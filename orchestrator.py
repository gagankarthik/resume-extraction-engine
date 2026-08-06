"""
ResumeOrchestrator — multi-agent pipeline for best-in-class resume extraction.

Stage 1 (one wave):    StructureAgent maps job boundaries and counts bullets, while
                       PersonalAgent, EducationAgent, SkillsAgent, CertificationsAgent
                       and SupplementalAgent read the document alongside it. Only
                       WorkAgent needs the structure map, so only WorkAgent waits for
                       it — the other five have no reason to, and used to anyway.
Stage 2 (pure Python): AnalyticsAgent computes tenure arithmetic from merged data.
Stage 3 (refinement):  ValidatorAgent re-extracts jobs that came back short.
Stage 4 (refinement):  CompletenessAuditorAgent grounds contact/client/project names
                       against the source text, measures coverage, and recovers missed
                       content additively.

Stages 3 and 4 improve a result that already exists. When the run is low on
budget they are skipped and said so in the report, because a resume that is
slightly less polished beats no resume at all — which is what a blown deadline
used to produce.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents import report
from agents.analytics import AnalyticsAgent
from agents.auditor import CompletenessAuditorAgent
from agents.certifications import CertificationsAgent
from agents.deadline import Deadline, DeadlineExceeded, set_deadline
from agents.education import EducationAgent
from agents.personal import PersonalInfoAgent
from agents.skills import SkillsAgent
from agents.structure import StructureAgent
from agents.supplemental import SupplementalAgent
from agents.validator_agent import ValidatorAgent
from agents.work import WorkExperienceAgent

logger = logging.getLogger(__name__)

# Roughly what each refinement stage needs to finish once started. Measured
# against the deadline before the stage begins, so a stage never starts work it
# cannot deliver.
_VALIDATION_BUDGET_SECONDS = 30.0
_AUDIT_BUDGET_SECONDS = 20.0


# Agent class name -> the name the user sees when a section did not come through.
_SECTION_LABELS = {
    "PersonalInfoAgent":    "Personal information",
    "WorkExperienceAgent":  "Work experience",
    "EducationAgent":       "Education",
    "SkillsAgent":          "Skills",
    "CertificationsAgent":  "Certifications",
    "SupplementalAgent":    "Additional sections",
}


async def _gather_within(deadline: Deadline, coros: tuple[Any, ...]) -> list[Any]:
    """Run everything concurrently, but stop waiting when the budget is gone.

    asyncio.gather waits for the slowest branch however long it takes, which is
    what let one slow section carry the whole run past its deadline and out
    through the caller as an error. Here the agents that finished are kept, the
    ones still running are cancelled, and the run continues with a hole in it —
    a resume missing one section beats no resume at all, and _unwrap records
    which section it is.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    remaining = deadline.remaining()
    _, pending = await asyncio.wait(
        tasks, timeout=None if remaining == float("inf") else remaining
    )

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    results: list[Any] = []
    for task in tasks:
        if task in pending:
            results.append(DeadlineExceeded("still running when the run ran out of time"))
            continue
        try:
            results.append(task.result())
        except Exception as exc:  # reported per section by _unwrap
            results.append(exc)
    return results


def _unwrap(result: Any, default: Any, agent_name: str) -> Any:
    """
    Return the agent's result, or the default if it raised.

    A failure here is recorded rather than only logged. An empty section in the
    response is otherwise indistinguishable from a section the resume never had,
    and the person reviewing the output is the one who needs to know which it is.
    """
    if isinstance(result, Exception):
        logger.warning("[Orchestrator] %s failed: %s", agent_name, result)
        ran_out_of_time = isinstance(result, DeadlineExceeded)
        report.record(
            _SECTION_LABELS.get(agent_name, agent_name),
            report.Status.FAILED,
            detail=(
                "This section was still being read when the run ran out of time, "
                "so it is empty here. Upload the file again to get it."
                if ran_out_of_time else
                "This section could not be extracted, so it is empty here even "
                "if the resume has one. Add it by hand, or run the file again."
            ),
        )
        return default
    return result


class ResumeOrchestrator:

    def __init__(self):
        self.structure_agent   = StructureAgent()
        self.personal_agent    = PersonalInfoAgent()
        self.work_agent        = WorkExperienceAgent()
        self.education_agent   = EducationAgent()
        self.skills_agent      = SkillsAgent()
        self.cert_agent        = CertificationsAgent()
        self.supp_agent        = SupplementalAgent()
        self.analytics_agent   = AnalyticsAgent()
        self.validator_agent   = ValidatorAgent()
        self.auditor_agent     = CompletenessAuditorAgent()

    async def run(self, normalized_text: str, deadline: Deadline | None = None) -> dict:
        deadline = deadline or Deadline.unlimited()
        set_deadline(deadline)

        # ── Stage 1: one wave, not two ────────────────────────────────────
        #
        # Structure discovery reads the whole document to find job boundaries,
        # and work extraction genuinely cannot begin until it has. The other
        # five agents read the same document for entirely different things and
        # never look at the structure map — yet they used to sit behind it,
        # adding a full round trip to every single extraction for nothing.
        logger.info("[Orchestrator] Stage 1 — structure discovery + section extraction")
        structure_task = asyncio.create_task(self.structure_agent.run(normalized_text))

        async def work_when_structured() -> list[dict]:
            structure = await structure_task
            logger.info(
                "[Orchestrator] Found %d job(s) in structure map", len(structure.get("jobs", []))
            )
            return await self.work_agent.run(normalized_text, structure)

        raw_results = await _gather_within(
            deadline,
            (
                self.personal_agent.run(normalized_text),
                work_when_structured(),
                self.education_agent.run(normalized_text),
                self.skills_agent.run(normalized_text),
                self.cert_agent.run(normalized_text),
                self.supp_agent.run(normalized_text),
            ),
        )
        if not structure_task.done():
            structure_task.cancel()

        # Always resolved by now — work_when_structured awaited it, and gather
        # waits for every branch. Reading it here keeps the map available to the
        # validation stage even when work extraction itself failed.
        try:
            structure = structure_task.result()
        except Exception as exc:  # a missing map degrades the run, never fails it
            logger.warning("[Orchestrator] Structure discovery failed: %s", exc)
            structure = {"jobs": []}
        if not isinstance(structure, dict):
            structure = {"jobs": []}

        personal_raw  = _unwrap(raw_results[0], {}, "PersonalInfoAgent")
        work_result   = _unwrap(raw_results[1], [], "WorkExperienceAgent")
        edu_result    = _unwrap(raw_results[2], [], "EducationAgent")
        skills_result = _unwrap(raw_results[3], {}, "SkillsAgent")
        cert_result   = _unwrap(raw_results[4], [], "CertificationsAgent")
        supp_result   = _unwrap(raw_results[5], {}, "SupplementalAgent")

        # PersonalInfoAgent now returns {"personal_information": {...}, "professional_summary": ..., "objective": ...}
        personal_info = personal_raw.get("personal_information", personal_raw) if isinstance(personal_raw, dict) else {}
        summary_from_personal   = personal_raw.get("professional_summary") if isinstance(personal_raw, dict) else None
        objective_from_personal = personal_raw.get("objective")            if isinstance(personal_raw, dict) else None

        # ── Merge results ──────────────────────────────────────────────────
        merged: dict[str, Any] = {
            "personal_information": personal_info,
            "work_experience":      work_result   if isinstance(work_result, list)   else [],
            "education":            edu_result    if isinstance(edu_result, list)    else [],
            "skills":               skills_result if isinstance(skills_result, dict) else {},
            "certifications":       cert_result   if isinstance(cert_result, list)   else [],
        }

        # Seed summary/objective from PersonalInfoAgent (guaranteed fast extraction)
        if summary_from_personal:
            merged["professional_summary"] = summary_from_personal
        if objective_from_personal:
            merged["objective"] = objective_from_personal

        # Merge supplemental — only overwrite with non-null/non-empty values so a
        # truncated SupplementalAgent response never wipes out the summary we already have.
        if isinstance(supp_result, dict):
            for key, val in supp_result.items():
                existing = merged.get(key)
                has_content = val is not None and val != [] and val != {}
                existing_empty = existing is None or existing == [] or existing == {}
                if has_content or existing_empty:
                    merged[key] = val

        # ── Stage 2: Analytics ────────────────────────────────────────────
        # Pure arithmetic over the dates already extracted — no model call, so
        # no budget check.
        logger.info("[Orchestrator] Stage 2 — analytics")
        try:
            analytics = await self.analytics_agent.run(merged)
            merged["analytics"] = analytics
        except Exception as exc:
            logger.warning("[Orchestrator] Analytics failed: %s", exc)
            merged["analytics"] = {}

        # ── Stage 3: Validation + re-extraction ───────────────────────────
        if deadline.allows(_VALIDATION_BUDGET_SECONDS):
            logger.info("[Orchestrator] Stage 3 — validation (%.0fs left)", deadline.remaining())
            try:
                merged = await self.validator_agent.run(merged, normalized_text, structure)
            except Exception as exc:
                logger.warning("[Orchestrator] Validation pass failed: %s", exc)
        else:
            logger.warning("[Orchestrator] Stage 3 — validation skipped, out of budget")
            report.record(
                "Validation",
                report.Status.SKIPPED,
                detail=(
                    "The bullet-count re-check did not run — this resume took long "
                    "enough to read that there was no time left for it. The work "
                    "history is here; compare it against the original."
                ),
            )

        # ── Stage 4: Completeness audit ───────────────────────────────────
        # Grounds hallucination-prone values against the source text, measures
        # how much of the resume actually made it into the JSON, and recovers
        # any missed content additively. Never fails the request.
        #
        # This stage always runs: its groundedness pass is what strips invented
        # metrics and unnamed technologies, and that guard is not optional at any
        # speed. Only the LLM-backed recovery call inside it yields to the clock.
        logger.info("[Orchestrator] Stage 4 — completeness audit")
        try:
            merged = await self.auditor_agent.run(
                merged, normalized_text, allow_recovery=deadline.allows(_AUDIT_BUDGET_SECONDS)
            )
        except Exception as exc:
            logger.warning("[Orchestrator] Completeness audit failed: %s", exc)

        # ── Attach the run report ─────────────────────────────────────────
        # Travels under a private key so it survives into _metadata without
        # becoming part of the resume schema itself.
        run_report = report.get_report()
        if run_report is not None:
            merged["_extraction_report"] = run_report.to_dict()

        # Final sanity log
        we = merged.get("work_experience", [])
        logger.info(
            "[Orchestrator] Final result: %d job(s), summary=%s, degraded=%s, %.0fs of budget unused",
            len(we),
            "present" if merged.get("professional_summary") else "absent",
            run_report.degraded if run_report else "unknown",
            deadline.remaining(),
        )

        return merged


# ── Module-level singleton for connection-pool reuse ──────────────────────
_orchestrator: ResumeOrchestrator | None = None


def get_orchestrator() -> ResumeOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResumeOrchestrator()
    return _orchestrator
