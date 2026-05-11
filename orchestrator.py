"""
ResumeOrchestrator — multi-agent pipeline for best-in-class resume extraction.

Stage 1 (sequential):  StructureAgent   → identifies ALL job boundaries, counts bullets
Stage 2 (parallel):    PersonalAgent, WorkAgent, EducationAgent, SkillsAgent,
                       CertificationsAgent, SupplementalAgent
Stage 3 (sequential):  AnalyticsAgent   → computes analytics from merged data
Stage 4 (sequential):  ValidatorAgent   → cross-validates bullet counts, re-extracts mismatches
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.structure import StructureAgent
from agents.personal import PersonalInfoAgent
from agents.work import WorkExperienceAgent
from agents.education import EducationAgent
from agents.skills import SkillsAgent
from agents.certifications import CertificationsAgent
from agents.supplemental import SupplementalAgent
from agents.analytics import AnalyticsAgent
from agents.validator_agent import ValidatorAgent

logger = logging.getLogger(__name__)


def _unwrap(result: Any, default: Any, agent_name: str) -> Any:
    """Return default and log a warning if result is an Exception."""
    if isinstance(result, Exception):
        logger.warning("[Orchestrator] %s failed: %s", agent_name, result)
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

    async def run(self, normalized_text: str) -> dict:
        # ── Stage 1: Structure discovery ──────────────────────────────────
        logger.info("[Orchestrator] Stage 1 — structure discovery")
        structure = await self.structure_agent.run(normalized_text)
        job_count = len(structure.get("jobs", []))
        logger.info("[Orchestrator] Found %d job(s) in structure map", job_count)

        # ── Stage 2: Parallel section extraction ──────────────────────────
        logger.info("[Orchestrator] Stage 2 — parallel extraction")
        raw_results = await asyncio.gather(
            self.personal_agent.run(normalized_text),
            self.work_agent.run(normalized_text, structure),
            self.education_agent.run(normalized_text),
            self.skills_agent.run(normalized_text),
            self.cert_agent.run(normalized_text),
            self.supp_agent.run(normalized_text),
            return_exceptions=True,  # never let one agent failure kill all results
        )

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

        # ── Stage 3: Analytics ────────────────────────────────────────────
        logger.info("[Orchestrator] Stage 3 — analytics")
        try:
            analytics = await self.analytics_agent.run(merged)
            merged["analytics"] = analytics
        except Exception as exc:
            logger.warning("[Orchestrator] Analytics failed: %s", exc)
            merged["analytics"] = {}

        # ── Stage 4: Validation + re-extraction ───────────────────────────
        logger.info("[Orchestrator] Stage 4 — validation")
        try:
            merged = await self.validator_agent.run(merged, normalized_text, structure)
        except Exception as exc:
            logger.warning("[Orchestrator] Validation pass failed: %s", exc)

        # Final sanity log
        we = merged.get("work_experience", [])
        logger.info(
            "[Orchestrator] Final result: %d job(s), summary=%s",
            len(we),
            "present" if merged.get("professional_summary") else "absent",
        )

        return merged


# ── Module-level singleton for connection-pool reuse ──────────────────────
_orchestrator: ResumeOrchestrator | None = None


def get_orchestrator() -> ResumeOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResumeOrchestrator()
    return _orchestrator
