"""
ResumeOrchestrator — multi-agent pipeline for best-in-class resume extraction.

Stage 1 (sequential):  StructureAgent   → identifies job boundaries, counts bullets
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
        logger.info(
            "[Orchestrator] Found %d job(s) in structure map",
            len(structure.get("jobs", [])),
        )

        # ── Stage 2: Parallel section extraction ──────────────────────────
        logger.info("[Orchestrator] Stage 2 — parallel extraction")
        (
            personal_result,
            work_result,
            edu_result,
            skills_result,
            cert_result,
            supp_result,
        ) = await asyncio.gather(
            self.personal_agent.run(normalized_text),
            self.work_agent.run(normalized_text, structure),
            self.education_agent.run(normalized_text),
            self.skills_agent.run(normalized_text),
            self.cert_agent.run(normalized_text),
            self.supp_agent.run(normalized_text),
            return_exceptions=False,
        )

        # ── Merge results ──────────────────────────────────────────────────
        merged: dict[str, Any] = {
            "personal_information": personal_result,
            "work_experience":      work_result,
            "education":            edu_result,
            "skills":               skills_result,
            "certifications":       cert_result,
        }
        # Supplemental agent returns a dict with many top-level keys
        if isinstance(supp_result, dict):
            merged.update(supp_result)

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

        return merged


# ── Module-level singleton for connection-pool reuse ──────────────────────
_orchestrator: ResumeOrchestrator | None = None


def get_orchestrator() -> ResumeOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResumeOrchestrator()
    return _orchestrator
