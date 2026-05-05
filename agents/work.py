"""
WorkExperienceAgent — extracts work experience with EXACT bullet point counts.
The bullet count per job is determined programmatically by StructureAgent, never by LLM estimation.
"""
from __future__ import annotations

import asyncio
import logging

from .base import BaseAgent

logger = logging.getLogger(__name__)

WORK_SYSTEM_BASE = """You are a work experience extraction specialist. Extract the work experience for ONE specific job entry from the resume.

CRITICAL BULLET COUNT RULE:
{bullet_instruction}

Extract into this JSON shape:
{{
  "company_name": "exact company name",
  "job_title": "exact job title",
  "start_date": "as written",
  "end_date": "as written or 'Present'",
  "is_current": false,
  "location": "city/state or null",
  "department": "department name or null",
  "employment_type": "Full-time/Contract/etc. or null",
  "duration": "duration string or null",
  "responsibilities": ["bullet 1", "bullet 2", ...],
  "achievements": ["measurable outcome 1", ...],
  "technologies_used": ["tech1", "tech2", ...],
  "description": "short prose description or null",
  "projects": [
    {{
      "projectName": "project name",
      "clientName": "client or null",
      "projectLocation": "location or null",
      "keyTechnologies": "comma-separated list or null",
      "projectResponsibilities": ["bullet 1", ...]
    }}
  ]
}}

Rules:
- Copy every responsibility VERBATIM — do not paraphrase, summarize, or merge.
- Strip the leading bullet character (•) from each entry.
- If this job uses a consulting sub-project structure, place individual project bullets inside the projects[] array.
- achievements[] should contain ONLY items with measurable results (%, $, headcount, time saved, etc.).
- technologies_used[] = every tool/language/platform mentioned in this job.
- Return ONLY valid JSON.
"""

WORK_SYSTEM_FULL_FALLBACK = """You are a work experience extraction specialist. Extract ALL work experience entries from the resume.

CRITICAL RULES:
1. NEVER skip or add bullet points. Extract EXACTLY the bullets that exist.
2. Copy all responsibilities VERBATIM — do not paraphrase or merge.
3. If a person worked on sub-projects under one company, structure them in projects[].

Return a JSON object: {{ "work_experience": [ <job objects> ] }}

Each job object:
{{
  "company_name": str, "job_title": str, "start_date": str, "end_date": str,
  "is_current": bool, "location": str|null, "department": str|null,
  "employment_type": str|null, "duration": str|null,
  "responsibilities": [str], "achievements": [str], "technologies_used": [str],
  "description": str|null, "projects": []
}}

Return ONLY valid JSON.
"""


class WorkExperienceAgent(BaseAgent):

    def __init__(self):
        super().__init__("WorkExperienceAgent")

    async def run(self, text: str, structure: dict) -> list[dict]:
        """
        Extract work experience using per-job extraction with exact bullet counts.
        Falls back to full-document extraction if structure is empty.
        """
        jobs_meta = structure.get("jobs", [])

        if not jobs_meta:
            return await self._extract_full_document(text)

        # Run per-job extraction in parallel
        tasks = [self._extract_single_job(text, meta) for meta in jobs_meta]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        extracted = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.warning("[WorkExperienceAgent] Job %d extraction failed: %s", i, res)
                # Fall through — the validator will flag this as missing
            else:
                extracted.append(res)

        return extracted

    # ------------------------------------------------------------------ #
    # Per-job extraction
    # ------------------------------------------------------------------ #

    async def _extract_single_job(self, full_text: str, meta: dict) -> dict:
        company = meta.get("company", "Unknown Company")
        title = meta.get("title", "Unknown Title")
        bullet_count = meta.get("bullet_count", 0)
        segment = meta.get("segment", "").strip()
        has_projects = meta.get("has_sub_projects", False)

        # Build bullet instruction
        if bullet_count > 0:
            bullet_instruction = (
                f"This job segment for {company} contains EXACTLY {bullet_count} bullet points. "
                f"You MUST extract all {bullet_count} bullets into responsibilities[] (or distribute them "
                f"between responsibilities[] and projects[].projectResponsibilities if sub-projects exist). "
                f"DO NOT add bullets that don't exist. DO NOT skip or merge any bullet."
            )
        else:
            bullet_instruction = (
                f"Extract all bullet points and responsibilities you find for {company}. "
                "Copy them verbatim without paraphrasing."
            )

        if has_projects:
            bullet_instruction += (
                " This job has a consulting sub-project structure. Distribute bullets across "
                "projects[].projectResponsibilities[] instead of a flat responsibilities[]."
            )

        system = WORK_SYSTEM_BASE.format(bullet_instruction=bullet_instruction)

        # Use the job segment text if available, otherwise full document
        context_text = segment if segment else full_text
        user_msg = (
            f"Extract work experience for: {company} | {title}\n"
            f"Date range: {meta.get('start_date', '')} – {meta.get('end_date', '')}\n\n"
            "=== TEXT SEGMENT ===\n"
            f"{context_text}\n"
            "=== END SEGMENT ===\n\n"
            "Return ONLY the JSON for this single job entry."
        )

        raw, _ = await self._call_llm(system, user_msg, max_tokens=4096)
        result = self._parse_json(raw)

        # Handle both wrapped and unwrapped responses
        if "work_experience" in result and isinstance(result["work_experience"], list):
            result = result["work_experience"][0] if result["work_experience"] else {}
        elif "company_name" not in result:
            # LLM returned a dict with a single job key — look for it
            for v in result.values():
                if isinstance(v, dict) and "company_name" in v:
                    result = v
                    break

        return result

    # ------------------------------------------------------------------ #
    # Full-document fallback
    # ------------------------------------------------------------------ #

    async def _extract_full_document(self, text: str) -> list[dict]:
        logger.info("[WorkExperienceAgent] No structure map — falling back to full-document extraction")
        user_msg = f"=== RESUME TEXT ===\n{text}\n=== END ===\n\nExtract all work experience. Return only JSON."
        raw, _ = await self._call_llm(WORK_SYSTEM_FULL_FALLBACK, user_msg, max_tokens=8192)
        result = self._parse_json(raw)
        return result.get("work_experience", [])
