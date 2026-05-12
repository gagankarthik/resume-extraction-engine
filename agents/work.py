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
  "description": "one brief sentence describing the role context only — NOT job duties",
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
- Strip leading bullet glyphs (•, ●, ▪, ▸, ‣, ○, ◦, ►, -, *, –, —) from each entry.
- Bullets may be marked by ANY of these glyphs, by numbered lists (1., 1)), or by no glyph at all (prose).
- If the job uses PROSE paragraphs (no explicit bullets): split each sentence into an individual item in responsibilities[]. Do NOT put job duties into description.
- Treat ANY narrative duty/action text in the segment as bullet content — even if it isn't formatted as a bulleted list. Past-tense action verbs (Designed, Led, Built, Managed, Migrated, Implemented, …) are duty signals; capture each as its own responsibility.
- NEVER leave responsibilities[] empty if the segment contains ANY duty/narrative text — every such job must have at least one entry. Only leave it empty if the segment is truly metadata-only (company/date/title/location/tech list with NO action sentences).
- description should be NULL or a single short sentence describing the COMPANY/ROLE context only (e.g. "Healthcare insurer in California"). NEVER put duty bullets, action verbs, or summary sentences into description — those belong in responsibilities[].
- projects[] rules (BE STRICT):
  • Use projects[] ONLY when the source resume EXPLICITLY labels sub-projects with their own headings (e.g. "Project 1: <Name>" or a discrete project name on its own line followed by its own bullet list).
  • DO NOT invent project names by grouping responsibilities by topic. A long flat list of bullets is NOT a multi-project structure — it's a single role's responsibilities. Put them all in responsibilities[].
  • If you find yourself synthesizing project names from bullet content (e.g. "Lakehouse Architecture Design", "BigID Implementation"), STOP — those are responsibility topics, not labeled sub-projects. Put them as flat entries in responsibilities[].
- achievements[] should contain ONLY items with measurable results (%, $, headcount, time saved, etc.).
- technologies_used[] = every tool/language/platform mentioned in this job.
- Return ONLY valid JSON.
"""

WORK_SYSTEM_FULL_FALLBACK = """You are a work experience extraction specialist. Extract ALL work experience entries from the resume. This is a long-career resume — include EVERY job, even old ones from 15-25 years ago.

CRITICAL RULES:
1. Include EVERY job entry — do not skip any role, even old ones.
2. NEVER skip or add bullet points. Extract EXACTLY the bullets that exist.
3. Copy all responsibilities VERBATIM — do not paraphrase or merge.
4. Bullets may be marked by ANY glyph (•, ●, ▪, ▸, ‣, ○, ◦, ►, -, *, –, —), numbered (1., 1)), or no glyph at all (prose). Treat them all as bullets.
5. If the job uses PROSE paragraphs (no bullet points): split each sentence into a separate item in responsibilities[]. Do NOT use the description field for job duties.
6. NEVER leave responsibilities[] empty if the job segment has ANY duty/narrative text — every such job must have at least one entry. Only leave it empty if the segment is truly metadata-only (company/date/title/location/tech list).
7. If a person worked on sub-projects under one company, structure them in projects[].

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
        Preserves document order; retries any job that failed on the first pass.
        """
        jobs_meta = structure.get("jobs", [])

        if not jobs_meta:
            return await self._extract_full_document(text)

        # First pass: all jobs in parallel
        first_pass = await asyncio.gather(
            *[self._extract_single_job(text, meta) for meta in jobs_meta],
            return_exceptions=True,
        )

        # Collect results and identify failures
        results: list[dict | None] = [None] * len(jobs_meta)
        retry_indices: list[int] = []

        for i, res in enumerate(first_pass):
            if isinstance(res, Exception):
                logger.warning(
                    "[WorkExperienceAgent] Job %d (%s) failed — will retry: %s",
                    i, jobs_meta[i].get("company", "?"), res,
                )
                retry_indices.append(i)
            else:
                results[i] = res

        # Retry pass for any failures
        if retry_indices:
            logger.info("[WorkExperienceAgent] Retrying %d failed job(s)", len(retry_indices))
            retry_pass = await asyncio.gather(
                *[self._extract_single_job(text, jobs_meta[i]) for i in retry_indices],
                return_exceptions=True,
            )
            for idx, res in zip(retry_indices, retry_pass):
                if isinstance(res, Exception):
                    logger.error(
                        "[WorkExperienceAgent] Job %d (%s) permanently failed after retry: %s",
                        idx, jobs_meta[idx].get("company", "?"), res,
                    )
                else:
                    results[idx] = res

        return [r for r in results if r is not None]

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
                f"Extract ALL responsibilities for {company} into the responsibilities[] array. "
                "If the job uses explicit bullet points (•, -, *, numbers): copy each bullet verbatim as a separate array item. "
                "If the job uses PROSE paragraphs instead of bullets: split every sentence into an individual item in responsibilities[]. "
                "Do NOT put job duties into the description field. "
                "NEVER leave responsibilities[] empty — every job must have at least one entry."
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

        raw, _ = await self._call_llm(system, user_msg, max_tokens=6144)
        result = self._parse_json(raw)

        # Handle bare JSON array (LLM skipped the wrapper object)
        if isinstance(result, list):
            return result[0] if result else {}

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
        raw, _ = await self._call_llm(WORK_SYSTEM_FULL_FALLBACK, user_msg, max_tokens=16384)
        result = self._parse_json(raw)
        if isinstance(result, list):
            return result
        return result.get("work_experience", [])
