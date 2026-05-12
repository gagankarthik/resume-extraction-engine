"""
ValidatorAgent — cross-validates extracted JSON against raw text.
Re-extracts any job whose bullet count doesn't match the structure map.
"""
from __future__ import annotations

import asyncio
import logging

from .base import BaseAgent

logger = logging.getLogger(__name__)

REEXTRACT_SYSTEM = """Re-extract ONE job entry from the resume. The previous extraction missed some bullet points.

CRITICAL: This job segment contains EXACTLY {expected_count} bullet points.
Extract all {expected_count} bullets verbatim into responsibilities[].
DO NOT skip, merge, or add any bullet.

Also return ALL the other fields you can find in the segment — company_name, job_title,
start_date, end_date, is_current, location, department, employment_type, duration,
technologies_used, description, projects, achievements. Use the same schema as the
work_experience items. Even if a field is unchanged from a prior extraction, return it
so the caller can fall back to it. Leave any field you can't find as null or [].

Return ONLY the JSON for this single job entry.
"""


class ValidatorAgent(BaseAgent):

    def __init__(self):
        super().__init__("ValidatorAgent")

    async def run(self, merged: dict, raw_text: str, structure: dict) -> dict:
        """
        Validates and optionally corrects the merged extraction.
        Returns the (potentially corrected) merged dict.
        """
        work = merged.get("work_experience", [])
        jobs_meta = structure.get("jobs", [])

        mismatches = self._find_bullet_mismatches(work, jobs_meta)
        if not mismatches:
            logger.info("[ValidatorAgent] All bullet counts match — no corrections needed")
            return merged

        logger.warning("[ValidatorAgent] %d bullet count mismatch(es) found — re-extracting", len(mismatches))
        corrected_work = list(work)

        tasks = [(idx, meta) for idx, meta in mismatches]
        results = await asyncio.gather(
            *[self._reextract_job(raw_text, meta, idx) for idx, meta in tasks],
            return_exceptions=True,
        )

        for (idx, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.warning("[ValidatorAgent] Re-extraction failed for job %d: %s", idx, result)
                continue
            if not result:
                continue
            # Merge: keep the original job's metadata as the floor, then overlay
            # any non-empty field from the re-extraction. The re-extract prompt is
            # bullet-focused; the LLM sometimes returns a sparse object without
            # company_name/job_title/location, and replacing wholesale would erase
            # good metadata from the first pass (the "Company / Role" placeholder bug).
            corrected_work[idx] = self._merge_job(corrected_work[idx], result)

        merged["work_experience"] = corrected_work

        # Final pass: drop any job whose bullet count STILL doesn't match the
        # structure map. Showing a half-extracted job is worse than omitting it.
        remaining_mismatches = self._find_bullet_mismatches(corrected_work, jobs_meta)
        if remaining_mismatches:
            drop_indices = {idx for idx, _ in remaining_mismatches}
            for idx, meta in remaining_mismatches:
                logger.warning(
                    "[ValidatorAgent] Dropping %s — bullet count still mismatched after re-extraction",
                    meta.get("company", f"job {idx}"),
                )
            merged["work_experience"] = [
                job for i, job in enumerate(corrected_work) if i not in drop_indices
            ]

        return merged

    @staticmethod
    def _merge_job(original: dict, replacement: dict) -> dict:
        """Overlay non-empty fields from `replacement` onto `original`."""
        merged = dict(original)
        for k, v in replacement.items():
            # An empty string/list/dict/None from the re-extraction should not
            # overwrite a populated value from the first pass.
            if v is None or v == "" or v == [] or v == {}:
                continue
            merged[k] = v
        return merged

    # ------------------------------------------------------------------ #

    def _find_bullet_mismatches(
        self, work: list[dict], jobs_meta: list[dict]
    ) -> list[tuple[int, dict]]:
        mismatches = []
        for i, (job, meta) in enumerate(zip(work, jobs_meta)):
            expected = meta.get("bullet_count", 0)
            if expected == 0:
                continue
            extracted = len(job.get("responsibilities", []))
            # Also count sub-project bullets
            for proj in job.get("projects", []):
                extracted += len(proj.get("projectResponsibilities", []))
            if extracted != expected:
                logger.debug(
                    "[ValidatorAgent] %s: expected %d bullets, got %d",
                    meta.get("company", "?"), expected, extracted,
                )
                mismatches.append((i, meta))
        return mismatches

    async def _reextract_job(self, full_text: str, meta: dict, job_idx: int) -> dict | None:
        expected = meta.get("bullet_count", 0)
        segment = meta.get("segment", "").strip() or full_text
        company = meta.get("company", "Unknown")

        system = REEXTRACT_SYSTEM.format(expected_count=expected)
        user_msg = (
            f"Job: {company} | {meta.get('title', '')} | "
            f"{meta.get('start_date', '')}–{meta.get('end_date', '')}\n\n"
            "=== TEXT SEGMENT ===\n"
            f"{segment}\n"
            "=== END ==="
        )
        raw, _ = await self._call_llm(system, user_msg, max_tokens=6144)
        result = self._parse_json(raw)

        # Unwrap if needed
        if "work_experience" in result and isinstance(result["work_experience"], list):
            result = result["work_experience"][0] if result["work_experience"] else None
        return result
