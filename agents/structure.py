"""
StructureAgent — discovers job boundaries in normalized resume text.
Programmatically counts bullets per job segment (never trusts LLM for counts).
"""
from __future__ import annotations

import logging

from .base import BaseAgent

logger = logging.getLogger(__name__)

STRUCTURE_SYSTEM = """You are a resume structure analyzer. Your ONLY job is to identify all work experience job entries in the resume text and locate their boundaries.

Return a JSON object with this exact shape:
{
  "jobs": [
    {
      "company": "Exact company name as written",
      "title": "Job title / role",
      "start_date": "as written",
      "end_date": "as written or 'Present'",
      "location": "city/state if present or null",
      "anchor_line": "the EXACT first line of text where this job begins (copy verbatim)",
      "has_sub_projects": true/false
    }
  ]
}

Rules:
- List jobs in ORDER as they appear in the resume (top to bottom).
- anchor_line must be the EXACT verbatim text of the first line of the job section (usually company name).
- Set has_sub_projects to true if the job contains labelled sub-projects or clients (consulting pattern).
- Return ONLY the JSON, no other text.
"""


class StructureAgent(BaseAgent):

    def __init__(self):
        super().__init__("StructureAgent")

    async def run(self, text: str) -> dict:
        """
        Returns:
        {
          "jobs": [
            {
              "company": str,
              "title": str,
              "start_date": str,
              "end_date": str,
              "location": str | None,
              "anchor_line": str,
              "has_sub_projects": bool,
              "segment": str,          # raw text segment for this job
              "bullet_count": int,     # programmatic count
            }
          ]
        }
        """
        user_msg = f"=== RESUME TEXT ===\n{text}\n=== END ==="
        try:
            raw, _ = await self._call_llm(STRUCTURE_SYSTEM, user_msg, max_tokens=4096)
            result = self._parse_json(raw)
            jobs = result.get("jobs", [])
        except Exception as exc:
            logger.warning("[StructureAgent] LLM failed to parse structure: %s — using empty job list", exc)
            jobs = []

        # Augment with programmatic bullet counts and text segments
        jobs = self._attach_segments_and_counts(text, jobs)
        return {"jobs": jobs}

    # ------------------------------------------------------------------ #
    # Segment extraction and bullet counting
    # ------------------------------------------------------------------ #

    def _attach_segments_and_counts(self, text: str, jobs: list[dict]) -> list[dict]:
        lines = text.split("\n")
        total_lines = len(lines)

        # Find start line for each job by matching anchor_line
        for job in jobs:
            anchor = (job.get("anchor_line") or "").strip()
            if not anchor:
                continue
            start = self._find_anchor_line(lines, anchor)
            if start is not None:
                job["_start"] = start

        # Sort by position in document
        anchored = [j for j in jobs if "_start" in j]
        unanchored = [j for j in jobs if "_start" not in j]
        anchored.sort(key=lambda j: j["_start"])

        # Attach segment text and bullet count
        for idx, job in enumerate(anchored):
            seg_start = job["_start"]
            seg_end = anchored[idx + 1]["_start"] if idx + 1 < len(anchored) else total_lines
            segment_lines = lines[seg_start:seg_end]
            job["segment"] = "\n".join(segment_lines)
            job["bullet_count"] = self._count_bullets(segment_lines)
            del job["_start"]

        # Unanchored jobs get empty segment (WorkAgent will fall back to full text)
        for job in unanchored:
            job["segment"] = ""
            job["bullet_count"] = 0

        return anchored + unanchored

    @staticmethod
    def _find_anchor_line(lines: list[str], anchor: str) -> int | None:
        anchor_lower = anchor.lower()
        # Exact match first
        for i, line in enumerate(lines):
            if line.strip().lower() == anchor_lower:
                return i
        # Substring match
        for i, line in enumerate(lines):
            if anchor_lower in line.lower():
                return i
        return None

    @staticmethod
    def _count_bullets(lines: list[str]) -> int:
        return sum(1 for line in lines if line.strip().startswith("•"))
