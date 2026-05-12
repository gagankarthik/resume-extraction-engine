"""
StructureAgent — discovers job boundaries in normalized resume text.
Programmatically counts bullets per job segment (never trusts LLM for counts).
"""
from __future__ import annotations

import logging
import re

from .base import BaseAgent

logger = logging.getLogger(__name__)

STRUCTURE_SYSTEM = """You are a resume structure analyzer. Your ONLY job is to identify ALL work experience job entries in the resume text.

CRITICAL: You MUST find EVERY single job entry — no matter how many. Long careers of 15–25+ years may have 10–20 or more jobs. Do NOT stop early. Scan the ENTIRE document from top to bottom.

Return a JSON object with this exact shape:
{
  "jobs": [
    {
      "company": "Exact company name as written",
      "title": "Job title / role",
      "start_date": "as written",
      "end_date": "as written or 'Present'",
      "location": "city/state if present or null",
      "anchor_line": "the EXACT first line of text where this job begins — copy it character-for-character verbatim",
      "has_sub_projects": true/false
    }
  ]
}

Rules:
- List EVERY job in ORDER as they appear in the resume (top to bottom), including old roles from 20-25 years ago.
- anchor_line MUST be the exact verbatim text of the very first line of the job entry (usually the company name line). Copy it character-for-character — no paraphrasing.
- If the resume has an "ORGANISATIONAL SCAN" or compact job-list section, list every company/role in that section.
- Set has_sub_projects to true if the job contains labelled sub-projects, client engagements, or consulting assignments.
- Include ALL positions: full-time, contract, consulting, part-time, internships.
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
            raw, _ = await self._call_llm(STRUCTURE_SYSTEM, user_msg, max_tokens=8192)
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
        anchor_lower = anchor.strip().lower()
        # Pass 1: exact match
        for i, line in enumerate(lines):
            if line.strip().lower() == anchor_lower:
                return i
        # Pass 2: line starts with anchor (line has trailing date/location after company name)
        for i, line in enumerate(lines):
            if line.strip().lower().startswith(anchor_lower):
                return i
        # Pass 3: anchor starts with the full line (anchor has more context than the actual line)
        # Only trigger when the line is at least 6 chars to prevent false positives on short words.
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if len(stripped) >= 6 and anchor_lower.startswith(stripped):
                return i
        # Pass 4: substring match (last resort)
        for i, line in enumerate(lines):
            if anchor_lower in line.lower():
                return i
        return None

    _NUMBERED_BULLET = re.compile(r"^\d+[\.\)]\s+\S")
    # Glyph and ASCII bullets the resume might use — broader than just "•".
    _BULLET_GLYPH = re.compile(r"^[•●▪▸‣○◦►◘■□◙*\-–—]\s+\S")

    @classmethod
    def _count_bullets(cls, lines: list[str]) -> int:
        count = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if (
                cls._BULLET_GLYPH.match(stripped)
                or cls._NUMBERED_BULLET.match(stripped)
            ):
                count += 1
        return count
