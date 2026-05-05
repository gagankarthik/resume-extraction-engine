"""
AnalyticsAgent — computes analytics from merged extraction data.
Uses pure Python for most fields; asks LLM only for industry/function classification.
"""
from __future__ import annotations

import re
import logging
from datetime import date, datetime
from typing import Any

from .base import BaseAgent

logger = logging.getLogger(__name__)

ANALYTICS_SYSTEM = """Given the work experience and education data below, classify:

1. career_level: one of "Entry-Level" / "Mid-Level" / "Senior" / "Director" / "Executive"
2. primary_industry: e.g. "Information Technology", "Healthcare", "Financial Services", "Consulting"
3. secondary_industries: list of any additional industries
4. job_functions: list of main functional areas e.g. ["Software Engineering", "Data Architecture"]
5. highest_education_level: e.g. "Master's Degree", "Bachelor's Degree", "Doctorate"
6. resume_language: ISO 639-1 code e.g. "en"

Return ONLY this JSON:
{
  "career_level": null,
  "primary_industry": null,
  "secondary_industries": [],
  "job_functions": [],
  "highest_education_level": null,
  "resume_language": "en"
}
"""

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    s = s.strip()
    if re.match(r"present|current|now", s, re.I):
        return date.today()
    # Year only
    m = re.fullmatch(r"\d{4}", s)
    if m:
        return date(int(s), 1, 1)
    # Month Year  e.g. "Jan 2020" or "2020-01"
    m = re.match(r"(\w+)\s+(\d{4})", s)
    if m:
        month_str, year = m.group(1).lower(), int(m.group(2))
        month = _MONTH_MAP.get(month_str[:3])
        if month:
            return date(year, month, 1)
    m = re.match(r"(\d{4})[/-](\d{1,2})", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)
    return None


def _tenure_months(start_str: str | None, end_str: str | None) -> float:
    s = _parse_date(start_str)
    e = _parse_date(end_str) or date.today()
    if s is None:
        return 0.0
    delta = (e.year - s.year) * 12 + (e.month - s.month)
    return max(0.0, float(delta))


class AnalyticsAgent(BaseAgent):

    def __init__(self):
        super().__init__("AnalyticsAgent")

    async def run(self, merged: dict) -> dict:
        work = merged.get("work_experience", [])
        edu = merged.get("education", [])

        # Compute numeric fields in Python (deterministic)
        tenures = [_tenure_months(j.get("start_date"), j.get("end_date")) for j in work]
        total_months = self._non_overlapping_months(work)
        total_years = round(total_months / 12, 1)
        num_companies = len({j.get("company_name", "") for j in work if j.get("company_name")})
        num_roles = len(work)
        avg_tenure = round(total_months / num_roles) if num_roles else None

        # Ask LLM for classification fields
        classification = await self._classify(work, edu)

        return {
            "total_years_of_experience": total_years,
            "total_months_of_experience": int(total_months),
            "career_level": classification.get("career_level"),
            "primary_industry": classification.get("primary_industry"),
            "secondary_industries": classification.get("secondary_industries", []),
            "job_functions": classification.get("job_functions", []),
            "highest_education_level": classification.get("highest_education_level"),
            "number_of_companies": num_companies,
            "number_of_roles": num_roles,
            "average_tenure_months": avg_tenure,
            "has_international_experience": self._has_international(work),
            "primary_location": self._primary_location(work),
            "salary_mentioned": None,
            "resume_language": classification.get("resume_language", "en"),
        }

    # ------------------------------------------------------------------ #

    async def _classify(self, work: list[dict], edu: list[dict]) -> dict:
        work_summary = "\n".join(
            f"- {j.get('company_name', '')} | {j.get('job_title', '')} | {j.get('start_date', '')}–{j.get('end_date', '')}"
            for j in work[:10]
        )
        edu_summary = "\n".join(
            f"- {e.get('degree', '')} in {e.get('field_of_study', '')} from {e.get('institution_name', '')}"
            for e in edu
        )
        user_msg = f"WORK EXPERIENCE:\n{work_summary}\n\nEDUCATION:\n{edu_summary}"
        try:
            raw, _ = await self._call_llm(ANALYTICS_SYSTEM, user_msg, max_tokens=1024)
            return self._parse_json(raw)
        except Exception as exc:
            logger.warning("[AnalyticsAgent] Classification failed: %s", exc)
            return {}

    @staticmethod
    def _non_overlapping_months(work: list[dict]) -> float:
        """Collapse overlapping date ranges before summing."""
        intervals = []
        for j in work:
            s = _parse_date(j.get("start_date"))
            e = _parse_date(j.get("end_date")) or date.today()
            if s:
                intervals.append((s, e))
        if not intervals:
            return 0.0
        intervals.sort()
        merged: list[tuple[date, date]] = [intervals[0]]
        for s, e in intervals[1:]:
            prev_s, prev_e = merged[-1]
            if s <= prev_e:
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))
        total = sum((e.year - s.year) * 12 + (e.month - s.month) for s, e in merged)
        return max(0.0, float(total))

    @staticmethod
    def _has_international(work: list[dict]) -> bool | None:
        locs = [j.get("location", "") or "" for j in work]
        countries = {"india", "uk", "united kingdom", "canada", "australia", "germany", "france", "singapore", "uae"}
        for loc in locs:
            if any(c in loc.lower() for c in countries):
                return True
        return None

    @staticmethod
    def _primary_location(work: list[dict]) -> str | None:
        for j in work:
            loc = j.get("location")
            if loc:
                return loc
        return None
