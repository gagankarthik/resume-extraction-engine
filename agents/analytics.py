"""
AnalyticsAgent — computes analytics from merged extraction data.
Uses pure Python for most fields; asks LLM only for industry/function classification.
"""
from __future__ import annotations

import logging
import re
from datetime import date

from .base import BaseAgent

logger = logging.getLogger(__name__)

# No LLM classification lives here any more.
#
# career_level, primary_industry, secondary_industries, job_functions and
# highest_education_level were all judgements ABOUT the candidate rather than
# anything the resume said. "Senior" / "Information Technology" appear nowhere
# in the source document, and the tool's contract is to return the resume's own
# text and nothing else. What remains below is arithmetic over dates and
# locations the resume does state.

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


# Quarter and season → representative start month, so spans like "Q3 2018" or
# "Spring 2019" still contribute to tenure instead of being parsed as 0 months.
_QUARTER_MONTH = {"q1": 1, "q2": 4, "q3": 7, "q4": 10}
_SEASON_MONTH = {"winter": 1, "spring": 4, "summer": 7, "fall": 10, "autumn": 10}


def _full_year(digits: str) -> int:
    """"2009" → 2009, "09" → 2009, "90" → 1990.

    A two-digit year on a resume is in the past, so the pivot is this year: a
    number that would land in the future belongs to the previous century.
    """
    year = int(digits)
    if len(digits) == 4:
        return year
    century = date.today().year // 100 * 100
    return year + century if year + century <= date.today().year else year + century - 100


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    s = s.strip()
    if re.search(r"present|current|now|till\s*date|to\s*date|ongoing", s, re.I):
        return date.today()
    # Quarter + year, e.g. "Q3 2018" / "2018 Q3"
    m = re.search(r"\b(q[1-4])\b.*?(\d{4})|(\d{4}).*?\b(q[1-4])\b", s, re.I)
    if m:
        q = (m.group(1) or m.group(4)).lower()
        year = int(m.group(2) or m.group(3))
        return date(year, _QUARTER_MONTH[q], 1)
    # Season + year, e.g. "Spring 2019"
    m = re.search(r"(winter|spring|summer|fall|autumn)\s+(\d{4})", s, re.I)
    if m:
        return date(int(m.group(2)), _SEASON_MONTH[m.group(1).lower()], 1)
    # Month Year  e.g. "Jan 2020", and the two-digit forms a long career is
    # written in: "Feb'09", "Nov ' 01", "Oct 02". A resume that spans decades
    # switches notation partway through, and every unparsed date used to count
    # as zero months — which is how thirty-five years of experience was
    # reported as sixteen.
    m = re.search(r"([A-Za-z]{3,9})\.?\s*[-–—]?\s*['’]?\s*(\d{4}|\d{2})(?!\d)", s)
    if m:
        month = _MONTH_MAP.get(m.group(1)[:3].lower())
        if month:
            return date(_full_year(m.group(2)), month, 1)
    # Numeric  e.g. "2020-01", "01/2020", "2020"
    m = re.search(r"(\d{4})[/-](\d{1,2})", s)
    if m:
        return date(int(m.group(1)), min(12, max(1, int(m.group(2)))), 1)
    m = re.search(r"(\d{1,2})[/-](\d{4})", s)
    if m:
        return date(int(m.group(2)), min(12, max(1, int(m.group(1)))), 1)
    # Year only (last, so it doesn't pre-empt the more specific patterns above)
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if m:
        return date(int(m.group(0)), 1, 1)
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

        # Compute numeric fields in Python (deterministic)
        total_months = self._non_overlapping_months(work)
        total_years = round(total_months / 12, 1)
        num_companies = len({j.get("company_name", "") for j in work if j.get("company_name")})
        num_roles = len(work)
        avg_tenure = round(total_months / num_roles) if num_roles else None

        return {
            "total_years_of_experience": total_years,
            "total_months_of_experience": int(total_months),
            "number_of_companies": num_companies,
            "number_of_roles": num_roles,
            "average_tenure_months": avg_tenure,
            "has_international_experience": self._has_international(work),
            "primary_location": self._primary_location(work),
        }

    # ------------------------------------------------------------------ #

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
        if not any(loc.strip() for loc in locs):
            return None  # no location data anywhere → genuinely unknown
        countries = {"india", "uk", "united kingdom", "canada", "australia", "germany", "france", "singapore", "uae"}
        for loc in locs:
            if any(c in loc.lower() for c in countries):
                return True
        return False  # we had locations and none were international

    @staticmethod
    def _primary_location(work: list[dict]) -> str | None:
        for j in work:
            loc = j.get("location")
            if loc:
                return loc
        return None
