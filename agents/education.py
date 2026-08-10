"""EducationAgent — focused extraction of education entries.

The prompt asks for the degree as the resume writes it, in full. What ships is
the abbreviation — BS, MS, AS — which `agents.degrees` derives from that full
phrase after extraction. Reading the family off the candidate's own words is the
part a table does better than a prompt: "Bachelor's degree in Accounting" is a
BS every time, whereas a model asked to shorten it sometimes decides it is a BBA.
"""
from __future__ import annotations

from .base import BaseAgent
from .sections import EDUCATION, slice_matching

EDUCATION_SYSTEM = """Extract ALL education entries from the resume.

Rules:
- degree = the degree EXACTLY as written in the resume, verbatim, INCLUDING the level word the resume uses ("Bachelor's degree in Accounting", "Associate of Applied Science", "B.Tech"). Copy it whole — do not shorten it yourself and do not swap one family for another.
- degree_type = the standard abbreviation of the SAME degree that is written. Use these forms: AA / AS / BA / BS / BE / BTech / BCom / BCA / BBA / MS / MA / MBA / MCA / MTech / MCom / PhD / Diploma.
  • NEVER convert between degree families. "BS" / "B.S." / "Bachelor of Science" → BS (NEVER BTech). "B.Tech" / "Bachelor of Technology" → BTech (NEVER BS). "Master of Business Administration" → MBA. "Bachelor of Engineering" → BE.
  • Abbreviate ONLY — do not change the degree itself. If unsure of the family, copy what is written.
- Extract field_of_study (e.g. "Computer Science", "Business Administration"). When the degree is written as "<level> in <subject>", the subject is the field_of_study — "Bachelor's degree in Accounting & Business Management" has field_of_study "Accounting & Business Management".
- Extract location: the institution's city/state/country if written ANYWHERE in the entry (including on the same line as the institution name). Use null only if no location is written.
- Extract end_date from any graduation year — even a standalone 4-digit year. Keep the month if one is written (e.g. "May 2018").
- Extract GPA, percentage, or grade when present.
- Do NOT invent any value that is not written in the resume — missing fields stay null.

Return ONLY this JSON:
{
  "education": [
    {
      "institution_name": "", "degree": null, "degree_type": null,
      "field_of_study": null, "major": null, "minor": null,
      "start_date": null, "end_date": null, "is_current": false,
      "gpa": null, "percentage": null, "grade": null,
      "honors": [], "relevant_coursework": [], "thesis_title": null,
      "dissertation": null, "location": null, "activities": [], "description": null
    }
  ]
}
"""


class EducationAgent(BaseAgent):
    def __init__(self):
        super().__init__("EducationAgent")

    async def run(self, text: str) -> list[dict]:
        # The education section when the resume marks one out, the whole
        # document when it does not. Reading only the relevant block is both
        # cheaper and stricter: a degree cannot be assembled out of a job
        # description that is no longer in front of the model.
        scoped = slice_matching(text, EDUCATION) or text
        user_msg = f"=== RESUME ===\n{scoped}\n=== END ===\n\nExtract education. Return JSON."
        result = await self._call_llm_json(
            EDUCATION_SYSTEM, user_msg, max_tokens=3072,
            section="Education",
        )
        if isinstance(result, list):
            return result
        return result.get("education", [])
