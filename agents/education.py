"""EducationAgent — focused extraction of education entries."""
from __future__ import annotations

from .base import BaseAgent

EDUCATION_SYSTEM = """Extract ALL education entries from the resume.

Rules:
- degree = the degree EXACTLY as written in the resume, verbatim.
- degree_type = the standard abbreviation of the SAME degree that is written. Use these forms: AA / AS / BA / BS / BE / BTech / MS / MA / MBA / MCA / MTech / PhD / Diploma / Associate.
  • NEVER convert between degree families. "BS" / "B.S." / "Bachelor of Science" → BS (NEVER BTech). "B.Tech" / "Bachelor of Technology" → BTech (NEVER BS). "Master of Business Administration" / "Masters of Business Administration" → MBA. "Bachelor of Engineering" → BE.
  • Abbreviate ONLY — do not change the degree itself. If unsure of the family, copy what is written.
- Extract field_of_study (e.g. "Computer Science", "Business Administration").
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
        user_msg = f"=== RESUME ===\n{text}\n=== END ===\n\nExtract education. Return JSON."
        result = await self._call_llm_json(
            EDUCATION_SYSTEM, user_msg, max_tokens=3072,
            section="Education",
        )
        if isinstance(result, list):
            return result
        return result.get("education", [])
