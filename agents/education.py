"""EducationAgent — focused extraction of education entries."""
from __future__ import annotations
from .base import BaseAgent

EDUCATION_SYSTEM = """Extract ALL education entries from the resume.

Rules:
- degree_type must be a standard abbreviation: B.S. / B.A. / B.E. / B.Tech / M.S. / M.A. / MBA / MCA / M.Tech / Ph.D. / Diploma / Associate / etc.
- Extract field_of_study (e.g. "Computer Science", "Business Administration").
- Extract end_date from any graduation year — even a standalone 4-digit year.
- Extract GPA, percentage, or grade when present.

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
        raw, _ = await self._call_llm(EDUCATION_SYSTEM, user_msg, max_tokens=3072)
        result = self._parse_json(raw)
        if isinstance(result, list):
            return result
        return result.get("education", [])
