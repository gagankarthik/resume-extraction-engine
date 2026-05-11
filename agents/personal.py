"""PersonalInfoAgent — focused extraction of personal information, professional summary, and objective."""
from __future__ import annotations
from .base import BaseAgent

PERSONAL_SYSTEM = """Extract personal information, the professional summary, and the objective statement from the resume. Scan ALL sections — headers, footers, sidebars, every column.

Return ONLY this JSON:
{
  "personal_information": {
    "full_name": null,
    "first_name": null,
    "last_name": null,
    "email": [],
    "phone": [],
    "address": {
      "full_address": null, "street": null, "city": null,
      "state": null, "country": null, "zip_code": null
    },
    "linkedin_url": null, "github_url": null, "portfolio_url": null,
    "twitter_url": null, "other_urls": [],
    "date_of_birth": null, "nationality": null, "gender": null,
    "marital_status": null, "profile_headline": null
  },
  "professional_summary": null,
  "objective": null
}

Rules for professional_summary:
- Extract the full text of any section labelled "Summary", "Professional Summary", "Profile", "About Me", "Career Summary", "Executive Summary", "Overview", or similar.
- Copy it verbatim — do not paraphrase or shorten.
- If the summary is written as multiple bullet points, join them into a single string separated by " | ".
- If no summary section exists, return null.

Rules for objective:
- Extract the full text of any section labelled "Objective", "Career Objective", "Goal", or similar.
- If no objective section exists, return null.
"""


class PersonalInfoAgent(BaseAgent):
    def __init__(self):
        super().__init__("PersonalInfoAgent")

    async def run(self, text: str) -> dict:
        user_msg = f"=== RESUME ===\n{text}\n=== END ===\n\nExtract personal information, professional summary, and objective. Return JSON."
        raw, _ = await self._call_llm(PERSONAL_SYSTEM, user_msg, max_tokens=3072)
        result = self._parse_json(raw)
        if isinstance(result, list):
            return {"personal_information": result[0] if result else {}}
        # Normalise: if LLM returned bare personal_information dict without the wrapper
        if "personal_information" not in result and "full_name" in result:
            return {"personal_information": result, "professional_summary": None, "objective": None}
        return result
