"""PersonalInfoAgent — focused extraction of personal information."""
from __future__ import annotations
from .base import BaseAgent

PERSONAL_SYSTEM = """Extract personal information from the resume. Scan ALL sections — headers, footers, sidebars, every column.

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
  }
}
"""


class PersonalInfoAgent(BaseAgent):
    def __init__(self):
        super().__init__("PersonalInfoAgent")

    async def run(self, text: str) -> dict:
        user_msg = f"=== RESUME ===\n{text}\n=== END ===\n\nExtract personal information only. Return JSON."
        raw, _ = await self._call_llm(PERSONAL_SYSTEM, user_msg, max_tokens=2048)
        result = self._parse_json(raw)
        return result.get("personal_information", result)
