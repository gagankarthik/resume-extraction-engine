"""
SupplementalAgent — extracts all non-core sections in one pass:
projects, publications, awards, volunteer, languages, interests,
references, patents, memberships, conferences, courses, training,
extracurricular, professional summary, objective, raw_sections.
"""
from __future__ import annotations
from .base import BaseAgent

SUPP_SYSTEM = """Extract all supplemental sections from the resume. Return ONLY this JSON (use [] for missing arrays, null for missing scalars):

{
  "professional_summary": null,
  "objective": null,
  "projects": [{"name":"","description":null,"role":null,"start_date":null,"end_date":null,"is_current":false,"technologies":[],"url":null,"repository_url":null,"highlights":[],"team_size":null,"type":null}],
  "publications": [{"title":"","authors":[],"publisher":null,"journal":null,"conference":null,"date":null,"url":null,"doi":null,"isbn":null,"description":null,"type":null}],
  "awards_and_honors": [{"title":"","issuer":null,"date":null,"description":null,"level":null}],
  "volunteer_experience": [{"organization":"","role":null,"start_date":null,"end_date":null,"is_current":false,"location":null,"description":null,"responsibilities":[],"cause":null}],
  "languages": [{"language":"","proficiency":null,"reading":null,"writing":null,"speaking":null}],
  "interests_and_hobbies": [],
  "references": [{"name":null,"title":null,"company":null,"email":null,"phone":null,"relationship":null,"available_on_request":false}],
  "patents": [{"title":"","patent_number":null,"date":null,"description":null,"status":null,"inventors":[],"url":null}],
  "professional_memberships": [{"organization":"","role":null,"membership_type":null,"start_date":null,"end_date":null,"is_current":false}],
  "conferences_and_talks": [{"title":"","event":null,"date":null,"location":null,"description":null,"url":null,"type":null}],
  "courses": [{"name":"","provider":null,"platform":null,"date":null,"url":null,"credential_id":null,"duration":null}],
  "training": [{"name":"","provider":null,"date":null,"duration":null,"description":null}],
  "extracurricular_activities": [{"organization":"","role":null,"start_date":null,"end_date":null,"description":null}],
  "raw_sections": {"section_names_found":[],"unclassified_content":null}
}

If a section has no data, use an empty array [] or null — never include placeholder items.
"""


class SupplementalAgent(BaseAgent):
    def __init__(self):
        super().__init__("SupplementalAgent")

    async def run(self, text: str) -> dict:
        user_msg = f"=== RESUME ===\n{text}\n=== END ===\n\nExtract all supplemental sections. Return JSON."
        raw, _ = await self._call_llm(SUPP_SYSTEM, user_msg, max_tokens=6144)
        return self._parse_json(raw)
