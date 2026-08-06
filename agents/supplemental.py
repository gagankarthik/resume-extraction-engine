"""
SupplementalAgent — extracts all non-core sections in one pass:
projects, references, patents, conferences, courses, training,
extracurricular, professional summary, objective, raw_sections.

Awards, volunteer experience, languages, publications, memberships and
interests are deliberately NOT extracted. They are not part of the output the
tool produces, so asking for them only spends tokens on content that is
discarded downstream.
"""
from __future__ import annotations

from .base import BaseAgent, output_budget
from .sections import EXPERIENCE, slice_excluding

SUPP_SYSTEM = """Extract all supplemental sections from the resume. Return ONLY this JSON (use [] for missing arrays, null for missing scalars):

{
  "professional_summary": null,
  "objective": null,
  "projects": [{"name":"","description":null,"role":null,"start_date":null,"end_date":null,"is_current":false,"technologies":[],"url":null,"repository_url":null,"highlights":[],"team_size":null,"type":null}],
  "references": [{"name":null,"title":null,"company":null,"email":null,"phone":null,"relationship":null,"available_on_request":false}],
  "patents": [{"title":"","patent_number":null,"date":null,"description":null,"status":null,"inventors":[],"url":null}],
  "conferences_and_talks": [{"title":"","event":null,"date":null,"location":null,"description":null,"url":null,"type":null}],
  "courses": [{"name":"","provider":null,"platform":null,"date":null,"url":null,"credential_id":null,"duration":null}],
  "training": [{"name":"","provider":null,"date":null,"duration":null,"description":null}],
  "extracurricular_activities": [{"organization":"","role":null,"start_date":null,"end_date":null,"description":null}],
  "raw_sections": {"section_names_found":[],"unclassified_content":null}
}

Rules:
- If a section has no data, use an empty array [] or null — never include placeholder items.
- NEVER invent data. Every value must be copied from the resume text; missing values stay null.
- These sections are NOT part of this schema. SKIP them entirely — do not return them, and do not relocate their content into any other field:
  • Awards / Honors / Recognition / Accolades
  • Volunteer Experience / Community Service
  • Languages / Language Proficiency
  • Publications / Papers
  • Professional Memberships / Affiliations
  • Interests / Hobbies / Activities
- projects: extract ANY standalone "Projects" / "Academic Projects" / "Personal Projects" / "Key Projects" section, with every bullet under each project copied verbatim into highlights[]. Do NOT skip this section when it exists. (Projects that are nested inside a specific job's work experience are handled elsewhere — only extract standalone project sections here.)
"""


class SupplementalAgent(BaseAgent):
    def __init__(self):
        super().__init__("SupplementalAgent")

    async def run(self, text: str) -> dict:
        # Everything this agent wants lives outside the work history, which is
        # usually most of the document by volume. Dropping it leaves a much
        # smaller read and takes with it the standing temptation to lift a job's
        # bullets into projects[] — the one thing the prompt above says twice
        # not to do. A summary-shaped heading is kept even when it matches, so
        # "EXPERIENCE SUMMARY" does not take the summary out with the jobs.
        scoped = slice_excluding(text, EXPERIENCE)
        user_msg = f"=== RESUME ===\n{scoped}\n=== END ===\n\nExtract all supplemental sections. Return JSON."
        return await self._call_llm_json(
            SUPP_SYSTEM, user_msg,
            max_tokens=output_budget(scoped, floor=8192, ceiling=16384),
            section="Additional sections",
        )
