"""CertificationsAgent — focused extraction of certifications."""
from __future__ import annotations

from .base import BaseAgent
from .sections import CERTIFICATION, slice_matching

CERT_SYSTEM = """Extract certifications, licenses, and certificates from the resume.

WHERE THEY MAY COME FROM — this is the whole rule:
- ONLY from a dedicated section headed "Certifications", "Certificates", "Licenses", "Credentials", or an equivalent title.
- If the resume has NO such section, return {"certifications": []}. An empty list is the correct answer. Do NOT go looking elsewhere to fill it.
- NEVER build a certification out of a job responsibility, a skill, a tool name, a degree, or a training course. "Certified Scrum practices adopted across the team" is a responsibility, not a certification. "AWS" in a skills list is a technology, not a certification.
- Extract EVERY entry inside a real certifications section, copied verbatim.

Return ONLY this JSON:
{
  "certifications": [
    {
      "name": "", "issuing_organization": null, "issue_date": null,
      "expiry_date": null, "credential_id": null, "credential_url": null, "description": null
    }
  ]
}
"""


class CertificationsAgent(BaseAgent):
    def __init__(self):
        super().__init__("CertificationsAgent")

    async def run(self, text: str) -> list[dict]:
        # This agent's entire rule is "only from a dedicated certifications
        # section", which the prompt could ask for but not guarantee — and when
        # it slipped, a job responsibility mentioning "certified" came back as a
        # credential. Finding the section here settles it: no section, no call,
        # no certifications. The audit stage already strips certifications from
        # a resume with no such heading, so this reaches the same answer without
        # spending a request to get there.
        scoped = slice_matching(text, CERTIFICATION)
        if scoped is None:
            return []

        user_msg = f"=== RESUME ===\n{scoped}\n=== END ===\n\nExtract certifications. Return JSON."
        result = await self._call_llm_json(
            CERT_SYSTEM, user_msg, max_tokens=2048,
            section="Certifications",
        )
        if isinstance(result, list):
            return result
        return result.get("certifications", [])
