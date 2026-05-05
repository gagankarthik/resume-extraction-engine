"""CertificationsAgent — focused extraction of certifications."""
from __future__ import annotations
from .base import BaseAgent

CERT_SYSTEM = """Extract ALL certifications, licenses, and certificates from the resume.

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
        user_msg = f"=== RESUME ===\n{text}\n=== END ===\n\nExtract certifications. Return JSON."
        raw, _ = await self._call_llm(CERT_SYSTEM, user_msg, max_tokens=2048)
        result = self._parse_json(raw)
        return result.get("certifications", [])
