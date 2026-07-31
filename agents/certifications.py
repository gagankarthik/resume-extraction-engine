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
        result = await self._call_llm_json(
            CERT_SYSTEM, user_msg, max_tokens=2048,
            section="Certifications",
        )
        if isinstance(result, list):
            return result
        return result.get("certifications", [])
