"""SkillsAgent — full-document skills extraction across all categories."""
from __future__ import annotations
from .base import BaseAgent

SKILLS_SYSTEM = """Extract ALL skills mentioned ANYWHERE in the resume — job descriptions, summaries, project sections, education, profile paragraphs.

Categorize into the following arrays:
- programming_languages: Python, Java, C#, SQL, R, JavaScript, TypeScript, etc.
- frameworks_and_libraries: React, Django, Spring Boot, TensorFlow, .NET, etc.
- databases: Oracle, SQL Server, PostgreSQL, MongoDB, MySQL, etc.
- cloud_platforms: AWS, Azure, GCP, Snowflake, Databricks, etc.
- tools_and_platforms: Jira, Confluence, Git, Docker, Kubernetes, Tableau, Power BI, etc.
- operating_systems: Windows, Linux, macOS, Unix, RHEL, etc.
- methodologies: Agile, Scrum, Kanban, SAFe, TOGAF, ITIL, Waterfall, etc.
- domain_skills: industry-specific knowledge (ETL, Data Warehousing, Risk Management, etc.)
- design_skills: UI/UX, Figma, Adobe XD, Photoshop, etc.
- soft_skills: Leadership, Communication, etc. (ONLY if explicitly stated)
- other_skills: anything that doesn't fit above
- all_skills_raw: union of everything, deduplicated
- technical_skills: union of programming_languages through design_skills, deduplicated

Return ONLY this JSON:
{
  "skills": {
    "all_skills_raw": [], "technical_skills": [], "soft_skills": [],
    "programming_languages": [], "frameworks_and_libraries": [], "databases": [],
    "cloud_platforms": [], "tools_and_platforms": [], "operating_systems": [],
    "methodologies": [], "domain_skills": [], "design_skills": [],
    "languages_spoken": [], "other_skills": []
  }
}
"""


class SkillsAgent(BaseAgent):
    def __init__(self):
        super().__init__("SkillsAgent")

    async def run(self, text: str) -> dict:
        user_msg = f"=== RESUME ===\n{text}\n=== END ===\n\nExtract all skills. Return JSON."
        raw, _ = await self._call_llm(SKILLS_SYSTEM, user_msg, max_tokens=6144)
        result = self._parse_json(raw)
        if isinstance(result, list):
            return {}
        return result.get("skills", result)
