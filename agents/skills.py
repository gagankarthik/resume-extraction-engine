"""SkillsAgent — full-document skills extraction across all categories."""
from __future__ import annotations
from .base import BaseAgent

SKILLS_SYSTEM = """Extract ONLY skills that are explicitly named in the resume. Do NOT infer, generate, or extract activities or tasks from job responsibilities.

A skill is a named technology, tool, platform, language, framework, methodology, or concept that appears verbatim (or near-verbatim) in the resume. Job duties, action verbs, and activity phrases are NOT skills.

You will produce TWO views of the same skills inventory:

(A) NORMALIZED CATEGORIES — fixed taxonomy, every skill placed into the best-fit bucket:
- programming_languages: e.g. Python, Java, C#, SQL, R, JavaScript, TypeScript
- frameworks_and_libraries: e.g. React, Django, Spring Boot, TensorFlow, .NET, FastAPI
- databases: e.g. Oracle, SQL Server, PostgreSQL, MongoDB, MySQL, Redshift
- cloud_platforms: e.g. AWS, Azure, GCP, Snowflake, Databricks
- tools_and_platforms: e.g. Jira, Git, Docker, Kubernetes, Tableau, Power BI, Airflow
- operating_systems: e.g. Windows, Linux, macOS, Unix, RHEL
- methodologies: e.g. Agile, Scrum, Kanban, SAFe, ITIL, CI/CD, DevOps
- domain_skills: named domain concepts only — short noun terms like ETL, Data Warehousing, Machine Learning, NLP, Computer Vision. NOT action phrases or multi-clause descriptions. Each entry must be a recognizable industry concept of 1-3 words.
- design_skills: e.g. UI/UX, Figma, Adobe XD, Photoshop
- soft_skills: ONLY if the resume explicitly lists them as skills (e.g. Leadership, Communication)
- other_skills: named skills that don't fit any category above
- all_skills_raw: deduplicated union of all categories above
- technical_skills: deduplicated union of programming_languages through design_skills

(B) VERBATIM CATEGORIES — preserve the resume's own section labels:
- If the resume's skills/technical-skills section uses its OWN category labels (e.g. "Cloud Datawarehouse", "Data Modeling Tool", "ETL Tool", "Big Data Technology"), copy each label and its skills VERBATIM into the `categories` array.
- Use the EXACT label text from the resume (case, punctuation, ampersands, parentheses) — do NOT normalize, expand, or rename.
- Preserve the resume's ORDER of categories.
- If the resume uses a single flat list with NO category labels, leave `categories` as an empty array.
- A skill may appear in BOTH normalized categories AND verbatim categories — the views are independent.

Return ONLY this JSON:
{
  "skills": {
    "all_skills_raw": [], "technical_skills": [], "soft_skills": [],
    "programming_languages": [], "frameworks_and_libraries": [], "databases": [],
    "cloud_platforms": [], "tools_and_platforms": [], "operating_systems": [],
    "methodologies": [], "domain_skills": [], "design_skills": [],
    "languages_spoken": [], "other_skills": [],
    "categories": [
      { "name": "<verbatim label from resume>", "skills": ["skill1", "skill2"] }
    ]
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
