"""
SkillsAgent — full-document skills extraction across all categories.

The output schema holds two views of one inventory: a fixed taxonomy, and the
resume's own section labels copied verbatim. Asking for both in one response
means every skill on the resume is written out twice, and a dense engineering
resume carries two hundred of them. That response ran to the token ceiling and
was cut off, which costs far more than the tokens — a truncated response is
re-asked from scratch with double the budget.

The two views are independent, so they are now two requests that run at the same
time. Each is half the size, neither comes near the ceiling, and the pair
finishes in about the time the slower one takes rather than the sum.
"""
from __future__ import annotations

import asyncio
import logging

from .base import BaseAgent

logger = logging.getLogger(__name__)

# What a skill is, and what it is not. Both requests need this, because both can
# invent one from a job duty.
_GROUNDING_RULES = """Extract ONLY skills that are explicitly named in the resume. Do NOT infer, generate, or extract activities or tasks from job responsibilities.

A skill is a named technology, tool, platform, language, framework, methodology, or concept that appears verbatim (or near-verbatim) in the resume. Job duties, action verbs, and activity phrases are NOT skills.

NEVER infer a skill from what a duty implies. "security and access management" is NOT IAM. "containers" is NOT Docker or Kubernetes. "cloud" is NOT AWS or Azure. "reporting" is NOT Power BI or Tableau. If the name is not printed in the resume, it does not belong in ANY category — and never add a related or typical technology alongside one that is written.

Skills sections come in MANY layouts — comma lists, bullet lists, two-column tables, skill matrices, category grids, sidebars. Scan the ENTIRE document for them (a "Technical Skills" / "Skills" / "Tools" section may appear anywhere, including after work experience). Never return empty output when the resume contains a skills section."""


SKILLS_NORMALIZED_SYSTEM = f"""{_GROUNDING_RULES}

Sort every skill into this fixed taxonomy, choosing the best-fit bucket:
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

Place each skill in exactly ONE bucket. Do not repeat it in another, and do not
return a combined or "all skills" list — those are assembled from what you return.

Return ONLY this JSON:
{{
  "skills": {{
    "soft_skills": [],
    "programming_languages": [], "frameworks_and_libraries": [], "databases": [],
    "cloud_platforms": [], "tools_and_platforms": [], "operating_systems": [],
    "methodologies": [], "domain_skills": [], "design_skills": [],
    "other_skills": []
  }}
}}
"""


SKILLS_VERBATIM_SYSTEM = f"""{_GROUNDING_RULES}

Preserve the resume's OWN skills-section labels and the skills under each.
- If the resume's skills/technical-skills section uses its own category labels (e.g. "Cloud Datawarehouse", "Data Modeling Tool", "ETL Tool", "Big Data Technology"), copy each label and its skills VERBATIM.
- Use the EXACT label text from the resume (case, punctuation, ampersands, parentheses) — do NOT normalize, expand, or rename.
- Preserve the resume's ORDER of categories.
- If the resume uses a single flat list with NO category labels, return an empty array.

Return ONLY this JSON:
{{
  "categories": [
    {{ "name": "<verbatim label from resume>", "skills": ["skill1", "skill2"] }}
  ]
}}
"""

# The six-plus buckets whose union is technical_skills, in the order the schema
# documents them.
_TECHNICAL_BUCKETS = (
    "programming_languages", "frameworks_and_libraries", "databases",
    "cloud_platforms", "tools_and_platforms", "operating_systems",
    "methodologies", "domain_skills", "design_skills",
)

_ALL_BUCKETS = (*_TECHNICAL_BUCKETS, "soft_skills", "other_skills")


def _union(skills: dict, buckets: tuple[str, ...]) -> list[str]:
    """Deduplicated union of the named buckets, first spelling and order kept."""
    seen: set[str] = set()
    out: list[str] = []
    for bucket in buckets:
        for skill in skills.get(bucket) or []:
            if not isinstance(skill, str) or not skill.strip():
                continue
            key = skill.strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(skill.strip())
    return out


def derive_union_fields(skills: dict) -> dict:
    """Fill technical_skills and all_skills_raw from the buckets.

    These two used to be asked of the model, which meant every skill on the
    resume was generated three or four times over — once in its bucket, again in
    each union, and again under the resume's own label in categories[]. On a
    dense engineering resume that alone ran the response past its token ceiling,
    and a response that runs out of room is re-asked from scratch with double
    the budget. One agent was spending minutes emitting the same words.

    A union is arithmetic. Computing it here also makes it correct by
    construction: the model used to be able to list a skill in a bucket and omit
    it from the union, and nothing downstream would notice.
    """
    if not isinstance(skills, dict):
        return skills
    skills["technical_skills"] = _union(skills, _TECHNICAL_BUCKETS)
    skills["all_skills_raw"] = _union(skills, _ALL_BUCKETS)
    return skills


class SkillsAgent(BaseAgent):
    def __init__(self):
        super().__init__("SkillsAgent")

    async def run(self, text: str) -> dict:
        user_msg = f"=== RESUME ===\n{text}\n=== END ===\n\nExtract all skills. Return JSON."

        normalized, verbatim = await asyncio.gather(
            self._call_llm_json(
                SKILLS_NORMALIZED_SYSTEM, user_msg, max_tokens=4096, section="Skills"
            ),
            self._call_llm_json(
                SKILLS_VERBATIM_SYSTEM, user_msg, max_tokens=4096, section="Skills (labels)"
            ),
            # The taxonomy is the part downstream depends on; the resume's own
            # labels are a presentation nicety. Losing one must not lose both.
            return_exceptions=True,
        )

        skills: dict = {}
        if isinstance(normalized, dict):
            skills = normalized.get("skills", normalized)
        elif isinstance(normalized, Exception):
            logger.warning("[SkillsAgent] Normalized pass failed: %s", normalized)
        if not isinstance(skills, dict):
            skills = {}

        if isinstance(verbatim, dict):
            categories = verbatim.get("categories", verbatim.get("skills", {}))
            if isinstance(categories, dict):
                categories = categories.get("categories")
            if isinstance(categories, list):
                skills["categories"] = categories
        elif isinstance(verbatim, Exception):
            logger.warning("[SkillsAgent] Verbatim-label pass failed: %s", verbatim)

        # A total failure has to surface as one, not as a resume with no skills.
        if not skills and isinstance(normalized, Exception):
            raise normalized

        return derive_union_fields(skills)
