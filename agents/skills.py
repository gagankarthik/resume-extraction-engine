"""
SkillsAgent — the resume's skills, read once and then sorted.

The output schema holds two views of one inventory: the resume's own section
labels copied verbatim, and a fixed taxonomy. Asking for both in one response
means every skill is written out twice, and a dense resume carries hundreds of
them — that response ran to the token ceiling and was cut off, which is
expensive, because a truncated response is re-asked from scratch.

Splitting them into two requests fixed the ceiling but not the cost. Both were
reading the whole document, and the taxonomy pass in particular treats every
technology named anywhere — including in passing inside a job bullet — as a
skill to emit. On a thirty-year consulting resume that is four hundred entries
generated from thirty thousand characters, and it measured at 85-250 seconds
against a 150-second budget for the entire extraction. It timed out on most
runs, which is how a resume with a full skills section arrived with thirteen
empty buckets.

So the passes are a pipeline rather than a race:

  1. INVENTORY reads the document and returns the skills the resume presents as
     skills, under the resume's own labels where it uses them. Ten seconds.
  2. TAXONOMY sorts that list — a few hundred words, not the whole resume —
     into the fixed buckets. It reads no prose, so it cannot invent from prose,
     and its output is bounded by what step 1 found.

Sequential, and still several times faster than the parallel version, because
the expensive half no longer regenerates the document's contents. It also makes
the two views agree by construction: they are now the same list, labelled twice.
A resume that presents no skills section at all falls back to reading the
document for named technologies, which is what step 2 used to do for everyone.
"""
from __future__ import annotations

import logging
import re

from agents import report

from .base import BaseAgent, output_budget

logger = logging.getLogger(__name__)

# What a skill is, and what it is not. Both requests need this, because both can
# invent one from a job duty.
_GROUNDING_RULES = """Extract ONLY skills that are explicitly named in the resume. Do NOT infer, generate, or extract activities or tasks from job responsibilities.

A skill is a named technology, tool, platform, language, framework, methodology, or concept that appears verbatim (or near-verbatim) in the resume. Job duties, action verbs, and activity phrases are NOT skills.

NEVER infer a skill from what a duty implies. "security and access management" is NOT IAM. "containers" is NOT Docker or Kubernetes. "cloud" is NOT AWS or Azure. "reporting" is NOT Power BI or Tableau. If the name is not printed in the resume, it does not belong in ANY category — and never add a related or typical technology alongside one that is written.

Skills sections come in MANY layouts — comma lists, bullet lists, two-column tables, skill matrices, category grids, sidebars. Scan the ENTIRE document for them (a "Technical Skills" / "Skills" / "Tools" section may appear anywhere, including after work experience). Never return empty output when the resume contains a skills section."""


_TAXONOMY = """Sort every skill into this fixed taxonomy, choosing the best-fit bucket:
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
{
  "skills": {
    "soft_skills": [],
    "programming_languages": [], "frameworks_and_libraries": [], "databases": [],
    "cloud_platforms": [], "tools_and_platforms": [], "operating_systems": [],
    "methodologies": [], "domain_skills": [], "design_skills": [],
    "other_skills": []
  }
}
"""


# The fallback for a resume that presents no skills section anywhere: read the
# document for named technologies, which is what the taxonomy pass used to do
# for every resume.
SKILLS_FROM_DOCUMENT_SYSTEM = f"{_GROUNDING_RULES}\n\n{_TAXONOMY}"


# Step 1 — read the document once.
SKILLS_INVENTORY_SYSTEM = f"""{_GROUNDING_RULES}

Preserve the resume's OWN skills-section labels and the skills under each.
- If the resume's skills/technical-skills section uses its own category labels (e.g. "Cloud Datawarehouse", "Data Modeling Tool", "ETL Tool", "Big Data Technology"), copy each label and its skills VERBATIM.
- Use the EXACT label text from the resume (case, punctuation, ampersands, parentheses) — do NOT normalize, expand, or rename.
- Preserve the resume's ORDER of categories.
- Keep a parenthetical that belongs to a name: "Procure to Pay (PTP)" and "SAP APO (DP/SNP)" are each ONE skill, not two.
- uncategorized[] is ONLY for a skills section that carries no labels at all — a single flat list or grid of skill names. If every skills section is labelled, uncategorized[] is []. Never put job-duty text there, and never repeat a skill that is already under a label.

Return ONLY this JSON:
{{
  "categories": [
    {{ "name": "<verbatim label from resume>", "skills": ["skill1", "skill2"] }}
  ],
  "uncategorized": ["skill3", "skill4"]
}}
"""


# Step 2 — sort the list from step 1. No resume text: the skills are already
# decided, and re-reading the document is what made this the slow half.
SKILLS_TAXONOMY_SYSTEM = f"""You are given the list of skills read off a candidate's resume. Sort it.

{_TAXONOMY}

RULES:
- Sort ONLY the skills in the list. Do NOT add, invent, expand, split, merge, or rename any of them — every entry you return must be one of the given strings, character for character.
- Every skill in the list must appear in exactly one bucket. Do not drop any.

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

# ── What a skill entry may look like ────────────────────────────────────────
#
# Two things get into the skills lists that are not skills, and neither comes
# from the extraction prompts — they arrive through the audit's recovery pass,
# which hands back whole resume LINES it found uncovered:
#
#   "Containerization: Docker, Kubernetes, Helm · IaC: Terraform, CDK"
#   "Work Authorization - US Permanent Resident (Green Card). No sponsorship required."
#
# The first is a real skills line that needs taking apart; the second is a
# sentence that is not a skill at all. Both are decidable from the text itself.

# Between categories on a packed line. Deliberately excludes "/" — "CI/CD" and
# "UI/UX" are single skills, not two.
_CATEGORY_SEPARATORS = frozenset("·|•;")

# Within a category. A comma separates skills — unless it is inside brackets,
# where it belongs to the name: "Plan to produce (PTM PP, PP-PI)" and
# "SAP APO (DP/SNP)" are each one skill the resume named, and splitting them
# leaves the reader with "PP-PI)" as a technology.
_ITEM_SEPARATORS = frozenset(",")

_OPENERS, _CLOSERS = "([{", ")]}"

# "Frontend: Next.js, React" — the label names the bucket, not a skill.
_INLINE_LABEL = re.compile(r"^[A-Za-z][\w &+/-]{0,40}\s*:\s*(?=\S)")

# A sentence, not a skill: something ends and something else begins.
_SENTENCE = re.compile(r"\.\s+\S")

# Resume statements that read like a skill line but describe status, not ability.
_NOT_A_SKILL = re.compile(
    r"\b(?:no\s+sponsorship|sponsorship\s+required|work\s+authoriz|permanent\s+resident"
    r"|green\s+card|citizenship|visa\s+status|willing\s+to\s+relocate|available\s+(?:from|immediately)"
    r"|references?\s+available|years?\s+of\s+experience)\b",
    re.I,
)

# Generous, because the cost of the two mistakes is not symmetric: a long entry
# kept is the resume's own wording sitting in the skills list, while one dropped
# is content the candidate wrote and no longer has. Prose is caught by the
# sentence and not-a-skill tests above, which do not depend on length.
_MAX_SKILL_WORDS = 10


def looks_like_skill(value: str) -> bool:
    """True when a string reads as the name of a skill rather than prose."""
    s = (value or "").strip().rstrip(".,;:")
    if not s or not any(c.isalnum() for c in s):
        return False
    if _NOT_A_SKILL.search(s) or _SENTENCE.search(s):
        return False
    return len(s.split()) <= _MAX_SKILL_WORDS


def _split_outside_brackets(value: str, separators: frozenset[str]) -> list[str]:
    """Split on the separators that sit at bracket depth zero."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in value:
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth = max(0, depth - 1)
        elif depth == 0 and ch in separators:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return parts


def split_packed_skills(value: str) -> list[str]:
    """Individual skills from one entry, which may be a whole category line.

    A plain name comes back unchanged in a one-item list. A packed line is
    taken apart on its own punctuation, with the category labels dropped —
    they name the bucket the skills already sit in. Punctuation inside brackets
    is part of the name and is never a split point.
    """
    out: list[str] = []
    for segment in _split_outside_brackets(value or "", _CATEGORY_SEPARATORS):
        segment = _INLINE_LABEL.sub("", segment.strip())
        for item in _split_outside_brackets(segment, _ITEM_SEPARATORS):
            item = item.strip().strip("•").strip().rstrip(".,;:")
            if item:
                out.append(item)
    return out


def tidy_skill_list(items: object) -> tuple[list[str], int, int]:
    """Explode packed lines, drop prose, and de-duplicate one skills bucket.

    Returns (tidied, split_count, dropped_count).
    """
    if not isinstance(items, list):
        return items, 0, 0

    out: list[str] = []
    seen: set[str] = set()
    split = 0
    dropped = 0
    for item in items:
        if not isinstance(item, str) or not item.strip():
            dropped += 1
            continue
        parts = split_packed_skills(item)
        if len(parts) > 1:
            split += 1
        elif not parts:
            dropped += 1
            continue
        for part in parts:
            if not looks_like_skill(part):
                dropped += 1
                continue
            key = part.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(part)
    return out, split, dropped


# The six-plus buckets whose union is technical_skills, in the order the schema
# documents them.
_TECHNICAL_BUCKETS = (
    "programming_languages", "frameworks_and_libraries", "databases",
    "cloud_platforms", "tools_and_platforms", "operating_systems",
    "methodologies", "domain_skills", "design_skills",
)

_ALL_BUCKETS = (*_TECHNICAL_BUCKETS, "soft_skills", "other_skills")


def _dedupe(values: list[str]) -> list[str]:
    """First spelling and order kept."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        key = value.strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value.strip())
    return out


def _union(skills: dict, buckets: tuple[str, ...]) -> list[str]:
    """Deduplicated union of the named buckets, first spelling and order kept."""
    return _dedupe([s for bucket in buckets for s in (skills.get(bucket) or [])])


def _category_skills(skills: dict) -> list[str]:
    """Every skill sitting under the resume's own section labels."""
    out: list[str] = []
    for cat in skills.get("categories") or []:
        if isinstance(cat, dict):
            out.extend(cat.get("skills") or [])
    return _dedupe(out)


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

    categories[] is the same inventory under the resume's own labels, so it is
    NOT folded in — it would add nothing and only risk double-counting. It is
    used when there is nothing else: the two passes fail independently, and when
    the taxonomy pass is the one that fails, every bucket is empty while the
    categories came through intact. An empty union there says the resume lists
    no skills, which is a different and much worse claim than saying they
    arrived ungrouped.
    """
    if not isinstance(skills, dict):
        return skills
    from_categories = _category_skills(skills)
    skills["technical_skills"] = _union(skills, _TECHNICAL_BUCKETS) or from_categories
    skills["all_skills_raw"] = _union(skills, _ALL_BUCKETS) or from_categories
    return skills


# The label a resume puts on the tech stack of ONE job, not on a skills
# category. The inventory pass reads the whole document — it has to, because
# real resumes scatter their skills blocks between the work sections — and
# "Environment: Tosca, SQL Developer, Jira, AutoSys" under a job has exactly
# the shape it is told to copy: a label with skills under it.
#
# It is already extracted as that job's technologies and printed beneath it as
# "Key Technologies/Skills". Copying it a second time gave one submitted resume
# four Technical Skills categories all called "Environment", each repeating a
# different job's stack. The stack belongs to the job; the category is noise.
_PER_JOB_TECH_LABEL = re.compile(
    r"^(?:key\s+)?(?:"
    r"environment|technolog(?:y|ies)|tech\s*stack|technology\s+stack"
    r"|tools?\s+used|software\s+used|platforms?\s+used"
    r")\b(?:\s*[/&,]\s*skills?)?$",
    re.I,
)


def _clean_categories(value: object) -> list[dict]:
    """The categories[] entries worth keeping, whatever shape they arrived in."""
    if isinstance(value, dict):
        value = value.get("categories")
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    for cat in value:
        if not isinstance(cat, dict):
            continue
        skills, _, _ = tidy_skill_list(cat.get("skills"))
        name = cat.get("name")
        if not skills or not isinstance(name, str) or not name.strip():
            continue
        label = name.strip().rstrip(":").strip()
        if _PER_JOB_TECH_LABEL.match(label):
            logger.info("[SkillsAgent] Dropping per-job tech line read as a category: %r", label)
            continue
        out.append({"name": name.strip(), "skills": skills})
    return out


class SkillsAgent(BaseAgent):
    def __init__(self):
        super().__init__("SkillsAgent")

    async def run(self, text: str) -> dict:
        inventory = await self._read_inventory(text)
        named = inventory["named"]

        if named:
            skills = await self._sort_into_taxonomy(named)
        else:
            # No skills section anywhere — the only place left to look is the
            # prose, which is the old behaviour kept for the resumes that need it.
            logger.info("[SkillsAgent] No skills section found — reading the document")
            skills = await self._read_taxonomy_from_document(text)

        if inventory["categories"]:
            skills["categories"] = inventory["categories"]
        return derive_union_fields(skills)

    # ------------------------------------------------------------------ #

    async def _read_inventory(self, text: str) -> dict:
        """The skills the resume presents, with its own labels kept.

        A failure here is not fatal: run() falls through to reading the
        document for named technologies, which is what every resume used to get.
        """
        user_msg = f"=== RESUME ===\n{text}\n=== END ===\n\nList the skills this resume presents. Return JSON."
        try:
            result = await self._call_llm_json(
                SKILLS_INVENTORY_SYSTEM, user_msg, max_tokens=4096, section="Skills"
            )
        except Exception as exc:
            logger.warning("[SkillsAgent] Inventory pass failed: %s", exc)
            return {"categories": [], "named": []}

        if not isinstance(result, dict):
            return {"categories": [], "named": []}

        categories = _clean_categories(result.get("categories", result))
        loose, _, _ = tidy_skill_list(result.get("uncategorized"))
        named = _dedupe([s for cat in categories for s in cat["skills"]] + loose)
        return {"categories": categories, "named": named}

    async def _sort_into_taxonomy(self, named: list[str]) -> dict:
        """Bucket a known list. Never fatal — the list itself is the fallback."""
        user_msg = (
            "=== SKILLS ===\n"
            + "\n".join(f"- {s}" for s in named)
            + "\n=== END ===\n\nSort these into the taxonomy. Return JSON."
        )
        try:
            result = await self._call_llm_json(
                SKILLS_TAXONOMY_SYSTEM, user_msg,
                max_tokens=output_budget("\n".join(named), floor=2048),
                section="Skills (grouping)",
            )
        except Exception as exc:
            logger.warning("[SkillsAgent] Taxonomy pass failed: %s", exc)
            # Recorded, not just logged. The run used to come back "not
            # degraded" with all thirteen buckets empty, which reads as a resume
            # that lists no skills rather than a pass that did not finish.
            report.record(
                "Skills",
                report.Status.PARTIAL,
                detail=(
                    "The skills could not be sorted into categories, so they are "
                    "listed under the resume's own headings instead. Nothing is "
                    "missing — check the grouping if you rely on it."
                ),
            )
            # Ungrouped beats absent. The buckets and categories[] hold the same
            # skills either way — grouping is the only thing lost — and a skill
            # the resume listed without a label lives nowhere else.
            return {"other_skills": named}

        skills = result.get("skills", result) if isinstance(result, dict) else {}
        return skills if isinstance(skills, dict) else {}

    async def _read_taxonomy_from_document(self, text: str) -> dict:
        user_msg = f"=== RESUME ===\n{text}\n=== END ===\n\nExtract all skills. Return JSON."
        result = await self._call_llm_json(
            SKILLS_FROM_DOCUMENT_SYSTEM, user_msg, max_tokens=4096, section="Skills"
        )
        skills = result.get("skills", result) if isinstance(result, dict) else {}
        return skills if isinstance(skills, dict) else {}
