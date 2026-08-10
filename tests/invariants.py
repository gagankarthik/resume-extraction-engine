"""
What must be true of ANY correct extraction, whatever the resume.

These are not assertions about one document. Each one is a bug that shipped,
generalised to the rule it broke — a bullet that absorbed a section heading, a
figure nobody wrote, a skill name split down the middle, thirty years of dates
counted as sixteen. A resume that violates none of them may still be imperfect,
but it is free of every failure this engine is known to have.

Two callers share them, and that is the point:

  * tests/test_extraction_golden.py runs them against a replayed cassette, so
    they gate every commit in CI without a network call;
  * tools/evaluate.py runs them against a live model, so "is this model better"
    is answered with a list rather than an impression.

Keep them universal. Anything true only of one fixture belongs in that
fixture's own test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from agents.auditor import _figures, _squash


@dataclass(frozen=True)
class Violation:
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


def bullets_of(extracted: dict) -> list[tuple[str, str]]:
    """(company, bullet) for every responsibility and project bullet."""
    out: list[tuple[str, str]] = []
    for job in extracted.get("work_experience") or []:
        if not isinstance(job, dict):
            continue
        company = job.get("company_name") or "?"
        for bullet in job.get("responsibilities") or []:
            if isinstance(bullet, str):
                out.append((company, bullet))
        for proj in job.get("projects") or []:
            if not isinstance(proj, dict):
                continue
            for bullet in proj.get("projectResponsibilities") or []:
                if isinstance(bullet, str):
                    out.append((company, bullet))
    return out


def skills_of(extracted: dict) -> list[str]:
    skills = extracted.get("skills") or {}
    if not isinstance(skills, dict):
        return []
    every = list(skills.get("all_skills_raw") or [])
    for cat in skills.get("categories") or []:
        if isinstance(cat, dict):
            every.extend(cat.get("skills") or [])
    return [s for s in every if isinstance(s, str)]


# Section scaffolding. A responsibility containing one of these has absorbed a
# heading — which is what unbounded bullet-block folding did, putting another
# job's employer and client inside this job's duties.
_SCAFFOLDING = re.compile(
    r"Employer\s*[-–—:]|Client\s*:|Project Description\s*:|===|Responsibilities\s*:",
    re.I,
)

_DATE_SENTENCE = re.compile(r"(?i)^\s*(since|from)\b")


def universal_violations(extracted: dict, resume_text: str) -> list[Violation]:
    """Every rule broken by this extraction. Empty means nothing known is wrong."""
    out: list[Violation] = []

    def fail(rule: str, detail: str) -> None:
        out.append(Violation(rule, detail))

    bullets = bullets_of(extracted)
    source_squashed = _squash(resume_text)
    source_figures = _figures(resume_text)
    longest_line = max((len(line) for line in resume_text.split("\n")), default=0)

    # ── Identity ────────────────────────────────────────────────────────────
    personal = extracted.get("personal_information") or {}
    name = (personal.get("full_name") or "").strip()
    if not name:
        fail("name-missing", "no full_name was extracted")
    elif _squash(name) not in source_squashed:
        fail("name-not-in-resume", repr(name))
    if re.search(r"[(\[\"]", name):
        fail("name-carries-nickname", repr(name))

    # ── Degrees ─────────────────────────────────────────────────────────────
    for entry in extracted.get("education") or []:
        if not isinstance(entry, dict):
            continue
        degree = (entry.get("degree") or "").strip()
        if not degree:
            continue
        if re.search(r"(?i)\b(bachelor|master|associate|doctor)", degree):
            fail("degree-not-abbreviated", repr(degree))
        if entry.get("degree_type") and entry["degree_type"] != degree:
            fail("degree-disagrees-with-type", f"{degree!r} vs {entry['degree_type']!r}")

    # ── Work history ────────────────────────────────────────────────────────
    for job in extracted.get("work_experience") or []:
        if not isinstance(job, dict):
            continue
        if not (job.get("company_name") or "").strip():
            fail("job-without-employer", repr(job.get("job_title")))
        if job.get("department"):
            label = re.search(r"(?im)^[ \t]*(?:department|dept\.?)[ \t]*[:\-]", resume_text)
            if not label:
                fail("department-invented", repr(job["department"]))
        for field in ("start_date", "end_date"):
            value = job.get(field)
            if isinstance(value, str) and value.strip():
                if _DATE_SENTENCE.match(value):
                    fail("date-is-a-sentence", f"{field}={value!r}")
                if value.strip().endswith(("-", "–", "—")):
                    fail("date-has-dangling-separator", f"{field}={value!r}")
        if job.get("is_current") and job.get("end_date") not in (None, "", "Till Date"):
            fail("current-role-not-till-date", repr(job.get("end_date")))

    # A resume may genuinely hold several stints at one employer — this document
    # has five at Cardinal Health — so the employer alone proves nothing. The
    # same employer STARTING on the same date is one role counted twice, which
    # is what happens when a summary table of engagements is read as jobs in its
    # own right alongside the detailed section describing the same work.
    seen_roles: dict[tuple[str, str], int] = {}
    for job in extracted.get("work_experience") or []:
        if not isinstance(job, dict):
            continue
        key = (_squash(job.get("company_name")), _squash(job.get("start_date")))
        if not key[0] or not key[1]:
            continue
        seen_roles[key] = seen_roles.get(key, 0) + 1
    for (company, start), count in seen_roles.items():
        if count > 1:
            fail("role-listed-twice", f"{company} starting {start} appears {count} times")

    # ── Bullets ─────────────────────────────────────────────────────────────
    for company, bullet in bullets:
        if _SCAFFOLDING.search(bullet):
            fail("bullet-absorbed-a-heading", f"{company}: {bullet[:120]}")
        if len(bullet) > longest_line + 80:
            fail("bullet-spans-multiple-source-lines", f"{company}: {len(bullet)} chars")
        if _squash(bullet) not in source_squashed:
            fail("bullet-not-in-resume", f"{company}: {bullet[:120]}")
        invented = _figures(bullet) - source_figures
        if invented:
            fail("figure-invented", f"{company}: {sorted(invented)} in {bullet[:90]}")

    # ── Skills ──────────────────────────────────────────────────────────────
    for skill in skills_of(extracted):
        if skill.count("(") != skill.count(")"):
            fail("skill-name-split", repr(skill))
        if re.search(r"\.\s+[A-Z]", skill):
            fail("skill-is-prose", repr(skill))
        if len(skill.split()) > 12:
            fail("skill-is-prose", repr(skill))

    # ── Analytics ───────────────────────────────────────────────────────────
    analytics = extracted.get("analytics") or {}
    years = analytics.get("total_years_of_experience")
    roles = extracted.get("work_experience") or []
    if roles and (years is None or years <= 0):
        fail("career-span-not-computed", f"{len(roles)} roles but {years!r} years")

    return out


def summarise(violations: list[Violation]) -> dict[str, int]:
    """How many times each rule was broken, worst first."""
    counts: dict[str, int] = {}
    for v in violations:
        counts[v.rule] = counts.get(v.rule, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
