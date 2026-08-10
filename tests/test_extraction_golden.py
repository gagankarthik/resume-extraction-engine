"""
One real resume, all the way through, with the model replayed from a cassette.

This is the test the suite was missing. Everything else here checks a function;
this checks the ANSWER — the JSON a person would be handed — and it does so
without a network call, so it runs on every change in under a second.

The resume behind it is a thirty-five-year SAP consulting CV: 397 lines, 33
roles, dates written four different ways, a skills section, two summary tables,
and a long stretch where the author stops using bullet glyphs. Every bug this
engine has had showed up on a document like it.

It is a real resume with the candidate's identity replaced — the name, email
and phone are invented, and nothing else was touched. The structure is what
makes it worth testing against, and the structure is intact; the person who
sent it in is not in a public repository.

WHAT IS ASSERTED, AND WHAT IS NOT

Not equality against a recorded blob. A golden file that fails whenever the
output improves gets regenerated without being read, and then it is not a test.
What is pinned here are properties that must hold for ANY correct extraction of
this document:

  * the candidate is named, contactable, and their degrees are abbreviated;
  * every role is present and every bullet traces back to the resume;
  * no bullet has absorbed another job's header, and no figure was invented;
  * the skills the resume lists are all there, spelled as it spells them;
  * a career that ran from 1990 is reported as thirty-odd years, not sixteen.

Each of those is a bug that shipped. Together they are the shape of a correct
answer, and a change that improves the extraction can still satisfy all of them.
"""
from __future__ import annotations

import asyncio
import os
import re

import pytest
from invariants import summarise, universal_violations
from replay import FIXTURES, Cassette, ReplayProvider

from agents import report
from agents.deadline import Deadline
from agents.llm.client import LLMClient
from config import LLMSettings

FIXTURE = "long-career"

# Generous, and irrelevant: the replay provider answers instantly, so the only
# thing this bounds is a hang in our own code.
BUDGET_SECONDS = 120.0


def _settings() -> LLMSettings:
    return LLMSettings(
        model="replay",
        max_concurrent=12, max_output_tokens=32000, call_timeout_seconds=90,
        transport_retries=0, truncation_escalations=2,
    )


@pytest.fixture(scope="module")
def resume_text() -> str:
    return (FIXTURES / f"{FIXTURE}.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def extracted(resume_text) -> dict:
    """The pipeline's answer for this resume, model replayed."""
    cassette = Cassette.load(FIXTURES / f"{FIXTURE}.cassette.json")
    provider = ReplayProvider(cassette)
    client = LLMClient(settings=_settings(), provider=provider)

    import agents.base as base
    import agents.llm.client as llm_client

    original = (base.get_client, llm_client.get_client)
    # The audit's recovery pass asks the model about whatever the coverage check
    # happened to miss, so its question changes with every improvement upstream
    # and no recording can cover it for long. It is exercised by unit tests
    # instead; here it would only produce cassette misses that say nothing.
    previous_recovery = os.environ.get("ENABLE_AUDIT_RECOVERY")
    os.environ["ENABLE_AUDIT_RECOVERY"] = "false"

    base.get_client = lambda: client
    llm_client.get_client = lambda: client
    try:
        from orchestrator import ResumeOrchestrator

        async def main() -> dict:
            report.reset_report()
            return await ResumeOrchestrator().run(resume_text, Deadline.in_seconds(BUDGET_SECONDS))

        return asyncio.run(main())
    finally:
        base.get_client, llm_client.get_client = original
        if previous_recovery is None:
            os.environ.pop("ENABLE_AUDIT_RECOVERY", None)
        else:
            os.environ["ENABLE_AUDIT_RECOVERY"] = previous_recovery


def _jobs(extracted) -> list[dict]:
    return extracted["work_experience"]


def _all_bullets(extracted) -> list[tuple[str, str]]:
    """(company, bullet) for every responsibility and project bullet."""
    out = []
    for job in _jobs(extracted):
        company = job.get("company_name") or "?"
        for bullet in job.get("responsibilities") or []:
            out.append((company, bullet))
        for proj in job.get("projects") or []:
            for bullet in proj.get("projectResponsibilities") or []:
                out.append((company, bullet))
    return out


# ── Who the candidate is ────────────────────────────────────────────────────

def test_the_candidate_is_identified(extracted):
    pi = extracted["personal_information"]
    assert pi["full_name"] == "Anil K Verma"
    assert pi["first_name"] == "Anil"
    assert pi["last_name"] == "Verma"
    assert pi["email"] == ["anil.verma@example.com"]
    assert pi["phone"], "the resume prints a cell number"
    assert "5550100" in pi["phone"][0].replace(" ", "").replace("-", "")


def test_the_summary_survives(extracted):
    summary = extracted.get("professional_summary") or ""
    assert len(summary) > 200, "the resume opens with several paragraphs about the candidate"
    assert "SAP" in summary


# ── Education ───────────────────────────────────────────────────────────────

def test_both_degrees_are_present_and_abbreviated(extracted):
    education = extracted["education"]
    assert len(education) == 2, [e.get("degree") for e in education]

    degrees = {e.get("degree") for e in education}
    assert degrees == {"MS", "MBA"}, degrees
    # The long form is what the resume writes; the output must not carry it.
    for entry in education:
        assert "Master" not in (entry.get("degree") or "")
        assert entry.get("degree") == entry.get("degree_type")

    institutions = " ".join(e.get("institution_name") or "" for e in education)
    assert "IGNOU" in institutions
    assert "Rani" in institutions


# ── Work history ────────────────────────────────────────────────────────────

def test_every_role_is_kept(extracted):
    jobs = _jobs(extracted)
    # 29 SAP engagements plus 3 manufacturing roles, less the one the resume
    # lists twice. Fewer than this means roles were dropped.
    assert len(jobs) >= 28, len(jobs)
    for job in jobs:
        assert (job.get("company_name") or "").strip(), job


def test_the_oldest_and_newest_roles_both_made_it(extracted):
    companies = " | ".join((j.get("company_name") or "") for j in _jobs(extracted))
    # The current engagement, and the 1990 one at the far end of the document.
    assert "Aircraft" in companies
    assert "Shaw Wallace" in companies
    assert "Boeing" in companies


def test_the_current_role_reads_till_date(extracted):
    current = [j for j in _jobs(extracted) if j.get("is_current")]
    assert current, "the resume has an engagement running to today"
    for job in current:
        assert job["end_date"] == "Till Date", job["end_date"]


def test_dates_are_not_left_as_sentences(extracted):
    for job in _jobs(extracted):
        for field in ("start_date", "end_date"):
            value = job.get(field)
            if not value:
                continue
            assert not re.match(r"(?i)^\s*(since|from)\b", value), (field, value)
            assert not value.strip().endswith("-"), (field, value)


def test_the_work_history_carries_its_bullets(extracted):
    bullets = _all_bullets(extracted)
    assert len(bullets) >= 150, len(bullets)


# ── The bugs that shipped ───────────────────────────────────────────────────

def test_every_universal_invariant_holds(extracted, resume_text):
    """The whole rule set from tests/invariants.py, in one assertion.

    Bullets that absorbed a heading or ran over several jobs, figures nobody
    wrote, degrees left in longhand, skill names split down the middle, a
    career span of zero — each was a real defect, and each is checked here
    against the same rules `tools/evaluate.py` applies to a live model.
    """
    violations = universal_violations(extracted, resume_text)
    assert violations == [], (
        f"{len(violations)} violation(s): {summarise(violations)}\n  "
        + "\n  ".join(str(v) for v in violations[:10])
    )


def test_no_department_was_conjured_from_a_client(extracted):
    # This resume labels no department anywhere, so every one of these would be
    # a client or an employer that landed in the wrong field.
    departments = [j.get("department") for j in _jobs(extracted) if j.get("department")]
    assert departments == [], departments


# ── Skills ──────────────────────────────────────────────────────────────────

def test_the_skills_section_is_not_empty(extracted):
    """Thirteen empty buckets is what a timed-out taxonomy pass looks like."""
    skills = extracted["skills"]
    assert len(skills["technical_skills"]) >= 20, skills["technical_skills"]
    assert len(skills["all_skills_raw"]) >= 20


def test_the_resumes_own_skill_headings_are_kept(extracted):
    names = [c["name"] for c in extracted["skills"]["categories"]]
    assert any("Skill sets" in n for n in names), names
    assert len(names) >= 2, names


def test_skill_names_are_not_split_down_the_middle(extracted):
    """Punctuation inside brackets belongs to the name.

    Splitting "Plan to produce (PTM PP, PP-PI)" on its comma left "PP-PI)" in
    the skills list — a technology that does not exist.
    """
    skills = extracted["skills"]
    every = skills["all_skills_raw"] + [
        s for c in skills["categories"] for s in c["skills"]
    ]
    for skill in every:
        assert skill.count("(") == skill.count(")"), skill
        assert not skill.startswith(")"), skill


def test_skills_are_named_not_narrated(extracted):
    """A sentence in the skills list is a line the resume was never asked for."""
    for skill in extracted["skills"]["all_skills_raw"]:
        assert not re.search(r"\.\s+[A-Z]", skill), skill
        assert len(skill.split()) <= 12, skill


# ── Analytics ───────────────────────────────────────────────────────────────

def test_the_career_span_matches_the_resume(extracted):
    """The summary says "more than 3 decades"; the dates say 1990 onwards.

    Two-digit years used to parse as nothing, so most of this career counted as
    zero months and thirty-five years was reported as sixteen.
    """
    analytics = extracted["analytics"]
    assert 30 <= analytics["total_years_of_experience"] <= 40, analytics
    assert analytics["number_of_roles"] == len(_jobs(extracted))


# ── The run as a whole ──────────────────────────────────────────────────────

def test_the_extraction_covers_the_document(extracted):
    audit = extracted.get("_audit") or {}
    assert audit.get("coverage_percent", 0) >= 95.0, audit.get("coverage_percent")


def test_nothing_was_dropped_for_want_of_a_recorded_answer(extracted):
    """A cassette miss degrades a section silently; say so instead."""
    run = extracted.get("_extraction_report") or {}
    assert run.get("failed_sections") == [], run.get("failed_sections")
