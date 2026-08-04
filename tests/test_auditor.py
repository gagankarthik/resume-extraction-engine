"""
Tests for the deterministic parts of CompletenessAuditorAgent.
No API keys or network needed. Run directly:  python tests/test_auditor.py
(also compatible with pytest if installed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.auditor import (
    _figure_key,
    _figures,
    _tokens,
    coverage_report,
    ground_check,
    is_grounded,
    merge_recovered,
    repair_figures,
    scrub_skills,
    source_line_index,
    source_terms,
    tech_is_grounded,
)

RESUME_TEXT = """
John Smith
john.smith@example.com | (555) 123-4567 | linkedin.com/in/john-smith

WORK EXPERIENCE
Acme Corporation                                Jan 2020 - Present
Senior Data Engineer
Client: Diligent Insurance
• Designed and implemented scalable ETL pipelines using Apache Spark and Airflow
• Reduced nightly batch processing time by 40 percent across all regions
• Mentored four junior engineers on data modeling best practices

EDUCATION
Bachelor of Science in Computer Science, State University, 2015
"""


def test_grounded_contacts_kept_and_fakes_dropped():
    merged = {
        "personal_information": {
            "email": ["john.smith@example.com", "fake.person@nowhere.org"],
            "phone": ["(555) 123-4567", "999-888-7777"],
            "linkedin_url": "https://linkedin.com/in/john-smith",
            "github_url": "https://github.com/totally-invented-handle",
        },
        "work_experience": [],
    }
    merged, warnings = ground_check(merged, RESUME_TEXT)
    pi = merged["personal_information"]
    assert pi["email"] == ["john.smith@example.com"], pi["email"]
    assert pi["phone"] == ["(555) 123-4567"], pi["phone"]
    assert pi["linkedin_url"] is not None
    assert pi["github_url"] is None
    assert len(warnings) == 3, warnings


def test_ungrounded_client_name_removed():
    merged = {
        "personal_information": {},
        "work_experience": [{
            "company_name": "Acme Corporation",
            "responsibilities": ["Designed and implemented scalable ETL pipelines"],
            "projects": [{
                "projectName": "Diligent Insurance",       # grounded — appears in text
                "clientName": "Deloitte",                  # hallucinated — NOT in text
                "projectResponsibilities": ["Reduced nightly batch processing time by 40 percent"],
            }],
        }],
    }
    merged, warnings = ground_check(merged, RESUME_TEXT)
    proj = merged["work_experience"][0]["projects"][0]
    assert proj["clientName"] is None
    assert proj["projectName"] == "Diligent Insurance"
    assert any("Deloitte" in w for w in warnings), warnings


def test_invented_project_heading_dropped_but_bullets_kept():
    merged = {
        "personal_information": {},
        "work_experience": [{
            "company_name": "Acme Corporation",
            "responsibilities": [],
            "projects": [{
                "projectName": "Lakehouse Modernization Initiative",  # invented — not in text
                "clientName": None,
                "projectResponsibilities": ["Mentored four junior engineers on data modeling best practices"],
            }],
        }],
    }
    merged, warnings = ground_check(merged, RESUME_TEXT)
    job = merged["work_experience"][0]
    assert job["projects"] == []
    assert job["responsibilities"] == ["Mentored four junior engineers on data modeling best practices"]
    assert any("invented project heading" in w.lower() for w in warnings), warnings


def test_duplicate_bullets_deduped():
    merged = {
        "personal_information": {},
        "work_experience": [{
            "company_name": "Acme",
            "responsibilities": ["Did the thing well", "Did the  thing   well", "Did another thing"],
        }],
    }
    merged, _ = ground_check(merged, RESUME_TEXT)
    assert len(merged["work_experience"][0]["responsibilities"]) == 2


def test_fabricated_metric_bullets_dropped():
    merged = {
        "personal_information": {},
        "work_experience": [{
            "company_name": "Acme Corporation",
            "responsibilities": [
                # Real, grounded bullets — must be KEPT (even with a real metric).
                "Designed and implemented scalable ETL pipelines using Apache Spark and Airflow",
                "Reduced nightly batch processing time by 40 percent across all regions",
                "Mentored four junior engineers on data modeling best practices",
                # AI-fabricated padding — ungrounded + impact/metric → must be DROPPED.
                "Improved release predictability by 40%",
                "Increased sprint velocity by 20%",
                "Delivered measurable cost optimization through license renegotiation and capacity right-sizing",
            ],
            "achievements": ["Accelerated time-to-market for 3 major platform launches"],
        }],
    }
    merged, warnings = ground_check(merged, RESUME_TEXT)
    resp = merged["work_experience"][0]["responsibilities"]
    assert resp == [
        "Designed and implemented scalable ETL pipelines using Apache Spark and Airflow",
        "Reduced nightly batch processing time by 40 percent across all regions",
        "Mentored four junior engineers on data modeling best practices",
    ], resp
    assert merged["work_experience"][0]["achievements"] == [], merged["work_experience"][0]["achievements"]
    assert any("fabricated" in w.lower() for w in warnings), warnings


def test_coverage_detects_missed_lines():
    full = {
        "work_experience": [{
            "company_name": "Acme Corporation",
            "job_title": "Senior Data Engineer",
            "responsibilities": [
                "Designed and implemented scalable ETL pipelines using Apache Spark and Airflow",
                "Reduced nightly batch processing time by 40 percent across all regions",
                "Mentored four junior engineers on data modeling best practices",
            ],
        }],
        "education": [{"institution_name": "State University", "degree": "Bachelor of Science in Computer Science"}],
        "personal_information": {"full_name": "John Smith", "email": ["john.smith@example.com"],
                                 "phone": ["(555) 123-4567"], "linkedin_url": "linkedin.com/in/john-smith",
                                 "address": None},
    }
    pct_full, missed_full = coverage_report(full, RESUME_TEXT)
    assert pct_full >= 80.0, (pct_full, missed_full)

    # Now delete a bullet — coverage must drop and the line must be reported.
    partial = {**full, "work_experience": [{
        **full["work_experience"][0],
        "responsibilities": full["work_experience"][0]["responsibilities"][:1],
    }]}
    pct_partial, missed = coverage_report(partial, RESUME_TEXT)
    assert pct_partial < pct_full
    assert any("Mentored four junior engineers" in m for m in missed), missed


def test_merge_recovered_is_additive_and_deduped():
    merged = {
        "work_experience": [{"company_name": "Acme Corporation", "responsibilities": ["Existing bullet one"]}],
        "education": [{"institution_name": "State University"}],
        "certifications": [],
        "skills": {"all_skills_raw": ["Python"], "other_skills": []},
    }
    recovered = {
        "work_bullets": [{
            "company_name": "Acme",
            "bullets": ["Existing bullet one", "Newly recovered bullet two"],
        }],
        "education": [
            {"institution_name": "State University"},                # dupe — skipped
            {"institution_name": "Community College", "degree": "AA"},
        ],
        "certifications": [{"name": "AWS Certified Solutions Architect"}],
        "skills": ["Python", "Terraform"],
        "professional_summary": "A summary.",
    }
    added = merge_recovered(merged, recovered)
    assert added["work_bullets"] == 1
    assert merged["work_experience"][0]["responsibilities"] == ["Existing bullet one", "Newly recovered bullet two"]
    assert added["education"] == 1 and len(merged["education"]) == 2
    assert added["certifications"] == 1
    assert added["skills"] == 1 and "Terraform" in merged["skills"]["other_skills"]
    assert merged["professional_summary"] == "A summary."

    # Summary must never be overwritten once present.
    merge_recovered(merged, {"professional_summary": "Different."})
    assert merged["professional_summary"] == "A summary."


def test_is_grounded_short_names_strict():
    toks = set(_tokens(RESUME_TEXT))
    assert is_grounded("Diligent Insurance", toks)
    assert not is_grounded("Deloitte", toks)
    assert is_grounded("", toks)  # nothing checkable


# ── Invented-figure guard ───────────────────────────────────────────────────

FIGURE_RESUME = """
WORK EXPERIENCE
Acme Corporation                                Jan 2020 - Present
Senior Data Engineer
• Designed and implemented scalable ETL pipelines using Apache Spark and Airflow
• Cut infrastructure spend by $2 million during the platform consolidation
• Improved dashboard load times by 15% for the analytics team
"""

FIG_SOURCE = _figures(FIGURE_RESUME)
FIG_INDEX = source_line_index(FIGURE_RESUME)


def _repair(bullet):
    return repair_figures(bullet, FIG_SOURCE, FIG_INDEX)


def test_figure_key_normalises_magnitude_and_spacing():
    # "$2 million" and "$2M" are the same claim written two ways.
    assert _figure_key("$2 million") == _figure_key("$2M")
    assert _figure_key("40 %") == _figure_key("40%")
    assert _figure_key("2.50%") == _figure_key("2.5%")
    assert _figure_key("15%") != _figure_key("50%")


def test_bullet_with_source_figures_is_untouched():
    # Every figure is written in the resume — nothing to repair.
    bullet = "Improved dashboard load times by 15% for the analytics team"
    assert _repair(bullet) == bullet
    assert _repair("Cut infrastructure spend by $2 million during the platform consolidation") is not None


def test_bullet_without_figures_is_untouched():
    bullet = "Designed and implemented scalable ETL pipelines using Apache Spark and Airflow"
    assert _repair(bullet) == bullet


def test_invented_percentage_restores_the_resume_line():
    # The model kept the real bullet and stapled a number the resume never gave.
    padded = (
        "Designed and implemented scalable ETL pipelines using Apache Spark "
        "and Airflow, improving throughput by 40%"
    )
    assert _repair(padded) == (
        "Designed and implemented scalable ETL pipelines using Apache Spark and Airflow"
    )


def test_invented_figure_is_excised_when_no_source_line_matches():
    # No resume line to fall back on, so the invented clause is cut and the
    # real head survives verbatim — no rewording, no added punctuation.
    padded = "Owned the quarterly capacity planning cycle, reducing cloud cost by 62%"
    assert _repair(padded) == "Owned the quarterly capacity planning cycle"


def test_wholly_invented_metric_bullet_is_dropped():
    # Nothing to anchor to and nothing worth keeping before the figure.
    assert _repair("Boosted revenue 300%") is None


def test_year_and_version_numbers_are_not_treated_as_figures():
    # Bare integers must not trip the guard: years, versions, team counts.
    assert _figures("Migrated to Python 3.9 in 2021 with a team of 12") == set()


# ── Technology guard ────────────────────────────────────────────────────────

TECH_RESUME = """
William Reed
WORK EXPERIENCE
Acme Corporation
Security Engineer
• Led security and access management across the payments platform
• Built reporting dashboards and automated the release pipeline
• Environment: Python, Node.js, CI/CD, C#, Go
"""

TECH_SRC = source_terms(TECH_RESUME)


def test_technology_named_in_resume_is_kept():
    for named in ("Python", "C#", "Go"):
        assert tech_is_grounded(named, TECH_SRC), named


def test_technology_inferred_from_a_duty_is_rejected():
    # The resume says "security and access management" and "reporting"; it never
    # names the products those duties imply.
    for inferred in ("IAM", "Kubernetes", "Power BI", "AWS", "Docker"):
        assert not tech_is_grounded(inferred, TECH_SRC), inferred


def test_short_acronym_is_not_matched_inside_a_word():
    # "William" contains "iam" — a substring check would wrongly keep IAM.
    assert not tech_is_grounded("IAM", TECH_SRC)


def test_spelling_variant_of_a_named_technology_survives():
    # Written "Node.js" / "CI/CD"; returned without the punctuation.
    assert tech_is_grounded("NodeJS", TECH_SRC)
    assert tech_is_grounded("CICD", TECH_SRC)


def test_ground_check_drops_inferred_job_technologies():
    merged = {
        "work_experience": [{
            "company_name": "Acme Corporation",
            "technologies_used": ["Python", "IAM", "Kubernetes", "Node.js"],
        }],
    }
    merged, warnings = ground_check(merged, TECH_RESUME)
    kept = merged["work_experience"][0]["technologies_used"]
    assert kept == ["Python", "Node.js"]
    assert any("not named in the resume" in w for w in warnings)


def test_scrub_skills_drops_inferred_skills_across_buckets():
    skills = {
        "programming_languages": ["Python", "Ruby"],
        "cloud_platforms": ["AWS"],
        "categories": [{"name": "Security", "skills": ["IAM", "C#"]}],
    }
    dropped = scrub_skills(skills, TECH_SRC)
    assert skills["programming_languages"] == ["Python"]
    assert skills["cloud_platforms"] == []
    assert skills["categories"][0]["skills"] == ["C#"]
    assert dropped == 3


def test_ground_check_strips_invented_percentage_from_a_job():
    merged = {
        "work_experience": [{
            "company_name": "Acme Corporation",
            "responsibilities": [
                "Designed and implemented scalable ETL pipelines using Apache "
                "Spark and Airflow, improving throughput by 40%",
                "Improved dashboard load times by 15% for the analytics team",
            ],
        }],
    }
    merged, warnings = ground_check(merged, FIGURE_RESUME)
    resp = merged["work_experience"][0]["responsibilities"]

    # The invented 40% is gone; the real 15% the resume states is preserved.
    assert not any("40%" in r for r in resp)
    assert any("15%" in r for r in resp)
    assert len(resp) == 2  # the real bullet was repaired, not dropped
    assert any("invented figures" in w for w in warnings)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok: {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL: {name}: {exc}")
    if failures:
        sys.exit(1)
    print("all auditor tests passed")
