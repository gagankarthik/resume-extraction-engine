"""
Tests for the deterministic parts of CompletenessAuditorAgent.
No API keys or network needed. Run directly:  python tests/test_auditor.py
(also compatible with pytest if installed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.auditor import (  # noqa: E402
    ground_check,
    coverage_report,
    merge_recovered,
    is_grounded,
    _tokens,
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
