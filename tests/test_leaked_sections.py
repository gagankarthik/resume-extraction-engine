"""
A resume section that ends up inside a job.

When the last job on a page absorbs the section printed after it, the leftover
arrives as a subsection of that job. On a submitted document that read as a
heading "CERTIFICATIONS" in the middle of the final role's duties, followed by
the candidate's five Tricentis credentials — all of which were already in the
certifications table three inches above.

The content is not lost by dropping it: it is extracted properly into
certifications[] / education[] / skills, which is where every template prints
it. What is dropped is the second copy, in the wrong place.

The discrimination this has to get right is between a resume heading and a
label a job legitimately uses. "Certification Testing" is work a QA engineer
does; "Project Description" and "Environment" head real per-job blocks. Only an
exact section name is dropped.
"""
from __future__ import annotations

import pytest

from validator import validate_resume_json

LEAKED = [
    "CERTIFICATIONS",
    "Certifications",
    "Certification",
    "LICENSES",
    "EDUCATION",
    "Academic",
    "Technical Skills",
    "Skills",
    "Core Competencies",
    "PROFESSIONAL SUMMARY",
    "Objective",
    "Work Experience",
    "Publications",
    "References",
]

KEPT = [
    "Environment",
    "Project Description",
    "Responsibilities",
    "Certification Testing",
    "Skills Transfer",
    "Key Achievements",
    "Client",
    "Technologies",
]


def _job(subsection_title: str) -> dict:
    return {
        "work_experience": [
            {
                "company_name": "Societe Generale",
                "job_title": "Automation Test Architect",
                "responsibilities": ["Led the Tosca migration."],
                "subsections": [
                    {"title": subsection_title, "content": ["Tricentis Certified Tosca AS1"]},
                ],
            }
        ]
    }


def _subsection_titles(payload: dict) -> list[str]:
    job = validate_resume_json(payload)[0]["work_experience"][0]
    return [s.get("title") for s in (job.get("subsections") or [])]


@pytest.mark.parametrize("title", LEAKED)
def test_a_resume_section_inside_a_job_is_dropped(title):
    assert _subsection_titles(_job(title)) == []


@pytest.mark.parametrize("title", KEPT)
def test_a_label_a_job_really_uses_survives(title):
    assert _subsection_titles(_job(title)) == [title]


def test_dropping_a_leaked_section_leaves_the_job_intact():
    """Only the subsection goes — the duties it was hiding among stay."""
    job = validate_resume_json(_job("CERTIFICATIONS"))[0]["work_experience"][0]

    assert job["company_name"] == "Societe Generale"
    assert job["responsibilities"] == ["Led the Tosca migration."]


def test_a_job_with_no_subsections_is_unaffected():
    payload = {"work_experience": [{"company_name": "NRG Energy", "responsibilities": ["Owned QA."]}]}
    job = validate_resume_json(payload)[0]["work_experience"][0]

    assert job.get("subsections") == []
    assert job["responsibilities"] == ["Owned QA."]
