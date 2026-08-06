"""
Routing each agent to the part of the resume it is responsible for.

The interesting cases are all failures to divide. Splitting a well-labelled
resume is easy; what matters is that a resume with unusual headings, or none at
all, falls back to the whole document instead of handing an agent an empty
string. A missed speed-up is invisible. A section silently emptied is not.
"""
from __future__ import annotations

from agents.sections import (
    CERTIFICATION,
    EDUCATION,
    EXPERIENCE,
    heading_text,
    slice_excluding,
    slice_matching,
    split_sections,
)

RESUME = """\
Jane Doe
jane@example.com | +1 555 0100

PROFESSIONAL SUMMARY
Data engineer with ten years building ingestion platforms.

PROFESSIONAL EXPERIENCE
Acme Corporation                     2019 - Present
Senior Data Engineer
• Built and maintained the ingestion pipeline
• Led the migration of the warehouse to Snowflake

EDUCATION
B.S. Computer Science, State University, 2008

CERTIFICATIONS
AWS Certified Solutions Architect, 2021
"""


def test_preamble_survives_the_split():
    """The name and contact block belongs to no section and must not vanish."""
    preamble, sections = split_sections(RESUME)

    assert "Jane Doe" in preamble
    assert "jane@example.com" in preamble
    assert len(sections) == 4


def test_education_slice_is_the_education_section():
    slice_ = slice_matching(RESUME, EDUCATION)

    assert slice_ is not None
    assert "State University" in slice_
    # The point of routing: the work history is not in front of the model.
    assert "Snowflake" not in slice_


def test_a_combined_heading_reaches_both_agents():
    """"EDUCATION & CERTIFICATIONS" is one block that two agents need."""
    resume = RESUME.replace("EDUCATION\n", "EDUCATION & CERTIFICATIONS\n").replace(
        "CERTIFICATIONS\nAWS", "AWS"
    )

    assert "State University" in (slice_matching(resume, EDUCATION) or "")
    assert "AWS Certified" in (slice_matching(resume, CERTIFICATION) or "")


def test_no_matching_section_falls_back_to_everything():
    """A resume with no education heading must not yield an empty read."""
    no_education = RESUME.replace("EDUCATION\n", "SCHOOLING HISTORY\n")

    # Nothing matched, so the caller is told to use the whole document.
    assert slice_matching(no_education, EDUCATION) is None


def test_a_heading_with_nothing_under_it_is_not_a_section():
    """A bare heading is a false match, not a section worth trusting."""
    stub = "EXPERIENCE\nAcme Corp\n\nEDUCATION\n"

    assert slice_matching(stub, EDUCATION) is None


def test_excluding_experience_keeps_the_rest():
    remainder = slice_excluding(RESUME, EXPERIENCE)

    assert "Jane Doe" in remainder, "the contact block was dropped"
    assert "State University" in remainder
    assert "Data engineer with ten years" in remainder
    assert "Snowflake" not in remainder, "the work history was not removed"


def test_an_experience_summary_is_not_a_work_history():
    """The heading matches EXPERIENCE, but dropping it loses the summary."""
    resume = RESUME.replace("PROFESSIONAL SUMMARY", "EXPERIENCE SUMMARY")

    remainder = slice_excluding(resume, EXPERIENCE)

    assert "Data engineer with ten years" in remainder
    assert "Snowflake" not in remainder, "the real work history should still go"


def test_a_resume_that_is_all_experience_is_still_read_whole():
    """Removing everything must return the document, not an empty string."""
    only_jobs = "EXPERIENCE\nAcme Corp\nSenior Engineer\n• Built the pipeline\n• Led the migration"

    assert slice_excluding(only_jobs, EXPERIENCE) == only_jobs


def test_body_text_is_not_mistaken_for_a_heading():
    """A sentence mentioning a section name must not split the document."""
    assert heading_text("Led the education technology platform team for three years.") is None
    assert heading_text("• Built the certification tracking service") is None


def test_title_case_headings_are_recognised():
    """Not every resume shouts its headings."""
    assert heading_text("Education", (EDUCATION,)) == "Education"
    assert heading_text("Certifications:", (CERTIFICATION,)) == "Certifications"
