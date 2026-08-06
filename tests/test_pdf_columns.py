"""
Reading a real two-column PDF, end to end.

test_columns.py checks the geometry rule against word boxes. This checks the
thing that actually runs: a PDF file, through pdfplumber, through the
normalizer, to the text an agent is handed. The two failures it pins down are
the ones that made two-column resumes quietly unusable —

  * a sidebar entry welded onto the front of a job bullet, and
  * a bullet glyph pushed off the start of its line, which drops the
    programmatic bullet count to zero and silently disables the validator for
    that job.

PDFs are built by hand (see pdfbuild.py) rather than checked in as fixtures, so
the layout under test is visible in the test itself.
"""
from __future__ import annotations

import extractor
from agents.structure import StructureAgent
from extractor import extract_text
from tests.pdfbuild import build_columns, build_single_column

SIDEBAR = ["SKILLS", "Python", "SQL", "Kubernetes", "Terraform", "Docker", "Airflow", "Kafka"]

BODY = [
    "PROFESSIONAL EXPERIENCE",
    "Acme Corp  2019 - Present",
    "Senior Data Engineer",
    "- Built the ingestion pipeline",
    "- Led the migration to Snowflake",
    "- Owned the on-call rotation",
    "Globex Inc  2016 - 2019",
    "Data Engineer",
]

SINGLE_COLUMN = [
    "PROFESSIONAL EXPERIENCE",
    "Acme Corporation      2019 - Present",
    "Senior Data Engineer",
    "- Built and maintained the ingestion pipeline end to end",
    "- Led the migration of the warehouse to Snowflake",
    "- Owned the on-call rotation for the platform team",
]


def _lines(text: str) -> list[str]:
    return text.split("\n")


def test_two_column_page_is_reported_as_such():
    _, _, info = extract_text(build_columns(left=SIDEBAR, right=BODY), "pdf")

    assert info["multi_column_pages"] == [1]


def test_sidebar_is_not_welded_onto_job_bullets():
    text, _, _ = extract_text(build_columns(left=SIDEBAR, right=BODY), "pdf")

    fused = [ln for ln in _lines(text) if "Kubernetes" in ln and "ingestion" in ln]
    assert not fused, f"a skill was merged into a responsibility: {fused}"


def test_bullet_counting_survives_a_two_column_layout():
    """The count is what the validator checks against; zero switches it off."""
    text, _, _ = extract_text(build_columns(left=SIDEBAR, right=BODY), "pdf")

    assert StructureAgent._count_bullets(_lines(text)) == 3


def test_without_column_detection_the_same_file_breaks(monkeypatch):
    """Pins the bug this fix exists for, so a regression is unambiguous."""
    monkeypatch.setattr(extractor, "detect_column_bands", lambda words: None)

    text, _, _ = extract_text(build_columns(left=SIDEBAR, right=BODY), "pdf")

    assert StructureAgent._count_bullets(_lines(text)) == 0
    assert any("Kubernetes" in ln and "ingestion" in ln for ln in _lines(text))


def test_both_columns_keep_all_their_content():
    text, _, _ = extract_text(build_columns(left=SIDEBAR, right=BODY), "pdf")

    for entry in [*SIDEBAR, "Snowflake", "Globex Inc", "Data Engineer"]:
        assert entry in text, f"{entry!r} was lost when the page was split"


def test_single_column_pdf_is_not_split():
    text, _, info = extract_text(build_single_column(SINGLE_COLUMN), "pdf")

    assert info["multi_column_pages"] == []
    assert StructureAgent._count_bullets(_lines(text)) == 3
