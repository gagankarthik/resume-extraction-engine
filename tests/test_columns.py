"""
Finding the gutter on a two-column resume — and not finding one where there
isn't.

Both directions matter, and the second matters more. A missed gutter garbles a
two-column resume; a phantom gutter cuts an ordinary single-column resume in
half, which is worse and affects far more files. The false-positive cases below
are the layouts that make a naive whitespace rule fail: a right-aligned date
column, and a page whose lines happen to break in similar places.

Tests run against word boxes rather than rendered PDFs, because the rule is
about geometry and building real PDFs would test pdfplumber instead.
"""
from __future__ import annotations

from extractor import detect_column_bands

CHAR_W = 6.0
LINE_H = 14.0


def _words(layout: str, origin_x: float = 50.0, origin_y: float = 60.0) -> list[dict]:
    """Word boxes from an ASCII page, where column position is x position.

    Each character cell is CHAR_W wide, so text drawn further right in the
    string sits further right on the page — which is exactly what the detector
    reads.
    """
    words: list[dict] = []
    for row, line in enumerate(layout.split("\n")):
        col = 0
        for token in line.split(" "):
            if token:
                words.append({
                    "x0": origin_x + col * CHAR_W,
                    "x1": origin_x + (col + len(token)) * CHAR_W,
                    "top": origin_y + row * LINE_H,
                })
            col += len(token) + 1
    return words


TWO_COLUMN = """\
SKILLS                              PROFESSIONAL EXPERIENCE
Python                              Acme Corp 2019 - Present
SQL                                 Senior Data Engineer
Kubernetes                          Built the ingestion pipeline
Terraform                           Led the migration to Snowflake
Docker                              Owned the on-call rotation
Airflow                             Globex Inc 2016 - 2019
Spark                               Data Engineer
Kafka                               Maintained the nightly ETL jobs
"""

# The layout that breaks a naive whitespace rule: the gap before the dates is
# wide and appears on several lines, but the bullets run straight through it.
RIGHT_ALIGNED_DATES = """\
PROFESSIONAL EXPERIENCE
Acme Corporation                                       2019 - Present
Senior Data Engineer
Built and maintained the ingestion pipeline end to end
Led the migration of the warehouse to Snowflake
Owned the on-call rotation for the platform team
Globex Incorporated                                    2016 - 2019
Data Engineer
Maintained the nightly ETL jobs and their alerting
Wrote the reconciliation tooling used by finance
"""

SINGLE_COLUMN = """\
PROFESSIONAL EXPERIENCE
Acme Corporation
Senior Data Engineer 2019 - Present
Built and maintained the ingestion pipeline end to end
Led the migration of the warehouse to Snowflake
Owned the on-call rotation for the platform team
Wrote the reconciliation tooling used by the finance team
Mentored two junior engineers through their first year
"""


def test_finds_the_gutter_on_a_two_column_page():
    bands = detect_column_bands(_words(TWO_COLUMN))

    assert bands is not None, "the sidebar and the body were read as one column"
    assert len(bands) == 2

    left, right = bands
    # The sidebar ends well before the body begins.
    assert left[1] < right[0]


def test_right_aligned_dates_are_not_a_gutter():
    """The most dangerous false positive: splitting dates off every job."""
    assert detect_column_bands(_words(RIGHT_ALIGNED_DATES)) is None


def test_ordinary_single_column_page_is_left_alone():
    assert detect_column_bands(_words(SINGLE_COLUMN)) is None


def test_a_nearly_empty_page_is_left_alone():
    """Too little on the page to tell a layout from an accident."""
    assert detect_column_bands(_words("Jane Doe\njane@example.com")) is None


def test_a_lone_heading_beside_a_body_is_not_a_column():
    """One short label to the left of the text is a margin note, not a column."""
    layout = """\
NOTE                                Acme Corporation 2019 - Present
                                    Senior Data Engineer
                                    Built the ingestion pipeline
                                    Led the migration to Snowflake
                                    Owned the on-call rotation
                                    Wrote the reconciliation tooling
"""
    assert detect_column_bands(_words(layout)) is None


def test_bands_cover_the_words_they_claim():
    """Every word must fall inside one of the returned bands.

    A band that clips its own column would silently drop text — the failure
    mode that is hardest to notice downstream, because the output still looks
    like a resume.
    """
    words = _words(TWO_COLUMN)
    bands = detect_column_bands(words)
    assert bands is not None

    for w in words:
        centre = (w["x0"] + w["x1"]) / 2
        assert any(x0 <= centre <= x1 for x0, x1 in bands), (
            f"word at x={centre:.0f} falls in no column"
        )
