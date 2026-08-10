"""
Tests for the date arithmetic behind the analytics block.

A long career is written in more than one notation — "Feb 1990" early on,
"Feb' 09" in the middle, "Aug 2013" late — and every date that fails to parse
silently contributes zero months. No API keys or network needed.
Run directly:  python tests/test_analytics_dates.py
"""
import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.analytics import AnalyticsAgent, _full_year, _parse_date


def test_four_digit_years_are_unchanged():
    assert _parse_date("Jan 2020") == date(2020, 1, 1)
    assert _parse_date("Feb 1990") == date(1990, 2, 1)
    assert _parse_date("August 2013") == date(2013, 8, 1)


def test_apostrophe_and_two_digit_years_parse():
    assert _parse_date("Feb'09") == date(2009, 2, 1)
    assert _parse_date("Feb' 09") == date(2009, 2, 1)
    assert _parse_date("Nov '01") == date(2001, 11, 1)
    assert _parse_date("Oct 02") == date(2002, 10, 1)
    assert _parse_date("Jun'03") == date(2003, 6, 1)


def test_two_digit_years_land_in_the_past():
    # A resume's dates have already happened, so the pivot is this year.
    assert _full_year("09") == 2009
    assert _full_year("90") == 1990
    assert _full_year("99") == 1999
    assert _full_year("1989") == 1989


def test_a_stranded_separator_does_not_hide_the_month():
    # "Nov - 2010" used to skip past its own month and match the END date's.
    assert _parse_date("Nov - 2010") == date(2010, 11, 1)


def test_present_still_wins_over_everything():
    for value in ("Present", "Till Date", "Current", "ongoing"):
        assert _parse_date(value) == date.today(), value


def test_a_three_decade_career_is_counted_as_one():
    work = [
        {"company_name": "Acme", "start_date": "Feb 1990", "end_date": "Feb 1999"},
        {"company_name": "Beta", "start_date": "Feb' 99", "end_date": "Feb' 01"},
        {"company_name": "Gamma", "start_date": "Oct 02", "end_date": "Apr 03"},
        {"company_name": "Delta", "start_date": "Feb'09", "end_date": "Nov' 10"},
        {"company_name": "Epsilon", "start_date": "Aug 2013", "end_date": "Jun 2015"},
    ]
    analytics = asyncio.run(AnalyticsAgent().run({"work_experience": work}))
    # The first two spans touch and collapse into Feb 1990 - Feb 2001 (132
    # months); then 6 + 21 + 22 = 181 months in all. Only the two four-digit
    # spans used to parse, so this read as 130 months.
    assert analytics["total_months_of_experience"] == 181, analytics
    assert analytics["total_years_of_experience"] == 15.1, analytics
    assert analytics["number_of_roles"] == 5
    assert analytics["number_of_companies"] == 5


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
