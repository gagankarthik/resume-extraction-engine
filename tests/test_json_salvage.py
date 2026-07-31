"""
Tests for recovering data from truncated model responses.

No API keys or network needed. Run directly:  python tests/test_json_salvage.py
(also compatible with pytest if installed)

The case that matters most is a long resume whose job array is cut mid-record.
Before salvage, that response failed to parse, the agent raised, the
orchestrator swallowed it, and every job vanished. These tests pin down that
the complete records survive.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.json_salvage import (
    UnsalvageableJSON,
    count_items,
    salvage,
)

logging.disable(logging.WARNING)


# ── Responses that arrived whole ──────────────────────────────────────────

def test_clean_json_is_not_marked_repaired():
    result = salvage('{"work_experience":[{"company_name":"Acme"}]}')
    assert result.repaired is False
    assert result.data["work_experience"][0]["company_name"] == "Acme"


def test_markdown_fences_are_stripped():
    result = salvage('```json\n{"a":[1,2,3]}\n```')
    assert result.repaired is False
    assert result.data == {"a": [1, 2, 3]}


def test_prose_around_json_is_ignored():
    result = salvage('Here is the result:\n{"x":[1,2]}')
    assert result.repaired is False
    assert result.data == {"x": [1, 2]}


# ── Responses that were cut off ───────────────────────────────────────────

def test_cut_mid_string_keeps_the_completed_records():
    result = salvage(
        '{"work_experience":[{"company_name":"Acme",'
        '"responsibilities":["Did a thing","Did another thi'
    )
    assert result.repaired is True
    assert result.data["work_experience"][0]["responsibilities"] == ["Did a thing"]


def test_cut_after_complete_elements_keeps_all_of_them():
    result = salvage('{"bullets":["one","two","three","fou')
    assert result.repaired is True
    assert result.data["bullets"] == ["one", "two", "three"]


def test_cut_directly_after_a_key_backs_off_to_the_previous_record():
    # A naive "cut at the last closed string" drops here: it leaves {"title"
    # with no value. The salvager must step further back.
    result = salvage('{"jobs":[{"c":"A"},{"c":"B"},{"title"')
    assert result.repaired is True
    assert result.data["jobs"] == [{"c": "A"}, {"c": "B"}]


def test_escaped_quotes_do_not_confuse_the_scanner():
    result = salvage('{"b":["say \\"hi\\" now","second","thir')
    assert result.repaired is True
    assert result.data["b"] == ['say "hi" now', "second"]


def test_nested_structures_are_closed_in_order():
    result = salvage('{"a":{"b":{"c":["x","y","z"')
    assert result.repaired is True
    assert result.data["a"]["b"]["c"] == ["x", "y", "z"]


def test_long_resume_cut_in_the_final_job_keeps_every_earlier_job():
    """
    The regression this module exists for: eight jobs, cut partway through the
    eighth. All eight survive, the eighth carrying the bullets that arrived.
    """
    complete = ",".join(
        '{"company_name":"Co%d","responsibilities":["r1","r2","r3"]}' % i
        for i in range(7)
    )
    payload = (
        '{"work_experience":['
        + complete
        + ',{"company_name":"Co7","responsibilities":["r1","r'
    )

    result = salvage(payload)
    jobs = result.data["work_experience"]

    assert result.repaired is True
    assert len(jobs) == 8, "earlier jobs must not be lost to a cut in the last one"
    assert jobs[6]["responsibilities"] == ["r1", "r2", "r3"]
    # The truncated record keeps what it had. It is reported as PARTIAL so the
    # reviewer knows to check it against the source.
    assert jobs[7]["responsibilities"] == ["r1"]


# ── Responses with nothing to recover ─────────────────────────────────────

def test_a_refusal_raises_rather_than_returning_junk():
    try:
        salvage("I cannot help with that.")
    except UnsalvageableJSON:
        return
    raise AssertionError("expected UnsalvageableJSON")


def test_cut_before_the_first_value_raises():
    try:
        salvage('{"a":')
    except UnsalvageableJSON:
        return
    raise AssertionError("expected UnsalvageableJSON")


def test_empty_response_raises():
    try:
        salvage("   ")
    except UnsalvageableJSON:
        return
    raise AssertionError("expected UnsalvageableJSON")


# ── Reporting helper ──────────────────────────────────────────────────────

def test_count_items_counts_records_not_keys():
    assert count_items([1, 2, 3]) == 3
    assert count_items({"jobs": [1, 2], "skills": [1, 2, 3]}) == 5
    assert count_items("not a container") == 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failure(s)'}")
    sys.exit(1 if failures else 0)
