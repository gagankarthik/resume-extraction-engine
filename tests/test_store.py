"""The extraction store: what an item holds, and that a failed save never fails an upload."""
import json
import zlib
from datetime import datetime, timezone

import store

RESULT = {
    "personal_information": {"full_name": "Alex Sample", "email": ["alex@example.test"]},
    "work_experience": [{"company_name": "Acme", "job_title": "Engineer", "location": "Columbus, OH"}],
    "_metadata": {"request_id": "req-123"},
}


def _fields(**over):
    base = dict(source="hire", subject="sub-1", origin="https://hire.oceanbluecorp.com",
                file_name="alex.pdf", file_type="pdf", client_id=None, project_id=None,
                now=datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc))
    base.update(over)
    return base


def test_item_round_trips_the_full_result():
    item = store.build_item(RESULT, **_fields())
    assert item["id"]["S"] == "req-123"
    assert item["source"]["S"] == "hire"
    assert item["gsi1pk"]["S"] == "EXTRACTION"
    assert item["gsi1sk"]["S"].startswith("2026-09-25T12:00:00")
    assert json.loads(zlib.decompress(item["result"]["B"])) == RESULT


def test_source_falls_back_to_origin_then_unknown():
    assert store.build_item(RESULT, **_fields(source=None))["source"]["S"] == "https://hire.oceanbluecorp.com"
    assert store.build_item(RESULT, **_fields(source=None, origin=None))["source"]["S"] == "unknown"


def test_disabled_without_a_table(monkeypatch):
    monkeypatch.delenv("EXTRACTIONS_TABLE", raising=False)
    assert store.save_extraction(RESULT, **_fields()) is None


def test_a_failed_write_is_swallowed(monkeypatch):
    class Boom:
        def put_item(self, **_):
            raise RuntimeError("dynamo down")

    monkeypatch.setenv("EXTRACTIONS_TABLE", "resume-extractions")
    monkeypatch.setattr(store, "_client", Boom())
    assert store.save_extraction(RESULT, **_fields()) is None
