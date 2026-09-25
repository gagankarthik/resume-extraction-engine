"""
Keep every extraction in one place.

WHY THIS EXISTS

Several products send resumes to this engine — the Hire tool, the company
website, the matching service. Each used to keep (or not keep) its own copy of
the result, so building anything across all of them (the talent heat map) meant
reading every product's tables, or paying to extract the same resume again.

Now the engine writes each successful extraction to a single DynamoDB table.
Anything that needs the whole picture reads that one table.

WHAT IS STORED

The full structured result, exactly as returned to the caller, compressed. Plus
who asked (the ticket's subject), which product (the `source` query parameter
or the calling origin), the file's name and type, and when. The uploaded file
itself is never stored.

FAILING OPEN, DELIBERATELY

Saving is a side effect. If the table is unset, unreachable, or the write
fails, the caller still gets their result — a person waiting on an upload must
never lose it because a log write failed. Failures are logged loudly instead.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
import zlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Items above this are skipped rather than failing the write: DynamoDB's hard
# limit is 400 KB, and a compressed resume is typically 5-30 KB.
_MAX_BLOB_BYTES = 350 * 1024

_client = None


def _table_name() -> str:
    return os.getenv("EXTRACTIONS_TABLE", "").strip()


def enabled() -> bool:
    return bool(_table_name())


def _dynamo():
    # Imported here so the service (and its tests) run without boto3 when the
    # table is not configured. The Lambda runtime ships boto3.
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config

        _client = boto3.client(
            "dynamodb",
            region_name=os.getenv("AWS_REGION", "us-east-2"),
            config=Config(connect_timeout=2, read_timeout=4, retries={"max_attempts": 2}),
        )
    return _client


def build_item(
    result: dict,
    *,
    source: str | None,
    subject: str | None,
    origin: str | None,
    file_name: str,
    file_type: str,
    client_id: str | None,
    project_id: str | None,
    now: datetime | None = None,
    extraction_id: str | None = None,
) -> dict | None:
    """The DynamoDB item for one extraction, or None if it is too large to keep.

    `extraction_id` pins the key, for imports that must be safe to re-run.
    """
    now = now or datetime.now(timezone.utc)
    created = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    meta = result.get("_metadata") if isinstance(result, dict) else None
    request_id = (meta or {}).get("request_id") if isinstance(meta, dict) else None
    extraction_id = str(extraction_id or request_id or uuid.uuid4())

    blob = zlib.compress(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6)
    if len(blob) > _MAX_BLOB_BYTES:
        logger.warning("[store] extraction %s is %d KB compressed; not stored", extraction_id, len(blob) // 1024)
        return None

    item: dict = {
        "id": {"S": extraction_id},
        "createdAt": {"S": created},
        # One partition for the time index: every reader asks "what is new since
        # my cursor", which is a single range query on this key.
        "gsi1pk": {"S": "EXTRACTION"},
        "gsi1sk": {"S": f"{created}#{extraction_id}"},
        "source": {"S": (source or origin or "unknown")[:80]},
        "fileName": {"S": (file_name or "resume")[:255]},
        "fileType": {"S": (file_type or "")[:20]},
        "result": {"B": blob},
        "schema": {"N": "1"},
    }
    if subject:
        item["subject"] = {"S": subject[:128]}
    if origin:
        item["origin"] = {"S": origin[:200]}
    if client_id:
        item["clientId"] = {"S": client_id[:128]}
    if project_id:
        item["projectId"] = {"S": project_id[:128]}

    days = os.getenv("EXTRACTIONS_RETENTION_DAYS", "").strip()
    if days.isdigit() and int(days) > 0:
        item["ttl"] = {"N": str(int(time.time()) + int(days) * 86400)}
    return item


def save_extraction(result: dict, **fields) -> str | None:
    """Write one extraction. Returns its id, or None if it was not stored."""
    if not enabled():
        return None
    try:
        item = build_item(result, **fields)
        if item is None:
            return None
        _dynamo().put_item(TableName=_table_name(), Item=item)
        return item["id"]["S"]
    except Exception:
        logger.exception("[store] could not save extraction to %s", _table_name())
        return None
