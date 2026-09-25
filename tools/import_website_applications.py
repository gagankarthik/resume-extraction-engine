"""
One-off import: website applications' resume analyses into resume-extractions.

The company website ran every applicant's resume through this engine and kept
the result on the application (oceanblue-applications.resumeAnalysis) before
the engine saved extractions itself. This copies those results into the one
extractions table, in exactly the shape the engine now writes, so everything
that reads extractions sees them too.

Safe to re-run: each application maps to a fixed id (website-<application id>).

    python tools/import_website_applications.py           # dry run: counts only
    python tools/import_website_applications.py --write
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import boto3
from boto3.dynamodb.types import TypeDeserializer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import store

SOURCE_TABLE = "oceanblue-applications"
TARGET_TABLE = "resume-extractions"
REGION = "us-east-2"


def plain(value):
    """DynamoDB numbers arrive as Decimal; JSON wants int or float."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def when(raw: str | None) -> datetime:
    try:
        return datetime.fromisoformat((raw or "").replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def main(write: bool) -> None:
    client = boto3.client("dynamodb", region_name=REGION)
    des = TypeDeserializer()
    pages = client.get_paginator("scan").paginate(
        TableName=SOURCE_TABLE, FilterExpression="attribute_exists(resumeAnalysis)"
    )

    seen = stored = skipped = 0
    for page in pages:
        for raw in page.get("Items", []):
            seen += 1
            app = plain({k: des.deserialize(v) for k, v in raw.items()})
            result = app.get("resumeAnalysis")
            if not isinstance(result, dict) or not isinstance(result.get("work_experience"), list):
                skipped += 1
                continue

            # The applicant's contact details live on the application, not in
            # the analysis; put them where an extraction keeps them.
            person = dict(result.get("personal_information") or {})
            person.setdefault("full_name", app.get("name") or " ".join(filter(None, [app.get("firstName"), app.get("lastName")])))
            if app.get("email"):
                person["email"] = [app["email"]]
            if app.get("phone"):
                person["phone"] = [app["phone"]]
            result = {**result, "personal_information": person}

            key = app.get("applicationId") or app.get("id")
            item = store.build_item(
                result,
                source="website",
                subject=app.get("userId"),
                origin=None,
                file_name=app.get("resumeFileName") or "resume",
                file_type=(result.get("_metadata") or {}).get("file_type", ""),
                client_id=None,
                project_id=app.get("jobId"),
                now=when(app.get("resumeAnalyzedAt") or app.get("appliedAt") or app.get("createdAt")),
                extraction_id=f"website-{key}",
            )
            if item is None:
                skipped += 1
                continue
            if write:
                client.put_item(TableName=TARGET_TABLE, Item=item)
            stored += 1

    verb = "stored" if write else "would store"
    print(f"read {seen} applications; {verb} {stored}; skipped {skipped}")


if __name__ == "__main__":
    main(write="--write" in sys.argv)
