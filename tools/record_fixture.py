"""
Record a real extraction so the golden test can replay it.

    python tools/record_fixture.py "C:/path/Resume.docx" --name long-career

Writes two files into tests/fixtures/:

    <name>.txt            the normalised resume text the pipeline reads
    <name>.cassette.json  every model answer, keyed by the call that asked

Run this when you add a resume to the suite, and again when a change alters
what the pipeline ASKS the model — a new segmentation, a different slice. It
costs one real extraction. Rewording a system prompt does not need a re-record;
the cassette is keyed on the user message.

Nothing in the normal test run touches this script, and nothing in the service
imports it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from replay import FIXTURES, Cassette, RecordingProvider  # noqa: E402

from agents import report  # noqa: E402
from agents.deadline import Deadline  # noqa: E402
from agents.llm.client import LLMClient  # noqa: E402
from agents.llm.providers import build_provider  # noqa: E402
from config import get_settings  # noqa: E402
from extractor import extract_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("record_fixture")


async def record(resume: Path, name: str, seconds: float) -> None:
    text, pages, info = extract_text(resume.read_bytes(), resume.suffix.lstrip(".").lower())
    logger.info("Extracted %d chars via %s", len(text), info.get("method"))

    settings = get_settings()
    cassette = Cassette(meta={
        "resume_file": resume.name,
        "model": settings.llm.model,
        "extraction_method": info.get("method"),
        "pages": pages,
    })
    client = LLMClient(
        settings=settings.llm,
        provider=RecordingProvider(build_provider(settings.llm), cassette),
    )

    import agents.base as base
    import agents.llm.client as llm_client

    base.get_client = lambda: client
    llm_client.get_client = lambda: client

    from orchestrator import ResumeOrchestrator

    report.reset_report()
    result = await ResumeOrchestrator().run(text, Deadline.in_seconds(seconds))

    (FIXTURES / f"{name}.txt").parent.mkdir(parents=True, exist_ok=True)
    (FIXTURES / f"{name}.txt").write_text(text, encoding="utf-8")
    cassette.save(FIXTURES / f"{name}.cassette.json")
    (FIXTURES / f"{name}.recorded-output.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )

    logger.info(
        "Recorded %d call(s) over %d job(s) → tests/fixtures/%s.cassette.json",
        sum(len(v) for v in cassette.calls.values()),
        len(result.get("work_experience") or []),
        name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resume", type=Path, help="the resume file to record")
    parser.add_argument("--name", required=True, help="fixture name, e.g. long-career")
    parser.add_argument(
        "--seconds", type=float, default=600.0,
        help="budget for the recording run; generous, since it is not the product path",
    )
    args = parser.parse_args()

    if not args.resume.is_file():
        parser.error(f"no such file: {args.resume}")
    asyncio.run(record(args.resume, args.name, args.seconds))


if __name__ == "__main__":
    main()
