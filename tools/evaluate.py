"""
Run a real extraction and score it. Use this to compare models or prompts.

    python tools/evaluate.py tests/fixtures/long-career.txt --model gpt-4.1-mini
    python tools/evaluate.py tests/fixtures/long-career.txt --model gpt-5.4-mini
    python tools/evaluate.py "C:/path/Resume.docx"          # any resume file

Reports three things, which together are what "better" has to mean here:

    accuracy   every universal invariant this extraction breaks (tests/invariants.py)
    speed      wall clock, and where it went
    shape      roles, bullets, skills, coverage — so a model that scores well by
               extracting half the resume is visibly doing that

It makes real calls and costs real money. Nothing in the test suite runs it.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from invariants import summarise, universal_violations  # noqa: E402

from agents import report  # noqa: E402
from agents.deadline import Deadline  # noqa: E402
from agents.llm.client import LLMClient  # noqa: E402
from agents.llm.providers import build_provider  # noqa: E402
from config import get_settings  # noqa: E402
from extractor import extract_text  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Which agent a call belongs to, read off its system prompt. Only for the
# breakdown — nothing depends on getting it right.
_STAGES = (
    ("structure analyzer", "structure"),
    ("work experience extraction specialist", "work"),
    ("Re-extract ONE job entry", "validator"),
    ("Preserve the resume's OWN skills-section labels", "skills:inventory"),
    ("given the list of skills read off", "skills:taxonomy"),
    ("personal information", "personal"),
    ("education entries", "education"),
    ("certifications, licenses", "certifications"),
    ("supplemental sections", "supplemental"),
    ("extraction auditor", "audit:recovery"),
)


def stage_of(system: str) -> str:
    return next((name for needle, name in _STAGES if needle in system), "other")


class TimingProvider:
    """Wraps a provider and times every call by stage."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.spans: list[tuple[str, float, int, int]] = []

    async def complete(self, system, user, *, max_tokens, temperature, json_mode, timeout=None):
        stage = stage_of(system)
        started = time.monotonic()
        try:
            completion = await self.inner.complete(
                system, user, max_tokens=max_tokens, temperature=temperature,
                json_mode=json_mode, timeout=timeout,
            )
            self.spans.append((stage, time.monotonic() - started,
                               len(user), completion.output_tokens))
            return completion
        except Exception:
            self.spans.append((stage, time.monotonic() - started, len(user), -1))
            raise


def load_resume(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    text, _, _ = extract_text(path.read_bytes(), path.suffix.lstrip(".").lower())
    return text


async def evaluate(path: Path, model: str | None, seconds: float) -> int:
    resume = load_resume(path)
    # get_settings() is cached and reads the environment, so the override has to
    # be in place before the first call — which main() guarantees.
    settings = get_settings()

    provider = TimingProvider(build_provider(settings.llm))
    client = LLMClient(settings=settings.llm, provider=provider)

    import agents.base as base
    import agents.llm.client as llm_client

    base.get_client = lambda: client
    llm_client.get_client = lambda: client

    from orchestrator import ResumeOrchestrator

    report.reset_report()
    started = time.monotonic()
    result = await ResumeOrchestrator().run(resume, Deadline.in_seconds(seconds))
    wall = time.monotonic() - started

    violations = universal_violations(result, resume)
    jobs = result.get("work_experience") or []
    bullets = sum(
        len(j.get("responsibilities") or [])
        + sum(len(p.get("projectResponsibilities") or []) for p in (j.get("projects") or []))
        for j in jobs if isinstance(j, dict)
    )
    audit = result.get("_audit") or {}

    label = settings.llm.model
    print(f"\n{'=' * 72}\n  {path.name}   model={label}\n{'=' * 72}")
    print(f"\nSPEED   wall {wall:.1f}s over {len(provider.spans)} call(s)\n")
    by: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for stage, elapsed, _, out in provider.spans:
        row = by[stage]
        row[0] += 1
        row[1] += elapsed
        row[2] = max(row[2], elapsed)
        row[3] += max(0, out)
    print(f"  {'stage':20} {'n':>3} {'sum_s':>8} {'slowest':>8} {'out_tok':>8}")
    for stage, (n, total, slowest, out) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"  {stage:20} {int(n):3d} {total:8.1f} {slowest:8.1f} {int(out):8d}")

    print(f"\nSHAPE   roles {len(jobs)}   bullets {bullets}   "
          f"skills {len(result.get('skills', {}).get('all_skills_raw') or [])}   "
          f"education {len(result.get('education') or [])}   "
          f"coverage {audit.get('coverage_percent', '?')}%")

    print(f"\nACCURACY   {len(violations)} violation(s)")
    if violations:
        for rule, count in summarise(violations).items():
            print(f"  {count:4d}  {rule}")
        print("\n  first few:")
        for v in violations[:8]:
            print(f"    - {v}")
    else:
        print("  none — every universal invariant holds")
    print()
    return 1 if violations else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resume", type=Path)
    parser.add_argument("--model", help="override OPENAI_MODEL for this run")
    parser.add_argument("--seconds", type=float, default=300.0)
    args = parser.parse_args()

    if not args.resume.is_file():
        parser.error(f"no such file: {args.resume}")
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model
        get_settings.cache_clear()
    sys.exit(asyncio.run(evaluate(args.resume, args.model, args.seconds)))


if __name__ == "__main__":
    main()
