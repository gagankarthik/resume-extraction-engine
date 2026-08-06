"""
What the pipeline does with the clock.

Two behaviours are worth pinning down, because both reached production and
neither is visible in a unit test of any single agent:

  * the stages that say they run in parallel actually do. With a concurrency cap
    of 2 they did not, and a resume with a dozen jobs took a quarter of an hour
    while every log line claimed six agents were running at once.
  * running out of time returns a resume. It used to raise, discarding every
    stage that had already succeeded and billing for all of them.

Both are timing properties, so both are tested against a fake provider with a
known per-call latency rather than against a real model. The suite has no async
plugin, so each test drives its own event loop.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from agents import report
from agents.deadline import Deadline, DeadlineExceeded
from agents.llm.client import LLMClient
from agents.llm.providers import Completion
from config import LLMSettings

# One call's simulated latency. Large enough that serialised calls are
# unmistakable in the total, small enough to keep the suite quick.
CALL_SECONDS = 0.20

JOB_COUNT = 12


def _resume_text() -> str:
    """A long-career resume: the shape that was taking 840s."""
    parts = ["Jane Doe", "jane@example.com", "+1 555 0100", "", "PROFESSIONAL EXPERIENCE", ""]
    for i in range(JOB_COUNT):
        parts += [
            f"Company {i}",
            f"Senior Engineer | Jan 20{10 + i // 2} - Dec 20{11 + i // 2}",
            "• Built and maintained the ingestion service",
            "• Led the migration off the legacy scheduler",
            "• Owned the on-call rotation for the platform team",
            "",
        ]
    parts += ["EDUCATION", "B.S. Computer Science, State University, 2008"]
    return "\n".join(parts)


class FakeProvider:
    """Answers every call after a fixed delay, and records overlap.

    `peak_concurrency` is the whole point: it is what distinguishes a pipeline
    that fans out from one that queues.
    """

    name = "fake"

    def __init__(self, delay: float = CALL_SECONDS):
        self.delay = delay
        self.calls = 0
        self.in_flight = 0
        self.peak_concurrency = 0

    async def complete(self, system, user, *, max_tokens, temperature, json_mode, timeout=None):
        self.calls += 1
        self.in_flight += 1
        self.peak_concurrency = max(self.peak_concurrency, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        return Completion(
            text=json.dumps(self._answer(system)),
            input_tokens=100,
            output_tokens=100,
            truncated=False,
            model="fake",
            provider=self.name,
        )

    @staticmethod
    def _answer(system: str) -> dict:
        """Shaped like the agent's expected response."""
        if "structure analyzer" in system:
            return {
                "jobs": [
                    {
                        "company": f"Company {i}",
                        "title": "Senior Engineer",
                        "start_date": f"Jan 20{10 + i // 2}",
                        "end_date": f"Dec 20{11 + i // 2}",
                        "location": None,
                        "anchor_line": f"Company {i}",
                        "has_sub_projects": False,
                    }
                    for i in range(JOB_COUNT)
                ]
            }
        if "work experience extraction specialist" in system:
            return {
                "company_name": "Company 0",
                "job_title": "Senior Engineer",
                "start_date": "Jan 2010",
                "end_date": "Dec 2011",
                "is_current": False,
                "responsibilities": [
                    "Built and maintained the ingestion service",
                    "Led the migration off the legacy scheduler",
                    "Owned the on-call rotation for the platform team",
                ],
                "achievements": [],
                "technologies_used": [],
                "projects": [],
            }
        return {}


def _settings(**overrides) -> LLMSettings:
    base = {
        "provider": "openai",
        "openai_model": "fake",
        "anthropic_model": "fake",
        "max_concurrent": 12,
        "max_output_tokens": 32000,
        "call_timeout_seconds": 90,
        "transport_retries": 0,
        "truncation_escalations": 1,
    }
    return LLMSettings(**{**base, **overrides})


@pytest.fixture
def fake_llm(monkeypatch):
    """Point every agent at a fake provider and hand the test its handle."""

    def install(delay: float = CALL_SECONDS, **setting_overrides) -> FakeProvider:
        provider = FakeProvider(delay)
        client = LLMClient(settings=_settings(**setting_overrides), provider=provider)
        monkeypatch.setattr("agents.llm.client.get_client", lambda: client)
        monkeypatch.setattr("agents.base.get_client", lambda: client)
        return provider

    return install


def _extract(text: str, seconds: float) -> dict:
    """Run one extraction on its own event loop, as the service would."""
    from orchestrator import ResumeOrchestrator

    async def main() -> dict:
        report.reset_report()
        return await ResumeOrchestrator().run(text, Deadline.in_seconds(seconds))

    return asyncio.run(main())


def test_stages_run_concurrently(fake_llm):
    """A dozen jobs must not cost a dozen round trips."""
    provider = fake_llm()

    started = time.monotonic()
    result = _extract(_resume_text(), 60)
    elapsed = time.monotonic() - started

    assert len(result["work_experience"]) == JOB_COUNT

    # Structure has to land before work extraction starts, and the audit's
    # recovery call cannot start before the merge exists, so a few sequential
    # waves are inherent. The old cap of 2 needed roughly ten for the same work.
    assert elapsed < CALL_SECONDS * 6, (
        f"{provider.calls} calls took {elapsed:.2f}s — the stages are queueing, not fanning out"
    )
    assert provider.peak_concurrency >= 6, (
        f"only {provider.peak_concurrency} call(s) were ever in flight at once"
    )


def test_section_agents_do_not_wait_for_structure(fake_llm):
    """The five agents that never read the structure map must not queue behind it."""
    provider = fake_llm()

    _extract(_resume_text(), 60)

    # Structure plus the five independent section agents: six calls that can all
    # be in flight during the very first wave.
    assert provider.peak_concurrency >= 6


def test_running_out_of_time_returns_a_resume(fake_llm):
    """A budget too small for the whole pipeline degrades — it does not raise."""
    fake_llm(delay=0.5)

    result = _extract(_resume_text(), 1.5)

    assert isinstance(result, dict)
    # The expensive part survives. This is precisely what the old timeout threw away.
    assert result.get("work_experience"), "work history was discarded instead of returned"
    assert "analytics" in result


def test_skipped_refinement_is_reported_not_hidden(fake_llm):
    """When a stage is dropped for time, the response says so."""
    fake_llm(delay=0.5)

    result = _extract(_resume_text(), 1.5)

    run_report = result.get("_extraction_report") or {}
    assert "Validation" in run_report.get("skipped_sections", []), (
        "validation was skipped silently — the reviewer has no way to know"
    )


def test_an_exhausted_budget_stops_spending(fake_llm):
    """Past the deadline, no further calls are issued."""
    provider = fake_llm(delay=0.5)

    from orchestrator import ResumeOrchestrator

    async def main() -> int:
        report.reset_report()
        await ResumeOrchestrator().run(_resume_text(), Deadline.in_seconds(1.2))
        spent = provider.calls
        # Nothing may still be running once run() has returned.
        await asyncio.sleep(0.8)
        return provider.calls - spent

    assert main and asyncio.run(main()) == 0, "calls were still being issued after the deadline"


def test_one_slow_agent_cannot_hold_the_whole_run():
    """The bug that turned a slow resume into a 500.

    gather waits for its slowest branch however long that takes, so a single
    section still generating at the deadline carried the run past it and out
    through the caller as an error — discarding five sections that had already
    succeeded. The stragglers are now cut loose and reported.
    """
    from orchestrator import _gather_within

    async def quick(value):
        return value

    async def never():
        await asyncio.sleep(30)
        return "too late"

    async def main():
        started = time.monotonic()
        results = await _gather_within(
            Deadline.in_seconds(0.5), (quick("a"), never(), quick("b"))
        )
        return results, time.monotonic() - started

    results, elapsed = asyncio.run(main())

    assert elapsed < 5, f"waited {elapsed:.1f}s for a branch the budget did not cover"
    assert results[0] == "a"
    assert results[2] == "b", "a finished section was lost with the slow one"
    assert isinstance(results[1], DeadlineExceeded)


def test_a_failing_agent_is_returned_not_raised():
    """One section's exception must not become the whole run's exception."""
    from orchestrator import _gather_within

    async def boom():
        raise ValueError("section failed")

    async def fine():
        return "kept"

    results = asyncio.run(_gather_within(Deadline.in_seconds(10), (boom(), fine())))

    assert isinstance(results[0], ValueError)
    assert results[1] == "kept"


def test_deterministic_guards_survive_a_tight_budget(fake_llm):
    """Groundedness is not a refinement — it runs at any speed.

    The audit's LLM recovery call yields to the clock; its anti-fabrication pass
    must not, because a fabricated figure on a submitted resume is not a cost
    worth trading for a faster response.
    """
    fake_llm(delay=0.5)

    result = _extract(_resume_text(), 1.5)

    assert "_audit" in result, "the groundedness pass was skipped under time pressure"
