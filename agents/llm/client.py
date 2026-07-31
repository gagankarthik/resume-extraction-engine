"""
The one path every model call goes through.

Responsibilities, each of which used to be duplicated or missing in the agents:

  * cap concurrency across the whole pipeline, so a resume with fifteen jobs
    cannot fire fifteen simultaneous calls and trip a rate limit;
  * retry transport failures with the server's own backoff hint when it gives
    one, exponential backoff when it does not;
  * treat a response that ran out of tokens as a distinct failure, re-ask with
    a larger budget, and salvage the complete records if the ceiling is reached;
  * account for tokens per request.

Agents call `complete_json` and get parsed data. Nothing above this layer knows
which vendor answered.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from config import LLMSettings, get_settings

from agents import report
from agents.json_salvage import UnsalvageableJSON, count_items, salvage
from agents.llm.providers import Completion, LLMProvider, build_provider

logger = logging.getLogger(__name__)


# ── Token accounting ──────────────────────────────────────────────────────
# A mutable dict in a ContextVar so every task spawned by asyncio.gather shares
# the caller's accumulator and usage rolls up to the originating request.
_token_usage: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "token_usage", default=None
)


def reset_token_usage() -> dict:
    acc = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    _token_usage.set(acc)
    return acc


def get_token_usage() -> dict | None:
    return _token_usage.get()


def _record_usage(completion: Completion) -> None:
    acc = _token_usage.get()
    if acc is None:
        return
    acc["input_tokens"] += completion.input_tokens
    acc["output_tokens"] += completion.output_tokens
    acc["calls"] += 1


# ── Errors ────────────────────────────────────────────────────────────────


class TruncatedCompletion(RuntimeError):
    """Generation stopped on the token ceiling; `text` is mid-structure."""

    def __init__(self, label: str, text: str, limit: int):
        super().__init__(f"[{label}] Response hit the {limit}-token ceiling and was cut off.")
        self.text = text
        self.limit = limit


class ExtractionFailed(RuntimeError):
    """Nothing usable came back for a section."""


# Providers state their own backoff in the error text; honouring it beats
# guessing, and avoids a thundering herd of retries waking together.
_RETRY_HINT_RE = re.compile(
    r"try again in (\d+(?:\.\d+)?)\s*(ms|s|second|seconds)", re.IGNORECASE
)


def _retry_after(exc: Exception) -> float | None:
    match = _RETRY_HINT_RE.search(str(exc))
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2).lower() == "ms" else value


@dataclass
class LLMClient:
    settings: LLMSettings
    provider: LLMProvider
    _semaphore: asyncio.Semaphore | None = field(default=None, repr=False)

    def _sem(self) -> asyncio.Semaphore:
        # Created on first use so it binds to the running event loop.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.max_concurrent)
        return self._semaphore

    # ── One call, with transport retries ──────────────────────────────────

    async def complete(
        self,
        system: str,
        user: str,
        *,
        label: str,
        max_tokens: int,
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> Completion:
        last_exc: Exception | None = None

        for attempt in range(self.settings.transport_retries + 1):
            try:
                async with self._sem():
                    completion = await self.provider.complete(
                        system,
                        user,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        json_mode=json_mode,
                    )
                _record_usage(completion)

                if completion.truncated:
                    raise TruncatedCompletion(label, completion.text, max_tokens)

                return completion

            except TruncatedCompletion:
                raise  # not a transport problem; the caller decides what to do

            except Exception as exc:
                last_exc = exc
                if attempt >= self.settings.transport_retries:
                    break
                wait = _retry_after(exc)
                wait = wait + 0.1 * (attempt + 1) if wait is not None else min(30.0, 2**attempt)
                logger.warning(
                    "[%s] attempt %d/%d failed (%s) — retrying in %.2fs",
                    label, attempt + 1, self.settings.transport_retries + 1, exc, wait,
                )
                await asyncio.sleep(wait)

        raise ExtractionFailed(f"[{label}] every attempt failed: {last_exc}") from last_exc

    # ── One call, parsed, with truncation handling ────────────────────────

    async def complete_json(
        self,
        system: str,
        user: str,
        *,
        label: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Any:
        """
        Return parsed JSON, growing the output budget when the model runs out
        of room and keeping the complete records when it still does.

        The outcome is recorded against `label` so the person reviewing the
        result can see which sections came back whole.
        """
        budget = max_tokens
        attempts = 0
        truncation: TruncatedCompletion | None = None

        for _ in range(self.settings.truncation_escalations + 1):
            attempts += 1
            try:
                completion = await self.complete(
                    system, user, label=label, max_tokens=budget, temperature=temperature
                )
            except TruncatedCompletion as exc:
                truncation = exc
                if budget >= self.settings.max_output_tokens:
                    break
                budget = min(budget * 2, self.settings.max_output_tokens)
                logger.warning("[%s] cut off — re-asking with %d tokens", label, budget)
                continue

            parsed = salvage(completion.text)
            report.record(
                label,
                report.Status.RECOVERED if attempts > 1 else report.Status.OK,
                items=count_items(parsed.data),
                attempts=attempts,
            )
            return parsed.data

        return self._salvage_or_fail(truncation, label, attempts)

    @staticmethod
    def _salvage_or_fail(
        truncation: TruncatedCompletion | None, label: str, attempts: int
    ) -> Any:
        """Keep the complete records from a response that never fit, or give up loudly."""
        if truncation is None:
            raise ExtractionFailed(f"[{label}] produced no response.")

        try:
            parsed = salvage(truncation.text)
        except UnsalvageableJSON as exc:
            report.record(
                label,
                report.Status.FAILED,
                attempts=attempts,
                detail="The response was cut off before a single complete record.",
            )
            raise ExtractionFailed(f"[{label}] {exc}") from exc

        kept = count_items(parsed.data)
        report.record(
            label,
            report.Status.PARTIAL,
            items=kept,
            attempts=attempts,
            detail=(
                f"Too long to finish. {kept} complete record(s) were kept; anything "
                "after that was not returned. Check this section against the original."
            ),
        )
        logger.warning("[%s] kept %d record(s) from a truncated response", label, kept)
        return parsed.data


_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Process-wide client, so connection pools and the semaphore are shared."""
    global _client
    if _client is None:
        llm_settings = get_settings().llm
        _client = LLMClient(settings=llm_settings, provider=build_provider(llm_settings))
    return _client
