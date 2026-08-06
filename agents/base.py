"""
BaseAgent — the shared surface every section agent builds on.

Transport, retries, concurrency, truncation handling, and token accounting live
in `agents.llm`. What remains here is the thin adapter the agents use, so a
section agent is only ever a prompt plus the shaping of its own result.
"""
from __future__ import annotations

import logging
from typing import Any

from agents import report
from agents.json_salvage import UnsalvageableJSON, count_items, salvage
from agents.llm import get_client, get_token_usage, reset_token_usage

logger = logging.getLogger(__name__)

# Re-exported so existing callers (processor.py) keep working unchanged.
__all__ = ["BaseAgent", "get_token_usage", "output_budget", "reset_token_usage"]


def output_budget(source: str, *, floor: int = 4096, ceiling: int = 16384) -> int:
    """Room to copy `source` back out as JSON, with scaffolding to spare.

    Extraction is near-verbatim, so a section's answer is about as long as the
    text it reads. Guessing low is expensive in a way guessing high is not: a
    response that runs out of room is re-asked from scratch with double the
    budget, so one bad guess costs an entire extra generation — and on a long
    resume that was happening on the longest jobs, the ones already slowest.
    Providers bill for tokens generated, not for the ceiling requested, so the
    only cost of a generous budget is that it must stay under the model's own.

    Roughly four characters per token, times 1.6 for JSON keys, quoting and
    escaping, plus a fixed allowance for the object's fields.
    """
    return max(floor, min(ceiling, int(len(source) / 4 * 1.6) + 768))


class BaseAgent:
    """All section agents inherit from this class."""

    def __init__(self, name: str):
        self.name = name

    @property
    def client(self):
        # Resolved per call rather than stored, so settings changes and test
        # doubles take effect without rebuilding every agent.
        return get_client()

    # ------------------------------------------------------------------ #
    # LLM calling
    # ------------------------------------------------------------------ #

    async def _call_llm(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
        max_tokens: int = 8192,
        temperature: float = 0,
        **_ignored: Any,
    ) -> tuple[str, dict]:
        """
        Raw text from the model.

        Prefer `_call_llm_json` — it handles a response that ran out of room,
        which this cannot do because it has already thrown away the structure.
        """
        completion = await self.client.complete(
            system,
            user,
            label=self.name,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        return completion.text, completion.usage()

    async def _call_llm_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 0,
        section: str | None = None,
    ) -> Any:
        """
        Parsed JSON, with the output budget grown automatically when a response
        is cut short and complete records kept when it still is.

        `section` names the row this call reports under, so the editor can tell
        the user which parts of their resume came back whole.
        """
        return await self.client.complete_json(
            system,
            user,
            label=section or self.name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ------------------------------------------------------------------ #
    # JSON parsing
    # ------------------------------------------------------------------ #

    def _parse_json(self, text: str) -> Any:
        """
        Parse a model response, recovering the complete records from one that
        was cut short rather than discarding the section wholesale.
        """
        try:
            result = salvage(text)
        except UnsalvageableJSON as exc:
            raise ValueError(f"[{self.name}] {exc}") from exc

        if result.repaired:
            logger.warning(
                "[%s] Response was incomplete — kept the records that parsed.", self.name
            )
            report.record(
                self.name,
                report.Status.PARTIAL,
                items=count_items(result.data),
                detail=(
                    "The model's response was cut off. The records that came "
                    "through were kept — check this section against the original."
                ),
            )
        return result.data

    # ------------------------------------------------------------------ #
    # Abstract run method
    # ------------------------------------------------------------------ #

    async def run(self, **kwargs: Any) -> Any:
        raise NotImplementedError
