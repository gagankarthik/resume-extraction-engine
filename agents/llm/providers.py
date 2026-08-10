"""
The OpenAI adapter, and the shape the rest of the pipeline sees.

Everything above this file works in `Completion`s and knows nothing about the
SDK: what a response costs, whether it was cut off, and what it said. That
boundary is worth keeping even with one vendor behind it, because it is what
lets the test suite replay a recorded run in place of a real one.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from config import LLMSettings


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response, in the terms the pipeline cares about."""

    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool
    """True when generation stopped on the token ceiling rather than finishing.
    The text is mid-structure and must not be treated as a complete answer."""

    model: str
    provider: str

    def usage(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "truncated": self.truncated,
        }


class LLMProvider(Protocol):
    """What the client needs from whatever is answering.

    The test suite supplies a recorded stand-in through this same interface, so
    it is not an abstraction over vendors so much as the seam that makes the
    pipeline testable without a network.
    """

    name: str

    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        timeout: float | None = None,
    ) -> Completion: ...


# OpenAI renamed the output ceiling when the reasoning models landed: gpt-5 and
# the o-series take `max_completion_tokens`, everything before them takes
# `max_tokens`, and sending the wrong one is a 400 rather than a warning.
#
# The name pattern below is the first guess, not the authority. Model names are
# not a stable API — a family this list has never heard of would be guessed
# wrong and fail every call — so the guess is corrected at runtime from the
# error the API itself returns, and remembered for the process. Getting it wrong
# costs one retry, once, ever.
_NEEDS_COMPLETION_TOKENS = re.compile(r"^(?:gpt-[5-9]|gpt-\d{2}|o[1-9])", re.I)

_UNSUPPORTED_MAX_TOKENS = re.compile(
    r"unsupported parameter:\s*'?(max_tokens|max_completion_tokens)'?", re.I
)

# model name -> the parameter that model actually accepts.
_TOKEN_PARAM: dict[str, str] = {}


def _token_param(model: str) -> str:
    remembered = _TOKEN_PARAM.get(model)
    if remembered is not None:
        return remembered
    return "max_completion_tokens" if _NEEDS_COMPLETION_TOKENS.match(model) else "max_tokens"


@dataclass
class OpenAIProvider:
    settings: LLMSettings
    name: str = "openai"
    _client: Any = field(default=None, repr=False)

    def _get_client(self) -> Any:
        # Built lazily and reused: one connection pool for the whole process
        # instead of one per agent call.
        if self._client is None:
            from openai import AsyncOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            # max_retries=0 because this layer already retries, and only this
            # layer knows the run's remaining budget. Left at the SDK default of
            # 2, a per-request timeout of 90s silently became a 270s call: the
            # deadline could not bound it, and the pipeline blew through its own
            # limit and returned a 500 instead of a partial result.
            self._client = AsyncOpenAI(api_key=api_key, max_retries=0)
        return self._client

    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        timeout: float | None = None,
    ) -> Completion:
        model = self.settings.model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # The SDK's own default is ten minutes, long enough for one stalled
        # request to outlast the entire run.
        if timeout is not None and timeout != float("inf"):
            kwargs["timeout"] = timeout

        resp = await self._create(kwargs, model, max_tokens)
        choice = resp.choices[0]

        return Completion(
            text=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            truncated=choice.finish_reason == "length",
            model=model,
            provider=self.name,
        )

    async def _create(self, kwargs: dict[str, Any], model: str, max_tokens: int) -> Any:
        """Send the request, correcting the output-ceiling parameter if it is wrong.

        The correction happens once per model per process: the API tells us which
        name it wanted, we remember it, and every later call gets it right first
        time.
        """
        param = _token_param(model)
        try:
            return await self._get_client().chat.completions.create(
                **kwargs, **{param: max_tokens}
            )
        except Exception as exc:
            other = "max_tokens" if param == "max_completion_tokens" else "max_completion_tokens"
            if not _UNSUPPORTED_MAX_TOKENS.search(str(exc)):
                raise
            _TOKEN_PARAM[model] = other
            return await self._get_client().chat.completions.create(
                **kwargs, **{other: max_tokens}
            )


def build_provider(settings: LLMSettings) -> LLMProvider:
    return OpenAIProvider(settings)
