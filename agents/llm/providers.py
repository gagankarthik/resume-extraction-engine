"""
Provider adapters.

Each adapter turns one vendor's SDK response into the same `Completion`, so the
rest of the pipeline never branches on which model is behind it. Adding a
provider means adding a class here and one line in `build_provider` — no agent
changes.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from config import LLMSettings


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response, normalised across providers."""

    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool
    """True when generation stopped on the token ceiling rather than finishing.
    The text is mid-structure and must not be treated as a complete answer."""

    model: str
    provider: str
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def usage(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "truncated": self.truncated,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
        }


class LLMProvider(Protocol):
    """What the client needs from any model vendor."""

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


# The resume block is identical across the parallel section agents, so marking
# it cacheable turns six full-price reads into one write and five cache hits.
_CACHE_BLOCK_RE = re.compile(
    r"(===\s*(?:RESUME(?:\s+TEXT)?|TEXT\s+SEGMENT)\s*===.*?===\s*END(?:\s+SEGMENT)?\s*===)",
    re.DOTALL,
)


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
            self._client = AsyncOpenAI(api_key=api_key)
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
        model = self.settings.openai_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # The SDK's own default is ten minutes, long enough for one stalled
        # request to outlast the entire run.
        if timeout is not None and timeout != float("inf"):
            kwargs["timeout"] = timeout

        resp = await self._get_client().chat.completions.create(**kwargs)
        choice = resp.choices[0]

        return Completion(
            text=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            truncated=choice.finish_reason == "length",
            model=model,
            provider=self.name,
        )


@dataclass
class AnthropicProvider:
    settings: LLMSettings
    name: str = "anthropic"
    _client: Any = field(default=None, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set.")
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    @staticmethod
    def _with_cache_breakpoint(user: str) -> Any:
        """Mark the resume block cacheable, leaving the per-agent instructions hot."""
        match = _CACHE_BLOCK_RE.search(user)
        if not match:
            return user

        parts: list[dict[str, Any]] = []
        head, block, tail = user[: match.start()], match.group(0), user[match.end() :]
        if head.strip():
            parts.append({"type": "text", "text": head})
        parts.append({"type": "text", "text": block, "cache_control": {"type": "ephemeral"}})
        if tail.strip():
            parts.append({"type": "text", "text": tail})
        return parts

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
        model = self.settings.anthropic_model
        extra: dict[str, Any] = {}
        if timeout is not None and timeout != float("inf"):
            extra["timeout"] = timeout

        resp = await self._get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": self._with_cache_breakpoint(user)}],
            **extra,
        )

        return Completion(
            text=next((b.text for b in resp.content if b.type == "text"), ""),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            truncated=resp.stop_reason == "max_tokens",
            model=model,
            provider=self.name,
            cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        )


def build_provider(settings: LLMSettings) -> LLMProvider:
    if settings.provider == "anthropic":
        return AnthropicProvider(settings)
    return OpenAIProvider(settings)
