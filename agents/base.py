"""
BaseAgent — shared LLM-calling logic, JSON parsing, and retry for all section agents.
"""
from __future__ import annotations

import json
import os
import re
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseAgent:
    """All section agents inherit from this class."""

    def __init__(self, name: str):
        self.name = name
        self.provider = os.getenv("MODEL_PROVIDER", "openai").lower()

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
        retries: int = 2,
    ) -> tuple[str, dict]:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                if self.provider == "anthropic":
                    return await self._call_anthropic(system, user, max_tokens=max_tokens)
                else:
                    return await self._call_openai(system, user, json_mode=json_mode, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                    logger.warning("[%s] LLM attempt %d failed: %s — retrying", self.name, attempt + 1, exc)
        raise RuntimeError(f"[{self.name}] All LLM attempts failed: {last_exc}") from last_exc

    async def _call_openai(self, system: str, user: str, *, json_mode: bool, max_tokens: int, temperature: float) -> tuple[str, dict]:
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        client = AsyncOpenAI(api_key=api_key)
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        usage = {
            "provider": "openai",
            "model": model,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
        return raw, usage

    async def _call_anthropic(self, system: str, user: str, *, max_tokens: int) -> tuple[str, dict]:
        import anthropic as _anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
        client = _anthropic.AsyncAnthropic(api_key=api_key)
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        raw = next((b.text for b in resp.content if b.type == "text"), "")
        usage = {
            "provider": "anthropic",
            "model": model,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return raw, usage

    # ------------------------------------------------------------------ #
    # JSON parsing
    # ------------------------------------------------------------------ #

    def _parse_json(self, text: str) -> Any:
        text = text.strip()
        # Strip accidental markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            end = len(lines) - 1 if lines and lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[1:end])
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting the first JSON object / array from the text
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"[{self.name}] Could not parse JSON from response:\n{text[:300]}")

    # ------------------------------------------------------------------ #
    # Abstract run method
    # ------------------------------------------------------------------ #

    async def run(self, **kwargs: Any) -> Any:
        raise NotImplementedError
