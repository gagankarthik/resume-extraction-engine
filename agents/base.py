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

# ── Module-level singleton clients — created once, reused for every API call ─
# Avoids creating a new HTTP connection pool per agent call (critical for
# 15+ parallel job extractions on a 25-year resume).
_anthropic_client: Any = None
_openai_client: Any = None

# Global concurrency cap for in-flight LLM calls. Stage 2 fires ~6 top-level
# agents in parallel and WorkExperienceAgent fires N per-job calls inside that —
# without this, an 8-job 25-year resume hits ~14 concurrent calls and trips
# the OpenAI 30K TPM ceiling on gpt-4o instantly. Default lowered to 2 so a
# single Stage-2 burst stays under ~14K active tokens, leaving headroom for
# WorkExperienceAgent's per-job retries. Configurable via env var.
_LLM_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENT", "2"))
_llm_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)
    return _llm_semaphore


# Parse the "Please try again in 704ms" hint from OpenAI 429 responses so we
# wait exactly the prescribed duration instead of a fixed exponential backoff.
_RETRY_HINT_RE = re.compile(r"try again in (\d+(?:\.\d+)?)\s*(ms|s|second|seconds)", re.IGNORECASE)


def _parse_retry_after(exc: Exception) -> float | None:
    msg = str(exc)
    m = _RETRY_HINT_RE.search(msg)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value / 1000.0 if unit == "ms" else value


def _get_anthropic_client() -> Any:
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic as _anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = _anthropic.AsyncAnthropic(api_key=api_key)
    return _anthropic_client


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


# Regex to locate the cacheable resume/segment block inside a user message.
# Matches any of the marker pairs the agents use.
_CACHE_BLOCK_RE = re.compile(
    r"(===\s*(?:RESUME(?:\s+TEXT)?|TEXT\s+SEGMENT)\s*===.*?===\s*END(?:\s+SEGMENT)?\s*===)",
    re.DOTALL,
)


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
        retries: int = 5,
    ) -> tuple[str, dict]:
        last_exc: Exception | None = None
        sem = _get_semaphore()
        for attempt in range(retries + 1):
            try:
                # Acquire the global semaphore so the whole pipeline can't exceed
                # LLM_MAX_CONCURRENT concurrent in-flight calls.
                async with sem:
                    if self.provider == "anthropic":
                        return await self._call_anthropic(system, user, max_tokens=max_tokens)
                    else:
                        return await self._call_openai(system, user, json_mode=json_mode, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    # Respect the server's "try again in Xms/Xs" hint if present.
                    # Fall back to exponential backoff capped at 30s otherwise.
                    hinted = _parse_retry_after(exc)
                    if hinted is not None:
                        # Add a small jitter so retries from multiple in-flight
                        # callers don't all wake simultaneously.
                        wait = hinted + 0.1 * (attempt + 1)
                    else:
                        wait = min(30.0, 2 ** attempt)
                    logger.warning(
                        "[%s] LLM attempt %d failed: %s — retrying in %.2fs",
                        self.name, attempt + 1, exc, wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(f"[{self.name}] All LLM attempts failed: {last_exc}") from last_exc

    async def _call_openai(self, system: str, user: str, *, json_mode: bool, max_tokens: int, temperature: float) -> tuple[str, dict]:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        client = _get_openai_client()
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
        model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
        client = _get_anthropic_client()

        # Auto-cache the resume/segment block so repeated calls with the same
        # resume text (e.g. Stage 2 parallel agents) benefit from KV cache hits.
        m = _CACHE_BLOCK_RE.search(user)
        if m:
            pre = user[:m.start()]
            block = m.group(0)
            post = user[m.end():]
            user_content: Any = []
            if pre.strip():
                user_content.append({"type": "text", "text": pre})
            user_content.append({"type": "text", "text": block, "cache_control": {"type": "ephemeral"}})
            if post.strip():
                user_content.append({"type": "text", "text": post})
        else:
            user_content = user

        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_content}],
        )
        raw = next((b.text for b in resp.content if b.type == "text"), "")
        usage = {
            "provider": "anthropic",
            "model": model,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_creation_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0),
            "cache_read_tokens":     getattr(resp.usage, "cache_read_input_tokens", 0),
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
        # Extract the first JSON object / array from surrounding text
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
