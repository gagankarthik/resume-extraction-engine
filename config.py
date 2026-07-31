"""
One place that reads the environment.

Before this, `os.getenv` calls were scattered across the agents, the processor,
and the orchestrator — so the same setting could be spelled two ways, a typo
silently fell back to a default, and there was nowhere to look up what the
service actually accepts. Settings are read once, validated once, and frozen.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

Provider = Literal["openai", "anthropic"]


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True, slots=True)
class LLMSettings:
    provider: Provider
    openai_model: str
    anthropic_model: str

    max_concurrent: int
    """In-flight LLM calls across the whole pipeline. Keeps a long resume with
    many jobs from firing enough parallel calls to trip a provider rate limit."""

    max_output_tokens: int
    """Ceiling for a single completion. Truncation escalation stops here."""

    transport_retries: int
    """Retries for network errors and rate limits, per call."""

    truncation_escalations: int
    """How many times to re-ask with a doubled budget when a response is cut off."""

    @property
    def model(self) -> str:
        return self.anthropic_model if self.provider == "anthropic" else self.openai_model


@dataclass(frozen=True, slots=True)
class Settings:
    llm: LLMSettings
    use_orchestrator: bool
    extraction_timeout_seconds: int
    max_file_mb: int
    log_level: str
    cors_origins: tuple[str, ...] = field(default=())

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Build the settings once per process.

    Cached rather than module-level so tests can clear it, and so importing this
    module never fails at import time on a bad value — the error surfaces on
    first use, where it can be reported properly.
    """
    provider_raw = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider_raw not in ("openai", "anthropic"):
        raise ValueError(
            f"MODEL_PROVIDER must be 'openai' or 'anthropic', got {provider_raw!r}"
        )

    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return Settings(
        llm=LLMSettings(
            provider=provider_raw,  # type: ignore[arg-type]
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
            max_concurrent=_int("LLM_MAX_CONCURRENT", 2),
            max_output_tokens=_int("LLM_MAX_OUTPUT_TOKENS", 32000),
            transport_retries=_int("LLM_TRANSPORT_RETRIES", 5),
            truncation_escalations=_int("LLM_TRUNCATION_ESCALATIONS", 2),
        ),
        use_orchestrator=_bool("USE_ORCHESTRATOR", True),
        extraction_timeout_seconds=_int("EXTRACTION_TIMEOUT_SECONDS", 360),
        max_file_mb=_int("MAX_FILE_SIZE_MB", 20),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
    )
