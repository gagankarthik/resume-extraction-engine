"""
The output-ceiling parameter OpenAI wants depends on the model.

gpt-5 and the o-series take `max_completion_tokens`; everything before them
takes `max_tokens`; sending the wrong one is a 400, not a warning. Model names
are not a stable API, so the provider guesses from the name and then corrects
itself from what the API says — these tests pin both halves, because a wrong
guess with no correction means every call fails.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agents.llm import providers
from agents.llm.providers import OpenAIProvider, _token_param
from config import LLMSettings


@pytest.fixture(autouse=True)
def _clear_memo():
    providers._TOKEN_PARAM.clear()
    yield
    providers._TOKEN_PARAM.clear()


def test_the_name_decides_the_first_guess():
    assert _token_param("gpt-4.1-mini") == "max_tokens"
    assert _token_param("gpt-4o") == "max_tokens"
    assert _token_param("gpt-5.4-mini") == "max_completion_tokens"
    assert _token_param("gpt-5") == "max_completion_tokens"
    assert _token_param("o3-mini") == "max_completion_tokens"
    # A family this pattern has never heard of still gets a guess, and the
    # correction below is what saves it.
    assert _token_param("gpt-12-turbo") == "max_completion_tokens"


class FakeCompletions:
    """Accepts exactly one of the two parameter names, like the real API."""

    def __init__(self, accepts: str):
        self.accepts = accepts
        self.attempts: list[str] = []

    async def create(self, **kwargs):
        sent = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
        self.attempts.append(sent)
        if sent != self.accepts:
            raise RuntimeError(
                f"Error code: 400 - Unsupported parameter: '{sent}' is not supported "
                f"with this model. Use '{self.accepts}' instead."
            )

        message = SimpleNamespace(content="{}")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class FakeClient:
    def __init__(self, accepts: str):
        self.chat = type("Chat", (), {"completions": FakeCompletions(accepts)})()


def _provider(model: str, accepts: str) -> tuple[OpenAIProvider, FakeClient]:
    settings = LLMSettings(
        model=model,
        max_concurrent=1, max_output_tokens=32000, call_timeout_seconds=10,
        transport_retries=0, truncation_escalations=1,
    )
    provider = OpenAIProvider(settings)
    client = FakeClient(accepts)
    provider._client = client
    return provider, client


def _call(provider: OpenAIProvider):
    return asyncio.run(provider.complete(
        "sys", "user", max_tokens=100, temperature=0, json_mode=True,
    ))


def test_a_correct_guess_costs_one_request():
    provider, client = _provider("gpt-4.1-mini", accepts="max_tokens")
    _call(provider)
    assert client.chat.completions.attempts == ["max_tokens"]


def test_a_wrong_guess_is_corrected_from_the_error():
    # The name says old-style, the API wants new-style.
    provider, client = _provider("mystery-model-1", accepts="max_completion_tokens")
    _call(provider)
    assert client.chat.completions.attempts == ["max_tokens", "max_completion_tokens"]


def test_the_correction_is_remembered_for_later_calls():
    provider, client = _provider("mystery-model-1", accepts="max_completion_tokens")
    _call(provider)
    _call(provider)
    _call(provider)
    # One wrong attempt in total, not one per call.
    assert client.chat.completions.attempts == [
        "max_tokens", "max_completion_tokens", "max_completion_tokens", "max_completion_tokens",
    ]


def test_an_unrelated_error_is_not_swallowed():
    provider, _ = _provider("gpt-4.1-mini", accepts="max_tokens")

    async def boom(**kwargs):
        raise RuntimeError("Error code: 429 - rate limit exceeded")

    provider._client.chat.completions.create = boom
    with pytest.raises(RuntimeError, match="429"):
        _call(provider)
