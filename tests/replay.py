"""
Record what the model said once; replay it forever.

WHY THIS EXISTS

Every stage after the model is ordinary code — segmentation, merging, the
bullet guards, validation, the audit, the shape pass — and every bug found in
this engine so far has been in that code, not in the model. But none of it could
be tested end to end, because exercising it meant a live call whose answer is
different every time. So the only way to know a change had not broken a real
resume was to run one and read the JSON, and changes shipped that quietly fused
bullets, emptied the skills section, or split a skill name down the middle.

A cassette removes the model from the loop. One recording of a real extraction
turns the entire pipeline into a pure function of the resume text, so a golden
test can assert what came out and mean it.

WHAT THE KEY IS, AND WHY

Calls are keyed by the USER message alone — never the system prompt. The system
prompt is the instruction we edit constantly; the user message is the resume,
the job segment, the skills list. Keying this way means rewording a prompt still
replays, which is what makes the harness usable: the test tells you whether the
pipeline still assembles the answer correctly, and it keeps telling you that
while you tune the prompts.

A user message that genuinely changes — because the code now segments jobs
differently, or asks about a different slice — misses, and the miss says so
loudly. That is the correct outcome: the input to the model changed, and nobody
can know what the model would have said without asking it. Re-record.

    python tools/record_fixture.py path/to/resume.docx --name long-career
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.llm.providers import Completion, LLMProvider

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def call_key(user: str) -> str:
    """The identity of one call: its user message, and nothing else."""
    return hashlib.sha1(user.encode("utf-8")).hexdigest()[:16]


class CassetteMiss(RuntimeError):
    """A call the cassette has no answer for."""


@dataclass
class Cassette:
    """Recorded answers, in the order they were given.

    A key maps to a LIST because one call can legitimately be made twice with
    the same user message — a response that hit the token ceiling is re-asked
    with a bigger budget, and the second answer is the one that parses. Replaying
    them in order reproduces that; collapsing them to one would hide it.
    """

    calls: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    _cursor: dict[str, int] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path) -> Cassette:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(calls=raw.get("calls", {}), meta=raw.get("meta", {}))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"meta": self.meta, "calls": self.calls}, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )

    def add(self, user: str, completion: Completion) -> None:
        entry = {
            "hint": user.strip().splitlines()[0][:90] if user.strip() else "",
            "text": completion.text,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "truncated": completion.truncated,
        }
        self.calls.setdefault(call_key(user), []).append(entry)

    def take(self, user: str) -> dict[str, Any]:
        key = call_key(user)
        answers = self.calls.get(key)
        if not answers:
            raise CassetteMiss(
                "No recorded answer for this call. The user message changed, so "
                "the recording no longer covers it.\n"
                f"  first line: {user.strip().splitlines()[0][:120] if user.strip() else '(empty)'}\n"
                "  re-record:  python tools/record_fixture.py <resume> --name <fixture>"
            )
        # Past the end, the last answer stands: a retry of an identical call
        # would have been served the same way.
        index = min(self._cursor.get(key, 0), len(answers) - 1)
        self._cursor[key] = index + 1
        return answers[index]

    def rewind(self) -> None:
        self._cursor.clear()


@dataclass
class ReplayProvider:
    """An LLMProvider that answers from a cassette. No network, no clock."""

    cassette: Cassette
    name: str = "replay"
    model: str = "recorded"
    served: int = 0

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
        answer = self.cassette.take(user)
        self.served += 1
        return Completion(
            text=answer["text"],
            input_tokens=answer.get("input_tokens", 0),
            output_tokens=answer.get("output_tokens", 0),
            # A recorded truncation is replayed as one, so the escalation path
            # is exercised rather than skipped.
            truncated=bool(answer.get("truncated")),
            model=self.model,
            provider=self.name,
        )


@dataclass
class RecordingProvider:
    """Wraps a real provider and writes every answer into a cassette."""

    inner: LLMProvider
    cassette: Cassette

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", "unknown")

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
        completion = await self.inner.complete(
            system, user, max_tokens=max_tokens, temperature=temperature,
            json_mode=json_mode, timeout=timeout,
        )
        self.cassette.add(user, completion)
        return completion
