"""
The run's clock, shared by every stage.

The pipeline used to hold exactly one notion of time: a single
`asyncio.wait_for` wrapped around the whole orchestrator. When it fired, every
stage that had already succeeded was thrown away and the caller got an error
instead of a resume — the worst possible trade, because the expensive work was
already done and paid for.

A Deadline replaces that with a budget every stage can read. The stages that
produce the resume take what they need; the stages that only refine it ask
whether the run can still afford them and step aside when it cannot. A slow
resume comes back less polished, with the report saying so. It does not come
back as an error.

The deadline lives in a ContextVar so the layer that actually spends the time —
the LLM client — can consult it without threading a parameter through every
agent signature, and so concurrent requests in one process never share a clock.
"""
from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass

INFINITY = float("inf")


class DeadlineExceeded(RuntimeError):
    """The run is out of time, so this call was never made."""


@dataclass(frozen=True, slots=True)
class Deadline:
    """A monotonic point in time the run must finish by."""

    ends_at: float

    @classmethod
    def in_seconds(cls, seconds: float | None) -> Deadline:
        if seconds is None or seconds <= 0:
            return cls(INFINITY)
        return cls(time.monotonic() + seconds)

    @classmethod
    def unlimited(cls) -> Deadline:
        return cls(INFINITY)

    def remaining(self) -> float:
        if self.ends_at == INFINITY:
            return INFINITY
        return max(0.0, self.ends_at - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def allows(self, seconds: float) -> bool:
        """Is there room left for a stage expected to take roughly this long?

        Refinement stages call this before starting. Answering honestly matters
        more than answering optimistically: a stage that begins with too little
        budget spends what is left and still gets cancelled, so the run loses
        the time twice.
        """
        return self.remaining() >= seconds


_deadline: contextvars.ContextVar[Deadline | None] = contextvars.ContextVar(
    "extraction_deadline", default=None
)


def set_deadline(deadline: Deadline) -> None:
    _deadline.set(deadline)


def get_deadline() -> Deadline:
    """The current run's deadline, or an unlimited one outside a run."""
    return _deadline.get() or Deadline.unlimited()
