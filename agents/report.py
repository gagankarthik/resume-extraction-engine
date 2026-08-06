"""
Per-request record of how each section actually went.

The pipeline is deliberately fault-tolerant: one agent failing must not take the
whole extraction down. The cost of that is silence — a section can come back
empty because the resume had nothing in it, or because the call failed, and the
JSON looks identical either way.

This module keeps the difference. Every agent reports its outcome, the report
travels out with the response, and the editor shows the user which sections it
should not trust.
"""
from __future__ import annotations

import contextvars
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    OK = "ok"
    """Completed and parsed cleanly."""

    PARTIAL = "partial"
    """The model ran out of room; complete records were kept, the tail was lost."""

    RECOVERED = "recovered"
    """Failed once, then succeeded on a retry or a narrower pass."""

    FAILED = "failed"
    """Produced nothing usable. The section is empty because of an error."""

    SKIPPED = "skipped"
    """A refinement pass the run had no time left for.

    Distinct from FAILED on purpose: nothing was lost, it simply was not
    double-checked. The extraction stands; the extra verification does not."""


@dataclass
class SectionOutcome:
    section: str
    status: Status
    items: int = 0
    attempts: int = 1
    detail: str | None = None
    """Plain-language explanation, shown to the user."""


@dataclass
class ExtractionReport:
    sections: list[SectionOutcome] = field(default_factory=list)

    def record(
        self,
        section: str,
        status: Status,
        *,
        items: int = 0,
        attempts: int = 1,
        detail: str | None = None,
    ) -> None:
        # One row per section — a later outcome replaces an earlier one.
        self.sections = [s for s in self.sections if s.section != section]
        self.sections.append(
            SectionOutcome(
                section=section,
                status=status,
                items=items,
                attempts=attempts,
                detail=detail,
            )
        )

    @property
    def degraded(self) -> bool:
        return any(s.status in (Status.PARTIAL, Status.FAILED) for s in self.sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [asdict(s) | {"status": s.status.value} for s in self.sections],
            "degraded": self.degraded,
            "failed_sections": [
                s.section for s in self.sections if s.status == Status.FAILED
            ],
            "partial_sections": [
                s.section for s in self.sections if s.status == Status.PARTIAL
            ],
            "skipped_sections": [
                s.section for s in self.sections if s.status == Status.SKIPPED
            ],
        }


_report: contextvars.ContextVar[ExtractionReport | None] = contextvars.ContextVar(
    "extraction_report", default=None
)


def reset_report() -> ExtractionReport:
    """Begin a fresh report for the current request."""
    report = ExtractionReport()
    _report.set(report)
    return report


def get_report() -> ExtractionReport | None:
    return _report.get()


def record(section: str, status: Status, **kwargs: Any) -> None:
    """Record an outcome if a report is active; a no-op otherwise."""
    report = _report.get()
    if report is not None:
        report.record(section, status, **kwargs)
