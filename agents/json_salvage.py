"""
Recovering usable data from a JSON response the model did not finish writing.

When a completion hits its token ceiling the text stops mid-structure. A strict
`json.loads` throws, and the caller that treats a throw as "this section is
empty" silently drops the whole section — thirty bullets lost because the
thirty-first was cut in half.

This module closes what is open and discards only the trailing fragment, so a
truncated response yields the complete elements it did contain. Every result is
labelled, because partial data that is not marked as partial is worse than none.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SalvageResult:
    data: Any
    """Whatever could be parsed."""

    repaired: bool
    """True when the text had to be closed off to parse."""

    dropped_fragment: str | None = None
    """The trailing text that was discarded, for logging."""


class UnsalvageableJSON(ValueError):
    """The response could not be parsed even after repair."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    end = len(lines) - 1 if lines and lines[-1].strip() == "```" else len(lines)
    return "\n".join(lines[1:end])


# How far back to search for a cut point that parses. Each candidate is one
# cheap parse of a prefix; a few hundred bounds the worst case on a long array
# while still reaching well past any single truncated record.
_MAX_CUT_CANDIDATES = 400


def _scan(text: str) -> tuple[list[str], list[int], bool]:
    """
    Walk the text once, tracking structure.

    Returns the stack of unclosed brackets, every index where the document could
    plausibly be cut, and whether the scan ended inside a string literal.

    A closed string is only a candidate when it is definitely a *value*: an
    element of an array, or the right-hand side of a colon. A string sitting
    directly inside an object with no colon before it is a key, and cutting
    after `"title"` would leave a key with no value. That distinction is what
    lets `["a","b","c"` keep all three bullets while `{"c":"C","t"` still backs
    off to the previous record.
    """
    stack: list[str] = []
    cuts: list[int] = []
    in_string = False
    escaped = False
    string_started_after_colon = False
    last_significant = ""

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                inside_array = bool(stack) and stack[-1] == "]"
                if inside_array or string_started_after_colon:
                    cuts.append(i + 1)
                last_significant = '"'
            continue

        if ch.isspace():
            continue

        if ch == '"':
            in_string = True
            string_started_after_colon = last_significant == ":"
            continue

        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            cuts.append(i + 1)  # just after a completed structure
        elif ch == ",":
            cuts.append(i)  # just before the separator, dropping what follows

        last_significant = ch

    return stack, cuts, in_string


def _close(head: str) -> str | None:
    """Close every structure still open in `head`, or None if it cannot be closed."""
    stack, _, in_string = _scan(head)
    if in_string:
        return None  # a cut candidate should never land inside a string
    if not stack:
        return head
    return head + "".join(reversed(stack))


def salvage(text: str) -> SalvageResult:
    """
    Parse `text` as JSON, repairing a truncated tail if needed.

    Raises UnsalvageableJSON when there is nothing coherent to recover.
    """
    cleaned = _strip_fences(text)
    if not cleaned:
        raise UnsalvageableJSON("Empty response.")

    try:
        return SalvageResult(data=json.loads(cleaned), repaired=False)
    except json.JSONDecodeError:
        pass

    # Some models wrap the object in prose. Take the outermost structure.
    start = min(
        (i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise UnsalvageableJSON(f"No JSON structure found in: {cleaned[:200]}")
    body = cleaned[start:]

    try:
        return SalvageResult(data=json.loads(body), repaired=False)
    except json.JSONDecodeError:
        pass

    stack, cuts, in_string = _scan(body)

    if not stack and not in_string:
        raise UnsalvageableJSON(f"Malformed JSON that is not truncated: {body[:200]}")

    # Work backwards from the last plausible cut point. The first one that
    # parses is the most data we can keep. Trying several is what makes this
    # robust: a record cut mid-key needs a cut further back than one cut
    # mid-value, and guessing a single point gets one of those cases wrong.
    for cut in reversed(cuts[-_MAX_CUT_CANDIDATES:]):
        head = body[:cut].rstrip().rstrip(",")
        if not head:
            continue

        candidate = _close(head)
        if candidate is None:
            continue

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        dropped = body[cut:]
        logger.warning(
            "Recovered a truncated JSON response, discarding %d trailing character(s).",
            len(dropped),
        )
        return SalvageResult(
            data=data, repaired=True, dropped_fragment=dropped[:200] or None
        )

    raise UnsalvageableJSON("Response was cut before the first complete value.")


def count_items(data: Any) -> int:
    """How many records survived, for the extraction report."""
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return sum(len(v) if isinstance(v, list) else 1 for v in data.values())
    return 0
