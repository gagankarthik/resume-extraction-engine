"""
Degrees, written the way a recruiter reads them.

Resumes spell a degree out — "Bachelor's degree in Accounting & Business
Management", "Associate's degree in Computer Science & Programming" — and the
extraction copies that verbatim, which is the right thing for it to do. What
goes on the formatted resume is the abbreviation: BS, MS, AS. Asking the model
for both the full phrase and its abbreviation put the choice of family in the
model's hands, and it drifts (a B.Tech comes back as a BS).

The mapping is a closed set, so it belongs here rather than in a prompt: the
family is decided by the words the candidate wrote, the same way every time.
"""
from __future__ import annotations

import re

# (abbreviation, pattern). Order is the whole design — the first match wins, so
# every specific family is listed before the generic level it belongs to.
# "Bachelor of Technology" must be read as BTech before the bare "bachelor" rule
# can call it a BS.
_RULES: tuple[tuple[str, str], ...] = (
    ("PhD",     r"ph\.?\s?d\.?|doctor(?:ate)?\s+of\s+philosophy|doctoral"),
    ("EdD",     r"ed\.?\s?d\.?|doctor\s+of\s+education"),
    ("DBA",     r"d\.?\s?b\.?\s?a\.?|doctor\s+of\s+business\s+administration"),
    ("MD",      r"m\.?\s?d\.?|doctor\s+of\s+medicine"),
    ("JD",      r"j\.?\s?d\.?|juris\s+doctor"),
    ("LLM",     r"ll\.?\s?m\.?|master(?:'?s)?\s+of\s+laws"),
    ("LLB",     r"ll\.?\s?b\.?|bachelor(?:'?s)?\s+of\s+laws"),
    ("MBA",     r"m\.?\s?b\.?\s?a\.?|master(?:'?s)?\s+(?:of|in)\s+business\s+administration"),
    ("BBA",     r"b\.?\s?b\.?\s?a\.?|bachelor(?:'?s)?\s+(?:of|in)\s+business\s+administration"),
    ("MCA",     r"m\.?\s?c\.?\s?a\.?|master(?:'?s)?\s+(?:of|in)\s+computer\s+applications?"),
    ("BCA",     r"b\.?\s?c\.?\s?a\.?|bachelor(?:'?s)?\s+(?:of|in)\s+computer\s+applications?"),
    ("MTech",   r"m\.?\s?tech|master(?:'?s)?\s+(?:of|in)\s+technology"),
    ("BTech",   r"b\.?\s?tech|bachelor(?:'?s)?\s+(?:of|in)\s+technology"),
    ("ME",      r"m\.?\s?e\.?|master(?:'?s)?\s+(?:of|in)\s+engineering"),
    ("BE",      r"b\.?\s?e\.?|bachelor(?:'?s)?\s+(?:of|in)\s+engineering"),
    ("MCom",    r"m\.?\s?com|master(?:'?s)?\s+(?:of|in)\s+commerce"),
    ("BCom",    r"b\.?\s?com|bachelor(?:'?s)?\s+(?:of|in)\s+commerce"),
    ("MFA",     r"m\.?\s?f\.?\s?a\.?|master(?:'?s)?\s+(?:of|in)\s+fine\s+arts"),
    ("MPH",     r"m\.?\s?p\.?\s?h\.?|master(?:'?s)?\s+(?:of|in)\s+public\s+health"),
    ("MSW",     r"m\.?\s?s\.?\s?w\.?|master(?:'?s)?\s+(?:of|in)\s+social\s+work"),
    ("MPhil",   r"m\.?\s?phil|master(?:'?s)?\s+(?:of|in)\s+philosophy"),
    ("MA",      r"m\.?\s?a\.?|master(?:'?s)?\s+(?:of|in)\s+arts"),
    ("BA",      r"b\.?\s?a\.?|bachelor(?:'?s)?\s+(?:of|in)\s+arts"),
    ("MS",      r"m\.?\s?s\.?c?\.?|master(?:'?s)?\s+(?:of|in)\s+science"),
    ("BS",      r"b\.?\s?s\.?c?\.?|bachelor(?:'?s)?\s+(?:of|in)\s+science"),
    ("PGDM",    r"p\.?\s?g\.?\s?d\.?\s?m\.?|post[\s-]?graduate\s+diploma"),
    ("AA",      r"a\.?\s?a\.?|associate(?:'?s)?\s+(?:of|in)\s+arts"),
    ("AS",      r"a\.?\s?s\.?|associate(?:'?s)?\s+(?:of|in)\s+(?:applied\s+)?science"),
    # Generic levels, last. An unqualified "Master's degree" is an MS, a
    # "Bachelor's degree" a BS — the resume did not say which family, so the
    # commonest one for that level is the only honest guess available.
    ("MS",      r"master(?:'?s)?|post[\s-]?graduate|pg\b"),
    ("BS",      r"bachelor(?:'?s)?|under[\s-]?graduate"),
    ("AS",      r"associate(?:'?s)?"),
    ("Diploma", r"diploma"),
)

_COMPILED = tuple(
    (abbrev, re.compile(rf"(?<!\w)(?:{pattern})(?!\w)", re.I))
    for abbrev, pattern in _RULES
)

# Where the degree stops and its subject begins. "in" first because it is what
# resumes overwhelmingly write; the punctuated forms cover "BS, Computer
# Science" and "BS - Computer Science".
_FIELD_SPLIT = re.compile(r"\s+in\s+|\s*[,;:]\s*|\s+-\s+", re.I)

# Filler that carries no family signal and only gets in the way of matching.
_NOISE = re.compile(r"\b(?:degree|programme?|course|studies|major|honou?rs?)\b", re.I)


def abbreviate(phrase: str | None) -> str | None:
    """The standard abbreviation for a written degree, or None if it isn't one."""
    if not isinstance(phrase, str) or not phrase.strip():
        return None
    cleaned = _NOISE.sub(" ", phrase)
    for abbrev, pattern in _COMPILED:
        if pattern.search(cleaned):
            return abbrev
    return None


def split_degree(degree: str | None) -> tuple[str | None, str | None]:
    """(abbreviation, field of study) read off a degree written out in full.

    "Bachelor's degree in Accounting & Business Management" becomes
    ("BS", "Accounting & Business Management"). Either half may come back None:
    a bare "BS" has no field, and a phrase no rule recognises has no
    abbreviation — in which case the caller keeps what the resume wrote.
    """
    if not isinstance(degree, str) or not degree.strip():
        return None, None
    head, field = _head_and_field(degree)
    abbrev = abbreviate(head)
    if abbrev is not None:
        return abbrev, field
    # The subject may have swallowed the family — "Engineering Diploma" has no
    # split point, so read the whole phrase and let the field go.
    return abbreviate(degree), None


def _head_and_field(degree: str) -> tuple[str, str | None]:
    parts = _FIELD_SPLIT.split(degree.strip(), maxsplit=1)
    head = parts[0].strip()
    field = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    return head, field


def normalize_entry(entry: dict) -> bool:
    """Rewrite one education entry's degree as its abbreviation, in place.

    Returns True when something changed. The subject is moved into
    field_of_study when the entry does not already carry one, so nothing the
    resume wrote is lost by shortening the degree line.
    """
    if not isinstance(entry, dict):
        return False

    written = entry.get("degree") or entry.get("degree_type")
    if not isinstance(written, str) or not written.strip():
        return False

    abbrev, field = split_degree(written)
    if abbrev is None:
        return False

    changed = False
    if entry.get("degree") != abbrev:
        entry["degree"] = abbrev
        changed = True
    if entry.get("degree_type") != abbrev:
        entry["degree_type"] = abbrev
        changed = True
    if field and not (entry.get("field_of_study") or "").strip():
        entry["field_of_study"] = field
        changed = True
    return changed


def normalize_education(education: object) -> int:
    """Abbreviate every degree in the list. Returns how many entries changed."""
    if not isinstance(education, list):
        return 0
    return sum(1 for entry in education if normalize_entry(entry))
