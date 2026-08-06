"""
Where one part of a resume ends and the next begins.

Every section agent used to be handed the whole document — six full passes over
the same text, each one reading past everything it was not looking for. That is
paid for twice: in tokens, and in the mistakes it invites. An agent that can see
the responsibilities of a job is an agent that can build a certification out of
one, which is a failure the audit stage then spends a model call cleaning up.

Splitting on headings turns most of that into a smaller question. The agent
that extracts education is given the education section, so the wrong answer is
no longer available to it.

The fallback is the point of the design: when a heading cannot be found with
confidence, the caller gets the whole document back and behaves exactly as it
did before. A resume with unusual headings loses the speed-up. It does not lose
its content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Headings are short, unpunctuated, and sit on their own line. These bounds are
# what keep an ordinary sentence from being read as a section break.
_MAX_HEADING_CHARS = 60
_MAX_HEADING_WORDS = 6


def heading_text(line: str, known: tuple[re.Pattern[str], ...] = ()) -> str | None:
    """The line's heading text, or None when it does not read as a heading.

    A heading is short, unpunctuated, and either fully capitalised or a name the
    caller recognises. Resumes label their sections in every style, so this is
    lenient about case and trailing colons and strict about length.

    `known` lets each caller bring its own vocabulary, because the cost of a
    mistake differs by caller: the audit stage deletes a whole section when it
    fails to spot a heading, so it recognises few names loosely, while routing
    merely falls back to the full document and can afford to recognise many.
    """
    s = (line or "").strip().lstrip("•").strip().rstrip(":").strip()
    if not s or len(s) > _MAX_HEADING_CHARS or len(s.split()) > _MAX_HEADING_WORDS:
        return None
    if s.endswith("."):
        return None
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return None
    if all(c.isupper() for c in letters):
        return s
    if any(p.search(s) for p in known):
        return s
    return None


# The section names resumes actually use. Recognising one that is not there
# costs nothing here — the section simply comes back empty and the caller falls
# back — so this list leans inclusive.
SECTION_NAME = re.compile(
    r"summary|profile|objective|about\s+me|overview"
    r"|experience|employment|work\s+history|career|organi[sz]ational\s+scan"
    r"|education|academic|qualification"
    r"|skill|competenc|expertise|technical|proficienc"
    r"|certificat|licen[cs]|credential|accredit"
    r"|project|portfolio"
    r"|training|course|workshop"
    r"|award|honou?r|recognition|accomplishment|achievement"
    # "present" is deliberately absent: it matches the "2019 - Present" on every
    # current job, which turned each job header into a section break.
    r"|publication|paper|patent|conference|presentation"
    r"|language|interest|hobb|volunteer|community"
    r"|membership|affiliation|association"
    r"|reference|activit|extra[\s-]?curricular"
    r"|personal\s+details?|contact",
    re.I,
)

EDUCATION = re.compile(r"education|academic|qualification|scholastic", re.I)

CERTIFICATION = re.compile(r"certificat|licen[cs]|credential|accredit", re.I)

# Deliberately loose, and matched by containment: "RELEVANT WORK EXPERIENCE" and
# "IT EXPERIENCE" are real headings that an anchored pattern would miss.
EXPERIENCE = re.compile(
    r"experience|employment|work\s+history|career\s+history"
    r"|professional\s+background|organi[sz]ational\s+scan",
    re.I,
)

# A heading matching EXPERIENCE that also says this is a summary, not a work
# history — "EXPERIENCE SUMMARY" and "CAREER PROFILE" head a paragraph about the
# candidate, and dropping one would take the professional summary with it.
_NOT_REALLY_EXPERIENCE = re.compile(r"summary|profile|objective|overview|highlight", re.I)


# Section headings do not carry years; job and education lines do. Used only
# when splitting for routing — never in the audit stage, where refusing to
# recognise "WORK EXPERIENCE (2015-2024)" would delete the entire work history
# rather than merely forgo a speed-up.
_HAS_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True, slots=True)
class Section:
    """One headed block: the heading line and everything under it."""

    heading: str
    text: str
    """The heading line together with its body, so an agent reading a slice can
    still see what the candidate called it."""


def split_sections(text: str) -> tuple[str, list[Section]]:
    """Split into (preamble, sections).

    The preamble is everything before the first heading — on nearly every resume
    that is the name and contact block, which belongs to no section and must not
    be lost when the document is taken apart.
    """
    lines = (text or "").split("\n")
    preamble: list[str] = []
    sections: list[Section] = []

    current_heading: str | None = None
    current: list[str] = []

    for line in lines:
        found = heading_text(line, (SECTION_NAME,))
        if found is not None and _HAS_YEAR.search(found):
            found = None  # a dated line is a job or a degree, not a heading
        if found is None:
            (current if current_heading is not None else preamble).append(line)
            continue
        if current_heading is not None:
            sections.append(Section(current_heading, "\n".join(current).strip()))
        current_heading = found
        current = [line]

    if current_heading is not None:
        sections.append(Section(current_heading, "\n".join(current).strip()))

    return "\n".join(preamble).strip(), sections


# A slice shorter than this is not a section, it is a heading with nothing under
# it — usually a false match. Falling back to the whole document is correct.
_MIN_USEFUL_SLICE = 40


def slice_matching(text: str, pattern: re.Pattern[str]) -> str | None:
    """Every section whose heading matches, joined — or None to use the lot.

    None is not an error. It means the document did not divide cleanly enough to
    be worth trusting, and the caller should read all of it.
    """
    _, sections = split_sections(text)
    hits = [s.text for s in sections if pattern.search(s.heading)]
    if not hits:
        return None
    joined = "\n\n".join(hits).strip()
    return joined if len(joined) >= _MIN_USEFUL_SLICE else None


def slice_excluding(text: str, pattern: re.Pattern[str]) -> str:
    """The document with the matching sections removed.

    For agents whose content is defined by where it is not: everything except
    the work history is a much smaller document, and none of what they are
    looking for lives there.

    A section is kept when its heading reads as a summary despite matching, and
    the whole document is returned if the removal would leave too little to be
    plausible — a resume that is nothing but work experience should still be
    read whole rather than read as nothing.
    """
    preamble, sections = split_sections(text)
    kept = [
        s.text
        for s in sections
        if not pattern.search(s.heading) or _NOT_REALLY_EXPERIENCE.search(s.heading)
    ]
    remainder = "\n\n".join([preamble, *kept]).strip()
    return remainder if len(remainder) >= _MIN_USEFUL_SLICE else text
