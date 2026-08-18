"""
The last pass over the merged result: shape, not content.

Everything here is deterministic and works only on what the extraction already
returned. It fixes the handful of presentation faults that survive every prompt
because they are not really extraction mistakes — a nickname the resume prints
in brackets, a degree written out in full, the same credential arriving once as
a certification and again as a training course, a client label copied along with
the client's name, an end date that says "Present" where the formatted resume
says "Till Date".

Prompts have been asked to handle these and cannot be relied on to: each is a
rule about the shape of the output, and a rule about shape is cheaper and more
honest to apply in code, where it holds every time.
"""
from __future__ import annotations

import re

from agents.degrees import normalize_education

# ── Names ───────────────────────────────────────────────────────────────────

# A nickname the resume prints beside the legal name: "BalaKrishna (Krishna)
# Jallipalli", 'Robert "Bob" Smith'. It is the same person written twice, and on
# a formatted resume it reads as a typo. Only paired brackets and double quotes
# are stripped — a lone apostrophe belongs to names like O'Brien.
_BRACKETED = re.compile(r"\s*[(\[]\s*[^)\]]{1,30}\s*[)\]]\s*")
_QUOTED = re.compile(r"\s*[\"“”]\s*[^\"“”]{1,30}\s*[\"“”]\s*")
_SPACES = re.compile(r"\s{2,}")


def clean_name(name: object) -> str | None:
    """A person's name with any parenthetical or quoted nickname removed."""
    if not isinstance(name, str) or not name.strip():
        return None
    out = _QUOTED.sub(" ", _BRACKETED.sub(" ", name))
    out = _SPACES.sub(" ", out).strip(" ,-")
    return out or None


# Resumes print the candidate's name in whatever case the header design used —
# "SHASHE KIRAN GANJI", "shashe kiran ganji" — and copying that verbatim onto a
# formatted resume reads as shouting or as sloppy. Only a token written
# ALL-CAPS or all-lowercase is recased; a token already mixed-case ("McDonald",
# "O'Brien", "DeSouza") was written that way on purpose and is left alone, since
# a blind title-case would flatten "O'Brien" to "O'brien".
def _titlecase_token(token: str) -> str:
    letters = [c for c in token if c.isalpha()]
    if not letters or not (all(c.isupper() for c in letters) or all(c.islower() for c in letters)):
        return token
    return re.sub(r"[A-Za-z]+", lambda m: m.group(0)[:1].upper() + m.group(0)[1:].lower(), token)


def titlecase_name(name: str | None) -> str | None:
    """A name with each shouted or all-lowercase token recased to Title Case."""
    if not name:
        return name
    return " ".join(_titlecase_token(tok) for tok in name.split(" "))


def polish_personal(personal: object) -> None:
    """Drop nicknames from the name fields, fix shouted/lowercase casing, and
    keep the three of them agreeing."""
    if not isinstance(personal, dict):
        return

    full = titlecase_name(clean_name(personal.get("full_name")))
    first = titlecase_name(clean_name(personal.get("first_name")))
    last = titlecase_name(clean_name(personal.get("last_name")))

    # A full name assembled from the parts is better than no full name; the
    # reverse — parts read off the full name — is how they are normally filled.
    if not full and (first or last):
        full = " ".join(p for p in (first, last) if p)

    if full:
        personal["full_name"] = full
        parts = full.split()
        if not first and parts:
            first = parts[0]
        # The surname is the last token, but only when the name has more than
        # one: a single-token full name is a first name, not a last name.
        if not last and len(parts) > 1:
            last = parts[-1]

    if first:
        personal["first_name"] = first
    if last:
        personal["last_name"] = last


# ── Labels copied along with the value ──────────────────────────────────────

# Resumes label these fields inline ("Client: Diligent Insurance"), and the
# label comes back attached to the value. It is scaffolding from the source
# document, not part of the name.
_FIELD_LABEL = re.compile(
    r"^\s*(?:end\s+)?(?:client|customer|employer|company|organi[sz]ation|account"
    r"|project|engagement|role|title|position|department|dept|location|duration"
    r"|technologies|environment)\s*[:\-–—]\s*",
    re.I,
)


def strip_label(value: object) -> object:
    """The value without the "Client:" / "Project:" prefix the resume printed."""
    if not isinstance(value, str):
        return value
    out = _FIELD_LABEL.sub("", value).strip()
    return out or None


# ── Current roles ───────────────────────────────────────────────────────────

_PRESENT = re.compile(
    r"^\s*(?:since\s+)?(?:present|current(?:ly)?|now|to\s*date|till\s*date|till\s*now"
    r"|to\s*present|ongoing|date)\s*$",
    re.I,
)

CURRENT_END_DATE = "Till Date"

# "Since June 2025 -" and "Nov - 2010" are how the resume prints a date beside
# its heading; the field wants the date. Leading connectives and dangling
# separators go, and a separator left stranded inside the date closes up.
_DATE_LEAD = re.compile(r"^\s*(?:since|from|w\.e\.f\.?|effective)\s+", re.I)
_DATE_EDGES = re.compile(r"^[\s\-–—:|,]+|[\s\-–—:|,]+$")
_DATE_STRANDED = re.compile(r"\s+[-–—]\s+(?=\d)")


def _is_present(value: object) -> bool:
    return isinstance(value, str) and bool(_PRESENT.match(value))


def tidy_date(value: object) -> object:
    """A date field with the resume's sentence scaffolding taken off."""
    if not isinstance(value, str):
        return value
    out = _DATE_STRANDED.sub(" ", _DATE_EDGES.sub("", _DATE_LEAD.sub("", value)))
    return out.strip() or None


def _squash(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def polish_work(work: object) -> None:
    """Clean labels off the job/client/project names and settle current roles.

    A job the candidate still holds is written "Till Date" — one spelling for
    what the resume writes half a dozen ways, so the formatted output does not
    show "Present" on one role and "Current" on the next.
    """
    if not isinstance(work, list):
        return

    for job in work:
        if not isinstance(job, dict):
            continue

        for field in ("company_name", "job_title", "department", "location"):
            if field in job:
                job[field] = strip_label(job.get(field))
        for field in ("start_date", "end_date"):
            if field in job:
                job[field] = tidy_date(job.get(field))

        if _is_present(job.get("end_date")) or job.get("is_current") is True:
            job["end_date"] = CURRENT_END_DATE
            job["is_current"] = True
        elif job.get("end_date"):
            job["is_current"] = False

        projects = job.get("projects")
        if not isinstance(projects, list):
            continue

        covered: set[str] = set()
        for proj in projects:
            if not isinstance(proj, dict):
                continue
            for field in ("projectName", "clientName", "projectLocation", "keyTechnologies"):
                if field in proj:
                    proj[field] = strip_label(proj.get(field))
            for bullet in proj.get("projectResponsibilities") or []:
                if isinstance(bullet, str):
                    covered.add(_squash(bullet))

        # A bullet that sits under a client/project must not also sit in the
        # job's flat list — the rendered resume would print it twice, once per
        # place it was found.
        resp = job.get("responsibilities")
        if covered and isinstance(resp, list):
            job["responsibilities"] = [
                r for r in resp
                if not (isinstance(r, str) and _squash(r) in covered)
            ]


# ── One role, listed once ───────────────────────────────────────────────────


def _bullet_count(job: dict) -> int:
    total = len(job.get("responsibilities") or [])
    for proj in job.get("projects") or []:
        if isinstance(proj, dict):
            total += len(proj.get("projectResponsibilities") or [])
    return total


def _absorb(winner: dict, loser: dict) -> None:
    """Fill the winner's empty fields from the duplicate before discarding it."""
    for key, value in loser.items():
        if value in (None, "", [], {}):
            continue
        if winner.get(key) in (None, "", [], {}):
            winner[key] = value


def dedupe_roles(work: object) -> int:
    """Collapse the same engagement listed twice. Returns how many went.

    Long consulting resumes carry a summary table of engagements AND a detailed
    section describing the same work, and a reader that treats both as job
    entries returns every role twice. The employer alone cannot decide this —
    this candidate has five separate stints at one client — but the same
    employer STARTING on the same date is one role written down twice.

    The copy with more bullets wins, since that is the detailed entry; anything
    the shorter copy knew and the longer one did not is carried across first, so
    a date or location that only the summary table printed is not lost with it.
    """
    if not isinstance(work, list):
        return 0

    best: dict[tuple[str, str], dict] = {}
    kept: list = []
    removed = 0

    for job in work:
        if not isinstance(job, dict):
            kept.append(job)
            continue
        key = (_squash(job.get("company_name")), _squash(job.get("start_date")))
        if not key[0] or not key[1]:
            kept.append(job)          # not enough identity to judge — keep it
            continue

        previous = best.get(key)
        if previous is None:
            best[key] = job
            kept.append(job)
            continue

        removed += 1
        if _bullet_count(job) > _bullet_count(previous):
            # The newcomer is the fuller entry: put it where the old one sat.
            _absorb(job, previous)
            kept[kept.index(previous)] = job
            best[key] = job
        else:
            _absorb(previous, job)

    work[:] = kept
    return removed


# ── Credentials ─────────────────────────────────────────────────────────────

# "Certified Agile Scrum Master - Scrum Alliance, USA" is a name and an issuer
# on one line. Split on a dash with space either side; a hyphen inside a word
# ("Post-Graduate") has no spaces and is left alone.
_ISSUER_SPLIT = re.compile(r"\s+[-–—]\s+|\s+\|\s+")

# Leading section noise the model sometimes carries into the first entry.
_CERT_PREFIX = re.compile(
    r"^\s*(?:certificat(?:ion|e)s?|licen[cs]es?|credentials?|training)\s*[:\-]\s*", re.I
)


def polish_certifications(certifications: object) -> int:
    """Split issuer off the name, drop label noise, and remove repeats.

    Returns how many duplicate entries were removed.
    """
    if not isinstance(certifications, list):
        return 0

    seen: set[str] = set()
    kept: list = []
    for cert in certifications:
        if not isinstance(cert, dict):
            continue
        name = cert.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = _CERT_PREFIX.sub("", name).strip().lstrip("•").strip()
        if not name:
            continue

        if not (cert.get("issuing_organization") or "").strip():
            parts = _ISSUER_SPLIT.split(name, maxsplit=1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                name = parts[0].strip()
                cert["issuing_organization"] = parts[1].strip()
        cert["name"] = name

        key = _squash(name)
        if key in seen:
            continue
        seen.add(key)
        kept.append(cert)

    removed = len(certifications) - len(kept)
    certifications[:] = kept
    return removed


def dedupe_against_certifications(merged: dict) -> int:
    """Remove training/course entries that repeat a certification.

    The same credential list reaches the output twice — once from the
    certifications section and once from the supplemental pass, which reads the
    same block under its "training" heading. Matching ignores punctuation, so an
    entry separated by an em dash in one list and a hyphen in the other still
    counts as the same credential.
    """
    certs = merged.get("certifications")
    if not isinstance(certs, list):
        return 0
    known = {_squash(c.get("name")) for c in certs if isinstance(c, dict)}
    known |= {
        _squash(f"{c.get('name')} {c.get('issuing_organization')}")
        for c in certs if isinstance(c, dict)
    }
    known.discard("")
    if not known:
        return 0

    removed = 0
    for section in ("training", "courses"):
        items = merged.get(section)
        if not isinstance(items, list):
            continue
        kept = [
            it for it in items
            if not (isinstance(it, dict) and (
                _squash(it.get("name")) in known
                or _squash(f"{it.get('name')} {it.get('provider')}") in known
            ))
        ]
        removed += len(items) - len(kept)
        merged[section] = kept
    return removed


# ── Bullet formatting for output ────────────────────────────────────────────

# The array survives the whole pipeline as separate items — StructureAgent's
# bullet count, the auditor's per-bullet grounding and split-metric rejoin, the
# validator's count cross-check — all depend on one array element per bullet.
# But the document tool downstream of this API has been rendering that array as
# one run-on paragraph instead of a bulleted list, and there is no fixing that
# from here. What IS in reach: matching the same fix professional_summary
# already uses for the identical problem — hand back one string with each
# bullet on its own "• "-prefixed line, so the bullets survive even a renderer
# that only knows how to print a string. This runs last, after every stage that
# needs the array form is done with it.


def bulletize(items: object) -> object:
    """A list of responsibility strings as one "• "-per-line string."""
    if not isinstance(items, list) or not items:
        return items
    lines = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    if not lines:
        return items
    return "\n".join(f"• {line}" for line in lines)


def bulletize_work_experience(work: object) -> None:
    """Turn each job's responsibilities, and each project's, into bullet text."""
    if not isinstance(work, list):
        return
    for job in work:
        if not isinstance(job, dict):
            continue
        if "responsibilities" in job:
            job["responsibilities"] = bulletize(job["responsibilities"])
        for proj in job.get("projects") or []:
            if isinstance(proj, dict) and "projectResponsibilities" in proj:
                proj["projectResponsibilities"] = bulletize(proj["projectResponsibilities"])


# ── Entry point ─────────────────────────────────────────────────────────────


def polish(merged: dict) -> list[str]:
    """Apply every shape fix to the merged result. Returns notes for the audit."""
    if not isinstance(merged, dict):
        return []

    notes: list[str] = []

    polish_personal(merged.get("personal_information"))

    changed = normalize_education(merged.get("education"))
    if changed:
        notes.append(f"Shortened {changed} degree name(s) to their standard abbreviation")

    polish_work(merged.get("work_experience"))

    # After polish_work, so the dates it tidies are the ones compared: a role
    # written "Since June 2025 -" in one place and "June 2025" in another is the
    # same role, and only reads that way once both have been cleaned.
    repeated = dedupe_roles(merged.get("work_experience"))
    if repeated:
        notes.append(f"Removed {repeated} role(s) the resume lists twice")

    duplicates = polish_certifications(merged.get("certifications"))
    if duplicates:
        notes.append(f"Removed {duplicates} duplicate certification(s)")

    repeats = dedupe_against_certifications(merged)
    if repeats:
        notes.append(f"Removed {repeats} training/course entr(y/ies) already listed as certifications")

    return notes
