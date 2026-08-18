"""
CompletenessAuditorAgent — final verification stage (Stage 5).

Closes the loop between the original resume text and the merged extraction:

1. GROUNDEDNESS (deterministic, no LLM): values that must literally exist in
   the source text — emails, phones, URLs, sub-project names, client names —
   are verified against it. Ungrounded values are removed (hallucination
   guard) and reported as warnings.
2. COVERAGE (deterministic, no LLM): finds significant resume lines whose
   content never made it into ANY extracted field.
3. RECOVERY (one targeted LLM call, only when coverage gaps exist): extracts
   ONLY the missed content and merges it ADDITIVELY — existing data is never
   overwritten or removed by this step.

The agent never raises out of run(): any internal failure leaves the merged
data unchanged and is reported in the audit dict attached at merged["_audit"].
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .base import BaseAgent, output_budget
from .sections import heading_text
from .skills import derive_union_fields, tidy_skill_list

logger = logging.getLogger(__name__)

# ── Tokenisation helpers (pure, unit-testable) ──────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall((s or "").lower())


def _squash(s: str) -> str:
    """Lowercase and remove all whitespace — for literal containment checks."""
    return re.sub(r"\s+", "", (s or "").lower())


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


# ── Groundedness ────────────────────────────────────────────────────────────

def is_grounded(value: str, text_token_set: set[str]) -> bool:
    """True if the value's significant tokens all appear in the source text.

    Short values (≤3 tokens, e.g. client names) require EVERY token present;
    longer values pass at ≥80% so minor punctuation/splitting differences in
    the source don't cause false positives.
    """
    toks = _tokens(value)
    if not toks:
        return True  # nothing checkable — don't judge
    present = sum(1 for t in toks if t in text_token_set)
    if len(toks) <= 3:
        return present == len(toks)
    return present / len(toks) >= 0.8


# ── Fabricated-bullet guard ─────────────────────────────────────────────────

# Quantified-impact phrasing the LLM tends to invent: "by 40%", "$2M", "3x".
_METRIC_RE = re.compile(r"\d+(?:\.\d+)?\s*%|\$\s*\d|\bby\s+\d|\b\d+\s*x\b", re.I)
# Achievement/impact verbs that lead AI-padded sentences.
_IMPACT_LEAD_RE = re.compile(
    r"^\W*(improv|reduc|increas|accelerat|deliver|optimi[sz]|enhanc|streamlin|"
    r"boost|driv|achiev|generat|grew|grow|cut|decreas|sav|spearhead|slash|"
    r"maximi[sz]|minimi[sz]|elevat|transform)",
    re.I,
)


def _bullet_is_fabricated(bullet: str, token_set: set[str]) -> bool:
    """True only when a responsibility/achievement looks like AI padding.

    Guard is deliberately conservative: a bullet is removed ONLY when BOTH
    hold, so genuine verbatim content is never lost:

    1. It is NOT grounded in the source text — its significant words largely
       fail to trace back to the resume (every real bullet is copied verbatim,
       so real content stays grounded even when it contains metrics).
    2. It is phrased as quantified/impact padding — it quotes a metric
       ("by 40%", "$2M") or opens with an achievement verb ("Improved…",
       "Reduced…", "Delivered measurable cost optimization…").

    This targets invented statistics and generic impact sentences while leaving
    ordinary duty bullets (even oddly tokenized ones) untouched.
    """
    if not isinstance(bullet, str) or not bullet.strip():
        return False
    if is_grounded(bullet, token_set):
        return False  # words trace to the resume → real content
    return bool(_METRIC_RE.search(bullet) or _IMPACT_LEAD_RE.match(bullet))


# ── Invented-figure guard ───────────────────────────────────────────────────
#
# The fabricated-bullet guard above only catches a WHOLLY invented sentence. The
# more common failure is subtler and survives it: the model copies a real bullet
# and staples a number onto the end — "Led the ETL migration" becomes "Led the
# ETL migration, improving throughput by 40%". Its words all trace back to the
# resume, so is_grounded() passes it, and a percentage the candidate never
# claimed goes out on a submitted document.
#
# Numbers are checkable in a way prose is not: a figure is either written in the
# resume or it is not. Everything below is deterministic — no LLM, no judgement.

# Figures the model invents as "impact": percentages, money, multipliers. Bare
# integers are deliberately excluded — years, versions ("Python 3.9") and team
# sizes would produce false positives.
_FIGURE_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*%"
    r"|(?:USD|INR|EUR|GBP|[$₹€£])\s*\d+(?:[.,]\d+)*\s*"
    r"(?:k\b|m\b|b\b|mn\b|bn\b|million|billion|thousand|crore|cr\b|lakh|lac\b)?"
    r"|\b\d+(?:\.\d+)?\s*x\b",
    re.I,
)

_MAGNITUDE = {
    "thousand": "k", "million": "m", "billion": "b", "crore": "cr",
    "lakh": "l", "lac": "l", "mn": "m", "bn": "b",
    "k": "k", "m": "m", "b": "b", "cr": "cr",
}


def _figure_key(fig: str) -> str:
    """Canonical form of a figure, so "$2 million" and "$2M" compare equal."""
    s = fig.lower().replace(",", "").replace(" ", "")
    num = re.search(r"\d+(?:\.\d+)?", s)
    if not num:
        return s
    n = num.group(0)
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    if "%" in s:
        return f"pct:{n}"
    if re.search(r"\d\s*x$", s):
        return f"mult:{n}"
    tail = s[num.end():]
    mag = next((v for k, v in _MAGNITUDE.items() if tail.startswith(k)), "")
    return f"money:{n}{mag}"


def _figures(text: str) -> set[str]:
    return {_figure_key(m.group(0)) for m in _FIGURE_RE.finditer(text or "")}


def source_line_index(raw_text: str) -> list[tuple[str, set[str]]]:
    """Substantial resume lines paired with their token sets, for re-anchoring.

    Whole bullets — continuation lines folded back in — are indexed alongside
    the individual lines. A bullet whose trailing metric wrapped onto its own
    line has to be matchable as the one thing the candidate wrote, or the metric
    it genuinely carries reads as invented.
    """
    index: list[tuple[str, set[str]]] = []
    seen: set[str] = set()
    lines = [
        re.sub(r"^[•●◦‣⁃∙·○▪▸\-–—*]\s*", "", line.strip()).strip()
        for line in (raw_text or "").split("\n")
    ]
    for chunk in [*lines, *source_bullet_blocks(raw_text)]:
        toks = set(_tokens(chunk))
        if len(toks) < 4 or chunk in seen:
            continue
        seen.add(chunk)
        index.append((chunk, toks))
    return index


def _source_match(bullet: str, index: list[tuple[str, set[str]]]) -> str | None:
    """The resume line this bullet was copied from, if one clearly matches.

    Requires agreement in BOTH directions — most of the source line's words
    appear in the bullet, and most of the bullet's words come from that line —
    so a bullet is never swapped for a different job's text.

    Ties go to the longer candidate. A wrapped bullet is indexed both as its
    first line and as the whole bullet, and both score perfectly against a
    faithful copy; the whole bullet is the one that accounts for all of it, and
    picking the first line instead would read its second half as invented.
    """
    btoks = set(_tokens(bullet))
    if not btoks:
        return None
    best: str | None = None
    best_rank = (0.0, 0)
    for line, ltoks in index:
        overlap = len(ltoks & btoks)
        rank = (overlap / len(ltoks), len(ltoks))
        if rank > best_rank and overlap / len(btoks) >= 0.5:
            best, best_rank = line, rank
    return best if best_rank[0] >= 0.6 else None


# Clause boundaries an appended impact phrase hangs off.
_CLAUSE_BOUNDARY_RE = re.compile(r"\s*[,;(]\s*|\s+[—–-]\s+")


def _excise_figure(bullet: str, bad: set[str]) -> str | None:
    """Cut the clause carrying an invented figure, keeping the real text before it.

    Only the trailing clause is removed, and only when a substantial head
    survives with no ungrounded figures left. Nothing is reworded and no
    punctuation is added — the head is returned exactly as it was written.
    """
    for m in _FIGURE_RE.finditer(bullet):
        if _figure_key(m.group(0)) not in bad:
            continue
        cut = None
        for b in _CLAUSE_BOUNDARY_RE.finditer(bullet):
            if b.start() >= m.start():
                break
            cut = b.start()
        if cut is None:
            return None
        head = bullet[:cut].rstrip(" ,;:-–—(")
        if len(head.split()) < 5 or _figures(head) & bad:
            return None
        return head
    return None


def _nearby_figures(bullet: str, index: list[tuple[str, set[str]]]) -> set[str]:
    """Figures written on resume lines that share real subject matter with this
    bullet — the figures it could plausibly have been copied with."""
    btoks = set(_tokens(bullet))
    out: set[str] = set()
    for line, ltoks in index:
        if len(ltoks & btoks) >= 3:
            out |= _figures(line)
    return out


def repair_figures(
    bullet: str,
    source_figures: set[str],
    index: list[tuple[str, set[str]]],
) -> str | None:
    """Return the bullet verbatim, a corrected version, or None to drop it.

    Order of preference: leave it alone → restore the resume's own line →
    cut the invented clause → drop. A bullet whose figures all appear in the
    resume is returned untouched, which is the overwhelmingly common case.

    Which figures count as "in the resume" is decided against the bullet's OWN
    source line, not the document. Checking the document was the leak that let
    padding through: the model copies a real bullet and staples "reducing
    onboarding time 40%" onto it, and because some unrelated line elsewhere on
    the resume happens to say 40%, the figure was accepted. A bullet is copied
    verbatim from one place, so that place is what its numbers must match. The
    document-wide set is still the fallback for a bullet whose source cannot be
    located at all — there is nothing tighter to compare it against.
    """
    if not isinstance(bullet, str) or not bullet.strip():
        return bullet
    figures = _figures(bullet)
    if not figures:
        return bullet

    original = _source_match(bullet, index)
    if original is not None:
        allowed = _figures(original)
    else:
        allowed = _nearby_figures(bullet, index) or source_figures

    bad = figures - allowed
    if not bad:
        return bullet

    if original and not (_figures(original) & bad):
        return original

    return _excise_figure(bullet, bad)


def _scrub_bullets(
    items: Any,
    token_set: set[str],
    source_figures: set[str] | None = None,
    index: list[tuple[str, set[str]]] | None = None,
) -> tuple[list, int, int]:
    """Return (kept_items, dropped_count, repaired_count) for a bullet list.

    Repair runs BEFORE the fabricated-bullet test, and the test then judges the
    repaired text. The order matters: an invented impact clause drags its whole
    bullet out of the source's vocabulary, so a real duty with padding stapled
    to it reads as ungrounded and used to be deleted outright — taking the
    candidate's own sentence with it. Cutting the padding first leaves the real
    half to be recognised for what it is.
    """
    if not isinstance(items, list):
        return items, 0, 0
    kept: list = []
    repaired = 0
    for b in items:
        fixed = b
        if source_figures is not None and index is not None:
            fixed = repair_figures(b, source_figures, index)
            if fixed is None:
                continue  # invented figure that could not be traced or cut
        if _bullet_is_fabricated(fixed, token_set):
            continue
        if isinstance(b, str) and fixed != b:
            repaired += 1
        kept.append(fixed)
    return kept, len(items) - len(kept), repaired


# ── Split-bullet repair ─────────────────────────────────────────────────────


# A line that opens something new rather than continuing the bullet above it.
# Without these, a resume that stops using glyphs part-way — and long ones
# nearly all do — folds every remaining line of the document into whichever
# bullet happened to be last, and that runaway block then replaces real
# responsibilities with a paragraph containing another job's header.
_SECTION_LABEL = re.compile(
    r"^\W*(?:employer|end\s+client|client|customer|role|responsibilit\w*|project\s+description"
    r"|area\s+of\s+work|work\s+done|important\s+(?:work|jobs|things)\s+done|period|duration"
    r"|environment|technologies|organi[sz]ation|education|skills?|certificat\w*)\b\s*[-:–—]",
    re.I,
)

# "Jun'07 - Jan' 09", "Aug 2013 - Jun 2015", "Nov 2015 - Present" on its own
# line: a job header, never the tail of the bullet above it.
_DATE_RANGE_LINE = re.compile(
    r"^\W*(?:since\s+)?[A-Za-z]{3,9}\.?\s*['’]?\s*\d{2,4}\s*[-–—]+\s*"
    r"(?:[A-Za-z]{3,9}\.?\s*['’]?\s*\d{2,4}|present|current|till\s*date|to\s*date)?\W*$",
    re.I,
)

# A bullet that wraps is a fragment of one line, not a paragraph. Continuation
# lines beyond this are new content that lost its glyph.
_MAX_CONTINUATION_LINES = 2


def _continues_bullet(line: str) -> bool:
    """Does this line read as the tail of the bullet above it?"""
    return not (
        "|" in line                       # a table row
        or _SECTION_LABEL.match(line)
        or _DATE_RANGE_LINE.match(line)
        or line.rstrip().endswith(":")    # "Responsibilities:" style heading
    )


def source_bullet_blocks(raw_text: str) -> list[str]:
    """The resume's bullets, each with its continuation lines folded back in.

    normalize_text() has already rewritten every leading glyph to "• " and
    rejoined every lowercase wrap, so what remains to fold is the wrap that
    starts with a capital — "Achieved 80% accuracy…" on its own line. That is a
    line or two, which is why the folding is bounded: an unbounded run swallows
    the rest of the resume the moment the candidate stops using glyphs.
    """
    blocks: list[str] = []
    current: str | None = None
    folded = 0
    for line in (raw_text or "").split("\n"):
        stripped = line.strip()
        if not stripped:
            current = None
            continue
        if stripped.startswith("•"):
            if current:
                blocks.append(current)
            current = stripped.lstrip("•").strip()
            folded = 0
        elif current is not None:
            if folded >= _MAX_CONTINUATION_LINES or not _continues_bullet(stripped):
                blocks.append(current)
                current = None
                continue
            current = f"{current} {stripped}"
            folded += 1
    if current:
        blocks.append(current)
    return [b for b in blocks if b]


def merge_split_bullets(
    responsibilities: Any,
    blocks: list[str],
    blocks_squashed: list[str],
) -> tuple[Any, int]:
    """Rejoin responsibilities that the model split out of one source bullet.

    A trailing detail — most often a metric, "Achieved 80% accuracy" — gets
    emitted as its own responsibility, so the rendered resume shows two bullets
    where the candidate wrote one. Consecutive responsibilities that trace back
    to the SAME source bullet are replaced by that bullet's own text, which
    restores both the count and the wording without composing anything.
    """
    if not isinstance(responsibilities, list) or len(responsibilities) < 2:
        return responsibilities, 0

    # Which source bullet did each responsibility come from? Exact containment
    # of the squashed text — responsibilities are copied verbatim, so a looser
    # test would risk fusing two genuinely separate bullets.
    origins: list[int | None] = []
    for item in responsibilities:
        squashed = _squash(item) if isinstance(item, str) else ""
        origins.append(
            next((i for i, block in enumerate(blocks_squashed) if squashed and squashed in block), None)
        )

    out: list = []
    merged = 0
    i = 0
    while i < len(responsibilities):
        j = i
        while j + 1 < len(responsibilities) and origins[i] is not None and origins[j + 1] == origins[i]:
            j += 1
        if j > i and _is_faithful_rejoin(responsibilities[i:j + 1], blocks[origins[i]]):
            out.append(blocks[origins[i]])
            merged += j - i
            i = j + 1
        else:
            out.extend(responsibilities[i:j + 1])
            i = j + 1
    return out, merged


# A rejoin puts back the glue between two halves of one bullet — a space, a
# comma, a couple of words. Anything substantially longer than the pieces it
# replaces is not the same bullet, and swapping it in would both lose the real
# text and import whatever else the block picked up.
_REJOIN_SLACK = 0.25
_REJOIN_ALLOWANCE = 20


def _is_faithful_rejoin(parts: list, block: str) -> bool:
    """True when `block` is the parts written as one bullet, not a paragraph."""
    written = sum(len(_squash(p)) for p in parts if isinstance(p, str))
    return len(_squash(block)) <= written * (1 + _REJOIN_SLACK) + _REJOIN_ALLOWANCE


# ── Technology guard ────────────────────────────────────────────────────────


def _flatten(s: str) -> str:
    """Lowercase and drop everything that is not a letter or digit."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def source_terms(raw_text: str) -> tuple[set[str], str, str]:
    """The three views of the source text the technology guard compares against."""
    return set(_tokens(raw_text)), (raw_text or "").lower(), _flatten(raw_text)


def tech_is_grounded(tech: str, src: tuple[set[str], str, str]) -> bool:
    """True only when the technology is NAMED in the resume.

    Technologies are the easiest field to hallucinate, because the model reads
    a duty and supplies the tool that duty implies: "security and access
    management" yields IAM, "containers" yields Kubernetes, "cloud" yields AWS.
    Those are the model's inferences about the candidate's stack, not the
    candidate's claims, and on a submitted resume they are fabricated skills.

    Three ways to pass, in order of strictness:
      1. every significant word of the name appears in the source;
      2. the name with punctuation removed appears in the equally flattened
         source — this rescues spelling variants ("NodeJS" for "Node.js",
         "CI/CD" for "CICD"). Restricted to 4+ characters so a short acronym
         like IAM can never be matched inside an unrelated word;
      3. names too short or too symbolic to tokenise at all (C#, R, Go, C++)
         get a word-boundary search against the raw text.

    Errors lean toward keeping: a false keep leaves the resume's own wording in
    place, whereas a false drop deletes something the candidate really wrote.
    """
    token_set, raw_lower, flat_source = src
    if not isinstance(tech, str) or not tech.strip():
        return False
    toks = _tokens(tech)
    if toks:
        if all(t in token_set for t in toks):
            return True
        flat = _flatten(tech)
        return len(flat) >= 4 and flat in flat_source
    return re.search(rf"(?<!\w){re.escape(tech.strip().lower())}(?!\w)", raw_lower) is not None


def _scrub_techs(items: Any, src: tuple[set[str], str, str]) -> tuple[list, int]:
    if not isinstance(items, list):
        return items, 0
    kept = [t for t in items if tech_is_grounded(t, src)]
    return kept, len(items) - len(kept)


def _dedupe_strs(items: Any) -> tuple[Any, int]:
    """The same string named more than once in one list, kept only the first time.

    The model occasionally names a technology twice within one job's
    technologies_used — nothing upstream removes the repeat, and a renderer
    that prints one "Key Technologies/Skills" line per array entry then shows
    the same technology twice back to back.
    """
    if not isinstance(items, list) or len(items) < 2:
        return items, 0
    seen: set[str] = set()
    kept = []
    for item in items:
        key = _squash(item) if isinstance(item, str) else repr(item)
        if key and key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept, len(items) - len(kept)


def _scrub_tech_string(value: Any, src: tuple[set[str], str, str]) -> tuple[Any, int]:
    """Same filter for the comma-separated keyTechnologies string on projects."""
    if not isinstance(value, str) or not value.strip():
        return value, 0
    parts = [p.strip() for p in value.split(",") if p.strip()]
    kept = [p for p in parts if tech_is_grounded(p, src)]
    if len(kept) == len(parts):
        return value, 0
    return (", ".join(kept) or None), len(parts) - len(kept)


# Every skills bucket is a plain list of named skills, so the same rule applies:
# a skill the resume never names is the model's inference, not the candidate's.
_SKILL_LIST_FIELDS = (
    "all_skills_raw", "technical_skills", "soft_skills", "programming_languages",
    "frameworks_and_libraries", "databases", "cloud_platforms", "tools_and_platforms",
    "operating_systems", "methodologies", "domain_skills", "design_skills", "other_skills",
)


def _tidy_and_scrub(items: Any, src: tuple[set[str], str, str]) -> tuple[Any, int]:
    """One skills bucket, taken apart and then grounded.

    Order matters. Tidying first turns a whole resume line — "Containerization:
    Docker, Kubernetes, Helm · IaC: Terraform" — into the skills it lists, so
    the groundedness check judges each name on its own instead of passing the
    line whole because most of its words appear in the resume. Prose that is not
    a skill at all ("Work Authorization - US Permanent Resident…") is dropped by
    the same step.
    """
    tidied, _, dropped_prose = tidy_skill_list(items)
    kept, dropped_ungrounded = _scrub_techs(tidied, src)
    return kept, dropped_prose + dropped_ungrounded


def scrub_skills(skills: Any, src: tuple[set[str], str, str]) -> int:
    """Normalise every skills bucket and drop what the resume does not name.

    Returns how many entries were removed.
    """
    if not isinstance(skills, dict):
        return 0
    dropped = 0
    for field in _SKILL_LIST_FIELDS:
        original = skills.get(field)
        if not isinstance(original, list):
            continue
        kept, n = _tidy_and_scrub(original, src)
        skills[field] = kept
        dropped += n
    for cat in skills.get("categories") or []:
        if not isinstance(cat, dict):
            continue
        original = cat.get("skills")
        if not isinstance(original, list):
            continue
        kept, n = _tidy_and_scrub(original, src)
        cat["skills"] = kept
        dropped += n

    # The unions were computed from the buckets before any of this ran. Deriving
    # them again is the only way they stay a union of what the buckets now hold.
    derive_union_fields(skills)
    return dropped


# An explicit department label, the only evidence that a resume states one.
_DEPARTMENT_LABEL = re.compile(r"^[ \t]*(?:department|dept\.?)[ \t]*[:\-–—]", re.I | re.M)


def _url_fragment(url: str) -> str:
    u = _squash(url)
    u = re.sub(r"^https?://", "", u).removeprefix("www.")
    return u.rstrip("/")[:40]


def ground_check(merged: dict, raw_text: str) -> tuple[dict, list[str]]:
    """Remove contact values and per-job client/project names that do not
    appear anywhere in the source text. Returns (merged, warnings)."""
    warnings: list[str] = []
    squashed = _squash(raw_text)
    text_digits = _digits(raw_text)
    token_set = set(_tokens(raw_text))
    source_figures = _figures(raw_text)
    line_index = source_line_index(raw_text)
    src = source_terms(raw_text)
    blocks = source_bullet_blocks(raw_text)
    blocks_squashed = [_squash(b) for b in blocks]

    # A section the resume does not have is a section the output must not have.
    # Without a Certifications heading the model was promoting responsibility
    # lines into certifications; without an Experience heading it was building
    # work history out of the skills section. Both are inventions, and both are
    # settled here rather than hoped for in a prompt.
    if merged.get("certifications") and not has_section(raw_text, _CERTIFICATION_HEADING):
        warnings.append(
            f"Removed {len(merged['certifications'])} certification(s) — the resume has no "
            "certifications section, so these came from elsewhere in the document"
        )
        merged["certifications"] = []

    if merged.get("work_experience") and not has_section(raw_text, _EXPERIENCE_HEADING):
        warnings.append(
            f"Removed {len(merged['work_experience'])} work experience entr(y/ies) — the resume "
            "has no experience section, so these were inferred from other sections"
        )
        merged["work_experience"] = []

    pi = merged.get("personal_information")
    if isinstance(pi, dict):
        emails = [e for e in (pi.get("email") or []) if isinstance(e, str)]
        kept = [e for e in emails if _squash(e) in squashed]
        for e in set(emails) - set(kept):
            warnings.append(f"Removed email not found in resume text: {e}")
        if emails:
            pi["email"] = kept

        phones = [p for p in (pi.get("phone") or []) if isinstance(p, str)]
        kept_p = [p for p in phones if len(_digits(p)) >= 7 and _digits(p)[-10:] in text_digits]
        for p in set(phones) - set(kept_p):
            warnings.append(f"Removed phone not found in resume text: {p}")
        if phones:
            pi["phone"] = kept_p

        for field in ("linkedin_url", "github_url", "portfolio_url", "twitter_url"):
            url = pi.get(field)
            if isinstance(url, str) and url.strip():
                frag = _url_fragment(url)
                # Match on the path part too when present — bare domains
                # (linkedin.com) are trivially "grounded" and not worth checking.
                if frag and frag not in squashed:
                    warnings.append(f"Removed {field} not found in resume text: {url}")
                    pi[field] = None

    # A resume that never writes "Department:" has no department to extract.
    # The field is a magnet: the model reads a client line — "Client: Department
    # of Workforce Development" — sees the word, and files the client as the
    # department, which then prints on the formatted resume as an org unit the
    # candidate never named.
    labels_a_department = bool(_DEPARTMENT_LABEL.search(raw_text or ""))

    # Per-job sub-projects: client/project names must exist in the source.
    for job in merged.get("work_experience") or []:
        if not isinstance(job, dict):
            continue
        company = job.get("company_name") or "?"

        if job.get("department") and not labels_a_department:
            warnings.append(
                f"Removed department '{job['department']}' — the resume does not "
                f"label a department for this role ({company})"
            )
            job["department"] = None

        # Drop AI-fabricated responsibilities/achievements (invented metrics or
        # ungrounded impact sentences) before any further processing.
        rejoined, split_count = merge_split_bullets(
            job.get("responsibilities"), blocks, blocks_squashed)
        if split_count:
            job["responsibilities"] = rejoined
            warnings.append(
                f"Rejoined {split_count} detail(s) split out of their own bullet ({company})"
            )

        resp_scrubbed, resp_dropped, resp_fixed = _scrub_bullets(
            job.get("responsibilities"), token_set, source_figures, line_index)
        if resp_dropped or resp_fixed:
            job["responsibilities"] = resp_scrubbed
            if resp_dropped:
                warnings.append(f"Removed {resp_dropped} fabricated/ungrounded responsibility bullet(s) ({company})")
            if resp_fixed:
                warnings.append(f"Removed invented figures from {resp_fixed} responsibility bullet(s) ({company})")
        ach_scrubbed, ach_dropped, ach_fixed = _scrub_bullets(
            job.get("achievements"), token_set, source_figures, line_index)
        if ach_dropped or ach_fixed:
            job["achievements"] = ach_scrubbed
            if ach_dropped:
                warnings.append(f"Removed {ach_dropped} fabricated/ungrounded achievement(s) ({company})")
            if ach_fixed:
                warnings.append(f"Removed invented figures from {ach_fixed} achievement(s) ({company})")

        tech_scrubbed, tech_dropped = _scrub_techs(job.get("technologies_used"), src)
        if tech_dropped:
            job["technologies_used"] = tech_scrubbed
            warnings.append(f"Removed {tech_dropped} technology/technologies not named in the resume ({company})")

        tech_deduped, tech_dupes = _dedupe_strs(job.get("technologies_used"))
        if tech_dupes:
            job["technologies_used"] = tech_deduped
            warnings.append(
                f"Removed {tech_dupes} duplicate technology/technologies from the same "
                f"job ({company}) — printed once under \"Key Technologies/Skills\", not twice"
            )

        kept_projects = []
        for proj in job.get("projects") or []:
            if not isinstance(proj, dict):
                continue
            client = proj.get("clientName")
            if isinstance(client, str) and client.strip() and not is_grounded(client, token_set):
                warnings.append(f"Removed ungrounded client name '{client}' ({company})")
                proj["clientName"] = None
            pname = proj.get("projectName")
            if isinstance(pname, str) and pname.strip() and not is_grounded(pname, token_set):
                # Invented project heading — keep its bullets, drop the fake name.
                bullets = [b for b in (proj.get("projectResponsibilities") or []) if isinstance(b, str)]
                resp = job.setdefault("responsibilities", [])
                existing = {_squash(r) for r in resp if isinstance(r, str)}
                resp.extend(b for b in bullets if _squash(b) not in existing)
                warnings.append(f"Dropped invented project heading '{pname}' ({company}); kept its bullets")
                continue
            kt_scrubbed, kt_dropped = _scrub_tech_string(proj.get("keyTechnologies"), src)
            if kt_dropped:
                proj["keyTechnologies"] = kt_scrubbed
                warnings.append(
                    f"Removed {kt_dropped} technology/technologies not named in the resume "
                    f"({company}/{pname or 'project'})"
                )

            pr_rejoined, pr_split_count = merge_split_bullets(
                proj.get("projectResponsibilities"), blocks, blocks_squashed)
            if pr_split_count:
                proj["projectResponsibilities"] = pr_rejoined
                warnings.append(
                    f"Rejoined {pr_split_count} detail(s) split out of their own bullet "
                    f"({company}/{pname or 'project'})"
                )

            pr_scrubbed, pr_dropped, pr_fixed = _scrub_bullets(
                proj.get("projectResponsibilities"), token_set, source_figures, line_index)
            if pr_dropped or pr_fixed:
                proj["projectResponsibilities"] = pr_scrubbed
                if pr_dropped:
                    warnings.append(f"Removed {pr_dropped} fabricated project bullet(s) ({company}/{pname or 'project'})")
                if pr_fixed:
                    warnings.append(f"Removed invented figures from {pr_fixed} project bullet(s) ({company}/{pname or 'project'})")
            kept_projects.append(proj)
        if job.get("projects") is not None:
            job["projects"] = kept_projects

        # Dedupe responsibilities within the job (keeps first occurrence).
        resp = job.get("responsibilities")
        if isinstance(resp, list):
            seen: set[str] = set()
            deduped = []
            for r in resp:
                key = _squash(r) if isinstance(r, str) else repr(r)
                if key and key in seen:
                    continue
                seen.add(key)
                deduped.append(r)
            if len(deduped) != len(resp):
                warnings.append(f"Removed {len(resp) - len(deduped)} duplicate bullet(s) ({company})")
            job["responsibilities"] = deduped

    skills_dropped = scrub_skills(merged.get("skills"), src)
    if skills_dropped:
        warnings.append(f"Removed {skills_dropped} skill(s) not named in the resume")

    # Standalone projects: scrub fabricated highlights the same way.
    for proj in merged.get("projects") or []:
        if not isinstance(proj, dict):
            continue
        pt_scrubbed, pt_dropped = _scrub_techs(proj.get("technologies"), src)
        if pt_dropped:
            proj["technologies"] = pt_scrubbed
            warnings.append(
                f"Removed {pt_dropped} technology/technologies not named in the resume "
                f"({proj.get('name') or 'project'})"
            )

        hl_scrubbed, hl_dropped, hl_fixed = _scrub_bullets(
            proj.get("highlights"), token_set, source_figures, line_index)
        if hl_dropped or hl_fixed:
            proj["highlights"] = hl_scrubbed
            if hl_dropped:
                warnings.append(f"Removed {hl_dropped} fabricated project highlight(s) ({proj.get('name') or 'project'})")
            if hl_fixed:
                warnings.append(f"Removed invented figures from {hl_fixed} project highlight(s) ({proj.get('name') or 'project'})")

    return merged, warnings


# ── Section detection ───────────────────────────────────────────────────────

_HEADER_LIKE = re.compile(r"^[A-Z\s&/:\-]{3,60}$")

# Sections the tool does not produce. Their lines must be excluded from the
# coverage audit — otherwise every awards line reads as "missed", gets handed to
# the recovery pass, and comes back merged into work bullets or the summary.
# Removing a section from the schema and leaving it in the audit input puts the
# content straight back in through the side door.
_DROPPED_SECTION_HEADING = re.compile(
    r"^(?:"
    r"awards?(?:\s*(?:&|and)\s*(?:honou?rs?|recognitions?|achievements?))?"
    r"|honou?rs?(?:\s*(?:&|and)\s*awards?)?"
    r"|recognitions?|accolades?"
    r"|volunteer(?:ing)?(?:\s+(?:experience|work))?|community\s+(?:service|involvement)"
    r"|languages?(?:\s+(?:known|spoken|proficiency))?"
    r"|publications?|papers?|research\s+publications?"
    r"|(?:professional\s+)?(?:memberships?|affiliations?|associations?)"
    r"|(?:personal\s+)?interests?(?:\s*(?:&|and)\s*hobbies)?|hobbies"
    r"|extra[\s-]?curricular(?:\s+activities)?"
    r")$",
    re.I,
)

# These two are matched by CONTAINMENT, not equality, and deliberately so.
# Their absence deletes a whole section of the output, so the cost of failing to
# recognise a heading is far higher than the cost of recognising a loose one:
# "EXPERIENCE SUMMARY", "RELEVANT WORK EXPERIENCE" and "IT EXPERIENCE" are all
# real headings that an anchored pattern would miss, taking the entire work
# history down with it.
_EXPERIENCE_HEADING = re.compile(
    r"experience|employment|work\s+history|career\s+history"
    r"|professional\s+background|organi[sz]ational\s+scan",
    re.I,
)

_CERTIFICATION_HEADING = re.compile(
    r"certificat|certification|licen[cs]|credential|accredit",
    re.I,
)


def _heading_text(line: str) -> str | None:
    """The line's heading text, or None when it does not read as a heading.

    Shares its rule with section routing, but deliberately not its vocabulary.
    A missed heading here deletes a whole section of the output, so only these
    three names are recognised beyond plain capitalisation — the longer list
    routing uses would start matching headings inside body text and cut real
    content out of the result.
    """
    return heading_text(
        line,
        (_DROPPED_SECTION_HEADING, _EXPERIENCE_HEADING, _CERTIFICATION_HEADING),
    )


def has_section(raw_text: str, pattern: re.Pattern[str]) -> bool:
    """Does the resume carry a heading of this kind?"""
    return any(
        (h := _heading_text(line)) is not None and pattern.search(h)
        for line in (raw_text or "").split("\n")
    )


def strip_dropped_sections(raw_text: str) -> str:
    """Remove awards / volunteer / languages / publications / memberships /
    interests blocks, from their heading to the next heading."""
    kept: list[str] = []
    skipping = False
    for line in (raw_text or "").split("\n"):
        heading = _heading_text(line)
        if heading is not None:
            skipping = bool(_DROPPED_SECTION_HEADING.search(heading))
            if skipping:
                continue
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


# ── Coverage ────────────────────────────────────────────────────────────────


def coverage_report(merged: dict, raw_text: str) -> tuple[float, list[str]]:
    """Which significant resume lines never made it into the extraction?

    Returns (coverage_percent, missed_lines). A line counts as covered when
    ≥65% of its significant tokens appear somewhere in the extracted JSON.
    """
    blob_tokens = set(_tokens(json.dumps(merged, ensure_ascii=False, default=str)))

    significant: list[tuple[str, list[str]]] = []
    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not stripped or _HEADER_LIKE.match(stripped):
            continue
        toks = _tokens(stripped)
        if len(toks) < 4:
            continue  # dates, headings, single names — not auditable lines
        significant.append((stripped, toks))

    if not significant:
        return 100.0, []

    missed = []
    covered = 0
    for line, toks in significant:
        frac = sum(1 for t in toks if t in blob_tokens) / len(toks)
        if frac >= 0.65:
            covered += 1
        else:
            missed.append(line)
    return round(100.0 * covered / len(significant), 1), missed


# ── Additive merge of recovered content ─────────────────────────────────────

# Legal suffixes carry no identifying signal, and the recovery pass and the
# extraction pass rarely agree on whether to include them.
_COMPANY_NOISE = {
    "inc", "llc", "ltd", "limited", "corp", "corporation", "company", "and",
    "the", "plc", "gmbh", "pvt", "private", "group", "holdings", "solutions",
    "services", "technologies", "systems", "international",
}


def _company_tokens(name: str) -> set[str]:
    toks = set(_tokens(name))
    stripped = toks - _COMPANY_NOISE
    # An employer genuinely called "Systems Limited" keeps its words rather
    # than reducing to nothing.
    return stripped or toks


def _best_job_match(company: str, jobs: list[dict]) -> dict | None:
    """The extracted job a recovered bullet belongs to, or None.

    Matching used to require one company name to be a strict subset of the
    other, so "Acme Corp" and "Acme Corporation" failed to match and the bullet
    was dropped. Scoring against the shorter name lets ordinary naming
    differences through while still refusing an unrelated employer.
    """
    comp_toks = _company_tokens(company)
    if not comp_toks:
        return None
    best: dict | None = None
    best_score = 0.0
    for job in jobs:
        job_toks = _company_tokens(str(job.get("company_name") or ""))
        overlap = len(comp_toks & job_toks)
        if not overlap:
            continue
        score = overlap / min(len(comp_toks), len(job_toks))
        if score > best_score:
            best, best_score = job, score
    return best if best_score >= 0.5 else None


def merge_recovered(merged: dict, recovered: dict) -> dict[str, int]:
    """Merge the recovery pass output into `merged` ADDITIVELY.
    Returns counts of what was added, for the audit report."""
    added = {
        "work_bullets": 0, "education": 0, "certifications": 0,
        "projects": 0, "skills": 0, "summary_lines": 0,
        "unplaced_bullets": 0,
    }
    if not isinstance(recovered, dict):
        return added

    # Missed work bullets → append to the job with the matching company.
    jobs = [j for j in (merged.get("work_experience") or []) if isinstance(j, dict)]
    for item in recovered.get("work_bullets") or []:
        if not isinstance(item, dict):
            continue
        bullets = [b.strip() for b in (item.get("bullets") or []) if isinstance(b, str) and b.strip()]
        if not bullets:
            continue

        target = _best_job_match(str(item.get("company_name") or ""), jobs)
        # The recovery pass names the company it read off the resume, which need
        # not be spelled the way the extraction spelled it. When there is only
        # one job on the resume the bullet cannot belong anywhere else, so a
        # naming mismatch should not cost us the line.
        if target is None and len(jobs) == 1:
            target = jobs[0]
        if target is None:
            # Counted, not silently dropped — an unplaceable bullet is content
            # the resume contains and the output does not, and the reviewer is
            # the one who needs to know.
            added["unplaced_bullets"] += len(bullets)
            continue

        resp = target.setdefault("responsibilities", [])
        existing = {_squash(r) for r in resp if isinstance(r, str)}
        for b in bullets:
            if _squash(b) not in existing:
                resp.append(b)
                existing.add(_squash(b))
                added["work_bullets"] += 1

    def _append_new(key: str, items: Any, name_field: str, counter_key: str) -> None:
        if not isinstance(items, list):
            return
        existing_list = merged.setdefault(key, [])
        if not isinstance(existing_list, list):
            return
        existing_names = {_squash(str(e.get(name_field) or "")) for e in existing_list if isinstance(e, dict)}
        for it in items:
            if not isinstance(it, dict):
                continue
            name = _squash(str(it.get(name_field) or ""))
            if name and name not in existing_names:
                existing_list.append(it)
                existing_names.add(name)
                added[counter_key] += 1

    _append_new("education", recovered.get("education"), "institution_name", "education")
    _append_new("certifications", recovered.get("certifications"), "name", "certifications")
    _append_new("projects", recovered.get("projects"), "name", "projects")

    # Missed skills → flat append into the skills inventory.
    #
    # The recovery pass returns uncovered resume LINES, so what arrives here is
    # often a whole category row — "App Servers: WebSphere, WebLogic, Tomcat,
    # JBoss" — or a sentence that merely sat in the skills block, like a work
    # authorization statement. Appending those verbatim is what put paragraphs
    # into other_skills. Taking them apart first turns the row into the four
    # skills it names and drops the sentence.
    skills = merged.get("skills")
    if isinstance(skills, dict):
        known = set(_tokens(json.dumps(skills, ensure_ascii=False, default=str)))
        recovered_skills, _, _ = tidy_skill_list(
            [s for s in (recovered.get("skills") or []) if isinstance(s, str)]
        )
        for s in recovered_skills:
            if set(_tokens(s)) <= known:
                continue
            skills.setdefault("other_skills", []).append(s)
            added["skills"] += 1
        if added["skills"]:
            derive_union_fields(skills)

    # Missed summary lines → append to the summary already extracted.
    #
    # This used to fire only when there was NO summary yet, which made a dropped
    # summary bullet unrecoverable: the resume has a summary, one of its bullets
    # goes missing, recovery returns that bullet, and the merge threw it away
    # because the field was non-empty. Every other bucket here is additive; this
    # one now is too. Overwriting is still not an option — the recovery pass
    # returns the missed fragment, not the whole section.
    recovered_summary = recovered.get("professional_summary")
    if isinstance(recovered_summary, str) and recovered_summary.strip():
        existing = str(merged.get("professional_summary") or "").strip()
        fresh = [ln.strip() for ln in recovered_summary.split("\n") if ln.strip()]
        if not existing:
            merged["professional_summary"] = "\n".join(fresh)
            added["summary_lines"] += len(fresh)
        else:
            # Substring match on the squashed text, so a bullet already sitting
            # inside a single-paragraph summary is not appended a second time.
            squashed_existing = _squash(existing)
            keeps_bullets = existing.lstrip().startswith("•")
            new_lines = [ln for ln in fresh if _squash(ln) not in squashed_existing]
            if new_lines:
                merged["professional_summary"] = "\n".join(
                    [existing] + [
                        f"• {ln.lstrip('•').strip()}" if keeps_bullets else ln
                        for ln in new_lines
                    ]
                )
                added["summary_lines"] += len(new_lines)

    return added


# ── The agent ───────────────────────────────────────────────────────────────

RECOVERY_SYSTEM = """You are an extraction auditor. A resume was parsed into JSON, but the lines listed under === MISSED LINES === were NOT captured.

Classify ONLY the missed content into this JSON (use [] / null when a bucket has nothing):
{
  "recovered": {
    "work_bullets": [{"company_name": "<existing company this bullet belongs to>", "bullets": ["verbatim bullet", ...]}],
    "education": [{"institution_name": "", "degree": null, "degree_type": null, "field_of_study": null, "end_date": null, "location": null}],
    "certifications": [{"name": "", "issuing_organization": null, "issue_date": null}],
    "projects": [{"name": "", "description": null, "technologies": [], "highlights": []}],
    "skills": [],
    "professional_summary": null
  }
}

Rules:
- Copy text VERBATIM from the resume. NEVER invent or rephrase anything.
- Only include content from the missed lines (use the full resume text just for context, e.g. to find which company a bullet belongs to).
- Ignore missed lines that are pure layout noise (page numbers, decoration).
- Return ONLY valid JSON.
"""


class CompletenessAuditorAgent(BaseAgent):

    def __init__(self):
        super().__init__("CompletenessAuditorAgent")

    async def run(self, merged: dict, raw_text: str, *, allow_recovery: bool = True) -> dict:
        """Ground, measure, and — when the run can still afford it — recover.

        `allow_recovery` gates only the model call at the end. Everything before
        it is deterministic and runs regardless: the groundedness pass is what
        strips invented metrics and technologies the resume never named, and a
        fabricated figure on a submitted resume is not a cost worth trading for
        a faster response.
        """
        report: dict[str, Any] = {"warnings": [], "recovered": {}}
        try:
            merged, warnings = ground_check(merged, raw_text)
            report["warnings"] = warnings

            # Grounding checks read the whole document; the coverage audit and
            # the recovery pass must not, or they spend their effort putting the
            # dropped sections back.
            audit_text = strip_dropped_sections(raw_text)
            coverage, missed = coverage_report(merged, audit_text)
            report["coverage_percent"] = coverage
            report["missed_line_count"] = len(missed)

            recovery_enabled = (
                allow_recovery
                and os.getenv("ENABLE_AUDIT_RECOVERY", "true").lower() in ("1", "true", "yes")
            )
            if missed and not allow_recovery:
                report["warnings"].append(
                    f"{len(missed)} line(s) were not matched to a field, and there was no "
                    "time left to sort them — check the resume against the original"
                )
            # Recovery used to require two or more missed lines, which meant the
            # single-line gap — the most common one, and the one a reviewer
            # actually reads on a 99.3% report — was never even attempted. One
            # dropped line is exactly the case worth an extra call.
            if missed and recovery_enabled:
                try:
                    recovered = await self._recover(audit_text, missed[:40])
                    report["recovered"] = merge_recovered(merged, recovered)
                    unplaced = report["recovered"].get("unplaced_bullets", 0)
                    if unplaced:
                        report["warnings"].append(
                            f"{unplaced} recovered bullet(s) could not be matched to a job "
                            "and were not added — add them by hand"
                        )
                    coverage, missed = coverage_report(merged, audit_text)
                    report["coverage_percent"] = coverage
                    report["missed_line_count"] = len(missed)
                except Exception as exc:
                    logger.warning("[CompletenessAuditorAgent] Recovery pass failed: %s", exc)
                    report["warnings"].append("Recovery pass failed — some content may be missing")

            report["missed_lines"] = missed[:15]
            if missed:
                logger.warning(
                    "[CompletenessAuditorAgent] %d line(s) not captured (coverage %.1f%%)",
                    len(missed), coverage,
                )
        except Exception as exc:
            logger.warning("[CompletenessAuditorAgent] Audit failed: %s", exc)
            report["warnings"].append(f"Audit stage failed: {exc}")

        merged["_audit"] = report
        return merged

    async def _recover(self, raw_text: str, missed_lines: list[str]) -> dict:
        user_msg = (
            "=== RESUME ===\n"
            f"{raw_text}\n"
            "=== END ===\n\n"
            "=== MISSED LINES ===\n"
            + "\n".join(f"- {ln}" for ln in missed_lines)
            + "\n=== END MISSED LINES ===\n\n"
            "Classify the missed content. Return JSON."
        )
        result = await self._call_llm_json(
            RECOVERY_SYSTEM, user_msg,
            max_tokens=output_budget("\n".join(missed_lines), floor=4096, ceiling=8192),
            section="Content recovery",
        )
        if isinstance(result, dict):
            return result.get("recovered", result)
        return {}
