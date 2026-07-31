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

from .base import BaseAgent

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


def _scrub_bullets(items: Any, token_set: set[str]) -> tuple[list, int]:
    """Return (kept_items, dropped_count) for a responsibility/achievement list."""
    if not isinstance(items, list):
        return items, 0
    kept = [b for b in items if not _bullet_is_fabricated(b, token_set)]
    return kept, len(items) - len(kept)


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

    # Per-job sub-projects: client/project names must exist in the source.
    for job in merged.get("work_experience") or []:
        if not isinstance(job, dict):
            continue
        company = job.get("company_name") or "?"

        # Drop AI-fabricated responsibilities/achievements (invented metrics or
        # ungrounded impact sentences) before any further processing.
        resp_scrubbed, resp_dropped = _scrub_bullets(job.get("responsibilities"), token_set)
        if resp_dropped:
            job["responsibilities"] = resp_scrubbed
            warnings.append(f"Removed {resp_dropped} fabricated/ungrounded responsibility bullet(s) ({company})")
        ach_scrubbed, ach_dropped = _scrub_bullets(job.get("achievements"), token_set)
        if ach_dropped:
            job["achievements"] = ach_scrubbed
            warnings.append(f"Removed {ach_dropped} fabricated/ungrounded achievement(s) ({company})")

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
            pr_scrubbed, pr_dropped = _scrub_bullets(proj.get("projectResponsibilities"), token_set)
            if pr_dropped:
                proj["projectResponsibilities"] = pr_scrubbed
                warnings.append(f"Removed {pr_dropped} fabricated project bullet(s) ({company}/{pname or 'project'})")
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

    # Standalone projects: scrub fabricated highlights the same way.
    for proj in merged.get("projects") or []:
        if not isinstance(proj, dict):
            continue
        hl_scrubbed, hl_dropped = _scrub_bullets(proj.get("highlights"), token_set)
        if hl_dropped:
            proj["highlights"] = hl_scrubbed
            warnings.append(f"Removed {hl_dropped} fabricated project highlight(s) ({proj.get('name') or 'project'})")

    return merged, warnings


# ── Coverage ────────────────────────────────────────────────────────────────

_HEADER_LIKE = re.compile(r"^[A-Z\s&/:\-]{3,60}$")


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

def merge_recovered(merged: dict, recovered: dict) -> dict[str, int]:
    """Merge the recovery pass output into `merged` ADDITIVELY.
    Returns counts of what was added, for the audit report."""
    added = {"work_bullets": 0, "education": 0, "certifications": 0, "projects": 0, "skills": 0}
    if not isinstance(recovered, dict):
        return added

    # Missed work bullets → append to the job with the matching company.
    jobs = merged.get("work_experience") or []
    for item in recovered.get("work_bullets") or []:
        if not isinstance(item, dict):
            continue
        comp_toks = set(_tokens(str(item.get("company_name") or "")))
        target = None
        for job in jobs:
            if not isinstance(job, dict):
                continue
            jt = set(_tokens(str(job.get("company_name") or "")))
            if comp_toks and jt and (comp_toks <= jt or jt <= comp_toks):
                target = job
                break
        if target is None:
            continue
        resp = target.setdefault("responsibilities", [])
        existing = {_squash(r) for r in resp if isinstance(r, str)}
        for b in item.get("bullets") or []:
            if isinstance(b, str) and b.strip() and _squash(b) not in existing:
                resp.append(b.strip())
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
    skills = merged.get("skills")
    if isinstance(skills, dict):
        known = set(_tokens(json.dumps(skills, ensure_ascii=False, default=str)))
        for s in recovered.get("skills") or []:
            if isinstance(s, str) and s.strip() and not set(_tokens(s)) <= known:
                skills.setdefault("other_skills", []).append(s.strip())
                skills.setdefault("all_skills_raw", []).append(s.strip())
                added["skills"] += 1

    if recovered.get("professional_summary") and not merged.get("professional_summary"):
        merged["professional_summary"] = recovered["professional_summary"]

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

    async def run(self, merged: dict, raw_text: str) -> dict:
        report: dict[str, Any] = {"warnings": [], "recovered": {}}
        try:
            merged, warnings = ground_check(merged, raw_text)
            report["warnings"] = warnings

            coverage, missed = coverage_report(merged, raw_text)
            report["coverage_percent"] = coverage
            report["missed_line_count"] = len(missed)

            recovery_enabled = os.getenv("ENABLE_AUDIT_RECOVERY", "true").lower() in ("1", "true", "yes")
            if missed and recovery_enabled and len(missed) >= 2:
                try:
                    recovered = await self._recover(raw_text, missed[:40])
                    report["recovered"] = merge_recovered(merged, recovered)
                    coverage, missed = coverage_report(merged, raw_text)
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
            RECOVERY_SYSTEM, user_msg, max_tokens=6144,
            section="Content recovery",
        )
        if isinstance(result, dict):
            return result.get("recovered", result)
        return {}
