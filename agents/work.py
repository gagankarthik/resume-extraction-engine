"""
WorkExperienceAgent — extracts work experience with EXACT bullet point counts.
The bullet count per job is determined programmatically by StructureAgent, never by LLM estimation.
"""
from __future__ import annotations

import asyncio
import logging

from .base import BaseAgent, output_budget
from .deadline import get_deadline

logger = logging.getLogger(__name__)

# A retry pass re-runs the failed jobs from scratch. Worth starting only with
# enough budget left to actually finish one.
_RETRY_BUDGET_SECONDS = 25.0

WORK_SYSTEM_BASE = """You are a work experience extraction specialist. Extract the work experience for ONE specific job entry from the resume.

CRITICAL BULLET COUNT RULE:
{bullet_instruction}

Extract into this JSON shape:
{{
  "company_name": "exact company name",
  "job_title": "exact job title",
  "start_date": "as written",
  "end_date": "as written or 'Present'",
  "is_current": false,
  "location": "city/state or null",
  "department": "department name or null",
  "employment_type": "Full-time/Contract/etc. or null",
  "duration": "duration string or null",
  "responsibilities": ["bullet 1", "bullet 2", ...],
  "achievements": ["measurable outcome 1", ...],
  "technologies_used": ["tech1", "tech2", ...],
  "description": "verbatim company/role description line from the resume, or null",
  "projects": [
    {{
      "projectName": "project name",
      "clientName": "client or null",
      "projectLocation": "location or null",
      "keyTechnologies": "comma-separated list or null",
      "projectResponsibilities": ["bullet 1", ...]
    }}
  ]
}}

Rules:
- NEVER fabricate content. Every value you return MUST appear in the source text. If a value is not in the segment, use null / [] — do not guess, infer, or fill in plausible-sounding data.
- NEVER invent metrics, percentages, dollar figures, or counts. Only keep a number (e.g. "40%", "$2M", "reduced defects by 35%", "3 platform launches") if that EXACT number is written in the source segment. Do not add, estimate, round, or embellish numbers, and do not append quantified impact ("by 40%", "improving efficiency", "increasing velocity") that the source does not literally state.
- NEVER add generic AI-style impact sentences that are not in the source (e.g. "Improved release predictability by 40%", "Reduced production defects by 35%", "Accelerated time-to-market", "Increased sprint velocity by 20%", "Delivered measurable cost optimization through license renegotiation and capacity right-sizing"). If the resume does not contain the sentence, it MUST NOT appear in the output.
- NEVER append a result clause to a real bullet. Copying "Standardized the onboarding checklist" and returning "Standardized the onboarding checklist, reducing onboarding time 40%" is fabrication even though the first half is genuine — the candidate never claimed the 40%. Clauses of the form "reducing X N%", "increasing X N%", "cutting X N%", "improving X N%", "over three quarters" are forbidden unless that exact clause is written in the segment. A bullet ends where the resume ends it.
- Copy every responsibility VERBATIM — do not paraphrase, summarize, or merge. Each responsibility must be the COMPLETE sentence/bullet from the source — never truncate or cut a sentence partway.
- ONE source bullet = ONE array item. Return EXACTLY as many items as the source has, no more. Never split a bullet into several items, never invent an extra item, and never pad a short list. If the job has a single responsibility, responsibilities[] has exactly one entry.
- A trailing detail belongs to the bullet it modifies. Result and metric clauses — "Achieved 80% accuracy", "with 99.9% uptime", "resulting in a 2-day turnaround" — are PART of the responsibility they follow, even when the PDF wrapped them onto their own line. Keep them inside that entry. NEVER emit a metric or result fragment as a responsibility of its own: the candidate wrote one bullet, and the output must show one bullet.
- start_date / end_date: copy EXACTLY as written, INCLUDING the month when one is present (e.g. "Jan 2020", not "2020").
  • ONE exception: if the candidate still holds the role — the resume writes "Present", "Current", "Till Date", "To Date", "Now" or leaves the end open — set end_date to "Till Date" and is_current to true.
- location: extract the job's city/state whenever it appears anywhere in the job header lines — do not leave it null if a location is written.
- department: leave it null unless the resume LITERALLY labels one ("Department: X", "Dept: X"). A client, an agency, or a government body named in the job is the CLIENT or the EMPLOYER — never the department, even when its own name contains the word "Department".
- EMPLOYER vs CLIENT. Consulting resumes name both ("Employer - IBM Global Services" on one line, "Client: Merck, NJ, USA" on the next). company_name is the EMPLOYER when one is written, and the client MUST still be captured — as a projects[] entry whose clientName is that client, carrying this block's bullets. NEVER drop the client, and NEVER put the client's name in location: location is a city/state/country only. If the resume names only a client and no employer, the client is company_name.
- Strip inline labels from the values you copy. The resume writes "Client: Acme Insurance" or "Project: Claims Modernization"; the value is "Acme Insurance" and "Claims Modernization". Never carry the "Client:" / "Project:" / "Role:" prefix into the field.
- Strip leading bullet glyphs (•, ●, ▪, ▸, ‣, ○, ◦, ►, -, *, –, —) from each entry.
- Bullets may be marked by ANY of these glyphs, by numbered lists (1., 1)), or by no glyph at all (prose).
- If the job uses PROSE paragraphs (no explicit bullets): each PARAGRAPH becomes one item in responsibilities[], copied verbatim. Do NOT sentence-split a paragraph into several items, and do NOT put job duties into description.
- Treat ANY narrative duty/action text in the segment as bullet content — even if it isn't formatted as a bulleted list. Past-tense action verbs (Designed, Led, Built, Managed, Migrated, Implemented, …) are duty signals; capture each as its own responsibility.
- NEVER leave responsibilities[] empty if the segment contains ANY duty/narrative text — every such job must have at least one entry. Only leave it empty if the segment is truly metadata-only (company/date/title/location/tech list with NO action sentences).
- description: NEVER write one. It is null unless the resume itself prints a company/role description line for this job (e.g. "Healthcare insurer in California"), in which case copy that line VERBATIM. Do not summarise, characterise, or compose a sentence of your own. NEVER put duty bullets, action verbs, or summary sentences into description — those belong in responsibilities[].
- projects[] rules (BE STRICT):
  • Use projects[] ONLY when the source resume EXPLICITLY labels sub-projects with their own headings (e.g. "Project 1: <Name>" or a discrete project name on its own line followed by its own bullet list).
  • A consulting role served for SEVERAL CLIENTS is exactly this case: each "Client: <Name>" (or "Engagement:", "End Client:") block, with the bullets that follow it, is ONE entry in projects[] — clientName is that client, projectName is the engagement/project name when one is written and null when it is not, and projectResponsibilities[] holds that block's bullets and no other block's. Keep the clients in the order the resume lists them, and never fold two clients into one entry or split one client across two.
  • Bullets that belong to a client/project block go in that block ONLY. Do not also repeat them in the job's flat responsibilities[] — that prints the same line twice.
  • When the segment DOES contain explicitly labeled sub-projects, you MUST extract EVERY one of them — each with its projectName, clientName (if written), projectLocation (if written), keyTechnologies (if written), and its COMPLETE bullet list copied verbatim into projectResponsibilities[]. NEVER return a project with an empty projectResponsibilities[] when bullets exist under it, and NEVER skip a labeled project.
  • clientName: copy CHARACTER-FOR-CHARACTER from the segment (look for "Client:", "Customer:", "End Client:" labels or the client named in the project heading). If no client is written for THIS project, return null — NEVER guess, NEVER substitute a similar-sounding company, and NEVER reuse a client from a different project or job.
  • DO NOT invent project names by grouping responsibilities by topic. A long flat list of bullets is NOT a multi-project structure — it's a single role's responsibilities. Put them all in responsibilities[].
  • If you find yourself synthesizing project names from bullet content (e.g. "Lakehouse Architecture Design", "BigID Implementation"), STOP — those are responsibility topics, not labeled sub-projects. Put them as flat entries in responsibilities[].
- achievements[] should contain ONLY items with measurable results (%, $, headcount, time saved, etc.).
- technologies_used[] = ONLY the tools/languages/platforms whose names are LITERALLY WRITTEN in this job's text, copied exactly as spelled there.
  • NEVER infer a technology from a duty. "security and access management" does NOT mean IAM. "containers" does NOT mean Docker or Kubernetes. "cloud" does NOT mean AWS. "reporting" does NOT mean Power BI. If the name is not printed in the segment, it does not go in the array.
  • NEVER add a related, implied, or typical technology, and never expand an abbreviation into a product name.
  • If the job names no technologies at all, return [] — an empty array is the correct answer, not a list you assembled from the responsibilities.
- Return ONLY valid JSON.
"""

WORK_SYSTEM_FULL_FALLBACK = """You are a work experience extraction specialist. Extract ALL work experience entries from the resume. This is a long-career resume — include EVERY job, even old ones from 15-25 years ago.

CRITICAL RULES:
1. Include EVERY job entry — do not skip any role, even old ones. Every job must have its job_title when one is written.
2. NEVER skip or add bullet points. Extract EXACTLY the bullets that exist. NEVER fabricate content that is not in the resume.
3. Copy all responsibilities VERBATIM — do not paraphrase or merge. Never truncate a sentence partway.
3b. Copy dates EXACTLY as written, INCLUDING months ("Jan 2020", not "2020"). Extract each job's location when written. For a role the candidate still holds, set end_date to "Till Date" and is_current to true.
3c. NEVER append a metric or result clause the resume does not literally write ("reducing onboarding time 40%", "increasing delivery velocity 15%"). A bullet ends where the resume ends it.
3d. department stays null unless the resume labels one ("Department: X"). A client or agency named in the job is not the department. Strip "Client:" / "Project:" prefixes from the values you copy.
4. Bullets may be marked by ANY glyph (•, ●, ▪, ▸, ‣, ○, ◦, ►, -, *, –, —), numbered (1., 1)), or no glyph at all (prose). Treat them all as bullets.
5. If the job uses PROSE paragraphs (no bullet points): each PARAGRAPH becomes one item in responsibilities[], copied verbatim — never sentence-split a paragraph into several items. Do NOT use the description field for job duties.
5b. ONE source bullet = ONE array item. Return exactly as many items as the source has — never split, never pad. A job with a single responsibility gets exactly one entry.
6. NEVER leave responsibilities[] empty if the job segment has ANY duty/narrative text — every such job must have at least one entry. Only leave it empty if the segment is truly metadata-only (company/date/title/location/tech list).
7. If a person worked on sub-projects or for several named clients under one company, structure them in projects[] — one entry per labeled client/project, with clientName set and that block's bullets in projectResponsibilities[]. Do not repeat those bullets in responsibilities[].

Return a JSON object: {{ "work_experience": [ <job objects> ] }}

Each job object:
{{
  "company_name": str, "job_title": str, "start_date": str, "end_date": str,
  "is_current": bool, "location": str|null, "department": str|null,
  "employment_type": str|null, "duration": str|null,
  "responsibilities": [str], "achievements": [str], "technologies_used": [str],
  "description": str|null, "projects": []
}}

Return ONLY valid JSON.
"""


class WorkExperienceAgent(BaseAgent):

    def __init__(self):
        super().__init__("WorkExperienceAgent")

    async def run(self, text: str, structure: dict) -> list[dict]:
        """
        Extract work experience using per-job extraction with exact bullet counts.
        Falls back to full-document extraction if structure is empty.
        Preserves document order; retries any job that failed on the first pass.
        """
        jobs_meta = structure.get("jobs", [])

        if not jobs_meta:
            return await self._extract_full_document(text)

        # First pass: all jobs in parallel
        first_pass = await asyncio.gather(
            *[self._extract_single_job(text, meta) for meta in jobs_meta],
            return_exceptions=True,
        )

        # Collect results and identify failures
        results: list[dict | None] = [None] * len(jobs_meta)
        retry_indices: list[int] = []

        for i, res in enumerate(first_pass):
            if isinstance(res, Exception):
                logger.warning(
                    "[WorkExperienceAgent] Job %d (%s) failed — will retry: %s",
                    i, jobs_meta[i].get("company", "?"), res,
                )
                retry_indices.append(i)
            else:
                results[i] = res

        # Retry pass for any failures
        if retry_indices and not get_deadline().allows(_RETRY_BUDGET_SECONDS):
            logger.warning(
                "[WorkExperienceAgent] %d job(s) failed but there is no budget to retry them",
                len(retry_indices),
            )
        elif retry_indices:
            logger.info("[WorkExperienceAgent] Retrying %d failed job(s)", len(retry_indices))
            retry_pass = await asyncio.gather(
                *[self._extract_single_job(text, jobs_meta[i]) for i in retry_indices],
                return_exceptions=True,
            )
            for idx, res in zip(retry_indices, retry_pass, strict=True):
                if isinstance(res, Exception):
                    logger.error(
                        "[WorkExperienceAgent] Job %d (%s) permanently failed after retry: %s",
                        idx, jobs_meta[idx].get("company", "?"), res,
                    )
                else:
                    results[idx] = res

        # Fill any still-failed slot with a metadata-only stub built from the
        # structure map. This preserves POSITIONAL alignment between
        # work_experience and structure["jobs"] — the ValidatorAgent zips the
        # two together by index, so a dropped slot would shift every later job
        # against the wrong bullet-count meta. It also keeps the failed job
        # visible (with company/title/dates) rather than silently vanishing.
        for i, res in enumerate(results):
            if res is None:
                results[i] = self._stub_from_meta(jobs_meta[i])

        return results

    @staticmethod
    def _stub_from_meta(meta: dict) -> dict:
        """Minimal job object from structure metadata when extraction fails."""
        end = meta.get("end_date", "") or ""
        return {
            "company_name": meta.get("company", "") or "",
            "job_title": meta.get("title", "") or "",
            "start_date": meta.get("start_date", "") or "",
            "end_date": end,
            "is_current": end.strip().lower() in ("present", "current", "now"),
            "location": meta.get("location"),
            "department": None,
            "employment_type": None,
            "duration": None,
            "responsibilities": [],
            "achievements": [],
            "technologies_used": [],
            "description": None,
            "projects": [],
        }

    # ------------------------------------------------------------------ #
    # Per-job extraction
    # ------------------------------------------------------------------ #

    async def _extract_single_job(self, full_text: str, meta: dict) -> dict:
        company = meta.get("company", "Unknown Company")
        title = meta.get("title", "Unknown Title")
        bullet_count = meta.get("bullet_count", 0)
        segment = meta.get("segment", "").strip()
        has_projects = meta.get("has_sub_projects", False)

        # Build bullet instruction
        if bullet_count > 0:
            bullet_instruction = (
                f"This job segment for {company} contains EXACTLY {bullet_count} bullet points. "
                f"You MUST extract all {bullet_count} bullets into responsibilities[] (or distribute them "
                f"between responsibilities[] and projects[].projectResponsibilities if sub-projects exist). "
                f"DO NOT add bullets that don't exist. DO NOT skip or merge any bullet."
            )
        else:
            bullet_instruction = (
                f"Extract ALL responsibilities for {company} into the responsibilities[] array. "
                "If the job uses explicit bullet points (•, -, *, numbers): copy each bullet verbatim as a separate array item. "
                "If the job uses PROSE paragraphs instead of bullets: copy each paragraph verbatim as ONE array item — never sentence-split it. "
                "Return exactly as many items as the source has: never split one bullet into several and never add an item the source does not have. "
                "Do NOT put job duties into the description field. "
                "NEVER leave responsibilities[] empty — every job must have at least one entry."
            )

        if has_projects:
            bullet_instruction += (
                " This job has a consulting sub-project structure. Distribute bullets across "
                "projects[].projectResponsibilities[] instead of a flat responsibilities[]."
            )

        system = WORK_SYSTEM_BASE.format(bullet_instruction=bullet_instruction)

        # Use the job segment text if available, otherwise full document
        context_text = segment if segment else full_text
        user_msg = (
            f"Extract work experience for: {company} | {title}\n"
            f"Date range: {meta.get('start_date', '')} – {meta.get('end_date', '')}\n\n"
            "=== TEXT SEGMENT ===\n"
            f"{context_text}\n"
            "=== END SEGMENT ===\n\n"
            "Return ONLY the JSON for this single job entry."
        )

        # Sized to the segment rather than fixed at 6144. A long consulting role
        # with thirty bullets overran that ceiling, and the recovery for
        # overrunning it is to generate the whole job again with twice the room —
        # so the slowest jobs on the resume were the ones being done twice.
        result = await self._call_llm_json(
            system, user_msg, max_tokens=output_budget(context_text, floor=6144),
            section="Work experience",
        )

        # Handle bare JSON array (LLM skipped the wrapper object)
        if isinstance(result, list):
            return result[0] if result else {}

        # Handle both wrapped and unwrapped responses
        if "work_experience" in result and isinstance(result["work_experience"], list):
            result = result["work_experience"][0] if result["work_experience"] else {}
        elif "company_name" not in result:
            # LLM returned a dict with a single job key — look for it
            for v in result.values():
                if isinstance(v, dict) and "company_name" in v:
                    result = v
                    break

        return result

    # ------------------------------------------------------------------ #
    # Full-document fallback
    # ------------------------------------------------------------------ #

    async def _extract_full_document(self, text: str) -> list[dict]:
        logger.info("[WorkExperienceAgent] No structure map — falling back to full-document extraction")
        user_msg = f"=== RESUME TEXT ===\n{text}\n=== END ===\n\nExtract all work experience. Return only JSON."
        result = await self._call_llm_json(
            WORK_SYSTEM_FULL_FALLBACK, user_msg,
            max_tokens=output_budget(text, floor=8192, ceiling=24576),
            section="Work experience",
        )
        if isinstance(result, list):
            return result
        return result.get("work_experience", [])
