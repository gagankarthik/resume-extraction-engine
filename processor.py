"""
LLM processor — OpenAI (default) or Anthropic, switchable via MODEL_PROVIDER env var.
The provider client is a module-level singleton so the connection pool is reused
across all concurrent requests (important for multi-client production use).
"""
import os
import json
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

from validator import validate_resume_json

load_dotenv()

# ------------------------------------------------------------------ #
# Singleton clients — created once, reused for every request
# ------------------------------------------------------------------ #
_openai_client = None
_anthropic_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        _anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _anthropic_client


# ------------------------------------------------------------------ #
# Extraction system prompt
# ------------------------------------------------------------------ #
SYSTEM_PROMPT = """You are an expert resume parser. Extract EVERY piece of information from the resume below and return it as a single valid JSON object.

STRICT RULES — follow all of them:
1. Extract EVERY item — never skip a bullet point, date, number, metric, technology, or section.
2. Copy descriptions, responsibilities, and achievements VERBATIM — do not paraphrase.
3. Preserve ALL numbers and metrics exactly (e.g. "grew revenue by 43%", "led team of 12 engineers").
4. Never invent or infer data not explicitly written in the resume.
5. For missing scalar fields → null. For missing arrays → [].
6. Scan headers, footers, sidebars, and columns for contact information.
7. Collect every skill mentioned ANYWHERE, including inside job descriptions.
8. Preserve dates exactly as written (e.g. "Jan 2020", "2020–Present", "Current", "Q3 2018").
9. Output ONLY valid JSON — no markdown, no code fences, no explanation text.

REQUIRED JSON STRUCTURE:

{
  "personal_information": {
    "full_name": null,
    "first_name": null,
    "last_name": null,
    "email": [],
    "phone": [],
    "address": {
      "full_address": null,
      "street": null,
      "city": null,
      "state": null,
      "country": null,
      "zip_code": null
    },
    "linkedin_url": null,
    "github_url": null,
    "portfolio_url": null,
    "twitter_url": null,
    "other_urls": [],
    "date_of_birth": null,
    "nationality": null,
    "gender": null,
    "marital_status": null,
    "profile_headline": null
  },
  "professional_summary": null,
  "objective": null,
  "work_experience": [
    {
      "company_name": "",
      "job_title": "",
      "employment_type": null,
      "start_date": "",
      "end_date": "",
      "is_current": false,
      "duration": null,
      "location": null,
      "remote": null,
      "department": null,
      "reporting_to": null,
      "team_size": null,
      "responsibilities": [],
      "achievements": [],
      "technologies_used": [],
      "description": null
    }
  ],
  "education": [
    {
      "institution_name": "",
      "degree": null,
      "degree_type": null,
      "field_of_study": null,
      "major": null,
      "minor": null,
      "start_date": null,
      "end_date": null,
      "is_current": false,
      "gpa": null,
      "percentage": null,
      "grade": null,
      "honors": [],
      "relevant_coursework": [],
      "thesis_title": null,
      "dissertation": null,
      "location": null,
      "activities": [],
      "description": null
    }
  ],
  "skills": {
    "all_skills_raw": [],
    "technical_skills": [],
    "soft_skills": [],
    "programming_languages": [],
    "frameworks_and_libraries": [],
    "databases": [],
    "cloud_platforms": [],
    "tools_and_platforms": [],
    "operating_systems": [],
    "methodologies": [],
    "domain_skills": [],
    "design_skills": [],
    "languages_spoken": [],
    "other_skills": []
  },
  "certifications": [
    {
      "name": "",
      "issuing_organization": null,
      "issue_date": null,
      "expiry_date": null,
      "credential_id": null,
      "credential_url": null,
      "description": null
    }
  ],
  "projects": [
    {
      "name": "",
      "description": null,
      "role": null,
      "start_date": null,
      "end_date": null,
      "is_current": false,
      "technologies": [],
      "url": null,
      "repository_url": null,
      "highlights": [],
      "team_size": null,
      "type": null
    }
  ],
  "publications": [
    {
      "title": "",
      "authors": [],
      "publisher": null,
      "journal": null,
      "conference": null,
      "date": null,
      "url": null,
      "doi": null,
      "isbn": null,
      "description": null,
      "type": null
    }
  ],
  "awards_and_honors": [
    {
      "title": "",
      "issuer": null,
      "date": null,
      "description": null,
      "level": null
    }
  ],
  "volunteer_experience": [
    {
      "organization": "",
      "role": null,
      "start_date": null,
      "end_date": null,
      "is_current": false,
      "location": null,
      "description": null,
      "responsibilities": [],
      "cause": null
    }
  ],
  "languages": [
    {
      "language": "",
      "proficiency": null,
      "reading": null,
      "writing": null,
      "speaking": null
    }
  ],
  "interests_and_hobbies": [],
  "references": [
    {
      "name": null,
      "title": null,
      "company": null,
      "email": null,
      "phone": null,
      "relationship": null,
      "available_on_request": false
    }
  ],
  "patents": [
    {
      "title": "",
      "patent_number": null,
      "date": null,
      "description": null,
      "status": null,
      "inventors": [],
      "url": null
    }
  ],
  "professional_memberships": [
    {
      "organization": "",
      "role": null,
      "membership_type": null,
      "start_date": null,
      "end_date": null,
      "is_current": false
    }
  ],
  "conferences_and_talks": [
    {
      "title": "",
      "event": null,
      "date": null,
      "location": null,
      "description": null,
      "url": null,
      "type": null
    }
  ],
  "courses": [
    {
      "name": "",
      "provider": null,
      "platform": null,
      "date": null,
      "url": null,
      "credential_id": null,
      "duration": null
    }
  ],
  "training": [
    {
      "name": "",
      "provider": null,
      "date": null,
      "duration": null,
      "description": null
    }
  ],
  "extracurricular_activities": [
    {
      "organization": "",
      "role": null,
      "start_date": null,
      "end_date": null,
      "description": null
    }
  ],
  "analytics": {
    "total_years_of_experience": null,
    "total_months_of_experience": null,
    "career_level": null,
    "primary_industry": null,
    "secondary_industries": [],
    "job_functions": [],
    "highest_education_level": null,
    "number_of_companies": null,
    "number_of_roles": null,
    "average_tenure_months": null,
    "has_international_experience": null,
    "primary_location": null,
    "salary_mentioned": null,
    "resume_language": null
  },
  "raw_sections": {
    "section_names_found": [],
    "unclassified_content": null
  }
}"""


# ------------------------------------------------------------------ #
# Provider implementations
# ------------------------------------------------------------------ #

async def _call_openai(user_message: str) -> tuple[str, dict]:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = _get_openai()

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        response_format={"type": "json_object"},
        max_tokens=16000,
        temperature=0,
    )

    raw = response.choices[0].message.content or ""
    return raw, {
        "provider": "openai",
        "model": model,
        "usage": {
            "input_tokens":  response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens":  response.usage.total_tokens,
        },
    }


async def _call_anthropic(user_message: str) -> tuple[str, dict]:
    import anthropic as _anthropic
    model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    client = _get_anthropic()

    async with client.messages.stream(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        resp = await stream.get_final_message()

    raw = next((b.text for b in resp.content if b.type == "text"), "")
    return raw, {
        "provider": "anthropic",
        "model": model,
        "usage": {
            "input_tokens":        resp.usage.input_tokens,
            "output_tokens":       resp.usage.output_tokens,
            "cache_creation_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0),
            "cache_read_tokens":     getattr(resp.usage, "cache_read_input_tokens", 0),
        },
    }


# ------------------------------------------------------------------ #
# Public entry point
# ------------------------------------------------------------------ #

async def process_resume(
    raw_text: str,
    file_name: str,
    file_type: str,
    page_count: int,
    extraction_info: dict,
    client_id: str | None = None,
    project_id: str | None = None,
) -> dict:
    """
    Run LLM extraction, validate output, attach metadata.
    Safe for concurrent invocation — no shared mutable state per request.
    """
    word_count = len(raw_text.split())
    request_id = str(uuid.uuid4())

    user_message = (
        f"Resume file: {file_name} | {page_count} page(s) | "
        f"~{word_count} words | extracted via {extraction_info.get('method', 'unknown')}\n\n"
        "=== RESUME TEXT START ===\n"
        f"{raw_text}\n"
        "=== RESUME TEXT END ===\n\n"
        "Extract every detail. Return valid JSON only."
    )

    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    if provider == "anthropic":
        raw_json_text, llm_info = await _call_anthropic(user_message)
    else:
        raw_json_text, llm_info = await _call_openai(user_message)

    # Strip accidental markdown fences
    text = raw_json_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = len(lines) - 1 if lines and lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])

    try:
        extracted: dict = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON ({exc}). "
            f"First 400 chars of response:\n{text[:400]}"
        )

    extracted["_metadata"] = {
        "request_id":        request_id,
        "file_name":         file_name,
        "file_type":         file_type,
        "page_count":        page_count,
        "word_count":        word_count,
        "extraction_method": extraction_info.get("method"),
        "sparse_pages":      extraction_info.get("sparse_pages", []),
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        **({"client_id":  client_id}  if client_id  else {}),
        **({"project_id": project_id} if project_id else {}),
        **llm_info,
    }

    cleaned, warnings = validate_resume_json(extracted)
    return cleaned
