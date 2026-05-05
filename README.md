# Resume Extraction Engine

A FastAPI backend that accepts a resume file (PDF, DOCX, DOC, or TXT) and returns a fully structured JSON with every detail extracted — personal info, work history, education, skills, certifications, projects, publications, and 20+ more sections.

Supports two modes: a fast **single-shot LLM call** or a high-accuracy **multi-agent orchestration pipeline** with 4 stages and 9 specialized agents running in parallel.

---

## Features

- **Complete extraction** — verbatim bullet points, all metrics, every section, nothing skipped
- **Multi-format** — PDF (layout-aware, multi-column), DOCX, DOC, plain text
- **Dual LLM providers** — OpenAI GPT-4o or Anthropic Claude Opus 4.7, switchable via env var
- **Multi-agent pipeline** — StructureAgent → 6 parallel section agents → AnalyticsAgent → ValidatorAgent
- **Pydantic validation** — every field type-checked and coerced after extraction
- **Text normalization** — ligatures, broken hyphens, page numbers, running headers cleaned before LLM sees the text
- **AWS Lambda ready** — in-memory file processing, Mangum adapter, no disk writes
- **Multi-client/project** — optional `client_id` and `project_id` per request included in response metadata

---

## Architecture

```
POST /extract
      │
      ▼
 extractor.py          ← reads PDF/DOCX/TXT bytes → raw text
      │
      ▼
 normalizer.py         ← cleans ligatures, broken lines, headers/footers
      │
      ▼
 orchestrator.py       ← 4-stage multi-agent pipeline (USE_ORCHESTRATOR=true)
  │                       or processor.py single-shot LLM call
  │
  ├─ Stage 1  StructureAgent      (sequential) — maps job boundaries
  ├─ Stage 2  PersonalInfoAgent   ┐
  │           WorkExperienceAgent │ (parallel asyncio.gather)
  │           EducationAgent      │
  │           SkillsAgent         │
  │           CertificationsAgent │
  │           SupplementalAgent   ┘
  ├─ Stage 3  AnalyticsAgent      (sequential) — computes derived fields
  └─ Stage 4  ValidatorAgent      (sequential) — cross-validates bullet counts
      │
      ▼
 validator.py          ← Pydantic validation + coercion on merged JSON
      │
      ▼
 JSON response         ← 20+ sections + _metadata
```

---

## Project Structure

```
resume-extraction-engine/
├── main.py              # FastAPI app, routes, CORS, file type resolution
├── extractor.py         # PDF / DOCX / TXT text extraction
├── normalizer.py        # Text normalization pipeline
├── processor.py         # Single-shot LLM call (OpenAI or Anthropic)
├── orchestrator.py      # Multi-agent pipeline coordinator
├── validator.py         # Pydantic models for every resume section
├── handler.py           # AWS Lambda entrypoint (Mangum wrapper)
├── agents/
│   ├── base.py          # BaseAgent with shared LLM call logic
│   ├── structure.py     # Identifies job boundaries and bullet counts
│   ├── personal.py      # Personal information extraction
│   ├── work.py          # Work experience extraction
│   ├── education.py     # Education extraction
│   ├── skills.py        # Skills categorisation (14 sub-categories)
│   ├── certifications.py
│   ├── supplemental.py  # Projects, publications, awards, languages, etc.
│   ├── analytics.py     # Computes years of experience, career level, etc.
│   └── validator_agent.py # Cross-validates and re-extracts mismatches
├── terraform/
│   ├── main.tf          # AWS Lambda, IAM, S3, Function URL, CloudWatch
│   ├── variables.tf
│   └── outputs.tf
├── .github/
│   └── workflows/
│       └── deploy.yml   # CI lint + Terraform deploy on push to main
├── requirements.txt
└── .env.example
```

---

## Quick Start

See [SETUP.md](SETUP.md) for full local and AWS Lambda setup instructions.

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your API key and set MODEL_PROVIDER

# 3. Run
uvicorn main:app --reload
# → http://localhost:8000
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | Required when `MODEL_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o` | Any OpenAI chat model |
| `ANTHROPIC_API_KEY` | — | Required when `MODEL_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-opus-4-7` | Any Anthropic messages model |
| `USE_ORCHESTRATOR` | `true` | `true` = multi-agent pipeline, `false` = single-shot |
| `MAX_FILE_SIZE_MB` | `20` | Upload size limit |

---

## API

### `POST /extract`

Upload a resume and receive structured JSON.

**Query parameters (optional)**

| Param | Type | Description |
|---|---|---|
| `client_id` | string | Tag extraction to a client |
| `project_id` | string | Tag extraction to a project |

**curl**

```bash
curl -X POST "http://localhost:8000/extract?client_id=acme&project_id=q4-hire" \
     -F "file=@resume.pdf"
```

**Python**

```python
import httpx

with open("resume.pdf", "rb") as f:
    r = httpx.post(
        "http://localhost:8000/extract",
        params={"client_id": "acme", "project_id": "q4-hire"},
        files={"file": ("resume.pdf", f, "application/pdf")},
    )
print(r.json())
```

**Response shape (abbreviated)**

```json
{
  "personal_information": {
    "full_name": "Jane Doe",
    "email": ["jane@example.com"],
    "phone": ["+1-555-0100"],
    "linkedin_url": "https://linkedin.com/in/janedoe",
    "address": { "city": "New York", "state": "NY", "country": "USA" }
  },
  "professional_summary": "...",
  "work_experience": [
    {
      "company_name": "Acme Corp",
      "job_title": "Senior Engineer",
      "start_date": "Jan 2020",
      "end_date": "Present",
      "is_current": true,
      "team_size": 12,
      "responsibilities": ["..."],
      "achievements": ["..."],
      "technologies_used": ["Python", "FastAPI"]
    }
  ],
  "education": [{ "institution_name": "MIT", "degree": "B.S.", "field_of_study": "Computer Science", "gpa": 3.9 }],
  "skills": {
    "programming_languages": ["Python", "Go"],
    "frameworks_and_libraries": ["FastAPI", "Django"],
    "databases": ["PostgreSQL", "Redis"],
    "cloud_platforms": ["AWS"],
    "tools_and_platforms": ["Docker", "Git"]
  },
  "certifications": [],
  "projects": [],
  "publications": [],
  "awards_and_honors": [],
  "analytics": {
    "total_years_of_experience": 7.0,
    "career_level": "Senior",
    "primary_industry": "Information Technology",
    "highest_education_level": "Bachelor's Degree",
    "number_of_companies": 3,
    "number_of_roles": 4,
    "average_tenure_months": 21
  },
  "_metadata": {
    "request_id": "3f2a...",
    "file_name": "resume.pdf",
    "page_count": 2,
    "word_count": 640,
    "timestamp": "2026-05-01T10:00:00+00:00",
    "client_id": "acme",
    "project_id": "q4-hire",
    "provider": "orchestrator",
    "model": "multi-agent"
  }
}
```

### `GET /health`

```json
{ "status": "healthy" }
```

### `GET /`

Returns service info — provider, model, supported formats.

---

## Supported Formats

| Format | Extraction method |
|---|---|
| PDF | pdfplumber — layout-aware, table-aware, multi-column, repeated header/footer removal |
| DOCX / DOC | python-docx — paragraphs + tables + headers/footers in document order |
| TXT | UTF-8 decode |

> Scanned / image-only PDFs are not supported. The file must contain selectable text.

---

## JSON Schema — All Sections

`personal_information` · `professional_summary` · `objective` · `work_experience` · `education` · `skills` (14 sub-categories) · `certifications` · `projects` · `publications` · `awards_and_honors` · `volunteer_experience` · `languages` · `interests_and_hobbies` · `references` · `patents` · `professional_memberships` · `conferences_and_talks` · `courses` · `training` · `extracurricular_activities` · `analytics` · `raw_sections` · `_metadata`

---

## Deployment

See [SETUP.md](SETUP.md) → **AWS Lambda Deployment** section for full Terraform + GitHub Actions instructions.
