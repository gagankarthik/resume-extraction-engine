# Resume Extraction Engine

A FastAPI backend that accepts a resume file (PDF, DOCX, DOC, TXT) and returns a fully structured JSON with every detail extracted — personal info, work history, education, skills, certifications, projects, publications, and 15+ more sections.

## Features

- **Complete extraction** — nothing skipped, verbatim bullet points, all metrics preserved
- **Multi-format** — PDF (layout-aware, multi-column), DOCX, DOC, plain text
- **Dual LLM providers** — OpenAI GPT-4o-mini (default) or Anthropic Claude Opus 4.7, switchable via env var
- **Pydantic validation** — every section and list item is type-validated and coerced (no data loss)
- **Multi-client/project** — optional `client_id` and `project_id` per request, included in response metadata
- **Concurrent-safe** — singleton LLM client with connection pool reuse across all requests
- **Text normalization** — ligatures, broken lines, hyphen-breaks, running headers/footers, page numbers all cleaned before the LLM sees the text

## Project Structure

```
resume-extraction-engine/
├── main.py          # FastAPI app, routes, file type resolution
├── extractor.py     # PDF / DOCX / TXT text extraction
├── normalizer.py    # Text normalization pipeline
├── processor.py     # LLM call (OpenAI or Anthropic), metadata assembly
├── validator.py     # Pydantic models for every resume section
├── requirements.txt
└── .env.example
```

## Quick Start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment**

```bash
cp .env.example .env
# Edit .env — add your API key and choose a provider
```

**3. Run the server**

```bash
python main.py
# or
uvicorn main:app --reload
```

Server starts at `http://localhost:8000`.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | Required when using OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI chat model |
| `ANTHROPIC_API_KEY` | — | Required when using Anthropic |
| `ANTHROPIC_MODEL` | `claude-opus-4-7` | Any Anthropic messages model |
| `MAX_FILE_SIZE_MB` | `20` | Upload size limit |

## API

### `POST /extract`

Upload a resume and receive structured JSON.

**Query parameters (optional)**

| Param | Type | Description |
|---|---|---|
| `client_id` | string | Tag the extraction to a client |
| `project_id` | string | Tag the extraction to a project |

**Example — curl**

```bash
curl -X POST "http://localhost:8000/extract?client_id=acme&project_id=q4-hire" \
     -F "file=@resume.pdf"
```

**Example — Python**

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
  "education": [ { "institution_name": "MIT", "degree": "B.S.", "gpa": 3.9 } ],
  "skills": {
    "programming_languages": ["Python", "Go"],
    "frameworks_and_libraries": ["FastAPI", "Django"],
    "databases": ["PostgreSQL", "Redis"]
  },
  "certifications": [],
  "projects": [],
  "publications": [],
  "awards_and_honors": [],
  "languages": [],
  "analytics": {
    "total_years_of_experience": 7.0,
    "career_level": "Senior",
    "primary_industry": "Software Engineering"
  },
  "_metadata": {
    "request_id": "3f2a...",
    "file_name": "resume.pdf",
    "page_count": 2,
    "word_count": 640,
    "timestamp": "2026-05-01T10:00:00+00:00",
    "client_id": "acme",
    "project_id": "q4-hire",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "usage": { "input_tokens": 1800, "output_tokens": 1200, "total_tokens": 3000 }
  }
}
```

### `GET /`

Service info — provider, model, supported formats.

### `GET /health`

```json
{ "status": "healthy" }
```

## Supported Formats

| Format | Extraction method |
|---|---|
| PDF | pdfplumber (layout-aware, table-aware, multi-column) |
| DOCX / DOC | python-docx (paragraphs + tables + headers/footers in document order) |
| TXT | UTF-8 decode |

Scanned / image-only PDFs are not supported. Use a text-based PDF or convert to DOCX first.

## JSON Schema

The response covers 20+ top-level sections:

`personal_information` · `professional_summary` · `objective` · `work_experience` · `education` · `skills` (14 sub-categories) · `certifications` · `projects` · `publications` · `awards_and_honors` · `volunteer_experience` · `languages` · `interests_and_hobbies` · `references` · `patents` · `professional_memberships` · `conferences_and_talks` · `courses` · `training` · `extracurricular_activities` · `analytics` · `raw_sections` · `_metadata`
