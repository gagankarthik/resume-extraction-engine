import asyncio
import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import filetypes
from auth import TOKEN_HEADER, require_upload_token
from extractor import extract_text
from processor import process_resume

load_dotenv(Path(__file__).parent / ".env", override=True)

# Configure root logging once so agent/orchestrator warnings (failed agents,
# bullet-count mismatches, dropped data) actually surface in uvicorn/Lambda logs.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Extraction Engine — Resume State Format Tool",
    description=(
        "Resume extraction service for the Resume State Format Tool. "
        "Upload any resume (PDF, DOCX, DOC, or TXT) "
        "and receive a fully structured JSON with every detail extracted."
    ),
    version="2.0.0",
)

# CORS for browser-based callers. The browser posts resumes here directly —
# see auth.py for why — so these origins are the real gate on who may call the
# engine from a page, alongside the upload token.
#
# The Lambda Function URL has its own CORS config in Terraform, and that is the
# one that applies in production; this middleware covers a local
# `uvicorn main:app` run. Keep the two lists saying the same thing.
_DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,https://hire.oceanbluecorp.com"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

# Only one layer may answer CORS. Under a Function URL both were: the URL added
# Access-Control-Allow-Origin and this middleware added it again, so the browser
# saw the header twice ("contains multiple values '<origin>, <origin>'") and
# failed every upload — a duplicate is rejected even when both copies agree.
# AWS_LAMBDA_FUNCTION_NAME is set by the runtime, so it marks the deployed case;
# a local uvicorn run has no Function URL in front of it and still needs this.
IN_LAMBDA = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

if not IN_LAMBDA:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["Content-Type", TOKEN_HEADER],
    )

MAX_FILE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024
MAX_FILE_MB = MAX_FILE_BYTES // (1024 * 1024)


@app.get("/")
async def root():
    return {
        "service": "Resume Extraction Engine",
        "belongs_to": "Resume State Format Tool",
        "description": "Resume extraction for the Resume State Format Tool — not a general-purpose endpoint.",
        "version": "2.0.0",
        "provider": "openai",
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "supported_formats":  filetypes.SUPPORTED_DISPLAY,
        "max_file_size_mb": MAX_FILE_MB,
        "endpoints": {
            "POST /extract": f"Upload a resume — returns complete structured JSON. Requires a {TOKEN_HEADER} ticket.",
            "GET  /health":  "Health check",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/extract")
async def extract_resume(
    file: UploadFile = File(...),
    client_id: str | None = Query(default=None, description="Optional client identifier for multi-tenant tracking"),
    project_id: str | None = Query(default=None, description="Optional project identifier for grouping extractions"),
    claims: dict | None = Depends(require_upload_token),
):
    # The upload ticket is verified before a byte is read: this route runs a
    # ten-agent LLM pipeline, so an unauthorised call costs real money.
    if claims:
        logger.info("[extract] authorised upload for sub=%s", claims.get("sub", "unknown"))

    # --- Read the file ---
    # The bytes have to be in hand before the format can be decided: what a
    # file *is* is settled by its first bytes, not by the name or the
    # Content-Type the browser guessed. See filetypes.resolve.
    file_bytes = await file.read()
    name = file.filename or "the file"

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"“{name}” is empty — it contains no data. Check the file and upload it again.",
        )

    if len(file_bytes) > MAX_FILE_BYTES:
        actual_mb = len(file_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"“{name}” is {actual_mb:.1f} MB, over the {MAX_FILE_MB} MB limit. "
                f"Resumes are almost always well under this — if yours is large, it "
                f"probably contains images; export it to PDF again without them."
            ),
        )

    # --- Decide what it is ---
    kind = filetypes.resolve(file.content_type, file.filename, file_bytes)
    if kind is None:
        raise HTTPException(
            status_code=415,
            detail=filetypes.unsupported_message(file.filename, file.content_type),
        )

    # --- Extract text ---
    # extract_text is sync and CPU-bound (pdfplumber). Run it in a worker
    # thread so it doesn't block the event loop and stall concurrent uploads.
    try:
        raw_text, page_count, extraction_info = await asyncio.to_thread(
            extract_text, file_bytes, kind.key
        )
    except RuntimeError as exc:
        # Unreadable file: scanned PDF, legacy .doc, corrupt archive. These
        # messages are written for the person uploading and pass through as-is.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Anything else is a fault in this service, not in their file. Log the
        # detail; do not hand an internal traceback string to the caller.
        logger.exception("[extract] unhandled failure reading a %s file", kind.label)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Something went wrong reading “{name}”. This is a fault on our side, "
                f"not with the file. Please try again."
            ),
        ) from exc

    if not raw_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                f"No readable text was found in “{name}”. This usually means the "
                f"resume is a scan or a picture of a document — the text has to be "
                f"selectable, not photographed. Upload the original, or export it to "
                f"PDF from the app that made it. A password-protected file will also "
                f"land here."
            ),
        )

    # --- LLM extraction ---
    try:
        result = await process_resume(
            raw_text=raw_text,
            file_name=file.filename or "resume",
            file_type=kind.key,
            page_count=page_count,
            extraction_info=extraction_info,
            client_id=client_id,
            project_id=project_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM extraction failed: {exc}") from exc

    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
