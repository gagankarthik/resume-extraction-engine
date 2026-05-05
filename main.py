import os
import uvicorn
from pathlib import Path
from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from extractor import extract_text
from processor import process_resume

load_dotenv(Path(__file__).parent / ".env", override=True)

app = FastAPI(
    title="Resume Extraction Engine",
    description=(
        "Upload any resume (PDF, DOCX, DOC, or TXT) "
        "and receive a fully structured JSON with every detail extracted."
    ),
    version="2.0.0",
)

MAX_FILE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024

# Maps MIME type → internal file_type string for the extractor
MIME_TO_TYPE: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "text/plain": "txt",
}

# Extension fallback when MIME is missing/wrong
EXT_TO_TYPE: dict[str, str] = {
    "pdf": "pdf",
    "docx": "docx",
    "doc": "doc",
    "txt": "txt",
}

SUPPORTED_DISPLAY = ["PDF", "DOCX", "DOC", "TXT"]


@app.get("/")
async def root():
    return {
        "service": "Resume Extraction Engine",
        "version": "2.0.0",
        "provider": os.getenv("MODEL_PROVIDER", "openai"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                 if os.getenv("MODEL_PROVIDER", "openai") == "openai"
                 else os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
        "supported_formats":  SUPPORTED_DISPLAY,
        "max_file_size_mb": MAX_FILE_BYTES // (1024 * 1024),
        "endpoints": {
            "POST /extract": "Upload a resume — returns complete structured JSON",
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
):
    # --- Resolve file type ---
    content_type = (file.content_type or "").split(";")[0].strip()
    file_type = MIME_TO_TYPE.get(content_type)

    if file_type is None:
        # Try to infer from file extension
        ext = (file.filename or "").rsplit(".", 1)[-1].lower() if file.filename else ""
        file_type = EXT_TO_TYPE.get(ext)

    if file_type is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{content_type}'. "
                f"Accepted formats: {', '.join(SUPPORTED_DISPLAY)}"
            ),
        )

    # --- Read file ---
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_BYTES // (1024 * 1024)} MB.",
        )

    # --- Extract text ---
    try:
        raw_text, page_count, extraction_info = extract_text(file_bytes, file_type)
    except RuntimeError as exc:
        # Tesseract not installed etc.
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {exc}")

    if not raw_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No readable text found. "
                "If this is a scanned document, install Tesseract OCR and set TESSERACT_CMD. "
                "For image-based PDFs, ensure the file is not password-protected."
            ),
        )

    # --- LLM extraction ---
    try:
        result = await process_resume(
            raw_text=raw_text,
            file_name=file.filename or "resume",
            file_type=file_type,
            page_count=page_count,
            extraction_info=extraction_info,
            client_id=client_id,
            project_id=project_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM extraction failed: {exc}")

    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
