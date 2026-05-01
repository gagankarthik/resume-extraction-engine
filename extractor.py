"""
Resume text extraction — no OCR dependency.
  PDF  → pdfplumber (layout-aware, table-aware, repeated-header removal)
  DOCX → python-docx (paragraphs + tables + headers/footers in doc order)
  TXT  → UTF-8 decode
All paths run through normalizer.py before returning.
"""
import io

import pdfplumber
from docx import Document
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from docx.oxml.ns import qn

from normalizer import normalize_text, deduplicate_page_content

# Characters of text below which a PDF page is considered unparseable
_SPARSE_THRESHOLD = 50

ExtractionResult = tuple[str, int, dict]


# ------------------------------------------------------------------ #
# Public entry point
# ------------------------------------------------------------------ #

def extract_text(file_bytes: bytes, file_type: str) -> ExtractionResult:
    """
    Returns (normalized_text, page_count, extraction_info).
    Raises ValueError for unsupported types.
    Raises RuntimeError for unreadable files.
    """
    ft = file_type.lower()
    if ft == "pdf":
        return _extract_pdf(file_bytes)
    if ft in ("docx", "doc"):
        return _extract_docx(file_bytes)
    if ft == "txt":
        return _extract_txt(file_bytes)
    raise ValueError(f"Unsupported file type: {file_type!r}")


# ------------------------------------------------------------------ #
# PDF
# ------------------------------------------------------------------ #

def _extract_pdf(file_bytes: bytes) -> ExtractionResult:
    page_texts: list[str | None] = []
    sparse_pages: list[int] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            text = _extract_one_pdf_page(page)
            if len(text.strip()) < _SPARSE_THRESHOLD:
                sparse_pages.append(i)
            page_texts.append(text or None)

    if sparse_pages and all(t is None or len(t.strip()) < _SPARSE_THRESHOLD for t in page_texts):
        raise RuntimeError(
            "This PDF appears to be a scanned image — no machine-readable text found. "
            "Please use a text-based PDF or a DOCX file."
        )

    # Remove running headers/footers that repeat across pages
    page_texts = deduplicate_page_content(page_texts)

    combined = "\n\n".join(t for t in page_texts if t)
    normalized = normalize_text(combined)

    return normalized, page_count, {
        "method": "pdfplumber",
        "sparse_pages": sparse_pages,
    }


def _extract_one_pdf_page(page) -> str:
    """
    Extract text from one pdfplumber Page.
    Uses layout-aware extraction (preserves multi-column order),
    then appends any table rows not already captured.
    """
    # layout=True preserves spatial ordering — crucial for multi-column resumes
    text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3) or ""

    if not text.strip():
        text = page.extract_text() or ""

    # Explicit table extraction — pdfplumber may miss grid-based tables in layout mode
    table_rows: list[str] = []
    for table in page.extract_tables():
        for row in table:
            cells = [str(c).strip() for c in row if c and str(c).strip()]
            if cells:
                table_rows.append(" | ".join(cells))

    if table_rows:
        table_block = "\n".join(table_rows)
        # Only append if the content is not already present in the main text
        if table_block[:60] not in text:
            text = text + "\n\n" + table_block

    return text.strip()


# ------------------------------------------------------------------ #
# DOCX
# ------------------------------------------------------------------ #

def _extract_docx(file_bytes: bytes) -> ExtractionResult:
    doc = Document(io.BytesIO(file_bytes))
    parts: list[str] = []

    # Section headers
    for section in doc.sections:
        for attr in ("header", "even_page_header", "first_page_header"):
            try:
                hdr = getattr(section, attr)
                for para in hdr.paragraphs:
                    t = para.text.strip()
                    if t:
                        parts.append(t)
            except Exception:
                pass

    # Body in document order: paragraphs then tables
    for block in _iter_docx_blocks(doc):
        if isinstance(block, DocxParagraph):
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name or "").lower()
            is_heading = "heading" in style or "title" in style or (
                text.isupper() and 2 < len(text) < 80
            )
            parts.append(("\n" + text) if is_heading else text)

        elif isinstance(block, DocxTable):
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

    # Section footers (skip pure page numbers)
    for section in doc.sections:
        for attr in ("footer", "even_page_footer", "first_page_footer"):
            try:
                ftr = getattr(section, attr)
                for para in ftr.paragraphs:
                    t = para.text.strip()
                    if t and not t.isdigit():
                        parts.append(t)
            except Exception:
                pass

    combined = "\n".join(parts)
    normalized = normalize_text(combined)
    return normalized, 1, {"method": "docx"}


def _iter_docx_blocks(doc: Document):
    """Yield Paragraph and Table in document body order."""
    P_TAG = qn("w:p")
    T_TAG = qn("w:tbl")
    for child in doc.element.body:
        if child.tag == P_TAG:
            yield DocxParagraph(child, doc)
        elif child.tag == T_TAG:
            yield DocxTable(child, doc)


# ------------------------------------------------------------------ #
# Plain text
# ------------------------------------------------------------------ #

def _extract_txt(file_bytes: bytes) -> ExtractionResult:
    text = file_bytes.decode("utf-8", errors="replace")
    return normalize_text(text), 1, {"method": "txt"}
