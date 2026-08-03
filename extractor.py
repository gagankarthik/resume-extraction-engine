"""
Resume text extraction — no OCR dependency.
  PDF  → pdfplumber (layout-aware, table-aware, repeated-header removal)
  DOCX → python-docx (paragraphs + tables + headers/footers in doc order)
           fallback: raw ZIP extraction of word/document.xml
  RTF  → striprtf (control words stripped, escapes and unicode decoded)
  DOC  → not parseable; explains how to convert. Files that merely *claim*
         .doc are usually RTF or DOCX and are routed by their real bytes,
         so this message is reached only by genuine OLE2 documents.
  TXT  → UTF-8 decode
All paths run through normalizer.py before returning.
"""
import io
import zipfile
from xml.etree import ElementTree as ET

import pdfplumber
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from striprtf.striprtf import rtf_to_text

from normalizer import deduplicate_page_content, normalize_text

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
    Raises RuntimeError for unreadable files, with a message meant for the
    person who uploaded them.

    `file_type` is filetypes.resolve()'s verdict — decided from the file's own
    bytes, not its name — so each branch here can trust what it is given.
    """
    handler = _HANDLERS.get(file_type.lower())
    if handler is None:
        raise ValueError(f"Unsupported file type: {file_type!r}")
    return handler(file_bytes)


# ------------------------------------------------------------------ #
# PDF
# ------------------------------------------------------------------ #

def _extract_pdf(file_bytes: bytes) -> ExtractionResult:
    page_texts: list[str | None] = []
    sparse_pages: list[int] = []

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                text = _extract_one_pdf_page(page)
                if len(text.strip()) < _SPARSE_THRESHOLD:
                    sparse_pages.append(i)
                page_texts.append(text or None)
    except Exception as exc:
        # A damaged or encrypted PDF is the uploader's problem to fix, not a
        # fault in this service — say which, and what to do, rather than 500.
        if "password" in str(exc).lower() or "encrypt" in str(exc).lower():
            raise RuntimeError(
                "This PDF is password-protected, so its text cannot be read. "
                "Remove the password, or export an unprotected copy, and upload that."
            ) from exc
        raise RuntimeError(
            "This PDF could not be opened — the file looks damaged or incomplete. "
            "Try re-downloading it, or open it and export a fresh PDF."
        ) from exc

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

# Legacy Word .doc files are OLE2 compound documents (magic D0 CF 11 E0 A1 B1 1A E1).
# Modern .docx files are ZIP archives (magic PK\x03\x04). python-docx only reads
# the latter, so detect the former up front and fail with an actionable message
# instead of letting python-docx + the ZIP fallback both throw opaque errors.
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# Reading OLE2 needs a Word-format parser (antiword, LibreOffice) that cannot be
# shipped in a Lambda zip. Saying exactly which conversions work is the whole
# value of this message — "unsupported" alone leaves someone stuck.
_LEGACY_DOC_MESSAGE = (
    "This is a legacy Word document (.doc), a format from before 2007 that this "
    "tool cannot read. Open it in Word or Google Docs and use “Save as” to make "
    "a .docx, or export it to PDF — either will upload fine."
)


def _extract_doc(file_bytes: bytes) -> ExtractionResult:
    """
    Genuine OLE2 .doc.

    Most uploads named .doc never reach here: filetypes.resolve() reads their
    bytes first, and a "resume.doc" that is really RTF or DOCX is routed to the
    parser that can read it. What is left is the real thing, which cannot be.
    """
    raise RuntimeError(_LEGACY_DOC_MESSAGE)


def _extract_docx(file_bytes: bytes) -> ExtractionResult:
    # Defence in depth: routing should have sent this to _extract_doc already.
    if file_bytes[:8] == _OLE2_MAGIC:
        raise RuntimeError(_LEGACY_DOC_MESSAGE)

    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as docx_exc:
        try:
            return _extract_docx_zip_fallback(file_bytes)
        except Exception:
            raise RuntimeError(
                f"Could not read this Word file ({docx_exc}). "
                "Try re-saving it as .docx in Microsoft Word, or convert to PDF."
            ) from docx_exc

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
            style = ((block.style.name if block.style else None) or "").lower()
            is_heading = "heading" in style or "title" in style or (
                text.isupper() and 2 < len(text) < 80
            )
            # Word stores bullet/numbered list items as paragraphs with a <w:numPr>
            # element but NO text glyph — the bullet is rendered from the numbering
            # definition. Without re-introducing a glyph here, the normalizer and
            # StructureAgent can't tell that "Designed and implemented X" was a
            # bullet vs a free-floating paragraph, and bullet_count comes out 0.
            # Prepend "• " so downstream regex bullet detection works.
            if not is_heading and _is_list_paragraph(block):
                text = f"• {text}"
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


def _extract_docx_zip_fallback(file_bytes: bytes) -> ExtractionResult:
    """Extract text from word/document.xml directly when python-docx rejects the file."""
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        names = z.namelist()
        doc_xml = next(
            (n for n in names if n == "word/document.xml"),
            next((n for n in names if "document" in n.lower() and n.endswith(".xml")), None),
        )
        if doc_xml is None:
            raise RuntimeError("No document.xml found inside ZIP archive")
        data = z.read(doc_xml)

    root = ET.fromstring(data)
    parts: list[str] = []
    for para in root.iter(f"{{{W}}}p"):
        text = "".join(t.text or "" for t in para.iter(f"{{{W}}}t")).strip()
        if not text:
            continue
        # Mirror the python-docx path: prepend "• " for paragraphs in a list.
        pPr = para.find(f"{{{W}}}pPr")
        is_list = pPr is not None and pPr.find(f"{{{W}}}numPr") is not None
        if is_list:
            text = f"• {text}"
        parts.append(text)

    combined = "\n".join(parts)
    return normalize_text(combined), 1, {"method": "docx_zip_fallback"}


def _iter_docx_blocks(doc: Document):
    """Yield Paragraph and Table in document body order."""
    P_TAG = qn("w:p")
    T_TAG = qn("w:tbl")
    for child in doc.element.body:
        if child.tag == P_TAG:
            yield DocxParagraph(child, doc)
        elif child.tag == T_TAG:
            yield DocxTable(child, doc)


def _is_list_paragraph(para: DocxParagraph) -> bool:
    """True if this paragraph is part of a bulleted or numbered list in Word.

    Detection is restricted to the presence of <w:numPr>, which is Word's
    definitive list-membership signal. Style name alone (e.g. "ListParagraph")
    is unreliable — Word applies it to plain indented prose too.
    """
    pPr = para._p.find(qn("w:pPr"))
    if pPr is None:
        return False
    return pPr.find(qn("w:numPr")) is not None


# ------------------------------------------------------------------ #
# Plain text
# ------------------------------------------------------------------ #

def _extract_txt(file_bytes: bytes) -> ExtractionResult:
    text = _decode_text(file_bytes)
    return normalize_text(text), 1, {"method": "txt"}


def _decode_text(file_bytes: bytes) -> str:
    """
    Decode bytes that are meant to be text.

    Resumes routinely arrive as Windows-1252 from Word or Notepad — smart
    quotes, em dashes and accented names are exactly the bytes UTF-8 rejects.
    Falling straight to errors="replace" would pepper those names with U+FFFD,
    so try the encodings that actually occur before giving up on any character.
    """
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


# ------------------------------------------------------------------ #
# RTF
# ------------------------------------------------------------------ #

def _extract_rtf(file_bytes: bytes) -> ExtractionResult:
    """
    RTF is text with markup, so it decodes before it parses.

    striprtf drops control words, skips destination groups such as font and
    colour tables, and turns \\'xx and \\uN escapes back into characters —
    including the \\bullet that list items are built from, which downstream
    bullet counting depends on.
    """
    try:
        text = rtf_to_text(_decode_text(file_bytes), errors="ignore")
    except Exception as exc:
        raise RuntimeError(
            "This RTF file could not be read — it may be damaged. Try opening it "
            "in Word or TextEdit and saving it again as .docx or PDF."
        ) from exc

    if not text.strip():
        raise RuntimeError(
            "This RTF file contains no readable text. If the resume is a picture "
            "pasted into the document, upload the original instead — text has to "
            "be selectable, not pictured."
        )

    return normalize_text(text), 1, {"method": "striprtf"}


# ------------------------------------------------------------------ #
# Dispatch
# ------------------------------------------------------------------ #
# Keys are FileKind.key values from filetypes.py. Adding a format means adding
# a FileKind there and one entry here; nothing else in the service changes.

_HANDLERS = {
    "pdf": _extract_pdf,
    "docx": _extract_docx,
    "doc": _extract_doc,
    "rtf": _extract_rtf,
    "txt": _extract_txt,
}
