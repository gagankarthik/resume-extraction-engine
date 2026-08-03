"""
What the engine accepts, and how it decides what a file actually is.

One table, read by everything. The MIME list, the extension list, the display
names in error messages and the dispatch in extractor.py all came from here, so
adding a format is one entry rather than five edits in four files.

WHY SNIFFING

A browser's Content-Type is a guess. It comes from the OS file association, so
the same resume arrives as application/msword, application/octet-stream or ""
depending on the machine that sent it — and .doc in particular is routinely a
lie: Word's "Save as .doc" wrote RTF for years, and plenty of "resume.doc"
files on disk are RTF or even a renamed .docx. Trusting the label means telling
someone their perfectly readable resume is unsupported.

So the declared type is only a hint. The first bytes of a file say what it
really is, and those win when they disagree.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileKind:
    """One accepted format."""

    key: str                      # internal name, used to dispatch extraction
    label: str                    # what the user sees: "PDF"
    extensions: tuple[str, ...]   # lowercase, no dot
    mimes: tuple[str, ...]


PDF = FileKind("pdf", "PDF", ("pdf",), ("application/pdf",))

DOCX = FileKind(
    "docx",
    "DOCX",
    ("docx",),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",),
)

# Legacy binary Word. Accepted at the door so the extractor can explain how to
# convert it, rather than the upload being refused with a generic type error.
DOC = FileKind("doc", "DOC", ("doc",), ("application/msword",))

RTF = FileKind("rtf", "RTF", ("rtf",), ("application/rtf", "text/rtf", "application/x-rtf"))

TXT = FileKind("txt", "TXT", ("txt", "text"), ("text/plain",))

KINDS: tuple[FileKind, ...] = (PDF, DOCX, DOC, RTF, TXT)

# Order matters only for how the list reads in a message.
SUPPORTED_DISPLAY: list[str] = [kind.label for kind in KINDS]

_BY_KEY = {kind.key: kind for kind in KINDS}
_BY_EXTENSION = {ext: kind for kind in KINDS for ext in kind.extensions}
_BY_MIME = {mime: kind for kind in KINDS for mime in kind.mimes}

# Deliberately not treated as a signal. Browsers send these for anything they
# cannot place, so matching on them would resolve every unknown file to one type.
_MEANINGLESS_MIMES = {"", "application/octet-stream", "binary/octet-stream"}


# ── Magic numbers ────────────────────────────────────────────────────────────
# Enough bytes to tell the container apart; none of these need the full header.

_PDF_MAGIC = b"%PDF"
_ZIP_MAGIC = b"PK\x03\x04"          # docx (and every other OOXML/zip)
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .doc/.xls compound file
_RTF_MAGIC = b"{\\rtf"


def sniff(file_bytes: bytes) -> str | None:
    """
    The format the bytes actually are, or None when they carry no signature.

    None means "no container magic", which for our purposes means plain text —
    but the caller decides that, because an empty or truncated upload also
    lands here and deserves a different message.
    """
    head = file_bytes[:8]
    if head.startswith(_PDF_MAGIC):
        return PDF.key
    if head.startswith(_RTF_MAGIC):
        return RTF.key
    if head.startswith(_ZIP_MAGIC):
        return DOCX.key
    if head.startswith(_OLE2_MAGIC):
        return DOC.key
    return None


def extension_of(filename: str | None) -> str:
    """Lowercase extension without the dot, or "" when there isn't one."""
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].strip().lower()


def declared_kind(content_type: str | None, filename: str | None) -> FileKind | None:
    """
    What the upload claims to be, by extension first and MIME second.

    Extension leads because it survives the trip intact, while Content-Type is
    rewritten by whatever chain the file passed through.
    """
    by_ext = _BY_EXTENSION.get(extension_of(filename))
    if by_ext is not None:
        return by_ext

    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in _MEANINGLESS_MIMES:
        return None
    return _BY_MIME.get(mime)


def resolve(content_type: str | None, filename: str | None, file_bytes: bytes) -> FileKind | None:
    """
    Decide what to parse this upload as, or None if it is not something we read.

    The bytes are the authority. The declaration only fills in for plain text,
    which has no signature to find.
    """
    # Nothing to go on. Callers check for an empty upload first and say so
    # plainly; this is here so the function is honest when used on its own.
    if not file_bytes:
        return None

    declared = declared_kind(content_type, filename)
    sniffed_key = sniff(file_bytes)

    if sniffed_key is not None:
        sniffed = _BY_KEY[sniffed_key]
        if declared is not None and sniffed is not declared:
            logger.info(
                "[filetypes] %s declared %s but the bytes are %s — reading it as %s",
                filename or "upload", declared.label, sniffed.label, sniffed.label,
            )
        return sniffed

    # No signature from here down. Only plain text legitimately looks like
    # this: a declared PDF, DOCX or DOC with no magic is truncated or corrupt,
    # and saying so is more use than trying to parse the wreckage.
    if declared is not None:
        return None if declared.key in ("pdf", "docx", "doc") else declared

    # An extension we don't know is a refusal, not an invitation to guess. The
    # file may well be text — a .csv usually is — but the person meant to
    # upload a resume, and reading whatever they dropped is how a spreadsheet
    # ends up in the pipeline. unsupported_message names the extension back.
    if extension_of(filename):
        return None

    # No extension and no signature: judge it by whether it reads as text, so a
    # resume exported without an extension still works.
    return TXT if _looks_like_text(file_bytes) else None


def _looks_like_text(file_bytes: bytes) -> bool:
    """True when the bytes decode cleanly and are not mostly control characters."""
    sample = file_bytes[:4096]
    if not sample:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def accepted_list() -> str:
    """"PDF, DOCX, DOC, RTF, or TXT" — for the end of a sentence."""
    labels = SUPPORTED_DISPLAY
    return f"{', '.join(labels[:-1])}, or {labels[-1]}"


def unsupported_message(filename: str | None, content_type: str | None) -> str:
    """
    Why this file was refused, in terms the person uploading it can act on.

    Names the file and what it looked like, because "unsupported file type" on
    its own leaves someone re-uploading the same thing.
    """
    ext = extension_of(filename)
    named = f"“{filename}”" if filename else "That file"

    if ext in ("pages", "odt", "wpd"):
        return (
            f"{named} is a {ext.upper()} document, which this tool cannot read. "
            f"Export it to PDF or DOCX and upload that instead."
        )
    if ext in ("png", "jpg", "jpeg", "gif", "heic", "webp", "tif", "tiff"):
        return (
            f"{named} is an image. Text has to be readable, not pictured — "
            f"upload the original document, or a PDF exported from it."
        )
    if ext:
        return (
            f"{named} is a .{ext} file. This tool reads {accepted_list()} — "
            f"save or export the resume as one of those and try again."
        )

    declared = (content_type or "").split(";")[0].strip()
    if declared and declared not in _MEANINGLESS_MIMES:
        return (
            f"{named} arrived as {declared}, which this tool cannot read. "
            f"Upload a {accepted_list()} file."
        )
    return (
        f"{named} has no file extension, and its contents do not match any format "
        f"this tool reads. Upload a {accepted_list()} file."
    )
