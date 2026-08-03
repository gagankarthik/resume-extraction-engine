"""
What a file is, versus what it says it is.

The cases that matter here are the mislabelled ones. A resume saved from Word
as "resume.doc" is very often RTF or a renamed .docx, and browsers report
whatever MIME the sending machine happens to associate with an extension. If
resolution trusted the label, those uploads would be turned away as legacy .doc
files that this service genuinely cannot read — so the routing is checked
against the bytes, not the name.
"""
from __future__ import annotations

import io
import zipfile

import pytest

import filetypes

RTF = (
    rb"{\rtf1\ansi\deff0{\fonttbl{\f0 Times;}}\f0\fs24 "
    rb"Jordan Avery\par Senior Backend Engineer\par}"
)
TXT = b"Jordan Avery\nSenior Engineer\n"
OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _docx() -> bytes:
    """The smallest thing that is genuinely a DOCX container."""
    buf = io.BytesIO()
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = (
        f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>'
        f"<w:p><w:r><w:t>Jordan Avery</w:t></w:r></w:p>"
        f"</w:body></w:document>"
    )
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("word/document.xml", body)
    return buf.getvalue()


DOCX = _docx()


@pytest.mark.parametrize(
    ("case", "filename", "content_type", "data", "expected"),
    [
        # Honest uploads.
        ("rtf", "resume.rtf", "application/rtf", RTF, "rtf"),
        ("docx", "resume.docx", DOCX_MIME, DOCX, "docx"),
        ("txt", "resume.txt", "text/plain", TXT, "txt"),
        ("legacy doc", "resume.doc", "application/msword", OLE2, "doc"),
        # Mislabelled, and the reason this function reads bytes at all.
        ("rtf wearing .doc", "resume.doc", "application/msword", RTF, "rtf"),
        ("docx wearing .doc", "resume.doc", "application/msword", DOCX, "docx"),
        # MIME variants that browsers actually send.
        ("rtf as text/rtf", "resume.rtf", "text/rtf", RTF, "rtf"),
        ("docx as octet-stream", "resume.docx", "application/octet-stream", DOCX, "docx"),
        ("txt with no extension", "resume", "", TXT, "txt"),
        # Not resumes we can read.
        ("image", "resume.png", "image/png", PNG, None),
        ("pages export", "resume.pages", "application/octet-stream", b"\x00\x01\x02" * 40, None),
        # Text-like, but an extension we do not accept. Refused rather than
        # guessed at, so a spreadsheet does not reach the extraction pipeline.
        ("csv", "contacts.csv", "text/csv", b"name,email\nJordan,j@x.com\n", None),
        ("unknown extension", "resume.xyz", "application/octet-stream", TXT, None),
    ],
)
def test_resolve(case, filename, content_type, data, expected):
    kind = filetypes.resolve(content_type, filename, data)
    assert (kind.key if kind else None) == expected, case


def test_truncated_pdf_is_not_read_as_text():
    """A .pdf with no %PDF header is damaged, not a text file to be parsed."""
    assert filetypes.resolve("application/pdf", "resume.pdf", b"not a pdf at all") is None


def test_empty_upload_resolves_to_nothing():
    assert filetypes.resolve("text/plain", "resume.txt", b"") is None


def test_unsupported_message_names_the_file_and_the_fix():
    message = filetypes.unsupported_message("resume.pages", "application/octet-stream")
    assert "resume.pages" in message
    assert "PDF" in message


def test_unsupported_message_recognises_an_image():
    assert "image" in filetypes.unsupported_message("scan.png", "image/png").lower()


def test_accepted_list_reads_as_a_sentence():
    assert filetypes.accepted_list().endswith("or TXT")
    assert "RTF" in filetypes.accepted_list()
