"""
A minimal PDF writer, for tests only.

The column detector reads word geometry, so testing it against hand-made word
dicts tests the arithmetic but not the thing that actually runs: pdfplumber's
reading of a real file. No PDF-writing library is in the dependency set and
adding one to ship in a Lambda to serve tests would be the wrong trade, so this
writes the handful of PDF objects needed to place text at chosen coordinates.

Enough PDF to be opened by pdfminer and no more: one page, one Type 1 font, a
content stream of positioned strings, and a correct xref table (pdfminer will
not parse the file without one).
"""
from __future__ import annotations

PAGE_WIDTH = 612
PAGE_HEIGHT = 792


def _escape(s: str) -> bytes:
    """PDF string literals escape their own delimiters."""
    out = s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return out.encode("latin-1", "replace")


def build_pdf(runs: list[tuple[float, float, str]], font_size: float = 10.0) -> bytes:
    """A one-page PDF placing each (x, y, text) run at that point.

    Coordinates are PDF user space: origin bottom-left, y increasing upward.
    """
    stream_parts = [b"BT\n"]
    for x, y, text in runs:
        stream_parts.append(
            b"/F1 %s Tf 1 0 0 1 %s %s Tm (%s) Tj\n"
            % (
                f"{font_size:g}".encode(),
                f"{x:g}".encode(),
                f"{y:g}".encode(),
                _escape(text),
            )
        )
    stream_parts.append(b"ET")
    stream = b"".join(stream_parts)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>" % (PAGE_WIDTH, PAGE_HEIGHT),
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )
    return bytes(out)


def build_columns(
    left: list[str],
    right: list[str],
    *,
    left_x: float = 60,
    right_x: float = 300,
    top_y: float = 720,
    leading: float = 16,
) -> bytes:
    """A two-column page: two blocks of lines side by side."""
    runs: list[tuple[float, float, str]] = []
    for i, line in enumerate(left):
        if line:
            runs.append((left_x, top_y - i * leading, line))
    for i, line in enumerate(right):
        if line:
            runs.append((right_x, top_y - i * leading, line))
    return build_pdf(runs)


def build_single_column(
    lines: list[str], *, x: float = 60, top_y: float = 720, leading: float = 16
) -> bytes:
    runs = [(x, top_y - i * leading, ln) for i, ln in enumerate(lines) if ln]
    return build_pdf(runs)
