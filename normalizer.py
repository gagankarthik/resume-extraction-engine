"""
Text normalization pipeline:
garbage in = garbage out — this runs before Claude sees any text.
"""
import re
import unicodedata

# Unicode ligatures and typographic characters to standardize
_CHAR_MAP = {
    "ﬁ": "fi",   # ﬁ
    "ﬂ": "fl",   # ﬂ
    "ﬀ": "ff",   # ﬀ
    "ﬃ": "ffi",  # ﬃ
    "ﬄ": "ffl",  # ﬄ
    " ": " ",    # non-breaking space
    "​": "",     # zero-width space
    "‌": "",     # zero-width non-joiner
    "‍": "",     # zero-width joiner
    "–": "-",    # en dash
    "—": "-",    # em dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "•": "•",  # bullet — keep as-is, normalized below
    "●": "•",
    "▪": "•",
    "▸": "•",
    "►": "•",
    "‣": "•",
    "⁃": "•",
    "⁌": "•",
    "∙": "•",
    "◦": "•",
    "✓": "•",  # check mark → bullet
    "✔": "•",
    "➤": "•",
    # Private-Use-Area glyphs commonly emitted by Wingdings / Symbol fonts in PDFs
    "": "•",   # Wingdings bullet
    "": "•",   # Wingdings v
    "": "•",   # Wingdings square
    "": "•",   # Wingdings diamond
    "": "•",   # Wingdings arrow
    "": "•",   # Wingdings check
    "": "•",   # Wingdings X-mark
    "": "•",
}

# Patterns for bullet characters at line start
_BULLET_PATTERN = re.compile(
    r"^[ \t]*[•●▪▸►‣⁃⁌∙◦✓✔➤‧⦿⦾―‖\*\-\+\>\~][ \t]+",
    re.MULTILINE,
)

# Orphan bullet glyph alone on its own line — a common PDF-extraction artifact
# where the glyph is rendered as a separate text run. Example:
#
#   •
#   Leading data architecture initiatives...
#
# Without this fix, the glyph stays orphaned and the following line is treated
# as a continuation line that won't merge (because it starts with a capital
# letter), so the LLM sees a single uncategorized sentence per job rather than
# a list of bullets.
_ORPHAN_BULLET = re.compile(
    r"^[ \t]*([•●▪▸►‣⁃⁌∙◦✓✔➤‧⦿⦾])[ \t]*\n[ \t]*(?=\S)",
    re.MULTILINE,
)

# Standalone page number lines: "1", "Page 2", "2 of 10", "- 3 -"
_PAGE_NUM_PATTERN = re.compile(
    r"^[ \t]*[-–—]?[ \t]*(?:page\s*)?\d+(?:\s*(?:of|\/)\s*\d+)?[ \t]*[-–—]?[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)

# Trailing whitespace per line
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)

# Multiple consecutive spaces/tabs (within a line)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# Hyphenated word break across lines: "soft-\nware" → "software"
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

# More than 3 consecutive blank lines → collapse to 2
_MANY_BLANKS = re.compile(r"\n{4,}")


def normalize_text(text: str) -> str:
    """
    Full normalization pipeline. Apply to extracted text before sending to Claude.
    Order matters — do not reorder steps.
    """
    if not text:
        return ""

    # 1. Unicode NFKC — decomposes compatibility characters
    text = unicodedata.normalize("NFKC", text)

    # 2. Replace ligatures and special chars
    for char, replacement in _CHAR_MAP.items():
        text = text.replace(char, replacement)

    # 3. Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Fix hyphenated line breaks (must run before line merging)
    text = _HYPHEN_BREAK.sub(lambda m: m.group(1) + m.group(2), text)

    # 5a. Re-attach orphan bullet glyphs to their text line. Run BEFORE bullet
    # normalization so the joined "• text" line matches _BULLET_PATTERN.
    # Iterate to collapse cascading orphans (e.g. two glyphs stacked above text).
    prev = None
    while prev != text:
        prev = text
        text = _ORPHAN_BULLET.sub(r"\1 ", text)

    # 5b. Normalize bullet characters at line start to "• "
    text = _BULLET_PATTERN.sub("• ", text)

    # 6. Remove standalone page number lines
    text = _PAGE_NUM_PATTERN.sub("", text)

    # 7. Merge broken continuation lines:
    #    A line is a continuation if it doesn't end a sentence/clause AND
    #    the next line starts lowercase without a bullet marker.
    text = _merge_broken_lines(text)

    # 8. Collapse multiple spaces within a line
    text = _MULTI_SPACE.sub(" ", text)

    # 9. Strip trailing whitespace from each line
    text = _TRAILING_WS.sub("", text)

    # 10. Collapse excess blank lines
    text = _MANY_BLANKS.sub("\n\n\n", text)

    # 11. Strip leading/trailing whitespace from the whole document
    return text.strip()


def _merge_broken_lines(text: str) -> str:
    """
    Merge lines that appear to be continuation of the previous line.
    Heuristic: current line does not end a clause, next line starts lowercase
    and is not a bullet/section header.
    """
    lines = text.split("\n")
    result = []
    i = 0
    SENTENCE_ENDERS = {".", "!", "?", ":", ";", ","}
    BULLET_START = re.compile(r"^[•\*\-\+\>][ \t]")

    while i < len(lines):
        current = lines[i]
        stripped = current.rstrip()

        if (
            i + 1 < len(lines)
            and stripped  # current line is not empty
            and len(stripped) < 120  # not already a very long line
            and stripped[-1] not in SENTENCE_ENDERS
        ):
            next_line = lines[i + 1].lstrip()

            # Merge only if next line is a lowercase continuation (not a new bullet/heading)
            if (
                next_line
                and next_line[0].islower()
                and not BULLET_START.match(next_line)
            ):
                result.append(stripped + " " + next_line)
                i += 2
                continue

        result.append(stripped)
        i += 1

    return "\n".join(result)


def deduplicate_page_content(page_texts: list[str | None]) -> list[str | None]:
    """
    Remove lines that appear verbatim on 3+ pages — these are running headers/footers.
    """
    if len(page_texts) < 3:
        return page_texts

    non_empty = [t for t in page_texts if t]
    if len(non_empty) < 3:
        return page_texts

    # Count line frequency across pages
    line_frequency: dict[str, int] = {}
    for page in non_empty:
        seen_on_this_page = set()
        for line in page.split("\n"):
            stripped = line.strip()
            if len(stripped) >= 3 and stripped not in seen_on_this_page:
                line_frequency[stripped] = line_frequency.get(stripped, 0) + 1
                seen_on_this_page.add(stripped)

    threshold = max(3, len(non_empty) // 2)
    repeated = {line for line, count in line_frequency.items() if count >= threshold}

    if not repeated:
        return page_texts

    cleaned = []
    for page in page_texts:
        if page is None:
            cleaned.append(None)
            continue
        lines = [
            ln for ln in page.split("\n")
            if ln.strip() not in repeated
        ]
        cleaned.append("\n".join(lines))

    return cleaned
