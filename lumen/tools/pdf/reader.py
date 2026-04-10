"""
PDF Reader — LUMEN v3
=====================
Extract text (and tables) from PDFs using pdfplumber.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import List, Optional

import pdfplumber
import structlog

logger = structlog.get_logger()


def extract_text(pdf_path: str, max_pages: int = 50,
                 strip_references: bool = False) -> str:
    """
    Extract text from a PDF file on disk.

    Each page is delimited by ``<!-- PAGE X -->`` markers.
    Tables detected by pdfplumber are appended as Markdown tables
    after the page's running text.

    Args:
        pdf_path: Path to a ``.pdf`` file.
        max_pages: Stop after this many pages (0 = unlimited).
        strip_references: If True, truncate from "References" heading onward.

    Returns:
        Combined text of the document.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    with pdfplumber.open(path) as pdf:
        text = _extract_pages(pdf, max_pages)

    if strip_references:
        text = _strip_reference_section(text)

    return text


def _strip_reference_section(text: str) -> str:
    """Remove the reference/bibliography section from extracted PDF text.

    Looks for common heading patterns (case-insensitive) that mark the start
    of the bibliography. Keeps everything before the first match.  Only cuts
    if the match is in the last 40% of the document (safety: avoids false
    positives from a "References" mention in the introduction).
    """
    import re
    # Patterns ordered from most specific to least
    _REF_PATTERNS = [
        # Page-marker followed by heading
        r'<!-- PAGE \d+ -->\s*\n\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*\n',
        # Standalone heading line (possibly numbered: "8. References")
        r'\n\s*(?:\d+\.?\s+)?(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography'
        r'|WORKS\s+CITED|Works\s+Cited|LITERATURE\s+CITED)\s*\n',
        # "References" at end of a line (e.g. after abbreviations table)
        r'\n[^\n]*\bReferences\s*\n\s*(?:\[?1[\].])',
        # All-caps on its own line (spaced or not)
        r'\n\s*R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S\s*\n',
        # Short line ending with References/REFERENCES (case-insensitive)
        r'(?i)\n[^\n]{0,50}references\s*\n',
    ]
    threshold = len(text) * 0.6  # must be in last 40%

    for pat in _REF_PATTERNS:
        for m in re.finditer(pat, text):
            if m.start() >= threshold:
                trimmed = text[:m.start()].rstrip()
                logger.info("pdf_references_stripped",
                            original_chars=len(text),
                            trimmed_chars=len(trimmed),
                            saved_chars=len(text) - len(trimmed))
                return trimmed
    return text


def extract_text_from_bytes(pdf_bytes: bytes, max_pages: int = 50) -> str:
    """
    Extract text from an in-memory PDF.

    Useful when a PDF has just been downloaded and is still held in
    memory (avoids writing a temp file yourself).

    Args:
        pdf_bytes: Raw PDF content.
        max_pages: Stop after this many pages (0 = unlimited).

    Returns:
        Combined text of the document.
    """
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return _extract_pages(pdf, max_pages)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _extract_pages(pdf: pdfplumber.PDF, max_pages: int) -> str:
    """Walk pages, pull text + tables, return combined Markdown-ish string."""
    pages = pdf.pages
    if max_pages > 0:
        pages = pages[:max_pages]

    parts: list[str] = []

    for idx, page in enumerate(pages, start=1):
        page_parts: list[str] = [f"<!-- PAGE {idx} -->"]

        # --- Running text ---
        text = page.extract_text() or ""
        if text.strip():
            page_parts.append(text)

        # --- Tables ---
        tables = _extract_tables(page)
        if tables:
            page_parts.append(tables)

        parts.append("\n\n".join(page_parts))

    full = "\n\n".join(parts)
    logger.info(
        "pdf_text_extracted",
        pages=len(pages),
        chars=len(full),
    )
    return full


def _extract_tables(page: pdfplumber.pdf.Page) -> Optional[str]:
    """
    Extract all tables on a page and format them as Markdown tables.

    Returns a single string with all tables separated by blank lines,
    or *None* if no tables were found.
    """
    raw_tables: List[list] = page.extract_tables() or []
    if not raw_tables:
        return None

    md_tables: list[str] = []
    for table in raw_tables:
        md = _table_to_markdown(table)
        if md:
            md_tables.append(md)

    return "\n\n".join(md_tables) if md_tables else None


def _table_to_markdown(table: list) -> Optional[str]:
    """
    Convert a pdfplumber table (list of rows, each row a list of cell
    strings) into a Markdown table.

    Returns *None* for degenerate tables (< 1 row or all-empty).
    """
    if not table or len(table) < 1:
        return None

    # Normalise cells: None → empty string, strip whitespace,
    # collapse internal newlines to spaces.
    def _clean(cell: object) -> str:
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").strip()

    rows = [[_clean(c) for c in row] for row in table if row]
    if not rows:
        return None

    # Skip if every cell is empty
    if all(c == "" for row in rows for c in row):
        return None

    # Ensure uniform column count
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
