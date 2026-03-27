"""
PDF Document Decomposition — LUMEN v2
=======================================
Decomposes PDFs into structured segments using:
- gmft (primary): GPU-free table detection via DETR + structure recognition
- pdfplumber (fallback): text extraction + simple table detection
- Section heading detection via font/formatting heuristics
"""

import hashlib
import json
import re
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentSegment:
    segment_type: str           # "text", "table", "figure_caption", "heading", "reference"
    content: str                # Markdown text or markdown table
    page_number: int
    section_heading: Optional[str] = None
    bbox: Optional[tuple] = None
    confidence: float = 0.95


def decompose_pdf(pdf_path: str) -> List[DocumentSegment]:
    """
    Decompose PDF into structured segments.

    Strategy:
    1. Try gmft for table detection (primary)
    2. Extract text with pdfplumber, excluding table regions
    3. Identify section headings by heuristics
    4. Fallback to pdfplumber tables if gmft unavailable
    """
    segments = []
    table_bboxes_by_page = {}

    # Step 1: Table detection with gmft
    gmft_tables = _extract_tables_gmft(pdf_path)
    if gmft_tables is not None:
        table_bboxes_by_page = gmft_tables["bboxes"]
        for table_seg in gmft_tables["segments"]:
            segments.append(table_seg)
    else:
        logger.info("gmft unavailable, using pdfplumber fallback for tables")

    # Step 2: Text extraction with pdfplumber
    text_segments = _extract_text_pdfplumber(
        pdf_path,
        table_bboxes_by_page,
        use_pdfplumber_tables=(gmft_tables is None),
    )
    segments.extend(text_segments)

    # Sort by page number, then by vertical position (bbox y0)
    segments.sort(key=lambda s: (
        s.page_number,
        s.bbox[1] if s.bbox else 0,
    ))

    # Assign section headings
    _assign_section_headings(segments)

    logger.info(
        f"Decomposed {pdf_path}: {len(segments)} segments "
        f"({sum(1 for s in segments if s.segment_type == 'table')} tables, "
        f"{sum(1 for s in segments if s.segment_type == 'heading')} headings)"
    )

    return segments


def _extract_tables_gmft(pdf_path: str) -> Optional[dict]:
    """Extract tables using gmft (GPU-free DETR table detection)."""
    try:
        from gmft.auto import CroppedTable, TableDetector, AutoTableFormatter
        from gmft.pdf_bindings import PyPDFium2Document
    except ImportError:
        return None

    try:
        detector = TableDetector()
        formatter = AutoTableFormatter()
        doc = PyPDFium2Document(pdf_path)

        bboxes = {}
        table_segments = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            tables = detector.extract(page)

            page_bboxes = []
            for table in tables:
                try:
                    ft = formatter.extract(table)
                    md_table = ft.df().to_markdown(index=False)

                    bbox = (
                        table.bbox[0], table.bbox[1],
                        table.bbox[2], table.bbox[3],
                    )
                    page_bboxes.append(bbox)

                    table_segments.append(DocumentSegment(
                        segment_type="table",
                        content=md_table,
                        page_number=page_idx + 1,
                        bbox=bbox,
                        confidence=getattr(table, "confidence_score", 0.9),
                    ))
                except Exception as e:
                    logger.debug(f"gmft table extraction failed on page {page_idx + 1}: {e}")
                    continue

            if page_bboxes:
                bboxes[page_idx] = page_bboxes

        doc.close()
        return {"bboxes": bboxes, "segments": table_segments}

    except Exception as e:
        logger.warning(f"gmft processing failed: {e}")
        return None


def _extract_text_pdfplumber(
    pdf_path: str,
    table_bboxes_by_page: dict,
    use_pdfplumber_tables: bool = False,
) -> List[DocumentSegment]:
    """Extract text segments using pdfplumber, optionally with table fallback."""
    import pdfplumber

    segments = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_bboxes = table_bboxes_by_page.get(page_idx, [])

            # Pdfplumber table fallback
            if use_pdfplumber_tables:
                for table in page.find_tables():
                    try:
                        extracted = table.extract()
                        if extracted:
                            md = _table_rows_to_markdown(extracted)
                            if md:
                                bbox = (
                                    table.bbox[0], table.bbox[1],
                                    table.bbox[2], table.bbox[3],
                                )
                                page_bboxes.append(bbox)
                                segments.append(DocumentSegment(
                                    segment_type="table",
                                    content=md,
                                    page_number=page_idx + 1,
                                    bbox=bbox,
                                    confidence=0.80,
                                ))
                    except Exception:
                        continue

            # Extract text outside table regions
            if page_bboxes:
                # Crop out table areas
                filtered_page = page
                for bbox in page_bboxes:
                    try:
                        # Keep text outside the table bbox
                        cropped = filtered_page.outside_bbox(bbox)
                        if cropped:
                            filtered_page = cropped
                    except Exception:
                        pass
                text = filtered_page.extract_text() or ""
            else:
                text = page.extract_text() or ""

            if not text.strip():
                continue

            # Split into paragraphs and classify
            paragraphs = _split_into_paragraphs(text)
            for para in paragraphs:
                para_type = _classify_paragraph(para)
                segments.append(DocumentSegment(
                    segment_type=para_type,
                    content=para,
                    page_number=page_idx + 1,
                    bbox=None,
                    confidence=0.90,
                ))

    return segments


def _split_into_paragraphs(text: str) -> List[str]:
    """Split extracted text into paragraphs."""
    # Split on double newlines or lines that look like section breaks
    raw_parts = re.split(r"\n{2,}", text.strip())
    paragraphs = []
    for part in raw_parts:
        part = part.strip()
        if part and len(part) > 10:
            paragraphs.append(part)
    return paragraphs


def _classify_paragraph(para: str) -> str:
    """Classify a paragraph as heading, text, reference, or figure_caption."""
    line = para.strip()

    # Heading patterns
    if len(line) < 80 and not line.endswith("."):
        heading_patterns = [
            r"^(?:\d+\.?\s+)?(?:abstract|introduction|background|methods?|"
            r"results?|discussion|conclusion|references|acknowledgements?|"
            r"materials?\s+and\s+methods?|statistical\s+analysis|"
            r"study\s+design|participants?|outcome\s+measures?|"
            r"data\s+extraction|risk\s+of\s+bias|supplementary)",
            r"^(?:\d+\.)+\s+\w",  # Numbered sections like "2.1 Methods"
            r"^(?:Table|Figure|Fig\.)\s+\d+",
        ]
        for pattern in heading_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return "heading"

    # Figure caption
    if re.match(r"^(?:Figure|Fig\.)\s+\d+[.:]", line, re.IGNORECASE):
        return "figure_caption"

    # Reference section entries
    if re.match(r"^\[\d+\]|\d+\.\s+[A-Z][a-z]+", line):
        if "doi" in line.lower() or "et al" in line.lower():
            return "reference"

    return "text"


def _assign_section_headings(segments: List[DocumentSegment]):
    """Assign section_heading to each segment based on preceding headings."""
    current_heading = None
    for seg in segments:
        if seg.segment_type == "heading":
            current_heading = seg.content.strip()[:100]
        seg.section_heading = current_heading


def _table_rows_to_markdown(rows: list) -> str:
    """Convert pdfplumber table rows to markdown table."""
    if not rows or len(rows) < 2:
        return ""

    # Clean cells
    def clean(cell):
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").strip()

    header = [clean(c) for c in rows[0]]
    if not any(header):
        return ""

    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in rows[1:]:
        cells = [clean(c) for c in row]
        # Pad/truncate to match header length
        while len(cells) < len(header):
            cells.append("")
        cells = cells[:len(header)]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_segments_for_llm(segments: List[DocumentSegment]) -> str:
    """Format segments into a structured text representation for LLM context."""
    parts = []
    current_page = 0

    for seg in segments:
        if seg.page_number != current_page:
            current_page = seg.page_number
            parts.append(f"\n=== PAGE {current_page} ===\n")

        if seg.segment_type == "heading":
            parts.append(f"[HEADING] {seg.content}")
        elif seg.segment_type == "table":
            label = f" ({seg.section_heading})" if seg.section_heading else ""
            parts.append(f"[TABLE{label}]\n{seg.content}")
        elif seg.segment_type == "figure_caption":
            parts.append(f"[FIGURE] {seg.content}")
        elif seg.segment_type == "reference":
            parts.append(f"[REF] {seg.content}")
        else:
            parts.append(seg.content)

    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (4 chars per token)."""
    return max(1, len(text) // 4)


# ======================================================================
# Caching
# ======================================================================

def get_or_decompose(pdf_path: str, cache_dir: str) -> List[DocumentSegment]:
    """Decompose PDF with disk caching (keyed by file hash)."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Compute file hash
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    pdf_hash = h.hexdigest()

    cached_file = cache_path / f"decomposed_{pdf_hash}.json"
    if cached_file.exists():
        with open(cached_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [DocumentSegment(**seg) for seg in data]

    segments = decompose_pdf(pdf_path)

    with open(cached_file, "w", encoding="utf-8") as f:
        json.dump([asdict(seg) for seg in segments], f, indent=2, ensure_ascii=False)

    return segments
