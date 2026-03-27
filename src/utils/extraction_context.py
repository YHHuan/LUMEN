"""
Smart Extraction Context Builder — LUMEN v2
=============================================
Builds optimal LLM context for data extraction:
- Short papers (<8 pages): use full text
- Long papers: vector retrieval for relevant chunks
- Always includes: all tables, methods, results sections
"""

import logging
from typing import List

from src.utils.pdf_decomposer import DocumentSegment, format_segments_for_llm, estimate_tokens
from src.utils.vector_index import DocumentVectorIndex, build_field_queries

logger = logging.getLogger(__name__)


def build_extraction_context(
    segments: List[DocumentSegment],
    vector_index: DocumentVectorIndex,
    extraction_guidance: dict = None,
    max_tokens: int = 6000,
    full_context_threshold: int = 8,
) -> str:
    """
    Build optimal context for the Extractor LLM.

    For short papers (<threshold pages): use full text (no retrieval overhead)
    For longer papers: use vector retrieval to select relevant chunks
    """
    if not segments:
        return ""

    total_pages = max(seg.page_number for seg in segments)

    if total_pages <= full_context_threshold:
        return format_segments_for_llm(segments)

    # Long paper — smart retrieval
    field_queries = build_field_queries(extraction_guidance)
    retrieved = vector_index.retrieve_for_extraction_fields(field_queries)
    merged_context = list(retrieved["_merged_context"])

    # Force-include certain segment types
    must_include_types = {"table", "heading"}
    must_include_sections = {
        "methods", "results", "statistical analysis", "outcomes",
        "materials and methods", "study design", "participants",
    }

    existing_ids = {id(seg) for seg in merged_context}

    for seg in segments:
        if id(seg) in existing_ids:
            continue

        if seg.segment_type in must_include_types:
            merged_context.append(seg)
            existing_ids.add(id(seg))
            continue

        if seg.section_heading:
            heading_lower = seg.section_heading.lower()
            if any(sec in heading_lower for sec in must_include_sections):
                merged_context.append(seg)
                existing_ids.add(id(seg))

    # Sort back to document order
    seg_order = {id(seg): i for i, seg in enumerate(segments)}
    merged_context.sort(key=lambda s: seg_order.get(id(s), 999))

    context_text = format_segments_for_llm(merged_context)

    # Truncate if still too long
    if estimate_tokens(context_text) > max_tokens:
        context_text = _truncate_to_token_limit(context_text, max_tokens)

    return context_text


def _truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate text to approximate token limit, preferring to cut references."""
    # First try: remove reference sections
    ref_markers = ["[REF]", "=== REFERENCES ==="]
    for marker in ref_markers:
        idx = text.find(marker)
        if idx > 0:
            truncated = text[:idx].rstrip()
            if estimate_tokens(truncated) <= max_tokens:
                return truncated + "\n\n[... references truncated ...]"

    # Hard truncate by character count
    char_limit = max_tokens * 4
    if len(text) > char_limit:
        return text[:char_limit] + "\n\n[... truncated ...]"

    return text
