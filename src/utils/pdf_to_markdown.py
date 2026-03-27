"""
PDF-to-Markdown via Gemini — LUMEN v2
======================================
Multimodal PDF→Markdown conversion using Gemini's vision capabilities.
Falls back to pdfplumber-based extraction if Gemini is unavailable.

Key advantages over raw text extraction:
- Preserves table structure as markdown tables
- Handles multi-column layouts
- Extracts figure captions and labels
- Better handling of mathematical notation and subscripts

Usage:
    from src.utils.pdf_to_markdown import convert_pdf_to_markdown

    md_text = convert_pdf_to_markdown("path/to/paper.pdf")
"""

import base64
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(
    pdf_path: str,
    cache_dir: str = None,
    model: str = "google/gemini-3.1-pro-preview",
    max_pages: int = 50,
    pages_per_batch: int = 5,
) -> str:
    """
    Convert PDF to well-structured Markdown using Gemini multimodal.

    Args:
        pdf_path: Path to PDF file
        cache_dir: Directory for caching results
        model: Gemini model to use via OpenRouter
        max_pages: Maximum pages to process
        pages_per_batch: Pages to send per API call

    Returns:
        Markdown string of the full document
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Check cache
    if cache_dir:
        cached = _check_cache(str(pdf_path), cache_dir)
        if cached:
            logger.info(f"PDF→MD cache hit: {pdf_path.name}")
            return cached

    # Try Gemini conversion
    try:
        md_text = _convert_with_gemini(
            str(pdf_path), model=model,
            max_pages=max_pages, pages_per_batch=pages_per_batch,
        )
        logger.info(f"Gemini PDF→MD: {pdf_path.name} ({len(md_text)} chars)")
    except Exception as e:
        logger.warning(f"Gemini conversion failed ({e}), falling back to pdfplumber")
        md_text = _convert_with_pdfplumber(str(pdf_path))

    # Cache result
    if cache_dir:
        _save_cache(str(pdf_path), cache_dir, md_text)

    return md_text


def _convert_with_gemini(
    pdf_path: str,
    model: str = "google/gemini-3.1-pro-preview",
    max_pages: int = 50,
    pages_per_batch: int = 5,
) -> str:
    """Convert PDF pages to markdown using Gemini's vision via OpenRouter."""
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Convert PDF pages to images
    page_images = _pdf_to_images(pdf_path, max_pages=max_pages)
    total_pages = len(page_images)
    logger.info(f"Converting {total_pages} pages via Gemini ({model})")

    all_markdown = []
    cost_total = 0.0

    for batch_start in range(0, total_pages, pages_per_batch):
        batch_end = min(batch_start + pages_per_batch, total_pages)
        batch_images = page_images[batch_start:batch_end]

        page_range = f"{batch_start + 1}-{batch_end}"
        logger.info(f"  Processing pages {page_range}/{total_pages}")

        content_parts = [
            {
                "type": "text",
                "text": (
                    f"Convert these academic paper pages (pages {page_range}) to clean Markdown.\n\n"
                    "INSTRUCTIONS:\n"
                    "- Preserve all text content faithfully\n"
                    "- Convert tables to markdown table format\n"
                    "- Mark section headings with ## or ###\n"
                    "- Preserve figure/table captions as '**Figure X:** ...'\n"
                    "- Use inline math notation where applicable\n"
                    "- For multi-column layouts, merge into single-column flow\n"
                    "- Include page markers: <!-- PAGE X -->\n"
                    "- Do NOT summarize or paraphrase — output the full text\n"
                ),
            }
        ]

        for i, img_b64 in enumerate(batch_images):
            page_num = batch_start + i + 1
            content_parts.append({
                "type": "text",
                "text": f"--- Page {page_num} ---",
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                },
            })

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content_parts}],
                max_tokens=8192,
                temperature=0.0,
            )

            md_chunk = response.choices[0].message.content or ""
            all_markdown.append(md_chunk)

            # Track cost
            usage = response.usage
            if usage:
                cost_total += (
                    getattr(usage, "prompt_tokens", 0) * 1.25 / 1_000_000 +
                    getattr(usage, "completion_tokens", 0) * 10.0 / 1_000_000
                )

            # Rate limiting
            if batch_end < total_pages:
                time.sleep(2)

        except Exception as e:
            logger.error(f"Gemini batch {page_range} failed: {e}")
            # Fallback to pdfplumber for this batch
            fallback = _pdfplumber_pages(pdf_path, batch_start, batch_end)
            all_markdown.append(fallback)

    logger.info(f"Gemini conversion cost: ${cost_total:.4f}")
    return "\n\n".join(all_markdown)


def _pdf_to_images(pdf_path: str, max_pages: int = 50, dpi: int = 200) -> List[str]:
    """Convert PDF pages to base64-encoded PNG images."""
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(pdf_path)
        n_pages = min(len(doc), max_pages)
        images_b64 = []

        for i in range(n_pages):
            page = doc[i]
            bitmap = page.render(scale=dpi / 72)
            pil_image = bitmap.to_pil()

            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG", optimize=True)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            images_b64.append(b64)

        doc.close()
        return images_b64

    except ImportError:
        logger.warning("pypdfium2 not available, trying pdf2image")
        return _pdf_to_images_fallback(pdf_path, max_pages, dpi)


def _pdf_to_images_fallback(pdf_path: str, max_pages: int, dpi: int) -> List[str]:
    """Fallback using pdf2image (requires poppler)."""
    try:
        from pdf2image import convert_from_path
        import io

        images = convert_from_path(
            pdf_path, dpi=dpi,
            first_page=1, last_page=max_pages,
        )
        images_b64 = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", optimize=True)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            images_b64.append(b64)
        return images_b64

    except ImportError:
        raise RuntimeError(
            "PDF→image conversion requires pypdfium2 or pdf2image. "
            "Install: pip install pypdfium2"
        )


def _convert_with_pdfplumber(pdf_path: str) -> str:
    """Fallback: convert PDF to markdown using pdfplumber."""
    from src.utils.pdf_decomposer import decompose_pdf, format_segments_for_llm

    segments = decompose_pdf(pdf_path)
    return format_segments_for_llm(segments)


def _pdfplumber_pages(pdf_path: str, start: int, end: int) -> str:
    """Extract specific page range via pdfplumber as markdown."""
    import pdfplumber

    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start, min(end, len(pdf.pages))):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"<!-- PAGE {i + 1} -->\n\n{text}")

    return "\n\n".join(parts)


# ======================================================================
# Batch Conversion
# ======================================================================

def batch_convert_pdfs(
    pdf_dir: str,
    output_dir: str,
    model: str = "google/gemini-3.1-pro-preview",
    max_pages: int = 50,
) -> dict:
    """
    Convert all PDFs in a directory to markdown.

    Returns summary dict with counts and costs.
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"Batch converting {len(pdfs)} PDFs to Markdown")

    results = {"converted": 0, "failed": 0, "cached": 0, "files": []}

    for pdf_path in pdfs:
        md_path = output_dir / f"{pdf_path.stem}.md"

        if md_path.exists():
            results["cached"] += 1
            results["files"].append({
                "pdf": pdf_path.name,
                "status": "cached",
            })
            continue

        try:
            md_text = convert_pdf_to_markdown(
                str(pdf_path),
                cache_dir=str(output_dir / ".cache"),
                model=model,
                max_pages=max_pages,
            )

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)

            results["converted"] += 1
            results["files"].append({
                "pdf": pdf_path.name,
                "md": md_path.name,
                "status": "converted",
                "chars": len(md_text),
            })
            logger.info(f"  {pdf_path.name} -> {md_path.name} ({len(md_text)} chars)")

        except Exception as e:
            results["failed"] += 1
            results["files"].append({
                "pdf": pdf_path.name,
                "status": "failed",
                "error": str(e),
            })
            logger.error(f"  {pdf_path.name} FAILED: {e}")

    logger.info(
        f"Batch complete: {results['converted']} converted, "
        f"{results['cached']} cached, {results['failed']} failed"
    )
    return results


# ======================================================================
# Cache
# ======================================================================

def _file_hash(pdf_path: str) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_cache(pdf_path: str, cache_dir: str) -> Optional[str]:
    cache_file = Path(cache_dir) / f"pdf_md_{_file_hash(pdf_path)}.md"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    return None


def _save_cache(pdf_path: str, cache_dir: str, md_text: str):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"pdf_md_{_file_hash(pdf_path)}.md"
    cache_file.write_text(md_text, encoding="utf-8")
