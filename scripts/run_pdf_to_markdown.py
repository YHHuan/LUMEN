"""
PDF-to-Markdown Batch Conversion — LUMEN v2
=============================================
Convert downloaded PDFs to structured Markdown using Gemini multimodal.

Usage:
    python scripts/run_pdf_to_markdown.py                     # Convert all PDFs
    python scripts/run_pdf_to_markdown.py --file study_01.pdf # Single file
    python scripts/run_pdf_to_markdown.py --fallback-only     # pdfplumber only (no API)
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.pdf_to_markdown import convert_pdf_to_markdown, batch_convert_pdfs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Convert a single PDF file")
    parser.add_argument("--fallback-only", action="store_true",
                        help="Use pdfplumber only (no Gemini API calls)")
    parser.add_argument("--max-pages", type=int, default=50)
    args = parser.parse_args()

    select_project()
    data_dir = Path(get_data_dir())
    dm = DataManager()

    pdf_dir = data_dir / "phase3_screening" / "stage2_fulltext" / "pdfs"
    md_dir = dm.phase_dir("phase3_screening", "markdown")

    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        logger.info("Run Phase 3 Stage 2 (--download) first.")
        return

    if args.file:
        pdf_path = pdf_dir / args.file
        if not pdf_path.exists():
            logger.error(f"File not found: {pdf_path}")
            return

        model = "__pdfplumber__" if args.fallback_only else "google/gemini-3.1-pro-preview"

        if args.fallback_only:
            from src.utils.pdf_to_markdown import _convert_with_pdfplumber
            md_text = _convert_with_pdfplumber(str(pdf_path))
        else:
            md_text = convert_pdf_to_markdown(
                str(pdf_path),
                cache_dir=str(md_dir / ".cache"),
                max_pages=args.max_pages,
            )

        out_path = md_dir / f"{pdf_path.stem}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        logger.info(f"Converted: {out_path} ({len(md_text)} chars)")

    else:
        if args.fallback_only:
            # Manual batch with pdfplumber
            from src.utils.pdf_to_markdown import _convert_with_pdfplumber
            pdfs = sorted(pdf_dir.glob("*.pdf"))
            for pdf_path in pdfs:
                out_path = md_dir / f"{pdf_path.stem}.md"
                if out_path.exists():
                    continue
                try:
                    md_text = _convert_with_pdfplumber(str(pdf_path))
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(md_text)
                    logger.info(f"  {pdf_path.name} -> {out_path.name}")
                except Exception as e:
                    logger.error(f"  {pdf_path.name} FAILED: {e}")
        else:
            results = batch_convert_pdfs(
                str(pdf_dir), str(md_dir),
                max_pages=args.max_pages,
            )
            dm.save("phase3_fulltext", "pdf_to_md_results.json", results,
                    subfolder="markdown")

    pdfs_total = len(list(pdf_dir.glob("*.pdf")))
    mds_total = len(list(md_dir.glob("*.md")))
    print(f"\n  PDF->MD: {mds_total}/{pdfs_total} converted")


if __name__ == "__main__":
    main()
