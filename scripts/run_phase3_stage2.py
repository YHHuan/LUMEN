"""
Phase 3.2: Full-Text Screening — LUMEN v2
============================================
PDF download + full-text review.

Usage:
    python scripts/run_phase3_stage2.py --download            # PDF download only
    python scripts/run_phase3_stage2.py --review              # Full-text screening only
    python scripts/run_phase3_stage2.py --finalize-pending    # Exclude no-PDF studies
    python scripts/run_phase3_stage2.py --all                 # Download + review
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.utils.pdf_downloader import PDFDownloader
from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--finalize-pending", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.download = True
        args.review = True
        args.finalize_pending = True

    select_project()
    dm = DataManager()

    # Load included studies from stage 1
    included = dm.load(
        "phase3_screening", "included_studies.json",
        subfolder="stage1_title_abstract",
    )

    # Include human review queue (temporarily included)
    if dm.exists("phase3_screening", "human_review_queue.json",
                 subfolder="stage1_title_abstract"):
        human_review = dm.load(
            "phase3_screening", "human_review_queue.json",
            subfolder="stage1_title_abstract",
        )
        included.extend(human_review)

    pdf_dir = Path(get_data_dir()) / "phase3_screening" / "stage2_fulltext" / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Download PDFs
    if args.download:
        logger.info(f"Downloading PDFs for {len(included)} studies...")
        with PDFDownloader(str(pdf_dir), email=cfg.unpaywall_email) as downloader:
            for study in included:
                pdf_path = downloader.download(study)
                if pdf_path:
                    study["pdf_path"] = pdf_path

            logger.info(
                f"Download stats: {downloader.stats['success']} success, "
                f"{downloader.stats['failed']} failed, "
                f"{downloader.stats['cached']} cached"
            )

    # Finalize pending — scan pdf_dir for matching PDFs
    if args.finalize_pending:
        if dm.exists("phase3_screening", "fulltext_review.json",
                     subfolder="stage2_fulltext"):
            finalize_source = dm.load("phase3_screening", "fulltext_review.json",
                                       subfolder="stage2_fulltext")
        else:
            finalize_source = included

        # Scan pdf_dir to match studies with downloaded PDFs
        available_pdfs = {p.stem: str(p) for p in pdf_dir.glob("*.pdf")
                          if p.stat().st_size > 1000}
        for study in finalize_source:
            sid = study.get("study_id", "")
            if not study.get("pdf_path") or not Path(study.get("pdf_path", "")).exists():
                if sid in available_pdfs:
                    study["pdf_path"] = available_pdfs[sid]

        with_pdf = [s for s in finalize_source
                    if s.get("pdf_path") and Path(s["pdf_path"]).exists()]
        no_pdf = [s for s in finalize_source
                  if not s.get("pdf_path") or not Path(s["pdf_path"]).exists()]

        logger.info(f"Finalizing: {len(with_pdf)} with PDF, {len(no_pdf)} without")

        dm.save("phase3_screening", "included_with_pdf.json",
                with_pdf, subfolder="stage2_fulltext")
        dm.save("phase3_screening", "excluded_no_pdf.json",
                no_pdf, subfolder="stage2_fulltext")

    # Full-text review (placeholder — uses LLM screening on PDF text)
    if args.review:
        logger.info("Full-text review — using PDF decomposition...")
        # This would integrate with pdf_decomposer and screener agents
        # For now, save the current state
        dm.save("phase3_screening", "fulltext_review.json",
                included, subfolder="stage2_fulltext")

    print("\n  Phase 3.2 complete.")


if __name__ == "__main__":
    main()
