"""
PRISMA 2020 Flow Diagram Generator — LUMEN v2
================================================
Generates PRISMA flow diagram from pipeline data.

Usage:
    python scripts/export_prisma_diagram.py
    python scripts/export_prisma_diagram.py --included 173  # Override count
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--included", type=int, default=None)
    args = parser.parse_args()

    select_project()
    dm = DataManager()
    data_dir = Path(get_data_dir())

    # Collect counts
    counts = {}

    # Phase 2: search
    if dm.exists("phase2_search", "all_studies.json", subfolder="deduplicated"):
        deduped = dm.load("phase2_search", "all_studies.json", subfolder="deduplicated")
        counts["after_dedup"] = len(deduped)

    # Phase 3.0: pre-screen
    if dm.exists("phase2_search", "prescreen_rescue_log.json", subfolder="prescreened"):
        log = dm.load("phase2_search", "prescreen_rescue_log.json", subfolder="prescreened")
        counts["prescreen_passed"] = log.get("passed", 0)
        counts["prescreen_excluded"] = log.get("excluded", 0)
        counts["rescued_regex"] = log.get("rescued_regex", 0)
        counts["rescued_llm"] = log.get("rescued_llm", 0)

    # Phase 3.1: T/A screening
    if dm.exists("phase3_screening", "included_studies.json",
                 subfolder="stage1_title_abstract"):
        inc = dm.load("phase3_screening", "included_studies.json",
                      subfolder="stage1_title_abstract")
        exc = dm.load("phase3_screening", "excluded_studies.json",
                      subfolder="stage1_title_abstract")
        counts["ta_included"] = len(inc)
        counts["ta_excluded"] = len(exc)

    # Phase 4: extraction
    if dm.exists("phase4_extraction", "extracted_data.json"):
        ext = dm.load("phase4_extraction", "extracted_data.json")
        counts["extracted"] = len(ext)

    # Override
    if args.included:
        counts["final_included"] = args.included

    print("\n  PRISMA 2020 Flow Diagram Data:")
    for k, v in counts.items():
        print(f"    {k}: {v}")
    print()


if __name__ == "__main__":
    main()
