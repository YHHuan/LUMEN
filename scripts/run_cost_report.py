"""
Cost & Performance Report — LUMEN v2
======================================
Generate detailed cost transparency report with visualizations.

Usage:
    python scripts/run_cost_report.py                  # Full report + plots
    python scripts/run_cost_report.py --json            # JSON only
    python scripts/run_cost_report.py --no-plots        # Text only, no charts
"""

import sys
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.cost_tracker import CostTracker, format_cost_report, generate_cost_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="JSON output only")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    select_project()
    data_dir = get_data_dir()
    dm = DataManager()

    # Count studies for efficiency metrics
    extracted = dm.load_if_exists("phase4_extraction", "extracted_data.json", default=[])
    n_studies = len(extracted) if isinstance(extracted, list) else 0

    # Generate report
    tracker = CostTracker(data_dir)
    n_entries = tracker.load()

    if n_entries == 0:
        print("\n  No audit log entries found. Run pipeline phases first.")
        return

    logger.info(f"Loaded {n_entries} audit log entries")
    report = tracker.full_report(n_studies=n_studies)

    # Save
    dm.save("transparency", "cost_report.json", report)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_cost_report(report))

    # Generate plots
    if not args.no_plots and not args.json:
        try:
            fig_dir = dm.phase_dir("transparency", "figures")
            generate_cost_plots(report, str(fig_dir))
            print(f"\n  Plots saved to {fig_dir}/")
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()
