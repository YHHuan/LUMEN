"""
Publication Readiness Check — LUMEN v2
=======================================
Evaluate manuscript and pipeline readiness for publication.

Usage:
    python scripts/run_readiness_check.py                # Full report
    python scripts/run_readiness_check.py --json          # JSON output only
    python scripts/run_readiness_check.py --cost-only     # Cost summary only
"""

import sys
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.readiness_scorer import (
    PublicationReadinessScorer,
    format_readiness_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    parser.add_argument("--cost-only", action="store_true", help="Show cost summary only")
    args = parser.parse_args()

    select_project()
    data_dir = get_data_dir()
    dm = DataManager()

    scorer = PublicationReadinessScorer(data_dir)
    report = scorer.score()

    # Save report
    dm.save("quality_assessment", "readiness_report.json", report)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cost_only:
        cost = report.get("cost_summary", {})
        print(f"\n  Total API Cost: ${cost.get('total_cost_usd', 0):.4f}")
        print(f"  API Calls: {cost.get('total_api_calls', 0)}")
        if cost.get("cost_per_phase"):
            print("\n  By Phase:")
            for phase, c in cost["cost_per_phase"].items():
                print(f"    {phase}: ${c:.4f}")
    else:
        print(format_readiness_report(report))


if __name__ == "__main__":
    main()
