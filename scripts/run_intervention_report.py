"""
Human Intervention Report — LUMEN v2
======================================
Displays summary of all human interventions for cost analysis.

Usage:
    python scripts/run_intervention_report.py
    python scripts/run_intervention_report.py --json
"""

import sys
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project
from src.utils.human_intervention_log import HumanInterventionLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    select_project()
    hil = HumanInterventionLogger()
    summary = hil.summary()

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("\n" + "=" * 50)
    print("  Human Intervention Report")
    print("=" * 50)
    print(f"  Total interventions: {summary['total_interventions']}")

    if summary["total_interventions"] == 0:
        print("  (No interventions recorded)")
        print()
        return

    print(f"  Total time:          {summary['total_time_minutes']:.1f} min")
    print(f"  Unique studies:      {summary['unique_studies']}")

    print("\n  By Phase:")
    for phase, data in summary.get("by_phase", {}).items():
        print(f"    {phase}: {data['count']} interventions, {data['time']:.0f}s")

    print("\n  By Action:")
    for action, data in summary.get("by_action", {}).items():
        print(f"    {action}: {data['count']} ({data['time']:.0f}s)")
    print()


if __name__ == "__main__":
    main()
