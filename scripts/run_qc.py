"""
QC Checkpoint Runner — LUMEN v2
=================================
Run QC checks on pipeline output for any phase or all phases.

Usage:
    python scripts/run_qc.py                    # All phases
    python scripts/run_qc.py --phase extraction # Specific phase
    python scripts/run_qc.py --phase all        # Explicit all
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project
from src.utils.file_handlers import DataManager
from src.utils.qc_engine import QCEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

PHASE_MAP = {
    "prescreen": "phase2_5_prescreen",
    "screening": "phase3_screening",
    "fulltext": "phase3_3_fulltext",
    "extraction": "phase4_extraction",
    "nma": "phase5_nma",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all",
                        choices=list(PHASE_MAP.keys()) + ["all"],
                        help="Which phase to QC check")
    args = parser.parse_args()

    select_project()
    dm = DataManager()
    pico = dm.load_if_exists("input", "pico.yaml", default={})
    qc = QCEngine(dm, pico)

    phases = list(PHASE_MAP.values()) if args.phase == "all" else [PHASE_MAP[args.phase]]
    all_issues = []

    for phase in phases:
        issues = qc.run_all_checks(phase)
        if issues:
            qc.present_issues(issues, phase)
            all_issues.extend(issues)

    # Summary
    if not all_issues:
        print(f"\n{'=' * 60}")
        print(f"  ✓ All QC checkpoints passed — no issues found.")
        print(f"{'=' * 60}\n")
    else:
        critical = sum(1 for i in all_issues if i.severity == "critical")
        warnings = sum(1 for i in all_issues if i.severity == "warning")
        info = sum(1 for i in all_issues if i.severity == "info")
        print(f"\n{'=' * 60}")
        print(f"  QC Summary: {critical} critical, {warnings} warnings, {info} info")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
