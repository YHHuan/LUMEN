"""
Phase 4 Data Quality Diagnostics — LUMEN v2
==============================================
Usage:
    python scripts/diagnose_phase4.py          # Report only
    python scripts/diagnose_phase4.py --fix    # Auto-fix SE->SD conversions
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project
from src.utils.file_handlers import DataManager

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    extracted = dm.load("phase4_extraction", "extracted_data.json")

    print("\n" + "=" * 60)
    print("  Phase 4 Data Quality Report")
    print("=" * 60)

    total_outcomes = 0
    complete = 0
    missing_fields = {"mean": 0, "sd": 0, "n": 0}
    low_confidence = 0
    evidence_not_found = 0

    for study in extracted:
        citation = study.get("canonical_citation", study.get("study_id", "?"))
        for outcome in study.get("outcomes", []):
            total_outcomes += 1
            for grp in ["intervention_group", "control_group"]:
                g = outcome.get(grp, {})

                has_mean = g.get("mean") is not None
                has_sd = g.get("sd") is not None
                has_n = g.get("n") is not None

                if has_mean and has_sd and has_n:
                    complete += 1
                else:
                    if not has_mean:
                        missing_fields["mean"] += 1
                    if not has_sd:
                        missing_fields["sd"] += 1
                    if not has_n:
                        missing_fields["n"] += 1

                if g.get("confidence") == "low":
                    low_confidence += 1

                if g.get("evidence_span", "").startswith("NOT FOUND"):
                    evidence_not_found += 1

    print(f"\n  Studies: {len(extracted)}")
    print(f"  Total outcome groups: {total_outcomes * 2}")
    print(f"  Complete (mean+SD+n): {complete}")
    print(f"  Missing mean: {missing_fields['mean']}")
    print(f"  Missing SD: {missing_fields['sd']}")
    print(f"  Missing n: {missing_fields['n']}")
    print(f"  Low confidence: {low_confidence}")
    print(f"  Evidence not found: {evidence_not_found}")

    if args.fix:
        import numpy as np
        fixed = 0
        for study in extracted:
            for outcome in study.get("outcomes", []):
                for grp in ["intervention_group", "control_group"]:
                    g = outcome.get(grp, {})
                    if g.get("sd") is None and g.get("se") is not None and g.get("n"):
                        g["sd"] = round(g["se"] * np.sqrt(g["n"]), 4)
                        g["sd_source"] = "converted_from_se"
                        fixed += 1
        if fixed:
            dm.save("phase4_extraction", "extracted_data.json", extracted)
            print(f"\n  Fixed {fixed} SE->SD conversions")

    print()


if __name__ == "__main__":
    main()
