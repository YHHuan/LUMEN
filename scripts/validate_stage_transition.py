"""
Stage Gate Validation — LUMEN v2
=================================
Validates that all prerequisites are met before transitioning between phases.

Usage:
    python scripts/validate_stage_transition.py --from 4 --to 5
    python scripts/validate_stage_transition.py --from 5 --to 6
    python scripts/validate_stage_transition.py --all          # Validate all gates
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


class StageGateResult:
    def __init__(self, gate_name: str):
        self.gate_name = gate_name
        self.checks = []
        self.passed = True

    def check(self, label: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        self.checks.append({
            "check": label,
            "status": status,
            "detail": detail,
        })
        if not condition:
            self.passed = False

    def to_dict(self) -> dict:
        return {
            "gate": self.gate_name,
            "passed": self.passed,
            "timestamp": datetime.now().isoformat(),
            "checks": self.checks,
            "pass_count": sum(1 for c in self.checks if c["status"] == "PASS"),
            "total_count": len(self.checks),
        }

    def print_report(self):
        icon = "PASS" if self.passed else "FAIL"
        print(f"\n{'=' * 50}")
        print(f"  Stage Gate: {self.gate_name} [{icon}]")
        print(f"{'=' * 50}")
        for c in self.checks:
            mark = "  [+]" if c["status"] == "PASS" else "  [X]"
            msg = f'{mark} {c["check"]}'
            if c["detail"]:
                msg += f' — {c["detail"]}'
            print(msg)
        passed = sum(1 for c in self.checks if c["status"] == "PASS")
        print(f"\n  Result: {passed}/{len(self.checks)} checks passed")
        print()


def validate_phase1_to_2(dm: DataManager) -> StageGateResult:
    gate = StageGateResult("Phase 1 -> Phase 2 (Strategy -> Search)")
    gate.check("PICO defined", dm.exists("input", "pico.yaml"))
    gate.check("Strategy generated",
               dm.exists("phase1_strategy", "search_strategy.json"))
    if dm.exists("phase1_strategy", "search_strategy.json"):
        strategy = dm.load("phase1_strategy", "search_strategy.json")
        gate.check("Search queries present",
                   bool(strategy.get("search_queries")),
                   f"{len(strategy.get('search_queries', {}))} databases")
        gate.check("Screening criteria defined",
                   bool(strategy.get("screening_criteria")))
    return gate


def validate_phase2_to_3(dm: DataManager) -> StageGateResult:
    gate = StageGateResult("Phase 2 -> Phase 3 (Search -> Screening)")
    gate.check("Search results exist",
               dm.exists("phase2_search", "all_studies.json",
                         subfolder="deduplicated"))
    if dm.exists("phase2_search", "all_studies.json",
                 subfolder="deduplicated"):
        studies = dm.load("phase2_search", "all_studies.json",
                          subfolder="deduplicated")
        gate.check("Studies found",
                   len(studies) > 0,
                   f"{len(studies)} unique studies")
    return gate


def validate_phase3_to_4(dm: DataManager) -> StageGateResult:
    gate = StageGateResult("Phase 3 -> Phase 4 (Screening -> Extraction)")
    gate.check("Screening results exist",
               dm.exists("phase3_screening", "screening_results.json"))
    if dm.exists("phase3_screening", "screening_results.json"):
        results = dm.load("phase3_screening", "screening_results.json")
        included = [s for s in results if s.get("final_decision") == "include"]
        gate.check("Included studies >= 2",
                   len(included) >= 2,
                   f"{len(included)} included")
    return gate


def validate_phase4_to_5(dm: DataManager) -> StageGateResult:
    gate = StageGateResult("Phase 4 -> Phase 5 (Extraction -> Statistics)")

    gate.check("Extraction file exists",
               dm.exists("phase4_extraction", "extracted_data.json"))

    if dm.exists("phase4_extraction", "extracted_data.json"):
        extracted = dm.load("phase4_extraction", "extracted_data.json")
        gate.check("Extracted studies >= 2",
                   len(extracted) >= 2,
                   f"{len(extracted)} studies extracted")

        # Check each study has at least 1 outcome with complete data
        studies_with_data = 0
        incomplete_studies = []
        for study in extracted:
            study_id = study.get("study_id", study.get("canonical_citation", "unknown"))
            has_complete = False
            for outcome in study.get("outcomes", []):
                for group_key in ["intervention_group", "control_group"]:
                    g = outcome.get(group_key, {})
                    if (g.get("mean") is not None and
                        g.get("sd") is not None and
                        g.get("n") is not None):
                        has_complete = True
                        break
                # Also accept pre-computed effect sizes
                if outcome.get("effect_size") is not None and outcome.get("se") is not None:
                    has_complete = True
                if outcome.get("effect_size") is not None and outcome.get("ci_lower") is not None:
                    has_complete = True
                if has_complete:
                    break
            if has_complete:
                studies_with_data += 1
            else:
                incomplete_studies.append(study_id)

        gate.check("Studies with computable outcomes >= 2",
                   studies_with_data >= 2,
                   f"{studies_with_data}/{len(extracted)} have complete outcome data")

        if incomplete_studies:
            gate.check("All studies have data (warning only)",
                       len(incomplete_studies) == 0,
                       f"Missing data: {', '.join(incomplete_studies[:5])}")

    return gate


def validate_phase5_to_6(dm: DataManager) -> StageGateResult:
    gate = StageGateResult("Phase 5 -> Phase 6 (Statistics -> Manuscript)")

    gate.check("Statistical results exist",
               dm.exists("phase5_analysis", "statistical_results.json"))

    if dm.exists("phase5_analysis", "statistical_results.json"):
        results = dm.load("phase5_analysis", "statistical_results.json")

        main = results.get("main", {})
        gate.check("Pooled estimate computed",
                   main.get("pooled_effect") is not None,
                   f"effect={main.get('pooled_effect')}")
        gate.check("Confidence interval present",
                   main.get("ci_lower") is not None and main.get("ci_upper") is not None)
        gate.check("Heterogeneity assessed",
                   main.get("I2") is not None,
                   f"I2={main.get('I2')}%")

        # Check for figures
        fig_dir = Path(get_data_dir()) / "phase5_analysis" / "figures"
        has_figs = fig_dir.exists() and any(fig_dir.iterdir()) if fig_dir.exists() else False
        gate.check("Figures generated", has_figs)

    gate.check("Extraction data available for manuscript",
               dm.exists("phase4_extraction", "extracted_data.json"))

    return gate


GATE_MAP = {
    (1, 2): validate_phase1_to_2,
    (2, 3): validate_phase2_to_3,
    (3, 4): validate_phase3_to_4,
    (4, 5): validate_phase4_to_5,
    (5, 6): validate_phase5_to_6,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", type=int, dest="from_phase")
    parser.add_argument("--to", type=int, dest="to_phase")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    gates_to_run = []
    if args.all:
        gates_to_run = sorted(GATE_MAP.keys())
    elif args.from_phase and args.to_phase:
        key = (args.from_phase, args.to_phase)
        if key not in GATE_MAP:
            print(f"Unknown gate: Phase {args.from_phase} -> {args.to_phase}")
            sys.exit(1)
        gates_to_run = [key]
    else:
        print("Usage: --from X --to Y  or  --all")
        sys.exit(1)

    all_results = []
    all_passed = True

    for key in gates_to_run:
        validator = GATE_MAP[key]
        result = validator(dm)
        result.print_report()
        all_results.append(result.to_dict())
        if not result.passed:
            all_passed = False

    # Save gate validation log
    gate_log_dir = Path(get_data_dir()) / ".meta"
    gate_log_dir.mkdir(parents=True, exist_ok=True)
    gate_log = gate_log_dir / "stage_gate_log.json"
    with open(gate_log, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Gate validation log saved: {gate_log}")

    if all_passed:
        print("All stage gates PASSED")
    else:
        print("Some stage gates FAILED — review issues above")
        sys.exit(1)


if __name__ == "__main__":
    main()
