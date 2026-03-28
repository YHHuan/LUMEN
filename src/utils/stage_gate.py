"""
Stage Gate Validation — LUMEN v2
=================================
Reusable gate checks for phase transitions.
Called by scripts/validate_stage_transition.py and individual phase scripts.
"""

import logging
from pathlib import Path
from typing import Optional

from src.utils.file_handlers import DataManager
from src.utils.project import get_data_dir

logger = logging.getLogger(__name__)


class GateResult:
    def __init__(self, gate_name: str):
        self.gate_name = gate_name
        self.checks = []
        self.passed = True

    def check(self, label: str, condition: bool, detail: str = ""):
        self.checks.append({"check": label, "passed": condition, "detail": detail})
        if not condition:
            self.passed = False
        return condition

    def log_summary(self):
        passed = sum(1 for c in self.checks if c["passed"])
        status = "PASS" if self.passed else "FAIL"
        logger.info(f"Stage gate [{self.gate_name}]: {status} ({passed}/{len(self.checks)})")
        for c in self.checks:
            if not c["passed"]:
                logger.warning(f"  FAIL: {c['check']} — {c['detail']}")


def validate_phase4_to_5(dm: Optional[DataManager] = None) -> GateResult:
    if dm is None:
        dm = DataManager()
    gate = GateResult("Phase 4 -> Phase 5")

    if not gate.check("extraction file exists",
                      dm.exists("phase4_extraction", "extracted_data.json")):
        gate.log_summary()
        return gate

    extracted = dm.load("phase4_extraction", "extracted_data.json")
    gate.check("extracted studies >= 2", len(extracted) >= 2,
               f"{len(extracted)} studies")

    studies_ok = 0
    for study in extracted:
        # NMA format: outcomes nested inside arms[]
        if "arms" in study and study["arms"]:
            for arm in study["arms"]:
                for o in arm.get("outcomes", []):
                    if (o.get("mean") is not None and o.get("sd") is not None) or \
                       (o.get("events") is not None and o.get("total") is not None):
                        studies_ok += 1
                        break
                else:
                    continue
                break
            continue
        # Pairwise format: outcomes at study root
        for outcome in study.get("outcomes", []):
            # Continuous: mean + sd + n
            has_continuous = False
            for gk in ["intervention_group", "control_group"]:
                g = outcome.get(gk) or {}
                if g.get("mean") is not None and g.get("sd") is not None and g.get("n") is not None:
                    has_continuous = True
                elif g.get("mean") is not None and g.get("se") is not None and g.get("n") is not None:
                    has_continuous = True
            # Binary: events + total per arm
            ig = outcome.get("intervention_group") or {}
            cg = outcome.get("control_group") or {}
            has_binary = (ig.get("events") is not None and ig.get("total") is not None and
                          cg.get("events") is not None and cg.get("total") is not None)
            # Time-to-event: HR + CI
            has_hr = (outcome.get("hr") is not None and
                      outcome.get("hr_ci_lower") is not None and
                      outcome.get("hr_ci_upper") is not None)
            # Pre-computed VE%
            has_ve = (outcome.get("ve_pct") is not None and
                      outcome.get("ve_ci_lower") is not None)
            # Pre-computed effect size + SE or CI
            has_precomputed = (
                (outcome.get("effect_size") is not None and outcome.get("se") is not None) or
                (outcome.get("effect_size") is not None and outcome.get("ci_lower") is not None)
            )
            if has_continuous or has_binary or has_hr or has_ve or has_precomputed:
                studies_ok += 1
                break

    gate.check("studies with computable outcomes >= 2", studies_ok >= 2,
               f"{studies_ok}/{len(extracted)} computable")
    gate.log_summary()
    return gate


def validate_phase5_to_6(dm: Optional[DataManager] = None) -> GateResult:
    if dm is None:
        dm = DataManager()
    gate = GateResult("Phase 5 -> Phase 6")

    if not gate.check("statistical results exist",
                      dm.exists("phase5_analysis", "statistical_results.json")):
        gate.log_summary()
        return gate

    results = dm.load("phase5_analysis", "statistical_results.json")

    # NMA format: per-outcome results instead of single pooled estimate
    if results.get("analysis_type") == "nma":
        nma = results.get("nma", {})
        per_outcome = nma.get("per_outcome", {})
        gate.check("NMA results exist",
                   len(per_outcome) > 0,
                   f"{len(per_outcome)} outcomes analysed")
        gate.check("NMA has successful outcomes",
                   nma.get("n_outcomes_analysed", 0) > 0,
                   f"{nma.get('n_outcomes_analysed', 0)} succeeded")
        gate.check("extraction data available",
                   dm.exists("phase4_extraction", "extracted_data.json"))
    else:
        # Pairwise format
        main = results.get("main", {})
        gate.check("pooled estimate computed",
                   main.get("pooled_effect") is not None,
                   f"effect={main.get('pooled_effect')}")
        gate.check("CI present",
                   main.get("ci_lower") is not None and main.get("ci_upper") is not None)
        gate.check("heterogeneity assessed",
                   main.get("I2") is not None,
                   f"I2={main.get('I2')}%")
        gate.check("extraction data available",
                   dm.exists("phase4_extraction", "extracted_data.json"))
    gate.log_summary()
    return gate
