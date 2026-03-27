"""
Quality Assessment: RoB-2 / ROBINS-I + GRADE — LUMEN v2
========================================================
Run Risk of Bias assessment and GRADE evidence certainty.

Auto-routes each study based on design:
  - RCT studies → RoB-2 (Cochrane Risk of Bias 2)
  - Non-RCT studies → ROBINS-I (Risk Of Bias In Non-randomized Studies)

Usage:
    python scripts/run_quality_assessment.py                  # Full assessment
    python scripts/run_quality_assessment.py --rob2-only      # RoB-2 only (skip GRADE)
    python scripts/run_quality_assessment.py --grade-only      # GRADE only (skip RoB)
    python scripts/run_quality_assessment.py --skip-llm        # No LLM calls (auto only)
"""

import sys
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.config import cfg
from src.utils.rob2 import RoB2Assessor, build_rob2_summary
from src.utils.robins_i import (
    classify_study_design,
    RobinsIAssessor,
    build_robins_i_summary,
)
from src.utils.grade import (
    GRADEAssessor,
    build_grade_evidence_profile,
    format_grade_table,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rob2-only", action="store_true")
    parser.add_argument("--grade-only", action="store_true")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Auto-assessment only, no LLM calls")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    # Load data
    extracted = dm.load_if_exists("phase4_extraction", "extracted_data.json", default=[])
    stats = dm.load_if_exists("phase5_analysis", "statistical_results.json", default={})
    pico = dm.load_if_exists("input", "pico.yaml", default={})

    if not extracted:
        logger.error("No extracted data found. Run Phase 4 first.")
        return

    budget = TokenBudget("quality_assessment", limit_usd=cfg.budget("phase5"), reset=True)

    # ── Risk of Bias Assessment (auto-routed) ─────────

    rob2_summary = None
    robins_i_summary = None

    if not args.grade_only:
        # Split studies by design
        rct_studies = []
        nrct_studies = []
        for study in extracted:
            design = classify_study_design(study)
            if design == "RCT":
                rct_studies.append(study)
            else:
                nrct_studies.append(study)

        logger.info(
            f"Study design classification: {len(rct_studies)} RCT, "
            f"{len(nrct_studies)} non-RCT"
        )

        # ── RoB-2 for RCTs ──
        rob2_assessments = []
        if rct_studies:
            logger.info(f"Running RoB-2 on {len(rct_studies)} RCT studies...")
            if args.skip_llm:
                from src.utils.rob2 import create_empty_assessment as rob2_empty
                for study in rct_studies:
                    a = rob2_empty(study.get("study_id", ""))
                    a["overall_judgment"] = "No information"
                    a["overall_justification"] = "Awaiting manual assessment"
                    rob2_assessments.append(a)
            else:
                assessor = RoB2Assessor(budget=budget)
                rob2_assessments = assessor.assess_batch(rct_studies)

            rob2_summary = build_rob2_summary(rob2_assessments)
            dm.save("quality_assessment", "rob2_assessments.json", rob2_assessments)
            dm.save("quality_assessment", "rob2_summary.json", rob2_summary)

            overall = rob2_summary["overall_counts"]
            logger.info(
                f"RoB-2 Summary: {overall.get('Low risk', 0)} low, "
                f"{overall.get('Some concerns', 0)} some concerns, "
                f"{overall.get('High risk', 0)} high risk"
            )

        # ── ROBINS-I for non-RCTs ──
        robins_i_assessments = []
        if nrct_studies:
            logger.info(f"Running ROBINS-I on {len(nrct_studies)} non-RCT studies...")
            if args.skip_llm:
                from src.utils.robins_i import create_empty_assessment as robins_empty
                for study in nrct_studies:
                    a = robins_empty(study.get("study_id", ""))
                    a["overall_judgment"] = "No information"
                    a["overall_justification"] = "Awaiting manual assessment"
                    robins_i_assessments.append(a)
            else:
                assessor = RobinsIAssessor(budget=budget)
                robins_i_assessments = assessor.assess_batch(nrct_studies)

            robins_i_summary = build_robins_i_summary(robins_i_assessments)
            dm.save("quality_assessment", "robins_i_assessments.json", robins_i_assessments)
            dm.save("quality_assessment", "robins_i_summary.json", robins_i_summary)

            overall = robins_i_summary["overall_counts"]
            logger.info(
                f"ROBINS-I Summary: {overall.get('Low', 0)} low, "
                f"{overall.get('Moderate', 0)} moderate, "
                f"{overall.get('Serious', 0)} serious, "
                f"{overall.get('Critical', 0)} critical"
            )

        if not rct_studies and not nrct_studies:
            logger.warning("No studies to assess.")

    # ── GRADE Assessment ──────────────────────────────

    if not args.rob2_only:
        logger.info("Running GRADE evidence certainty assessment...")

        # Load RoB summaries if not just computed
        if rob2_summary is None:
            rob2_summary = dm.load_if_exists(
                "quality_assessment", "rob2_summary.json", default={}
            )
        if robins_i_summary is None:
            robins_i_summary = dm.load_if_exists(
                "quality_assessment", "robins_i_summary.json", default={}
            )
        # Merge RoB summaries for GRADE input
        combined_rob_summary = rob2_summary or robins_i_summary or {}

        # Determine outcomes to assess
        outcomes = _extract_outcomes(extracted, pico)
        if not outcomes:
            outcomes = ["Primary outcome"]

        logger.info(f"GRADE outcomes: {outcomes}")

        # LLM agent for indirectness (optional)
        from src.agents.base_agent import BaseAgent
        llm_agent = None
        if not args.skip_llm:
            llm_agent = BaseAgent(role_name="statistician", budget=budget)

        grade_assessor = GRADEAssessor(llm_agent=llm_agent)
        grade_results = grade_assessor.assess_all_outcomes(
            outcomes=outcomes,
            statistical_results=stats,
            rob2_summary=combined_rob_summary,
            pico=pico,
            n_studies=len(extracted),
            study_design=_infer_study_design(extracted),
        )

        # Build evidence profile
        profile = build_grade_evidence_profile(grade_results, stats)

        # Save
        dm.save("quality_assessment", "grade_assessments.json", grade_results)
        dm.save("quality_assessment", "grade_evidence_profile.json", profile)

        # Save markdown table
        grade_md = format_grade_table(profile)
        dm.save("quality_assessment", "grade_summary.md", grade_md)

        # Print
        for grade in grade_results:
            logger.info(
                f"  {grade['outcome']}: {grade['final_certainty']} certainty"
            )

        print(f"\n{grade_md}")

    print(f"\n  Quality assessment complete. Budget: {budget.summary()['total_cost_usd']}")


def _extract_outcomes(extracted: list, pico: dict) -> list:
    """Extract unique outcome names for GRADE assessment.
    Priority: analysis plan outcomes > PICO > extracted data.
    """
    # Priority 1: Use analysis plan outcomes (aligned with Phase 5 analyses)
    from src.utils.file_handlers import DataManager
    dm = DataManager()
    plan = dm.load_if_exists("phase4_5_planning", "analysis_plan.yaml", default=None)
    if plan and plan.get("analyses"):
        plan_outcomes = []
        for analysis in plan["analyses"]:
            outcome_label = analysis.get("outcome") or analysis.get("label", "")
            if outcome_label and outcome_label not in plan_outcomes:
                plan_outcomes.append(outcome_label)
        if plan_outcomes:
            return plan_outcomes[:5]

    # Priority 2: From PICO
    outcomes = set()
    pico_outcome = pico.get("outcome")
    if pico_outcome:
        if isinstance(pico_outcome, str):
            outcomes.add(pico_outcome)
        elif isinstance(pico_outcome, dict):
            for v in pico_outcome.values():
                if isinstance(v, str):
                    outcomes.add(v)
    if outcomes:
        return sorted(outcomes)[:5]

    # Priority 3: From extracted data (broad names preferred)
    from collections import Counter
    measure_counts = Counter()
    for study in extracted:
        for outcome in study.get("outcomes", []):
            measure = outcome.get("measure_broad") or outcome.get("measure", "")
            if measure:
                measure_counts[measure] += 1

    return [m for m, _ in measure_counts.most_common(5)]


def _infer_study_design(extracted: list) -> str:
    """Infer predominant study design."""
    designs = [s.get("study_design", "").lower() for s in extracted if s.get("study_design")]

    rct_keywords = ("rct", "randomized", "randomised", "random")
    n_rct = sum(1 for d in designs if any(k in d for k in rct_keywords))

    return "RCT" if n_rct > len(designs) / 2 else "observational"


if __name__ == "__main__":
    main()
