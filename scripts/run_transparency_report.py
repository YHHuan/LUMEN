"""
Transparency Report — LUMEN v2
================================
Unified transparency & reproducibility report combining:
- Cost & performance metrics
- Inter-rater reliability
- Human-AI agreement
- PRISMA-S compliance
- Reproducibility manifest
- Publication readiness

Usage:
    python scripts/run_transparency_report.py           # Full report
    python scripts/run_transparency_report.py --json     # JSON only
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
from src.utils.reproducibility import compute_config_hash, format_manifest
from src.utils.prisma_s import PrismaSChecker, format_prisma_s_report
from src.utils.readiness_scorer import PublicationReadinessScorer, format_readiness_report
from src.utils.human_review import HumanReviewOverlay
from src.utils.agreement import (
    compute_screening_agreement,
    compute_extraction_consistency,
    format_agreement_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    select_project()
    data_dir = get_data_dir()
    dm = DataManager()

    extracted = dm.load_if_exists("phase4_extraction", "extracted_data.json", default=[])
    n_studies = len(extracted) if isinstance(extracted, list) else 0

    full_report = {}

    # 1. Cost & Performance
    logger.info("Generating cost report...")
    tracker = CostTracker(data_dir)
    tracker.load()
    full_report["cost"] = tracker.full_report(n_studies=n_studies)

    # 2. Reproducibility
    logger.info("Computing reproducibility hash...")
    pico_path = Path(data_dir) / "input" / "pico.yaml"
    full_report["reproducibility"] = compute_config_hash(
        project_root=".",
        pico_path=str(pico_path) if pico_path.exists() else None,
    )

    # 3. PRISMA-S
    logger.info("Checking PRISMA-S compliance...")
    prisma_checker = PrismaSChecker(data_dir)
    full_report["prisma_s"] = prisma_checker.check()

    # 4. Inter-rater reliability
    logger.info("Computing inter-rater reliability...")
    screening_data = dm.load_if_exists("phase3_screening", "screening_results.json", default={})
    irr = {}
    if screening_data:
        screening_list = screening_data.get("results", []) if isinstance(screening_data, dict) else screening_data
        if screening_list:
            irr["screening"] = compute_screening_agreement(screening_list)
    if extracted:
        irr["extraction"] = compute_extraction_consistency(extracted)
    full_report["inter_rater_reliability"] = irr

    # 5. Human-AI agreement
    logger.info("Computing human-AI agreement...")
    overlay = HumanReviewOverlay(data_dir)
    full_report["human_ai_agreement"] = overlay.compute_agreement()

    # 6. Readiness
    logger.info("Scoring publication readiness...")
    scorer = PublicationReadinessScorer(data_dir)
    full_report["readiness"] = scorer.score()

    # Save
    dm.save("transparency", "full_transparency_report.json", full_report)

    if args.json:
        print(json.dumps(full_report, indent=2, ensure_ascii=False, default=str))
    else:
        # Print each section
        print(format_cost_report(full_report["cost"]))
        print()
        print(format_manifest(full_report["reproducibility"]))
        print()
        print(format_prisma_s_report(full_report["prisma_s"]))
        print()
        print(format_agreement_report(
            screening_agreement=irr.get("screening"),
            extraction_consistency=irr.get("extraction"),
            human_ai_agreement=full_report["human_ai_agreement"],
        ))
        print()
        print(format_readiness_report(full_report["readiness"]))

    # Generate plots
    try:
        fig_dir = dm.phase_dir("transparency", "figures")
        generate_cost_plots(full_report["cost"], str(fig_dir))
        logger.info(f"Plots saved to {fig_dir}")
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    # Final summary
    r = full_report["readiness"]
    c = full_report["cost"]["cost_summary"]
    print(f"\n  Readiness: {r['overall_score']}/100 ({r['grade']})")
    print(f"  Total Cost: ${c['total_cost_usd']:.4f}")
    print(f"  Config Hash: {full_report['reproducibility']['config_hash_short']}")
    print(f"  PRISMA-S: {full_report['prisma_s']['compliance_pct']:.0f}% compliant")


if __name__ == "__main__":
    main()
