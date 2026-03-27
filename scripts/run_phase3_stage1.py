"""
Phase 3.1: Title/Abstract Screening — LUMEN v2
=================================================
Dual-screener with 5-point confidence scale + Arbiter for firm conflicts.

Usage:
    python scripts/run_phase3_stage1.py                    # Dual screening (default)
    python scripts/run_phase3_stage1.py --single           # Single-agent mode (benchmark)
    python scripts/run_phase3_stage1.py --single --model gemini  # Single with specific model
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.config import cfg
from src.agents.screener import (
    ScreenerAgent, ArbiterAgent,
    run_dual_screening, run_single_screening,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# Model presets for benchmark comparisons
SINGLE_MODEL_PRESETS = {
    "gemini":   "screener_1",     # Gemini 3.1 Pro (default screener_1)
    "gpt":      "screener_2",     # GPT-4.1 Mini (default screener_2)
    "claude":   "arbiter",        # Claude Sonnet 4.6 (reuses arbiter config)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true",
                        help="Single-agent screening mode (benchmark/ablation)")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(SINGLE_MODEL_PRESETS.keys()),
                        help="Model preset for single-agent mode: gemini, gpt, claude")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    # Load pre-screened studies
    if dm.exists("phase2_search", "filtered_studies.json", subfolder="prescreened"):
        studies = dm.load("phase2_search", "filtered_studies.json", subfolder="prescreened")
    else:
        studies = dm.load("phase2_search", "all_studies.json", subfolder="deduplicated")

    # Load screening criteria
    criteria = dm.load("phase1_strategy", "screening_criteria.json")

    logger.info(f"Screening {len(studies)} studies...")

    # Budget
    budget = TokenBudget("phase3_ta", limit_usd=cfg.budget("phase3_ta"), reset=False)

    from src.utils.project import get_data_dir

    if args.single:
        # --- Single-agent mode (hidden benchmark) ---
        role = SINGLE_MODEL_PRESETS.get(args.model, "screener_1")
        screener = ScreenerAgent(role_name=role, budget=budget)
        model_label = args.model or "gemini"

        logger.info(f"Single-agent mode: role={role}, model_preset={model_label}")

        checkpoint_path = str(
            Path(get_data_dir()) / "phase3_screening" / "stage1_title_abstract"
            / f"_checkpoint_single_{model_label}.json"
        )
        results = run_single_screening(studies, criteria, screener,
                                       checkpoint_path=checkpoint_path)

        # Save with model suffix for benchmark comparison
        phase_dir = "phase3_screening"
        subfolder = "stage1_title_abstract"
        suffix = f"_single_{model_label}"

        dm.save(phase_dir, f"included_studies{suffix}.json", results["included"], subfolder=subfolder)
        dm.save(phase_dir, f"excluded_studies{suffix}.json", results["excluded"], subfolder=subfolder)
        dm.save(phase_dir, f"human_review_queue{suffix}.json", results["human_review_queue"], subfolder=subfolder)
        dm.save(phase_dir, f"screening_results{suffix}.json", {
            "results": results["screening_results"],
            "stats": results["stats"],
        }, subfolder=subfolder)

        print("\n" + "=" * 50)
        print(f"  Phase 3.1 Single-Agent Screening ({model_label})")
        print("=" * 50)
        print(f"  Mode:          single ({role})")
        print(f"  Screened:      {len(studies)}")
        print(f"  Included:      {len(results['included'])}")
        print(f"  Excluded:      {len(results['excluded'])}")
        print(f"  Human review:  {len(results['human_review_queue'])}")
        print(f"  Budget:        {budget.summary()['total_cost_usd']}")
        print()
    else:
        # --- Dual screening (default) ---
        screener1 = ScreenerAgent(role_name="screener_1", budget=budget)
        screener2 = ScreenerAgent(role_name="screener_2", budget=budget)
        arbiter = ArbiterAgent(budget=budget)

        checkpoint_path = str(
            Path(get_data_dir()) / "phase3_screening" / "stage1_title_abstract" / "_checkpoint.json"
        )
        results = run_dual_screening(studies, criteria, screener1, screener2, arbiter,
                                     checkpoint_path=checkpoint_path)

        # Save
        phase_dir = "phase3_screening"
        subfolder = "stage1_title_abstract"

        dm.save(phase_dir, "included_studies.json", results["included"], subfolder=subfolder)
        dm.save(phase_dir, "excluded_studies.json", results["excluded"], subfolder=subfolder)
        dm.save(phase_dir, "human_review_queue.json", results["human_review_queue"], subfolder=subfolder)
        dm.save(phase_dir, "screening_results.json", {
            "results": results["screening_results"],
            "stats": results["stats"],
        }, subfolder=subfolder)

        print("\n" + "=" * 50)
        print("  Phase 3.1 T/A Screening Complete")
        print("=" * 50)
        print(f"  Screened:      {len(studies)}")
        print(f"  Included:      {len(results['included'])}")
        print(f"  Excluded:      {len(results['excluded'])}")
        print(f"  Human review:  {len(results['human_review_queue'])}")
        print(f"  Cohen's kappa: {results['stats'].get('cohens_kappa', 'N/A')}")
        print(f"  PABAK:         {results['stats'].get('pabak', 'N/A')}")
        print(f"  Budget:        {budget.summary()['total_cost_usd']}")
        print()


if __name__ == "__main__":
    main()
