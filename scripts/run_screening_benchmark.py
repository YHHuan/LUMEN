"""
Screening Benchmark — LUMEN v2
================================
Compare screening arms with ROC curves and threshold-sweep metrics.

Usage:
    python scripts/run_screening_benchmark.py                         # Extract arms from existing dual + compute
    python scripts/run_screening_benchmark.py --ground-truth gt.csv   # With ground truth for sens/spec
    python scripts/run_screening_benchmark.py --asreview results.csv  # Add ASReview arm
    python scripts/run_screening_benchmark.py --plot-only             # Just plot from saved data
"""

import sys
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.screening_benchmark import (
    ScreeningBenchmark,
    extract_single_arm_from_dual,
    load_asreview_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="Path to ground truth CSV/JSON (study_id, included)")
    parser.add_argument("--asreview", type=str, default=None,
                        help="Path to ASReview results CSV")
    parser.add_argument("--plot-only", action="store_true",
                        help="Just re-plot from saved benchmark data")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    output_dir = str(dm.phase_dir("benchmark", "screening"))
    bench = ScreeningBenchmark()

    if args.plot_only:
        # Load saved results and re-plot
        saved = Path(output_dir) / "screening_benchmark.json"
        if saved.exists():
            from dataclasses import fields
            from src.utils.screening_benchmark import ArmResult
            with open(saved) as f:
                data = json.load(f)
            arm_results = {}
            for name, d in data.items():
                valid_keys = {f.name for f in fields(ArmResult)}
                filtered = {k: v for k, v in d.items() if k in valid_keys}
                arm_results[name] = ArmResult(**filtered)
            bench.plot_roc(arm_results, output_path=str(Path(output_dir) / "roc_curves.png"))
            print(bench.export_table(arm_results))
        else:
            print(f"No saved data at {saved}")
        return

    # Load dual screening results
    phase_dir = "phase3_screening"
    subfolder = "stage1_title_abstract"

    dual_data = dm.load_if_exists(phase_dir, "screening_results.json", subfolder=subfolder, default={})
    dual_results = dual_data.get("results", [])

    if not dual_results:
        logger.error("No dual screening results found. Run Phase 3.1 first.")
        return

    logger.info(f"Loaded {len(dual_results)} dual screening results")

    # --- Arm A: Single Gemini (from dual screener1) ---
    arm_gemini = extract_single_arm_from_dual(
        dual_results, "screener1", "single_gemini", "Gemini 3.1 Pro"
    )
    bench.add_arm("single_gemini", arm_gemini, model="Gemini 3.1 Pro")

    # --- Arm B: Single GPT (from dual screener2) ---
    arm_gpt = extract_single_arm_from_dual(
        dual_results, "screener2", "single_gpt", "GPT-4.1 Mini"
    )
    bench.add_arm("single_gpt", arm_gpt, model="GPT-4.1 Mini")

    # --- Arm C: Single Claude (from dedicated run, if available) ---
    claude_data = dm.load_if_exists(
        phase_dir, "screening_results_single_claude.json",
        subfolder=subfolder, default={}
    )
    if claude_data.get("results"):
        bench.add_arm("single_claude", claude_data["results"],
                       model="Claude Sonnet 4.6")
        logger.info(f"Loaded {len(claude_data['results'])} single Claude results")
    else:
        logger.info("No single Claude results found (run --single --model claude)")

    # --- Arm D: Dual + Arbiter ---
    bench.add_arm("dual_arbiter", dual_results, model="Gemini+GPT+Claude")

    # --- Arm E: ASReview (if provided) ---
    if args.asreview:
        asreview_results = load_asreview_results(args.asreview)
        bench.add_arm("asreview", asreview_results, model="ASReview (AL)")
        logger.info(f"Loaded {len(asreview_results)} ASReview results")

    # --- Ground truth ---
    if args.ground_truth:
        bench.load_ground_truth_from_file(args.ground_truth)
        logger.info(f"Loaded {len(bench.ground_truth)} ground truth labels")
    else:
        # Try to load from data dir
        gt_path = Path(get_data_dir()) / "input" / "ground_truth_screening.json"
        if gt_path.exists():
            bench.load_ground_truth_from_file(str(gt_path))
            logger.info(f"Loaded ground truth from {gt_path}")
        else:
            logger.warning("No ground truth provided — metrics will only show distributions")

    # Compute
    arm_results = bench.compute_all()

    # Display
    table = bench.export_table(arm_results)
    print("\n" + table)

    # ROC data
    roc_data = bench.export_roc_data(arm_results)
    for name, rd in roc_data.items():
        if rd["points"]:
            print(f"\n  ROC [{name}] AUC={rd['auc']:.3f}")
            for p in rd["points"]:
                marker = " ← operating point" if p["threshold"] == 3.5 else ""
                print(f"    threshold={p['threshold']} → Sens={p['sensitivity']:.3f} "
                      f"Spec={p['specificity']:.3f} F1={p['f1']:.3f}{marker}")

    # Save
    bench.save_results(arm_results, output_dir)

    print(f"\n  Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
