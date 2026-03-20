#!/usr/bin/env python3
"""
Phase 3 Stage 1: Title/Abstract Screening — v5
================================================
  python scripts/run_phase3_stage1.py
  python scripts/run_phase3_stage1.py --resume

v5: 自動偵測 Phase 2.5 prescreened data

前置需求: Phase 2 (+ optional Phase 2.5)
輸入: prescreened/filtered_studies.json 或 deduplicated/all_studies.json
輸出: data/phase3_screening/stage1_title_abstract/
"""

import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project
from src.agents.screener import run_dual_screening
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.utils.langfuse_client import log_phase_start, log_phase_end

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Stage 1: Title/Abstract Screening")
    parser.add_argument("--reset-budget", action="store_true",
                        help="Reset accumulated token spend before running")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    # === Prerequisite checks ===
    has_prescreened = dm.exists("phase2_search", "filtered_studies.json", subfolder="prescreened")
    has_deduped     = dm.exists("phase2_search", "all_studies.json", subfolder="deduplicated")
    has_criteria    = dm.exists("phase1_strategy", "screening_criteria.json")

    if not has_criteria:
        logger.error(
            "❌ screening_criteria.json not found. "
            "Run Phase 1 first: python scripts/run_phase1.py"
        )
        sys.exit(1)

    if not has_prescreened and not has_deduped:
        logger.error(
            "❌ No study list found (checked prescreened/ and deduplicated/). "
            "Run Phase 2 first: python scripts/run_phase2.py"
        )
        sys.exit(1)

    # === Load inputs ===
    logger.info("📋 Loading studies and criteria...")

    if has_prescreened:
        logger.info("✅ Found pre-screened data from Phase 2.5")
        studies = dm.load("phase2_search", "filtered_studies.json", subfolder="prescreened")
    else:
        logger.info("ℹ️  Using raw deduplicated data (Phase 2.5 was skipped)")
        studies = dm.load("phase2_search", "all_studies.json", subfolder="deduplicated")

    criteria = dm.load("phase1_strategy", "screening_criteria.json")
    
    logger.info(f"Loaded {len(studies)} studies for screening")

    # === Langfuse: phase start stamp ===
    import yaml
    from pathlib import Path as _Path
    _models_cfg = yaml.safe_load((_Path(__file__).parent.parent / "config/models.yaml").read_text())
    _lf_span = log_phase_start("phase3_stage1", {
        "input_study_count": len(studies),
        "screener1_model":   _models_cfg["models"]["screener1"]["model_id"],
        "screener2_model":   _models_cfg["models"]["screener2"]["model_id"],
        "arbiter_model":     _models_cfg["models"]["arbiter"]["model_id"],
        "budget_usd":        float(_models_cfg.get("token_budgets", {}).get("phase3_ta", 10.0)),
        "data_source":       "prescreened" if has_prescreened else "deduplicated",
    })

    # === Budget ===
    from src.config import cfg
    budget = TokenBudget(phase="phase3_stage1", limit_usd=cfg.budget_phase3_ta,
                         reset=args.reset_budget)

    # === Run dual screening ===
    _phase_start = time.time()
    results = run_dual_screening(studies, criteria, budget=budget)
    _elapsed = time.time() - _phase_start
    
    # === Save ===
    dm.save("phase3_screening", "screening_results.json", results,
            subfolder="stage1_title_abstract")

    included_ids = set(results["included"])
    excluded_ids = set(results["excluded"])
    human_queue_ids = set(results.get("human_review_queue", []))

    included_studies = [s for s in studies if s["study_id"] in included_ids]
    dm.save("phase3_screening", "included_studies.json", included_studies,
            subfolder="stage1_title_abstract")

    excluded_studies = [s for s in studies if s["study_id"] in excluded_ids]
    dm.save("phase3_screening", "excluded_studies.json", excluded_studies,
            subfolder="stage1_title_abstract")

    # Human review queue — studies where screeners gave conflicting signals and
    # at least one was "undecided". Temporarily included; edit this file then
    # re-run deduplication or manually move to excluded if appropriate.
    if human_queue_ids:
        human_queue_studies = [s for s in studies if s["study_id"] in human_queue_ids]
        dm.save("phase3_screening", "human_review_queue.json", human_queue_studies,
                subfolder="stage1_title_abstract")
        logger.info(
            f"⚠️  {len(human_queue_studies)} studies need human review — "
            f"saved to stage1_title_abstract/human_review_queue.json"
        )

    # === Save validation metrics (#10) ===
    _bsum_metrics = budget.summary()
    validation_metrics = {
        "phase":                 "phase3_stage1",
        "run_date":              datetime.now().isoformat(),
        "total_screened":        results["total_screened"],
        "included":              len(results["included"]),
        "excluded":              len(results["excluded"]),
        "human_review_queue":    len(results.get("human_review_queue", [])),
        "firm_conflicts":        len(results.get("conflicts_firm", [])),
        "side_agreement_rate":   round(results.get("agreement_rate", 0), 4),
        "exact_agreement_rate":  round(results.get("exact_agreement_rate", 0), 4),
        "cohens_kappa":          round(results.get("cohens_kappa", 0), 4),
        "duration_seconds":      round(_elapsed, 1),
        "duration_minutes":      round(_elapsed / 60, 1),
        "cost_usd":              _bsum_metrics.get("total_cost_usd", 0),
        "total_llm_calls":       _bsum_metrics.get("total_calls", 0),
    }
    dm.save("phase3_screening", "validation_metrics.json", validation_metrics,
            subfolder="stage1_title_abstract")

    # === Summary ===
    print("\n" + "=" * 60)
    print("Phase 3 Stage 1 Complete!")
    print("=" * 60)
    print(f"  Total screened:         {results['total_screened']}")
    print(f"  Included:               {len(results['included'])}")
    print(f"  Excluded:               {len(results['excluded'])}")
    print(f"  → Human review queue:   {len(results.get('human_review_queue', []))} "
          f"(undecided conflicts, temporarily included)")
    print(f"  Firm conflicts → Arbiter: {len(results.get('conflicts_firm', []))}")
    print(f"  Side agreement rate:    {results['agreement_rate']:.1%}")
    print(f"  Exact agreement rate:   {results.get('exact_agreement_rate', 0):.1%}")
    print(f"  Cohen's κ (binary):     {results.get('cohens_kappa', 0):.3f}")
    print(f"  ⏱️  Duration:            {_elapsed/60:.1f} min ({_elapsed:.0f}s)")
    print(f"\n  Token budget: {json.dumps(budget.summary(), indent=2)}")
    print(f"\nNext: python scripts/run_phase3_stage2.py --download")

    # === Langfuse: phase end stamp ===
    _bsum = budget.summary()
    log_phase_end(_lf_span, "phase3_stage1", {
        "total_screened":      results["total_screened"],
        "included":            len(results["included"]),
        "excluded":            len(results["excluded"]),
        "human_review_queue":  len(results.get("human_review_queue", [])),
        "arbiter_calls":       len(results.get("conflicts_firm", [])),
        "cohens_kappa":        results.get("cohens_kappa"),
        "side_agreement_rate": results.get("agreement_rate"),
        "total_cost_usd":      _bsum.get("total_cost_usd", 0),
        "cache_savings_usd":   _bsum.get("cache_savings_usd", 0),
    })


if __name__ == "__main__":
    main()
