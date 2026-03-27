"""
Phase 3.0: Pre-screening + Rescue Pipeline — LUMEN v2
========================================================
Context-aware keyword filtering + rescue pipeline for quarantined studies.

v2 improvements:
- Bigram context matching (no more false "protocol" exclusions)
- Quarantine pool for ambiguous matches
- Two-stage rescue: regex positive signals + LLM-lite
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.utils.prescreen import (
    run_prescreen, regex_rescue, llm_lite_rescue, build_positive_signals,
)
from src.config import cfg
from src.agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    select_project()
    dm = DataManager()

    # Load studies
    studies = dm.load("phase2_search", "all_studies.json", subfolder="deduplicated")
    logger.info(f"Loaded {len(studies)} deduplicated studies")

    # Load PICO and rescue keywords
    pico = dm.load("input", "pico.yaml")
    rescue_keywords = dm.load_if_exists(
        "phase1_strategy", "rescue_keywords.json", default={}
    )

    # Step 1: Context-aware pre-screening
    logger.info("Running context-aware pre-screening...")
    prescreen_result = run_prescreen(studies)

    passed = prescreen_result["passed"]
    excluded = prescreen_result["excluded"]
    quarantined = prescreen_result["quarantined"]

    # Step 2: Rescue pipeline
    rescue_mode = cfg.prescreen_settings.get("rescue_mode", "llm_lite")

    if quarantined and rescue_mode != "disabled":
        # Stage A: Regex rescue (zero cost)
        logger.info("Running regex rescue on quarantine pool...")
        rescued_regex, still_quarantined = regex_rescue(quarantined, pico)
        passed.extend(rescued_regex)

        # Stage B: LLM-lite rescue (if enabled)
        if still_quarantined and rescue_mode == "llm_lite":
            logger.info("Running LLM-lite rescue...")
            budget = TokenBudget("phase2_5", limit_usd=cfg.budget("phase2_5"), reset=True)
            rescue_agent = BaseAgent(role_name="rescue_screener", budget=budget)

            pico_summary = pico.get("pico", {})
            summary_str = (
                f"Population: {pico_summary.get('population', '')}, "
                f"Intervention: {pico_summary.get('intervention', '')}, "
                f"Comparison: {pico_summary.get('comparison', '')}, "
                f"Outcome: {pico_summary.get('outcome', '')}"
            )

            rescued_llm, final_excluded = llm_lite_rescue(
                still_quarantined, summary_str, rescue_agent
            )
            passed.extend(rescued_llm)
            excluded.extend(final_excluded)
        else:
            excluded.extend(still_quarantined)

    # Save results
    dm.save("phase2_search", "filtered_studies.json", passed, subfolder="prescreened")
    dm.save("phase2_search", "prescreen_excluded.json", excluded, subfolder="prescreened")

    # Rescue log
    rescue_log = {
        "total_input": len(studies),
        "passed": len(passed),
        "excluded": len(excluded),
        "quarantined_initial": len(quarantined),
        "rescued_regex": len([s for s in passed if s.get("rescue_stage") == "regex"]),
        "rescued_llm": len([s for s in passed if s.get("rescue_stage") == "llm_lite"]),
    }
    dm.save("phase2_search", "prescreen_rescue_log.json", rescue_log, subfolder="prescreened")

    print("\n" + "=" * 50)
    print("  Phase 3.0 Pre-screening Complete")
    print("=" * 50)
    print(f"  Input:           {len(studies)}")
    print(f"  Passed:          {len(passed)}")
    print(f"  Excluded:        {len(excluded)}")
    print(f"  Quarantined:     {len(quarantined)}")
    print(f"  Rescued (regex): {rescue_log['rescued_regex']}")
    print(f"  Rescued (LLM):   {rescue_log['rescued_llm']}")
    print()


if __name__ == "__main__":
    main()
