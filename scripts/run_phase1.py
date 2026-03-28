"""
Phase 1: Strategy Generation — LUMEN v2
==========================================
PICO -> search strategy (MeSH terms, queries, screening criteria)

Modes:
    python scripts/run_phase1.py                # Auto: interactive if no pico.yaml, else direct
    python scripts/run_phase1.py --interactive   # Force interactive PICO builder
    python scripts/run_phase1.py --direct        # Skip interactive, require existing pico.yaml

v2: Also generates positive rescue keywords for pre-screening.
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.config import cfg
from src.agents.strategist import StrategistAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true",
                        help="Force interactive PICO builder (even if pico.yaml exists)")
    parser.add_argument("--direct", action="store_true",
                        help="Skip interactive, require existing pico.yaml")
    args = parser.parse_args()

    select_project()
    dm = DataManager()
    data_dir = get_data_dir()

    # Budget
    budget = TokenBudget("phase1", limit_usd=cfg.budget("phase1"), reset=True)

    # ─── PICO loading: interactive or direct ─────────────────
    pico_exists = dm.exists("input", "pico.yaml")

    if args.interactive or (not pico_exists and not args.direct):
        # Interactive PICO builder
        from src.utils.pico_builder import run_interactive_pico_builder
        logger.info("Starting interactive PICO builder...")
        pico_data = run_interactive_pico_builder(data_dir, budget=budget)
        if not pico_data:
            logger.error("PICO builder cancelled")
            return
    elif pico_exists:
        pico = dm.load("input", "pico.yaml")
        pico_data = pico.get("pico", pico)
        logger.info(f"Loaded existing pico.yaml")
    else:
        logger.error(
            "No pico.yaml found. Run with --interactive to create one, "
            "or create data/<project>/input/pico.yaml manually."
        )
        return

    logger.info(f"PICO: {pico_data}")

    # Agent
    strategist = StrategistAgent(budget=budget)

    # Generate strategy
    logger.info("Generating search strategy...")
    strategy = strategist.generate_strategy(pico_data)

    if not strategy:
        logger.error("Strategy generation failed")
        return

    dm.save("phase1_strategy", "search_strategy.json", strategy)

    # Extract screening criteria
    criteria = strategy.get("screening_criteria", {})
    dm.save("phase1_strategy", "screening_criteria.json", criteria)

    # v2: Generate rescue keywords
    logger.info("Generating rescue keywords for pre-screening...")
    rescue_keywords = strategist.generate_rescue_keywords(pico_data, strategy)
    dm.save("phase1_strategy", "rescue_keywords.json", rescue_keywords)

    # Summary
    print("\n" + "=" * 50)
    print("  Phase 1 Complete")
    print("=" * 50)
    print(f"  MeSH terms: {len(strategy.get('mesh_terms', []))}")
    print(f"  Search queries: {len(strategy.get('search_queries', {}))}")
    print(f"  Rescue keywords: {sum(len(v) for v in rescue_keywords.values())} total")
    print(f"  Budget: {budget.summary()['total_cost_usd']}")
    print()


if __name__ == "__main__":
    main()
