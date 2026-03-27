"""
Phase 1: Strategy Generation — LUMEN v2
==========================================
PICO -> search strategy (MeSH terms, queries, screening criteria)
v2: Also generates positive rescue keywords for pre-screening.
"""

import sys
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
    select_project()
    dm = DataManager()

    # Load PICO
    pico = dm.load("input", "pico.yaml")
    pico_data = pico.get("pico", pico)

    logger.info(f"PICO: {pico_data}")

    # Budget
    budget = TokenBudget("phase1", limit_usd=cfg.budget("phase1"), reset=True)

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
