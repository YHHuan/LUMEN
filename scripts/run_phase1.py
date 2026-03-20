#!/usr/bin/env python3
"""
Phase 1: Research Strategy Planning
=====================================
啟動方式:
  cd meta-analysis-generator
  python -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  cp .env.example .env  # 填入 API keys
  python scripts/run_phase1.py

輸入: data/input/pico.yaml
輸出: data/phase1_strategy/
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project
from src.agents.strategist import StrategistAgent
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.utils.query_syntax import generate_all_queries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    project_dir = select_project()
    dm = DataManager()
    budget = TokenBudget(phase="phase1", limit_usd=5.0)
    
    # === 1. Load PICO ===
    logger.info("📋 Loading PICO input...")
    pico = dm.load("input", "pico.yaml")
    logger.info(f"Population: {pico['pico']['population'][:80]}...")
    logger.info(f"Intervention: {pico['pico']['intervention'][:80]}...")
    
    # === 2. Generate search strategy ===
    logger.info("🤖 Generating search strategy...")
    strategist = StrategistAgent(budget=budget)
    
    strategy = strategist.generate_search_strategy(pico["pico"] | {
        "study_design": pico.get("study_design", {}),
        "date_range": pico.get("date_range", {}),
        "language": pico.get("language", []),
        "subgroup_analyses": pico.get("subgroup_analyses", []),
    })
    
    if "error" in strategy:
        logger.error(f"Strategy generation failed: {strategy}")
        sys.exit(1)
    
    # === 3. Save outputs ===
    dm.save("phase1_strategy", "search_strategy.json", strategy)
    
    # Extract and save MeSH terms separately for review
    mesh_terms = strategy.get("mesh_terms", {})
    dm.save("phase1_strategy", "mesh_terms.json", mesh_terms)
    
    # Save PRISMA protocol
    prisma = strategy.get("prisma_protocol", {})
    dm.save("phase1_strategy", "prisma_protocol.json", prisma)
    
    # Save inclusion/exclusion criteria
    criteria = {
        "inclusion_criteria": strategy.get("inclusion_criteria", []),
        "exclusion_criteria": strategy.get("exclusion_criteria", []),
    }
    dm.save("phase1_strategy", "screening_criteria.json", criteria)

    # Save study design filter (used by Phase 2.5 pre-screen)
    design_filter = strategy.get("study_design_filter", {})
    if design_filter:
        dm.save("phase1_strategy", "study_design_filter.json", design_filter)
        logger.info(f"📋 Study design filter: mode={design_filter.get('filter_mode', 'N/A')}, "
                     f"designs={design_filter.get('required_designs', [])}")

    # Save extraction guidance (used by Phase 4 extractor)
    extraction_guidance = strategy.get("extraction_guidance", {})
    if extraction_guidance:
        dm.save("phase1_strategy", "extraction_guidance.json", extraction_guidance)
        n_measures = len(extraction_guidance.get("outcome_measures", []))
        logger.info(f"📋 Extraction guidance: {n_measures} outcome measures, "
                     f"timepoint={extraction_guidance.get('preferred_timepoint', 'N/A')}")
    
    # === 4. Validate MeSH terms and apply corrections back to queries ===
    if mesh_terms:
        logger.info("🔍 Validating MeSH terms...")
        validation = strategist.validate_mesh_terms(mesh_terms)
        dm.save("phase1_strategy", "mesh_validation.json", validation)

        # Extract corrected terms from validation and rebuild queries if any
        # changes were found (status == "corrected").
        corrected_pop = []
        corrected_int = []
        any_correction = False

        for entry in validation.get("validated_terms", {}).get("population", []):
            t = entry.get("term", "")
            if entry.get("status") == "corrected":
                logger.warning(f"MeSH correction (population): '{mesh_terms.get('population', [])}' → '{t}'  ({entry.get('note','')})")
                any_correction = True
            if t:
                corrected_pop.append(t)

        for entry in validation.get("validated_terms", {}).get("intervention", []):
            t = entry.get("term", "")
            if entry.get("status") == "corrected":
                logger.warning(f"MeSH correction (intervention): → '{t}'  ({entry.get('note','')})")
                any_correction = True
            if t:
                corrected_int.append(t)

        if any_correction and (corrected_pop or corrected_int):
            logger.info("🔄 Rebuilding queries with corrected MeSH terms...")
            flat = strategy.get("_flat_terms", {})
            date_range = pico.get("date_range", {})
            rebuilt_queries = generate_all_queries(
                population_terms=flat.get("population", []),
                intervention_terms=flat.get("intervention", []),
                mesh_population=corrected_pop or mesh_terms.get("population", []),
                mesh_intervention=corrected_int or mesh_terms.get("intervention", []),
                date_start=str(date_range.get("start", "2000"))[:4],
                date_end=str(date_range.get("end", "2025"))[:4],
                languages=pico.get("language", ["English"]),
                study_types=pico.get("study_design", {}).get("include", []),
                broad_terms=set(flat.get("broad_terms", [])),
            )
            strategy["search_queries"] = rebuilt_queries
            strategy["mesh_terms"] = {
                "population": corrected_pop,
                "intervention": corrected_int,
            }
            dm.save("phase1_strategy", "search_strategy.json", strategy)
            dm.save("phase1_strategy", "mesh_terms.json", strategy["mesh_terms"])
            logger.info("✅ Queries rebuilt with validated MeSH terms.")
    
    # === 5. Save Cochrane and Embase manual search instructions ===
    queries = strategy.get("search_queries", {})
    cochrane_txt = queries.get("cochrane", "")
    embase_txt   = queries.get("embase_ovid", "")
    if cochrane_txt:
        Path(project_dir, "phase1_strategy/manual_search_cochrane.txt").write_text(
            cochrane_txt, encoding="utf-8"
        )
        logger.info("📄 Cochrane search strategy saved.")
    if embase_txt:
        Path(project_dir, "phase1_strategy/manual_search_embase.txt").write_text(
            embase_txt, encoding="utf-8"
        )
        logger.info("📄 Embase/Ovid search strategy saved.")

    # === Summary ===
    print("\n" + "="*60)
    print("✅ Phase 1 Complete!")
    print("="*60)
    print(f"\nOutputs saved to: data/phase1_strategy/")
    print(f"\n📌 IMPORTANT: Please review these files before proceeding:")
    print(f"   1. mesh_terms.json            — Confirm MeSH terms are correct")
    print(f"   2. search_strategy.json       — Review auto-run queries (PubMed/Europe PMC/Scopus)")
    print(f"   3. screening_criteria.json    — Review inclusion/exclusion criteria")
    print(f"   4. manual_search_cochrane.txt — Copy-paste strategy for Cochrane Library")
    print(f"   5. manual_search_embase.txt   — Numbered Ovid strategy for Embase")
    print(f"\n💰 Token budget: {json.dumps(budget.summary(), indent=2)}")
    print(f"\nNext step: python scripts/run_phase2.py")


if __name__ == "__main__":
    main()
