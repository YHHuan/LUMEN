"""
Phase 4.5: Analysis Planner — LUMEN v2
========================================
Profiles extracted data and proposes a structured analysis plan
for Phase 5 to execute (instead of blindly pooling everything).

Usage:
    python scripts/run_phase4_5.py                    # Interactive terminal review
    python scripts/run_phase4_5.py --auto-approve      # Auto-approve for validation/ablation
    python scripts/run_phase4_5.py --profile-only      # Show data profile, no LLM
    python scripts/run_phase4_5.py --from-plan plan.yaml  # Load existing plan, skip to review
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
from src.utils.analysis_planner import (
    profile_extracted_data,
    propose_analysis_plan,
    display_plan_terminal,
    save_analysis_plan,
    load_analysis_plan,
    harmonize_outcomes_and_interventions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-approve", action="store_true",
                        help="Skip human review, auto-approve the LLM plan")
    parser.add_argument("--profile-only", action="store_true",
                        help="Show data profile without generating plan")
    parser.add_argument("--from-plan", type=str, default=None,
                        help="Load existing plan YAML instead of generating")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    # Load extracted data
    extracted = dm.load("phase4_extraction", "extracted_data.json")
    logger.info(f"Loaded {len(extracted)} extracted studies")

    # Load PICO
    pico = dm.load_if_exists("input", "pico.yaml", default={})

    # Step 0.5: Harmonize outcome & intervention names via LLM
    from src.agents.base_agent import BaseAgent
    harmonize_budget = TokenBudget("phase4_5_harmonize", limit_usd=1.0, reset=True)
    harmonize_agent = BaseAgent(role_name="strategist", budget=harmonize_budget)

    logger.info("Harmonizing outcome and intervention names...")
    harmonization = harmonize_outcomes_and_interventions(extracted, pico, harmonize_agent)
    dm.save("phase4_5_planning", "harmonization_map.json", harmonization)

    n_out = len(harmonization.get("outcome_map", {}))
    n_int = len(harmonization.get("intervention_map", {}))
    logger.info(f"Harmonization complete: {n_out} outcome mappings, {n_int} intervention mappings")

    # Save harmonized extracted data so Phase 5 can read the updated names
    dm.save("phase4_extraction", "extracted_data.json", extracted)
    logger.info("Saved harmonized extracted data back to phase4_extraction/extracted_data.json")

    # Step 1: Profile (now using harmonized names)
    profile = profile_extracted_data(extracted, pico)

    print("\n" + "=" * 60)
    print("  Phase 4.5: Data Profile")
    print("=" * 60)
    print(f"\n  Studies extracted:    {profile['n_studies']}")
    print(f"  With outcomes:       {profile['n_with_outcomes']}")
    print(f"\n  Interventions (broad):")
    for name, count in list(profile.get("broad_interventions", profile["interventions"]).items())[:10]:
        print(f"    {name}: {count}")
    print(f"\n  Outcomes (broad):")
    for name, count in list(profile.get("broad_outcomes", profile["outcomes"]).items())[:10]:
        print(f"    {name}: {count}")
    print(f"\n  Feasible combinations (k >= 3, broad grouping):")
    feasible_3 = [f for f in profile["feasible_analyses"] if f["k"] >= 3]
    for fa in feasible_3:
        print(f"    {fa['intervention']} × {fa['outcome']}: k={fa['k']} {fa.get('study_ids', [])}")
    if not feasible_3:
        print("    (none with k >= 3)")
    print()

    # Save profile
    dm.save("phase4_5_planning", "data_profile.json", profile)

    if args.profile_only:
        print("  Profile saved. Use --auto-approve or interactive mode to generate plan.")
        return

    # Step 2: Generate or load plan
    if args.from_plan:
        plan = load_analysis_plan(args.from_plan)
        if not plan:
            logger.error(f"Could not load plan from {args.from_plan}")
            return
        logger.info(f"Loaded plan from {args.from_plan}")
    else:
        # Use LLM to propose plan
        budget = TokenBudget("phase4_5", limit_usd=1.0, reset=True)
        planner_agent = BaseAgent(role_name="strategist", budget=budget)

        logger.info("Generating analysis plan via LLM...")
        plan = propose_analysis_plan(profile, pico, planner_agent)

        analyses = plan.get("analyses", [])
        logger.info(f"LLM proposed {len(analyses)} primary analyses")

    # Step 3: Human review or auto-approve
    if args.auto_approve:
        plan["human_approved"] = True
        print("  Auto-approved analysis plan.")
    else:
        plan = display_plan_terminal(plan)

    # Step 4: Save
    plan_path = str(Path(get_data_dir()) / "phase4_5_planning" / "analysis_plan.yaml")
    save_analysis_plan(plan, plan_path)

    # Also save as JSON for programmatic access
    dm.save("phase4_5_planning", "analysis_plan.json", plan)

    # Summary
    approved = plan.get("human_approved", False)
    analyses = plan.get("analyses", [])
    print("\n" + "=" * 60)
    print("  Phase 4.5 Analysis Planning Complete")
    print("=" * 60)
    print(f"  Analyses:    {len(analyses)}")
    print(f"  Subgroups:   {len(plan.get('subgroup_analyses', []))}")
    print(f"  Sensitivity: {len(plan.get('sensitivity_analyses', []))}")
    print(f"  Figures:     {len(plan.get('figures', []))}")
    print(f"  Approved:    {'Yes' if approved else 'No (Phase 5 will skip planned analyses)'}")
    print(f"  Plan saved:  {plan_path}")
    print()

    if not approved:
        print("  To approve, re-run with --auto-approve or edit the YAML manually")
        print("  and set 'human_approved: true'")
        print()


if __name__ == "__main__":
    main()
