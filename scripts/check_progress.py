"""
Pipeline Progress Check — LUMEN v2
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir

logging.basicConfig(level=logging.INFO)


def main():
    select_project()
    data_dir = Path(get_data_dir())

    print("\n" + "=" * 60)
    print("  LUMEN v2 — Pipeline Progress")
    print("=" * 60)

    checks = [
        ("Phase 1: Strategy", "phase1_strategy/search_strategy.json"),
        ("Phase 1: Rescue KW", "phase1_strategy/rescue_keywords.json"),
        ("Phase 2: Raw search", "phase2_search/raw"),
        ("Phase 2: Deduplicated", "phase2_search/deduplicated/all_studies.json"),
        ("Phase 3.0: Pre-screened", "phase2_search/prescreened/filtered_studies.json"),
        ("Phase 3.1: Included", "phase3_screening/stage1_title_abstract/included_studies.json"),
        ("Phase 3.1: Excluded", "phase3_screening/stage1_title_abstract/excluded_studies.json"),
        ("Phase 3.1: Human review", "phase3_screening/stage1_title_abstract/human_review_queue.json"),
        ("Phase 3.2: Full-text", "phase3_screening/stage2_fulltext/fulltext_review.json"),
        ("Phase 4: Extracted", "phase4_extraction/extracted_data.json"),
        ("Phase 4: Evidence val.", "phase4_extraction/evidence_validation.json"),
        ("Phase 5: Statistics", "phase5_analysis/statistical_results.json"),
        ("Phase 5: Figures", "phase5_analysis/figures"),
        ("Phase 6: Manuscript", "phase6_manuscript/drafts"),
    ]

    for label, rel_path in checks:
        full_path = data_dir / rel_path
        if full_path.exists():
            if full_path.is_file():
                try:
                    with open(full_path) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        count = f"{len(data)} items"
                    elif isinstance(data, dict):
                        count = f"{len(data)} keys"
                    else:
                        count = "exists"
                except Exception:
                    count = "exists"
                print(f"  [OK] {label}: {count}")
            elif full_path.is_dir():
                files = list(full_path.glob("*"))
                print(f"  [OK] {label}: {len(files)} files")
        else:
            print(f"  [ ] {label}: not yet")

    # Budget summary
    budget_dir = data_dir / ".budget"
    if budget_dir.exists():
        print("\n  Budget Summary:")
        total_cost = 0.0
        for bf in sorted(budget_dir.glob("*_budget.json")):
            with open(bf) as f:
                b = json.load(f)
            cost = b.get("total_cost_usd", 0)
            total_cost += cost
            calls = len(b.get("calls", []))
            print(f"    {bf.stem}: ${cost:.4f} ({calls} calls)")
        print(f"    TOTAL: ${total_cost:.4f}")

    print()


if __name__ == "__main__":
    main()
