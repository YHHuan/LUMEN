"""
Phase 3.3: Full-Text Screening — LUMEN v2
============================================
LLM-based full-text screening using PDF content + PICO criteria.
Runs AFTER Phase 3.2 (PDF download). Filters studies that passed T/A
screening but don't actually meet inclusion criteria on full-text review.

Uses Claude Sonnet 4.6 for stricter PICO verification against full text.

Usage:
    python scripts/run_phase3_3_fulltext_screen.py                # Full-text screen
    python scripts/run_phase3_3_fulltext_screen.py --dry-run      # Preview without LLM calls
    python scripts/run_phase3_3_fulltext_screen.py --threshold 4  # Custom threshold (1-5, default 3=undecided+)
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
from src.utils.pdf_decomposer import get_or_decompose, format_segments_for_llm
from src.config import cfg
from src.agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

FULLTEXT_SCREEN_SYSTEM_PROMPT = """\
You are a systematic review screener performing FULL-TEXT screening.
These studies already passed title/abstract screening. Your job is to verify
they truly meet the inclusion criteria based on the full paper content.

Be MORE STRICT than title/abstract screening — you now have the full text.
Exclude studies that:
- Do not actually report primary data on the target PICO
- Are secondary analyses of already-included cohorts (same data, different paper)
- Have no extractable quantitative outcomes relevant to meta-analysis
- Are review articles, editorials, or guidelines that slipped through T/A screening
- Study a related but different population, intervention, or outcome

Use a 5-point confidence scale:
- most_likely_include (5): Clearly reports relevant primary data with extractable outcomes
- likely_include (4): Likely relevant, minor uncertainty about overlap or outcome type
- undecided (3): Cannot determine — may need human review
- likely_exclude (2): Probably not relevant on full-text review
- most_likely_exclude (1): Clearly does not meet criteria on closer inspection

Return JSON:
{
  "confidence": "<level>",
  "reasoning": "<2-3 sentence justification based on full text>",
  "study_design": "<RCT|cohort|case-control|cross-sectional|other>",
  "has_extractable_outcomes": true/false,
  "exclusion_reason": "<null or brief reason if excluding>"
}"""

CONFIDENCE_MAP = {
    "most_likely_include": 5,
    "likely_include": 4,
    "undecided": 3,
    "likely_exclude": 2,
    "most_likely_exclude": 1,
}


def build_fulltext_prompt(study: dict, pico: dict, content: str,
                          criteria: dict) -> str:
    """Build the user prompt for full-text screening."""
    return (
        f"Study ID: {study.get('study_id', 'unknown')}\n"
        f"Title: {study.get('title', 'N/A')}\n\n"
        f"--- PICO ---\n"
        f"Population: {pico.get('population', 'N/A')}\n"
        f"Intervention: {pico.get('intervention', 'N/A')}\n"
        f"Comparison: {pico.get('comparison', 'N/A')}\n"
        f"Outcome: {json.dumps(pico.get('outcome', 'N/A'))}\n\n"
        f"--- Inclusion criteria ---\n"
        f"{json.dumps(criteria.get('inclusion', []), indent=2)}\n\n"
        f"--- Exclusion criteria ---\n"
        f"{json.dumps(criteria.get('exclusion', []), indent=2)}\n\n"
        f"--- Full Text Content (truncated) ---\n"
        f"{content}\n\n"
        f"Based on the FULL TEXT above, rate this study for inclusion."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview studies without LLM calls")
    parser.add_argument("--threshold", type=int, default=3,
                        help="Confidence threshold for inclusion (1-5, default 3 = include undecided+)")
    parser.add_argument("--max-chars", type=int, default=30000,
                        help="Max characters of PDF content to send (default 30000)")
    args = parser.parse_args()

    select_project()
    dm = DataManager()
    data_dir = get_data_dir()

    # Load studies with PDFs
    if dm.exists("phase3_screening", "included_with_pdf.json",
                 subfolder="stage2_fulltext"):
        studies = dm.load("phase3_screening", "included_with_pdf.json",
                          subfolder="stage2_fulltext")
    else:
        studies = dm.load("phase3_screening", "included_studies.json",
                          subfolder="stage1_title_abstract")

    # Filter to only studies with actual PDFs
    studies_with_pdf = [s for s in studies
                        if s.get("pdf_path") and Path(s["pdf_path"]).exists()]
    studies_no_pdf = [s for s in studies
                      if not s.get("pdf_path") or not Path(s["pdf_path"]).exists()]

    logger.info(f"Full-text screening: {len(studies_with_pdf)} with PDF, "
                f"{len(studies_no_pdf)} without PDF (will skip)")

    if not studies_with_pdf:
        logger.error("No studies with valid pdf_path found. "
                     "Run Phase 3.2 --download + --finalize-pending first.")
        return

    if args.dry_run:
        print(f"\n  Dry run: would screen {len(studies_with_pdf)} studies")
        print(f"  Threshold: {args.threshold} ({_threshold_label(args.threshold)})")
        print(f"  Studies without PDF (skipped): {len(studies_no_pdf)}")
        return

    # Load PICO and criteria
    pico = dm.load_if_exists("input", "pico.yaml", default={})
    criteria = dm.load_if_exists("phase1_strategy", "screening_criteria.json", default={})

    cache_dir = str(Path(data_dir) / ".cache" / "decomposed")
    budget = TokenBudget("phase3_ft", limit_usd=cfg.budget("phase3"), reset=True)

    # Use Sonnet (arbiter role = Claude Sonnet 4.6)
    agent = BaseAgent(role_name="arbiter", budget=budget)

    results = []

    from tqdm import tqdm
    for study in tqdm(studies_with_pdf, desc="Full-text screening"):
        study_id = study.get("study_id", "unknown")
        pdf_path = study["pdf_path"]

        try:
            segments = get_or_decompose(pdf_path, cache_dir=cache_dir)
            content = format_segments_for_llm(segments)[:args.max_chars]
        except Exception as e:
            logger.warning(f"  {study_id}: PDF decompose failed: {e}")
            results.append({
                "study_id": study_id,
                "confidence": "undecided",
                "confidence_score": 3,
                "reasoning": f"PDF decomposition failed: {e}",
                "has_extractable_outcomes": None,
                "exclusion_reason": None,
            })
            continue

        prompt = build_fulltext_prompt(study, pico, content, criteria)

        response = agent.call_llm(
            prompt=prompt,
            system_prompt=FULLTEXT_SCREEN_SYSTEM_PROMPT,
            expect_json=True,
            cache_namespace="fulltext_screening",
            description=f"FT screen {study_id}",
        )

        parsed = response.get("parsed", {})
        confidence = parsed.get("confidence", "undecided")
        score = CONFIDENCE_MAP.get(confidence, 3)

        results.append({
            "study_id": study_id,
            "title": study.get("title", ""),
            "confidence": confidence,
            "confidence_score": score,
            "reasoning": parsed.get("reasoning", ""),
            "study_design": parsed.get("study_design", ""),
            "has_extractable_outcomes": parsed.get("has_extractable_outcomes"),
            "exclusion_reason": parsed.get("exclusion_reason"),
        })

    # Apply threshold
    included = []
    excluded = []
    human_review = []

    for r in results:
        study = next((s for s in studies_with_pdf
                      if s.get("study_id") == r["study_id"]), None)
        if not study:
            continue

        entry = {**study, "_fulltext_screening": r}

        if r["confidence_score"] >= args.threshold:
            included.append(entry)
        elif r["confidence"] == "undecided":
            human_review.append(entry)
        else:
            excluded.append(entry)

    # Save results
    dm.save("phase3_screening", "fulltext_screening_results.json",
            results, subfolder="stage2_fulltext")
    dm.save("phase3_screening", "included_fulltext.json",
            included, subfolder="stage2_fulltext")
    dm.save("phase3_screening", "excluded_fulltext.json",
            excluded, subfolder="stage2_fulltext")
    if human_review:
        dm.save("phase3_screening", "human_review_fulltext.json",
                human_review, subfolder="stage2_fulltext")

    # Stats
    from collections import Counter
    dist = Counter(r["confidence"] for r in results)

    cost = budget.summary()

    print("\n" + "=" * 50)
    print("  Phase 3.3 Full-Text Screening Complete")
    print("=" * 50)
    print(f"  Total screened:    {len(results)}")
    print(f"  Included:          {len(included)}  (score >= {args.threshold})")
    print(f"  Excluded:          {len(excluded)}")
    print(f"  Human review:      {len(human_review)}")
    print(f"  No PDF (skipped):  {len(studies_no_pdf)}")
    print(f"\n  Confidence distribution:")
    for level in ["most_likely_include", "likely_include", "undecided",
                   "likely_exclude", "most_likely_exclude"]:
        print(f"    {level}: {dist.get(level, 0)}")
    print(f"\n  Budget: {cost['total_cost_usd']}")
    print()


def _threshold_label(t):
    labels = {1: "include everything", 2: "exclude only most_likely_exclude",
              3: "include undecided+", 4: "include likely_include+",
              5: "include only most_likely_include"}
    return labels.get(t, "?")


if __name__ == "__main__":
    main()
