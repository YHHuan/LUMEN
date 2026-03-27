"""
Phase 4-5 Extraction Ablation — LUMEN v2
==========================================
Compare multi-pass pipeline vs single-model extraction+planning.

Arms:
  A. full_pipeline   — existing 3-pass Gemini + tiebreaker + Phase 4.5 planner (already have data)
  C. single_sonnet   — Sonnet 4.6 per-study, single-pass extraction + inline planning
  D. single_gemini   — Gemini Pro per-study, single-pass extraction + fallback planning

Usage:
    python scripts/run_extraction_ablation.py --arm C              # Run Sonnet arm
    python scripts/run_extraction_ablation.py --arm D              # Run Gemini arm
    python scripts/run_extraction_ablation.py --compare            # Compare all arms
    python scripts/run_extraction_ablation.py --compare --ground-truth gt.json
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.utils.pdf_decomposer import get_or_decompose, format_segments_for_llm
from src.utils.vector_index import DocumentVectorIndex, build_field_queries
from src.utils.extraction_context import build_extraction_context
from src.utils.deduplication import generate_canonical_citation
from src.utils.analysis_planner import profile_extracted_data, _build_fallback_plan, harmonize_outcomes_and_interventions
from src.utils.statistics import MetaAnalysisEngine, is_r_available, run_r_metafor
from src.utils.effect_sizes import compute_effect_auto
from src.utils.deduplication import deduplicate_for_meta_analysis
from src.config import cfg
from src.agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# Arm configurations
ARM_CONFIGS = {
    "C": {
        "name": "single_sonnet",
        "role": "strategist",  # reuse strategist config → Claude Sonnet 4.6
        "label": "Single Sonnet 4.6 (per-study)",
        "suffix": "_ablation_sonnet",
    },
    "D": {
        "name": "single_gemini",
        "role": "extractor",   # reuse extractor config → Gemini 3.1 Pro
        "label": "Single Gemini Pro (per-study, 1-pass)",
        "suffix": "_ablation_gemini",
    },
}

SINGLE_MODEL_SYSTEM_PROMPT = """\
You are a data extraction specialist for systematic reviews and meta-analyses.
Extract structured data from the provided study content.

CRITICAL RULES:
1. For EVERY numeric value, provide evidence_span — the exact text or table cell.
2. If a value is not found, set field to null and evidence_span to "NOT FOUND IN SOURCE".
3. NEVER estimate, calculate, or infer values not explicitly stated.
4. For table data: "Table X, Row Y: [exact content]"
5. For text data: quote the exact sentence (max 30 words).

OUTCOME TYPE DETECTION:
For each outcome, determine the outcome_type FIRST:
- "continuous": means, SDs, sample sizes (e.g., depression scores)
- "binary": events/totals or proportions (e.g., remission, response, mortality, infection rate)
- "time_to_event": hazard ratios, survival data
- "precomputed": paper reports effect sizes directly (OR, RR, HR, VE%, SMD with CI)

IMPORTANT — EXTRACT BOTH CONTINUOUS AND BINARY WHEN AVAILABLE:
Many studies report BOTH continuous scores AND binary rates for the same construct.
For example, a depression study may report mean BDI-II scores (continuous) AND
number achieving remission (binary). Extract BOTH as separate outcome entries.
For binary outcomes, fill events (number with event) and total (group size) in each group.

ANALYSIS PLANNING:
In addition to extraction, classify each outcome with:
- intervention_normalized: canonical name (e.g., "Drug A", "Drug B", "Placebo")
- outcome_normalized: canonical name (e.g., "Primary outcome", "Mortality", "Response rate")
These will be used to group studies for separate meta-analyses.
NEVER pool different interventions or different outcome types together.

Return valid JSON with this structure:
{
  "study_id": "<id>",
  "study_design": "<RCT|cohort|case-control|...>",
  "population_description": "<string>",
  "total_n": <integer>,
  "intervention_description": "<string>",
  "control_description": "<string>",
  "canonical_citation": "<LastName et al. (Year)>",
  "outcomes": [
    {
      "measure": "<outcome description>",
      "outcome_type": "<continuous|binary|time_to_event|precomputed>",
      "intervention_normalized": "<canonical intervention name>",
      "outcome_normalized": "<canonical outcome name>",
      "intervention_group": {"mean": null, "sd": null, "n": null, "events": null, "total": null, "evidence_span": "..."},
      "control_group": {"mean": null, "sd": null, "n": null, "events": null, "total": null, "evidence_span": "..."},
      "ve_pct": null, "ve_ci_lower": null, "ve_ci_upper": null,
      "effect_size": null, "effect_measure": null, "ci_lower": null, "ci_upper": null,
      "hr": null, "hr_ci_lower": null, "hr_ci_upper": null,
      "evidence_span": "...",
      "confidence": "<high|medium|low>"
    }
  ]
}"""


def run_single_arm(dm, arm_key, studies, pico):
    """Run single-model extraction for one arm."""
    arm = ARM_CONFIGS[arm_key]
    suffix = arm["suffix"]

    budget = TokenBudget(f"phase4{suffix}", limit_usd=15.0, reset=True)
    agent = BaseAgent(role_name=arm["role"], budget=budget)

    cache_dir = str(Path(get_data_dir()) / ".cache" / "decomposed")

    all_extracted = []
    start_time = time.time()

    from tqdm import tqdm
    for study in tqdm(studies, desc=f"Extracting ({arm['label']})"):
        study_id = study.get("study_id", "unknown")
        pdf_path = study.get("pdf_path", "")

        # Build context (same as full pipeline)
        if pdf_path and Path(pdf_path).exists():
            segments = get_or_decompose(pdf_path, cache_dir=cache_dir)
            context_text = format_segments_for_llm(segments)[:40000]
        else:
            context_text = study.get("abstract", "")

        user_prompt = (
            f"Study ID: {study_id}\n"
            f"Title: {study.get('title', '')}\n\n"
            f"PICO Context:\n"
            f"  Population: {pico.get('population', 'N/A')}\n"
            f"  Intervention: {pico.get('intervention', 'N/A')}\n"
            f"  Comparison: {pico.get('comparison', 'N/A')}\n"
            f"  Outcome: {json.dumps(pico.get('outcome', 'N/A'))}\n\n"
            f"Study Content:\n{context_text}\n\n"
            f"Extract all relevant outcomes. For each outcome, include "
            f"intervention_normalized and outcome_normalized fields."
        )

        result = agent.call_llm(
            prompt=user_prompt,
            system_prompt=SINGLE_MODEL_SYSTEM_PROMPT,
            expect_json=True,
            cache_namespace=f"extraction{suffix}",
            description=f"Extract {study_id}",
        )

        parsed = result.get("parsed", {})
        if parsed:
            parsed["study_id"] = study_id
            if not parsed.get("canonical_citation"):
                parsed["canonical_citation"] = generate_canonical_citation(study)
            all_extracted.append(parsed)
        else:
            logger.warning(f"  {study_id}: extraction failed (parse error)")

    elapsed = time.time() - start_time

    # Save extracted data
    dm.save("phase4_extraction", f"extracted_data{suffix}.json", all_extracted)

    # Harmonize outcomes before profiling (same as full pipeline)
    harmonize_agent = BaseAgent(role_name=arm["role"], budget=budget)
    harmonization = harmonize_outcomes_and_interventions(all_extracted, pico, harmonize_agent)
    dm.save("phase4_5_planning", f"harmonization_map{suffix}.json", harmonization)

    # Profile + fallback plan
    profile = profile_extracted_data(all_extracted, pico)
    plan = _build_fallback_plan(profile, pico)
    plan["human_approved"] = True

    dm.save(f"phase4_5_planning", f"data_profile{suffix}.json", profile)
    dm.save(f"phase4_5_planning", f"analysis_plan{suffix}.json", plan)

    # Run Phase 5 on each planned analysis
    _run_stats_for_arm(dm, all_extracted, plan, suffix)

    # Save cost summary
    cost_summary = budget.summary()
    cost_summary["elapsed_seconds"] = round(elapsed, 1)
    cost_summary["arm"] = arm["label"]
    cost_summary["n_studies"] = len(all_extracted)
    dm.save("benchmark", f"cost{suffix}.json", cost_summary)

    cost_val = cost_summary.get('total_cost_usd', 0)
    logger.info(f"Arm {arm_key} ({arm['label']}): {len(all_extracted)} studies, "
                f"${cost_val:.2f}, {elapsed:.0f}s")

    return all_extracted, cost_summary


def _match_outcome_for_analysis(study: dict, analysis: dict, measure: str):
    """Find the outcome in a study that matches the analysis definition.

    Priority:
      1. Exact measure_broad or outcome_normalized match
      2. Substring match on measure_broad only
      3. Substring match on raw measure
      4. First computable outcome (fallback)
    """
    target_outcome = (analysis.get("outcome") or "").lower().strip()
    outcomes = study.get("outcomes", [])

    # Pass 1: exact measure_broad or outcome_normalized match
    if target_outcome:
        for o in outcomes:
            on = (o.get("outcome_normalized") or "").lower().strip()
            mb = (o.get("measure_broad") or "").lower().strip()
            if on == target_outcome or mb == target_outcome:
                es = compute_effect_auto(o, preferred_measure=measure)
                if es and es.get("yi") is not None:
                    return es
        # Pass 2: substring match on measure_broad only
        for o in outcomes:
            mb = (o.get("measure_broad") or "").lower().strip()
            if mb and (target_outcome in mb or mb in target_outcome):
                es = compute_effect_auto(o, preferred_measure=measure)
                if es and es.get("yi") is not None:
                    return es
        # Pass 3: substring match on raw measure
        for o in outcomes:
            raw = (o.get("measure") or "").lower().strip()
            if raw and (target_outcome in raw or raw in target_outcome):
                es = compute_effect_auto(o, preferred_measure=measure)
                if es and es.get("yi") is not None:
                    return es

    # Pass 4: first computable outcome (last resort fallback)
    for o in outcomes:
        es = compute_effect_auto(o, preferred_measure=measure)
        if es and es.get("yi") is not None:
            return es

    return None


def _run_stats_for_arm(dm, extracted, plan, suffix):
    """Run Phase 5 statistics for an ablation arm."""
    deduped, _ = deduplicate_for_meta_analysis(extracted)
    study_lookup = {s.get("study_id"): s for s in deduped}

    p5 = cfg.phase5_settings
    estimator = p5.get("estimator", "REML")
    hksj = p5.get("hartung_knapp", True)

    all_results = {}
    analyses = plan.get("analyses", [])

    for analysis in analyses:
        aid = analysis.get("id", "unnamed")
        study_ids = analysis.get("study_ids", [])
        measure = analysis.get("effect_measure", "OR")

        analysis_studies = [study_lookup[sid] for sid in study_ids if sid in study_lookup]
        if len(analysis_studies) < 2:
            continue

        effects_list, variances_list, labels_list = [], [], []
        for study in analysis_studies:
            slabel = study.get("canonical_citation", study.get("study_id", ""))
            es = _match_outcome_for_analysis(study, analysis, measure)
            if es:
                effects_list.append(es["yi"])
                variances_list.append(es["vi"])
                labels_list.append(slabel)

        if len(effects_list) < 2:
            continue

        effects = np.array(effects_list)
        variances = np.array(variances_list)

        try:
            if is_r_available():
                result = run_r_metafor(effects, variances, labels_list,
                                       method=estimator, knha=hksj)
            else:
                engine = MetaAnalysisEngine(estimator=estimator, hartung_knapp=hksj)
                result = engine.run_full_analysis(effects, variances, labels_list)
        except Exception as e:
            logger.error(f"  Stats failed for {aid}: {e}")
            continue

        all_results[aid] = {
            "analysis_id": aid,
            "label": analysis.get("label", aid),
            "measure": measure,
            "k": len(effects_list),
            "results": result,
        }

    dm.save("phase5_analysis", f"planned_results{suffix}.json", all_results)


def compare_arms(dm):
    """Compare all available arms and generate comparison table."""
    arms_data = {}

    # Arm A: full pipeline (existing data)
    full_extracted = dm.load_if_exists("phase4_extraction", "extracted_data.json", default=[])
    full_results = dm.load_if_exists("phase5_analysis", "planned_results.json", default={})
    if full_extracted:
        arms_data["A_full_pipeline"] = {
            "label": "Full pipeline (3-pass + planner)",
            "n_extracted": len(full_extracted),
            "results": full_results,
        }

    # Arm C & D
    for arm_key, arm_cfg in ARM_CONFIGS.items():
        suffix = arm_cfg["suffix"]
        ext = dm.load_if_exists("phase4_extraction", f"extracted_data{suffix}.json", default=[])
        res = dm.load_if_exists("phase5_analysis", f"planned_results{suffix}.json", default={})
        cost = dm.load_if_exists("benchmark", f"cost{suffix}.json", default={})
        if ext:
            arms_data[f"{arm_key}_{arm_cfg['name']}"] = {
                "label": arm_cfg["label"],
                "n_extracted": len(ext),
                "results": res,
                "cost": cost,
            }

    if not arms_data:
        logger.error("No arm data found")
        return

    # Build comparison table
    lines = [
        "# Phase 4-5 Extraction Ablation Comparison",
        "",
        "| Arm | Model | Studies | Analyses | Cost | Time |",
        "|-----|-------|---------|----------|------|------|",
    ]

    for key, data in arms_data.items():
        n_analyses = len(data.get("results", {}))
        cost = data.get("cost", {})
        raw_cost = cost.get('total_cost_usd', 0) if cost else 0
        cost_str = f"${float(raw_cost):.2f}" if cost else "—"
        raw_time = cost.get('elapsed_seconds', 0) if cost else 0
        time_str = f"{float(raw_time):.0f}s" if cost else "—"
        lines.append(f"| {key} | {data['label']} | {data['n_extracted']} | "
                      f"{n_analyses} | {cost_str} | {time_str} |")

    # Per-analysis comparison
    lines.extend(["", "## Per-Analysis Results", ""])

    # Collect all analysis IDs across arms
    all_aids = set()
    for data in arms_data.values():
        all_aids.update(data.get("results", {}).keys())

    if all_aids:
        lines.append("| Analysis | " + " | ".join(arms_data.keys()) + " |")
        lines.append("|----------|" + "|".join(["---"] * len(arms_data)) + "|")

        for aid in sorted(all_aids):
            row = [aid]
            for key, data in arms_data.items():
                res = data.get("results", {}).get(aid, {})
                if res:
                    r = res.get("results", {})
                    # Handle both R engine (r_raw.main) and Python engine (main) formats
                    main = {}
                    if isinstance(r.get("r_raw"), dict) and "main" in r["r_raw"]:
                        main = r["r_raw"]["main"]
                    elif "main" in r:
                        main = r["main"]
                    if main:
                        pooled = main.get("pooled_effect") or 0
                        i2 = main.get("I2") or 0
                        k = res.get("k", 0)
                        try:
                            pooled = float(pooled)
                            i2 = float(i2)
                            m = res.get("measure", "")
                            if m.startswith("VE"):
                                ve = (1 - np.exp(pooled)) * 100
                                row.append(f"k={k} VE={ve:.0f}% I²={i2:.0f}%")
                            elif m in ("OR", "RR", "HR"):
                                row.append(f"k={k} {m}={np.exp(pooled):.2f} I²={i2:.0f}%")
                            else:
                                row.append(f"k={k} ES={pooled:.3f} I²={i2:.0f}%")
                        except (TypeError, ValueError):
                            row.append(f"k={k}")
                    else:
                        row.append(f"k={res.get('k', '?')}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    dm.save("benchmark", "extraction_ablation_comparison.md", table)
    print("\n" + table + "\n")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, choices=["C", "D"],
                        help="Which arm to run")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all available arms")
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="Ground truth JSON for accuracy comparison")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    if args.compare:
        compare_arms(dm)
        return

    if not args.arm:
        print("Specify --arm C or --arm D, or --compare")
        return

    # Load included studies (auto-selects best available: 3.3 > 3.2 > 3.1)
    studies = dm.load_best_included()

    pico = dm.load_if_exists("input", "pico.yaml", default={})

    logger.info(f"Running Arm {args.arm}: {ARM_CONFIGS[args.arm]['label']}")
    logger.info(f"Studies: {len(studies)}")

    run_single_arm(dm, args.arm, studies, pico)

    # Auto-compare after running
    compare_arms(dm)


if __name__ == "__main__":
    main()
