"""
Phase 5: Statistical Analysis — LUMEN v2
===========================================
Random-effects meta-analysis with REML, HKSJ, sensitivity analyses,
publication bias tests, subgroup analysis, and meta-regression.
Network meta-analysis (NMA) via R netmeta when enabled.

Usage:
    python scripts/run_phase5.py                  # Full analysis (pairwise)
    python scripts/run_phase5.py --nma            # NMA analysis
    python scripts/run_phase5.py --builtin-only   # Skip LLM interpretation
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.utils.statistics import MetaAnalysisEngine, is_r_available, run_r_metafor, run_dual_engine
from src.utils.effect_sizes import compute_effect_from_study, compute_effect_auto, detect_outcome_type
from src.utils.deduplication import deduplicate_for_meta_analysis
from src.utils import visualizations as viz
from src.utils.stage_gate import validate_phase4_to_5
from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--builtin-only", action="store_true")
    parser.add_argument("--use-r", action="store_true", help="Use R metafor engine (primary)")
    parser.add_argument("--use-python", action="store_true", help="Use Python engine only")
    parser.add_argument("--use-both", action="store_true", help="Run both engines and compare")
    parser.add_argument("--nma", action="store_true", help="Run NMA instead of pairwise MA")
    parser.add_argument("--planned", action="store_true",
                        help="Use analysis_plan.yaml from Phase 4.5 to run subgroup-aware analyses")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    # Stage gate validation (Phase 4 -> 5)
    gate = validate_phase4_to_5(dm)
    if not gate.passed:
        logger.error("Stage gate Phase 4 -> 5 FAILED. Fix extraction data before running Phase 5.")
        return

    # Check NMA mode
    nma_mode = args.nma or cfg.v2.get("nma", {}).get("enabled", False)
    if nma_mode:
        _run_nma(dm, args)
        return

    # Planned mode: run per-analysis from Phase 4.5 plan
    if args.planned:
        _run_planned_analyses(dm, args)
        return

    # Load extracted data
    extracted = dm.load("phase4_extraction", "extracted_data.json")
    logger.info(f"Loaded {len(extracted)} extracted studies")

    # Deduplicate for meta-analysis
    deduped, dedup_log = deduplicate_for_meta_analysis(extracted)
    if dedup_log:
        logger.info(f"Removed {len(dedup_log)} duplicate citations")
        dm.save("phase5_analysis", "meta_dedup_log.json", dedup_log)

    # Settings
    p5 = cfg.phase5_settings

    # Detect preferred effect measure from PICO or config
    pico = dm.load_if_exists("input", "pico.yaml", default={})
    preferred_measure = (
        pico.get("effect_measure") or
        pico.get("preferred_measure") or
        p5.get("effect_measure") or
        None
    )

    # Compute effect sizes with auto-routing
    effects_list = []
    variances_list = []
    labels = []
    years = []
    study_metadata = {}
    detected_measures = []

    for study in deduped:
        study_id = study.get("study_id", "")
        label = study.get("canonical_citation", study_id)

        for outcome in study.get("outcomes", []):
            es = compute_effect_auto(outcome, preferred_measure=preferred_measure)
            if es and es.get("yi") is not None:
                effects_list.append(es["yi"])
                variances_list.append(es["vi"])
                labels.append(label)
                years.append(int(study.get("year", 0)) if study.get("year") else 0)
                study_metadata[label] = study
                detected_measures.append(es.get("measure", "SMD"))
                break  # Use primary outcome

    if len(effects_list) < 2:
        logger.error(f"Only {len(effects_list)} computable effects, need >= 2")
        return

    effects = np.array(effects_list)
    variances = np.array(variances_list)

    # Determine the dominant effect measure for labeling
    from collections import Counter
    measure_counts = Counter(detected_measures)
    dominant_measure = measure_counts.most_common(1)[0][0] if measure_counts else "SMD"
    logger.info(f"Effect measures detected: {dict(measure_counts)}, using '{dominant_measure}' as label")

    # Map to R metafor display label
    MEASURE_LABELS = {
        "SMD": "SMD", "MD": "MD",
        "OR": "log(OR)", "RR": "log(RR)", "RD": "RD",
        "HR": "log(HR)",
        "VE_OR": "log(OR) [VE]", "VE_RR": "log(RR) [VE]",
    }
    measure_label = MEASURE_LABELS.get(dominant_measure, dominant_measure)

    logger.info(f"Computing meta-analysis with {len(effects)} studies")

    estimator = p5.get("estimator", "REML")
    hksj = p5.get("hartung_knapp", True)
    engine_pref = p5.get("engine", "auto")  # "r" | "python" | "both" | "auto"

    # CLI flags override config
    if args.use_r:
        engine_pref = "r"
    elif args.use_python:
        engine_pref = "python"
    elif args.use_both:
        engine_pref = "both"

    # Auto-detect: prefer R if available
    if engine_pref == "auto":
        engine_pref = "r" if is_r_available() else "python"
        logger.info(f"Auto-detected engine: {engine_pref}")

    py_engine = MetaAnalysisEngine(estimator=estimator, hartung_knapp=hksj)

    # Subgroup analysis
    subgroups = None
    subgroup_vars = p5.get("subgroup_by", [])
    # For now, use first available subgroup variable
    if subgroup_vars:
        for var in subgroup_vars:
            sg = [study_metadata.get(l, {}).get(var, "unknown") for l in labels]
            if len(set(sg)) > 1:
                subgroups = sg
                break

    # Meta-regression moderators
    moderators = None
    moderator_names = None
    if p5.get("meta_regression", False):
        # Use year as a moderator (example)
        if any(y > 0 for y in years):
            moderators = np.array(years, dtype=float).reshape(-1, 1)
            moderator_names = ["publication_year"]

    valid_years = years if any(y > 0 for y in years) else None

    # Run analysis with selected engine
    if engine_pref == "both":
        logger.info("Running dual-engine analysis (R + Python)")
        results = run_dual_engine(
            effects, variances, labels,
            method=estimator, knha=hksj,
            years=valid_years, subgroups=subgroups,
            moderators=moderators, moderator_names=moderator_names,
            python_engine=py_engine,
        )
    elif engine_pref == "r":
        logger.info("Running R metafor engine (primary)")
        fig_dir = dm.phase_dir("phase5_analysis", "figures")
        try:
            results = run_r_metafor(
                effects, variances, labels,
                method=estimator, knha=hksj,
                measure_label=measure_label,
                years=valid_years, subgroups=subgroups,
                moderators=moderators,
                figures_dir=str(fig_dir),
            )
        except Exception as e:
            logger.warning(f"R engine failed ({e}), falling back to Python")
            results = py_engine.run_full_analysis(
                effects, variances, labels,
                subgroups=subgroups, moderators=moderators,
                moderator_names=moderator_names, years=valid_years,
            )
    else:
        logger.info("Running Python engine")
        results = py_engine.run_full_analysis(
            effects, variances, labels,
            subgroups=subgroups, moderators=moderators,
            moderator_names=moderator_names, years=valid_years,
        )

    # Save results
    dm.save("phase5_analysis", "statistical_results.json", results)

    # Generate figures (skip matplotlib if R already produced them)
    r_figs_exist = results.get("r_raw", {}).get("figures_generated") if engine_pref == "r" else False
    fig_dir = dm.phase_dir("phase5_analysis", "figures")

    if r_figs_exist:
        logger.info("R engine produced publication-quality figures (300 DPI), skipping matplotlib")
    else:
        logger.info("Generating matplotlib figures (fallback)")

    if not r_figs_exist:
        viz.forest_plot(
            effects, effects - 1.96 * np.sqrt(variances),
            effects + 1.96 * np.sqrt(variances),
            labels, pooled=results["main"],
            title="Forest Plot — Random-Effects Meta-Analysis",
            save_path=str(fig_dir / "forest_plot.png"),
        )
        plt_close()

        viz.funnel_plot(
            effects, np.sqrt(variances), labels,
            pooled_effect=results["main"]["pooled_effect"],
            egger_result=results.get("egger"),
            trim_fill_result=results.get("trim_and_fill"),
            save_path=str(fig_dir / "funnel_plot.png"),
        )
        plt_close()

        if results.get("leave_one_out"):
            viz.leave_one_out_plot(
                results["leave_one_out"],
                overall_result=results["main"],
                save_path=str(fig_dir / "leave_one_out.png"),
            )
            plt_close()

        if results.get("cumulative"):
            viz.cumulative_forest_plot(
                results["cumulative"],
                save_path=str(fig_dir / "cumulative_meta.png"),
            )
            plt_close()

        if results.get("influence"):
            viz.influence_plot(
                results["influence"],
                save_path=str(fig_dir / "influence_diagnostics.png"),
            )
            plt_close()

    if not r_figs_exist and results.get("subgroup"):
        viz.subgroup_forest_plot(
            results["subgroup"],
            save_path=str(fig_dir / "subgroup_forest.png"),
        )
        plt_close()

    # LLM interpretation
    if not args.builtin_only:
        try:
            budget = TokenBudget("phase5", limit_usd=cfg.budget("phase5"), reset=True)
            from src.agents.statistician import StatisticianAgent
            statistician = StatisticianAgent(budget=budget)
            interpretation = statistician.interpret_results(results, {
                "pico": dm.load_if_exists("input", "pico.yaml", default={}),
                "k": len(effects),
            })
            dm.save("phase5_analysis", "interpretation.json", interpretation)
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e}")

    # Store measure info in results
    results["effect_measure"] = dominant_measure
    results["measure_label"] = measure_label
    if "main" in results:
        results["main"]["effect_measure"] = dominant_measure

    # Summary
    main_result = results["main"]
    print("\n" + "=" * 50)
    print("  Phase 5 Statistical Analysis Complete")
    print("=" * 50)
    print(f"  Measure:         {dominant_measure} ({measure_label})")
    print(f"  Studies (k):     {main_result['k']}")
    print(f"  Estimator:       {main_result.get('estimator', '?')}")
    print(f"  Adjustment:      {main_result.get('adjustment', 'none')}")
    pooled = main_result['pooled_effect']
    ci_lo, ci_hi = main_result['ci_lower'], main_result['ci_upper']
    print(f"  Pooled effect:   {pooled:.4f}")
    print(f"  95% CI:          [{ci_lo:.4f}, {ci_hi:.4f}]")
    # For ratio measures, also show exponentiated values
    if dominant_measure in ("OR", "RR", "HR", "VE_OR", "VE_RR"):
        exp_est = np.exp(pooled)
        exp_lo, exp_hi = np.exp(ci_lo), np.exp(ci_hi)
        print(f"  Exp(effect):     {exp_est:.4f} [{exp_lo:.4f}, {exp_hi:.4f}]")
        if dominant_measure.startswith("VE"):
            ve = (1 - exp_est) * 100
            ve_lo = (1 - exp_hi) * 100  # inverted
            ve_hi = (1 - exp_lo) * 100
            print(f"  VE%:             {ve:.1f}% [{ve_lo:.1f}%, {ve_hi:.1f}%]")
    print(f"  p-value:         {main_result['p_value']:.6f}")
    print(f"  I2:              {main_result['I2']:.1f}%")
    print(f"  tau2:            {main_result['tau2']:.4f}")
    if results.get("egger"):
        print(f"  Egger's test:    p={results['egger']['p_value']:.4f} ({results['egger']['interpretation']})")
    if results.get("meta_regression"):
        mr = results["meta_regression"]
        print(f"  Meta-regression: R2={mr.get('R2_analog', 0):.3f}, QM p={mr.get('QM_p', 1):.4f}")
    print()

    # Generate renv.lock for reproducibility
    if engine_pref in ("r", "both"):
        renv_script = Path(__file__).parent / "r_bridge" / "generate_renv_lock.R"
        renv_path = dm.phase_dir("phase5_analysis") / "renv.lock"
        try:
            import subprocess
            subprocess.run(
                ["Rscript", str(renv_script), str(renv_path)],
                capture_output=True, text=True, timeout=30,
            )
            if renv_path.exists():
                logger.info(f"renv.lock saved: {renv_path}")
        except Exception as e:
            logger.warning(f"renv.lock generation failed: {e}")


def _run_planned_analyses(dm, args):
    """Run multiple analyses from Phase 4.5 analysis plan."""
    from src.utils.analysis_planner import load_analysis_plan

    plan_path = str(Path(get_data_dir()) / "phase4_5_planning" / "analysis_plan.yaml")
    plan = load_analysis_plan(plan_path)

    if not plan:
        logger.error(f"No analysis plan found at {plan_path}. Run Phase 4.5 first.")
        return

    if not plan.get("human_approved", False):
        logger.error("Analysis plan not approved. Run Phase 4.5 to approve or set human_approved: true")
        return

    analyses = plan.get("analyses", [])
    if not analyses:
        logger.error("Analysis plan has no analyses defined.")
        return

    # Load extracted data
    extracted = dm.load("phase4_extraction", "extracted_data.json")
    deduped, dedup_log = deduplicate_for_meta_analysis(extracted)
    if dedup_log:
        dm.save("phase5_analysis", "meta_dedup_log.json", dedup_log)

    # Build study lookup by study_id
    study_lookup = {s.get("study_id"): s for s in deduped}

    # Settings
    p5 = cfg.phase5_settings
    estimator = p5.get("estimator", "REML")
    hksj = p5.get("hartung_knapp", True)

    all_results = {}
    total_k = 0

    MEASURE_LABELS = {
        "SMD": "SMD", "MD": "MD",
        "OR": "log(OR)", "RR": "log(RR)", "RD": "RD",
        "HR": "log(HR)",
        "VE_OR": "log(OR) [VE]", "VE_RR": "log(RR) [VE]",
    }

    print("\n" + "=" * 60)
    print("  Phase 5: Planned Multi-Analysis Mode")
    print("=" * 60)

    for analysis in analyses:
        aid = analysis.get("id", "unnamed")
        label = analysis.get("label", aid)
        study_ids = analysis.get("study_ids", [])
        measure = analysis.get("effect_measure", "SMD")

        # Filter studies for this analysis
        analysis_studies = [study_lookup[sid] for sid in study_ids if sid in study_lookup]
        if len(analysis_studies) < 2:
            logger.warning(f"Analysis '{aid}': only {len(analysis_studies)} studies found, skipping")
            continue

        # Compute effects for these studies
        effects_list = []
        variances_list = []
        labels_list = []

        target_outcome = (analysis.get("outcome") or "").lower().strip()

        for study in analysis_studies:
            study_id = study.get("study_id", "")
            slabel = study.get("canonical_citation", study_id)

            matched_es = None
            # Pass 1: exact match on measure_broad or outcome_normalized
            if target_outcome:
                for outcome in study.get("outcomes", []):
                    on = (outcome.get("outcome_normalized") or "").lower().strip()
                    mb = (outcome.get("measure_broad") or "").lower().strip()
                    if on == target_outcome or mb == target_outcome:
                        matched_es = compute_effect_auto(outcome, preferred_measure=measure)
                        if matched_es and matched_es.get("yi") is not None:
                            break
                        matched_es = None
                # Pass 2: substring match on measure_broad only (avoid raw measure ambiguity)
                if not matched_es:
                    for outcome in study.get("outcomes", []):
                        mb = (outcome.get("measure_broad") or "").lower().strip()
                        if mb and (target_outcome in mb or mb in target_outcome):
                            matched_es = compute_effect_auto(outcome, preferred_measure=measure)
                            if matched_es and matched_es.get("yi") is not None:
                                break
                            matched_es = None
                # Pass 3: substring match on raw measure (broader, less precise)
                if not matched_es:
                    for outcome in study.get("outcomes", []):
                        raw = (outcome.get("measure") or "").lower().strip()
                        if raw and (target_outcome in raw or raw in target_outcome):
                            matched_es = compute_effect_auto(outcome, preferred_measure=measure)
                            if matched_es and matched_es.get("yi") is not None:
                                break
                            matched_es = None
            # Pass 4: first computable outcome (last resort fallback)
            if not matched_es:
                for outcome in study.get("outcomes", []):
                    matched_es = compute_effect_auto(outcome, preferred_measure=measure)
                    if matched_es and matched_es.get("yi") is not None:
                        break

            if matched_es and matched_es.get("yi") is not None:
                # Check measure compatibility: don't mix MD with SMD or OR with SMD
                actual_measure = matched_es.get("measure", "SMD")
                compatible = True
                if measure == "SMD" and actual_measure == "MD":
                    logger.warning(f"  {study_id}: skipping precomputed MD (={matched_es['yi']:.3f}) — incompatible with SMD pooling")
                    compatible = False
                elif measure == "OR" and actual_measure not in ("OR", "RR", "RD"):
                    logger.warning(f"  {study_id}: skipping {actual_measure} — incompatible with OR pooling")
                    compatible = False
                elif measure in ("SMD", "MD") and actual_measure in ("OR", "RR"):
                    logger.warning(f"  {study_id}: skipping {actual_measure} — incompatible with {measure} pooling")
                    compatible = False

                if compatible:
                    effects_list.append(matched_es["yi"])
                    variances_list.append(matched_es["vi"])
                    labels_list.append(slabel)

        if len(effects_list) < 2:
            logger.warning(f"Analysis '{aid}': only {len(effects_list)} computable effects, skipping")
            continue

        effects = np.array(effects_list)
        variances = np.array(variances_list)
        measure_label = MEASURE_LABELS.get(measure, measure)

        logger.info(f"Running analysis '{aid}': k={len(effects)}, measure={measure_label}")

        # Run R metafor
        fig_dir = dm.phase_dir("phase5_analysis", f"figures_{aid}")
        try:
            if is_r_available():
                result = run_r_metafor(
                    effects, variances, labels_list,
                    method=estimator, knha=hksj,
                    measure_label=measure_label,
                    figures_dir=str(fig_dir),
                )
            else:
                py_engine = MetaAnalysisEngine(estimator=estimator, hartung_knapp=hksj)
                result = py_engine.run_full_analysis(effects, variances, labels_list)
        except Exception as e:
            logger.error(f"Analysis '{aid}' failed: {e}")
            continue

        all_results[aid] = {
            "analysis_id": aid,
            "label": label,
            "measure": measure,
            "k": len(effects_list),
            "results": result,
        }
        total_k += len(effects_list)

        # Print summary
        main = result.get("r_raw", result).get("main", result.get("main", {}))
        if main:
            pooled = main.get("pooled_effect", 0)
            ci_lo = main.get("ci_lower", 0)
            ci_hi = main.get("ci_upper", 0)
            i2 = main.get("I2", 0)
            print(f"\n  [{aid}] {label}")
            print(f"    k={len(effects_list)}, pooled={pooled:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], I²={i2:.1f}%")
            if measure.startswith("VE"):
                ve = (1 - np.exp(pooled)) * 100
                ve_lo = (1 - np.exp(ci_hi)) * 100
                ve_hi = (1 - np.exp(ci_lo)) * 100
                print(f"    VE% = {ve:.1f}% [{ve_lo:.1f}%, {ve_hi:.1f}%]")

    # Save all results
    dm.save("phase5_analysis", "planned_results.json", all_results)

    # Also save combined statistical_results.json for downstream compatibility
    # Use first analysis as the "primary" result
    if all_results:
        first_key = list(all_results.keys())[0]
        dm.save("phase5_analysis", "statistical_results.json", all_results[first_key]["results"])

    print("\n" + "=" * 60)
    print("  Phase 5: Planned Analyses Complete")
    print("=" * 60)
    print(f"  Analyses run:    {len(all_results)}/{len(analyses)}")
    print(f"  Total studies:   {total_k}")
    for aid, res in all_results.items():
        print(f"  [{aid}]: k={res['k']}")
    print()


def _run_nma(dm, args):
    """Run NMA analysis via R netmeta."""
    from src.utils.nma import (
        prepare_nma_data, validate_network, run_nma_from_settings,
        is_netmeta_available, run_nma, load_nma_settings,
    )

    # Check R + netmeta availability
    if not is_netmeta_available():
        logger.error(
            "R netmeta not available. Install with:\n"
            "  Rscript -e 'install.packages(c(\"netmeta\", \"jsonlite\"))'"
        )
        return

    # Load NMA contrast data (generated by Phase 4 --nma)
    if dm.exists("phase4_extraction", "nma_contrasts.json"):
        contrasts = dm.load("phase4_extraction", "nma_contrasts.json")
        logger.info(f"Loaded {len(contrasts)} NMA contrasts from Phase 4")
    else:
        # Fall back: try to generate from extracted_data.json
        logger.info("No nma_contrasts.json found, generating from extracted_data.json")
        extracted = dm.load("phase4_extraction", "extracted_data.json")
        contrasts = prepare_nma_data(extracted)

    # Validate network
    validation = validate_network(contrasts)
    if not validation["valid"]:
        for err in validation["errors"]:
            logger.error(f"Network validation: {err}")
        logger.error("Cannot run NMA — fix network issues first")
        dm.save("phase5_analysis", "nma_validation_error.json", validation)
        return

    logger.info(
        f"Network: {validation['n_treatments']} treatments, "
        f"{validation['n_studies']} studies, {validation['n_contrasts']} contrasts"
    )
    logger.info(f"Treatments: {', '.join(validation['treatments'])}")

    # Output directory
    output_dir = str(dm.phase_dir("phase5_analysis", "nma"))

    # Run NMA
    nma_cfg = load_nma_settings()
    try:
        results = run_nma_from_settings(contrasts, output_dir)
    except ValueError as e:
        # NMA disabled in settings — run with defaults
        logger.warning(f"Settings issue ({e}), running with defaults")
        results = run_nma(
            contrasts, output_dir,
            effect_measure=nma_cfg.get("effect_measure", "SMD"),
            method_tau=nma_cfg.get("method_tau", "REML"),
            reference_group=nma_cfg.get("reference_group"),
        )

    # Save results
    dm.save("phase5_analysis", "nma_results.json", results)

    # Also save as statistical_results.json for Phase 6 compatibility
    phase5_results = {
        "analysis_type": "nma",
        "engine": "netmeta",
        "nma": results,
        "main": {
            "pooled_effect": None,
            "k": results.get("n_studies", 0),
            "tau2": results.get("tau2"),
            "I2": results.get("I2"),
            "Q": results.get("Q"),
            "estimator": results.get("method_tau", "REML"),
            "adjustment": "none",
        },
    }
    dm.save("phase5_analysis", "statistical_results.json", phase5_results)

    # LLM interpretation
    if not args.builtin_only:
        try:
            budget = TokenBudget("phase5", limit_usd=cfg.budget("phase5"), reset=True)
            from src.agents.statistician import StatisticianAgent
            statistician = StatisticianAgent(budget=budget)
            interpretation = statistician.interpret_results(phase5_results, {
                "pico": dm.load_if_exists("input", "pico.yaml", default={}),
                "k": results.get("n_studies", 0),
                "analysis_type": "nma",
                "treatments": results.get("treatments", []),
                "rankings": results.get("rankings"),
            })
            dm.save("phase5_analysis", "interpretation.json", interpretation)
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("  Phase 5 NMA Analysis Complete")
    print("=" * 50)
    print(f"  Studies:         {results.get('n_studies', '?')}")
    print(f"  Treatments:      {results.get('n_treatments', '?')}")
    print(f"  Contrasts:       {results.get('n_contrasts', '?')}")
    print(f"  Effect measure:  {results.get('effect_measure', '?')}")
    print(f"  tau2:            {results.get('tau2', '?')}")
    print(f"  I2:              {results.get('I2', '?')}%")

    if results.get("rankings"):
        rankings = results["rankings"]
        if isinstance(rankings, list):
            print("\n  Treatment Rankings (P-score):")
            for r in rankings:
                print(f"    {r.get('rank', '?')}. {r.get('treatment', '?')} (P={r.get('p_score', '?')})")
        elif isinstance(rankings, dict) and "treatment" in rankings:
            print(f"\n  Top treatment: {rankings.get('treatment', '?')}")

    consistency = results.get("consistency", {})
    if consistency:
        pval = consistency.get("pval_between", "?")
        incon = consistency.get("inconsistency_detected", False)
        print(f"\n  Consistency:     p={pval} ({'INCONSISTENCY' if incon else 'OK'})")

    print(f"\n  Figures: {output_dir}/figures/")
    print(f"  Tables:  {output_dir}/tables/")
    print()


def plt_close():
    import matplotlib.pyplot as plt
    plt.close("all")


if __name__ == "__main__":
    main()
