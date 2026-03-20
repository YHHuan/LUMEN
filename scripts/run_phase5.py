#!/usr/bin/env python3
"""
Phase 5: Statistical Analysis — v5
====================================
  python scripts/run_phase5.py                 # 完整分析
  python scripts/run_phase5.py --builtin-only  # 只跑內建引擎

v5 changes:
  - Iterates ALL measures per outcome (ADAS-Cog, MMSE, MoCA each get analysis)
  - Iterates both primary AND secondary outcomes
  - Safety outcomes handled descriptively (no meta-analysis)
  - Subgroup labels sanitized (no NoneType crash)
  - Robust error handling throughout
"""

import sys, json, logging, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.langfuse_client import log_phase_start, log_phase_end
from src.utils.file_handlers import DataManager
from src.utils.normalizer import SanityChecker, prepare_meta_data
from src.utils.statistics import run_full_meta_analysis
from src.utils.visualizations import forest_plot, funnel_plot, rob_traffic_light, rob_domain_barplot
from src.utils.cache import TokenBudget
from src.agents.statistician import StatisticianAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_subgroup_labels(meta_data, extracted, pico):
    """Build subgroup labels with None-safe handling."""
    subgroup_vars = {}
    for sg in pico.get("subgroup_analyses", []):
        labels = []
        for md in meta_data:
            s = next((x for x in extracted if x.get("study_id") == md["study_id"]), {})
            var = sg["variable"]
            
            if var == "Stimulation type":
                val = s.get("intervention", {}).get("type")
            elif var == "Population":
                val = s.get("population", {}).get("diagnosis")
            elif var == "Stimulation target":
                val = s.get("intervention", {}).get("target_area")
            elif var == "Session count":
                n_sessions = s.get("intervention", {}).get("sessions_total")
                if n_sessions is not None:
                    try:
                        val = "≤10 sessions" if int(n_sessions) <= 10 else ">10 sessions"
                    except (ValueError, TypeError):
                        val = None
                else:
                    val = None
            elif var == "Combined with cognitive training":
                combined = s.get("intervention", {}).get("combined_with", "none")
                if combined and combined.lower() not in ["none", "n/a", "", "null"]:
                    val = "Yes"
                else:
                    val = "No"
            else:
                val = None
            
            # Sanitize: None → "unknown", empty string → "unknown"
            labels.append(str(val).strip() if val else "unknown")
        
        # Only include if ≥2 distinct non-unknown groups
        unique_real = set(labels) - {"unknown"}
        if len(unique_real) >= 2:
            subgroup_vars[sg["variable"]] = labels
    
    return subgroup_vars


def _run_single_analysis(meta_data, subgroup_vars, outcome_name, measure,
                         effect_type, all_results, fig_dir):
    """Run meta-analysis for one measure, generate figures."""
    key = f"{outcome_name}_{measure}".replace(" ", "_")
    
    try:
        results = run_full_meta_analysis(meta_data, subgroup_vars)
    except Exception as e:
        logger.error(f"Meta-analysis failed for {key}: {e}")
        return
    
    results["measure"] = measure
    results["outcome_name"] = outcome_name
    results["k"] = len(meta_data)
    all_results[key] = results
    
    # --- Figures ---
    pa = results.get("primary_analysis", {})
    if not pa:
        return
    
    # Weights: try DL, then fallback to equal weights
    weights = results.get("random_effects_dl", {}).get("weights_percent")
    if weights is None or len(weights) != len(meta_data):
        weights = [100.0 / len(meta_data)] * len(meta_data)
    
    study_vis = []
    for md, w in zip(meta_data, weights):
        eff = md["effect"]
        se = md["se"]
        study_vis.append({
            "citation": md.get("citation", md["study_id"]),
            "effect": eff,
            "ci_lower": md.get("ci_lower", eff - 1.96 * se),
            "ci_upper": md.get("ci_upper", eff + 1.96 * se),
            "weight_pct": w,
        })
    
    try:
        forest_plot(study_vis, pa,
                    str(fig_dir / f"forest_{key}.png"),
                    f"{outcome_name} ({measure})", effect_type)
    except Exception as e:
        logger.warning(f"Forest plot failed for {key}: {e}")
    
    try:
        eff_arr = np.array([md["effect"] for md in meta_data])
        se_arr = np.array([md["se"] for md in meta_data])
        funnel_plot(eff_arr, se_arr, pa["pooled_effect"],
                    str(fig_dir / f"funnel_{key}.png"),
                    f"Funnel Plot — {measure}", effect_type,
                    results.get("eggers_test", {}).get("intercept_p"))
    except Exception as e:
        logger.warning(f"Funnel plot failed for {key}: {e}")


def _summarize_safety(extracted, dm):
    """Summarize adverse events descriptively (no meta-analysis)."""
    safety_data = []
    for s in extracted:
        ae = s.get("adverse_events", {})
        if not ae:
            continue
        # Check if any AE data exists
        has_data = any(v is not None for v in ae.values() if not isinstance(v, str))
        if has_data:
            safety_data.append({
                "study_id": s.get("study_id"),
                **ae,
            })
    
    if safety_data:
        dm.save("phase5_analysis", "safety_summary.json", safety_data)
        logger.info(f"Safety data: {len(safety_data)} studies with AE data (descriptive)")
    else:
        logger.info("Safety: no adverse event data found")
    
    return safety_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--builtin-only", action="store_true")
    args = parser.parse_args()
    
    select_project()
    dm = DataManager()
    budget = TokenBudget(phase="phase5", limit_usd=5.0)

    extracted = dm.load("phase4_extraction", "extracted_data.json")
    pico = dm.load("input", "pico.yaml")
    rob_data = dm.load("phase4_extraction", "risk_of_bias.json") if dm.exists("phase4_extraction", "risk_of_bias.json") else []

    # Langfuse: phase start stamp
    import yaml
    from pathlib import Path as _Path
    _models_cfg = yaml.safe_load((_Path(__file__).parent.parent / "config/models.yaml").read_text())
    _lf_span = log_phase_start("phase5_analysis", {
        "input_study_count": len(extracted),
        "statistician_model": _models_cfg["models"]["statistician"]["model_id"],
        "budget_usd":         5.0,
        "builtin_only":       args.builtin_only,
    })

    # --- Sanity check ---
    sanity = SanityChecker.check_all(extracted)
    dm.save("phase5_analysis", "sanity_check.json", sanity)
    logger.info(f"Sanity: {sanity['clean']} clean, {sanity['with_warnings']} warn, {sanity['with_errors']} err")
    
    # --- Setup ---
    all_results = {}
    fig_dir = Path(get_data_dir()) / "phase5_analysis" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Combine primary + secondary outcomes ---
    outcome_defs = []
    for o in pico["pico"]["outcome"].get("primary", []):
        outcome_defs.append({**o, "_priority": "primary"})
    for o in pico["pico"]["outcome"].get("secondary", []):
        outcome_defs.append({**o, "_priority": "secondary"})
    
    # --- Run meta-analysis for each outcome × measure ---
    for outcome_def in outcome_defs:
        outcome_name = outcome_def["name"]
        measures = outcome_def.get("measures", [])
        effect_type = "SMD" if "SMD" in outcome_def.get("metric", "SMD") else "MD"
        priority = outcome_def["_priority"]
        
        # Safety → descriptive only
        if any(kw in outcome_name.lower() for kw in ["safety", "tolerability", "adverse"]):
            safety_data = _summarize_safety(extracted, dm)
            continue
        
        logger.info(f"\n--- [{priority}] {outcome_name} ---")
        
        for measure in measures:
            meta_data = prepare_meta_data(extracted, outcome_name, measure, effect_type)
            
            if len(meta_data) < 2:
                logger.info(f"  {measure}: {len(meta_data)} studies — need ≥2, skipping")
                continue
            
            logger.info(f"  {measure}: {len(meta_data)} studies → running meta-analysis")
            
            subgroup_vars = _build_subgroup_labels(meta_data, extracted, pico)
            _run_single_analysis(meta_data, subgroup_vars, outcome_name, measure,
                                effect_type, all_results, fig_dir)
    
    # --- RoB figures ---
    if rob_data:
        try:
            rob_traffic_light(rob_data)
        except Exception as e:
            logger.warning(f"RoB traffic light failed: {e}")
        try:
            rob_domain_barplot(rob_data)
        except Exception as e:
            logger.warning(f"RoB barplot failed: {e}")
    
    # --- Save ---
    dm.save("phase5_analysis", "statistical_results.json", all_results)
    
    # --- Optional LLM supplementary code ---
    if not args.builtin_only:
        try:
            stat_agent = StatisticianAgent(budget=budget)
            code = stat_agent.generate_analysis_code(
                extracted, pico, pico.get("subgroup_analyses", []))
            Path(get_data_dir(), "phase5_analysis/supplementary_analysis.py").write_text(
                code, encoding="utf-8")
        except Exception as e:
            logger.warning(f"LLM code generation skipped: {e}")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("✅ Phase 5 Complete!")
    print("=" * 60)
    
    if not all_results:
        print("\n  ⚠️  No analyses could be run (insufficient data)")
        print("  Check: Are PDFs properly downloaded and extracted?")
        print("  Check: Does extracted_data.json have outcome data with mean/sd/n?")
    else:
        for key, res in all_results.items():
            pa = res.get("primary_analysis", {})
            het = pa.get("heterogeneity", {})
            k = res.get("k", "?")
            priority = "🔵" if any(
                key.startswith(o["name"].replace(" ", "_"))
                for o in pico["pico"]["outcome"].get("primary", [])
            ) else "⚪"
            print(f"\n  {priority} {key}:")
            print(f"     Pooled = {pa.get('pooled_effect', 'N/A'):.4f} "
                  f"[{pa.get('ci_lower', ''):.4f}, {pa.get('ci_upper', ''):.4f}]"
                  if isinstance(pa.get('pooled_effect'), (int, float)) else
                  f"     Pooled = {pa.get('pooled_effect', 'N/A')}")
            print(f"     p = {pa.get('p_value', 'N/A')}, "
                  f"I² = {het.get('I2', 'N/A')}%, k = {k}")
    
    print(f"\n💰 {json.dumps(budget.summary(), indent=2)}")
    print(f"\nNext step: python scripts/run_phase6.py")

    # Langfuse: phase end stamp
    _bsum = budget.summary()
    log_phase_end(_lf_span, "phase5_analysis", {
        "analyses_run":       len(all_results),
        "input_studies":      len(extracted),
        "total_cost_usd":     _bsum.get("total_cost_usd", 0),
        "cache_savings_usd":  _bsum.get("cache_savings_usd", 0),
    })


if __name__ == "__main__":
    main()
