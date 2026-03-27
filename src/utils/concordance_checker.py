"""
Concordance Checker — LUMEN v2
================================
Compares LUMEN outputs against published ground truth for:
  1. RoB agreement — Cohen's kappa per domain (Table 9)
  2. Synthesis concordance — effect size, CI overlap, conclusion (Figure 7)

Ground truth files:
  - RoB: data/<domain>/quality_assessment/rob_ground_truth.json
  - Synthesis: data/<domain>/phase5_analysis/ground_truth_estimates.json
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# RoB Agreement (Table 9)
# ======================================================================

def compare_rob_assessments(
    lumen_assessments: List[dict],
    ground_truth: List[dict],
    tool: str = "rob2",
) -> Dict:
    """
    Compute Cohen's kappa between LUMEN RoB assessments and published GT.

    Args:
        lumen_assessments: LUMEN RoB output (rob2_assessments.json or robins_i_assessments.json)
        ground_truth: published RoB judgments in same format
        tool: "rob2" or "robins_i"

    Returns:
        overall_kappa, per_domain_kappa, confusion_matrix
    """
    from src.utils.agreement import cohens_kappa

    gt_by_id = {}
    for a in ground_truth:
        gt_by_id[a.get("study_id", "")] = a

    # Collect paired judgments
    overall_lumen = []
    overall_gt = []
    domain_pairs = {}

    if tool == "rob2":
        domain_ids = ["D1", "D2", "D3", "D4", "D5"]
    else:
        domain_ids = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]

    for d_id in domain_ids:
        domain_pairs[d_id] = {"lumen": [], "gt": []}

    for la in lumen_assessments:
        sid = la.get("study_id", "")
        ga = gt_by_id.get(sid)
        if not ga:
            continue

        # Overall judgment
        lj = la.get("overall_judgment", "")
        gj = ga.get("overall_judgment", "")
        if lj and gj:
            overall_lumen.append(lj)
            overall_gt.append(gj)

        # Per-domain judgments
        for d_id in domain_ids:
            l_dom = la.get("domains", {}).get(d_id, {})
            g_dom = ga.get("domains", {}).get(d_id, {})
            lj_d = l_dom.get("judgment", "")
            gj_d = g_dom.get("judgment", "")
            if lj_d and gj_d:
                domain_pairs[d_id]["lumen"].append(lj_d)
                domain_pairs[d_id]["gt"].append(gj_d)

    # Compute kappas
    overall_kappa = (
        cohens_kappa(overall_lumen, overall_gt)
        if len(overall_lumen) >= 2 else None
    )

    per_domain_kappa = {}
    for d_id in domain_ids:
        dp = domain_pairs[d_id]
        if len(dp["lumen"]) >= 2:
            per_domain_kappa[d_id] = cohens_kappa(dp["lumen"], dp["gt"])
        else:
            per_domain_kappa[d_id] = None

    return {
        "tool": tool,
        "n_matched_studies": len(overall_lumen),
        "overall_kappa": round(overall_kappa, 4) if overall_kappa is not None else None,
        "per_domain_kappa": {
            k: round(v, 4) if v is not None else None
            for k, v in per_domain_kappa.items()
        },
        "overall_agreement": _agreement_rate(overall_lumen, overall_gt),
    }


def _agreement_rate(a: list, b: list) -> Optional[float]:
    if not a:
        return None
    agreed = sum(1 for x, y in zip(a, b) if x == y)
    return round(agreed / len(a), 4)


def format_rob_agreement_table(results: List[Dict]) -> str:
    """Format Table 9: RoB agreement across domains."""
    # Determine domain columns from first result
    if not results:
        return ""

    tool = results[0].get("tool", "rob2")
    if tool == "rob2":
        domains = ["D1", "D2", "D3", "D4", "D5"]
        headers = ["Domain", "Overall κ"] + [f"{d} κ" for d in domains]
    else:
        domains = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
        headers = ["Domain", "Overall κ"] + [f"{d} κ" for d in domains]

    lines = [
        "# Table 9: Risk-of-Bias Agreement",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["-----"] * len(headers)) + "|",
    ]

    for r in results:
        vals = [r.get("domain_label", "")]
        k = r.get("overall_kappa")
        vals.append(f"{k:.3f}" if k is not None else "N/A")
        for d in domains:
            dk = r.get("per_domain_kappa", {}).get(d)
            vals.append(f"{dk:.3f}" if dk is not None else "N/A")
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


# ======================================================================
# Synthesis Concordance (Figure 7)
# ======================================================================

def compare_synthesis(
    lumen_results: dict,
    ground_truth_estimates: List[dict],
) -> List[Dict]:
    """
    Compare LUMEN pooled estimates against published ground truth.

    Args:
        lumen_results: phase5_analysis/planned_results.json
        ground_truth_estimates: list of published estimates with fields:
            analysis_id, published_effect, published_ci_lower,
            published_ci_upper, published_measure, published_I2,
            published_conclusion

    Returns:
        list of comparison dicts per analysis
    """
    comparisons = []

    # Extract LUMEN analyses
    lumen_analyses = {}
    for analysis in lumen_results.get("analyses", []):
        aid = analysis.get("analysis_id", "")
        if aid:
            lumen_analyses[aid] = analysis

    for gt in ground_truth_estimates:
        aid = gt.get("analysis_id", "")
        lumen = lumen_analyses.get(aid)

        if not lumen:
            comparisons.append({
                "analysis_id": aid,
                "matched": False,
            })
            continue

        # Extract LUMEN values
        l_effect = lumen.get("pooled_effect")
        l_ci_lo = lumen.get("ci_lower")
        l_ci_hi = lumen.get("ci_upper")
        l_i2 = lumen.get("I2")

        # GT values
        g_effect = gt.get("published_effect")
        g_ci_lo = gt.get("published_ci_lower")
        g_ci_hi = gt.get("published_ci_upper")
        g_i2 = gt.get("published_I2")
        g_conclusion = gt.get("published_conclusion", "")

        # Effect size difference
        es_diff = None
        if l_effect is not None and g_effect is not None:
            es_diff = round(abs(l_effect - g_effect), 4)

        # CI overlap
        ci_overlap = _compute_ci_overlap(l_ci_lo, l_ci_hi, g_ci_lo, g_ci_hi)

        # Conclusion concordance — use effect_measure to determine null value
        l_measure = lumen.get("effect_measure", gt.get("published_measure", ""))
        l_conclusion = _infer_conclusion(l_effect, l_ci_lo, l_ci_hi, l_measure)
        conclusion_concordant = (
            _conclusions_match(l_conclusion, g_conclusion)
            if l_conclusion and g_conclusion else None
        )

        # I2 difference
        i2_diff = None
        if l_i2 is not None and g_i2 is not None:
            i2_diff = round(abs(l_i2 - g_i2), 1)

        comparisons.append({
            "analysis_id": aid,
            "matched": True,
            "lumen_effect": l_effect,
            "published_effect": g_effect,
            "effect_size_diff": es_diff,
            "lumen_ci": [l_ci_lo, l_ci_hi],
            "published_ci": [g_ci_lo, g_ci_hi],
            "ci_overlap_pct": ci_overlap,
            "lumen_conclusion": l_conclusion,
            "published_conclusion": g_conclusion,
            "conclusion_concordant": conclusion_concordant,
            "lumen_I2": l_i2,
            "published_I2": g_i2,
            "I2_diff": i2_diff,
        })

    return comparisons


def _compute_ci_overlap(l_lo, l_hi, g_lo, g_hi) -> Optional[float]:
    """Compute percentage overlap between two confidence intervals."""
    if any(v is None for v in [l_lo, l_hi, g_lo, g_hi]):
        return None

    overlap_lo = max(l_lo, g_lo)
    overlap_hi = min(l_hi, g_hi)

    if overlap_lo >= overlap_hi:
        return 0.0

    overlap_width = overlap_hi - overlap_lo
    union_width = max(l_hi, g_hi) - min(l_lo, g_lo)

    if union_width <= 0:
        return 100.0

    return round(overlap_width / union_width * 100, 1)


RATIO_MEASURES = {"OR", "RR", "HR", "or", "rr", "hr"}
DIFFERENCE_MEASURES = {"SMD", "MD", "RD", "WMD", "smd", "md", "rd", "wmd"}


def _infer_conclusion(effect, ci_lo, ci_hi, effect_measure: str = "") -> str:
    """Infer statistical conclusion from effect + CI + measure type."""
    if any(v is None for v in [effect, ci_lo, ci_hi]):
        return ""

    # Determine null value from effect_measure
    measure_upper = effect_measure.strip().upper()
    if measure_upper in {"OR", "RR", "HR"}:
        null = 1.0
    elif measure_upper in {"SMD", "MD", "RD", "WMD", "VE"}:
        null = 0.0
    else:
        # Fallback: if measure not specified, infer from value range
        # Log-scale ratios are typically around 0, raw ratios around 1
        null = 0.0

    if ci_lo <= null <= ci_hi:
        return "no significant difference"
    elif effect > null:
        return "significant increase"
    else:
        return "significant decrease"


def _conclusions_match(c1: str, c2: str) -> bool:
    """Check if two conclusions are concordant."""
    c1 = c1.lower().strip()
    c2 = c2.lower().strip()

    sig_terms = ["significant", "difference", "increase", "decrease",
                 "reduction", "improvement", "superior", "inferior"]
    no_sig = ["no significant", "not significant", "no difference",
              "non-significant", "nonsignificant"]

    c1_sig = any(t in c1 for t in no_sig)
    c2_sig = any(t in c2 for t in no_sig)

    # Both significant or both non-significant
    return c1_sig == c2_sig


def format_synthesis_comparison(comparisons: List[Dict]) -> str:
    """Format synthesis concordance table."""
    lines = [
        "# Synthesis Concordance",
        "",
        "| Analysis | ES Diff | CI Overlap (%) | Conclusion Match | I² Diff |",
        "|----------|---------|---------------|-----------------|---------|",
    ]

    for c in comparisons:
        if not c.get("matched"):
            lines.append(f"| {c['analysis_id']} | — | — | not matched | — |")
            continue

        es = f"{c['effect_size_diff']:.4f}" if c.get("effect_size_diff") is not None else "N/A"
        ci = f"{c['ci_overlap_pct']:.1f}" if c.get("ci_overlap_pct") is not None else "N/A"
        conc = "Yes" if c.get("conclusion_concordant") else ("No" if c.get("conclusion_concordant") is False else "N/A")
        i2 = f"{c['I2_diff']:.1f}" if c.get("I2_diff") is not None else "N/A"

        lines.append(f"| {c['analysis_id']} | {es} | {ci} | {conc} | {i2} |")

    return "\n".join(lines)
