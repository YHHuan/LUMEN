"""
Inter-Rater Reliability Metrics — LUMEN v2
============================================
Computes agreement statistics between:
- Screener 1 vs Screener 2 (Phase 3 dual screening)
- Extraction passes (Phase 4 self-consistency)
- Human vs LLM decisions
- Multi-rater scenarios (Fleiss' kappa)

Metrics:
- Cohen's kappa (2 raters, categorical)
- Weighted kappa (ordinal scales)
- Fleiss' kappa (3+ raters)
- Percent agreement
- Prevalence-adjusted bias-adjusted kappa (PABAK)
- Specific agreement (positive/negative)
"""

import logging
import math
from collections import Counter
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


# ======================================================================
# Cohen's Kappa (2 raters)
# ======================================================================

def cohens_kappa(rater1: List[str], rater2: List[str]) -> dict:
    """
    Compute Cohen's kappa for two raters.

    Args:
        rater1: List of categorical decisions from rater 1
        rater2: List of categorical decisions from rater 2

    Returns:
        dict with kappa, p_observed, p_expected, interpretation
    """
    if len(rater1) != len(rater2):
        raise ValueError("Rater lists must have equal length")

    n = len(rater1)
    if n == 0:
        return {"kappa": 0.0, "n": 0}

    # Categories
    categories = sorted(set(rater1) | set(rater2))

    # Contingency table
    table = {}
    for cat1 in categories:
        table[cat1] = {}
        for cat2 in categories:
            table[cat1][cat2] = 0

    for r1, r2 in zip(rater1, rater2):
        table[r1][r2] += 1

    # Observed agreement
    p_o = sum(table[c][c] for c in categories) / n

    # Expected agreement
    p_e = 0.0
    for c in categories:
        row_sum = sum(table[c].values()) / n
        col_sum = sum(table[r][c] for r in categories) / n
        p_e += row_sum * col_sum

    # Kappa
    if p_e >= 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    return {
        "kappa": round(kappa, 4),
        "p_observed": round(p_o, 4),
        "p_expected": round(p_e, 4),
        "n": n,
        "n_categories": len(categories),
        "categories": categories,
        "interpretation": _interpret_kappa(kappa),
    }


def weighted_kappa(rater1: List[int], rater2: List[int],
                   weights: str = "linear") -> dict:
    """
    Weighted kappa for ordinal scales (e.g., 1-5 confidence scores).

    Args:
        rater1, rater2: Lists of ordinal ratings
        weights: "linear" or "quadratic"
    """
    if len(rater1) != len(rater2):
        raise ValueError("Rater lists must have equal length")

    n = len(rater1)
    if n == 0:
        return {"weighted_kappa": 0.0, "n": 0}

    categories = sorted(set(rater1) | set(rater2))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Weight matrix
    w = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            diff = abs(i - j) / max(k - 1, 1)
            if weights == "quadratic":
                w[i][j] = diff ** 2
            else:
                w[i][j] = diff

    # Observed and expected weighted disagreement
    obs_table = [[0] * k for _ in range(k)]
    for r1, r2 in zip(rater1, rater2):
        obs_table[cat_idx[r1]][cat_idx[r2]] += 1

    row_sums = [sum(obs_table[i]) for i in range(k)]
    col_sums = [sum(obs_table[j][i] for j in range(k)) for i in range(k)]

    w_observed = sum(
        w[i][j] * obs_table[i][j] / n
        for i in range(k) for j in range(k)
    )

    w_expected = sum(
        w[i][j] * row_sums[i] * col_sums[j] / (n * n)
        for i in range(k) for j in range(k)
    )

    if w_expected == 0:
        wk = 1.0
    else:
        wk = 1.0 - w_observed / w_expected

    return {
        "weighted_kappa": round(wk, 4),
        "weight_type": weights,
        "n": n,
        "n_categories": k,
        "interpretation": _interpret_kappa(wk),
    }


# ======================================================================
# Fleiss' Kappa (3+ raters)
# ======================================================================

def fleiss_kappa(ratings_matrix: List[List[int]]) -> dict:
    """
    Compute Fleiss' kappa for multiple raters.

    Args:
        ratings_matrix: NxK matrix where N=items, K=categories.
                       Each cell = number of raters who assigned that category.

    Returns:
        dict with kappa, interpretation
    """
    n_items = len(ratings_matrix)
    if n_items == 0:
        return {"kappa": 0.0, "n_items": 0}

    n_raters = sum(ratings_matrix[0])
    n_cats = len(ratings_matrix[0])

    # P_j: proportion of all assignments to category j
    p_j = []
    for j in range(n_cats):
        total_j = sum(row[j] for row in ratings_matrix)
        p_j.append(total_j / (n_items * n_raters))

    # P_i: extent of agreement for each item
    p_i = []
    for row in ratings_matrix:
        sum_sq = sum(r * r for r in row)
        pi = (sum_sq - n_raters) / (n_raters * (n_raters - 1))
        p_i.append(pi)

    p_bar = sum(p_i) / n_items
    p_e = sum(pj * pj for pj in p_j)

    if p_e >= 1.0:
        kappa = 1.0
    else:
        kappa = (p_bar - p_e) / (1.0 - p_e)

    return {
        "kappa": round(kappa, 4),
        "p_observed": round(p_bar, 4),
        "p_expected": round(p_e, 4),
        "n_items": n_items,
        "n_raters": n_raters,
        "n_categories": n_cats,
        "interpretation": _interpret_kappa(kappa),
    }


# ======================================================================
# PABAK (Prevalence-Adjusted Bias-Adjusted Kappa)
# ======================================================================

def pabak(rater1: List[str], rater2: List[str]) -> dict:
    """
    Prevalence-adjusted bias-adjusted kappa.
    Addresses the kappa paradox in high-prevalence settings.
    """
    n = len(rater1)
    if n == 0:
        return {"pabak": 0.0, "n": 0}

    agree = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == r2)
    p_o = agree / n

    pabak_value = 2 * p_o - 1

    return {
        "pabak": round(pabak_value, 4),
        "p_observed": round(p_o, 4),
        "n": n,
        "interpretation": _interpret_kappa(pabak_value),
    }


# ======================================================================
# Specific Agreement
# ======================================================================

def specific_agreement(rater1: List[str], rater2: List[str],
                       positive_label: str = "include") -> dict:
    """
    Compute positive and negative specific agreement.

    Useful when prevalence is very high or low (kappa paradox).
    """
    n = len(rater1)
    if n == 0:
        return {}

    # 2x2 contingency table
    a = sum(1 for r1, r2 in zip(rater1, rater2)
            if r1 == positive_label and r2 == positive_label)
    b = sum(1 for r1, r2 in zip(rater1, rater2)
            if r1 == positive_label and r2 != positive_label)
    c = sum(1 for r1, r2 in zip(rater1, rater2)
            if r1 != positive_label and r2 == positive_label)
    d = sum(1 for r1, r2 in zip(rater1, rater2)
            if r1 != positive_label and r2 != positive_label)

    pos_agree = 2 * a / max(2 * a + b + c, 1)
    neg_agree = 2 * d / max(2 * d + b + c, 1)

    return {
        "positive_agreement": round(pos_agree, 4),
        "negative_agreement": round(neg_agree, 4),
        "contingency": {"a": a, "b": b, "c": c, "d": d},
        "n": n,
    }


# ======================================================================
# Pipeline-Specific Agreement Analysis
# ======================================================================

def compute_screening_agreement(screening_results: list) -> dict:
    """
    Compute inter-rater agreement for dual screening (Phase 3).

    Expects screening results with screener_1_decision and screener_2_decision.
    """
    r1_decisions = []
    r2_decisions = []
    r1_scores = []
    r2_scores = []

    for study in screening_results:
        s1 = study.get("screener_1", {})
        s2 = study.get("screener_2", {})

        d1 = s1.get("decision", s1.get("recommendation", ""))
        d2 = s2.get("decision", s2.get("recommendation", ""))

        if d1 and d2:
            r1_decisions.append(d1)
            r2_decisions.append(d2)

        # Confidence scores (1-5 scale)
        c1 = s1.get("confidence", s1.get("score"))
        c2 = s2.get("confidence", s2.get("score"))
        if c1 is not None and c2 is not None:
            r1_scores.append(int(c1))
            r2_scores.append(int(c2))

    result = {
        "n_dual_screened": len(r1_decisions),
        "percent_agreement": round(
            sum(1 for a, b in zip(r1_decisions, r2_decisions) if a == b)
            / max(len(r1_decisions), 1), 4
        ),
    }

    if r1_decisions:
        result["cohens_kappa"] = cohens_kappa(r1_decisions, r2_decisions)
        result["pabak"] = pabak(r1_decisions, r2_decisions)
        result["specific_agreement"] = specific_agreement(
            r1_decisions, r2_decisions, positive_label="include"
        )

    if r1_scores:
        result["weighted_kappa_linear"] = weighted_kappa(
            r1_scores, r2_scores, weights="linear"
        )
        result["weighted_kappa_quadratic"] = weighted_kappa(
            r1_scores, r2_scores, weights="quadratic"
        )

    return result


def compute_extraction_consistency(extracted_data: list) -> dict:
    """
    Compute extraction pass consistency (Phase 4 self-consistency).

    Checks agreement across extraction passes for key numeric fields.
    """
    consistent = 0
    inconsistent = 0
    field_agreement: Dict[str, List[bool]] = {}

    for study in extracted_data:
        passes = study.get("extraction_passes", [])
        if len(passes) < 2:
            continue

        for field in ["total_n", "study_design"]:
            values = [p.get(field) for p in passes if p.get(field) is not None]
            if len(values) >= 2:
                all_same = all(v == values[0] for v in values)
                if field not in field_agreement:
                    field_agreement[field] = []
                field_agreement[field].append(all_same)

                if all_same:
                    consistent += 1
                else:
                    inconsistent += 1

    total = consistent + inconsistent
    result = {
        "total_comparisons": total,
        "consistent": consistent,
        "inconsistent": inconsistent,
        "consistency_rate": round(consistent / max(total, 1), 4),
        "per_field": {},
    }

    for field, agreements in field_agreement.items():
        result["per_field"][field] = {
            "n": len(agreements),
            "agreement_rate": round(
                sum(agreements) / max(len(agreements), 1), 4
            ),
        }

    return result


# ======================================================================
# Interpretation
# ======================================================================

def _interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) interpretation scale."""
    if kappa < 0:
        return "Poor"
    if kappa < 0.20:
        return "Slight"
    if kappa < 0.40:
        return "Fair"
    if kappa < 0.60:
        return "Moderate"
    if kappa < 0.80:
        return "Substantial"
    return "Almost perfect"


def format_agreement_report(
    screening_agreement: dict = None,
    extraction_consistency: dict = None,
    human_ai_agreement: dict = None,
) -> str:
    """Format a comprehensive agreement report."""
    lines = [
        "=" * 60,
        "  LUMEN v2 — Inter-Rater Reliability Report",
        "=" * 60,
        "",
    ]

    if screening_agreement:
        sa = screening_agreement
        lines.extend([
            "  Dual Screening Agreement (Phase 3):",
            "  " + "-" * 50,
            f"  Studies dual-screened:    {sa['n_dual_screened']}",
            f"  Percent agreement:       {sa['percent_agreement']:.1%}",
        ])
        if "cohens_kappa" in sa:
            ck = sa["cohens_kappa"]
            lines.append(
                f"  Cohen's kappa:           {ck['kappa']:.3f} ({ck['interpretation']})"
            )
        if "weighted_kappa_quadratic" in sa:
            wk = sa["weighted_kappa_quadratic"]
            lines.append(
                f"  Weighted kappa (quad):   {wk['weighted_kappa']:.3f} ({wk['interpretation']})"
            )
        if "pabak" in sa:
            lines.append(
                f"  PABAK:                   {sa['pabak']['pabak']:.3f}"
            )
        lines.append("")

    if extraction_consistency:
        ec = extraction_consistency
        lines.extend([
            "  Extraction Consistency (Phase 4):",
            "  " + "-" * 50,
            f"  Total comparisons:       {ec['total_comparisons']}",
            f"  Consistency rate:        {ec['consistency_rate']:.1%}",
        ])
        for field, data in ec.get("per_field", {}).items():
            lines.append(
                f"    {field:<25} {data['agreement_rate']:.1%} ({data['n']} comparisons)"
            )
        lines.append("")

    if human_ai_agreement:
        ha = human_ai_agreement.get("overall", {})
        lines.extend([
            "  Human-AI Agreement:",
            "  " + "-" * 50,
            f"  Total reviews:           {ha.get('total_reviews', 0)}",
            f"  Agreement rate:          {ha.get('agreement_rate', 0):.1%}",
            f"  Overrides:               {ha.get('overrides', 0)}",
        ])
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
