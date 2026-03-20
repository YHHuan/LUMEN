"""
Effect Size Calculations — v5
==============================
v5 CRITICAL FIX: auto_compute_effect now handles LLM key variants:
  intervention_mean_post / intervention_mean_change / intervention_mean
  → all resolved to intervention_mean before computation.

Reference:
- Borenstein et al. (2009)
- Cochrane Handbook Ch. 6 & 10
"""

import numpy as np
from scipy import stats as sp_stats
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# 1. SE Back-Calculation
# ======================================================================

def se_from_ci(lower: float, upper: float, level: float = 0.95) -> float:
    z = sp_stats.norm.ppf(1 - (1 - level) / 2)
    return (upper - lower) / (2 * z)

def se_from_ci_log(lower: float, upper: float, level: float = 0.95) -> float:
    z = sp_stats.norm.ppf(1 - (1 - level) / 2)
    return (np.log(upper) - np.log(lower)) / (2 * z)

def se_from_p_value(effect: float, p_value: float, is_log_scale: bool = False) -> Optional[float]:
    if p_value <= 0 or p_value >= 1:
        return None
    z = sp_stats.norm.ppf(1 - p_value / 2)
    return abs(np.log(effect) if is_log_scale else effect) / z


# ======================================================================
# 2. Core Effect Size Functions
# ======================================================================

def hedges_g(m1, sd1, n1, m2, sd2, n2) -> dict:
    m1, sd1, n1 = float(m1), float(sd1), int(n1)
    m2, sd2, n2 = float(m2), float(sd2), int(n2)
    sd_pooled = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
    if sd_pooled == 0:
        return {"effect_type": "SMD", "effect": 0, "se": None}
    d = (m1 - m2) / sd_pooled
    df = n1 + n2 - 2
    j = 1 - 3 / (4*df - 1)
    g = d * j
    se = np.sqrt(1/n1 + 1/n2 + g**2 / (2*(n1+n2)))
    return {
        "effect_type": "SMD", "effect": round(g, 4),
        "se": round(se, 4), "variance": round(se**2, 6),
        "ci_lower": round(g - 1.96*se, 4), "ci_upper": round(g + 1.96*se, 4),
        "cohens_d": round(d, 4), "hedges_correction": round(j, 4),
    }

def mean_difference(m1, sd1, n1, m2, sd2, n2) -> dict:
    m1, sd1, n1 = float(m1), float(sd1), int(n1)
    m2, sd2, n2 = float(m2), float(sd2), int(n2)
    md = m1 - m2
    se = np.sqrt(sd1**2/n1 + sd2**2/n2)
    return {
        "effect_type": "MD", "effect": round(md, 4),
        "se": round(se, 4), "variance": round(se**2, 6),
        "ci_lower": round(md - 1.96*se, 4), "ci_upper": round(md + 1.96*se, 4),
    }

def odds_ratio(e1, t1, e2, t2, correction=0.5) -> dict:
    a, c, b, d = e1, e2, t1-e1, t2-e2
    if a==0 or b==0 or c==0 or d==0:
        a += correction; b += correction; c += correction; d += correction
    or_val = (a*d)/(b*c)
    log_or = np.log(or_val)
    se_log = np.sqrt(1/a + 1/b + 1/c + 1/d)
    return {
        "effect_type": "OR", "effect": round(or_val, 4),
        "log_effect": round(log_or, 4), "se_log": round(se_log, 4),
        "variance_log": round(se_log**2, 6),
        "ci_lower": round(np.exp(log_or - 1.96*se_log), 4),
        "ci_upper": round(np.exp(log_or + 1.96*se_log), 4),
    }

def risk_ratio(e1, t1, e2, t2, correction=0.5) -> dict:
    a, c, n1, n2 = e1, e2, t1, t2
    if a==0 or c==0:
        a += correction; c += correction; n1 += 2*correction; n2 += 2*correction
    rr = (a/n1)/(c/n2)
    log_rr = np.log(rr)
    se_log = np.sqrt(1/a - 1/n1 + 1/c - 1/n2)
    return {
        "effect_type": "RR", "effect": round(rr, 4),
        "log_effect": round(log_rr, 4), "se_log": round(se_log, 4),
        "variance_log": round(se_log**2, 6),
        "ci_lower": round(np.exp(log_rr - 1.96*se_log), 4),
        "ci_upper": round(np.exp(log_rr + 1.96*se_log), 4),
    }

def risk_difference(e1, t1, e2, t2) -> dict:
    p1, p2 = e1/t1, e2/t2
    rd = p1-p2
    se = np.sqrt(p1*(1-p1)/t1 + p2*(1-p2)/t2)
    return {
        "effect_type": "RD", "effect": round(rd, 4), "se": round(se, 4),
        "variance": round(se**2, 6),
        "ci_lower": round(rd-1.96*se, 4), "ci_upper": round(rd+1.96*se, 4),
    }


# ======================================================================
# 3. Conversion
# ======================================================================

def or_to_smd(log_or, se_log_or):
    smd = log_or * np.sqrt(3) / np.pi
    se = se_log_or * np.sqrt(3) / np.pi
    return {"effect_type": "SMD (from OR)", "effect": round(smd, 4), "se": round(se, 4)}

def smd_to_or(smd, se_smd):
    log_or = smd * np.pi / np.sqrt(3)
    se = se_smd * np.pi / np.sqrt(3)
    return {"effect_type": "OR (from SMD)", "effect": round(np.exp(log_or), 4),
            "log_effect": round(log_or, 4), "se_log": round(se, 4)}


# ======================================================================
# 4. Key Normalization (v5 CRITICAL FIX)
# ======================================================================

def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _normalize_outcome_data(data: dict) -> dict:
    """
    v5 FIX: LLM extractor produces keys like:
      intervention_mean_post, intervention_sd_post,
      intervention_mean_change, intervention_sd_change
    But auto_compute_effect needs:
      intervention_mean, intervention_sd

    Priority: change scores > post scores > direct keys
    """
    norm = dict(data)

    for group in ["intervention", "control"]:
        mean_key = f"{group}_mean"
        sd_key = f"{group}_sd"

        if _safe_float(norm.get(mean_key)) is not None:
            continue  # Direct key exists

        # Prefer change scores (more appropriate for pre-post RCT)
        change_m = _safe_float(norm.get(f"{group}_mean_change"))
        change_sd = _safe_float(norm.get(f"{group}_sd_change"))
        post_m = _safe_float(norm.get(f"{group}_mean_post"))
        post_sd = _safe_float(norm.get(f"{group}_sd_post"))

        if change_m is not None and change_sd is not None and change_sd > 0:
            norm[mean_key] = change_m
            norm[sd_key] = change_sd
            norm["_data_source"] = "change_scores"
        elif post_m is not None and post_sd is not None and post_sd > 0:
            norm[mean_key] = post_m
            norm[sd_key] = post_sd
            norm["_data_source"] = "post_scores"

        # N fallback
        n_key = f"{group}_n"
        if norm.get(n_key) is None:
            norm[n_key] = norm.get(f"{group}_n_analyzed")

    return norm


# ======================================================================
# 5. Auto-Detect & Compute (v5)
# ======================================================================

def auto_compute_effect(data: dict, preferred_type: str = "SMD") -> Optional[dict]:
    """
    v5: normalizes LLM keys (_post/_change → direct) then computes.
    """
    data = _normalize_outcome_data(data)

    # --- Case 1: Mean/SD/N → continuous ---
    m1 = _safe_float(data.get("intervention_mean"))
    s1 = _safe_float(data.get("intervention_sd"))
    n1 = _safe_float(data.get("intervention_n"))
    m2 = _safe_float(data.get("control_mean"))
    s2 = _safe_float(data.get("control_sd"))
    n2 = _safe_float(data.get("control_n"))

    if all(v is not None for v in [m1, s1, n1, m2, s2, n2]) and s1 > 0 and s2 > 0 and n1 > 0 and n2 > 0:
        src = data.get("_data_source", "direct")
        if preferred_type == "MD":
            result = mean_difference(m1, s1, int(n1), m2, s2, int(n2))
            return {**result, "method": f"direct_md_{src}"}
        else:
            result = hedges_g(m1, s1, int(n1), m2, s2, int(n2))
            return {**result, "method": f"direct_hedges_g_{src}"}

    # --- Case 2: Binary ---
    if all(data.get(k) is not None for k in [
        "intervention_events", "intervention_total", "control_events", "control_total"
    ]):
        if preferred_type == "RR":
            r = risk_ratio(int(data["intervention_events"]), int(data["intervention_total"]),
                           int(data["control_events"]), int(data["control_total"]))
            return {**r, "method": "direct_rr"}
        else:
            r = odds_ratio(int(data["intervention_events"]), int(data["intervention_total"]),
                           int(data["control_events"]), int(data["control_total"]))
            return {**r, "method": "direct_or"}

    # --- Case 3: SMD + CI ---
    smd, smd_lo, smd_hi = _safe_float(data.get("smd")), _safe_float(data.get("smd_95ci_lower")), _safe_float(data.get("smd_95ci_upper"))
    if smd is not None and smd_lo is not None and smd_hi is not None:
        se = se_from_ci(smd_lo, smd_hi)
        return {"effect_type": "SMD", "effect": smd, "se": round(se, 4),
                "variance": round(se**2, 6), "ci_lower": smd_lo, "ci_upper": smd_hi,
                "method": "back_calc_smd_ci"}

    # --- Case 4: MD + CI ---
    md, md_lo, md_hi = _safe_float(data.get("mean_difference")), _safe_float(data.get("md_95ci_lower")), _safe_float(data.get("md_95ci_upper"))
    if md is not None and md_lo is not None and md_hi is not None:
        se = se_from_ci(md_lo, md_hi)
        return {"effect_type": "MD", "effect": md, "se": round(se, 4),
                "variance": round(se**2, 6), "ci_lower": md_lo, "ci_upper": md_hi,
                "method": "back_calc_md_ci"}

    # --- Case 4b: MD + SE ---
    md_se = _safe_float(data.get("md_se"))
    if md is not None and md_se is not None and md_se > 0:
        return {"effect_type": "MD", "effect": md, "se": round(md_se, 4),
                "variance": round(md_se**2, 6),
                "ci_lower": round(md - 1.96*md_se, 4), "ci_upper": round(md + 1.96*md_se, 4),
                "method": "direct_md_se"}

    # --- Case 5: Effect + p ---
    for ekey in ["smd", "mean_difference", "hazard_ratio"]:
        ev = _safe_float(data.get(ekey))
        pv = _safe_float(data.get("p_value"))
        if ev is not None and pv is not None and 0 < pv < 1:
            is_log = ekey == "hazard_ratio"
            se = se_from_p_value(ev, pv, is_log_scale=is_log)
            if se and se > 0:
                if is_log:
                    return {"effect_type": "HR", "effect": ev,
                            "log_effect": round(np.log(ev), 4), "se_log": round(se, 4),
                            "method": f"back_calc_p_{ekey}"}
                etype = "SMD" if ekey == "smd" else "MD"
                return {"effect_type": etype, "effect": ev, "se": round(se, 4),
                        "method": f"back_calc_p_{ekey}"}

    logger.warning(f"Could not compute effect: non-null keys = "
                   f"{[k for k, v in data.items() if v is not None and not k.startswith('_')]}")
    return {"effect_type": "unknown", "effect": None, "se": None, "method": "insufficient_data"}


# ======================================================================
# 6. Direction
# ======================================================================

def correct_direction(effect_data: dict, scale_direction: str) -> dict:
    result = dict(effect_data)
    if scale_direction == "lower_better":
        if result.get("effect") is not None:
            result["effect"] = -result["effect"]
        if result.get("ci_lower") is not None and result.get("ci_upper") is not None:
            lo, hi = result["ci_lower"], result["ci_upper"]
            result["ci_lower"], result["ci_upper"] = -hi, -lo
        result["direction_corrected"] = True
        result["original_scale_direction"] = "lower_better"
    else:
        result["direction_corrected"] = False
        result["original_scale_direction"] = "higher_better"
    return result

SCALE_DIRECTIONS = {
    "ADAS-Cog": "lower_better", "ADAS": "lower_better",
    "MMSE": "higher_better", "MoCA": "higher_better",
    "CDR": "lower_better", "CDR-SB": "lower_better",
    "NPI": "lower_better", "GDS": "lower_better",
    "ADL": "higher_better", "IADL": "higher_better", "ADCS-ADL": "higher_better",
    "QoL-AD": "higher_better", "ZBI": "lower_better",
    "HAM-D": "lower_better", "HAMD": "lower_better",
    "BDI": "lower_better", "UPDRS": "lower_better",
}

def get_scale_direction(measure_name: str) -> str:
    if not measure_name:
        return "higher_better"
    for key, direction in SCALE_DIRECTIONS.items():
        if key.lower() in measure_name.lower():
            return direction
    return "higher_better"
