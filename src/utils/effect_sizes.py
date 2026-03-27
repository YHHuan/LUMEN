"""
Effect Size Calculations — LUMEN v2
=====================================
Supports: SMD (Hedges' g), OR, RR, RD, HR, MD, VE%.
Auto-routing based on outcome_type field.
"""

import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# ======================================================================
# Outcome type constants
# ======================================================================
CONTINUOUS = "continuous"
BINARY = "binary"
TIME_TO_EVENT = "time_to_event"
PRECOMPUTED = "precomputed"

# ======================================================================
# Continuous outcomes — SMD (Hedges' g) and MD
# ======================================================================

def compute_hedges_g(m1: float, sd1: float, n1: int,
                      m2: float, sd2: float, n2: int) -> Dict:
    """
    Compute Hedges' g (bias-corrected standardized mean difference).

    Args:
        m1, sd1, n1: intervention group mean, SD, sample size
        m2, sd2, n2: control group mean, SD, sample size

    Returns:
        {"yi": float, "vi": float, "se": float, "measure": "SMD", ...}
    """
    if sd1 <= 0 or sd2 <= 0 or n1 <= 1 or n2 <= 1:
        return {"yi": None, "vi": None, "se": None, "measure": "SMD", "error": "invalid_input"}

    # Pooled SD
    sp = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
    if sp == 0:
        return {"yi": 0.0, "vi": 0.0, "se": 0.0, "measure": "SMD",
                "g": 0.0, "d": 0.0, "J": 1.0}

    # Cohen's d
    d = (m1 - m2) / sp

    # Small-sample bias correction (Hedges' J)
    df = n1 + n2 - 2
    J = 1 - 3 / (4 * df - 1)

    g = d * J
    var_d = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
    var_g = var_d * J ** 2

    return {
        "yi": round(float(g), 6),
        "vi": round(float(var_g), 6),
        "se": round(float(np.sqrt(var_g)), 6),
        "measure": "SMD",
        "g": round(float(g), 4),
        "d": round(float(d), 4),
        "J": round(float(J), 6),
        # Legacy aliases
        "var_g": round(float(var_g), 6),
        "se_g": round(float(np.sqrt(var_g)), 4),
    }


def compute_md(m1: float, sd1: float, n1: int,
               m2: float, sd2: float, n2: int) -> Dict:
    """
    Compute raw mean difference (unstandardized).
    Used when all studies use the same measurement scale.
    """
    if n1 <= 0 or n2 <= 0:
        return {"yi": None, "vi": None, "se": None, "measure": "MD", "error": "invalid_input"}

    md = m1 - m2
    var_md = (sd1 ** 2 / n1) + (sd2 ** 2 / n2)

    return {
        "yi": round(float(md), 6),
        "vi": round(float(var_md), 6),
        "se": round(float(np.sqrt(var_md)), 6),
        "measure": "MD",
    }


# ======================================================================
# Binary outcomes — OR, RR, RD
# ======================================================================

def compute_log_or(e1: int, n1: int, e2: int, n2: int,
                   cc: float = 0.5) -> Dict:
    """
    Compute log odds ratio from a 2x2 table.

    Args:
        e1, n1: events and total in intervention group
        e2, n2: events and total in control group
        cc: continuity correction added when any cell is 0
    """
    if n1 <= 0 or n2 <= 0 or e1 < 0 or e2 < 0 or e1 > n1 or e2 > n2:
        return {"yi": None, "vi": None, "se": None, "measure": "OR", "error": "invalid_input"}

    a, b = float(e1), float(n1 - e1)
    c, d = float(e2), float(n2 - e2)

    # Apply continuity correction if any cell is zero
    if a == 0 or b == 0 or c == 0 or d == 0:
        a += cc; b += cc; c += cc; d += cc

    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return {"yi": None, "vi": None, "se": None, "measure": "OR", "error": "zero_cell"}

    log_or = np.log(a * d / (b * c))
    var_log_or = 1/a + 1/b + 1/c + 1/d

    return {
        "yi": round(float(log_or), 6),
        "vi": round(float(var_log_or), 6),
        "se": round(float(np.sqrt(var_log_or)), 6),
        "measure": "OR",
        "OR": round(float(np.exp(log_or)), 4),
    }


def compute_log_rr(e1: int, n1: int, e2: int, n2: int,
                   cc: float = 0.5) -> Dict:
    """
    Compute log risk ratio from a 2x2 table.

    Args:
        e1, n1: events and total in intervention group
        e2, n2: events and total in control group
        cc: continuity correction added when any cell is 0
    """
    if n1 <= 0 or n2 <= 0 or e1 < 0 or e2 < 0 or e1 > n1 or e2 > n2:
        return {"yi": None, "vi": None, "se": None, "measure": "RR", "error": "invalid_input"}

    a, c = float(e1), float(e2)
    n1f, n2f = float(n1), float(n2)

    if a == 0 or c == 0:
        a += cc; c += cc
        n1f += cc; n2f += cc

    if a <= 0 or c <= 0:
        return {"yi": None, "vi": None, "se": None, "measure": "RR", "error": "zero_cell"}

    log_rr = np.log((a / n1f) / (c / n2f))
    var_log_rr = (1/a - 1/n1f) + (1/c - 1/n2f)

    return {
        "yi": round(float(log_rr), 6),
        "vi": round(float(var_log_rr), 6),
        "se": round(float(np.sqrt(var_log_rr)), 6),
        "measure": "RR",
        "RR": round(float(np.exp(log_rr)), 4),
    }


def compute_rd(e1: int, n1: int, e2: int, n2: int) -> Dict:
    """
    Compute risk difference from a 2x2 table.
    """
    if n1 <= 0 or n2 <= 0 or e1 < 0 or e2 < 0 or e1 > n1 or e2 > n2:
        return {"yi": None, "vi": None, "se": None, "measure": "RD", "error": "invalid_input"}

    p1 = e1 / n1
    p2 = e2 / n2
    rd = p1 - p2
    var_rd = (p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2)

    return {
        "yi": round(float(rd), 6),
        "vi": round(float(var_rd), 6),
        "se": round(float(np.sqrt(var_rd)), 6),
        "measure": "RD",
        "RD": round(float(rd), 4),
    }


# ======================================================================
# Time-to-event — HR (pre-computed from papers)
# ======================================================================

def use_precomputed_hr(hr: float, ci_lower: float, ci_upper: float) -> Dict:
    """
    Convert a reported HR + 95% CI to log-HR + variance for meta-analysis.
    Most HR data is reported directly; we just need log-scale conversion.
    """
    if hr is None or hr <= 0 or ci_lower is None or ci_upper is None:
        return {"yi": None, "vi": None, "se": None, "measure": "HR", "error": "invalid_input"}

    log_hr = np.log(hr)
    # SE from CI: (log(upper) - log(lower)) / (2 * 1.96)
    se_log_hr = (np.log(ci_upper) - np.log(ci_lower)) / (2 * 1.96)
    var_log_hr = se_log_hr ** 2

    return {
        "yi": round(float(log_hr), 6),
        "vi": round(float(var_log_hr), 6),
        "se": round(float(se_log_hr), 6),
        "measure": "HR",
        "HR": round(float(hr), 4),
    }


# ======================================================================
# Vaccine Effectiveness — VE% (derived from OR or RR)
# ======================================================================

def compute_ve_from_or(e1: int, n1: int, e2: int, n2: int) -> Dict:
    """
    Compute Vaccine Effectiveness from OR: VE = (1 - OR) * 100.
    Meta-analysis is done on log-OR scale; VE is a display transform.
    """
    result = compute_log_or(e1, n1, e2, n2)
    if result.get("yi") is None:
        result["measure"] = "VE_OR"
        return result

    or_val = np.exp(result["yi"])
    result["VE_pct"] = round(float((1 - or_val) * 100), 1)
    result["measure"] = "VE_OR"
    return result


def compute_ve_from_rr(e1: int, n1: int, e2: int, n2: int) -> Dict:
    """
    Compute Vaccine Effectiveness from RR: VE = (1 - RR) * 100.
    Meta-analysis is done on log-RR scale; VE is a display transform.
    """
    result = compute_log_rr(e1, n1, e2, n2)
    if result.get("yi") is None:
        result["measure"] = "VE_RR"
        return result

    rr_val = np.exp(result["yi"])
    result["VE_pct"] = round(float((1 - rr_val) * 100), 1)
    result["measure"] = "VE_RR"
    return result


def use_precomputed_ve(ve_pct: float, ci_lower_pct: float, ci_upper_pct: float,
                       base_measure: str = "OR") -> Dict:
    """
    Convert reported VE% + 95% CI to log-OR or log-RR for meta-analysis.
    VE = (1 - OR) * 100  =>  OR = 1 - VE/100  =>  log-OR = log(1 - VE/100)
    """
    if ve_pct is None or ci_lower_pct is None or ci_upper_pct is None:
        return {"yi": None, "vi": None, "se": None, "measure": f"VE_{base_measure}", "error": "invalid_input"}

    # Convert VE% to ratio scale
    ratio = 1 - ve_pct / 100.0
    ratio_lower = 1 - ci_upper_pct / 100.0  # note: VE CI is inverted
    ratio_upper = 1 - ci_lower_pct / 100.0

    if ratio <= 0 or ratio_lower <= 0 or ratio_upper <= 0:
        return {"yi": None, "vi": None, "se": None, "measure": f"VE_{base_measure}", "error": "invalid_ve"}

    log_ratio = np.log(ratio)
    se_log = (np.log(ratio_upper) - np.log(ratio_lower)) / (2 * 1.96)
    var_log = se_log ** 2

    return {
        "yi": round(float(log_ratio), 6),
        "vi": round(float(var_log), 6),
        "se": round(float(se_log), 6),
        "measure": f"VE_{base_measure}",
        "VE_pct": round(float(ve_pct), 1),
        f"{base_measure}": round(float(ratio), 4),
    }


# ======================================================================
# Pre-computed effect sizes (generic)
# ======================================================================

def use_precomputed_effect(effect: float, se: float = None,
                           ci_lower: float = None, ci_upper: float = None,
                           measure: str = "SMD",
                           log_scale: bool = False) -> Optional[Dict]:
    """
    Use a pre-computed effect size + SE or CI directly.
    For ratio measures (OR, RR, HR), set log_scale=True if already on log scale,
    or False if on natural scale (will be log-transformed).
    """
    if effect is None:
        return None

    yi = float(effect)
    ratio_measures = {"OR", "RR", "HR", "VE_OR", "VE_RR"}

    if measure in ratio_measures and not log_scale:
        if yi <= 0:
            return None
        yi = np.log(yi)

    if se is not None:
        vi = float(se) ** 2
    elif ci_lower is not None and ci_upper is not None:
        if measure in ratio_measures and not log_scale:
            if ci_lower <= 0 or ci_upper <= 0:
                return None
            se_calc = (np.log(ci_upper) - np.log(ci_lower)) / (2 * 1.96)
        else:
            se_calc = (float(ci_upper) - float(ci_lower)) / (2 * 1.96)
        vi = se_calc ** 2
    else:
        return None

    return {
        "yi": round(float(yi), 6),
        "vi": round(float(vi), 6),
        "se": round(float(np.sqrt(vi)), 6),
        "measure": measure,
    }


# ======================================================================
# Utilities
# ======================================================================

def se_to_sd(se: float, n: int) -> float:
    """Convert standard error to standard deviation."""
    return se * np.sqrt(n)


def ci_to_sd(ci_lower: float, ci_upper: float, n: int) -> float:
    """Convert 95% CI to SD (assuming normal distribution)."""
    se = (ci_upper - ci_lower) / (2 * 1.96)
    return se * np.sqrt(n)


def compute_change_score_effect(
    m_pre1: float, m_post1: float, sd_pre1: float, sd_post1: float, n1: int,
    m_pre2: float, m_post2: float, sd_pre2: float, sd_post2: float, n2: int,
    r: float = 0.5,
) -> Optional[Dict]:
    """
    Compute effect size from pre-post change scores.
    """
    change1 = m_post1 - m_pre1
    change2 = m_post2 - m_pre2

    sd_change1 = np.sqrt(sd_pre1 ** 2 + sd_post1 ** 2 - 2 * r * sd_pre1 * sd_post1)
    sd_change2 = np.sqrt(sd_pre2 ** 2 + sd_post2 ** 2 - 2 * r * sd_pre2 * sd_post2)

    if sd_change1 <= 0 or sd_change2 <= 0:
        return None

    return compute_hedges_g(change1, sd_change1, n1, change2, sd_change2, n2)


# ======================================================================
# Auto-routing: detect outcome type and compute appropriate effect size
# ======================================================================

def detect_outcome_type(outcome: dict) -> str:
    """
    Detect the outcome type from available fields.
    Returns: 'continuous', 'binary', 'time_to_event', or 'precomputed'.
    """
    ig = outcome.get("intervention_group", {})
    cg = outcome.get("control_group", {})

    # Check for binary data (events/total)
    if (ig.get("events") is not None and ig.get("total") is not None and
        cg.get("events") is not None and cg.get("total") is not None):
        return BINARY

    # Check for continuous data (mean/sd/n)
    if (ig.get("mean") is not None and ig.get("n") is not None and
        cg.get("mean") is not None and cg.get("n") is not None):
        return CONTINUOUS

    # Check for pre-computed HR
    if outcome.get("hr") is not None or outcome.get("hazard_ratio") is not None:
        return TIME_TO_EVENT

    # Check for pre-computed VE%
    if outcome.get("ve_pct") is not None or outcome.get("vaccine_effectiveness") is not None:
        return PRECOMPUTED

    # Check for any pre-computed effect size
    if outcome.get("effect_size") is not None:
        return PRECOMPUTED

    return CONTINUOUS  # default fallback


def compute_effect_auto(outcome: dict, preferred_measure: str = None) -> Optional[Dict]:
    """
    Auto-route to the correct effect size computation based on outcome data.

    Args:
        outcome: study outcome dict with flexible fields
        preferred_measure: hint from PICO/config (e.g., "OR", "RR", "SMD", "HR", "VE")

    Returns:
        {"yi": float, "vi": float, "se": float, "measure": str, ...} or None
    """
    outcome_type = outcome.get("outcome_type") or detect_outcome_type(outcome)

    if outcome_type == BINARY:
        return _compute_binary_effect(outcome, preferred_measure)
    elif outcome_type == TIME_TO_EVENT:
        return _compute_tte_effect(outcome)
    elif outcome_type == PRECOMPUTED:
        return _compute_precomputed_effect(outcome, preferred_measure)
    else:
        return _compute_continuous_effect(outcome, preferred_measure)


def _compute_continuous_effect(outcome: dict, preferred_measure: str = None) -> Optional[Dict]:
    """Compute effect from continuous outcome data (mean/sd/n)."""
    ig = outcome.get("intervention_group", {})
    cg = outcome.get("control_group", {})

    m1 = ig.get("mean")
    sd1 = ig.get("sd")
    n1 = ig.get("n")
    m2 = cg.get("mean")
    sd2 = cg.get("sd")
    n2 = cg.get("n")

    # SE-to-SD conversion
    if sd1 is None and ig.get("se") is not None and n1:
        sd1 = se_to_sd(ig["se"], n1)
    if sd2 is None and cg.get("se") is not None and n2:
        sd2 = se_to_sd(cg["se"], n2)

    # CI-to-SD conversion
    if sd1 is None and ig.get("ci_lower") is not None and ig.get("ci_upper") is not None and n1:
        sd1 = ci_to_sd(ig["ci_lower"], ig["ci_upper"], n1)
    if sd2 is None and cg.get("ci_lower") is not None and cg.get("ci_upper") is not None and n2:
        sd2 = ci_to_sd(cg["ci_lower"], cg["ci_upper"], n2)

    if any(v is None for v in [m1, sd1, n1, m2, sd2, n2]):
        return None

    if preferred_measure == "MD":
        return compute_md(m1, sd1, int(n1), m2, sd2, int(n2))
    return compute_hedges_g(m1, sd1, int(n1), m2, sd2, int(n2))


def _compute_binary_effect(outcome: dict, preferred_measure: str = None) -> Optional[Dict]:
    """Compute effect from binary outcome data (events/total per arm)."""
    ig = outcome.get("intervention_group", {})
    cg = outcome.get("control_group", {})

    e1 = ig.get("events")
    n1 = ig.get("total") or ig.get("n")
    e2 = cg.get("events")
    n2 = cg.get("total") or cg.get("n")

    if any(v is None for v in [e1, n1, e2, n2]):
        return None

    e1, n1, e2, n2 = int(e1), int(n1), int(e2), int(n2)

    if preferred_measure in ("RR", "VE_RR"):
        result = compute_log_rr(e1, n1, e2, n2)
    elif preferred_measure == "RD":
        return compute_rd(e1, n1, e2, n2)
    elif preferred_measure == "VE_OR":
        return compute_ve_from_or(e1, n1, e2, n2)
    elif preferred_measure == "VE_RR":
        return compute_ve_from_rr(e1, n1, e2, n2)
    else:
        # Default to OR for binary outcomes
        result = compute_log_or(e1, n1, e2, n2)

    return result


def _compute_tte_effect(outcome: dict) -> Optional[Dict]:
    """Compute effect from time-to-event data (HR from paper)."""
    hr = outcome.get("hr") or outcome.get("hazard_ratio")
    ci_lower = outcome.get("hr_ci_lower") or outcome.get("ci_lower")
    ci_upper = outcome.get("hr_ci_upper") or outcome.get("ci_upper")

    if hr is not None and ci_lower is not None and ci_upper is not None:
        return use_precomputed_hr(float(hr), float(ci_lower), float(ci_upper))

    # Try SE-based
    se = outcome.get("hr_se") or outcome.get("se")
    if hr is not None and se is not None:
        return use_precomputed_effect(float(hr), se=float(se), measure="HR", log_scale=False)

    return None


def _compute_precomputed_effect(outcome: dict, preferred_measure: str = None) -> Optional[Dict]:
    """Use pre-computed effect sizes reported directly in the paper."""
    # VE%
    ve = outcome.get("ve_pct") or outcome.get("vaccine_effectiveness")
    if ve is not None:
        ci_lower = outcome.get("ve_ci_lower") or outcome.get("ci_lower")
        ci_upper = outcome.get("ve_ci_upper") or outcome.get("ci_upper")
        if ci_lower is not None and ci_upper is not None:
            base = "RR" if preferred_measure in ("RR", "VE_RR") else "OR"
            return use_precomputed_ve(float(ve), float(ci_lower), float(ci_upper), base_measure=base)

    # Generic pre-computed
    es = outcome.get("effect_size")
    if es is not None:
        measure = outcome.get("effect_measure") or preferred_measure or "SMD"
        se = outcome.get("se")
        ci_lower = outcome.get("ci_lower")
        ci_upper = outcome.get("ci_upper")
        log_scale = outcome.get("log_scale", False)
        return use_precomputed_effect(
            float(es), se=float(se) if se else None,
            ci_lower=float(ci_lower) if ci_lower else None,
            ci_upper=float(ci_upper) if ci_upper else None,
            measure=measure, log_scale=log_scale,
        )

    return None


# ======================================================================
# Legacy compatibility wrapper
# ======================================================================

def compute_effect_from_study(outcome: dict) -> Optional[Dict]:
    """
    Legacy wrapper — compute effect size from a study outcome dict.
    Now delegates to compute_effect_auto for full measure support.

    Returns dict with 'yi' and 'vi' keys (and legacy 'g'/'var_g' for SMD).
    """
    result = compute_effect_auto(outcome)
    if result is None:
        return None

    # Legacy compatibility: map yi/vi to g/var_g if SMD
    if result.get("measure") == "SMD" and result.get("g") is None:
        result["g"] = result["yi"]
        result["var_g"] = result["vi"]
        result["se_g"] = result["se"]

    return result
