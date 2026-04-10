"""
Unified effect size calculations for LUMEN v3.

Single implementation — replaces v2's duplicated effect_sizes.py + nma.py.
Fixes v2 audit #4 (SE/SD double correction) and #5 (duplicate implementations).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def hedges_g(n1: int, mean1: float, sd1: float,
             n2: int, mean2: float, sd2: float) -> dict:
    """Hedges' g with small-sample correction factor J."""
    df = n1 + n2 - 2
    if df <= 0:
        raise ValueError(f"Total n must be > 2, got n1={n1}, n2={n2}")

    pooled_sd = math.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / df)
    if pooled_sd == 0:
        raise ValueError("Pooled SD is zero — cannot compute effect size")

    d = (mean1 - mean2) / pooled_sd

    # Small-sample correction factor (Hedges & Olkin 1985)
    j = 1 - 3 / (4 * df - 1)
    g = d * j

    se = math.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * df)) * j
    ci_lower = g - 1.96 * se
    ci_upper = g + 1.96 * se

    return {
        "g": g,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "correction_factor": j,
        "cohens_d": d,
    }


def mean_difference(mean1: float, sd1: float, n1: int,
                    mean2: float, sd2: float, n2: int) -> dict:
    """Raw (unstandardized) mean difference."""
    md = mean1 - mean2
    se = math.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
    ci_lower = md - 1.96 * se
    ci_upper = md + 1.96 * se
    return {"md": md, "se": se, "ci_lower": ci_lower, "ci_upper": ci_upper}


def log_risk_ratio(a: int, n1: int, c: int, n2: int) -> dict:
    """Log risk ratio for 2×2 table (a/n1 vs c/n2)."""
    if a <= 0 or c <= 0 or n1 <= 0 or n2 <= 0:
        raise ValueError("All cell counts must be > 0 for log RR")
    rr = (a / n1) / (c / n2)
    log_rr = math.log(rr)
    se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
    ci_lower = log_rr - 1.96 * se
    ci_upper = log_rr + 1.96 * se
    return {"log_rr": log_rr, "se": se, "ci_lower": ci_lower, "ci_upper": ci_upper}


def risk_ratio(a: int, n1: int, c: int, n2: int) -> dict:
    """Risk ratio with CI on natural scale."""
    lr = log_risk_ratio(a, n1, c, n2)
    return {
        "rr": math.exp(lr["log_rr"]),
        "ci_lower": math.exp(lr["ci_lower"]),
        "ci_upper": math.exp(lr["ci_upper"]),
        "log_rr": lr["log_rr"],
        "se_log": lr["se"],
    }


def odds_ratio(a: int, b: int, c: int, d: int) -> dict:
    """Odds ratio from 2×2 table."""
    if any(v <= 0 for v in [a, b, c, d]):
        raise ValueError("All cell counts must be > 0 for OR")
    or_val = (a * d) / (b * c)
    log_or = math.log(or_val)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_lower = math.exp(log_or - 1.96 * se)
    ci_upper = math.exp(log_or + 1.96 * se)
    return {
        "or": or_val,
        "log_or": log_or,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def risk_difference(a: int, n1: int, c: int, n2: int) -> dict:
    """Risk difference (absolute risk reduction)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Group sizes must be > 0")
    p1 = a / n1
    p2 = c / n2
    rd = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_lower = rd - 1.96 * se
    ci_upper = rd + 1.96 * se
    return {"rd": rd, "se": se, "ci_lower": ci_lower, "ci_upper": ci_upper}


def check_and_correct_se_sd(value: float, n: int,
                            reported_as: str,
                            reference_mean: float | None = None) -> dict:
    """
    Detect and correct SE/SD confusion. Fixes v2 audit #4.

    Heuristic: SE ≈ SD / sqrt(n). If reported SE is suspiciously large
    relative to what an SE should be, or reported SD is suspiciously small
    relative to what an SD should be, apply a one-time correction.

    The key insight: we need BOTH conditions to be true:
    1. sqrt(n) is large enough that SE and SD are meaningfully different
    2. The reported value is in the WRONG range for what it claims to be

    When reference_mean is provided, we use CV (coefficient of variation)
    to validate: typical CV for biomedical data is 0.1-1.0.
    Without reference_mean, we require a more conservative threshold.

    Returns was_corrected flag so downstream never double-corrects.
    """
    if n <= 1:
        return {
            "corrected_value": value,
            "was_corrected": False,
            "original_value": value,
            "correction_type": None,
        }

    sqrt_n = math.sqrt(n)
    result = {
        "original_value": value,
        "was_corrected": False,
        "correction_type": None,
    }

    # Only attempt correction when SE and SD differ by > 3x
    if sqrt_n <= 3 or value <= 0:
        result["corrected_value"] = value
        return result

    if reported_as == "se":
        # If reported as SE but value looks like SD:
        # A real SE should be ~ SD/sqrt(n). If this "SE" is actually a SD,
        # then value/sqrt(n) would be the true SE.
        # Check: if reference_mean provided, SE should give a reasonable CV
        # when multiplied back to SD: CV = (SE * sqrt(n)) / mean
        if reference_mean and reference_mean > 0:
            implied_cv = value / reference_mean  # if this is really SE, CV_se = SE/mean
            implied_cv_as_sd = (value / sqrt_n) / reference_mean
            # SD typically gives CV 0.05-1.5. If treating as SD gives more
            # plausible CV, it's likely a SD mislabeled as SE
            if implied_cv >= 1.5 and 0.01 < implied_cv_as_sd < 1.5:
                result["corrected_value"] = value / sqrt_n
                result["was_corrected"] = True
                result["correction_type"] = "se_was_sd"
                return result
        # Without reference: no correction (can't distinguish)
        result["corrected_value"] = value
        return result

    elif reported_as == "sd":
        # If reported as SD but value looks like SE:
        # A real SD should be ~ SE * sqrt(n). If this "SD" is actually SE,
        # then value * sqrt(n) would be the true SD.
        if reference_mean and reference_mean > 0:
            implied_cv = value / reference_mean
            implied_cv_as_se = (value * sqrt_n) / reference_mean
            # If treating as SE gives more plausible CV
            if implied_cv < 0.01 and 0.01 < implied_cv_as_se < 1.5:
                result["corrected_value"] = value * sqrt_n
                result["was_corrected"] = True
                result["correction_type"] = "sd_was_se"
                return result
        result["corrected_value"] = value
        return result

    result["corrected_value"] = value
    return result
