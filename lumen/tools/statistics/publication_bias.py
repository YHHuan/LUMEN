"""
Publication bias detection: Egger's test and trim-and-fill.

Fixes v2 audit #23: trim-and-fill direction flip warning.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def egger_test(effects: list[float], ses: list[float]) -> dict:
    """
    Egger's regression test for funnel plot asymmetry.

    Regresses standardized effect (effect/SE) on precision (1/SE).
    Tests whether intercept differs from zero.
    """
    effects_arr = np.asarray(effects, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    k = len(effects_arr)

    if k < 3:
        raise ValueError("Egger's test requires k >= 3")

    precision = 1.0 / ses_arr
    std_effect = effects_arr / ses_arr

    # Weighted linear regression: std_effect = intercept + slope * precision
    slope, intercept, r_value, p_value, std_err = stats.linregress(precision, std_effect)

    # The intercept tests asymmetry; use t-test
    t_stat = intercept / std_err
    p_intercept = float(2 * stats.t.sf(abs(t_stat), k - 2))

    return {
        "intercept": float(intercept),
        "se": float(std_err),
        "t_stat": float(t_stat),
        "p_value": p_intercept,
        "significant": p_intercept < 0.10,  # conventional threshold
        "k": k,
    }


def trim_and_fill(effects: list[float], ses: list[float],
                  estimator: str = "L0") -> dict:
    """
    Duval & Tweedie trim-and-fill method.

    Estimates the number of missing studies on the side of the funnel
    where studies appear to be suppressed, imputes them, and re-estimates
    the pooled effect.

    Fixes v2 audit #23: warns when filled effect flips direction.
    """
    from lumen.tools.statistics.meta_analysis import random_effects_meta

    effects_arr = np.asarray(effects, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    k = len(effects_arr)

    if k < 3:
        raise ValueError("Trim-and-fill requires k >= 3")

    # Original pooled estimate
    original = random_effects_meta(effects_arr.tolist(), ses_arr.tolist(),
                                   apply_hksj=False)
    pooled_orig = original["pooled_effect"]

    # Estimate number of missing studies using L0 estimator
    # Sort by effect size distance from pooled
    diffs = effects_arr - pooled_orig
    abs_diffs = np.abs(diffs)
    order = np.argsort(abs_diffs)

    # Determine which side has fewer studies (the suppressed side)
    n_right = np.sum(diffs > 0)
    n_left = np.sum(diffs < 0)
    suppress_right = n_right < n_left  # fewer on right → missing on right

    # Rank-based estimation (simplified L0)
    sorted_diffs = diffs[order]
    if suppress_right:
        ranks = np.array([i for i, d in enumerate(sorted_diffs) if d > 0])
    else:
        ranks = np.array([i for i, d in enumerate(sorted_diffs) if d < 0])

    # L0 estimator: k0 = number of studies on one side minus expected
    # Simplified: estimate as max(0, 2*n_extreme - k + 1) where n_extreme
    # counts studies on the "missing" side
    n_missing_side = n_right if suppress_right else n_left
    n_other_side = k - n_missing_side
    k0 = max(0, n_other_side - n_missing_side)

    # Impute mirror studies
    imputed_effects = list(effects_arr)
    imputed_ses = list(ses_arr)

    if k0 > 0:
        # Mirror the most extreme studies from the over-represented side
        if suppress_right:
            # Missing on right → mirror leftmost studies to the right
            extreme_idx = np.argsort(diffs)[:k0]
        else:
            # Missing on left → mirror rightmost studies to the left
            extreme_idx = np.argsort(-diffs)[:k0]

        for idx in extreme_idx:
            mirror_effect = 2 * pooled_orig - effects_arr[idx]
            imputed_effects.append(float(mirror_effect))
            imputed_ses.append(float(ses_arr[idx]))

    # Re-estimate with imputed studies
    adjusted = random_effects_meta(imputed_effects, imputed_ses,
                                   apply_hksj=False)

    # Check direction flip (v2 audit #23)
    direction_flipped = (pooled_orig > 0) != (adjusted["pooled_effect"] > 0)
    warning = None
    if direction_flipped:
        warning = (
            f"Trim-and-fill reversed the direction of the effect "
            f"(original: {pooled_orig:.4f}, adjusted: {adjusted['pooled_effect']:.4f}). "
            f"This suggests substantial publication bias or model instability. "
            f"Interpret with caution."
        )

    return {
        "adjusted_effect": adjusted["pooled_effect"],
        "adjusted_ci": (adjusted["ci_lower"], adjusted["ci_upper"]),
        "n_imputed": k0,
        "original_effect": float(pooled_orig),
        "original_ci": (original["ci_lower"], original["ci_upper"]),
        "direction_flipped": direction_flipped,
        "warning": warning,
        "k_total": k + k0,
    }
