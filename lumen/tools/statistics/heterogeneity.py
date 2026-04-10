"""
Heterogeneity statistics: I², τ², Q, prediction interval.

Standalone utility functions — meta_analysis.py computes these inline,
but this module provides them for independent use.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cochran_q(effects: list[float], ses: list[float]) -> dict:
    """Cochran's Q test for heterogeneity."""
    effects_arr = np.asarray(effects, dtype=float)
    variances = np.asarray(ses, dtype=float) ** 2
    w = 1.0 / variances
    mu = np.sum(w * effects_arr) / w.sum()
    q = float(np.sum(w * (effects_arr - mu) ** 2))
    k = len(effects_arr)
    df = k - 1
    p = float(1 - stats.chi2.cdf(q, df)) if df > 0 else 1.0
    return {"q": q, "df": df, "p_value": p}


def i_squared(q: float, df: int) -> float:
    """I² from Q and degrees of freedom."""
    if q <= 0 or df <= 0:
        return 0.0
    return max(0.0, (q - df) / q * 100)


def prediction_interval(pooled_effect: float, tau2: float,
                        pooled_se: float, k: int) -> dict | None:
    """
    Prediction interval for a new study's true effect.

    PI = pooled ± t(k-2, 0.975) * sqrt(tau² + se_pooled²)
    Requires k >= 3.
    """
    if k < 3:
        return None
    t_crit = stats.t.ppf(0.975, k - 2)
    pi_se = np.sqrt(tau2 + pooled_se ** 2)
    return {
        "lower": float(pooled_effect - t_crit * pi_se),
        "upper": float(pooled_effect + t_crit * pi_se),
        "t_critical": float(t_crit),
        "pi_se": float(pi_se),
    }
