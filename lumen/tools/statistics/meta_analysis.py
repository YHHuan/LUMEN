"""
Random-effects meta-analysis with REML + DL fallback tracking.

Fixes v2 audit #6 (REML fallback not tracked), #25 (HKSJ formula),
#24 (prediction interval).
"""

from __future__ import annotations

import numpy as np
from scipy import stats, optimize


def _dersimonian_laird(effects: np.ndarray, variances: np.ndarray) -> float:
    """DerSimonian-Laird estimator for tau²."""
    w = 1.0 / variances
    w_sum = w.sum()
    q = np.sum(w * (effects - np.average(effects, weights=w)) ** 2)
    k = len(effects)
    c = w_sum - np.sum(w ** 2) / w_sum
    tau2 = max(0.0, (q - (k - 1)) / c)
    return tau2


def _reml_tau2(effects: np.ndarray, variances: np.ndarray) -> float:
    """REML estimator for tau² using L-BFGS-B optimization."""
    k = len(effects)

    def neg_reml_log_likelihood(log_tau2: float) -> float:
        tau2 = np.exp(log_tau2)
        total_var = variances + tau2
        w = 1.0 / total_var
        w_sum = w.sum()
        mu_hat = np.sum(w * effects) / w_sum
        resid = effects - mu_hat
        ll = -0.5 * (
            np.sum(np.log(total_var))
            + np.sum(w * resid ** 2)
            + np.log(w_sum)
        )
        return -ll

    # Initialize with DL estimate
    tau2_init = _dersimonian_laird(effects, variances)
    log_tau2_init = np.log(max(tau2_init, 1e-10))

    result = optimize.minimize(
        neg_reml_log_likelihood,
        x0=log_tau2_init,
        method="L-BFGS-B",
        bounds=[(np.log(1e-12), np.log(1e6))],
    )

    if not result.success:
        raise RuntimeError(f"REML optimization failed: {result.message}")

    return max(0.0, np.exp(result.x[0]))


def random_effects_meta(effects: list[float], ses: list[float],
                        method: str = "REML",
                        apply_hksj: bool = True) -> dict:
    """
    Random-effects meta-analysis.

    Parameters
    ----------
    effects : effect sizes
    ses : standard errors
    method : "REML" or "DL"
    apply_hksj : apply Hartung-Knapp-Sidik-Jonkman adjustment (default True)

    Returns
    -------
    dict with pooled_effect, pooled_se, ci_lower, ci_upper, tau2,
    tau2_method_used, i2, q, q_p, prediction_interval, weights, method_log.
    """
    effects_arr = np.asarray(effects, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    k = len(effects_arr)

    if k == 0:
        raise ValueError("No studies provided")
    if k != len(ses_arr):
        raise ValueError("effects and ses must have same length")
    if np.any(ses_arr <= 0):
        raise ValueError("All standard errors must be > 0")

    variances = ses_arr ** 2
    method_log = []

    # --- Estimate tau² ---
    if k == 1:
        method_log.append("k=1: tau2 set to 0, no pooling possible")
        tau2 = 0.0
        tau2_method = "none_k1"
    elif method == "REML":
        try:
            tau2 = _reml_tau2(effects_arr, variances)
            tau2_method = "REML"
            method_log.append("REML converged successfully")
        except (RuntimeError, ValueError) as e:
            method_log.append(f"REML failed ({e}), falling back to DL")
            tau2 = _dersimonian_laird(effects_arr, variances)
            tau2_method = "DL_fallback"
            method_log.append("DL fallback succeeded")
    else:
        tau2 = _dersimonian_laird(effects_arr, variances)
        tau2_method = "DL"
        method_log.append("DL estimation used (requested)")

    # --- Pool ---
    total_var = variances + tau2
    w = 1.0 / total_var
    w_sum = w.sum()
    pooled = np.sum(w * effects_arr) / w_sum
    pooled_se = np.sqrt(1.0 / w_sum)

    # --- Heterogeneity ---
    w_fe = 1.0 / variances
    q = float(np.sum(w_fe * (effects_arr - np.average(effects_arr, weights=w_fe)) ** 2))
    q_df = max(k - 1, 1)
    q_p = float(1 - stats.chi2.cdf(q, q_df)) if k > 1 else 1.0
    i2 = max(0.0, (q - q_df) / q * 100) if q > 0 else 0.0

    # --- HKSJ adjustment (fixes v2 audit #25) ---
    if apply_hksj and k > 1:
        q_hksj = np.sum(w * (effects_arr - pooled) ** 2) / (k - 1)
        pooled_se_hksj = pooled_se * np.sqrt(q_hksj)
        # Use t-distribution with k-1 df
        t_crit = stats.t.ppf(0.975, k - 1)
        ci_lower = float(pooled - t_crit * pooled_se_hksj)
        ci_upper = float(pooled + t_crit * pooled_se_hksj)
        pooled_se_final = float(pooled_se_hksj)
        method_log.append(f"HKSJ adjustment applied (t-dist, df={k-1})")
    else:
        ci_lower = float(pooled - 1.96 * pooled_se)
        ci_upper = float(pooled + 1.96 * pooled_se)
        pooled_se_final = float(pooled_se)
        if k == 1:
            method_log.append("k=1: no HKSJ, using z-based CI")
        else:
            method_log.append("HKSJ not applied (disabled)")

    # --- Prediction interval (fixes v2 audit #24) ---
    if k >= 3:
        t_crit_pi = stats.t.ppf(0.975, k - 2)
        pi_se = np.sqrt(tau2 + pooled_se ** 2)
        pi_lower = float(pooled - t_crit_pi * pi_se)
        pi_upper = float(pooled + t_crit_pi * pi_se)
    else:
        pi_lower = None
        pi_upper = None
        method_log.append(f"k={k}: prediction interval requires k>=3")

    return {
        "pooled_effect": float(pooled),
        "pooled_se": pooled_se_final,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "tau2": float(tau2),
        "tau2_method_used": tau2_method,
        "i2": float(i2),
        "q": q,
        "q_p": q_p,
        "prediction_interval": (pi_lower, pi_upper) if pi_lower is not None else None,
        "weights": (w / w_sum).tolist(),
        "method_log": method_log,
        "k": k,
    }


def subgroup_meta(effects: list[float], ses: list[float],
                  groups: list[str], **kwargs) -> dict:
    """Subgroup analysis with between-group Q test."""
    effects_arr = np.asarray(effects, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    groups_arr = np.asarray(groups)

    unique_groups = sorted(set(groups))
    subgroup_results = {}
    pooled_effects = []
    pooled_vars = []

    for grp in unique_groups:
        mask = groups_arr == grp
        if mask.sum() == 0:
            continue
        res = random_effects_meta(
            effects_arr[mask].tolist(),
            ses_arr[mask].tolist(),
            **kwargs,
        )
        subgroup_results[grp] = res
        pooled_effects.append(res["pooled_effect"])
        pooled_vars.append(res["pooled_se"] ** 2)

    # Between-group Q test
    pe = np.array(pooled_effects)
    pv = np.array(pooled_vars)
    if len(pe) > 1:
        w_bg = 1.0 / pv
        mu_bg = np.sum(w_bg * pe) / w_bg.sum()
        q_between = float(np.sum(w_bg * (pe - mu_bg) ** 2))
        q_between_p = float(1 - stats.chi2.cdf(q_between, len(pe) - 1))
    else:
        q_between = 0.0
        q_between_p = 1.0

    return {
        "subgroups": subgroup_results,
        "q_between": q_between,
        "q_between_p": q_between_p,
        "n_subgroups": len(unique_groups),
    }


def leave_one_out(effects: list[float], ses: list[float],
                  labels: list[str] | None = None, **kwargs) -> list[dict]:
    """Leave-one-out sensitivity analysis."""
    k = len(effects)
    if k < 2:
        raise ValueError("Leave-one-out requires k >= 2")

    if labels is None:
        labels = [f"study_{i}" for i in range(k)]

    results = []
    for i in range(k):
        eff_i = effects[:i] + effects[i + 1:]
        se_i = ses[:i] + ses[i + 1:]
        res = random_effects_meta(eff_i, se_i, **kwargs)
        results.append({
            "omitted": labels[i],
            "omitted_index": i,
            "pooled_effect": res["pooled_effect"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"],
            "i2": res["i2"],
            "tau2": res["tau2"],
        })

    return results


def cumulative_meta(effects: list[float], ses: list[float],
                    labels: list[str] | None = None,
                    sort_by: list[int] | None = None,
                    **kwargs) -> list[dict]:
    """Cumulative meta-analysis, adding one study at a time."""
    k = len(effects)
    if labels is None:
        labels = [f"study_{i}" for i in range(k)]

    order = sort_by if sort_by is not None else list(range(k))

    results = []
    for j in range(1, len(order) + 1):
        idx = order[:j]
        eff_j = [effects[i] for i in idx]
        se_j = [ses[i] for i in idx]
        res = random_effects_meta(eff_j, se_j, **kwargs)
        results.append({
            "added": labels[order[j - 1]],
            "k": j,
            "pooled_effect": res["pooled_effect"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"],
            "i2": res["i2"],
        })

    return results
