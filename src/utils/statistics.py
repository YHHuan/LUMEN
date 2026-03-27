"""
Meta-Analysis Statistical Engine — LUMEN v2
=============================================
Pure Python implementation (no R dependency).

Features:
1. Fixed-effect (Inverse Variance)
2. Random-effects (DerSimonian-Laird / REML)
3. Hartung-Knapp-Sidik-Jonkman adjustment
4. Heterogeneity (I2, Q, tau2, H2, prediction interval)
5. Sensitivity analyses (leave-one-out, cumulative, influence)
6. Subgroup analysis (Q_between)
7. Meta-regression (mixed-effects, WLS)
8. Publication bias (Egger, Begg, trim-and-fill, failsafe-N)

REML Implementation Notes:
- Uses scipy.optimize.minimize with L-BFGS-B (bounded, gradient-based)
- Falls back to Nelder-Mead if L-BFGS-B fails to converge
- Starting value: DerSimonian-Laird tau2 estimate
- Meta-regression uses iteratively reweighted least squares (IRLS)
"""

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize, minimize_scalar
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
import logging
import warnings
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


# ======================================================================
# Core: tau2 Estimators
# ======================================================================

def _tau2_dl(Q: float, w_fe: np.ndarray, df: int) -> float:
    """DerSimonian-Laird tau2 estimator."""
    c = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
    tau2 = max(0.0, (Q - df) / c)
    return float(tau2)


def _tau2_reml(effects: np.ndarray, variances: np.ndarray,
               optimizer: str = "L-BFGS-B") -> float:
    """
    REML (Restricted Maximum Likelihood) tau2 estimator.

    Maximizes the restricted log-likelihood:
        l_R(tau2) = -0.5 * [sum(log(vi + tau2)) + log(sum(1/(vi+tau2)))
                            + sum(wi*(yi - mu_hat)^2)]

    where wi = 1/(vi + tau2), mu_hat = sum(wi*yi)/sum(wi).
    """
    k = len(effects)
    if k < 2:
        return 0.0

    # DL estimate as starting value
    w_fe = 1.0 / variances
    mu_fe = np.sum(effects * w_fe) / np.sum(w_fe)
    Q = float(np.sum(w_fe * (effects - mu_fe) ** 2))
    tau2_start = max(0.0, _tau2_dl(Q, w_fe, k - 1))

    def neg_reml_ll(log_tau2):
        """Negative REML log-likelihood (parameterized in log-space for stability)."""
        tau2 = np.exp(log_tau2)
        wi = 1.0 / (variances + tau2)
        sum_wi = np.sum(wi)
        mu_hat = np.sum(wi * effects) / sum_wi

        ll = -0.5 * (
            np.sum(np.log(variances + tau2))
            + np.log(sum_wi)
            + np.sum(wi * (effects - mu_hat) ** 2)
        )
        return -ll  # Minimize negative LL

    # Try L-BFGS-B first (fast, gradient-based)
    x0 = np.log(max(tau2_start, 1e-8))

    try:
        result = minimize(neg_reml_ll, x0, method=optimizer,
                          options={"maxiter": 1000, "ftol": 1e-12})
        if result.success:
            return max(0.0, float(np.exp(result.x[0])))
    except Exception:
        pass

    # Fallback: Nelder-Mead (derivative-free, more robust)
    try:
        result = minimize(neg_reml_ll, x0, method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-10})
        if result.success:
            return max(0.0, float(np.exp(result.x[0])))
    except Exception:
        pass

    # Final fallback: DL estimate
    logger.warning("REML optimization failed, falling back to DL estimator")
    return tau2_start


# ======================================================================
# Core: Meta-Analysis
# ======================================================================

def fixed_effect_meta(effects: np.ndarray, variances: np.ndarray) -> dict:
    """Fixed-effect meta-analysis (Inverse Variance method)."""
    weights = 1.0 / variances
    pooled = float(np.sum(effects * weights) / np.sum(weights))
    se = float(np.sqrt(1.0 / np.sum(weights)))
    z = pooled / se
    p = float(2 * (1 - sp_stats.norm.cdf(abs(z))))

    return {
        "model": "fixed",
        "pooled_effect": round(pooled, 4),
        "se": round(se, 4),
        "ci_lower": round(pooled - 1.96 * se, 4),
        "ci_upper": round(pooled + 1.96 * se, 4),
        "z": round(z, 4),
        "p_value": round(p, 6),
        "weights": np.round(weights / np.sum(weights) * 100, 2).tolist(),
    }


def random_effects_meta(effects: np.ndarray, variances: np.ndarray,
                         method: str = "reml",
                         hartung_knapp: bool = True) -> dict:
    """
    Random-effects meta-analysis.

    Args:
        effects: array of effect sizes
        variances: array of sampling variances
        method: "dl" (DerSimonian-Laird) or "reml"
        hartung_knapp: use Hartung-Knapp-Sidik-Jonkman adjustment for CI
    """
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)
    k = len(effects)

    if k < 2:
        if k == 1:
            se = float(np.sqrt(variances[0]))
            return {
                "model": "single_study",
                "pooled_effect": round(float(effects[0]), 4),
                "se": round(se, 4),
                "ci_lower": round(float(effects[0] - 1.96 * se), 4),
                "ci_upper": round(float(effects[0] + 1.96 * se), 4),
                "k": 1,
            }
        return {"model": "insufficient_data", "k": 0}

    # Step 1: Fixed-effect for Q statistic
    w_fe = 1.0 / variances
    mu_fe = np.sum(effects * w_fe) / np.sum(w_fe)
    Q = float(np.sum(w_fe * (effects - mu_fe) ** 2))
    df = k - 1
    p_het = float(1 - sp_stats.chi2.cdf(Q, df))

    # Step 2: Estimate tau2
    if method == "reml":
        tau2 = _tau2_reml(effects, variances)
    else:
        tau2 = _tau2_dl(Q, w_fe, df)

    # Step 3: Random-effects pooling
    w_re = 1.0 / (variances + tau2)
    mu_re = float(np.sum(effects * w_re) / np.sum(w_re))

    # Step 4: CI calculation
    if hartung_knapp and k >= 3:
        se_re, ci_lower, ci_upper, p_val = _hartung_knapp_ci(
            effects, variances, tau2, mu_re, w_re, k
        )
    else:
        se_re = float(np.sqrt(1.0 / np.sum(w_re)))
        ci_lower = mu_re - 1.96 * se_re
        ci_upper = mu_re + 1.96 * se_re
        z = mu_re / se_re if se_re > 0 else 0
        p_val = float(2 * (1 - sp_stats.norm.cdf(abs(z))))

    # Heterogeneity stats
    I2 = _compute_I2(Q, df)
    H2 = Q / df if df > 0 else 1.0

    # Prediction interval
    pi_lower, pi_upper = _prediction_interval(mu_re, tau2, se_re, k)

    return {
        "model": "random",
        "estimator": method.upper(),
        "adjustment": "HKSJ" if (hartung_knapp and k >= 3) else "none",
        "pooled_effect": round(mu_re, 4),
        "se": round(se_re, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_val, 6),
        "tau2": round(tau2, 4),
        "tau": round(np.sqrt(tau2), 4),
        "I2": round(I2, 2),
        "H2": round(H2, 2),
        "Q": round(Q, 4),
        "Q_df": df,
        "Q_p": round(p_het, 6),
        "prediction_interval": [round(pi_lower, 4), round(pi_upper, 4)],
        "k": k,
        "weights": np.round(w_re / np.sum(w_re) * 100, 2).tolist(),
    }


def _hartung_knapp_ci(effects, variances, tau2, mu, w_re, k):
    """Hartung-Knapp-Sidik-Jonkman adjusted CI using t-distribution."""
    q_hk = float(np.sum(w_re * (effects - mu) ** 2))
    df = k - 1

    # HKSJ variance estimator
    var_hk = q_hk / (df * np.sum(w_re))
    se_hk = float(np.sqrt(max(var_hk, 0)))

    # Use t-distribution instead of normal
    t_crit = float(sp_stats.t.ppf(0.975, df))
    ci_lower = mu - t_crit * se_hk
    ci_upper = mu + t_crit * se_hk

    # p-value from t-distribution
    t_val = mu / se_hk if se_hk > 0 else 0
    p_val = float(2 * sp_stats.t.sf(abs(t_val), df))

    return se_hk, ci_lower, ci_upper, p_val


def _compute_I2(Q, df):
    """Higgins I-squared statistic."""
    if Q <= df or df == 0:
        return 0.0
    return float((Q - df) / Q * 100)


def _prediction_interval(mu, tau2, se, k):
    """Prediction interval for the true effect in a new study."""
    if k < 3:
        return float('-inf'), float('inf')
    t_crit = float(sp_stats.t.ppf(0.975, k - 2))
    pi_se = np.sqrt(tau2 + se ** 2)
    return float(mu - t_crit * pi_se), float(mu + t_crit * pi_se)


# ======================================================================
# Sensitivity Analyses
# ======================================================================

def leave_one_out(effects: np.ndarray, variances: np.ndarray,
                  study_labels: list, method: str = "reml",
                  hartung_knapp: bool = True) -> list:
    """Leave-one-out sensitivity analysis."""
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)
    results = []

    for i in range(len(effects)):
        mask = np.array([j for j in range(len(effects)) if j != i])
        sub = random_effects_meta(
            effects[mask], variances[mask],
            method=method, hartung_knapp=hartung_knapp,
        )
        sub["excluded_study"] = study_labels[i]
        sub["excluded_index"] = i
        results.append(sub)

    return results


def cumulative_meta_analysis(effects: np.ndarray, variances: np.ndarray,
                              study_labels: list, years: list = None,
                              method: str = "reml") -> list:
    """Cumulative meta-analysis (adding studies chronologically)."""
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)

    # Sort by year if provided
    if years:
        order = np.argsort(years)
    else:
        order = np.arange(len(effects))

    results = []
    for k in range(2, len(order) + 1):
        indices = order[:k]
        sub = random_effects_meta(
            effects[indices], variances[indices],
            method=method, hartung_knapp=False,
        )
        sub["studies_included"] = k
        sub["latest_study"] = study_labels[order[k - 1]]
        results.append(sub)

    return results


def influence_diagnostics(effects: np.ndarray, variances: np.ndarray,
                           study_labels: list,
                           method: str = "reml") -> list:
    """Influence diagnostics (hat values, Cook's distance, DFBETAS)."""
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)
    k = len(effects)

    full = random_effects_meta(effects, variances, method=method, hartung_knapp=False)
    mu_full = full["pooled_effect"]
    tau2 = full["tau2"]

    w = 1.0 / (variances + tau2)
    sum_w = np.sum(w)

    # Hat values (leverage)
    hat_vals = w / sum_w

    # Leave-one-out for Cook's distance
    loo = leave_one_out(effects, variances, study_labels, method=method,
                        hartung_knapp=False)

    diagnostics = []
    for i, loo_result in enumerate(loo):
        mu_loo = loo_result["pooled_effect"]
        se_full = full["se"]

        # Cook's distance analog
        cooks_d = (mu_full - mu_loo) ** 2 / (se_full ** 2) if se_full > 0 else 0

        # DFBETAS
        dfbetas = (mu_full - mu_loo) / se_full if se_full > 0 else 0

        diagnostics.append({
            "study": study_labels[i],
            "hat_value": round(float(hat_vals[i]), 4),
            "cooks_distance": round(float(cooks_d), 4),
            "dfbetas": round(float(dfbetas), 4),
            "pooled_without": round(float(mu_loo), 4),
        })

    return diagnostics


# ======================================================================
# Subgroup Analysis
# ======================================================================

def subgroup_analysis(effects: np.ndarray, variances: np.ndarray,
                       study_labels: list, subgroups: list,
                       method: str = "reml",
                       hartung_knapp: bool = True) -> dict:
    """
    Between-subgroup heterogeneity test (Q_between).

    Args:
        subgroups: list of group labels (same length as effects)
    """
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)

    groups = defaultdict(list)
    for i, grp in enumerate(subgroups):
        groups[grp].append(i)

    subgroup_results = {}
    for grp_name, indices in groups.items():
        if len(indices) < 2:
            subgroup_results[grp_name] = {
                "k": len(indices),
                "note": "insufficient studies for meta-analysis",
            }
            continue

        idx = np.array(indices)
        sub = random_effects_meta(
            effects[idx], variances[idx],
            method=method, hartung_knapp=hartung_knapp,
        )
        sub["studies"] = [study_labels[i] for i in indices]
        subgroup_results[grp_name] = sub

    # Q_between
    Q_between, df_between, p_between = _compute_Q_between(
        subgroup_results, effects, variances, groups
    )

    return {
        "subgroups": subgroup_results,
        "Q_between": round(Q_between, 4),
        "df_between": df_between,
        "p_between": round(p_between, 6),
    }


def _compute_Q_between(subgroup_results, effects, variances, groups):
    """Compute between-subgroup Q statistic."""
    # Overall pooled effect (fixed-effect within subgroups)
    overall_effects = []
    overall_weights = []

    for grp_name, result in subgroup_results.items():
        if "pooled_effect" not in result:
            continue
        mu_g = result["pooled_effect"]
        se_g = result.get("se", 0)
        if se_g > 0:
            w_g = 1.0 / (se_g ** 2)
            overall_effects.append(mu_g)
            overall_weights.append(w_g)

    if len(overall_effects) < 2:
        return 0.0, 0, 1.0

    eff = np.array(overall_effects)
    wts = np.array(overall_weights)
    mu_overall = np.sum(wts * eff) / np.sum(wts)
    Q_between = float(np.sum(wts * (eff - mu_overall) ** 2))
    df = len(overall_effects) - 1
    p = float(1 - sp_stats.chi2.cdf(Q_between, df))

    return Q_between, df, p


# ======================================================================
# Meta-Regression (Mixed-Effects)
# ======================================================================

def meta_regression(effects: np.ndarray, variances: np.ndarray,
                     moderators: np.ndarray,
                     moderator_names: list = None,
                     method: str = "reml") -> dict:
    """
    Mixed-effects meta-regression.

    Model: y_i = X_i @ beta + u_i + e_i
    where u_i ~ N(0, tau2), e_i ~ N(0, v_i)

    Estimation:
    1. Estimate tau2 via REML (profile likelihood)
    2. Compute beta = (X'WX)^{-1} X'Wy
    3. Var(beta) = (X'WX)^{-1}
    4. Test coefficients with t-distribution (Knapp-Hartung)

    Args:
        effects: (k,) array of effect sizes
        variances: (k,) array of sampling variances
        moderators: (k, p) array of moderator values (design matrix without intercept)
        moderator_names: list of moderator variable names
    """
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)
    X_raw = np.asarray(moderators, dtype=float)

    k = len(effects)
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(-1, 1)

    # Add intercept
    X = np.column_stack([np.ones(k), X_raw])
    p = X.shape[1]

    if moderator_names is None:
        moderator_names = [f"x{i}" for i in range(X_raw.shape[1])]
    coef_names = ["intercept"] + list(moderator_names)

    if k <= p:
        return {"error": f"Need k > p (k={k}, p={p}) for meta-regression"}

    # Step 1: Estimate tau2 via REML profile likelihood
    tau2 = _meta_regression_reml_tau2(effects, variances, X, method)

    # Step 2: WLS estimation
    W = np.diag(1.0 / (variances + tau2))
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ effects

    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse for near-singular matrices
        XtWX_inv = np.linalg.pinv(XtWX)
        logger.warning("XtWX near-singular, using pseudo-inverse")

    beta = XtWX_inv @ XtWy

    # Step 3: Variance and inference
    # Knapp-Hartung adjustment
    residuals = effects - X @ beta
    w = 1.0 / (variances + tau2)
    RSS = float(np.sum(w * residuals ** 2))

    # Adjusted variance (Knapp-Hartung)
    df_resid = k - p
    s2_kh = RSS / df_resid if df_resid > 0 else 1.0
    var_beta_kh = s2_kh * XtWX_inv

    se_beta = np.sqrt(np.diag(var_beta_kh))

    # t-tests for each coefficient
    coefficients = []
    for j in range(p):
        t_val = beta[j] / se_beta[j] if se_beta[j] > 0 else 0.0
        p_val = float(2 * sp_stats.t.sf(abs(t_val), df_resid)) if df_resid > 0 else 1.0
        t_crit = float(sp_stats.t.ppf(0.975, df_resid)) if df_resid > 0 else 1.96

        coefficients.append({
            "name": coef_names[j],
            "estimate": round(float(beta[j]), 4),
            "se": round(float(se_beta[j]), 4),
            "t_value": round(float(t_val), 4),
            "p_value": round(float(p_val), 6),
            "ci_lower": round(float(beta[j] - t_crit * se_beta[j]), 4),
            "ci_upper": round(float(beta[j] + t_crit * se_beta[j]), 4),
        })

    # Model fit: QM (omnibus test for moderators)
    # Test H0: beta_1 = ... = beta_{p-1} = 0
    if p > 1:
        beta_mod = beta[1:]  # Exclude intercept
        var_mod = var_beta_kh[1:, 1:]
        try:
            QM = float(beta_mod @ np.linalg.inv(var_mod) @ beta_mod)
        except np.linalg.LinAlgError:
            QM = 0.0
        df_mod = p - 1
        p_QM = float(1 - sp_stats.f.cdf(QM / df_mod, df_mod, df_resid)) if df_resid > 0 else 1.0
    else:
        QM, df_mod, p_QM = 0.0, 0, 1.0

    # R-squared analog (proportion of heterogeneity explained)
    tau2_base = _tau2_reml(effects, variances)
    R2 = max(0.0, 1 - tau2 / tau2_base) if tau2_base > 0 else 0.0

    # QE: residual heterogeneity
    QE = RSS
    p_QE = float(1 - sp_stats.chi2.cdf(QE, df_resid)) if df_resid > 0 else 1.0

    return {
        "model": "meta_regression",
        "method": method.upper(),
        "adjustment": "Knapp-Hartung",
        "k": k,
        "p": p,
        "tau2": round(float(tau2), 4),
        "tau2_baseline": round(float(tau2_base), 4),
        "R2_analog": round(float(R2), 4),
        "coefficients": coefficients,
        "QM": round(float(QM), 4),
        "QM_df": df_mod,
        "QM_p": round(float(p_QM), 6),
        "QE": round(float(QE), 4),
        "QE_df": df_resid,
        "QE_p": round(float(p_QE), 6),
    }


def _meta_regression_reml_tau2(effects, variances, X, method="reml"):
    """Estimate tau2 for meta-regression model via REML."""
    k = len(effects)
    p = X.shape[1]

    if method != "reml":
        # Method of moments (DL analog for regression)
        W0 = np.diag(1.0 / variances)
        P0 = W0 - W0 @ X @ np.linalg.solve(X.T @ W0 @ X, X.T @ W0)
        RSS0 = float(effects @ P0 @ effects)
        return max(0.0, (RSS0 - (k - p)) / np.trace(P0))

    def neg_reml_ll(log_tau2):
        tau2 = np.exp(log_tau2)
        W = np.diag(1.0 / (variances + tau2))
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ effects

        try:
            L = np.linalg.cholesky(XtWX)
            log_det_XtWX = 2 * np.sum(np.log(np.diag(L)))
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            return 1e10

        residuals = effects - X @ beta
        ll = -0.5 * (
            np.sum(np.log(variances + tau2))
            + log_det_XtWX
            + float(residuals @ W @ residuals)
        )
        return -ll

    # Starting value from DL
    tau2_start = _meta_regression_reml_tau2(effects, variances, X, method="dl")
    x0 = np.log(max(tau2_start, 1e-8))

    try:
        result = minimize(neg_reml_ll, x0, method="L-BFGS-B",
                          options={"maxiter": 1000, "ftol": 1e-12})
        if result.success:
            return max(0.0, float(np.exp(result.x[0])))
    except Exception:
        pass

    try:
        result = minimize(neg_reml_ll, x0, method="Nelder-Mead",
                          options={"maxiter": 2000})
        if result.success:
            return max(0.0, float(np.exp(result.x[0])))
    except Exception:
        pass

    return tau2_start


# ======================================================================
# Publication Bias
# ======================================================================

def egger_test(effects: np.ndarray, variances: np.ndarray) -> dict:
    """Egger's regression test for funnel plot asymmetry."""
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)

    se = np.sqrt(variances)
    precision = 1.0 / se
    z = effects / se

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(precision, z)

    # t-test for intercept
    k = len(effects)
    t_val = intercept / std_err if std_err > 0 else 0
    p_two_sided = float(2 * sp_stats.t.sf(abs(t_val), k - 2)) if k > 2 else 1.0

    return {
        "test": "Egger",
        "intercept": round(float(intercept), 4),
        "se_intercept": round(float(std_err), 4),
        "t_value": round(float(t_val), 4),
        "p_value": round(float(p_two_sided), 6),
        "interpretation": "asymmetry_detected" if p_two_sided < 0.10 else "no_asymmetry",
    }


def begg_test(effects: np.ndarray, variances: np.ndarray) -> dict:
    """Begg's rank correlation test."""
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)

    se = np.sqrt(variances)
    k = len(effects)

    # Standardized effects
    w = 1.0 / variances
    mu = np.sum(w * effects) / np.sum(w)
    std_effects = (effects - mu) / se

    tau, p_value = sp_stats.kendalltau(std_effects, variances)

    return {
        "test": "Begg",
        "kendall_tau": round(float(tau), 4),
        "p_value": round(float(p_value), 6),
        "interpretation": "asymmetry_detected" if p_value < 0.10 else "no_asymmetry",
    }


def trim_and_fill(effects: np.ndarray, variances: np.ndarray,
                   side: str = "left", estimator: str = "L0") -> dict:
    """
    Duval & Tweedie's trim-and-fill method.

    Estimates the number of missing studies and imputes them to
    make the funnel plot symmetric.
    """
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)
    k = len(effects)

    # Step 1: Estimate pooled effect
    w = 1.0 / variances
    mu0 = np.sum(w * effects) / np.sum(w)

    # Step 2: Rank |yi - mu| by effect size
    centered = effects - mu0

    if side == "left":
        # Look for missing studies on the left (negative) side
        # Find right-side outliers and mirror them
        right_idx = np.where(centered > 0)[0]
        if len(right_idx) == 0:
            return {"k0": 0, "imputed_effects": [], "adjusted": None}

        # L0 estimator
        n_ranks = np.abs(centered)
        rank_order = np.argsort(n_ranks)
        ranks = np.zeros(k)
        ranks[rank_order] = np.arange(1, k + 1)

        # Positive side ranks
        T_plus = np.sum(ranks[centered > 0])
        # Estimated missing studies (L0)
        k0_raw = (4 * T_plus - k * (k + 1) / 2) / (2 * k - 1)
        k0 = max(0, int(round(k0_raw)))
    else:
        # Mirror left side
        left_idx = np.where(centered < 0)[0]
        if len(left_idx) == 0:
            return {"k0": 0, "imputed_effects": [], "adjusted": None}

        n_ranks = np.abs(centered)
        rank_order = np.argsort(n_ranks)
        ranks = np.zeros(k)
        ranks[rank_order] = np.arange(1, k + 1)

        T_minus = np.sum(ranks[centered < 0])
        k0_raw = (4 * T_minus - k * (k + 1) / 2) / (2 * k - 1)
        k0 = max(0, int(round(k0_raw)))

    if k0 == 0:
        orig_result = random_effects_meta(effects, variances, hartung_knapp=False)
        return {
            "k0": 0,
            "side": side,
            "imputed_effects": [],
            "imputed_variances": [],
            "adjusted": orig_result,
        }

    # Step 3: Impute missing studies by mirroring
    sorted_centered = np.sort(centered)
    if side == "left":
        # Mirror the k0 most extreme right-side studies
        extreme = np.sort(centered[centered > 0])[-k0:]
        imputed_effects = mu0 - extreme  # Mirror around mu0
        imputed_variances = np.sort(variances[centered > 0])[-k0:]
    else:
        extreme = np.sort(centered[centered < 0])[:k0]
        imputed_effects = mu0 - extreme
        imputed_variances = np.sort(variances[centered < 0])[:k0]

    # Step 4: Re-run meta-analysis with imputed studies
    all_effects = np.concatenate([effects, imputed_effects])
    all_variances = np.concatenate([variances, imputed_variances])
    adjusted = random_effects_meta(all_effects, all_variances, hartung_knapp=False)

    return {
        "k0": k0,
        "side": side,
        "imputed_effects": np.round(imputed_effects, 4).tolist(),
        "imputed_variances": np.round(imputed_variances, 4).tolist(),
        "adjusted": adjusted,
    }


def failsafe_n(effects: np.ndarray, variances: np.ndarray,
                alpha: float = 0.05) -> dict:
    """Rosenthal's fail-safe N."""
    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)

    z = effects / np.sqrt(variances)
    z_sum = float(np.sum(z))
    k = len(effects)
    z_alpha = float(sp_stats.norm.ppf(1 - alpha))

    n_fs = max(0, int(np.ceil((z_sum / z_alpha) ** 2 - k)))

    # Orwin's fail-safe N (effect size approach)
    mean_d = float(np.mean(effects))
    trivial_d = 0.10  # Trivial effect size threshold
    if abs(mean_d) > trivial_d:
        orwin_n = max(0, int(np.ceil(k * (mean_d / trivial_d - 1))))
    else:
        orwin_n = 0

    return {
        "rosenthal_n": n_fs,
        "orwin_n": orwin_n,
        "k": k,
        "interpretation": (
            f"Need {n_fs} null studies to overturn significance (Rosenthal). "
            f"Need {orwin_n} studies with trivial effect to reduce below d={trivial_d} (Orwin)."
        ),
    }


# ======================================================================
# Effect Sizes
# ======================================================================

def cohens_d(m1, sd1, n1, m2, sd2, n2) -> dict:
    """Compute Cohen's d (standardized mean difference) and its variance."""
    sp = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp if sp > 0 else 0.0
    var_d = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
    return {"d": float(d), "var_d": float(var_d), "se_d": float(np.sqrt(var_d))}


def hedges_g(m1, sd1, n1, m2, sd2, n2) -> dict:
    """Compute Hedges' g (bias-corrected SMD) and its variance."""
    result = cohens_d(m1, sd1, n1, m2, sd2, n2)
    d = result["d"]
    df = n1 + n2 - 2
    # Small-sample correction factor J
    J = 1 - 3 / (4 * df - 1) if df > 1 else 1.0
    g = d * J
    var_g = result["var_d"] * J ** 2
    return {"g": float(g), "var_g": float(var_g), "se_g": float(np.sqrt(var_g)), "J": float(J)}


def se_to_sd(se: float, n: int) -> float:
    """Convert standard error to standard deviation."""
    return se * np.sqrt(n)


def ci_to_sd(ci_lower: float, ci_upper: float, n: int) -> float:
    """Convert 95% CI to SD (assuming normal distribution)."""
    se = (ci_upper - ci_lower) / (2 * 1.96)
    return se * np.sqrt(n)


# ======================================================================
# MetaAnalysisEngine (Convenience Class)
# ======================================================================

class MetaAnalysisEngine:
    """High-level interface for running complete meta-analysis."""

    def __init__(self, estimator: str = "REML", hartung_knapp: bool = True,
                 optimizer: str = "L-BFGS-B"):
        self.estimator = estimator.lower()
        self.hartung_knapp = hartung_knapp
        self.optimizer = optimizer

    def run(self, effects, variances, study_labels=None, **kwargs):
        """Run random-effects meta-analysis."""
        effects = np.asarray(effects, dtype=float)
        variances = np.asarray(variances, dtype=float)
        if study_labels is None:
            study_labels = [f"Study_{i+1}" for i in range(len(effects))]

        return random_effects_meta(
            effects, variances,
            method=self.estimator,
            hartung_knapp=self.hartung_knapp,
        )

    def run_full_analysis(self, effects, variances, study_labels,
                           subgroup_var=None, subgroups=None,
                           moderators=None, moderator_names=None,
                           years=None) -> dict:
        """Run complete analysis suite: main + sensitivity + bias + subgroup."""
        effects = np.asarray(effects, dtype=float)
        variances = np.asarray(variances, dtype=float)

        results = {}

        # Main analysis
        results["main"] = self.run(effects, variances, study_labels)
        results["fixed"] = fixed_effect_meta(effects, variances)

        # Sensitivity
        results["leave_one_out"] = leave_one_out(
            effects, variances, study_labels,
            method=self.estimator, hartung_knapp=self.hartung_knapp,
        )

        results["cumulative"] = cumulative_meta_analysis(
            effects, variances, study_labels, years=years,
            method=self.estimator,
        )

        results["influence"] = influence_diagnostics(
            effects, variances, study_labels, method=self.estimator,
        )

        # Publication bias
        if len(effects) >= 3:
            results["egger"] = egger_test(effects, variances)
            results["begg"] = begg_test(effects, variances)
            results["trim_and_fill"] = trim_and_fill(effects, variances)
            results["failsafe_n"] = failsafe_n(effects, variances)

        # Subgroup analysis
        if subgroups is not None:
            results["subgroup"] = subgroup_analysis(
                effects, variances, study_labels, subgroups,
                method=self.estimator, hartung_knapp=self.hartung_knapp,
            )

        # Meta-regression
        if moderators is not None:
            results["meta_regression"] = meta_regression(
                effects, variances, moderators,
                moderator_names=moderator_names,
                method=self.estimator,
            )

        return results


# ======================================================================
# R metafor Bridge
# ======================================================================

R_BRIDGE_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "r_bridge" / "meta_analysis.R"


def is_r_available() -> bool:
    """Check if Rscript is available and metafor is installed."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", "library(metafor); cat('OK')"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0 and "OK" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_r_metafor(effects, variances, labels, *,
                  method="REML", knha=True, measure_label="SMD",
                  years=None, subgroups=None, moderators=None,
                  figures_dir=None, timeout=120) -> dict:
    """
    Run meta-analysis via R metafor bridge.

    Returns dict matching MetaAnalysisEngine output schema, or raises RuntimeError.
    """
    effects = np.asarray(effects, dtype=float).tolist()
    variances = np.asarray(variances, dtype=float).tolist()

    input_data = {
        "effects": effects,
        "variances": variances,
        "labels": list(labels),
        "method": method,
        "knha": knha,
        "measure_label": measure_label,
    }
    if years is not None:
        input_data["years"] = [int(y) for y in years]
    if subgroups is not None:
        input_data["subgroups"] = list(subgroups)
    if moderators is not None:
        mod = np.asarray(moderators, dtype=float)
        input_data["moderators"] = mod.flatten().tolist() if mod.ndim <= 1 else mod[:, 0].tolist()

    with tempfile.TemporaryDirectory(prefix="lumen_r_") as tmpdir:
        in_path = Path(tmpdir) / "input.json"
        out_path = Path(tmpdir) / "output.json"

        with open(in_path, "w") as f:
            json.dump(input_data, f)

        cmd = ["Rscript", str(R_BRIDGE_SCRIPT), "--input", str(in_path), "--output", str(out_path)]
        if figures_dir:
            cmd.extend(["--figures-dir", str(figures_dir)])
        logger.info(f"Running R metafor: {' '.join(cmd)}")

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if proc.returncode != 0:
            raise RuntimeError(f"R metafor failed (rc={proc.returncode}): {proc.stderr}")

        if not out_path.exists():
            raise RuntimeError(f"R metafor produced no output. stderr: {proc.stderr}")

        with open(out_path) as f:
            r_results = json.load(f)

    return _normalize_r_results(r_results)


def _normalize_r_results(r: dict) -> dict:
    """Convert R metafor JSON output to match Python MetaAnalysisEngine schema."""
    results = {"engine": "r_metafor", "r_raw": r}

    # Main analysis
    if "main" in r and "error" not in r["main"]:
        m = r["main"]
        results["main"] = {
            "pooled_effect": m["pooled_effect"],
            "se": m["se"],
            "ci_lower": m["ci_lower"],
            "ci_upper": m["ci_upper"],
            "z_value": m.get("z_value"),
            "p_value": m["p_value"],
            "tau2": m["tau2"],
            "I2": m["I2"],
            "H2": m.get("H2"),
            "Q": m["Q"],
            "Q_df": m.get("Q_df"),
            "Q_p": m["Q_p"],
            "k": m["k"],
            "estimator": m.get("method", "REML"),
            "adjustment": "HKSJ" if m.get("knha") else "none",
            "pi_lower": m.get("pi_lower"),
            "pi_upper": m.get("pi_upper"),
        }

    # Leave-one-out
    if "leave_one_out" in r and not isinstance(r["leave_one_out"], dict):
        results["leave_one_out"] = r["leave_one_out"]
    elif "leave_one_out" in r and isinstance(r["leave_one_out"], list):
        results["leave_one_out"] = r["leave_one_out"]

    # Cumulative
    if "cumulative" in r and isinstance(r["cumulative"], list):
        results["cumulative"] = r["cumulative"]

    # Egger's test
    if "egger_test" in r and "error" not in r["egger_test"]:
        e = r["egger_test"]
        results["egger"] = {
            "intercept": e["intercept"],
            "se": e["se"],
            "z_value": e.get("z_value"),
            "p_value": e["p_value"],
            "interpretation": "significant asymmetry" if e.get("significant") else "no significant asymmetry",
        }

    # Begg's test
    if "begg_test" in r and "error" not in r["begg_test"]:
        b = r["begg_test"]
        results["begg"] = {
            "tau": b["tau"],
            "p_value": b["p_value"],
            "interpretation": "significant" if b.get("significant") else "not significant",
        }

    # Trim-and-fill
    if "trim_and_fill" in r and "error" not in r["trim_and_fill"]:
        tf = r["trim_and_fill"]
        results["trim_and_fill"] = {
            "k_original": tf["k_original"],
            "k_filled": tf["k_filled"],
            "k_total": tf["k_total"],
            "adjusted_estimate": tf["adjusted_estimate"],
            "adjusted_ci_lower": tf["adjusted_ci_lower"],
            "adjusted_ci_upper": tf["adjusted_ci_upper"],
            "side": tf.get("side"),
        }

    # Failsafe-N
    if "failsafe_n" in r and "error" not in r["failsafe_n"]:
        results["failsafe_n"] = r["failsafe_n"]

    # Subgroup analysis
    if "subgroup_analysis" in r and "error" not in r["subgroup_analysis"]:
        results["subgroup"] = r["subgroup_analysis"]

    # Meta-regression
    if "meta_regression" in r and "error" not in r["meta_regression"]:
        mr = r["meta_regression"]
        results["meta_regression"] = {
            "intercept": mr["intercept"],
            "slope": mr["slope"],
            "slope_se": mr["slope_se"],
            "slope_p": mr["slope_p"],
            "QM": mr["QM"],
            "QM_p": mr["QM_p"],
            "QE": mr["QE"],
            "QE_p": mr["QE_p"],
            "R2_analog": mr.get("R2", 0),
            "tau2_residual": mr["tau2_residual"],
        }

    # Influence diagnostics
    if "influence" in r and isinstance(r["influence"], list):
        results["influence"] = r["influence"]

    # Study weights
    if "study_weights" in r:
        results["study_weights"] = r["study_weights"]

    # Audit
    if "audit" in r:
        results["audit"] = r["audit"]

    return results


def run_dual_engine(effects, variances, labels, *,
                    method="REML", knha=True, measure_label="SMD",
                    years=None, subgroups=None, moderators=None,
                    moderator_names=None, python_engine=None) -> dict:
    """
    Run both Python and R engines, compare results, return R as primary.

    Flags discrepancies > 1% in key metrics.
    """
    # Python results
    if python_engine is None:
        python_engine = MetaAnalysisEngine(estimator=method, hartung_knapp=knha)

    py_results = python_engine.run_full_analysis(
        effects, variances, labels,
        subgroups=subgroups, moderators=moderators,
        moderator_names=moderator_names,
        years=years,
    )

    # R results
    r_results = run_r_metafor(
        effects, variances, labels,
        method=method, knha=knha, measure_label=measure_label,
        years=years, subgroups=subgroups, moderators=moderators,
    )

    # Compare key metrics
    discrepancies = []
    if "main" in py_results and "main" in r_results:
        py_m = py_results["main"]
        r_m = r_results["main"]
        for key in ["pooled_effect", "ci_lower", "ci_upper", "tau2", "I2"]:
            py_val = py_m.get(key)
            r_val = r_m.get(key)
            if py_val is not None and r_val is not None:
                if abs(py_val) > 1e-8 or abs(r_val) > 1e-8:
                    denom = max(abs(py_val), abs(r_val), 1e-8)
                    pct_diff = abs(py_val - r_val) / denom * 100
                    if pct_diff > 1.0:
                        discrepancies.append({
                            "metric": key,
                            "python": round(py_val, 6),
                            "r": round(r_val, 6),
                            "pct_diff": round(pct_diff, 2),
                        })

    # R is primary, attach comparison
    r_results["python_comparison"] = {
        "python_main": py_results.get("main"),
        "discrepancies": discrepancies,
        "concordant": len(discrepancies) == 0,
    }

    if discrepancies:
        logger.warning(f"R vs Python discrepancies: {discrepancies}")
    else:
        logger.info("R and Python engines concordant (all metrics within 1%)")

    return r_results
