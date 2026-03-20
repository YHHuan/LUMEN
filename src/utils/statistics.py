"""
Meta-Analysis Statistical Engine
==================================
完整的統計引擎，移植自 R 的 meta/metafor 邏輯。

功能:
1. Fixed-effect (Inverse Variance)
2. Random-effects (DerSimonian-Laird / REML)
3. Heterogeneity (I², Q, tau², H², prediction interval)
4. Subgroup analysis
5. Sensitivity analysis (leave-one-out, influential studies)
6. Meta-regression (mixed-effects)
7. Publication bias (Egger's test, Begg's test, trim-and-fill)

所有結果格式統一為 dict，方便 JSON 輸出。
"""

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# Core Meta-Analysis
# ======================================================================

def fixed_effect_meta(effects: np.ndarray, variances: np.ndarray) -> dict:
    """
    Fixed-effect meta-analysis (Inverse Variance method)。
    
    假設所有研究估計的是同一個 true effect。
    """
    weights = 1.0 / variances
    pooled = np.sum(effects * weights) / np.sum(weights)
    se = np.sqrt(1.0 / np.sum(weights))
    ci_lower = pooled - 1.96 * se
    ci_upper = pooled + 1.96 * se
    z = pooled / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    
    return {
        "model": "fixed",
        "pooled_effect": round(float(pooled), 4),
        "se": round(float(se), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "z": round(float(z), 4),
        "p_value": round(float(p), 6),
        "weights": np.round(weights / np.sum(weights) * 100, 2).tolist(),
    }


def random_effects_meta(effects: np.ndarray, variances: np.ndarray,
                        method: str = "dl") -> dict:
    """
    Random-effects meta-analysis。
    
    Methods:
    - "dl": DerSimonian-Laird (最常用，Cochrane 預設)
    - "reml": Restricted Maximum Likelihood (較精確)
    """
    k = len(effects)
    
    if k < 2:
        logger.warning("Need ≥2 studies for meta-analysis")
        if k == 1:
            return {
                "model": "single_study",
                "pooled_effect": round(float(effects[0]), 4),
                "se": round(float(np.sqrt(variances[0])), 4),
                "ci_lower": round(float(effects[0] - 1.96 * np.sqrt(variances[0])), 4),
                "ci_upper": round(float(effects[0] + 1.96 * np.sqrt(variances[0])), 4),
                "k": 1,
            }
        return {"model": "insufficient_data", "k": 0}
    
    # --- Step 1: Fixed-effect for Q statistic ---
    w_fe = 1.0 / variances
    mu_fe = np.sum(effects * w_fe) / np.sum(w_fe)
    Q = float(np.sum(w_fe * (effects - mu_fe) ** 2))
    df = k - 1
    p_het = float(1 - sp_stats.chi2.cdf(Q, df))
    
    # --- Step 2: Estimate tau² ---
    if method == "dl":
        tau2 = _tau2_dl(Q, w_fe, df)
    elif method == "reml":
        tau2 = _tau2_reml(effects, variances)
    else:
        tau2 = _tau2_dl(Q, w_fe, df)
    
    # --- Step 3: Random-effects weights ---
    w_re = 1.0 / (variances + tau2)
    mu_re = float(np.sum(effects * w_re) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    
    ci_lower = mu_re - 1.96 * se_re
    ci_upper = mu_re + 1.96 * se_re
    z = mu_re / se_re if se_re > 0 else 0
    p = float(2 * (1 - sp_stats.norm.cdf(abs(z))))
    
    # --- Step 4: Heterogeneity ---
    I2 = max(0, 100 * (Q - df) / Q) if Q > 0 else 0
    H2 = Q / df if df > 0 else 1
    
    # Prediction interval (where the true effect of a NEW study might fall)
    if tau2 > 0 and k > 2:
        t_crit = sp_stats.t.ppf(0.975, df=k-2)
        pi_se = np.sqrt(tau2 + se_re**2)
        pi_lower = mu_re - t_crit * pi_se
        pi_upper = mu_re + t_crit * pi_se
    else:
        pi_lower = ci_lower
        pi_upper = ci_upper
    
    return {
        "model": f"random_effects_{method}",
        "pooled_effect": round(mu_re, 4),
        "se": round(se_re, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "z": round(z, 4),
        "p_value": round(p, 6),
        "k": k,
        "heterogeneity": {
            "Q": round(Q, 4),
            "df": df,
            "p_heterogeneity": round(p_het, 6),
            "I2": round(I2, 1),
            "I2_interpretation": _interpret_i2(I2),
            "H2": round(H2, 2),
            "tau2": round(float(tau2), 6),
            "tau": round(float(np.sqrt(tau2)), 4),
        },
        "prediction_interval": {
            "lower": round(float(pi_lower), 4),
            "upper": round(float(pi_upper), 4),
        },
        "weights_percent": np.round(w_re / np.sum(w_re) * 100, 2).tolist(),
    }


def _tau2_dl(Q: float, w: np.ndarray, df: int) -> float:
    """DerSimonian-Laird estimator for tau²"""
    C = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0, (Q - df) / C)
    return tau2


def _tau2_reml(effects: np.ndarray, variances: np.ndarray) -> float:
    """REML estimator for tau² (iterative)"""
    def neg_reml_ll(tau2):
        if tau2 < 0:
            return 1e10
        w = 1.0 / (variances + tau2)
        mu = np.sum(effects * w) / np.sum(w)
        ll = -0.5 * (np.sum(np.log(variances + tau2)) + 
                      np.sum(w * (effects - mu)**2) +
                      np.log(np.sum(w)))
        return -ll
    
    result = minimize_scalar(neg_reml_ll, bounds=(0, 10), method='bounded')
    return max(0, result.x)


def _interpret_i2(i2: float) -> str:
    if i2 < 25:
        return "low"
    elif i2 < 50:
        return "moderate"
    elif i2 < 75:
        return "substantial"
    else:
        return "considerable"


# ======================================================================
# Subgroup Analysis
# ======================================================================

def subgroup_analysis(effects: np.ndarray, variances: np.ndarray,
                      groups: np.ndarray, method: str = "dl") -> dict:
    """
    Subgroup analysis (分組分析)。
    
    Args:
        groups: array of group labels (same length as effects)
    
    Returns:
        Per-group results + test for subgroup differences
    """
    unique_groups = np.unique(groups)
    subgroup_results = {}
    pooled_effects = []
    pooled_variances = []
    
    for group in unique_groups:
        mask = groups == group
        if np.sum(mask) < 1:
            continue
        
        group_effects = effects[mask]
        group_variances = variances[mask]
        
        if len(group_effects) >= 2:
            result = random_effects_meta(group_effects, group_variances, method)
        elif len(group_effects) == 1:
            result = {
                "pooled_effect": float(group_effects[0]),
                "se": float(np.sqrt(group_variances[0])),
                "k": 1,
            }
        else:
            continue
        
        subgroup_results[str(group)] = result
        pooled_effects.append(result["pooled_effect"])
        pooled_variances.append(result.get("se", 0)**2)
    
    # Test for subgroup differences (Q_between)
    test_diff = None
    if len(pooled_effects) >= 2:
        pe = np.array(pooled_effects)
        pv = np.array(pooled_variances)
        w = 1.0 / pv
        mu_overall = np.sum(pe * w) / np.sum(w)
        Q_between = float(np.sum(w * (pe - mu_overall)**2))
        df_between = len(pe) - 1
        p_between = float(1 - sp_stats.chi2.cdf(Q_between, df_between))
        
        test_diff = {
            "Q_between": round(Q_between, 4),
            "df": df_between,
            "p_value": round(p_between, 6),
            "significant": p_between < 0.05,
        }
    
    return {
        "subgroups": subgroup_results,
        "test_for_differences": test_diff,
    }


# ======================================================================
# Sensitivity Analyses
# ======================================================================

def leave_one_out(effects: np.ndarray, variances: np.ndarray,
                  study_labels: List[str], method: str = "dl") -> dict:
    """
    Leave-one-out sensitivity analysis.
    每次移除一篇，重算合併效應。
    """
    k = len(effects)
    results = []
    
    for i in range(k):
        mask = np.ones(k, dtype=bool)
        mask[i] = False
        
        loo_result = random_effects_meta(effects[mask], variances[mask], method)
        
        results.append({
            "excluded_study": study_labels[i],
            "pooled_effect": loo_result["pooled_effect"],
            "ci_lower": loo_result["ci_lower"],
            "ci_upper": loo_result["ci_upper"],
            "I2": loo_result.get("heterogeneity", {}).get("I2"),
        })
    
    # Identify influential studies
    full_result = random_effects_meta(effects, variances, method)
    full_effect = full_result["pooled_effect"]
    
    influential = []
    for r in results:
        # Study is influential if removing it changes significance
        full_sig = (full_result["ci_lower"] > 0) or (full_result["ci_upper"] < 0)
        loo_sig = (r["ci_lower"] > 0) or (r["ci_upper"] < 0)
        
        if full_sig != loo_sig:
            influential.append(r["excluded_study"])
    
    return {
        "leave_one_out": results,
        "influential_studies": influential,
        "full_analysis": {
            "pooled_effect": full_effect,
            "ci_lower": full_result["ci_lower"],
            "ci_upper": full_result["ci_upper"],
        },
    }


def cumulative_meta(effects: np.ndarray, variances: np.ndarray,
                    study_labels: List[str], sort_by: str = "year",
                    years: List[int] = None) -> dict:
    """
    累積 meta-analysis (按年份排序，逐步加入研究)。
    """
    if years is not None:
        order = np.argsort(years)
    else:
        order = np.arange(len(effects))
    
    results = []
    for i in range(1, len(order) + 1):
        idx = order[:i]
        r = random_effects_meta(effects[idx], variances[idx])
        results.append({
            "added_study": study_labels[order[i-1]],
            "k": i,
            "pooled_effect": r["pooled_effect"],
            "ci_lower": r["ci_lower"],
            "ci_upper": r["ci_upper"],
        })
    
    return {"cumulative": results}


# ======================================================================
# Publication Bias
# ======================================================================

def eggers_test(effects: np.ndarray, variances: np.ndarray) -> dict:
    """
    Egger's regression test for funnel plot asymmetry.
    
    回歸: z_i = a + b * precision_i
    其中 z_i = effect_i / se_i, precision_i = 1/se_i
    
    顯著的 intercept (a) → funnel plot asymmetry → 可能有 publication bias
    """
    se = np.sqrt(variances)
    precision = 1.0 / se
    z_scores = effects / se
    
    # Weighted linear regression
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(precision, z_scores)
    
    return {
        "test": "Egger's regression",
        "intercept": round(intercept, 4),
        "intercept_se": round(std_err, 4),
        "intercept_p": round(p_value, 6),
        "significant": p_value < 0.1,  # Use 0.1 for Egger's (convention)
        "interpretation": (
            "Significant funnel plot asymmetry detected (potential publication bias)"
            if p_value < 0.1
            else "No significant funnel plot asymmetry"
        ),
    }


def beggs_test(effects: np.ndarray, variances: np.ndarray) -> dict:
    """
    Begg's rank correlation test.
    Kendall's tau between standardized effect sizes and variances.
    """
    se = np.sqrt(variances)
    
    # Standardize effects
    w = 1.0 / variances
    mu = np.sum(effects * w) / np.sum(w)
    std_effects = (effects - mu) / se
    
    tau, p = sp_stats.kendalltau(std_effects, variances)
    
    return {
        "test": "Begg's rank correlation",
        "kendall_tau": round(float(tau), 4),
        "p_value": round(float(p), 6),
        "significant": p < 0.1,
    }


def trim_and_fill(effects: np.ndarray, variances: np.ndarray,
                  side: str = "left") -> dict:
    """
    Trim-and-fill 方法估計缺失研究數。
    
    簡化版 — 完整版建議用 R 的 metafor::trimfill()
    """
    k = len(effects)
    
    # Sort by effect size
    order = np.argsort(effects)
    sorted_effects = effects[order]
    sorted_variances = variances[order]
    
    # Estimate number of missing studies (R0 estimator)
    result = random_effects_meta(sorted_effects, sorted_variances)
    mu = result["pooled_effect"]
    
    # Count studies on each side of pooled effect
    n_left = np.sum(sorted_effects < mu)
    n_right = k - n_left
    
    # Estimated missing
    if side == "left":
        n_missing = max(0, n_right - n_left)
    else:
        n_missing = max(0, n_left - n_right)
    
    # Adjusted analysis (add imputed studies)
    if n_missing > 0:
        # Mirror the most extreme studies
        if side == "left":
            to_mirror = sorted_effects[-n_missing:]
            mirrored = 2 * mu - to_mirror
            mirrored_var = sorted_variances[-n_missing:]
        else:
            to_mirror = sorted_effects[:n_missing]
            mirrored = 2 * mu - to_mirror
            mirrored_var = sorted_variances[:n_missing]
        
        adj_effects = np.concatenate([effects, mirrored])
        adj_variances = np.concatenate([variances, mirrored_var])
        adj_result = random_effects_meta(adj_effects, adj_variances)
    else:
        adj_result = result
    
    return {
        "method": "trim_and_fill",
        "side": side,
        "n_missing_estimated": int(n_missing),
        "original_k": k,
        "adjusted_k": k + n_missing,
        "original_effect": result["pooled_effect"],
        "adjusted_effect": adj_result["pooled_effect"],
        "adjusted_ci_lower": adj_result["ci_lower"],
        "adjusted_ci_upper": adj_result["ci_upper"],
    }


# ======================================================================
# Meta-Regression (簡化版)
# ======================================================================

def meta_regression(effects: np.ndarray, variances: np.ndarray,
                    moderator: np.ndarray, moderator_name: str = "x") -> dict:
    """
    Mixed-effects meta-regression (method of moments).
    
    model: effect_i = beta0 + beta1 * moderator_i + u_i + e_i
    
    這是簡化版，完整版建議用 statsmodels 的 WLS 或 rpy2 + metafor。
    """
    from scipy.stats import linregress
    
    k = len(effects)
    se = np.sqrt(variances)
    
    # Weighted regression (inverse variance weights)
    w = 1.0 / variances
    
    # WLS: weight both x and y
    w_sqrt = np.sqrt(w)
    slope, intercept, r_value, p_value, std_err = linregress(
        moderator * w_sqrt, effects * w_sqrt
    )
    
    # R² analog (proportion of heterogeneity explained)
    full_result = random_effects_meta(effects, variances)
    tau2_full = full_result.get("heterogeneity", {}).get("tau2", 0)
    
    # Residual heterogeneity
    residuals = effects - (intercept + slope * moderator)
    w_res = 1.0 / variances
    Q_res = float(np.sum(w_res * residuals**2))
    
    R2 = max(0, 1 - Q_res / full_result.get("heterogeneity", {}).get("Q", Q_res))
    
    return {
        "moderator": moderator_name,
        "intercept": round(intercept, 4),
        "slope": round(slope, 4),
        "slope_se": round(std_err, 4),
        "slope_p": round(p_value, 6),
        "R2_analog": round(R2 * 100, 1),
        "interpretation": (
            f"{moderator_name} {'significantly' if p_value < 0.05 else 'does not significantly'} "
            f"moderate the effect (p={p_value:.4f}), explaining {R2*100:.1f}% of heterogeneity"
        ),
    }


# ======================================================================
# Convenience: Run Full Analysis Pipeline
# ======================================================================

def run_full_meta_analysis(meta_data: List[dict], 
                           subgroup_vars: dict = None) -> dict:
    """
    一步到位：跑完所有統計分析。
    
    Args:
        meta_data: list of {study_id, citation, effect, se, variance, ...}
        subgroup_vars: {"var_name": [group_labels]} for subgroup analysis
    
    Returns:
        Complete statistical results dict
    """
    # Prepare arrays
    valid = [d for d in meta_data if d.get("effect") is not None and d.get("se") is not None]
    
    if len(valid) < 2:
        return {"error": f"Only {len(valid)} valid studies, need ≥2"}
    
    effects = np.array([d["effect"] for d in valid])
    ses = np.array([d["se"] for d in valid])
    variances = ses ** 2
    labels = [d.get("citation", d["study_id"]) for d in valid]
    
    results = {"k": len(valid), "studies": labels}
    
    # --- Primary analysis ---
    results["fixed_effect"] = fixed_effect_meta(effects, variances)
    results["random_effects_dl"] = random_effects_meta(effects, variances, "dl")
    results["random_effects_reml"] = random_effects_meta(effects, variances, "reml")
    
    # Choose recommended model
    i2 = results["random_effects_dl"]["heterogeneity"]["I2"]
    if i2 < 25:
        results["recommended_model"] = "fixed_effect"
        results["primary_analysis"] = results["fixed_effect"]
    else:
        results["recommended_model"] = "random_effects_dl"
        results["primary_analysis"] = results["random_effects_dl"]
    
    # --- Sensitivity ---
    results["leave_one_out"] = leave_one_out(effects, variances, labels)
    
    # --- Publication bias ---
    if len(valid) >= 10:  # Egger's test needs ≥10 studies
        results["eggers_test"] = eggers_test(effects, variances)
        results["beggs_test"] = beggs_test(effects, variances)
        results["trim_and_fill"] = trim_and_fill(effects, variances)
    else:
        results["publication_bias_note"] = (
            f"Only {len(valid)} studies — publication bias tests unreliable with <10 studies"
        )
    
    # --- Subgroup analyses ---
    if subgroup_vars:
        results["subgroup_analyses"] = {}
        for var_name, group_labels in subgroup_vars.items():
            if len(group_labels) == len(valid):
                # Sanitize: replace None with "unknown"
                sanitized = [str(g) if g is not None else "unknown" for g in group_labels]
                groups = np.array(sanitized)
                # Only run if ≥2 distinct non-unknown groups
                unique_real = set(sanitized) - {"unknown"}
                if len(unique_real) >= 2:
                    results["subgroup_analyses"][var_name] = subgroup_analysis(
                        effects, variances, groups
                    )
    
    # --- Summary ---
    pa = results["primary_analysis"]
    sig = "significant" if pa["p_value"] < 0.05 else "not significant"
    results["summary"] = (
        f"Pooled effect: {pa['pooled_effect']:.3f} "
        f"(95% CI: {pa['ci_lower']:.3f} to {pa['ci_upper']:.3f}), "
        f"p={pa['p_value']:.4f} ({sig}). "
        f"Heterogeneity: I²={i2:.1f}% ({_interpret_i2(i2)})."
    )
    
    return results
