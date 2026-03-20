import os
import json
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2, t
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Configuration for input/output
# -----------------------------
INPUT_PATH = "data/PSY_MCI_rTMS/phase4_extraction/extracted_data.json"
OUTPUT_DIR = "data/PSY_MCI_rTMS/phase5_analysis"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
JSON_PATH = os.path.join(OUTPUT_DIR, "meta_analysis_results.json")
CSV_EXTRACTED_PATH = os.path.join(OUTPUT_DIR, "derived_effect_sizes.csv")

PRIMARY_OUTCOMES = ["ADAS-Cog", "MMSE", "MoCA"]

SUBGROUP_SPECS = [
    {
        "variable": "Stimulation type",
        "groups": ["rTMS", "tDCS", "tACS"]
    },
    {
        "variable": "Population",
        "groups": ["MCI only", "Mild AD only", "Mixed"]
    },
    {
        "variable": "Stimulation target",
        "groups": ["DLPFC", "Temporal", "Parietal", "Multi-site"]
    },
    {
        "variable": "Session count",
        "groups": ["≤10 sessions", ">10 sessions"]
    },
    {
        "variable": "Combined with cognitive training",
        "groups": ["Yes", "No"]
    }
]

# -----------------------------
# Utility helpers
# -----------------------------
def ensure_directories() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None

def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None

def sanitize_filename(name: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    out = str(name)
    for ch in bad:
        out = out.replace(ch, "_")
    out = out.replace(" ", "_")
    return out

def p_to_str(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def format_prisma_result(k: int, n_total: int, pooled: float, ci_l: float, ci_u: float,
                         p_value: float, i2: float, tau2: float, pi_l: float, pi_u: float) -> str:
    return (
        f"k={k} studies, N={n_total} participants; pooled g={pooled:.2f} "
        f"(95% CI: {ci_l:.2f}–{ci_u:.2f}), p={p_to_str(p_value)}, I²={i2:.0f}%, "
        f"tau²={tau2:.3f}, prediction interval: {pi_l:.2f}–{pi_u:.2f}"
    )

def extract_source_text(outcome_dict: Dict[str, Any]) -> str:
    src = outcome_dict.get("source_location", {}) if isinstance(outcome_dict, dict) else {}
    quote = src.get("quote")
    section = src.get("section")
    page = src.get("page")
    items = []
    if section is not None:
        items.append(f"section={section}")
    if page is not None:
        items.append(f"page={page}")
    if quote is not None:
        items.append(f"quote={quote}")
    return "; ".join(items)

# -----------------------------
# Data validation and loading
# -----------------------------
def validate_input_data(data: Any) -> None:
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of study records.")
    for i, study in enumerate(data):
        if not isinstance(study, dict):
            raise ValueError(f"Study record at index {i} is not a dictionary.")
        for key in ["study_id", "citation", "outcomes"]:
            if key not in study:
                raise ValueError(f"Missing required study-level key '{key}' in study index {i}.")
        if not isinstance(study["outcomes"], dict):
            raise ValueError(f"'outcomes' must be a dictionary in study index {i}.")

def load_data(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_input_data(data)
    return data

# -----------------------------
# Subgroup extraction heuristics
# -----------------------------
def infer_stimulation_type(study: Dict[str, Any]) -> Optional[str]:
    text = json.dumps(study, ensure_ascii=False).lower()
    if "tacs" in text:
        return "tACS"
    if "tdcs" in text:
        return "tDCS"
    if "rtms" in text or "repetitive transcranial magnetic stimulation" in text or "tms" in text:
        return "rTMS"
    return None

def infer_population(study: Dict[str, Any]) -> Optional[str]:
    text = json.dumps(study, ensure_ascii=False).lower()
    has_mci = "mci" in text or "mild cognitive impairment" in text
    has_ad = "alzheimer" in text or "ad dementia" in text or "mild ad" in text
    if has_mci and has_ad:
        return "Mixed"
    if has_mci:
        return "MCI only"
    if has_ad:
        return "Mild AD only"
    return None

def infer_stimulation_target(study: Dict[str, Any]) -> Optional[str]:
    text = json.dumps(study, ensure_ascii=False).lower()
    if "multi-site" in text or "multisite" in text or "multiple site" in text or "multiple target" in text:
        return "Multi-site"
    if "dlpfc" in text or "dorsolateral prefrontal" in text:
        return "DLPFC"
    if "temporal" in text:
        return "Temporal"
    if "parietal" in text:
        return "Parietal"
    return None

def infer_session_count(study: Dict[str, Any]) -> Optional[str]:
    candidates = []
    for key in ["session_count", "sessions", "n_sessions", "number_of_sessions"]:
        if key in study:
            candidates.append(study.get(key))
    text = json.dumps(study, ensure_ascii=False).lower()
    numeric = None
    for c in candidates:
        val = safe_float(c)
        if val is not None:
            numeric = val
            break
    if numeric is None:
        import re
        patterns = [
            r'(\d+)\s+sessions',
            r'(\d+)\s-session',
            r'(\d+)\s*session'
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                numeric = safe_float(m.group(1))
                break
    if numeric is None:
        return None
    return "≤10 sessions" if numeric <= 10 else ">10 sessions"

def infer_combined_cognitive_training(study: Dict[str, Any]) -> Optional[str]:
    text = json.dumps(study, ensure_ascii=False).lower()
    yes_terms = [
        "cognitive training", "cognitive rehabilitation", "computerized cognitive",
        "combined with training", "adjunctive cognitive", "paired with cognitive"
    ]
    no_terms = ["without cognitive training", "no cognitive training"]
    if any(term in text for term in no_terms):
        return "No"
    if any(term in text for term in yes_terms):
        return "Yes"
    return None

def derive_subgroup_metadata(study: Dict[str, Any]) -> Dict[str, Optional[str]]:
    return {
        "Stimulation type": infer_stimulation_type(study),
        "Population": infer_population(study),
        "Stimulation target": infer_stimulation_target(study),
        "Session count": infer_session_count(study),
        "Combined with cognitive training": infer_combined_cognitive_training(study),
    }

# -----------------------------
# Effect size calculations
# -----------------------------
def hedges_g_from_group_data(m1: float, sd1: float, n1: int, m0: float, sd0: float, n0: int) -> Optional[Tuple[float, float]]:
    if any(v is None for v in [m1, sd1, n1, m0, sd0, n0]):
        return None
    if n1 <= 1 or n0 <= 1 or sd1 <= 0 or sd0 <= 0:
        return None
    df = n1 + n0 - 2
    if df <= 0:
        return None
    sp2 = (((n1 - 1) * sd1 ** 2) + ((n0 - 1) * sd0 ** 2)) / df
    if sp2 <= 0:
        return None
    sp = math.sqrt(sp2)
    d = (m1 - m0) / sp
    j = 1 - (3 / (4 * df - 1)) if (4 * df - 1) != 0 else 1.0
    g = j * d
    var_d = (n1 + n0) / (n1 * n0) + (d ** 2) / (2 * (n1 + n0 - 2))
    var_g = (j ** 2) * var_d
    if var_g <= 0:
        return None
    return g, var_g

def approximate_g_from_md_ci(md: float, ci_l: float, ci_u: float, n1: Optional[int], n0: Optional[int]) -> Optional[Tuple[float, float]]:
    if any(v is None for v in [md, ci_l, ci_u]):
        return None
    se_md = (ci_u - ci_l) / (2 * 1.96)
    if se_md <= 0:
        return None
    if n1 is not None and n0 is not None and n1 > 1 and n0 > 1:
        # Approximate pooled SD from SE(MD) under equal-scale post means:
        # SE(MD)^2 = SDp^2 * (1/n1 + 1/n0)
        denom = (1.0 / n1) + (1.0 / n0)
        if denom <= 0:
            return None
        sdp = se_md / math.sqrt(denom)
        if sdp <= 0:
            return None
        df = n1 + n0 - 2
        if df <= 0:
            return None
        d = md / sdp
        j = 1 - (3 / (4 * df - 1)) if (4 * df - 1) != 0 else 1.0
        g = j * d
        var_d = (n1 + n0) / (n1 * n0) + (d ** 2) / (2 * (n1 + n0 - 2))
        var_g = (j ** 2) * var_d
        if var_g <= 0:
            return None
        return g, var_g
    return None

def derive_effect_size_for_outcome(study: Dict[str, Any], outcome_name: str, outcome_dict: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "study_id": study.get("study_id"),
        "citation": study.get("citation"),
        "outcome": outcome_name,
        "measure": outcome_dict.get("measure"),
        "timepoint": outcome_dict.get("timepoint"),
        "effect_size_g": None,
        "variance_g": None,
        "se_g": None,
        "n_intervention": None,
        "n_control": None,
        "n_total": None,
        "effect_source": None,
        "source_text": extract_source_text(outcome_dict)
    }

    subgroup_meta = derive_subgroup_metadata(study)
    row.update(subgroup_meta)

    m1_post = safe_float(outcome_dict.get("intervention_mean_post"))
    sd1_post = safe_float(outcome_dict.get("intervention_sd_post"))
    m1_change = safe_float(outcome_dict.get("intervention_mean_change"))
    sd1_change = safe_float(outcome_dict.get("intervention_sd_change"))
    n1 = safe_int(outcome_dict.get("intervention_n"))

    m0_post = safe_float(outcome_dict.get("control_mean_post"))
    sd0_post = safe_float(outcome_dict.get("control_sd_post"))
    m0_change = safe_float(outcome_dict.get("control_mean_change"))
    sd0_change = safe_float(outcome_dict.get("control_sd_change"))
    n0 = safe_int(outcome_dict.get("control_n"))

    md = safe_float(outcome_dict.get("mean_difference"))
    md_l = safe_float(outcome_dict.get("md_95ci_lower"))
    md_u = safe_float(outcome_dict.get("md_95ci_upper"))

    result = None
    source = None

    if all(v is not None for v in [m1_change, sd1_change, n1, m0_change, sd0_change, n0]):
        result = hedges_g_from_group_data(m1_change, sd1_change, n1, m0_change, sd0_change, n0)
        source = "change_scores"
    elif all(v is not None for v in [m1_post, sd1_post, n1, m0_post, sd0_post, n0]):
        result = hedges_g_from_group_data(m1_post, sd1_post, n1, m0_post, sd0_post, n0)
        source = "post_scores"
    elif all(v is not None for v in [md, md_l, md_u]):
        result = approximate_g_from_md_ci(md, md_l, md_u, n1, n0)
        source = "mean_difference_ci_approx"

    if result is not None:
        g, var_g = result
        row["effect_size_g"] = g
        row["variance_g"] = var_g
        row["se_g"] = math.sqrt(var_g) if var_g > 0 else None
        row["n_intervention"] = n1
        row["n_control"] = n0
        row["n_total"] = (n1 if n1 is not None else 0) + (n0 if n0 is not None else 0)
        row["effect_source"] = source
    else:
        row["n_intervention"] = n1
        row["n_control"] = n0
        row["n_total"] = (n1 if n1 is not None else 0) + (n0 if n0 is not None else 0)
        row["effect_source"] = None

    return row

def build_effect_size_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for study in data:
        outcomes = study.get("outcomes", {})
        if not isinstance(outcomes, dict):
            continue
        for outcome_name, outcome_dict in outcomes.items():
            if not isinstance(outcome_dict, dict):
                continue
            rows.append(derive_effect_size_for_outcome(study, outcome_name, outcome_dict))
    df = pd.DataFrame(rows)
    return df

# -----------------------------
# Meta-analysis core functions
# -----------------------------
def dersimonian_laird(y: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)

    valid = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y = y[valid]
    v = v[valid]

    k = len(y)
    if k == 0:
        return {
            "k": 0, "pooled_effect": None, "se": None, "ci_lower": None, "ci_upper": None,
            "z": None, "p_value": None, "Q": None, "Q_df": None, "Q_p_value": None,
            "I2": None, "tau2": None, "prediction_interval_lower": None,
            "prediction_interval_upper": None, "weights_fe": [], "weights_re": []
        }

    w_fe = 1.0 / v
    pooled_fe = np.sum(w_fe * y) / np.sum(w_fe)
    q = np.sum(w_fe * (y - pooled_fe) ** 2)
    df = k - 1
    c = np.sum(w_fe) - (np.sum(w_fe ** 2) / np.sum(w_fe))
    tau2 = max(0.0, (q - df) / c) if (k > 1 and c > 0) else 0.0

    w_re = 1.0 / (v + tau2)
    pooled = np.sum(w_re * y) / np.sum(w_re)
    se = math.sqrt(1.0 / np.sum(w_re))
    z = pooled / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    ci_lower = pooled - 1.96 * se
    ci_upper = pooled + 1.96 * se

    q_p = 1 - chi2.cdf(q, df) if df > 0 else np.nan
    i2 = max(0.0, min(100.0, ((q - df) / q) * 100.0)) if (q > 0 and df > 0) else 0.0

    if k >= 3:
        tcrit = t.ppf(0.975, df=k - 2)
        pred_se = math.sqrt(tau2 + se ** 2)
        pi_l = pooled - tcrit * pred_se
        pi_u = pooled + tcrit * pred_se
    else:
        pi_l = np.nan
        pi_u = np.nan

    return {
        "k": int(k),
        "pooled_effect": float(pooled),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "z": float(z),
        "p_value": float(p_value),
        "Q": float(q),
        "Q_df": int(df),
        "Q_p_value": float(q_p) if np.isfinite(q_p) else None,
        "I2": float(i2),
        "tau2": float(tau2),
        "prediction_interval_lower": float(pi_l) if np.isfinite(pi_l) else None,
        "prediction_interval_upper": float(pi_u) if np.isfinite(pi_u) else None,
        "weights_fe": w_fe.tolist(),
        "weights_re": w_re.tolist()
    }

def egger_test(y: np.ndarray, se: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    valid = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y = y[valid]
    se = se[valid]
    k = len(y)

    if k < 3:
        return {"k": k, "intercept": None, "slope": None, "p_value": None, "note": "Insufficient studies for Egger's test"}

    snd = y / se
    precision = 1.0 / se
    X = sm.add_constant(precision)
    model = sm.OLS(snd, X).fit()
    intercept = model.params[0]
    slope = model.params[1] if len(model.params) > 1 else None
    p_value = model.pvalues[0] if len(model.pvalues) > 0 else None

    return {
        "k": int(k),
        "intercept": float(intercept) if intercept is not None else None,
        "slope": float(slope) if slope is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "note": None
    }

def trim_and_fill_l0(y: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    valid = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y = y[valid]
    v = v[valid]
    k = len(y)

    if k < 10:
        return {
            "k0": 0,
            "filled_effects": [],
            "filled_variances": [],
            "adjusted_meta": None,
            "note": "Trim-and-fill not performed; fewer than 10 studies"
        }

    base = dersimonian_laird(y, v)
    center = base["pooled_effect"]
    left = np.sum(y < center)
    right = np.sum(y > center)
    k0 = int(abs(right - left) // 2)

    if k0 <= 0:
        return {
            "k0": 0,
            "filled_effects": [],
            "filled_variances": [],
            "adjusted_meta": base,
            "note": "No missing studies imputed"
        }

    if right > left:
        sorted_idx = np.argsort(y)[-k0:]
    else:
        sorted_idx = np.argsort(y)[:k0]

    filled_effects = []
    filled_variances = []
    for idx in sorted_idx:
        yi = y[idx]
        vi = v[idx]
        mirrored = 2 * center - yi
        filled_effects.append(float(mirrored))
        filled_variances.append(float(vi))

    y_aug = np.concatenate([y, np.array(filled_effects)])
    v_aug = np.concatenate([v, np.array(filled_variances)])
    adjusted = dersimonian_laird(y_aug, v_aug)

    return {
        "k0": int(k0),
        "filled_effects": filled_effects,
        "filled_variances": filled_variances,
        "adjusted_meta": adjusted,
        "note": None
    }

def leave_one_out_analysis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    results = []
    valid_df = df.dropna(subset=["effect_size_g", "variance_g"]).copy()
    valid_df = valid_df[(valid_df["variance_g"] > 0)]
    if len(valid_df) < 2:
        return results

    for i in range(len(valid_df)):
        omitted = valid_df.iloc[i]
        loo_df = valid_df.drop(valid_df.index[i])
        ma = dersimonian_laird(loo_df["effect_size_g"].values, loo_df["variance_g"].values)
        results.append({
            "omitted_study_id": omitted["study_id"],
            "omitted_citation": omitted["citation"],
            "k": ma["k"],
            "pooled_effect": ma["pooled_effect"],
            "ci_lower": ma["ci_lower"],
            "ci_upper": ma["ci_upper"],
            "p_value": ma["p_value"],
            "I2": ma["I2"],
            "tau2": ma["tau2"]
        })
    return results

# -----------------------------
# Plotting functions
# -----------------------------
def create_forest_plot(df: pd.DataFrame, meta_result: Dict[str, Any], outcome: str, outpath: str) -> None:
    plot_df = df.dropna(subset=["effect_size_g", "variance_g"]).copy()
    plot_df = plot_df[plot_df["variance_g"] > 0].copy()
    if plot_df.empty:
        return

    plot_df["se"] = np.sqrt(plot_df["variance_g"])
    plot_df["ci_l"] = plot_df["effect_size_g"] - 1.96 * plot_df["se"]
    plot_df["ci_u"] = plot_df["effect_size_g"] + 1.96 * plot_df["se"]
    plot_df = plot_df.sort_values("effect_size_g")

    k = len(plot_df)
    fig_h = max(4, k * 0.4 + 3)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=300)

    y_positions = np.arange(k, 0, -1)

    if meta_result.get("weights_re") and len(meta_result["weights_re"]) == k:
        weights = np.array(meta_result["weights_re"])
    else:
        weights = 1.0 / (plot_df["variance_g"].values + (meta_result.get("tau2", 0.0) or 0.0))
    weights_pct = 100 * weights / np.sum(weights)

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        y = y_positions[idx]
        ax.plot([row["ci_l"], row["ci_u"]], [y, y], color="black", lw=1)
        ax.scatter(row["effect_size_g"], y, s=20 + 2.5 * weights_pct[idx], color="black", zorder=3)

    pooled = meta_result["pooled_effect"]
    ci_l = meta_result["ci_lower"]
    ci_u = meta_result["ci_upper"]
    diamond_y = 0.3
    diamond_x = [ci_l, pooled, ci_u, pooled]
    diamond_y_coords = [diamond_y, diamond_y + 0.25, diamond_y, diamond_y - 0.25]
    ax.fill(diamond_x, diamond_y_coords, color="dimgray", alpha=0.8)

    ax.axvline(0, color="gray", linestyle="--", lw=1)
    ax.set_ylim(-1, k + 1)
    x_min = min(plot_df["ci_l"].min(), ci_l) - 0.5
    x_max = max(plot_df["ci_u"].max(), ci_u) + 0.5
    ax.set_xlim(x_min, x_max)

    ax.set_yticks([])
    ax.set_xlabel("Hedges' g (SMD)")
    ax.set_title(f"Forest plot: {outcome}")

    left_x = x_min
    right_x = x_max

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        y = y_positions[idx]
        label = f"{row['citation']} [{row['study_id']}]"
        effect_text = f"{row['effect_size_g']:.2f} ({row['ci_l']:.2f}, {row['ci_u']:.2f})"
        ax.text(left_x, y, label, ha="left", va="center", fontsize=8)
        ax.text(right_x, y, effect_text, ha="right", va="center", fontsize=8)

    pooled_text = f"{pooled:.2f} ({ci_l:.2f}, {ci_u:.2f})"
    ax.text(left_x, diamond_y, "Random-effects pooled", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.text(right_x, diamond_y, pooled_text, ha="right", va="center", fontsize=9, fontweight="bold")

    summary = (
        f"k={meta_result['k']}, I²={meta_result['I2']:.1f}%, "
        f"tau²={meta_result['tau2']:.3f}, p={p_to_str(meta_result['p_value'])}"
    )
    ax.text(0.5, -0.08, summary, transform=ax.transAxes, ha="center", va="top", fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def create_funnel_plot(df: pd.DataFrame, meta_result: Dict[str, Any], egger_result: Dict[str, Any],
                       trimfill_result: Dict[str, Any], outcome: str, outpath: str) -> None:
    plot_df = df.dropna(subset=["effect_size_g", "variance_g"]).copy()
    plot_df = plot_df[plot_df["variance_g"] > 0].copy()
    if plot_df.empty:
        return

    plot_df["se"] = np.sqrt(plot_df["variance_g"])
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.scatter(plot_df["effect_size_g"], plot_df["se"], color="black", alpha=0.8, label="Observed studies")
    pooled = meta_result["pooled_effect"]
    ax.axvline(pooled, color="gray", linestyle="--", lw=1, label="Pooled effect")

    se_range = np.linspace(plot_df["se"].min(), plot_df["se"].max(), 100)
    ax.plot(pooled - 1.96 * se_range, se_range, linestyle=":", color="gray", lw=1)
    ax.plot(pooled + 1.96 * se_range, se_range, linestyle=":", color="gray", lw=1)

    if trimfill_result is not None and trimfill_result.get("k0", 0) > 0:
        filled_effects = trimfill_result.get("filled_effects", [])
        filled_vars = trimfill_result.get("filled_variances", [])
        if len(filled_effects) > 0:
            filled_se = np.sqrt(np.array(filled_vars))
            ax.scatter(filled_effects, filled_se, marker="o", facecolors="none", edgecolors="red", s=50, label="Trim-and-fill")
            adj = trimfill_result.get("adjusted_meta", {})
            if adj and adj.get("pooled_effect") is not None:
                ax.axvline(adj["pooled_effect"], color="red", linestyle="--", lw=1, label="Adjusted pooled")

    egger_p = egger_result.get("p_value")
    annotation = f"Egger's test p={p_to_str(egger_p)}"
    if trimfill_result is not None and trimfill_result.get("k0") is not None:
        annotation += f"\nTrim-and-fill k0={trimfill_result.get('k0', 0)}"
    ax.text(0.03, 0.03, annotation, transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="lightgray"))

    ax.set_title(f"Funnel plot: {outcome}")
    ax.set_xlabel("Hedges' g (SMD)")
    ax.set_ylabel("Standard Error")
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def create_risk_of_bias_summary_figure(studies: List[Dict[str, Any]], outpath: str) -> None:
    domains = [
        "Randomization process",
        "Deviations from intended interventions",
        "Missing outcome data",
        "Measurement of the outcome",
        "Selection of the reported result",
        "Overall bias"
    ]
    levels = ["Low", "Some concerns", "High"]

    records = []
    for study in studies:
        rob = study.get("risk_of_bias", {}) if isinstance(study, dict) else {}
        row = {"study_id": study.get("study_id", "Unknown")}
        for d in domains:
            val = rob.get(d) or rob.get(d.lower()) or rob.get(d.replace(" ", "_").lower())
            if val is None:
                val = "Some concerns"
            val_str = str(val).strip().lower()
            if "low" in val_str:
                row[d] = "Low"
            elif "high" in val_str:
                row[d] = "High"
            else:
                row[d] = "Some concerns"
        records.append(row)

    if len(records) == 0:
        return

    rob_df = pd.DataFrame(records)
    pct = []
    for d in domains:
        counts = rob_df[d].value_counts(dropna=False)
        total = len(rob_df)
        pct.append([100 * counts.get(level, 0) / total for level in levels])
    pct = np.array(pct)

    colors = ["#4CAF50", "#FFC107", "#F44336"]
    fig_h = max(4, len(domains) * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=300)

    left = np.zeros(len(domains))
    y = np.arange(len(domains))
    for i, level in enumerate(levels):
        ax.barh(y, pct[:, i], left=left, color=colors[i], edgecolor="white", label=level)
        left += pct[:, i]

    ax.set_yticks(y)
    ax.set_yticklabels(domains)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of studies")
    ax.set_title("Risk of Bias Summary")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Analysis wrappers
# -----------------------------
def analyze_outcome(df_all: pd.DataFrame, outcome: str) -> Dict[str, Any]:
    df = df_all[df_all["outcome"] == outcome].copy()
    valid_df = df.dropna(subset=["effect_size_g", "variance_g"]).copy()
    valid_df = valid_df[valid_df["variance_g"] > 0].copy()

    result = {
        "outcome": outcome,
        "n_records_total": int(len(df)),
        "n_records_analyzable": int(len(valid_df)),
        "primary_meta_analysis": None,
        "prisma_summary": None,
        "subgroup_analyses": {},
        "leave_one_out": [],
        "egger_test": None,
        "trim_and_fill": None,
        "forest_plot_path": None,
        "funnel_plot_path": None,
        "included_studies": []
    }

    if valid_df.empty:
        result["primary_meta_analysis"] = {"k": 0, "note": "No analyzable studies for this outcome"}
        return result

    ma = dersimonian_laird(valid_df["effect_size_g"].values, valid_df["variance_g"].values)
    total_n = int(valid_df["n_total"].fillna(0).sum())
    prisma = format_prisma_result(
        k=ma["k"],
        n_total=total_n,
        pooled=ma["pooled_effect"],
        ci_l=ma["ci_lower"],
        ci_u=ma["ci_upper"],
        p_value=ma["p_value"],
        i2=ma["I2"],
        tau2=ma["tau2"],
        pi_l=ma["prediction_interval_lower"] if ma["prediction_interval_lower"] is not None else float("nan"),
        pi_u=ma["prediction_interval_upper"] if ma["prediction_interval_upper"] is not None else float("nan")
    )

    result["primary_meta_analysis"] = ma
    result["prisma_summary"] = prisma
    result["included_studies"] = valid_df[["study_id", "citation", "effect_size_g", "variance_g", "n_total"]].to_dict(orient="records")

    # Subgroup analyses
    subgroup_results = {}
    for spec in SUBGROUP_SPECS:
        var = spec["variable"]
        subgroup_results[var] = {}
        for grp in spec["groups"]:
            sub_df = valid_df[valid_df[var] == grp].copy()
            if len(sub_df) == 0:
                subgroup_results[var][grp] = {"k": 0, "note": "No studies in subgroup"}
            else:
                sub_ma = dersimonian_laird(sub_df["effect_size_g"].values, sub_df["variance_g"].values)
                subgroup_results[var][grp] = {
                    "k": sub_ma["k"],
                    "n_total": int(sub_df["n_total"].fillna(0).sum()),
                    "pooled_effect": sub_ma["pooled_effect"],
                    "ci_lower": sub_ma["ci_lower"],
                    "ci_upper": sub_ma["ci_upper"],
                    "p_value": sub_ma["p_value"],
                    "Q": sub_ma["Q"],
                    "Q_p_value": sub_ma["Q_p_value"],
                    "I2": sub_ma["I2"],
                    "tau2": sub_ma["tau2"],
                    "prediction_interval_lower": sub_ma["prediction_interval_lower"],
                    "prediction_interval_upper": sub_ma["prediction_interval_upper"],
                    "prisma_summary": format_prisma_result(
                        k=sub_ma["k"],
                        n_total=int(sub_df["n_total"].fillna(0).sum()),
                        pooled=sub_ma["pooled_effect"],
                        ci_l=sub_ma["ci_lower"],
                        ci_u=sub_ma["ci_upper"],
                        p_value=sub_ma["p_value"],
                        i2=sub_ma["I2"],
                        tau2=sub_ma["tau2"],
                        pi_l=sub_ma["prediction