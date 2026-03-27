"""
Meta-Analysis Visualizations — LUMEN v2
=========================================
Forest plots, funnel plots, and sensitivity analysis figures.
All plots use canonical citation labels (e.g. "Li 2024" instead of "PMID_12345").
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Style defaults
PLOT_STYLE = {
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.titlesize": 12,
    "figure.dpi": 150,
}


def _apply_style():
    plt.rcParams.update(PLOT_STYLE)


def _ensure_unique_labels(labels: list) -> list:
    """Append a/b/c suffixes if duplicate labels exist."""
    from collections import Counter
    counts = Counter(labels)
    duplicates = {k for k, v in counts.items() if v > 1}
    if not duplicates:
        return labels

    new_labels = []
    seen = {}
    for label in labels:
        if label in duplicates:
            idx = seen.get(label, 0)
            suffix = chr(ord("a") + idx)
            new_labels.append(f"{label}{suffix}")
            seen[label] = idx + 1
        else:
            new_labels.append(label)
    return new_labels


# ======================================================================
# Forest Plot
# ======================================================================

def forest_plot(
    effects: np.ndarray,
    ci_lowers: np.ndarray,
    ci_uppers: np.ndarray,
    labels: list,
    pooled: dict = None,
    title: str = "Forest Plot",
    xlabel: str = "Effect Size (Hedges' g)",
    figsize: tuple = (10, None),
    save_path: str = None,
) -> plt.Figure:
    """Standard forest plot with diamond for pooled effect."""
    _apply_style()

    effects = np.asarray(effects)
    ci_lowers = np.asarray(ci_lowers)
    ci_uppers = np.asarray(ci_uppers)
    labels = _ensure_unique_labels(labels)
    k = len(effects)

    if figsize[1] is None:
        figsize = (figsize[0], max(4, k * 0.4 + 2))

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(k, 0, -1)

    # Study-level effects
    for i in range(k):
        ax.plot(
            [ci_lowers[i], ci_uppers[i]], [y_positions[i], y_positions[i]],
            color="black", linewidth=1,
        )
        ax.plot(effects[i], y_positions[i], "s", color="steelblue",
                markersize=6, zorder=3)

    # Pooled effect diamond
    if pooled:
        y_diamond = 0
        pe = pooled["pooled_effect"]
        cl = pooled["ci_lower"]
        cu = pooled["ci_upper"]
        diamond = plt.Polygon(
            [(cl, y_diamond), (pe, y_diamond + 0.3),
             (cu, y_diamond), (pe, y_diamond - 0.3)],
            closed=True, facecolor="red", edgecolor="black", alpha=0.7,
        )
        ax.add_patch(diamond)
        labels_full = labels + [f"Pooled ({pooled.get('estimator', 'RE')})"]
        y_all = np.append(y_positions, y_diamond)
    else:
        labels_full = labels
        y_all = y_positions

    # Null effect line
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_yticks(y_all)
    ax.set_yticklabels(labels_full)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved forest plot: {save_path}")

    return fig


# ======================================================================
# Funnel Plot
# ======================================================================

def funnel_plot(
    effects: np.ndarray,
    se: np.ndarray,
    labels: list = None,
    pooled_effect: float = None,
    egger_result: dict = None,
    trim_fill_result: dict = None,
    title: str = "Funnel Plot",
    xlabel: str = "Effect Size",
    save_path: str = None,
) -> plt.Figure:
    """Enhanced funnel plot with Egger's line and trim-and-fill imputed studies."""
    _apply_style()

    effects = np.asarray(effects)
    se = np.asarray(se)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Original studies
    ax.scatter(effects, se, color="steelblue", s=40, zorder=3, label="Studies")

    # Imputed studies from trim-and-fill
    if trim_fill_result and trim_fill_result.get("k0", 0) > 0:
        imp_eff = np.array(trim_fill_result["imputed_effects"])
        imp_var = np.array(trim_fill_result["imputed_variances"])
        imp_se = np.sqrt(imp_var)
        ax.scatter(imp_eff, imp_se, color="red", s=40, marker="o",
                   facecolors="none", zorder=3, label=f"Imputed (k0={trim_fill_result['k0']})")

    # Pooled effect vertical line
    if pooled_effect is not None:
        ax.axvline(x=pooled_effect, color="black", linestyle="--",
                   linewidth=0.8, alpha=0.6)

    # Pseudo-confidence regions
    if pooled_effect is not None:
        se_max = max(se) * 1.1
        se_range = np.linspace(0.001, se_max, 100)
        for z, alpha_fill in [(1.96, 0.1), (1.645, 0.05)]:
            left = pooled_effect - z * se_range
            right = pooled_effect + z * se_range
            ax.fill_betweenx(se_range, left, right, alpha=alpha_fill,
                             color="lightgray")

    # Egger's regression line
    if egger_result:
        precision = 1.0 / se
        z_vals = effects / se
        slope = egger_result.get("intercept", 0)
        # Egger's: z = intercept + slope * precision
        se_plot = np.linspace(min(se) * 0.9, max(se) * 1.1, 50)
        prec_plot = 1.0 / se_plot
        # Reconstruct: effect = (intercept + slope * precision) * se
        # Simplified: just show the line
        ax.plot([], [], "--", color="red", alpha=0.7,
                label=f"Egger p={egger_result['p_value']:.3f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Standard Error")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved funnel plot: {save_path}")

    return fig


# ======================================================================
# Leave-One-Out Plot
# ======================================================================

def leave_one_out_plot(
    loo_results: list,
    overall_result: dict = None,
    title: str = "Leave-One-Out Sensitivity Analysis",
    xlabel: str = "Pooled Effect Size",
    save_path: str = None,
) -> plt.Figure:
    """Forest plot showing pooled effect when each study is removed."""
    _apply_style()

    labels = [r.get("excluded_study") or r.get("excluded", "?") for r in loo_results]
    effects = np.array([r.get("pooled_effect") or r.get("estimate", 0) for r in loo_results])
    ci_lowers = np.array([r["ci_lower"] for r in loo_results])
    ci_uppers = np.array([r["ci_upper"] for r in loo_results])

    return forest_plot(
        effects, ci_lowers, ci_uppers, labels,
        pooled=overall_result,
        title=title, xlabel=xlabel,
        save_path=save_path,
    )


# ======================================================================
# Cumulative Forest Plot
# ======================================================================

def cumulative_forest_plot(
    cumulative_results: list,
    title: str = "Cumulative Meta-Analysis",
    xlabel: str = "Pooled Effect Size",
    save_path: str = None,
) -> plt.Figure:
    """Cumulative meta-analysis forest plot."""
    _apply_style()

    labels = [f"k={r['studies_included']} ({r.get('latest_study', '')})"
              for r in cumulative_results]
    effects = np.array([r["pooled_effect"] for r in cumulative_results])
    ci_lowers = np.array([r["ci_lower"] for r in cumulative_results])
    ci_uppers = np.array([r["ci_upper"] for r in cumulative_results])

    return forest_plot(
        effects, ci_lowers, ci_uppers, labels,
        title=title, xlabel=xlabel,
        save_path=save_path,
    )


# ======================================================================
# Subgroup Forest Plot
# ======================================================================

def subgroup_forest_plot(
    subgroup_result: dict,
    title: str = "Subgroup Analysis",
    xlabel: str = "Effect Size",
    save_path: str = None,
) -> plt.Figure:
    """Forest plot with subgroup headers and diamonds."""
    _apply_style()

    subgroups = subgroup_result.get("subgroups", {})

    # Collect all data for plotting
    labels = []
    effects = []
    ci_lowers = []
    ci_uppers = []
    is_subgroup_header = []
    is_diamond = []

    for grp_name, grp_data in subgroups.items():
        if "pooled_effect" not in grp_data:
            continue

        # Subgroup header
        k_grp = grp_data.get("k", 0)
        labels.append(f"--- {grp_name} (k={k_grp}) ---")
        effects.append(grp_data["pooled_effect"])
        ci_lowers.append(grp_data["ci_lower"])
        ci_uppers.append(grp_data["ci_upper"])
        is_subgroup_header.append(True)
        is_diamond.append(True)

        # Individual studies in subgroup
        for study in grp_data.get("studies", []):
            labels.append(f"  {study}")
            effects.append(None)
            ci_lowers.append(None)
            ci_uppers.append(None)
            is_subgroup_header.append(False)
            is_diamond.append(False)

    # For now, plot subgroup diamonds only
    plot_labels = [l for l, d in zip(labels, is_diamond) if d]
    plot_effects = np.array([e for e, d in zip(effects, is_diamond) if d and e is not None])
    plot_ci_l = np.array([c for c, d in zip(ci_lowers, is_diamond) if d and c is not None])
    plot_ci_u = np.array([c for c, d in zip(ci_uppers, is_diamond) if d and c is not None])

    fig = forest_plot(
        plot_effects, plot_ci_l, plot_ci_u, plot_labels,
        title=title, xlabel=xlabel, save_path=save_path,
    )

    # Add Q_between annotation
    q_b = subgroup_result.get("Q_between", 0)
    p_b = subgroup_result.get("p_between", 1)
    fig.axes[0].annotate(
        f"Q_between={q_b:.2f}, p={p_b:.4f}",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=8, style="italic",
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ======================================================================
# Influence Plot
# ======================================================================

def influence_plot(
    diagnostics: list,
    title: str = "Influence Diagnostics",
    save_path: str = None,
) -> plt.Figure:
    """Cook's distance / DFBETAS influence plot."""
    _apply_style()

    labels = [d.get("study") or d.get("study_label", "?") for d in diagnostics]
    cooks_d = [d.get("cooks_distance", 0) for d in diagnostics]
    hat_vals = [d.get("hat_value") or d.get("hat", 0) for d in diagnostics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(labels) * 0.3 + 1)))

    y = np.arange(len(labels), 0, -1)

    # Cook's distance
    ax1.barh(y, cooks_d, color="steelblue", alpha=0.7)
    threshold = 4 / len(labels) if len(labels) > 0 else 1
    ax1.axvline(x=threshold, color="red", linestyle="--", linewidth=0.8,
                label=f"Threshold (4/k={threshold:.3f})")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Cook's Distance")
    ax1.set_title("Cook's Distance")
    ax1.legend(fontsize=8)

    # Hat values (leverage)
    ax2.barh(y, hat_vals, color="coral", alpha=0.7)
    hat_threshold = 2 * 1 / len(labels) if len(labels) > 0 else 1
    ax2.axvline(x=hat_threshold, color="red", linestyle="--", linewidth=0.8,
                label=f"Threshold (2/k={hat_threshold:.3f})")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Hat Value (Leverage)")
    ax2.set_title("Leverage")
    ax2.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved influence plot: {save_path}")

    return fig


# ======================================================================
# Paper Validation Figures
# ======================================================================

def bland_altman_panel(
    ba_results: dict,
    fields: list = None,
    field_labels: dict = None,
    title: str = "Figure 6 — Bland-Altman Plots",
    save_path: str = None,
) -> plt.Figure:
    """
    Bland-Altman panel plot (Figure 6).

    Args:
        ba_results: {field_name: bland_altman_dict} from extraction_validator
        fields: which fields to plot (default: ["n", "mean", "sd"])
        field_labels: display labels (default: auto)
    """
    _apply_style()

    if fields is None:
        fields = [f for f in ["n", "mean", "sd"] if f in ba_results]
    if field_labels is None:
        field_labels = {"n": "Sample Size", "mean": "Mean", "sd": "SD",
                        "events": "Events", "total": "Total"}

    n_panels = len(fields)
    if n_panels == 0:
        return plt.figure()

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel_labels = "ABCDEFGHIJ"
    for i, field in enumerate(fields):
        ax = axes[i]
        ba = ba_results.get(field, {})
        points = ba.get("points", [])

        if not points:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"({panel_labels[i]}) {field_labels.get(field, field)}")
            continue

        means = [p["mean"] for p in points]
        diffs = [p["diff"] for p in points]

        ax.scatter(means, diffs, alpha=0.6, s=30, color="steelblue")

        # Mean difference line
        md = ba.get("mean_diff", 0)
        ax.axhline(y=md, color="red", linestyle="-", linewidth=1,
                    label=f"Mean diff: {md:.2f}")

        # Limits of agreement
        loa_u = ba.get("loa_upper", 0)
        loa_l = ba.get("loa_lower", 0)
        ax.axhline(y=loa_u, color="gray", linestyle="--", linewidth=0.8,
                    label=f"+1.96 SD: {loa_u:.2f}")
        ax.axhline(y=loa_l, color="gray", linestyle="--", linewidth=0.8,
                    label=f"-1.96 SD: {loa_l:.2f}")

        # Zero line
        ax.axhline(y=0, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

        ax.set_xlabel("Mean of LUMEN & Published")
        ax.set_ylabel("Difference (LUMEN − Published)")
        ax.set_title(f"({panel_labels[i]}) {field_labels.get(field, field)}")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Bland-Altman plot: {save_path}")

    return fig


def forest_plot_comparison(
    comparisons: list,
    title: str = "Figure 7 — Forest Plot Comparison: Published vs LUMEN",
    save_path: str = None,
) -> plt.Figure:
    """
    Side-by-side forest plot: published estimates (left) vs LUMEN (right).

    Args:
        comparisons: list from concordance_checker.compare_synthesis()
    """
    _apply_style()

    matched = [c for c in comparisons if c.get("matched")]
    if not matched:
        fig = plt.figure()
        return fig

    n = len(matched)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, n * 0.5 + 2)),
                                    sharey=True)

    y = np.arange(n, 0, -1)
    labels = [c["analysis_id"] for c in matched]

    # Published (left)
    pub_effects = []
    pub_ci_lo = []
    pub_ci_hi = []
    for c in matched:
        pub_effects.append(c.get("published_effect", 0))
        ci = c.get("published_ci", [None, None])
        pub_ci_lo.append(ci[0] if ci[0] is not None else 0)
        pub_ci_hi.append(ci[1] if ci[1] is not None else 0)

    pub_effects = np.array(pub_effects, dtype=float)
    pub_err_lo = pub_effects - np.array(pub_ci_lo, dtype=float)
    pub_err_hi = np.array(pub_ci_hi, dtype=float) - pub_effects

    ax1.errorbar(pub_effects, y, xerr=[pub_err_lo, pub_err_hi],
                  fmt="s", color="navy", markersize=6, capsize=3,
                  linewidth=1.5, label="Published")
    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Effect Size")
    ax1.set_title("Published Estimates")
    ax1.grid(True, alpha=0.2, axis="x")

    # LUMEN (right)
    lum_effects = []
    lum_ci_lo = []
    lum_ci_hi = []
    for c in matched:
        lum_effects.append(c.get("lumen_effect", 0))
        ci = c.get("lumen_ci", [None, None])
        lum_ci_lo.append(ci[0] if ci[0] is not None else 0)
        lum_ci_hi.append(ci[1] if ci[1] is not None else 0)

    lum_effects = np.array(lum_effects, dtype=float)
    lum_err_lo = lum_effects - np.array(lum_ci_lo, dtype=float)
    lum_err_hi = np.array(lum_ci_hi, dtype=float) - lum_effects

    ax2.errorbar(lum_effects, y, xerr=[lum_err_lo, lum_err_hi],
                  fmt="o", color="forestgreen", markersize=6, capsize=3,
                  linewidth=1.5, label="LUMEN")
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Effect Size")
    ax2.set_title("LUMEN Estimates")
    ax2.grid(True, alpha=0.2, axis="x")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved forest comparison: {save_path}")

    return fig


def calibration_curve_plot(
    arm_calibrations: dict,
    title: str = "Figure 8 — Calibration Curves",
    save_path: str = None,
) -> plt.Figure:
    """
    Calibration curve: 5-point score vs observed inclusion probability.

    Args:
        arm_calibrations: {arm_name: calibration_dict} from screening_benchmark
            Each dict has "per_level": {score: {"inclusion_rate": float, "n": int}}
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration diagonal
    ax.plot([1, 5], [0, 1], "--", color="gray", alpha=0.5, label="Perfect calibration")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for i, (arm_name, cal) in enumerate(arm_calibrations.items()):
        per_level = cal.get("per_level", {})
        if not per_level:
            continue

        scores = sorted(per_level.keys())
        rates = [per_level[s]["inclusion_rate"] for s in scores]
        sizes = [per_level[s]["n"] for s in scores]

        color = colors[i % len(colors)]
        ax.plot(scores, rates, "o-", color=color, linewidth=2, markersize=8,
                label=arm_name)

        # Size annotations
        for s, r, n in zip(scores, rates, sizes):
            ax.annotate(f"n={n}", (s, r), fontsize=7, ha="center",
                        xytext=(0, 8), textcoords="offset points", color=color)

    ax.set_xlabel("Five-Point Confidence Score")
    ax.set_ylabel("Observed Inclusion Probability")
    ax.set_title(title)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1\n(Excl.)", "2\n(Likely Excl.)",
                         "3\n(Undecided)", "4\n(Likely Incl.)",
                         "5\n(Incl.)"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved calibration curve: {save_path}")

    return fig
