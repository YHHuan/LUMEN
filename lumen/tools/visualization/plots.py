"""
Forest plot and funnel plot generators for meta-analysis results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


def forest_plot(
    effects: list[float],
    ci_lowers: list[float],
    ci_uppers: list[float],
    labels: list[str],
    pooled_effect: float | None = None,
    pooled_ci: tuple[float, float] | None = None,
    title: str = "Forest Plot",
    xlabel: str = "Effect Size",
    output_path: str | Path | None = None,
) -> Any:
    """
    Generate a forest plot.

    Parameters
    ----------
    effects : individual study effect sizes
    ci_lowers, ci_uppers : confidence interval bounds
    labels : study labels
    pooled_effect, pooled_ci : pooled estimate with CI
    title : plot title
    xlabel : x-axis label
    output_path : if provided, save figure to file

    Returns matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(effects)
    n_rows = k + (2 if pooled_effect is not None else 0)

    fig, ax = plt.subplots(figsize=(8, max(4, n_rows * 0.5)))

    y_positions = list(range(k - 1, -1, -1))

    # Individual studies
    for i, (eff, lo, hi, y) in enumerate(zip(effects, ci_lowers, ci_uppers, y_positions)):
        ax.plot([lo, hi], [y, y], color="#2C3E50", linewidth=1.2)
        ax.plot(eff, y, "s", color="#3498DB", markersize=7)

    # Pooled estimate (diamond)
    if pooled_effect is not None and pooled_ci is not None:
        diamond_y = -1.5
        diamond_x = [pooled_ci[0], pooled_effect, pooled_ci[1], pooled_effect]
        diamond_dy = [0, 0.3, 0, -0.3]
        ax.fill(diamond_x, [diamond_y + d for d in diamond_dy],
                color="#E74C3C", alpha=0.7)
        y_positions.append(diamond_y)
        labels = list(labels) + ["", "Pooled"]

    # Reference line at null
    ax.axvline(x=0, color="#95A5A6", linestyle="--", linewidth=0.8)

    # Labels
    all_y = list(range(k - 1, -1, -1))
    if pooled_effect is not None:
        all_y.append(-1.5)
        ax.set_yticks(all_y)
        ax.set_yticklabels(list(labels[:k]) + ["Pooled"], fontsize=9)
    else:
        ax.set_yticks(all_y)
        ax.set_yticklabels(labels[:k], fontsize=9)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("forest_plot_saved", path=str(output_path))

    return fig


def funnel_plot(
    effects: list[float],
    ses: list[float],
    pooled_effect: float | None = None,
    title: str = "Funnel Plot",
    xlabel: str = "Effect Size",
    output_path: str | Path | None = None,
) -> Any:
    """
    Generate a funnel plot (effect size vs SE).

    Parameters
    ----------
    effects : effect sizes
    ses : standard errors
    pooled_effect : vertical line at pooled estimate
    title : plot title
    xlabel : x-axis label
    output_path : save path

    Returns matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(effects, ses, color="#3498DB", s=40, zorder=3, edgecolor="#2C3E50")

    # Invert y-axis (SE decreases upward = more precise)
    ax.invert_yaxis()

    # Pooled estimate line
    if pooled_effect is not None:
        ax.axvline(x=pooled_effect, color="#E74C3C", linestyle="--",
                   linewidth=1, label=f"Pooled = {pooled_effect:.3f}")

        # Pseudo 95% CI funnel
        se_range = np.linspace(0.001, max(ses) * 1.2, 100)
        ci_lo = pooled_effect - 1.96 * se_range
        ci_hi = pooled_effect + 1.96 * se_range
        ax.plot(ci_lo, se_range, color="#BDC3C7", linestyle=":", linewidth=0.8)
        ax.plot(ci_hi, se_range, color="#BDC3C7", linestyle=":", linewidth=0.8)
        ax.fill_betweenx(se_range, ci_lo, ci_hi, color="#ECF0F1", alpha=0.4)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Standard Error", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if pooled_effect is not None:
        ax.legend(fontsize=9)

    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("funnel_plot_saved", path=str(output_path))

    return fig
