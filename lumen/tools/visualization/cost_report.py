"""
Cost report generator for Paper 1 figures.

Generates formatted cost breakdowns and summary tables.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


def generate_cost_report(cost_summary: dict, n_studies: int = 0) -> dict:
    """
    Generate a structured cost report from CostTracker summary.

    Parameters
    ----------
    cost_summary : output of CostTracker.summary()
    n_studies : number of included studies (for per-study cost)

    Returns dict with formatted tables and totals.
    """
    phases = {}
    grand_total = {"calls": 0, "tokens": 0, "cost": 0.0}

    for phase, agents in sorted(cost_summary.items()):
        phase_total = {"calls": 0, "tokens": 0, "cost": 0.0}
        agent_rows = []

        for agent, stats in sorted(agents.items()):
            calls = stats.get("calls", 0)
            tokens = stats.get("tokens", 0)
            cost = stats.get("cost", 0.0)
            phase_total["calls"] += calls
            phase_total["tokens"] += tokens
            phase_total["cost"] += cost
            agent_rows.append({
                "agent": agent,
                "calls": calls,
                "tokens": tokens,
                "cost": cost,
            })

        phases[phase] = {
            "agents": agent_rows,
            "total": phase_total,
        }
        grand_total["calls"] += phase_total["calls"]
        grand_total["tokens"] += phase_total["tokens"]
        grand_total["cost"] += phase_total["cost"]

    per_study = None
    if n_studies > 0:
        per_study = {
            "cost_per_study": grand_total["cost"] / n_studies,
            "tokens_per_study": grand_total["tokens"] / n_studies,
            "calls_per_study": grand_total["calls"] / n_studies,
        }

    return {
        "phases": phases,
        "grand_total": grand_total,
        "per_study": per_study,
        "n_studies": n_studies,
    }


def format_cost_table(report: dict) -> str:
    """Format cost report as a readable text table."""
    lines = [
        "LUMEN v3 — Cost Report",
        "=" * 70,
    ]

    for phase, data in sorted(report["phases"].items()):
        lines.append(f"\n  Phase: {phase}")
        lines.append(f"  {'Agent':<30s} {'Calls':>6s} {'Tokens':>10s} {'Cost':>10s}")
        lines.append(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*10}")
        for row in data["agents"]:
            lines.append(
                f"  {row['agent']:<30s} {row['calls']:>6d} "
                f"{row['tokens']:>10d} ${row['cost']:>9.4f}"
            )
        t = data["total"]
        lines.append(
            f"  {'SUBTOTAL':<30s} {t['calls']:>6d} "
            f"{t['tokens']:>10d} ${t['cost']:>9.4f}"
        )

    gt = report["grand_total"]
    lines.append(f"\n{'=' * 70}")
    lines.append(
        f"  {'GRAND TOTAL':<30s} {gt['calls']:>6d} "
        f"{gt['tokens']:>10d} ${gt['cost']:>9.4f}"
    )

    if report.get("per_study"):
        ps = report["per_study"]
        lines.append(f"\n  Per-study averages (n={report['n_studies']}):")
        lines.append(f"    Cost/study:   ${ps['cost_per_study']:.4f}")
        lines.append(f"    Tokens/study: {ps['tokens_per_study']:.0f}")
        lines.append(f"    Calls/study:  {ps['calls_per_study']:.1f}")

    return "\n".join(lines)


def generate_cost_figure(report: dict, output_path: str | Path | None = None) -> Any:
    """
    Generate a cost breakdown bar chart for Paper 1.

    Returns matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = list(report["phases"].keys())
    costs = [report["phases"][p]["total"]["cost"] for p in phases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: cost by phase
    colors = plt.cm.Set2(range(len(phases)))
    bars = ax1.barh(phases, costs, color=colors, edgecolor="#2C3E50")
    ax1.set_xlabel("Cost (USD)", fontsize=10)
    ax1.set_title("Cost by Phase", fontsize=12, fontweight="bold")
    for bar, cost in zip(bars, costs):
        ax1.text(bar.get_width() + max(costs) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"${cost:.3f}", va="center", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Pie chart: proportion by phase
    if sum(costs) > 0:
        ax2.pie(costs, labels=phases, autopct="%1.1f%%", colors=colors,
                startangle=90, textprops={"fontsize": 9})
        ax2.set_title("Cost Distribution", fontsize=12, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No cost data", ha="center", va="center")

    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("cost_figure_saved", path=str(output_path))

    return fig
