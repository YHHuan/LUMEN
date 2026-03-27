"""
Cost & Performance Tracker — LUMEN v2
=======================================
Aggregates API costs, token efficiency, latency, and cache performance
from the audit log. Produces detailed transparency reports.

Key metrics:
- Per-phase and per-model cost breakdown
- Token efficiency: tokens/study for each phase
- Cache hit rate and cost savings from caching
- Latency: mean, P50, P95, total wall-clock time
- Cumulative cost curve over time
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CostTracker:
    """Parse audit log and compute transparency metrics."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.audit_path = self.data_dir / ".audit" / "prompt_log.jsonl"
        self.entries: List[dict] = []

    def load(self) -> int:
        """Load audit log entries. Returns number of entries loaded."""
        self.entries = []
        if not self.audit_path.exists():
            logger.warning(f"No audit log found at {self.audit_path}")
            return 0

        with open(self.audit_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return len(self.entries)

    def full_report(self, n_studies: int = 0) -> dict:
        """Generate comprehensive cost & performance report."""
        if not self.entries:
            self.load()

        return {
            "generated_at": datetime.now().isoformat(),
            "total_api_calls": len(self.entries),
            "cost_summary": self._cost_summary(),
            "cost_by_phase": self._cost_by_phase(),
            "cost_by_model": self._cost_by_model(),
            "token_efficiency": self._token_efficiency(n_studies),
            "cache_performance": self._cache_performance(),
            "latency_stats": self._latency_stats(),
            "cost_timeline": self._cost_timeline(),
        }

    # ── Cost Breakdown ────────────────────────────────

    def _cost_summary(self) -> dict:
        total_cost = sum(e.get("estimated_cost_usd", 0) for e in self.entries)
        total_input = sum(e.get("input_tokens", 0) for e in self.entries)
        total_output = sum(e.get("output_tokens", 0) for e in self.entries)
        total_cache_read = sum(e.get("cache_read_tokens", 0) for e in self.entries)
        total_cache_write = sum(e.get("cache_write_tokens", 0) for e in self.entries)

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_write_tokens": total_cache_write,
            "avg_cost_per_call": round(total_cost / max(len(self.entries), 1), 6),
        }

    def _cost_by_phase(self) -> dict:
        phase_data: Dict[str, dict] = defaultdict(lambda: {
            "cost_usd": 0.0, "calls": 0, "input_tokens": 0,
            "output_tokens": 0, "latency_total": 0.0,
            "retries": 0, "failures": 0,
        })

        for e in self.entries:
            phase = _role_to_phase(e.get("role", "unknown"))
            d = phase_data[phase]
            d["cost_usd"] += e.get("estimated_cost_usd", 0)
            d["calls"] += 1
            d["input_tokens"] += e.get("input_tokens", 0)
            d["output_tokens"] += e.get("output_tokens", 0)
            d["latency_total"] += e.get("latency_seconds", 0)
            d["retries"] += e.get("retry_count", 0)
            if e.get("failed", False):
                d["failures"] += 1

        result = {}
        for phase in sorted(phase_data.keys()):
            d = phase_data[phase]
            result[phase] = {
                "cost_usd": round(d["cost_usd"], 6),
                "calls": d["calls"],
                "input_tokens": d["input_tokens"],
                "output_tokens": d["output_tokens"],
                "avg_latency_s": round(d["latency_total"] / max(d["calls"], 1), 2),
                "wall_clock_s": round(d["latency_total"], 1),
                "retry_rate_pct": round(d["retries"] / max(d["calls"], 1) * 100, 1),
                "failure_rate_pct": round(d["failures"] / max(d["calls"], 1) * 100, 1),
                "cost_pct": 0.0,  # filled below
            }

        total = sum(v["cost_usd"] for v in result.values())
        if total > 0:
            for v in result.values():
                v["cost_pct"] = round(v["cost_usd"] / total * 100, 1)

        return result

    def _cost_by_model(self) -> dict:
        model_data: Dict[str, dict] = defaultdict(lambda: {
            "cost_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0,
        })

        for e in self.entries:
            model = e.get("actual_model", e.get("model_id", "unknown"))
            d = model_data[model]
            d["cost_usd"] += e.get("estimated_cost_usd", 0)
            d["calls"] += 1
            d["input_tokens"] += e.get("input_tokens", 0)
            d["output_tokens"] += e.get("output_tokens", 0)

        return {
            model: {
                "cost_usd": round(d["cost_usd"], 6),
                "calls": d["calls"],
                "input_tokens": d["input_tokens"],
                "output_tokens": d["output_tokens"],
            }
            for model, d in sorted(model_data.items(),
                                   key=lambda x: x[1]["cost_usd"], reverse=True)
        }

    # ── Token Efficiency ──────────────────────────────

    def _token_efficiency(self, n_studies: int) -> dict:
        """Compute tokens-per-study and related efficiency metrics."""
        total_input = sum(e.get("input_tokens", 0) for e in self.entries)
        total_output = sum(e.get("output_tokens", 0) for e in self.entries)
        total_cost = sum(e.get("estimated_cost_usd", 0) for e in self.entries)

        result = {
            "total_studies": n_studies,
            "tokens_per_study": None,
            "cost_per_study_usd": None,
            "input_output_ratio": round(
                total_input / max(total_output, 1), 2
            ),
        }

        if n_studies > 0:
            result["tokens_per_study"] = round(
                (total_input + total_output) / n_studies
            )
            result["cost_per_study_usd"] = round(total_cost / n_studies, 6)

        # Per-phase efficiency
        phase_studies = {}
        for e in self.entries:
            phase = _role_to_phase(e.get("role", "unknown"))
            if phase not in phase_studies:
                phase_studies[phase] = {"tokens": 0, "cost": 0.0}
            phase_studies[phase]["tokens"] += (
                e.get("input_tokens", 0) + e.get("output_tokens", 0)
            )
            phase_studies[phase]["cost"] += e.get("estimated_cost_usd", 0)

        if n_studies > 0:
            result["per_phase"] = {
                phase: {
                    "tokens_per_study": round(d["tokens"] / n_studies),
                    "cost_per_study": round(d["cost"] / n_studies, 6),
                }
                for phase, d in sorted(phase_studies.items())
            }

        return result

    # ── Cache Performance ─────────────────────────────

    def _cache_performance(self) -> dict:
        """Analyze prompt cache hit rates and savings."""
        total_input = sum(e.get("input_tokens", 0) for e in self.entries)
        total_cache_read = sum(e.get("cache_read_tokens", 0) for e in self.entries)
        total_cache_write = sum(e.get("cache_write_tokens", 0) for e in self.entries)

        # Cache read tokens are charged at 10% of normal rate
        # Savings = cache_read * 0.9 * avg_input_price
        avg_prices = defaultdict(list)
        for e in self.entries:
            model = e.get("actual_model", "")
            cr = e.get("cache_read_tokens", 0)
            if cr > 0:
                avg_prices[model].append(cr)

        cache_hit_rate = (
            total_cache_read / max(total_input, 1)
        ) if total_input > 0 else 0.0

        return {
            "total_input_tokens": total_input,
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": total_cache_write,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "cache_hit_rate_pct": f"{cache_hit_rate * 100:.1f}%",
            "estimated_savings_factor": round(
                cache_hit_rate * 0.9, 4
            ),
        }

    # ── Latency Statistics ────────────────────────────

    def _latency_stats(self) -> dict:
        """Compute latency statistics across all API calls."""
        latencies = [
            e.get("latency_seconds", 0)
            for e in self.entries
            if e.get("latency_seconds", 0) > 0
        ]

        if not latencies:
            return {
                "total_wall_clock_s": 0,
                "mean_s": 0, "median_s": 0,
                "p95_s": 0, "max_s": 0,
                "n_calls": 0,
            }

        latencies.sort()
        n = len(latencies)

        return {
            "total_wall_clock_s": round(sum(latencies), 1),
            "mean_s": round(sum(latencies) / n, 2),
            "median_s": round(latencies[n // 2], 2),
            "p95_s": round(latencies[int(n * 0.95)], 2),
            "max_s": round(latencies[-1], 2),
            "min_s": round(latencies[0], 2),
            "n_calls": n,
        }

    # ── Cost Timeline ─────────────────────────────────

    def _cost_timeline(self) -> list:
        """Build cumulative cost over time for cost curve plotting."""
        if not self.entries:
            return []

        # Sort by timestamp
        sorted_entries = sorted(
            self.entries, key=lambda e: e.get("timestamp", "")
        )

        timeline = []
        cumulative = 0.0

        for e in sorted_entries:
            cost = e.get("estimated_cost_usd", 0)
            cumulative += cost
            timeline.append({
                "timestamp": e.get("timestamp", ""),
                "role": e.get("role", ""),
                "cost_usd": round(cost, 6),
                "cumulative_usd": round(cumulative, 6),
            })

        return timeline


# ======================================================================
# Report Formatting
# ======================================================================

def format_cost_report(report: dict) -> str:
    """Format cost report as human-readable text."""
    lines = [
        "=" * 65,
        "  LUMEN v2 — Cost & Performance Transparency Report",
        "=" * 65,
        "",
    ]

    cs = report["cost_summary"]
    lines.extend([
        f"  Total Cost:          ${cs['total_cost_usd']:.4f}",
        f"  Total API Calls:     {report['total_api_calls']}",
        f"  Total Tokens:        {cs['total_tokens']:,}",
        f"    Input:             {cs['total_input_tokens']:,}",
        f"    Output:            {cs['total_output_tokens']:,}",
        f"  Avg Cost/Call:       ${cs['avg_cost_per_call']:.6f}",
        "",
    ])

    # Token efficiency
    te = report["token_efficiency"]
    if te.get("tokens_per_study"):
        lines.extend([
            "  Token Efficiency:",
            "  " + "-" * 50,
            f"  Studies Processed:   {te['total_studies']}",
            f"  Tokens/Study:        {te['tokens_per_study']:,}",
            f"  Cost/Study:          ${te['cost_per_study_usd']:.4f}",
            f"  Input/Output Ratio:  {te['input_output_ratio']}:1",
            "",
        ])

    # Cost by phase
    cbp = report["cost_by_phase"]
    if cbp:
        lines.extend(["  Cost by Phase:", "  " + "-" * 55])
        for phase, data in cbp.items():
            bar_len = int(data["cost_pct"] / 5)
            bar = "#" * bar_len + "." * (20 - bar_len)
            lines.append(
                f"  {phase:<20} ${data['cost_usd']:>8.4f}  "
                f"{data['cost_pct']:>5.1f}%  [{bar}]  "
                f"({data['calls']} calls)"
            )
        lines.append("")

    # Cost by model
    cbm = report["cost_by_model"]
    if cbm:
        lines.extend(["  Cost by Model:", "  " + "-" * 55])
        for model, data in cbm.items():
            lines.append(
                f"  {model:<40} ${data['cost_usd']:>8.4f}  ({data['calls']} calls)"
            )
        lines.append("")

    # Cache performance
    cp = report["cache_performance"]
    lines.extend([
        "  Cache Performance:",
        "  " + "-" * 50,
        f"  Cache Hit Rate:      {cp['cache_hit_rate_pct']}",
        f"  Cache Read Tokens:   {cp['cache_read_tokens']:,}",
        f"  Savings Factor:      {cp['estimated_savings_factor']:.1%}",
        "",
    ])

    # Latency
    ls = report["latency_stats"]
    if ls["n_calls"] > 0:
        lines.extend([
            "  Latency:",
            "  " + "-" * 50,
            f"  Total Wall Clock:    {ls['total_wall_clock_s']:.0f}s "
            f"({ls['total_wall_clock_s']/60:.1f} min)",
            f"  Mean:                {ls['mean_s']:.2f}s",
            f"  Median (P50):        {ls['median_s']:.2f}s",
            f"  P95:                 {ls['p95_s']:.2f}s",
            f"  Max:                 {ls['max_s']:.2f}s",
            "",
        ])

    lines.append("=" * 65)
    return "\n".join(lines)


# ======================================================================
# Visualization
# ======================================================================

def generate_cost_plots(report: dict, output_dir: str):
    """Generate cost visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cost by phase bar chart
    cbp = report.get("cost_by_phase", {})
    if cbp:
        fig, ax = plt.subplots(figsize=(10, 5))
        phases = list(cbp.keys())
        costs = [cbp[p]["cost_usd"] for p in phases]
        colors = plt.cm.Set3([i / len(phases) for i in range(len(phases))])
        bars = ax.barh(phases, costs, color=colors)
        ax.set_xlabel("Cost (USD)")
        ax.set_title("LUMEN v2 — Cost by Pipeline Phase")
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_width() + max(costs) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"${cost:.4f}", va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / "cost_by_phase.png", dpi=150)
        plt.close()

    # 2. Cumulative cost curve
    timeline = report.get("cost_timeline", [])
    if timeline:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(len(timeline)))
        y = [t["cumulative_usd"] for t in timeline]
        ax.plot(x, y, "b-", linewidth=1.5)
        ax.fill_between(x, y, alpha=0.1)
        ax.set_xlabel("API Call Number")
        ax.set_ylabel("Cumulative Cost (USD)")
        ax.set_title("LUMEN v2 — Cumulative Cost Curve")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "cost_cumulative.png", dpi=150)
        plt.close()

    # 3. Cost by model pie chart
    cbm = report.get("cost_by_model", {})
    if cbm:
        fig, ax = plt.subplots(figsize=(8, 6))
        models = list(cbm.keys())
        costs = [cbm[m]["cost_usd"] for m in models]
        # Shorten model names for display
        labels = [m.split("/")[-1] if "/" in m else m for m in models]
        ax.pie(costs, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("LUMEN v2 — Cost Distribution by Model")
        plt.tight_layout()
        plt.savefig(output_dir / "cost_by_model.png", dpi=150)
        plt.close()

    logger.info(f"Cost plots saved to {output_dir}")


# ======================================================================
# Paper Figures — Multi-Domain Cost Visualizations
# ======================================================================

def generate_paper_cost_figures(
    domain_reports: Dict[str, dict],
    output_dir: str,
):
    """
    Generate paper-ready cost figures from multiple domain reports.

    Args:
        domain_reports: {domain_name: full_report_dict} from CostTracker.full_report()
        output_dir: where to save PNG files

    Produces:
        fig2_cost_by_phase_stacked.png  — Figure 2: stacked bar per dataset
        fig3_cost_by_model_tier.png     — Figure 3: grouped bar by model
        fig4_wallclock_by_phase.png     — Figure 4: box plot
        fig9_cross_domain_profile.png   — Figure 9: grouped bar per domain
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = list(domain_reports.keys())
    if not domains:
        logger.warning("No domain reports to plot")
        return

    # ── Figure 2: Per-phase cost breakdown (stacked bar) ──
    all_phases = set()
    for report in domain_reports.values():
        all_phases.update(report.get("cost_by_phase", {}).keys())
    phases = sorted(all_phases)

    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 1.5), 6))
    x = np.arange(len(domains))
    width = 0.6
    bottom = np.zeros(len(domains))

    colors = plt.cm.Set2(np.linspace(0, 1, len(phases)))
    for i, phase in enumerate(phases):
        costs = [
            domain_reports[d].get("cost_by_phase", {}).get(phase, {}).get("cost_usd", 0)
            for d in domains
        ]
        ax.bar(x, costs, width, bottom=bottom, label=phase, color=colors[i])
        bottom += np.array(costs)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Cost (USD)")
    ax.set_title("Figure 2 — Per-Phase Cost Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_cost_by_phase_stacked.png", dpi=200)
    plt.close()

    # ── Figure 3: Cost by model tier (grouped bar) ──
    all_models = set()
    for report in domain_reports.values():
        all_models.update(report.get("cost_by_model", {}).keys())
    models = sorted(all_models)
    short_models = [m.split("/")[-1] if "/" in m else m for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Token count
    model_tokens = {m: 0 for m in models}
    model_costs = {m: 0.0 for m in models}
    for report in domain_reports.values():
        cbm = report.get("cost_by_model", {})
        for m in models:
            data = cbm.get(m, {})
            model_tokens[m] += data.get("input_tokens", 0) + data.get("output_tokens", 0)
            model_costs[m] += data.get("cost_usd", 0)

    ax1.barh(short_models, [model_tokens[m] for m in models], color="steelblue")
    ax1.set_xlabel("Total Tokens")
    ax1.set_title("Token Distribution by Model")

    ax2.barh(short_models, [model_costs[m] for m in models], color="coral")
    ax2.set_xlabel("Cost (USD)")
    ax2.set_title("Cost Distribution by Model")

    plt.suptitle("Figure 3 — Cost Composition by Model Tier", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_cost_by_model_tier.png", dpi=200)
    plt.close()

    # ── Figure 4: Wall-clock time by phase (box plot) ──
    phase_times = {p: [] for p in phases}
    for report in domain_reports.values():
        cbp = report.get("cost_by_phase", {})
        for p in phases:
            wc = cbp.get(p, {}).get("wall_clock_s", 0)
            if wc > 0:
                phase_times[p].append(wc / 60.0)  # convert to minutes

    active_phases = [p for p in phases if phase_times[p]]
    if active_phases:
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [phase_times[p] for p in active_phases]
        bp = ax.boxplot(data, labels=active_phases, patch_artist=True)
        for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(active_phases)))):
            patch.set_facecolor(color)
        ax.set_ylabel("Wall-Clock Time (minutes)")
        ax.set_title("Figure 4 — Wall-Clock Time Distribution by Phase")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / "fig4_wallclock_by_phase.png", dpi=200)
        plt.close()

    # ── Figure 9: Cross-domain cost profile (grouped bar) ──
    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 2), 6))
    n_phases = len(phases)
    x = np.arange(len(domains))
    bar_width = 0.8 / max(n_phases, 1)

    for i, phase in enumerate(phases):
        costs = [
            domain_reports[d].get("cost_by_phase", {}).get(phase, {}).get("cost_usd", 0)
            for d in domains
        ]
        offset = (i - n_phases / 2 + 0.5) * bar_width
        ax.bar(x + offset, costs, bar_width, label=phase, color=colors[i])

    ax.set_xlabel("Domain")
    ax.set_ylabel("Cost (USD)")
    ax.set_title("Figure 9 — Cross-Domain Cost Profile")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "fig9_cross_domain_profile.png", dpi=200)
    plt.close()

    logger.info(f"Paper cost figures saved to {output_dir}")


def generate_cost_quality_scatter(
    arm_data: List[Dict],
    output_path: str,
):
    """
    Generate Figure 5: Cost vs Quality trade-off scatter plot.

    Args:
        arm_data: list of {arm_name, dataset, cost_usd, quality_score}
        output_path: save path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not arm_data:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    datasets = list(set(d.get("dataset", "") for d in arm_data))
    markers = ["o", "s", "D", "^", "v", "P", "*"]
    colors = plt.cm.tab10(range(len(datasets)))

    for i, ds in enumerate(datasets):
        points = [d for d in arm_data if d.get("dataset") == ds]
        x = [p["cost_usd"] for p in points]
        y = [p["quality_score"] for p in points]
        labels = [p.get("arm_name", "") for p in points]
        ax.scatter(x, y, marker=markers[i % len(markers)],
                   color=colors[i], s=80, label=ds, zorder=3)
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(lab, (xi, yi), fontsize=7, ha="left",
                        xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Total Cost (USD)")
    ax.set_ylabel("Quality Score")
    ax.set_title("Figure 5 — Cost vs. Quality Trade-off")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ======================================================================
# Role -> Phase Mapping (centralized in phase_mapping.py)
# ======================================================================

def _role_to_phase(role: str) -> str:
    from src.utils.phase_mapping import role_to_phase
    return role_to_phase(role, display=True)
