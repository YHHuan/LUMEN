"""
Publication-Quality Visualizations
====================================
Forest plot, Funnel plot, Risk of Bias summary — 
仿 Lancet / NEJM 風格。

所有圖表存為 300 DPI PNG，直接可投稿。
"""

import re
from pathlib import Path
from src.utils.project import get_data_dir
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import List, Optional
import logging

import textwrap

logger = logging.getLogger(__name__)


def _default_fig_path(filename: str) -> str:
    return str(Path(get_data_dir()) / "phase5_analysis" / "figures" / filename)


# Journal-grade style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})


def forest_plot(studies: List[dict], meta_result: dict,
                output_path: str = None,
                title: str = "",
                effect_label: str = "SMD",
                null_value: float = 0,
                figsize: tuple = None) -> str:
    """
    Publication-quality Forest Plot.
    
    Args:
        studies: [{citation, effect, ci_lower, ci_upper, weight_pct, n}, ...]
        meta_result: from random_effects_meta()
        effect_label: "SMD", "MD", "OR", "RR"
        null_value: 0 for SMD/MD, 1 for OR/RR
    """
    if output_path is None:
        output_path = _default_fig_path("forest_plot.png")
    k = len(studies)
    if figsize is None:
        figsize = (10, max(4, k * 0.45 + 2))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Y positions (top to bottom)
    y_positions = list(range(k + 1, 0, -1))  # +1 for diamond
    
    # --- Individual studies ---
    max_weight = max(s.get("weight_pct", 5) for s in studies) if studies else 10
    
    for i, study in enumerate(studies):
        y = y_positions[i]
        effect = study["effect"]
        ci_l = study["ci_lower"]
        ci_u = study["ci_upper"]
        weight = study.get("weight_pct", 5)
        
        # Square size proportional to weight
        square_size = max(3, min(15, weight / max_weight * 12))
        
        # Error bar
        ax.plot([ci_l, ci_u], [y, y], color='#333333', linewidth=0.8, zorder=1)
        
        # Square marker (size = weight)
        ax.plot(effect, y, 's', color='#2166AC', markersize=square_size, 
                markeredgecolor='#2166AC', zorder=2)
        
        # Study label (left)
        ax.text(-0.02, y, study.get("citation", f"Study {i+1}"),
                ha='right', va='center', fontsize=8, transform=ax.get_yaxis_transform())
        
        # Effect + CI text (right)
        ci_text = f"{effect:.2f} [{ci_l:.2f}, {ci_u:.2f}]"
        ax.text(1.02, y, ci_text, ha='left', va='center', fontsize=8,
                transform=ax.get_yaxis_transform(), family='monospace')
    
    # --- Diamond for pooled effect ---
    y_diamond = y_positions[-1] - 0.5
    pe = meta_result["pooled_effect"]
    ci_l = meta_result["ci_lower"]
    ci_u = meta_result["ci_upper"]
    
    diamond_height = 0.3
    diamond = plt.Polygon([
        [ci_l, y_diamond],
        [pe, y_diamond + diamond_height],
        [ci_u, y_diamond],
        [pe, y_diamond - diamond_height],
    ], closed=True, facecolor='#B2182B', edgecolor='#B2182B', zorder=3)
    ax.add_patch(diamond)
    
    # Pooled label
    ax.text(-0.02, y_diamond, "Pooled estimate",
            ha='right', va='center', fontsize=8, fontweight='bold',
            transform=ax.get_yaxis_transform())
    
    pooled_text = f"{pe:.2f} [{ci_l:.2f}, {ci_u:.2f}]"
    ax.text(1.02, y_diamond, pooled_text, ha='left', va='center', fontsize=8,
            fontweight='bold', transform=ax.get_yaxis_transform(), family='monospace')
    
    # --- Null line ---
    ax.axvline(x=null_value, color='#999999', linestyle='--', linewidth=0.8, zorder=0)
    
    # --- Formatting ---
    ax.set_xlabel(effect_label, fontsize=10)
    ax.set_yticks([])
    ax.set_ylim(y_diamond - 1, y_positions[0] + 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
    
    # Header labels
    ax.text(-0.02, y_positions[0] + 0.7, "Study",
            ha='right', va='center', fontsize=9, fontweight='bold',
            transform=ax.get_yaxis_transform())
    ax.text(1.02, y_positions[0] + 0.7, f"{effect_label} [95% CI]",
            ha='left', va='center', fontsize=9, fontweight='bold',
            transform=ax.get_yaxis_transform())
    
    # Favor labels
    xlim = ax.get_xlim()
    mid = (xlim[0] + null_value) / 2
    ax.text(mid, y_diamond - 0.8, "Favors intervention", ha='center', fontsize=7, 
            style='italic', color='#555555')
    mid2 = (xlim[1] + null_value) / 2
    ax.text(mid2, y_diamond - 0.8, "Favors control", ha='center', fontsize=7,
            style='italic', color='#555555')
    
    # Heterogeneity annotation
    het = meta_result.get("heterogeneity", {})
    if het:
        het_text = (
            f"Heterogeneity: I² = {het.get('I2', 'N/A')}%, "
            f"τ² = {het.get('tau2', 'N/A')}, "
            f"p = {het.get('p_heterogeneity', 'N/A')}"
        )
        fig.text(0.12, 0.02, het_text, fontsize=7, color='#666666')
    
    plt.tight_layout(rect=[0.12, 0.05, 0.82, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Forest plot saved: {output_path}")
    return output_path


def funnel_plot(effects: np.ndarray, se: np.ndarray,
                pooled_effect: float,
                output_path: str = None,
                title: str = "Funnel Plot",
                effect_label: str = "Effect Size",
                egger_p: float = None) -> str:
    """
    Publication-quality Funnel Plot.
    """
    if output_path is None:
        output_path = _default_fig_path("funnel_plot.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Points
    ax.scatter(effects, se, s=30, c='#2166AC', edgecolors='white', 
               linewidth=0.5, zorder=3, alpha=0.8)
    
    # Vertical line at pooled effect
    ax.axvline(x=pooled_effect, color='#B2182B', linewidth=1, linestyle='-', zorder=1)
    
    # Pseudo 95% CI region (triangle)
    se_range = np.linspace(0, max(se) * 1.1, 100)
    ci_lower = pooled_effect - 1.96 * se_range
    ci_upper = pooled_effect + 1.96 * se_range
    
    ax.fill_betweenx(se_range, ci_lower, ci_upper, alpha=0.08, color='#2166AC', zorder=0)
    ax.plot(ci_lower, se_range, '--', color='#999999', linewidth=0.5, zorder=1)
    ax.plot(ci_upper, se_range, '--', color='#999999', linewidth=0.5, zorder=1)
    
    # Invert y-axis (larger SE at bottom = less precise)
    ax.invert_yaxis()
    
    ax.set_xlabel(effect_label, fontsize=10)
    ax.set_ylabel("Standard Error", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Egger's test annotation
    if egger_p is not None:
        ax.text(0.95, 0.95, f"Egger's test p = {egger_p:.4f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Funnel plot saved: {output_path}")
    return output_path


def rob_traffic_light(rob_data: List[dict],
                      output_path: str = None,
                      title: str = "Risk of Bias Summary") -> str:
    """
    Risk of Bias traffic light figure (Cochrane RoB 2 style).
    
    Each study is a row, each domain is a column.
    Green = low, Yellow = some concerns, Red = high.
    """
    if output_path is None:
        output_path = _default_fig_path("rob_summary.png")
    domains = [
        ("D1_randomization", "Randomization"),
        ("D2_deviations", "Deviations"),
        ("D3_missing_data", "Missing data"),
        ("D4_measurement", "Measurement"),
        ("D5_selection_reporting", "Reporting"),
    ]
    
    colors = {
        "low": "#4DAF4A",           # Green
        "some_concerns": "#FFD700", # Yellow
        "high": "#E41A1C",          # Red
        "unknown": "#CCCCCC",       # Grey
    }
    symbols = {
        "low": "+",
        "some_concerns": "~",
        "high": "-",
        "unknown": "?",
    }
    
    n_studies = len(rob_data)
    n_domains = len(domains)
    
    if n_studies == 0:
        logger.warning("No RoB data to plot")
        return ""
    
    fig_height = max(3, n_studies * 0.4 + 2)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    for i, study_rob in enumerate(rob_data):
        y = n_studies - i - 1
        study_label = study_rob.get("study_id", f"Study {i+1}")
        
        # Study label
        ax.text(-0.5, y, study_label, ha='right', va='center', fontsize=7)
        
        rob_domains = study_rob.get("rob2_domains", {})
        
        for j, (domain_key, domain_label) in enumerate(domains):
            domain_data = rob_domains.get(domain_key, {})
            judgement = domain_data.get("judgement", "unknown")
            
            color = colors.get(judgement, colors["unknown"])
            symbol = symbols.get(judgement, "?")
            
            circle = plt.Circle((j, y), 0.35, color=color, ec='white', linewidth=1)
            ax.add_patch(circle)
            ax.text(j, y, symbol, ha='center', va='center', fontsize=10,
                    fontweight='bold', color='white')
    
    # Domain headers
    for j, (_, label) in enumerate(domains):
        ax.text(j, n_studies + 0.3, label, ha='center', va='bottom',
                fontsize=8, fontweight='bold', rotation=30)
    
    # Overall column
    for i, study_rob in enumerate(rob_data):
        y = n_studies - i - 1
        overall = study_rob.get("overall_rob", "unknown")
        color = colors.get(overall, colors["unknown"])
        symbol = symbols.get(overall, "?")
        
        j_overall = n_domains
        circle = plt.Circle((j_overall, y), 0.35, color=color, ec='white', linewidth=1)
        ax.add_patch(circle)
        ax.text(j_overall, y, symbol, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
    
    ax.text(n_domains, n_studies + 0.3, "Overall", ha='center', va='bottom',
            fontsize=8, fontweight='bold', rotation=30)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors["low"], label="Low risk"),
        mpatches.Patch(facecolor=colors["some_concerns"], label="Some concerns"),
        mpatches.Patch(facecolor=colors["high"], label="High risk"),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)
    
    ax.set_xlim(-0.6, n_domains + 0.6)
    ax.set_ylim(-1, n_studies + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ RoB summary saved: {output_path}")
    return output_path


def rob_domain_barplot(rob_data: List[dict],
                       output_path: str = None) -> str:
    """
    Stacked bar chart showing proportion of low/some_concerns/high per domain.
    """
    if output_path is None:
        output_path = _default_fig_path("rob_domains.png")
    domains = [
        ("D1_randomization", "Randomization"),
        ("D2_deviations", "Deviations"),
        ("D3_missing_data", "Missing data"),
        ("D4_measurement", "Measurement"),
        ("D5_selection_reporting", "Reporting"),
    ]
    
    n = len(rob_data)
    if n == 0:
        return ""
    
    domain_counts = {label: {"low": 0, "some_concerns": 0, "high": 0} for _, label in domains}
    
    for study in rob_data:
        rob_domains = study.get("rob2_domains", {})
        for key, label in domains:
            j = rob_domains.get(key, {}).get("judgement", "unknown")
            if j in domain_counts[label]:
                domain_counts[label][j] += 1
    
    labels = [label for _, label in domains]
    low_pct = [domain_counts[l]["low"] / n * 100 for l in labels]
    some_pct = [domain_counts[l]["some_concerns"] / n * 100 for l in labels]
    high_pct = [domain_counts[l]["high"] / n * 100 for l in labels]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    y = np.arange(len(labels))
    
    ax.barh(y, low_pct, color='#4DAF4A', label='Low risk', height=0.6)
    ax.barh(y, some_pct, left=low_pct, color='#FFD700', label='Some concerns', height=0.6)
    ax.barh(y, high_pct, left=np.array(low_pct) + np.array(some_pct), 
            color='#E41A1C', label='High risk', height=0.6)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Percentage of studies (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ RoB domain barplot saved: {output_path}")
    return output_path


def prisma_flow_diagram(prisma_data: dict,
                        output_path: str = None) -> str:
    """
    PRISMA 2020 flow diagram.

    Args:
        prisma_data: dict from prisma_flow.json with keys:
            identification, total_identified, after_deduplication,
            title_abstract_screened, title_abstract_excluded,
            fulltext_assessed, fulltext_excluded, fulltext_no_pdf,
            included_in_synthesis, exclusion_breakdown
        output_path: save path

    Returns path to saved PNG.
    """
    if output_path is None:
        output_path = _default_fig_path("prisma_flow.png")
    # ── Extract counts ────────────────────────────────────────────────────────
    sources      = prisma_data.get("identification", {})
    total_id     = prisma_data.get("total_identified", 0)
    after_dedup  = prisma_data.get("after_deduplication", 0)
    duplicates   = total_id - after_dedup
    ta_screened  = prisma_data.get("title_abstract_screened", after_dedup)
    ta_excluded  = prisma_data.get("title_abstract_excluded", 0)
    ft_assessed  = prisma_data.get("fulltext_assessed", 0)
    ft_excluded  = prisma_data.get("fulltext_excluded", 0)
    ft_no_pdf    = prisma_data.get("fulltext_no_pdf", 0)
    included     = prisma_data.get("included_in_synthesis", 0)

    # Group source counts (aggregate any entries that look like "records X-Y")
    _parts_re = re.compile(r'^records?\s+\d', re.IGNORECASE)
    grouped: dict[str, int] = {}
    embase_total = 0
    for src, cnt in sources.items():
        if _parts_re.match(str(src)):
            embase_total += int(cnt)
        else:
            grouped[src] = int(cnt)
    if embase_total:
        grouped["embase (combined)"] = embase_total

    # Build source label (up to 6 lines)
    src_lines = [f"{name.replace('_', ' ').title()} (n={cnt:,})"
                 for name, cnt in list(grouped.items())[:6]]
    if len(grouped) > 6:
        src_lines.append(f"…and {len(grouped) - 6} more")
    src_label = "Records identified:\n" + "\n".join(src_lines)

    # Build exclusion reasons (top 6 clean reasons)
    raw_breakdown = prisma_data.get("exclusion_breakdown", {})
    clean_reasons: list[tuple[str, int]] = []
    for reason, cnt in raw_breakdown.items():
        # Skip very long keys (these are full arbiter reasoning texts)
        if len(str(reason)) > 60:
            clean_reasons.append(("Other", int(cnt)))
        else:
            clean_reasons.append((str(reason).replace("_", " ").capitalize(), int(cnt)))
    # Sort by count descending, merge duplicate "Other" entries
    other_total = sum(c for r, c in clean_reasons if r == "Other")
    named = [(r, c) for r, c in clean_reasons if r != "Other"]
    named.sort(key=lambda x: -x[1])
    reason_lines = [f"{r} (n={c})" for r, c in named[:6]]
    if other_total:
        reason_lines.append(f"Other reasons (n={other_total})")
    ft_excl_label = f"Full-texts excluded (n={ft_excluded + ft_no_pdf})"
    if reason_lines:
        ft_excl_label += "\n" + "\n".join(reason_lines[:5])
    if ft_no_pdf:
        ft_excl_label += f"\nNo PDF available (n={ft_no_pdf})"

    # ── Layout constants ──────────────────────────────────────────────────────
    import matplotlib.patches as mpatch

    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    BOX_W  = 4.2
    BOX_H  = 1.0
    LEFT_X = 0.4
    MID_X  = 2.8 - BOX_W / 2   # left-column x for main flow
    RIGHT_X = 5.8               # right-column x for exclusion boxes
    BORDER = dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor="#2E86C1", linewidth=1.4)
    EXCL_BORDER = dict(boxstyle="round,pad=0.4", facecolor="#FDEDEC", edgecolor="#C0392B", linewidth=1.2)
    INCL_BORDER = dict(boxstyle="round,pad=0.4", facecolor="#EAFAF1", edgecolor="#27AE60", linewidth=1.4)

    def _box(x, y, w, h, text, style, fontsize=8.5):
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                wrap=True, bbox=style,
                transform=ax.transData,
                multialignment="center",
                clip_on=False)
        return (x + w / 2, y)   # bottom center

    def _arrow(x, y_top, y_bot, label=None):
        ax.annotate("", xy=(x, y_bot + 0.05), xytext=(x, y_top),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
        if label:
            ax.text(x + 0.12, (y_top + y_bot) / 2, label,
                    ha="left", va="center", fontsize=7, color="#555555")

    def _horiz_arrow(x_left, x_right, y):
        ax.annotate("", xy=(x_right, y), xytext=(x_left, y),
                    arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.0,
                                    connectionstyle="arc3,rad=0"))

    # ── Section labels ────────────────────────────────────────────────────────
    ax.text(0.15, 12.7, "Identification", fontsize=9, fontweight="bold", color="#2E86C1",
            rotation=90, va="top")
    ax.text(0.15, 10.2, "Screening", fontsize=9, fontweight="bold", color="#2E86C1",
            rotation=90, va="top")
    ax.text(0.15, 7.5, "Eligibility", fontsize=9, fontweight="bold", color="#2E86C1",
            rotation=90, va="top")
    ax.text(0.15, 5.0, "Included", fontsize=9, fontweight="bold", color="#27AE60",
            rotation=90, va="top")
    # Divider lines
    for y_line in [12.5, 10.0, 7.2, 5.3]:
        ax.axhline(y=y_line, xmin=0.03, xmax=0.97, color="#CCCCCC", linewidth=0.6, linestyle="--")

    # ── Box 1: Records identified ─────────────────────────────────────────────
    bx, by = LEFT_X + 0.8, 11.4
    ax.text(bx + BOX_W / 2, by + BOX_H / 2, src_label,
            ha="center", va="center", fontsize=8, bbox=BORDER,
            multialignment="center", clip_on=False)
    id_bot = (bx + BOX_W / 2, by)

    # Arrow down
    _arrow(id_bot[0], id_bot[1], 10.15)

    # ── Box 2: Records screened (after dedup) ─────────────────────────────────
    bx2, by2 = LEFT_X + 0.8, 9.1
    _box(bx2, by2, BOX_W, BOX_H,
         f"Records screened\n(after deduplication removed n={duplicates:,})\nn = {after_dedup:,}",
         BORDER)
    # Arrow down
    _arrow(bx2 + BOX_W / 2, by2, 7.8)
    # Horizontal arrow to exclusion
    _horiz_arrow(bx2 + BOX_W, RIGHT_X, by2 + BOX_H / 2)
    # Exclusion box (right)
    ax.text(RIGHT_X + 0.1, by2 + BOX_H / 2,
            f"Records excluded\n(T/A screening)\nn = {ta_excluded:,}",
            ha="left", va="center", fontsize=8, bbox=EXCL_BORDER,
            multialignment="center", clip_on=False)

    # ── Box 3: Full texts assessed ────────────────────────────────────────────
    bx3, by3 = LEFT_X + 0.8, 6.65
    _box(bx3, by3, BOX_W, BOX_H,
         f"Full-texts assessed for eligibility\nn = {ft_assessed:,}",
         BORDER)
    _arrow(bx3 + BOX_W / 2, by3, 5.4)
    _horiz_arrow(bx3 + BOX_W, RIGHT_X, by3 + BOX_H / 2)
    ax.text(RIGHT_X + 0.1, by3 + BOX_H / 2,
            ft_excl_label,
            ha="left", va="center", fontsize=7.5, bbox=EXCL_BORDER,
            multialignment="center", clip_on=False)

    # ── Box 4: Included ───────────────────────────────────────────────────────
    bx4, by4 = LEFT_X + 0.8, 4.3
    ax.text(bx4 + BOX_W / 2, by4 + BOX_H / 2,
            f"Studies included in synthesis\nn = {included:,}",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            bbox=INCL_BORDER, multialignment="center", clip_on=False)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title("PRISMA 2020 Flow Diagram", fontsize=13, fontweight="bold", pad=12)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"✅ PRISMA flow diagram saved: {output_path}")
    return output_path
