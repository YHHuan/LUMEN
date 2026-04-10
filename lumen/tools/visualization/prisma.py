"""
PRISMA 2020 flow diagram generator.

Generates a PRISMA flow diagram from pipeline state counts.
Output: matplotlib figure or text-based summary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# PRISMA 2020 box labels
PRISMA_BOXES = [
    "identification",
    "screening",
    "eligibility",
    "included",
]


def compute_prisma_counts(state: dict) -> dict:
    """
    Extract PRISMA flow counts from pipeline state.

    Returns dict with all counts needed for the flow diagram.
    """
    raw = state.get("raw_results", [])
    deduped = state.get("deduplicated_studies", [])
    prescreen = state.get("prescreen_results", [])
    screening = state.get("screening_results", [])
    fulltext = state.get("fulltext_results", [])
    included = state.get("included_studies", [])

    n_identified = len(raw)
    n_duplicates = n_identified - len(deduped)
    n_after_dedup = len(deduped)

    n_prescreen_excluded = sum(
        1 for p in prescreen if p.get("prescreen") == "excluded"
    )
    n_after_prescreen = n_after_dedup - n_prescreen_excluded

    n_screened = len(screening)
    n_screen_excluded = sum(
        1 for s in screening if s.get("final_decision") == "exclude"
    )
    n_screen_human_review = sum(
        1 for s in screening if s.get("final_decision") == "human_review"
    )
    n_sought_fulltext = n_screened - n_screen_excluded

    n_fulltext_assessed = len(fulltext)
    n_fulltext_excluded = sum(
        1 for f in fulltext if f.get("decision") == "exclude"
    )
    n_included = len(included)

    return {
        "n_identified": n_identified,
        "n_duplicates": n_duplicates,
        "n_after_dedup": n_after_dedup,
        "n_prescreen_excluded": n_prescreen_excluded,
        "n_after_prescreen": n_after_prescreen,
        "n_screened": n_screened,
        "n_screen_excluded": n_screen_excluded,
        "n_screen_human_review": n_screen_human_review,
        "n_sought_fulltext": n_sought_fulltext,
        "n_fulltext_assessed": n_fulltext_assessed,
        "n_fulltext_excluded": n_fulltext_excluded,
        "n_included": n_included,
    }


def generate_prisma_text(counts: dict) -> str:
    """Generate a text-based PRISMA flow summary."""
    lines = [
        "PRISMA 2020 Flow Diagram",
        "=" * 50,
        "",
        "IDENTIFICATION",
        f"  Records identified from databases: {counts['n_identified']}",
        f"  Duplicates removed: {counts['n_duplicates']}",
        f"  Records after deduplication: {counts['n_after_dedup']}",
        "",
        "SCREENING",
        f"  Excluded by prescreen (keywords): {counts['n_prescreen_excluded']}",
        f"  Records screened (T/A): {counts['n_screened']}",
        f"  Records excluded (T/A): {counts['n_screen_excluded']}",
        f"  Records to human review: {counts['n_screen_human_review']}",
        "",
        "ELIGIBILITY",
        f"  Reports sought for retrieval: {counts['n_sought_fulltext']}",
        f"  Reports assessed (full-text): {counts['n_fulltext_assessed']}",
        f"  Reports excluded (full-text): {counts['n_fulltext_excluded']}",
        "",
        "INCLUDED",
        f"  Studies included in review: {counts['n_included']}",
    ]
    return "\n".join(lines)


def generate_prisma_figure(counts: dict, output_path: str | Path | None = None) -> Any:
    """
    Generate a PRISMA 2020 flow diagram using matplotlib.

    Returns the matplotlib Figure object.
    If output_path is provided, saves to file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")
    fig.suptitle("PRISMA 2020 Flow Diagram", fontsize=14, fontweight="bold", y=0.97)

    def _box(x, y, w, h, text, color="#E8F4FD"):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.15", facecolor=color,
            edgecolor="#2C3E50", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                wrap=True, fontfamily="sans-serif")

    def _arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))

    def _side_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2))

    # Identification
    _box(5, 15, 5, 0.9,
         f"Records identified\n(n = {counts['n_identified']})")
    _arrow(5, 14.55, 5, 13.75)

    _box(5, 13.3, 5, 0.9,
         f"After duplicates removed\n(n = {counts['n_after_dedup']})")
    _box(8.5, 13.3, 2.5, 0.7,
         f"Duplicates\n(n = {counts['n_duplicates']})", color="#FADBD8")
    _side_arrow(7.5, 13.3, 7.25, 13.3)

    _arrow(5, 12.85, 5, 12.05)

    # Screening
    _box(5, 11.6, 5, 0.9,
         f"Records screened (T/A)\n(n = {counts['n_screened']})")
    _box(8.5, 11.6, 2.5, 0.7,
         f"Excluded (T/A)\n(n = {counts['n_screen_excluded']})", color="#FADBD8")
    _side_arrow(7.5, 11.6, 7.25, 11.6)

    if counts["n_prescreen_excluded"] > 0:
        _box(8.5, 12.5, 2.5, 0.7,
             f"Prescreen excluded\n(n = {counts['n_prescreen_excluded']})", color="#FCF3CF")
        _side_arrow(7.5, 12.5, 7.25, 12.5)

    _arrow(5, 11.15, 5, 10.35)

    # Eligibility
    _box(5, 9.9, 5, 0.9,
         f"Full-text assessed\n(n = {counts['n_fulltext_assessed']})")
    _box(8.5, 9.9, 2.5, 0.7,
         f"Excluded (full-text)\n(n = {counts['n_fulltext_excluded']})", color="#FADBD8")
    _side_arrow(7.5, 9.9, 7.25, 9.9)

    _arrow(5, 9.45, 5, 8.65)

    # Included
    _box(5, 8.2, 5, 0.9,
         f"Studies included\n(n = {counts['n_included']})", color="#D5F5E3")

    # Phase labels
    for label, y in [("IDENTIFICATION", 15.5), ("SCREENING", 12.2),
                     ("ELIGIBILITY", 10.4), ("INCLUDED", 8.7)]:
        ax.text(0.3, y, label, fontsize=7, fontweight="bold",
                rotation=90, va="center", color="#7F8C8D")

    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("prisma_saved", path=str(output_path))

    return fig
