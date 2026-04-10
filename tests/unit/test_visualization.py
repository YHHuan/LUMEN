"""Tests for visualization tools: PRISMA, forest/funnel plots, cost report."""

import pytest
import json
import tempfile
from pathlib import Path

from lumen.tools.visualization.prisma import (
    compute_prisma_counts, generate_prisma_text, generate_prisma_figure,
)
from lumen.tools.visualization.plots import forest_plot, funnel_plot
from lumen.tools.visualization.cost_report import (
    generate_cost_report, format_cost_table, generate_cost_figure,
)


# ──── PRISMA ────

SAMPLE_STATE = {
    "raw_results": [{"id": i} for i in range(500)],
    "deduplicated_studies": [{"id": i} for i in range(420)],
    "prescreen_results": [
        {"prescreen": "excluded"} for _ in range(200)
    ] + [
        {"prescreen": "passed"} for _ in range(220)
    ],
    "screening_results": [
        {"final_decision": "include"} for _ in range(50)
    ] + [
        {"final_decision": "exclude"} for _ in range(170)
    ],
    "fulltext_results": [
        {"decision": "include"} for _ in range(30)
    ] + [
        {"decision": "exclude"} for _ in range(20)
    ],
    "included_studies": [{"id": i} for i in range(30)],
}


class TestPrismaCounts:
    def test_basic_counts(self):
        counts = compute_prisma_counts(SAMPLE_STATE)
        assert counts["n_identified"] == 500
        assert counts["n_duplicates"] == 80
        assert counts["n_after_dedup"] == 420
        assert counts["n_prescreen_excluded"] == 200
        assert counts["n_included"] == 30

    def test_empty_state(self):
        counts = compute_prisma_counts({})
        assert counts["n_identified"] == 0
        assert counts["n_included"] == 0

    def test_text_output(self):
        counts = compute_prisma_counts(SAMPLE_STATE)
        text = generate_prisma_text(counts)
        assert "PRISMA 2020" in text
        assert "500" in text
        assert "30" in text

    def test_figure_generation(self):
        counts = compute_prisma_counts(SAMPLE_STATE)
        fig = generate_prisma_figure(counts)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_figure_save(self):
        counts = compute_prisma_counts(SAMPLE_STATE)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        fig = generate_prisma_figure(counts, output_path=path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
        import matplotlib.pyplot as plt
        plt.close(fig)
        Path(path).unlink()


# ──── Forest & Funnel Plots ────

EFFECTS = [-0.5, -0.3, -0.8, -0.2, -0.6]
CI_LO = [-0.9, -0.7, -1.3, -0.6, -1.0]
CI_HI = [-0.1, 0.1, -0.3, 0.2, -0.2]
SES = [0.2, 0.2, 0.25, 0.2, 0.2]
LABELS = ["Study A", "Study B", "Study C", "Study D", "Study E"]


class TestForestPlot:
    def test_basic_forest(self):
        fig = forest_plot(EFFECTS, CI_LO, CI_HI, LABELS)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_forest_with_pooled(self):
        fig = forest_plot(EFFECTS, CI_LO, CI_HI, LABELS,
                          pooled_effect=-0.48, pooled_ci=(-0.7, -0.26))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_forest_save(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        fig = forest_plot(EFFECTS, CI_LO, CI_HI, LABELS, output_path=path)
        assert Path(path).exists()
        import matplotlib.pyplot as plt
        plt.close(fig)
        Path(path).unlink()


class TestFunnelPlot:
    def test_basic_funnel(self):
        fig = funnel_plot(EFFECTS, SES)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_funnel_with_pooled(self):
        fig = funnel_plot(EFFECTS, SES, pooled_effect=-0.48)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_funnel_save(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        fig = funnel_plot(EFFECTS, SES, pooled_effect=-0.48, output_path=path)
        assert Path(path).exists()
        import matplotlib.pyplot as plt
        plt.close(fig)
        Path(path).unlink()


# ──── Cost Report ────

SAMPLE_COST_SUMMARY = {
    "screening": {
        "screener": {"calls": 200, "tokens": 300000, "cost": 1.50},
        "arbiter": {"calls": 50, "tokens": 100000, "cost": 2.00},
    },
    "extraction": {
        "extractor": {"calls": 80, "tokens": 640000, "cost": 3.20},
    },
    "writing": {
        "writer": {"calls": 11, "tokens": 88000, "cost": 0.88},
    },
}


class TestCostReport:
    def test_basic_report(self):
        report = generate_cost_report(SAMPLE_COST_SUMMARY)
        assert report["grand_total"]["cost"] == pytest.approx(7.58, abs=0.01)
        assert report["grand_total"]["calls"] == 341

    def test_per_study_cost(self):
        report = generate_cost_report(SAMPLE_COST_SUMMARY, n_studies=20)
        assert report["per_study"]["cost_per_study"] == pytest.approx(0.379, abs=0.01)

    def test_empty_summary(self):
        report = generate_cost_report({})
        assert report["grand_total"]["cost"] == 0.0

    def test_text_format(self):
        report = generate_cost_report(SAMPLE_COST_SUMMARY, n_studies=20)
        text = format_cost_table(report)
        assert "LUMEN v3" in text
        assert "GRAND TOTAL" in text
        assert "Per-study" in text

    def test_figure_generation(self):
        report = generate_cost_report(SAMPLE_COST_SUMMARY)
        fig = generate_cost_figure(report)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_figure_save(self):
        report = generate_cost_report(SAMPLE_COST_SUMMARY)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        fig = generate_cost_figure(report, output_path=path)
        assert Path(path).exists()
        import matplotlib.pyplot as plt
        plt.close(fig)
        Path(path).unlink()
