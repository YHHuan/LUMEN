"""Honest-cost accounting tests (stabilize P3.6)."""
from lumen.core.cost import CostTracker
from lumen.tools.visualization.cost_report import generate_cost_report


class TestCachedTokenAccounting:
    def test_cached_tokens_default_zero(self):
        t = CostTracker()
        t.record("phase3", "screener",
                 {"input_tokens": 500, "output_tokens": 100, "cost": 0.05})
        s = t.summary()
        assert s["grand_total_cached_tokens"] == 0
        assert s["caching_enabled"] is False
        assert s["cost_basis"] == "actual"

    def test_cached_tokens_tracked_when_present(self):
        t = CostTracker()
        t.record("phase3", "screener", {
            "input_tokens": 0, "output_tokens": 100,
            "cached_input_tokens": 400, "cost": 0.0,
        })
        s = t.summary()
        assert s["grand_total_cached_tokens"] == 400
        assert s["caching_enabled"] is True

    def test_existing_summary_keys_unchanged(self):
        """Baseline keys must survive for the existing test suite / callers."""
        t = CostTracker()
        t.record("p", "a", {"input_tokens": 10, "output_tokens": 5, "cost": 0.01})
        s = t.summary()
        assert s["grand_total_calls"] == 1
        assert s["grand_total_tokens"] == 15
        assert s["grand_total_cost"] == 0.01

    def test_estimate_remaining_labelled_as_assumption(self):
        t = CostTracker()
        t.record("phase1", "pico", {"input_tokens": 1000, "output_tokens": 500, "cost": 0.10})
        est = t.estimate_remaining("phase1", n_studies=10)
        assert est["basis"] == "extrapolated"  # honestly marked as a projection


class TestCostReportTokenKeyFix:
    """The token-key mismatch bug (same class as V4's weight-vocab bug)."""

    def test_real_costtracker_shape_reports_tokens(self):
        # Real CostTracker.summary()["by_phase"] uses input_tokens/output_tokens,
        # NOT a flat "tokens" key. Previously this reported 0 tokens.
        t = CostTracker()
        t.record("screening", "screener",
                 {"input_tokens": 300, "output_tokens": 100, "cost": 1.0})
        by_phase = t.summary()["by_phase"]
        report = generate_cost_report(by_phase)
        assert report["grand_total"]["tokens"] == 400  # not 0
        assert report["grand_total"]["cost"] == 1.0

    def test_legacy_flat_tokens_still_supported(self):
        legacy = {"screening": {"screener": {"calls": 2, "tokens": 5000, "cost": 1.5}}}
        report = generate_cost_report(legacy)
        assert report["grand_total"]["tokens"] == 5000

    def test_report_declares_actual_basis(self):
        report = generate_cost_report({})
        assert report["cost_basis"] == "actual"
