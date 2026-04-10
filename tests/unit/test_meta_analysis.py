"""Tests for meta-analysis — fixes v2 audit #6, #24, #25."""

import pytest
import numpy as np
from lumen.tools.statistics.meta_analysis import (
    random_effects_meta,
    subgroup_meta,
    leave_one_out,
    cumulative_meta,
)


# Reusable test data (5 studies, moderate heterogeneity)
EFFECTS = [0.5, 0.3, 0.8, 0.1, 0.6]
SES = [0.1, 0.15, 0.2, 0.12, 0.18]


class TestRandomEffectsMeta:
    def test_basic_output_fields(self):
        result = random_effects_meta(EFFECTS, SES)
        required = [
            "pooled_effect", "pooled_se", "ci_lower", "ci_upper",
            "tau2", "tau2_method_used", "i2", "q", "q_p",
            "prediction_interval", "weights", "method_log", "k",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_reml_convergence(self):
        result = random_effects_meta(EFFECTS, SES, method="REML")
        assert result["tau2_method_used"] in ("REML", "DL_fallback")
        assert len(result["method_log"]) > 0

    def test_reml_fallback_tracking(self):
        """v2 audit #6: fallback must be recorded in method_log."""
        result = random_effects_meta([0.5], [0.1])
        assert "method_log" in result
        assert len(result["method_log"]) > 0

    def test_dl_method(self):
        result = random_effects_meta(EFFECTS, SES, method="DL")
        assert result["tau2_method_used"] == "DL"

    def test_hksj_wider_than_dl(self):
        """v2 audit #25: HKSJ CI usually wider than DL CI."""
        result_hksj = random_effects_meta(EFFECTS, SES, method="REML", apply_hksj=True)
        result_no_hksj = random_effects_meta(EFFECTS, SES, method="DL", apply_hksj=False)
        ci_hksj = result_hksj["ci_upper"] - result_hksj["ci_lower"]
        ci_no = result_no_hksj["ci_upper"] - result_no_hksj["ci_lower"]
        # HKSJ is generally wider (allow 10% tolerance for edge cases)
        assert ci_hksj >= ci_no * 0.9

    def test_prediction_interval_wider_than_ci(self):
        """v2 audit #24: PI always >= CI width."""
        result = random_effects_meta(EFFECTS, SES)
        ci_width = result["ci_upper"] - result["ci_lower"]
        pi = result["prediction_interval"]
        assert pi is not None
        pi_width = pi[1] - pi[0]
        assert pi_width >= ci_width

    def test_prediction_interval_requires_k3(self):
        """PI needs k >= 3."""
        result = random_effects_meta([0.5, 0.3], [0.1, 0.15])
        assert result["prediction_interval"] is None

    def test_single_study(self):
        result = random_effects_meta([0.5], [0.1])
        assert result["k"] == 1
        assert result["pooled_effect"] == pytest.approx(0.5)

    def test_weights_sum_to_one(self):
        result = random_effects_meta(EFFECTS, SES)
        assert sum(result["weights"]) == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No studies"):
            random_effects_meta([], [])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            random_effects_meta([0.5, 0.3], [0.1])

    def test_zero_se_raises(self):
        with pytest.raises(ValueError, match="standard errors must be > 0"):
            random_effects_meta([0.5, 0.3], [0.1, 0.0])

    def test_i2_range(self):
        result = random_effects_meta(EFFECTS, SES)
        assert 0 <= result["i2"] <= 100

    def test_pooled_within_ci(self):
        result = random_effects_meta(EFFECTS, SES)
        assert result["ci_lower"] <= result["pooled_effect"] <= result["ci_upper"]


class TestSubgroupMeta:
    def test_basic(self):
        groups = ["A", "A", "B", "B", "B"]
        result = subgroup_meta(EFFECTS, SES, groups)
        assert result["n_subgroups"] == 2
        assert "A" in result["subgroups"]
        assert "B" in result["subgroups"]
        assert "q_between" in result

    def test_single_group(self):
        groups = ["A"] * 5
        result = subgroup_meta(EFFECTS, SES, groups)
        assert result["n_subgroups"] == 1
        assert result["q_between"] == 0.0


class TestLeaveOneOut:
    def test_basic(self):
        results = leave_one_out(EFFECTS, SES)
        assert len(results) == 5
        for r in results:
            assert "omitted" in r
            assert "pooled_effect" in r

    def test_labels(self):
        labels = ["Study A", "Study B", "Study C", "Study D", "Study E"]
        results = leave_one_out(EFFECTS, SES, labels=labels)
        assert results[0]["omitted"] == "Study A"

    def test_insufficient_studies(self):
        with pytest.raises(ValueError, match="k >= 2"):
            leave_one_out([0.5], [0.1])


class TestCumulativeMeta:
    def test_basic(self):
        results = cumulative_meta(EFFECTS, SES)
        assert len(results) == 5
        assert results[0]["k"] == 1
        assert results[-1]["k"] == 5

    def test_custom_order(self):
        results = cumulative_meta(EFFECTS, SES, sort_by=[4, 3, 2, 1, 0])
        assert results[0]["added"] == "study_4"
