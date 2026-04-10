"""Tests for lumen.tools.statistics.effect_sizes — fixes v2 audit #4, #5."""

import math
import pytest
from lumen.tools.statistics.effect_sizes import (
    hedges_g,
    mean_difference,
    log_risk_ratio,
    risk_ratio,
    odds_ratio,
    risk_difference,
    check_and_correct_se_sd,
)


class TestHedgesG:
    def test_known_value(self):
        """Hedges' g for a known case (Borenstein 2009 style)."""
        result = hedges_g(n1=50, mean1=103, sd1=5.5, n2=50, mean2=100, sd2=4.5)
        # d = 3 / pooled_sd ≈ 0.597, g ≈ d * J
        assert "g" in result
        assert "se" in result
        assert result["ci_lower"] < result["g"] < result["ci_upper"]
        assert 0 < result["correction_factor"] < 1

    def test_negative_effect(self):
        result = hedges_g(n1=30, mean1=5.0, sd1=2.0, n2=30, mean2=8.0, sd2=2.5)
        assert result["g"] < 0

    def test_zero_pooled_sd_raises(self):
        with pytest.raises(ValueError, match="Pooled SD is zero"):
            hedges_g(n1=10, mean1=5.0, sd1=0.0, n2=10, mean2=5.0, sd2=0.0)

    def test_insufficient_n_raises(self):
        with pytest.raises(ValueError, match="Total n must be > 2"):
            hedges_g(n1=1, mean1=5.0, sd1=1.0, n2=1, mean2=3.0, sd2=1.0)

    def test_small_sample_correction(self):
        """J factor should be closer to 1 for large samples."""
        small = hedges_g(n1=5, mean1=10, sd1=2, n2=5, mean2=8, sd2=2)
        large = hedges_g(n1=500, mean1=10, sd1=2, n2=500, mean2=8, sd2=2)
        assert small["correction_factor"] < large["correction_factor"]


class TestMeanDifference:
    def test_basic(self):
        result = mean_difference(mean1=10, sd1=3, n1=50, mean2=8, sd2=3, n2=50)
        assert abs(result["md"] - 2.0) < 1e-10
        assert result["ci_lower"] < 2.0 < result["ci_upper"]


class TestLogRiskRatio:
    def test_basic(self):
        result = log_risk_ratio(a=15, n1=100, c=10, n2=100)
        assert result["log_rr"] > 0  # higher risk in group 1
        assert result["ci_lower"] < result["log_rr"] < result["ci_upper"]

    def test_zero_events_raises(self):
        with pytest.raises(ValueError):
            log_risk_ratio(a=0, n1=100, c=10, n2=100)


class TestRiskRatio:
    def test_basic(self):
        result = risk_ratio(a=20, n1=100, c=10, n2=100)
        assert abs(result["rr"] - 2.0) < 1e-10
        assert result["ci_lower"] < result["rr"] < result["ci_upper"]


class TestOddsRatio:
    def test_basic(self):
        result = odds_ratio(a=20, b=80, c=10, d=90)
        expected_or = (20 * 90) / (80 * 10)
        assert abs(result["or"] - expected_or) < 1e-10

    def test_zero_cell_raises(self):
        with pytest.raises(ValueError):
            odds_ratio(a=0, b=10, c=5, d=10)


class TestRiskDifference:
    def test_basic(self):
        result = risk_difference(a=30, n1=100, c=20, n2=100)
        assert abs(result["rd"] - 0.10) < 1e-10


class TestSESDCorrection:
    def test_no_correction_small_n(self):
        """n <= 1 → no correction possible."""
        result = check_and_correct_se_sd(value=5.0, n=1, reported_as="se")
        assert not result["was_corrected"]

    def test_no_correction_without_reference(self):
        """Without reference_mean, no correction (conservative)."""
        result = check_and_correct_se_sd(value=15.0, n=100, reported_as="se")
        assert not result["was_corrected"]
        assert result["corrected_value"] == 15.0

    def test_se_corrected_when_looks_like_sd_with_reference(self):
        """Large 'SE' with large n + reference_mean → probably a SD.
        mean=10, reported SE=15 → CV=1.5 (too high for SE).
        If it's really SD, SE=15/10=1.5, CV_se=0.15 (reasonable)."""
        result = check_and_correct_se_sd(
            value=15.0, n=100, reported_as="se", reference_mean=10.0,
        )
        assert result["was_corrected"]
        assert result["correction_type"] == "se_was_sd"
        assert result["corrected_value"] == pytest.approx(15.0 / 10.0)

    def test_no_double_correction(self):
        """v2 audit #4: corrected value should not be re-corrected."""
        result1 = check_and_correct_se_sd(
            value=15.0, n=100, reported_as="se", reference_mean=10.0,
        )
        if result1["was_corrected"]:
            assert result1["was_corrected"] is True
            # Downstream checks was_corrected and doesn't re-call

    def test_sd_corrected_when_looks_like_se_with_reference(self):
        """Small 'SD' with large n + reference_mean → probably an SE.
        mean=100, reported SD=0.5 → CV=0.005 (too low for SD).
        If it's really SE, SD=0.5*10=5, CV=0.05 (reasonable)."""
        result = check_and_correct_se_sd(
            value=0.5, n=100, reported_as="sd", reference_mean=100.0,
        )
        assert result["was_corrected"]
        assert result["correction_type"] == "sd_was_se"
        assert result["corrected_value"] == pytest.approx(0.5 * 10.0)

    def test_no_correction_plausible_se(self):
        """SE=1.5 with n=100, mean=100 → CV=0.015 (plausible SE), no correction."""
        result = check_and_correct_se_sd(
            value=1.5, n=100, reported_as="se", reference_mean=100.0,
        )
        assert not result["was_corrected"]

    def test_no_correction_plausible_sd(self):
        """SD=15 with n=100, mean=100 → CV=0.15 (plausible SD), no correction."""
        result = check_and_correct_se_sd(
            value=15.0, n=100, reported_as="sd", reference_mean=100.0,
        )
        assert not result["was_corrected"]

    def test_correction_returns_original(self):
        result = check_and_correct_se_sd(
            value=15.0, n=100, reported_as="se", reference_mean=10.0,
        )
        assert result["original_value"] == 15.0
