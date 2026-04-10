"""Tests for RoB-2 — fixes v2 audit #10."""

import pytest
from lumen.tools.quality.rob2 import (
    assess_rob2,
    summarize_rob2_across_studies,
    REQUIRED_DOMAINS,
)


def _full_domains(level: str = "low") -> dict:
    """Helper: all 5 domains at the same level."""
    return {d: level for d in REQUIRED_DOMAINS}


class TestAssessRob2:
    def test_all_low(self):
        result = assess_rob2(_full_domains("low"))
        assert result["overall"] == "low"
        assert result["n_low"] == 5

    def test_all_high(self):
        result = assess_rob2(_full_domains("high"))
        assert result["overall"] == "high"
        assert result["n_high"] == 5

    def test_one_some_concerns(self):
        domains = _full_domains("low")
        domains["missing_outcome_data"] = "some_concerns"
        result = assess_rob2(domains)
        assert result["overall"] == "some_concerns"

    def test_one_high_overrides(self):
        domains = _full_domains("some_concerns")
        domains["randomization_process"] = "high"
        result = assess_rob2(domains)
        assert result["overall"] == "high"

    def test_rejects_incomplete(self):
        """v2 audit #10: missing domains must raise."""
        with pytest.raises(ValueError, match="Missing"):
            assess_rob2({"randomization_process": "low"})

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Missing"):
            assess_rob2({})

    def test_rejects_invalid_judgement(self):
        domains = _full_domains("low")
        domains["randomization_process"] = "medium"
        with pytest.raises(ValueError, match="Invalid"):
            assess_rob2(domains)

    def test_missing_domains_list(self):
        result = assess_rob2(_full_domains("low"))
        assert result["missing_domains"] == []

    def test_all_5_domains_in_output(self):
        result = assess_rob2(_full_domains("low"))
        for d in REQUIRED_DOMAINS:
            assert d in result["domains"]


class TestSummarizeRob2:
    def test_basic(self):
        assessments = [
            assess_rob2(_full_domains("low")),
            assess_rob2(_full_domains("high")),
        ]
        summary = summarize_rob2_across_studies(assessments)
        assert summary["k"] == 2
        assert summary["overall"]["low"] == pytest.approx(0.5)
        assert summary["overall"]["high"] == pytest.approx(0.5)

    def test_empty(self):
        summary = summarize_rob2_across_studies([])
        assert summary["k"] == 0

    def test_proportion_high(self):
        assessments = [assess_rob2(_full_domains("high"))] * 3
        summary = summarize_rob2_across_studies(assessments)
        assert summary["proportion_high_overall"] == pytest.approx(1.0)
