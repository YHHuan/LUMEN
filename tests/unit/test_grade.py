"""Tests for GRADE — fixes v2 audit #3, #11."""

import pytest
from lumen.tools.quality.grade import assess_grade


def _domain(level: str = "no_concern", reason: str = "") -> dict:
    return {"level": level, "reason": reason}


class TestAssessGrade:
    def test_all_no_concern_rct(self):
        result = assess_grade(
            rob_data=_domain(),
            inconsistency_data=_domain(),
            indirectness_data=_domain(),
            imprecision_data=_domain(),
            publication_bias_data=_domain(),
            study_design="rct",
        )
        assert result["grade"] == "high"
        assert result["certainty"] == 4
        assert result["total_downgrade"] == 0

    def test_rejects_missing_indirectness(self):
        """v2 audit #3: indirectness=None must not default to no_concern."""
        result = assess_grade(
            rob_data=_domain(),
            inconsistency_data=_domain(),
            indirectness_data=None,
            imprecision_data=_domain(),
            publication_bias_data=_domain(),
        )
        assert result.get("error") is not None
        assert result.get("grade") is None
        assert "indirectness" in result["error"]

    def test_rejects_other_missing_domain(self):
        result = assess_grade(
            rob_data=None,
            inconsistency_data=_domain(),
            indirectness_data=_domain(),
            imprecision_data=_domain(),
            publication_bias_data=_domain(),
        )
        assert result.get("error") is not None

    def test_serious_downgrade(self):
        result = assess_grade(
            rob_data=_domain("serious", "50% high RoB"),
            inconsistency_data=_domain(),
            indirectness_data=_domain(),
            imprecision_data=_domain(),
            publication_bias_data=_domain(),
            study_design="rct",
        )
        assert result["grade"] == "moderate"
        assert result["certainty"] == 3
        assert result["total_downgrade"] == 1

    def test_very_serious_downgrade(self):
        result = assess_grade(
            rob_data=_domain("very_serious"),
            inconsistency_data=_domain(),
            indirectness_data=_domain(),
            imprecision_data=_domain(),
            publication_bias_data=_domain(),
            study_design="rct",
        )
        assert result["grade"] == "low"
        assert result["certainty"] == 2

    def test_multiple_downgrades(self):
        result = assess_grade(
            rob_data=_domain("serious"),
            inconsistency_data=_domain("serious"),
            indirectness_data=_domain("serious"),
            imprecision_data=_domain(),
            publication_bias_data=_domain(),
            study_design="rct",
        )
        assert result["grade"] == "very_low"
        assert result["certainty"] == 1
        assert result["total_downgrade"] == 3

    def test_floor_at_very_low(self):
        """Cannot go below very_low (certainty=1)."""
        result = assess_grade(
            rob_data=_domain("very_serious"),
            inconsistency_data=_domain("very_serious"),
            indirectness_data=_domain("very_serious"),
            imprecision_data=_domain("very_serious"),
            publication_bias_data=_domain("very_serious"),
            study_design="rct",
        )
        assert result["grade"] == "very_low"
        assert result["certainty"] == 1

    def test_observational_starts_low(self):
        result = assess_grade(
            rob_data=_domain(),
            inconsistency_data=_domain(),
            indirectness_data=_domain(),
            imprecision_data=_domain(),
            publication_bias_data=_domain(),
            study_design="observational",
        )
        assert result["grade"] == "low"
        assert result["certainty"] == 2

    def test_threshold_sources_documented(self):
        """v2 audit #11: thresholds must cite literature sources."""
        result = assess_grade(
            rob_data=_domain(),
            inconsistency_data=_domain(),
            indirectness_data=_domain(),
            imprecision_data=_domain(),
            publication_bias_data=_domain(),
        )
        sources = result["threshold_sources"]
        assert "Cochrane" in sources["imprecision"]
        assert "Cochrane" in sources["inconsistency"]
        assert "Schünemann" in sources["risk_of_bias"]

    def test_downgrades_detail(self):
        result = assess_grade(
            rob_data=_domain("serious", "Many high RoB studies"),
            inconsistency_data=_domain(),
            indirectness_data=_domain(),
            imprecision_data=_domain(),
            publication_bias_data=_domain(),
        )
        assert "risk_of_bias" in result["downgrades"]
        assert result["downgrades"]["risk_of_bias"]["reason"] == "Many high RoB studies"
