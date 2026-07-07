"""Deterministic quality bucketing tests (stabilize P0)."""
import pytest

from lumen.tools.quality.quality_bucket import (
    robins_i_overall, quality_bucket, bucket_studies, grade_rob_downgrade,
    BUCKET_LOW, BUCKET_MODERATE, BUCKET_HIGH, ROBINS_I_DOMAINS,
)


def _robins(**overrides):
    base = {d: "Low" for d in ROBINS_I_DOMAINS}
    base.update(overrides)
    return base


class TestRobinsIOverall:
    def test_all_low(self):
        assert robins_i_overall(_robins()) == "Low"

    def test_worst_domain_wins(self):
        assert robins_i_overall(_robins(confounding="Serious")) == "Serious"
        assert robins_i_overall(_robins(missing_data="Critical")) == "Critical"

    def test_three_serious_escalates_to_critical(self):
        d = _robins(confounding="Serious", missing_data="Serious",
                    measurement_of_outcome="Serious")
        assert robins_i_overall(d) == "Critical"

    def test_case_insensitive(self):
        assert robins_i_overall(_robins(confounding="serious")) == "Serious"

    def test_missing_domain_raises(self):
        d = _robins()
        del d["confounding"]
        with pytest.raises(ValueError, match="requires all 7"):
            robins_i_overall(d)

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            robins_i_overall(_robins(confounding="nonsense"))


class TestQualityBucket:
    def test_rob2_labels(self):
        assert quality_bucket("low") == BUCKET_LOW
        assert quality_bucket("some_concerns") == BUCKET_MODERATE
        assert quality_bucket("high") == BUCKET_HIGH

    def test_robins_labels(self):
        assert quality_bucket("Low") == BUCKET_LOW
        assert quality_bucket("Moderate") == BUCKET_MODERATE
        assert quality_bucket("Serious") == BUCKET_HIGH
        assert quality_bucket("Critical") == BUCKET_HIGH

    def test_unknown_is_conservative_moderate(self):
        assert quality_bucket("???") == BUCKET_MODERATE


class TestBucketStudies:
    def test_tally_and_proportion(self):
        res = bucket_studies(["low", "high", "Serious", "some_concerns"])
        assert res["k"] == 4
        assert res["buckets"][BUCKET_HIGH] == 2  # high + Serious
        assert res["proportion_high"] == 0.5
        assert res["any_high"] is True

    def test_empty(self):
        res = bucket_studies([])
        assert res["k"] == 0
        assert res["proportion_high"] == 0.0
        assert res["any_high"] is False


class TestGradeDowngrade:
    def test_majority_high_downgrades_two(self):
        assert grade_rob_downgrade(0.6)["downgrade"] == 2

    def test_quarter_high_downgrades_one(self):
        assert grade_rob_downgrade(0.25)["downgrade"] == 1

    def test_any_high_downgrades_one(self):
        # V4 rule: even a single high-RoB study triggers a downgrade
        assert grade_rob_downgrade(0.1, any_high=True)["downgrade"] == 1

    def test_none_high_no_downgrade(self):
        assert grade_rob_downgrade(0.0, any_high=False)["downgrade"] == 0
