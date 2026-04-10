"""Tests for statistician agent."""

import json
import pytest
from unittest.mock import MagicMock, patch

from lumen.agents.statistician import StatisticianAgent, I2_HIGH_THRESHOLD, WEIGHT_DOMINANCE_THRESHOLD


def _make_statistician(llm_responses: list[dict] | None = None):
    router = MagicMock()
    cost_tracker = MagicMock()
    if llm_responses:
        side_effects = []
        for r in llm_responses:
            side_effects.append((json.dumps(r), {
                "model": "test", "input_tokens": 300,
                "output_tokens": 150, "cost": 0.004, "latency_ms": 400,
            }))
        router.call.side_effect = side_effects
    return StatisticianAgent(router=router, cost_tracker=cost_tracker, config={})


# ---- Sample data ----
HARMONIZED_CONTINUOUS = [
    {
        "study_id": "S1",
        "extractions": [{
            "canonical_outcome": "systolic_BP",
            "outcome_name": "SBP",
            "arm1": {"name": "intervention", "n": 50, "mean": 125.0, "sd": 12.0},
            "arm2": {"name": "control", "n": 50, "mean": 132.0, "sd": 13.0},
        }],
    },
    {
        "study_id": "S2",
        "extractions": [{
            "canonical_outcome": "systolic_BP",
            "outcome_name": "systolic blood pressure",
            "arm1": {"name": "intervention", "n": 40, "mean": 120.0, "sd": 10.0},
            "arm2": {"name": "control", "n": 40, "mean": 130.0, "sd": 11.0},
        }],
    },
    {
        "study_id": "S3",
        "extractions": [{
            "canonical_outcome": "systolic_BP",
            "outcome_name": "SBP",
            "arm1": {"name": "intervention", "n": 60, "mean": 122.0, "sd": 11.0},
            "arm2": {"name": "control", "n": 60, "mean": 131.0, "sd": 12.0},
        }],
    },
]

HARMONIZED_BINARY = [
    {
        "study_id": "S1",
        "extractions": [{
            "canonical_outcome": "mortality",
            "outcome_name": "mortality",
            "arm1": {"name": "intervention", "n": 100, "events": 5, "total": 100},
            "arm2": {"name": "control", "n": 100, "events": 12, "total": 100},
        }],
    },
    {
        "study_id": "S2",
        "extractions": [{
            "canonical_outcome": "mortality",
            "outcome_name": "mortality",
            "arm1": {"name": "intervention", "n": 80, "events": 3, "total": 80},
            "arm2": {"name": "control", "n": 80, "events": 10, "total": 80},
        }],
    },
]

PICO = {"population": "adults with hypertension", "intervention": "exercise"}


class TestDataProfile:
    def test_continuous_profile(self):
        profile = StatisticianAgent._data_profile(HARMONIZED_CONTINUOUS)
        assert profile["n_outcomes"] == 1
        assert profile["outcomes"][0]["name"] == "systolic_BP"
        assert profile["outcomes"][0]["k"] == 3
        assert profile["outcomes"][0]["has_continuous"] is True

    def test_binary_profile(self):
        profile = StatisticianAgent._data_profile(HARMONIZED_BINARY)
        assert profile["outcomes"][0]["has_binary"] is True
        assert profile["outcomes"][0]["k"] == 2

    def test_empty_data(self):
        profile = StatisticianAgent._data_profile([])
        assert profile["n_outcomes"] == 0
        assert profile["outcomes"] == []

    def test_mixed_outcomes(self):
        combined = HARMONIZED_CONTINUOUS + HARMONIZED_BINARY
        profile = StatisticianAgent._data_profile(combined)
        assert profile["n_outcomes"] == 2
        # S1/S2 appear in both continuous and binary data → 3 unique continuous + 2 binary study_ids
        assert profile["total_studies"] == len(
            {s for o in profile["outcomes"] for s in o["study_ids"]}
        )


class TestCollectOutcomeData:
    def test_continuous_smd(self):
        effects, ses, labels = StatisticianAgent._collect_outcome_data(
            HARMONIZED_CONTINUOUS, "systolic_BP", "SMD",
        )
        assert len(effects) == 3
        assert len(ses) == 3
        assert all(e < 0 for e in effects)  # intervention < control → negative

    def test_continuous_md(self):
        effects, ses, labels = StatisticianAgent._collect_outcome_data(
            HARMONIZED_CONTINUOUS, "systolic_BP", "MD",
        )
        assert len(effects) == 3
        assert effects[0] == pytest.approx(-7.0, abs=0.1)

    def test_binary_log_or(self):
        effects, ses, labels = StatisticianAgent._collect_outcome_data(
            HARMONIZED_BINARY, "mortality", "logOR",
        )
        assert len(effects) == 2
        assert all(e < 0 for e in effects)  # fewer events in intervention

    def test_missing_outcome(self):
        effects, ses, labels = StatisticianAgent._collect_outcome_data(
            HARMONIZED_CONTINUOUS, "nonexistent", "SMD",
        )
        assert len(effects) == 0


class TestAnomalyDetection:
    def test_high_i2_flagged(self):
        results = {
            "outcome_x": {
                "k": 5,
                "meta": {"i2": 95.0, "weights": [0.2] * 5,
                         "ci_lower": -1.0, "ci_upper": 0.5,
                         "prediction_interval": None},
            },
        }
        flags = StatisticianAgent._detect_anomalies(results, {}, {})
        types = [f["type"] for f in flags]
        assert "high_heterogeneity" in types

    def test_weight_dominance_flagged(self):
        results = {
            "outcome_x": {
                "k": 5,
                "meta": {"i2": 30.0, "weights": [0.5, 0.2, 0.1, 0.1, 0.1],
                         "ci_lower": 0.1, "ci_upper": 0.8,
                         "prediction_interval": None},
            },
        }
        flags = StatisticianAgent._detect_anomalies(results, {}, {})
        types = [f["type"] for f in flags]
        assert "weight_dominance" in types

    def test_underpowered_flagged(self):
        results = {
            "outcome_x": {
                "k": 2,
                "meta": {"i2": 10.0, "weights": [0.5, 0.5],
                         "ci_lower": -0.5, "ci_upper": 0.5,
                         "prediction_interval": None},
            },
        }
        flags = StatisticianAgent._detect_anomalies(results, {}, {})
        types = [f["type"] for f in flags]
        assert "underpowered" in types

    def test_prediction_interval_warning(self):
        results = {
            "outcome_x": {
                "k": 5,
                "meta": {"i2": 60.0, "weights": [0.2] * 5,
                         "ci_lower": 0.1, "ci_upper": 0.8,
                         "prediction_interval": (-0.5, 1.2)},
            },
        }
        flags = StatisticianAgent._detect_anomalies(results, {}, {})
        types = [f["type"] for f in flags]
        assert "prediction_interval_warning" in types

    def test_trim_fill_direction_flip(self):
        results = {
            "outcome_x": {
                "k": 10,
                "meta": {"i2": 30.0, "weights": [0.1] * 10,
                         "ci_lower": 0.1, "ci_upper": 0.5,
                         "prediction_interval": None},
                "trim_and_fill": {"direction_flipped": True, "warning": "flipped"},
            },
        }
        flags = StatisticianAgent._detect_anomalies(results, {}, {})
        types = [f["type"] for f in flags]
        assert "trim_fill_direction_flip" in types

    def test_no_anomalies_for_clean_data(self):
        results = {
            "outcome_x": {
                "k": 5,
                "meta": {"i2": 30.0, "weights": [0.2] * 5,
                         "ci_lower": -0.8, "ci_upper": -0.1,
                         "prediction_interval": (-1.2, -0.05)},
            },
        }
        flags = StatisticianAgent._detect_anomalies(results, {}, {})
        assert len(flags) == 0


class TestDefaultPlan:
    def test_default_plan_continuous(self):
        profile = StatisticianAgent._data_profile(HARMONIZED_CONTINUOUS)
        plan = StatisticianAgent._default_plan(profile)
        assert len(plan["outcomes"]) == 1
        assert plan["outcomes"][0]["effect_size"] == "SMD"
        assert plan["outcomes"][0]["apply_hksj"] is True

    def test_default_plan_binary(self):
        profile = StatisticianAgent._data_profile(HARMONIZED_BINARY)
        plan = StatisticianAgent._default_plan(profile)
        assert plan["outcomes"][0]["effect_size"] == "logOR"


class TestFullAnalyze:
    def test_analyze_with_mocked_llm(self):
        """Full pipeline with mocked LLM for plan + interpretation."""
        plan_response = {
            "outcomes": [{
                "name": "systolic_BP",
                "measure": "continuous",
                "effect_size": "SMD",
                "model": "random_effects",
                "method": "REML",
                "apply_hksj": True,
                "k": 3,
                "subgroup_variables": [],
                "sensitivity_analyses": ["leave_one_out"],
                "run_egger": False,
                "run_trim_and_fill": False,
                "notes": "",
            }],
            "global_notes": "Standard analysis",
        }
        interpret_response = {
            "interpretations": [{
                "outcome": "systolic_BP",
                "main_finding": "Exercise reduces SBP",
                "effect_magnitude": "moderate",
                "statistical_significance": True,
                "clinical_significance": "Clinically meaningful",
                "heterogeneity_interpretation": "Low heterogeneity",
                "certainty_of_evidence": "moderate",
                "caveats": [],
                "subgroup_findings": [],
                "sensitivity_findings": "Robust",
            }],
            "overall_summary": "Exercise reduces blood pressure",
            "limitations": [],
        }
        agent = _make_statistician([plan_response, interpret_response])
        result = agent.analyze(HARMONIZED_CONTINUOUS, PICO)

        assert "statistics_results" in result
        assert "systolic_BP" in result["statistics_results"]
        meta = result["statistics_results"]["systolic_BP"]["meta"]
        assert meta["pooled_effect"] < 0  # intervention better
        assert meta["k"] == 3
        assert "anomaly_flags" in result

    def test_analyze_empty_data(self):
        agent = _make_statistician()
        result = agent.analyze([], PICO)
        assert result["statistics_results"] == {}
        assert result["anomaly_flags"] == []
