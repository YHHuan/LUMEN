"""Integration tests for statistics pipeline."""

import json
import pytest
from unittest.mock import MagicMock

from lumen.agents.statistician import StatisticianAgent
from lumen.agents.quality_node import QualityAssessorAgent


def _mock_router(*responses):
    router = MagicMock()
    side_effects = []
    for r in responses:
        side_effects.append((json.dumps(r), {
            "model": "test", "input_tokens": 300,
            "output_tokens": 150, "cost": 0.004, "latency_ms": 400,
        }))
    router.call.side_effect = side_effects
    return router


# 5 studies for robust testing
HARMONIZED_5STUDY = [
    {
        "study_id": f"S{i}",
        "extractions": [{
            "canonical_outcome": "systolic_BP",
            "outcome_name": "SBP",
            "arm1": {"name": "intervention", "n": 50 + i * 5,
                     "mean": 120.0 + i, "sd": 10.0 + i * 0.5},
            "arm2": {"name": "control", "n": 50 + i * 5,
                     "mean": 130.0 + i, "sd": 11.0 + i * 0.5},
        }],
    }
    for i in range(5)
]

PICO = {"population": "adults with hypertension", "intervention": "exercise"}


class TestAnomalyDetectionHighI2:
    def test_high_i2_triggers_leave_one_out(self):
        """I² > 90% triggers leave-one-out analysis."""
        # Create data with very different effect sizes to produce high I²
        heterogeneous_data = [
            {
                "study_id": "S1",
                "extractions": [{
                    "canonical_outcome": "outcome_x",
                    "arm1": {"n": 50, "mean": 100.0, "sd": 10.0},
                    "arm2": {"n": 50, "mean": 105.0, "sd": 10.0},
                }],
            },
            {
                "study_id": "S2",
                "extractions": [{
                    "canonical_outcome": "outcome_x",
                    "arm1": {"n": 50, "mean": 100.0, "sd": 10.0},
                    "arm2": {"n": 50, "mean": 150.0, "sd": 10.0},
                }],
            },
            {
                "study_id": "S3",
                "extractions": [{
                    "canonical_outcome": "outcome_x",
                    "arm1": {"n": 50, "mean": 100.0, "sd": 10.0},
                    "arm2": {"n": 50, "mean": 102.0, "sd": 10.0},
                }],
            },
        ]
        plan_response = {
            "outcomes": [{
                "name": "outcome_x",
                "measure": "continuous",
                "effect_size": "SMD",
                "method": "REML",
                "apply_hksj": True,
                "k": 3,
                "subgroup_variables": [],
                "sensitivity_analyses": ["leave_one_out"],
                "run_egger": False,
                "run_trim_and_fill": False,
                "notes": "",
            }],
        }
        interpret_response = {
            "interpretations": [],
            "overall_summary": "High heterogeneity detected",
            "limitations": ["Very high I²"],
        }
        router = _mock_router(plan_response, interpret_response)
        agent = StatisticianAgent(
            router=router, cost_tracker=MagicMock(), config={},
        )
        result = agent.analyze(heterogeneous_data, PICO)

        stats = result["statistics_results"]["outcome_x"]
        assert stats["meta"]["i2"] > 90
        assert "leave_one_out" in stats
        assert any(f["type"] == "high_heterogeneity" for f in result["anomaly_flags"])


class TestStatisticianUsesSprint2Functions:
    def test_uses_sprint2_meta_analysis(self):
        """Verify statistician calls tools/statistics/*, not its own math."""
        plan_response = {
            "outcomes": [{
                "name": "systolic_BP",
                "measure": "continuous",
                "effect_size": "MD",
                "method": "REML",
                "apply_hksj": True,
                "k": 5,
                "subgroup_variables": [],
                "sensitivity_analyses": ["leave_one_out"],
                "run_egger": False,
                "run_trim_and_fill": False,
                "notes": "",
            }],
        }
        interpret_response = {
            "interpretations": [],
            "overall_summary": "OK",
            "limitations": [],
        }
        router = _mock_router(plan_response, interpret_response)
        agent = StatisticianAgent(
            router=router, cost_tracker=MagicMock(), config={},
        )
        result = agent.analyze(HARMONIZED_5STUDY, PICO)

        meta = result["statistics_results"]["systolic_BP"]["meta"]
        # These fields come from Sprint 2's random_effects_meta
        assert "pooled_effect" in meta
        assert "tau2" in meta
        assert "tau2_method_used" in meta
        assert "i2" in meta
        assert "prediction_interval" in meta
        assert "method_log" in meta
        # MD should be negative (intervention < control)
        assert meta["pooled_effect"] < 0


class TestQualityAssessorIntegration:
    def test_rob2_grade_integration(self):
        """RoB-2 assessment feeds into GRADE correctly."""
        rob2_response = {
            "assessments": [
                {
                    "study_id": f"S{i}",
                    "domains": {
                        "randomization_process": "low",
                        "deviations_from_intervention": "low",
                        "missing_outcome_data": "low" if i < 3 else "some_concerns",
                        "measurement_of_outcome": "low",
                        "selection_of_reported_result": "low",
                    },
                    "reasoning": {d: "OK" for d in [
                        "randomization_process", "deviations_from_intervention",
                        "missing_outcome_data", "measurement_of_outcome",
                        "selection_of_reported_result",
                    ]},
                }
                for i in range(5)
            ],
        }
        router = _mock_router(rob2_response)
        agent = QualityAssessorAgent(
            router=router, cost_tracker=MagicMock(), config={},
        )

        stats = {
            "systolic_BP": {
                "k": 5,
                "meta": {
                    "i2": 35.0,
                    "ci_lower": -0.84,
                    "ci_upper": -0.26,
                },
            },
        }
        result = agent.assess(
            extractions=[{"study_id": f"S{i}", "skeleton": {"design": "RCT"},
                          "extractions": [{"outcome_name": "SBP"}]}
                         for i in range(5)],
            statistics_results=stats,
            pico=PICO,
        )

        assert len(result["rob2"]) == 5
        assert "grade" in result
        assert "systolic_BP" in result["grade"]
        grade = result["grade"]["systolic_BP"]
        assert grade["grade"] is not None
        assert grade["certainty"] is not None
