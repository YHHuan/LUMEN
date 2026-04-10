"""Integration tests for extraction + harmonization pipeline."""

import json
import pytest
from unittest.mock import MagicMock, patch

from lumen.agents.extractor import ExtractorAgent, EVIDENCE_SPAN_THRESHOLD
from lumen.agents.harmonizer import HarmonizerAgent


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


SKELETON = {
    "design": "RCT",
    "arms": [{"name": "intervention"}, {"name": "control"}],
    "n_per_arm": {"intervention": 40, "control": 40},
    "primary_outcomes": ["weight_loss"],
    "secondary_outcomes": [],
    "follow_up_duration": "6 months",
}

EXTRACTIONS = [{
    "outcome_name": "weight_loss",
    "measure": "mean",
    "timepoint": "6 months",
    "arm1": {"name": "intervention", "n": 40, "mean": -5.2, "sd": 3.1},
    "arm2": {"name": "control", "n": 40, "mean": -1.1, "sd": 2.8},
}]

CROSSCHECK_OK = {"checks_passed": True, "issues": []}

CROSSCHECK_ISSUES = {
    "checks_passed": False,
    "issues": [
        {"field": "arm1.events", "problem": "events (60) > total (40)",
         "severity": "critical", "suggestion": "Check Table 3"},
    ],
}


class TestExtractionPipeline:
    def test_4round_produces_complete_output(self):
        """All 4 rounds complete and output has required fields."""
        spans = [
            {"outcome_name": "weight_loss", "field": "arm1.mean", "value": -5.2,
             "pdf_page": 4, "pdf_text_span": "mean weight loss -5.2 kg",
             "match_confidence": 0.92},
        ]
        router = _mock_router(SKELETON, EXTRACTIONS, CROSSCHECK_OK, spans)
        agent = ExtractorAgent(router=router, cost_tracker=MagicMock(), config={})

        result = agent.extract(
            {"study_id": "INT001"}, "Full PDF text...",
            {"population": "obese adults", "intervention": "diet"},
        )
        assert result["rounds_completed"] == 4
        assert "skeleton" in result
        assert "extractions" in result
        assert "crosscheck" in result
        assert "evidence_spans" in result
        assert result["crosscheck"]["checks_passed"]

    def test_round3_catches_impossible_values(self):
        """events > total should be flagged in round 3."""
        spans = [{"outcome_name": "x", "field": "f", "value": 1,
                  "pdf_page": 1, "pdf_text_span": "x", "match_confidence": 0.9}]
        router = _mock_router(SKELETON, EXTRACTIONS, CROSSCHECK_ISSUES, spans)
        agent = ExtractorAgent(router=router, cost_tracker=MagicMock(), config={})

        result = agent.extract({"study_id": "INT002"}, "PDF...", {})
        assert not result["crosscheck"]["checks_passed"]
        assert any(i["severity"] == "critical" for i in result["crosscheck"]["issues"])

    def test_evidence_span_low_confidence_flagged(self):
        """v2 audit #8: match_score < 0.7 should be in low_confidence_spans."""
        spans = [
            {"outcome_name": "weight_loss", "field": "arm1.mean", "value": -5.2,
             "pdf_page": 4, "pdf_text_span": "around 5 kg",
             "match_confidence": 0.45},
            {"outcome_name": "weight_loss", "field": "arm2.mean", "value": -1.1,
             "pdf_page": 4, "pdf_text_span": "-1.1 kg",
             "match_confidence": 0.95},
        ]
        router = _mock_router(SKELETON, EXTRACTIONS, CROSSCHECK_OK, spans)
        agent = ExtractorAgent(router=router, cost_tracker=MagicMock(), config={})

        result = agent.extract({"study_id": "INT003"}, "PDF...", {})
        assert len(result["low_confidence_spans"]) == 1
        assert result["low_confidence_spans"][0]["match_confidence"] < EVIDENCE_SPAN_THRESHOLD


class TestHarmonizationPipeline:
    def test_similar_outcomes_grouped(self):
        """'BMI change' and 'change in BMI' → same cluster."""
        extractions = [
            {"study_id": "S1", "extractions": [
                {"outcome_name": "BMI change", "measure": "mean"},
            ]},
            {"study_id": "S2", "extractions": [
                {"outcome_name": "change in BMI", "measure": "mean"},
            ]},
        ]
        llm_refined = {
            "clusters": {"BMI_change": ["BMI change", "change in BMI"]},
            "unmapped": [],
            "reasoning": "Same construct",
        }
        agent = HarmonizerAgent(
            router=MagicMock(), cost_tracker=MagicMock(), config={},
        )
        agent.router.call.return_value = (json.dumps(llm_refined), {
            "model": "test", "input_tokens": 200,
            "output_tokens": 100, "cost": 0.003, "latency_ms": 300,
        })

        with patch.object(agent, '_cluster_by_embedding', return_value={
            "BMI change": ["BMI change", "change in BMI"],
        }):
            result = agent.harmonize(extractions, {"outcome": "BMI"})

        assert "BMI_change" in result["outcome_clusters"]
        # Both studies should have canonical_outcome set
        for ext in result["harmonized_data"]:
            for item in ext["extractions"]:
                assert item["canonical_outcome"] == "BMI_change"

    def test_extraction_to_harmonization_flow(self):
        """Extraction output feeds into harmonization correctly."""
        # Simulate extraction results from 2 studies
        extraction_results = [
            {
                "study_id": "S1",
                "extractions": [
                    {"outcome_name": "SBP", "measure": "mean",
                     "arm1": {"mean": 120}, "arm2": {"mean": 130}},
                ],
            },
            {
                "study_id": "S2",
                "extractions": [
                    {"outcome_name": "systolic blood pressure", "measure": "mean",
                     "arm1": {"mean": 118}, "arm2": {"mean": 128}},
                ],
            },
        ]
        llm_refined = {
            "clusters": {"systolic_BP": ["SBP", "systolic blood pressure"]},
            "unmapped": [],
        }
        agent = HarmonizerAgent(
            router=MagicMock(), cost_tracker=MagicMock(), config={},
        )
        agent.router.call.return_value = (json.dumps(llm_refined), {
            "model": "test", "input_tokens": 200,
            "output_tokens": 100, "cost": 0.003, "latency_ms": 300,
        })

        with patch.object(agent, '_cluster_by_embedding', return_value={
            "SBP": ["SBP", "systolic blood pressure"],
        }):
            result = agent.harmonize(extraction_results, {"outcome": "BP"})

        assert "systolic_BP" in result["outcome_clusters"]
        # All extractions should have canonical_outcome
        for ext in result["harmonized_data"]:
            for item in ext["extractions"]:
                assert "canonical_outcome" in item
