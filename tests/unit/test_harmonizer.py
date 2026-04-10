"""Tests for outcome harmonization agent."""

import json
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from lumen.agents.harmonizer import HarmonizerAgent


def _make_harmonizer(llm_response: dict | None = None):
    router = MagicMock()
    cost_tracker = MagicMock()
    if llm_response is not None:
        router.call.return_value = (json.dumps(llm_response), {
            "model": "test", "input_tokens": 200,
            "output_tokens": 100, "cost": 0.003, "latency_ms": 300,
        })
    return HarmonizerAgent(router=router, cost_tracker=cost_tracker, config={})


SAMPLE_EXTRACTIONS = [
    {"study_id": "S1", "extractions": [
        {"outcome_name": "BMI change", "measure": "mean"},
        {"outcome_name": "systolic blood pressure", "measure": "mean"},
    ]},
    {"study_id": "S2", "extractions": [
        {"outcome_name": "change in BMI", "measure": "mean"},
        {"outcome_name": "SBP", "measure": "mean"},
    ]},
    {"study_id": "S3", "extractions": [
        {"outcome_name": "body mass index change", "measure": "mean"},
        {"outcome_name": "diastolic blood pressure", "measure": "mean"},
    ]},
]

PICO = {"outcome": "blood pressure, BMI"}


class TestCollectOutcomeNames:
    def test_collects_from_nested(self):
        names = HarmonizerAgent._collect_outcome_names(SAMPLE_EXTRACTIONS)
        assert len(names) == 6
        assert "BMI change" in names

    def test_collects_from_flat(self):
        flat = [
            {"outcome_name": "BP", "measure": "mean"},
            {"outcome_name": "BMI", "measure": "mean"},
        ]
        names = HarmonizerAgent._collect_outcome_names(flat)
        assert len(names) == 2

    def test_empty(self):
        names = HarmonizerAgent._collect_outcome_names([])
        assert names == []


class TestFallbackClustering:
    def test_exact_match_groups(self):
        result = HarmonizerAgent._fallback_cluster(["BMI", "bmi", "BMI", "SBP"])
        assert len(result) == 2  # BMI and SBP

    def test_single_name(self):
        result = HarmonizerAgent._fallback_cluster(["BMI"])
        assert "BMI" in result


class TestApplyMapping:
    def test_applies_canonical_names(self):
        clusters = {
            "BMI_change": ["BMI change", "change in BMI", "body mass index change"],
            "SBP": ["systolic blood pressure", "SBP"],
        }
        result = HarmonizerAgent._apply_mapping(SAMPLE_EXTRACTIONS, clusters)
        # Check S1's first extraction got canonical name
        s1_bmi = result[0]["extractions"][0]
        assert s1_bmi["canonical_outcome"] == "BMI_change"

    def test_unmapped_keeps_original(self):
        clusters = {"SBP": ["systolic blood pressure", "SBP"]}
        result = HarmonizerAgent._apply_mapping(SAMPLE_EXTRACTIONS, clusters)
        # BMI change not in clusters → keeps original
        s1_bmi = result[0]["extractions"][0]
        assert s1_bmi["canonical_outcome"] == "BMI change"

    def test_flat_extractions(self):
        flat = [{"outcome_name": "SBP", "measure": "mean"}]
        clusters = {"SBP": ["SBP", "systolic blood pressure"]}
        result = HarmonizerAgent._apply_mapping(flat, clusters)
        assert result[0]["canonical_outcome"] == "SBP"


class TestHarmonizeIntegration:
    def test_empty_extractions(self):
        agent = _make_harmonizer()
        result = agent.harmonize([], PICO)
        assert result["outcome_clusters"] == {}
        assert result["harmonized_data"] == []
        assert result["unmapped"] == []

    def test_full_flow_with_llm_refinement(self):
        """Full harmonization with mocked embedding + LLM."""
        llm_refined = {
            "clusters": {
                "BMI_change": ["BMI change", "change in BMI", "body mass index change"],
                "systolic_BP": ["systolic blood pressure", "SBP"],
                "diastolic_BP": ["diastolic blood pressure"],
            },
            "unmapped": [],
            "reasoning": "Grouped similar BMI and BP outcomes",
        }
        agent = _make_harmonizer(llm_refined)

        # Mock embedding to avoid loading model
        with patch.object(agent, '_cluster_by_embedding', return_value={
            "BMI change": ["BMI change", "change in BMI", "body mass index change"],
            "systolic blood pressure": ["systolic blood pressure", "SBP"],
            "diastolic blood pressure": ["diastolic blood pressure"],
        }):
            result = agent.harmonize(SAMPLE_EXTRACTIONS, PICO)

        assert len(result["outcome_clusters"]) == 3
        assert "BMI_change" in result["outcome_clusters"]
        assert len(result["unmapped"]) == 0

    def test_llm_failure_uses_embedding_clusters(self):
        """If LLM refinement fails, use embedding clusters as-is."""
        router = MagicMock()
        router.call.return_value = ("NOT JSON {{{", {
            "model": "test", "input_tokens": 200,
            "output_tokens": 100, "cost": 0.003, "latency_ms": 300,
        })
        agent = HarmonizerAgent(router=router, cost_tracker=MagicMock(), config={})

        embedding_clusters = {
            "BMI change": ["BMI change", "change in BMI"],
            "SBP": ["SBP"],
        }
        with patch.object(agent, '_cluster_by_embedding', return_value=embedding_clusters):
            result = agent.harmonize(SAMPLE_EXTRACTIONS, PICO)

        # Should fallback to embedding clusters
        assert len(result["outcome_clusters"]) >= 1
