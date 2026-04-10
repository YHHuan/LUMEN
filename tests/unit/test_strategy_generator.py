"""Tests for strategy generator agent."""

import json
import pytest
from unittest.mock import MagicMock

from lumen.agents.strategy_generator import StrategyGeneratorAgent


def _make_generator(llm_response: dict | None = None):
    router = MagicMock()
    cost_tracker = MagicMock()
    if llm_response:
        router.call.return_value = (json.dumps(llm_response), {
            "model": "test", "input_tokens": 300,
            "output_tokens": 200, "cost": 0.005, "latency_ms": 500,
        })
    return StrategyGeneratorAgent(router=router, cost_tracker=cost_tracker, config={})


PICO = {
    "population": "adults with type 2 diabetes",
    "intervention": "GLP-1 receptor agonists",
    "comparator": "placebo",
    "outcome": "HbA1c reduction",
    "study_design": "RCT",
}


class TestStrategyGeneration:
    def test_generates_strategy(self):
        llm_response = {
            "search_queries": [
                {"database": "PubMed", "query": "(GLP-1 OR liraglutide) AND diabetes AND RCT",
                 "notes": "Main search"},
                {"database": "Embase", "query": "glp-1 AND diabetes AND randomized",
                 "notes": "Embase equivalent"},
            ],
            "screening_criteria": {
                "inclusion": ["RCT", "adults >= 18", "T2DM diagnosis"],
                "exclusion": ["type 1 diabetes", "pediatric"],
                "required_keywords": ["diabetes", "GLP-1"],
                "exclusion_keywords": ["type 1", "pediatric", "animal"],
            },
            "expected_yield": {
                "estimated_total_hits": 500,
                "estimated_included": 20,
                "reasoning": "Based on similar reviews",
            },
            "mesh_terms": ["Diabetes Mellitus, Type 2", "Glucagon-Like Peptide 1"],
            "reasoning": "Comprehensive search using MeSH and free text",
        }
        agent = _make_generator(llm_response)
        result = agent.generate(PICO)

        assert "search_strategy" in result
        assert "screening_criteria" in result
        assert len(result["search_strategy"]["queries"]) == 2
        assert "PubMed" in result["search_strategy"]["queries"][0]["database"]
        assert len(result["screening_criteria"]["required_keywords"]) >= 1

    def test_llm_failure_returns_empty(self):
        """LLM failure → empty but valid structure."""
        router = MagicMock()
        router.call.return_value = ("INVALID", {
            "model": "test", "input_tokens": 100,
            "output_tokens": 50, "cost": 0.001, "latency_ms": 200,
        })
        agent = StrategyGeneratorAgent(router=router, cost_tracker=MagicMock(), config={})
        result = agent.generate(PICO)
        assert result["search_strategy"]["queries"] == []
        assert result["screening_criteria"]["inclusion"] == []

    def test_output_structure(self):
        """Result has expected keys."""
        llm_response = {
            "search_queries": [{"database": "PubMed", "query": "test", "notes": ""}],
            "screening_criteria": {
                "inclusion": ["a"], "exclusion": ["b"],
                "required_keywords": ["c"], "exclusion_keywords": ["d"],
            },
            "mesh_terms": ["Term1"],
        }
        agent = _make_generator(llm_response)
        result = agent.generate(PICO)

        strategy = result["search_strategy"]
        assert "queries" in strategy
        assert "mesh_terms" in strategy

        criteria = result["screening_criteria"]
        assert "inclusion" in criteria
        assert "exclusion" in criteria
        assert "required_keywords" in criteria
        assert "exclusion_keywords" in criteria
