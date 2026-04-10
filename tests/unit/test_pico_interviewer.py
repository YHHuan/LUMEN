"""Tests for PICO interviewer agent."""

import json
import pytest
from unittest.mock import MagicMock

from lumen.agents.pico_interviewer import (
    PICOInterviewerAgent, COMPLETENESS_THRESHOLD, SCORING_RUBRIC,
)


def _make_interviewer(llm_response: dict | None = None):
    router = MagicMock()
    cost_tracker = MagicMock()
    if llm_response:
        router.call.return_value = (json.dumps(llm_response), {
            "model": "test", "input_tokens": 300,
            "output_tokens": 200, "cost": 0.005, "latency_ms": 500,
        })
    return PICOInterviewerAgent(router=router, cost_tracker=cost_tracker, config={})


COMPLETE_PICO = {
    "population": "adults with type 2 diabetes",
    "intervention": "GLP-1 receptor agonists",
    "comparator": "placebo or standard care",
    "outcome": "HbA1c reduction",
    "study_design": "RCT",
    "inclusion_criteria": ["age >= 18", "diagnosed T2DM"],
    "exclusion_criteria": ["type 1 diabetes"],
    "timing": "12 weeks minimum",
}

INCOMPLETE_PICO = {
    "population": "diabetes",
    "intervention": "medication",
}


class TestCompletenessScoring:
    def test_complete_pico_scores_high(self):
        agent = _make_interviewer()
        score = agent.assess_completeness(COMPLETE_PICO)
        assert score >= COMPLETENESS_THRESHOLD

    def test_full_pico_scores_100(self):
        agent = _make_interviewer()
        score = agent.assess_completeness(COMPLETE_PICO)
        assert score == 100

    def test_empty_pico_scores_zero(self):
        agent = _make_interviewer()
        score = agent.assess_completeness({})
        assert score == 0

    def test_incomplete_pico_scores_low(self):
        agent = _make_interviewer()
        score = agent.assess_completeness(INCOMPLETE_PICO)
        assert score < COMPLETENESS_THRESHOLD

    def test_population_specificity_matters(self):
        agent = _make_interviewer()
        # Single word = half credit
        score_vague = agent.assess_completeness({"population": "patients"})
        # Multi-word = full credit
        score_specific = agent.assess_completeness({"population": "adults with hypertension"})
        assert score_specific > score_vague

    def test_list_outcome_handled(self):
        agent = _make_interviewer()
        pico = {**COMPLETE_PICO, "outcome": ["HbA1c", "weight loss"]}
        score = agent.assess_completeness(pico)
        assert score >= COMPLETENESS_THRESHOLD

    def test_scoring_capped_at_100(self):
        agent = _make_interviewer()
        score = agent.assess_completeness(COMPLETE_PICO)
        assert score <= 100


class TestElicitation:
    def test_complete_pico_returns_quickly(self):
        """Score >= 80 → LLM standardizes but no questions needed."""
        refined = {
            "refined_pico": COMPLETE_PICO,
            "completeness_score": 95,
            "questions": [],
            "reasoning": "PICO is complete",
        }
        agent = _make_interviewer(refined)
        result = agent.elicit(COMPLETE_PICO)
        assert result["completeness_score"] >= COMPLETENESS_THRESHOLD
        assert result["pico"]["population"] == "adults with type 2 diabetes"

    def test_incomplete_pico_generates_questions(self):
        """Score < 80 → LLM generates refinement questions."""
        refined = {
            "refined_pico": {
                "population": "adults with diabetes mellitus",
                "intervention": "pharmacological treatment",
                "comparator": "",
                "outcome": "",
            },
            "completeness_score": 40,
            "questions": [
                "What specific comparator group?",
                "What is the primary outcome?",
            ],
            "reasoning": "Missing comparator and outcome",
        }
        agent = _make_interviewer(refined)
        result = agent.elicit(INCOMPLETE_PICO)
        assert len(result["questions"]) >= 1

    def test_none_pico_handled(self):
        """None initial PICO → starts fresh."""
        refined = {
            "refined_pico": {},
            "completeness_score": 0,
            "questions": ["What is your research question?"],
            "reasoning": "No PICO provided",
        }
        agent = _make_interviewer(refined)
        result = agent.elicit(None)
        assert "pico" in result

    def test_llm_failure_returns_original(self):
        """LLM parse failure → return original with score."""
        router = MagicMock()
        router.call.return_value = ("NOT JSON {{{", {
            "model": "test", "input_tokens": 100,
            "output_tokens": 50, "cost": 0.001, "latency_ms": 200,
        })
        agent = PICOInterviewerAgent(router=router, cost_tracker=MagicMock(), config={})
        result = agent.elicit(INCOMPLETE_PICO)
        assert result["pico"] == INCOMPLETE_PICO
        assert result["completeness_score"] < COMPLETENESS_THRESHOLD
