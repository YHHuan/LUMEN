"""Tests for screener agent."""

import json
import pytest
from unittest.mock import MagicMock, patch

from lumen.agents.screener import ScreenerAgent


def _make_screener(llm_response: dict | str | None = None):
    """Create a ScreenerAgent with mocked router."""
    router = MagicMock()
    cost_tracker = MagicMock()

    if llm_response is not None:
        if isinstance(llm_response, dict):
            text = json.dumps(llm_response)
        else:
            text = llm_response
        router.call.return_value = (text, {
            "model": "test", "input_tokens": 100,
            "output_tokens": 50, "cost": 0.001, "latency_ms": 200,
        })

    agent = ScreenerAgent(router=router, cost_tracker=cost_tracker, config={})
    return agent


SAMPLE_STUDY = {
    "study_id": "S001",
    "title": "Effect of Exercise on Blood Pressure",
    "abstract": "A randomized controlled trial of 200 patients...",
    "year": 2023,
}
SAMPLE_PICO = {"population": "adults with hypertension", "intervention": "exercise"}
SAMPLE_CRITERIA = {"include": ["RCT"], "exclude": ["pediatric"]}


class TestScreenerTierOverride:
    def test_default_tier_is_smart(self):
        agent = _make_screener()
        assert agent.tier == "smart"

    def test_tier_override_fast(self):
        router = MagicMock()
        agent = ScreenerAgent(router=router, cost_tracker=MagicMock(),
                              config={}, tier_override="fast")
        assert agent.tier == "fast"

    def test_tier_override_none_keeps_default(self):
        router = MagicMock()
        agent = ScreenerAgent(router=router, cost_tracker=MagicMock(),
                              config={}, tier_override=None)
        assert agent.tier == "smart"


class TestScreenerAgent:
    def test_screen_single_include(self):
        agent = _make_screener({
            "decision": "include",
            "confidence": 85,
            "reasoning": "RCT of exercise in adults",
            "key_factors": ["RCT", "adults", "exercise"],
        })
        result = agent.screen_single(SAMPLE_STUDY, SAMPLE_PICO, SAMPLE_CRITERIA)
        assert result["decision"] == "include"
        assert result["confidence"] == 85
        assert result["study_id"] == "S001"
        assert result["screener_id"] == "1"

    def test_screen_single_exclude(self):
        agent = _make_screener({
            "decision": "exclude",
            "confidence": 90,
            "reasoning": "Not an RCT",
            "key_factors": ["observational"],
        })
        result = agent.screen_single(SAMPLE_STUDY, SAMPLE_PICO, SAMPLE_CRITERIA)
        assert result["decision"] == "exclude"

    def test_invalid_decision_forced_to_include(self):
        """Ambiguous decision should be forced to include (conservative)."""
        agent = _make_screener({
            "decision": "maybe",
            "confidence": 70,
            "reasoning": "Unclear",
            "key_factors": [],
        })
        result = agent.screen_single(SAMPLE_STUDY, SAMPLE_PICO, SAMPLE_CRITERIA)
        assert result["decision"] == "include"
        assert result["confidence"] <= 50  # penalized

    def test_confidence_clamped(self):
        agent = _make_screener({
            "decision": "include",
            "confidence": 150,
            "reasoning": "Very sure",
            "key_factors": [],
        })
        result = agent.screen_single(SAMPLE_STUDY, SAMPLE_PICO, SAMPLE_CRITERIA)
        assert result["confidence"] == 100

    def test_confidence_negative_clamped(self):
        agent = _make_screener({
            "decision": "exclude",
            "confidence": -10,
            "reasoning": "Uncertain",
            "key_factors": [],
        })
        result = agent.screen_single(SAMPLE_STUDY, SAMPLE_PICO, SAMPLE_CRITERIA)
        assert result["confidence"] == 0

    def test_screen_batch(self):
        agent = _make_screener({
            "decision": "include",
            "confidence": 80,
            "reasoning": "Relevant",
            "key_factors": ["RCT"],
        })
        studies = [
            {"study_id": "S001", "title": "Study 1", "abstract": "..."},
            {"study_id": "S002", "title": "Study 2", "abstract": "..."},
        ]
        results = agent.screen_batch(studies, SAMPLE_PICO, SAMPLE_CRITERIA)
        assert len(results) == 2
        assert all(r["decision"] == "include" for r in results)

    def test_batch_parse_failure_conservative(self):
        """Parse failure in batch → include conservatively."""
        router = MagicMock()
        router.call.return_value = ("not json at all!!!", {
            "model": "test", "input_tokens": 100,
            "output_tokens": 50, "cost": 0.001, "latency_ms": 200,
        })
        agent = ScreenerAgent(router=router, cost_tracker=MagicMock(), config={})
        results = agent.screen_batch(
            [{"study_id": "S001", "title": "T", "abstract": "A"}],
            SAMPLE_PICO, SAMPLE_CRITERIA,
        )
        assert len(results) == 1
        assert results[0]["decision"] == "include"
        assert results[0]["confidence"] == 0
        assert results[0].get("parse_error") is True
