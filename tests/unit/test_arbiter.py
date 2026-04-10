"""Tests for arbiter agent — union-hybrid screening strategy.

Union-hybrid logic:
- Either include → auto-include (union, max sensitivity)
- Both exclude + high confidence → auto-exclude
- Both exclude + low confidence → arbiter reviews
- Arbiter low confidence → human_review

Fixes v2 audit #7: parse failure → human_review.
"""

import json
import pytest
from unittest.mock import MagicMock

from lumen.agents.arbiter import (
    ArbiterAgent,
    needs_arbiter,
    resolve_screening,
    CONFIDENCE_AUTO_ACCEPT,
    CONFIDENCE_ARBITER_MIN,
    CONFIDENCE_HUMAN_REVIEW,
)


def _make_arbiter(llm_response: dict | str | None = None):
    router = MagicMock()
    cost_tracker = MagicMock()
    if llm_response is not None:
        text = json.dumps(llm_response) if isinstance(llm_response, dict) else llm_response
        router.call.return_value = (text, {
            "model": "test", "input_tokens": 100,
            "output_tokens": 50, "cost": 0.002, "latency_ms": 300,
        })
    return ArbiterAgent(router=router, cost_tracker=cost_tracker, config={})


STUDY = {"study_id": "S001", "title": "Test Study", "abstract": "..."}
PICO = {"population": "adults"}
CRITERIA = {"include": ["RCT"]}


class TestNeedsArbiter:
    def test_both_include_no_arbiter(self):
        """Union: both include → auto-include, no arbiter needed."""
        s1 = {"decision": "include", "confidence": 90}
        s2 = {"decision": "include", "confidence": 85}
        assert not needs_arbiter(s1, s2)

    def test_one_includes_no_arbiter(self):
        """Union: either includes → auto-include, no arbiter needed."""
        s1 = {"decision": "include", "confidence": 90}
        s2 = {"decision": "exclude", "confidence": 85}
        assert not needs_arbiter(s1, s2)

    def test_one_includes_low_confidence_no_arbiter(self):
        """Union: even low-confidence include → auto-include."""
        s1 = {"decision": "include", "confidence": 40}
        s2 = {"decision": "exclude", "confidence": 90}
        assert not needs_arbiter(s1, s2)

    def test_both_exclude_high_confidence_no_arbiter(self):
        """Both exclude with high confidence → auto-exclude."""
        s1 = {"decision": "exclude", "confidence": 90}
        s2 = {"decision": "exclude", "confidence": 85}
        assert not needs_arbiter(s1, s2)

    def test_both_exclude_low_confidence_needs_arbiter(self):
        """Both exclude but uncertain → arbiter reviews for false negatives."""
        s1 = {"decision": "exclude", "confidence": 65}
        s2 = {"decision": "exclude", "confidence": 70}
        assert needs_arbiter(s1, s2)

    def test_both_exclude_one_low_needs_arbiter(self):
        """Both exclude, one uncertain → arbiter reviews."""
        s1 = {"decision": "exclude", "confidence": 50}
        s2 = {"decision": "exclude", "confidence": 90}
        assert needs_arbiter(s1, s2)


class TestArbiterAgent:
    def test_resolve_include(self):
        arbiter = _make_arbiter({
            "decision": "include",
            "confidence": 80,
            "reasoning": "Aligns with PICO",
            "agreed_with": "screener1",
        })
        s1 = {"decision": "exclude", "confidence": 70, "reasoning": "Uncertain"}
        s2 = {"decision": "exclude", "confidence": 60, "reasoning": "Not sure"}
        result = arbiter.resolve(STUDY, s1, s2, PICO, CRITERIA)
        assert result["decision"] == "include"
        assert result["confidence"] == 80
        assert result["study_id"] == "S001"

    def test_low_confidence_routes_human(self):
        """Arbiter confidence < 60 → human_review."""
        arbiter = _make_arbiter({
            "decision": "include",
            "confidence": 40,
            "reasoning": "Not sure",
            "agreed_with": "neither",
        })
        s1 = {"decision": "exclude", "confidence": 50, "reasoning": "Maybe"}
        s2 = {"decision": "exclude", "confidence": 50, "reasoning": "Maybe not"}
        result = arbiter.resolve(STUDY, s1, s2, PICO, CRITERIA)
        assert result["decision"] == "human_review"

    def test_parse_failure_routes_human(self):
        """v2 audit #7: parse failure → human_review, NOT include."""
        router = MagicMock()
        router.call.return_value = ("absolutely not json {{{", {
            "model": "test", "input_tokens": 100,
            "output_tokens": 50, "cost": 0.002, "latency_ms": 300,
        })
        arbiter = ArbiterAgent(router=router, cost_tracker=MagicMock(), config={})
        s1 = {"decision": "exclude", "confidence": 60, "reasoning": "R1"}
        s2 = {"decision": "exclude", "confidence": 60, "reasoning": "R2"}
        result = arbiter.resolve(STUDY, s1, s2, PICO, CRITERIA)
        assert result["decision"] == "human_review"
        assert result.get("parse_error") is True

    def test_invalid_decision_to_human_review(self):
        arbiter = _make_arbiter({
            "decision": "maybe",
            "confidence": 50,
            "reasoning": "Ambiguous",
            "agreed_with": "neither",
        })
        s1 = {"decision": "exclude", "confidence": 60, "reasoning": "R"}
        s2 = {"decision": "exclude", "confidence": 60, "reasoning": "R"}
        result = arbiter.resolve(STUDY, s1, s2, PICO, CRITERIA)
        assert result["decision"] == "human_review"


class TestResolveScreening:
    def test_both_include(self):
        """Both include → include via dual_include."""
        s1 = {"decision": "include", "confidence": 90}
        s2 = {"decision": "include", "confidence": 85}
        result = resolve_screening(s1, s2)
        assert result["final_decision"] == "include"
        assert result["method"] == "dual_include"
        assert not result["arbiter_used"]

    def test_union_one_includes(self):
        """One includes, one excludes → include via union."""
        s1 = {"decision": "include", "confidence": 80}
        s2 = {"decision": "exclude", "confidence": 70}
        result = resolve_screening(s1, s2)
        assert result["final_decision"] == "include"
        assert result["method"] == "union_include"
        assert result["confidence"] == 80
        assert not result["arbiter_used"]

    def test_union_reversed(self):
        """Second screener includes → still union include."""
        s1 = {"decision": "exclude", "confidence": 70}
        s2 = {"decision": "include", "confidence": 60}
        result = resolve_screening(s1, s2)
        assert result["final_decision"] == "include"
        assert result["method"] == "union_include"
        assert result["confidence"] == 60

    def test_dual_exclude_high_confidence(self):
        """Both exclude with high confidence → exclude without arbiter."""
        s1 = {"decision": "exclude", "confidence": 90}
        s2 = {"decision": "exclude", "confidence": 85}
        result = resolve_screening(s1, s2)
        assert result["final_decision"] == "exclude"
        assert result["method"] == "dual_exclude_high_confidence"
        assert not result["arbiter_used"]

    def test_dual_exclude_low_confidence_arbiter_decides(self):
        """Both exclude, low confidence → arbiter decides."""
        s1 = {"decision": "exclude", "confidence": 65}
        s2 = {"decision": "exclude", "confidence": 60}
        arb = {"decision": "include", "confidence": 75, "agreed_with": "neither"}
        result = resolve_screening(s1, s2, arb)
        assert result["final_decision"] == "include"
        assert result["method"] == "arbiter"
        assert result["arbiter_used"]

    def test_missing_arbiter_human_review(self):
        """Both exclude low confidence + no arbiter → human_review."""
        s1 = {"decision": "exclude", "confidence": 65}
        s2 = {"decision": "exclude", "confidence": 60}
        result = resolve_screening(s1, s2, None)
        assert result["final_decision"] == "human_review"
