"""Integration tests for the full screening pipeline (union-hybrid strategy)."""

import json
import pytest
from unittest.mock import MagicMock

from lumen.agents.screener import ScreenerAgent
from lumen.agents.arbiter import ArbiterAgent, needs_arbiter, resolve_screening
from lumen.agents.screening_node import prescreen_node, screen_ta_node


def _mock_agent(cls, response: dict, **kwargs):
    router = MagicMock()
    router.call.return_value = (json.dumps(response), {
        "model": "test", "input_tokens": 100,
        "output_tokens": 50, "cost": 0.001, "latency_ms": 200,
    })
    return cls(router=router, cost_tracker=MagicMock(), config={}, **kwargs)


class TestPrescreenNode:
    def test_excludes_by_keyword(self):
        state = {
            "deduplicated_studies": [
                {"study_id": "1", "title": "Animal model of diabetes", "abstract": "Mice study"},
                {"study_id": "2", "title": "Human RCT of exercise", "abstract": "Randomized trial"},
            ],
            "screening_criteria": {
                "exclusion_keywords": ["animal", "mice"],
                "required_keywords": [],
            },
        }
        result = prescreen_node(state)
        assert len(result["deduplicated_studies"]) == 1
        assert result["deduplicated_studies"][0]["study_id"] == "2"

    def test_requires_keywords(self):
        state = {
            "deduplicated_studies": [
                {"study_id": "1", "title": "Unrelated chemistry paper", "abstract": "No match"},
                {"study_id": "2", "title": "RCT of exercise", "abstract": "Blood pressure"},
            ],
            "screening_criteria": {
                "exclusion_keywords": [],
                "required_keywords": ["exercise", "blood pressure"],
            },
        }
        result = prescreen_node(state)
        assert len(result["deduplicated_studies"]) == 1
        assert result["deduplicated_studies"][0]["study_id"] == "2"

    def test_empty_criteria_passes_all(self):
        state = {
            "deduplicated_studies": [
                {"study_id": "1", "title": "Any study", "abstract": "Anything"},
            ],
            "screening_criteria": {},
        }
        result = prescreen_node(state)
        assert len(result["deduplicated_studies"]) == 1


class TestUnionHybridRouting:
    def test_both_include_auto_include(self):
        """Union: both include → auto-include, no arbiter."""
        s1 = {"decision": "include", "confidence": 90}
        s2 = {"decision": "include", "confidence": 85}
        assert not needs_arbiter(s1, s2)
        result = resolve_screening(s1, s2)
        assert result["final_decision"] == "include"
        assert result["method"] == "dual_include"

    def test_one_includes_union_include(self):
        """Union: one includes → auto-include, no arbiter needed."""
        s1 = {"decision": "include", "confidence": 80}
        s2 = {"decision": "exclude", "confidence": 85}
        assert not needs_arbiter(s1, s2)
        result = resolve_screening(s1, s2)
        assert result["final_decision"] == "include"
        assert result["method"] == "union_include"

    def test_both_exclude_high_confidence(self):
        """Both exclude + high confidence → auto-exclude."""
        s1 = {"decision": "exclude", "confidence": 90}
        s2 = {"decision": "exclude", "confidence": 85}
        assert not needs_arbiter(s1, s2)
        result = resolve_screening(s1, s2)
        assert result["final_decision"] == "exclude"
        assert result["method"] == "dual_exclude_high_confidence"

    def test_both_exclude_low_confidence_arbiter_saves(self):
        """Both exclude + uncertain → arbiter catches false negative."""
        s1 = {"decision": "exclude", "confidence": 65}
        s2 = {"decision": "exclude", "confidence": 60}
        assert needs_arbiter(s1, s2)
        arb = {"decision": "include", "confidence": 75, "agreed_with": "neither"}
        result = resolve_screening(s1, s2, arb)
        assert result["final_decision"] == "include"
        assert result["arbiter_used"]

    def test_arbiter_low_confidence_routes_human(self):
        """v2 audit #7: arbiter not confident → human_review."""
        arbiter = _mock_agent(ArbiterAgent, {
            "decision": "include",
            "confidence": 40,
            "reasoning": "Not sure",
            "agreed_with": "screener1",
        })
        s1 = {"decision": "exclude", "confidence": 60, "reasoning": "R1"}
        s2 = {"decision": "exclude", "confidence": 55, "reasoning": "R2"}
        study = {"study_id": "S1", "title": "T", "abstract": "A"}
        result = arbiter.resolve(study, s1, s2, {"population": "adults"}, {})
        assert result["decision"] == "human_review"

    def test_arbiter_parse_failure_routes_human(self):
        """v2 audit #7: parse failure → human_review."""
        router = MagicMock()
        router.call.return_value = ("garbage not json {{{!!!", {
            "model": "test", "input_tokens": 100,
            "output_tokens": 50, "cost": 0.001, "latency_ms": 200,
        })
        arbiter = ArbiterAgent(router=router, cost_tracker=MagicMock(), config={})
        s1 = {"decision": "exclude", "confidence": 60, "reasoning": "R"}
        s2 = {"decision": "exclude", "confidence": 60, "reasoning": "R"}
        study = {"study_id": "S1", "title": "T", "abstract": "A"}
        result = arbiter.resolve(study, s1, s2, {}, {})
        assert result["decision"] == "human_review"


class TestScreenTaNode:
    def test_both_include_no_arbiter(self):
        """Both screeners include → included via union, arbiter not called."""
        include_response = {
            "decision": "include", "confidence": 90,
            "reasoning": "Relevant", "key_factors": ["RCT"],
        }
        screener1 = _mock_agent(ScreenerAgent, include_response)
        screener2 = _mock_agent(ScreenerAgent, include_response)
        arbiter = _mock_agent(ArbiterAgent, {
            "decision": "include", "confidence": 80,
            "reasoning": "N/A", "agreed_with": "screener1",
        })

        state = {
            "deduplicated_studies": [
                {"study_id": "S1", "title": "Good Study", "abstract": "Relevant RCT"},
            ],
            "pico": {"population": "adults"},
            "screening_criteria": {"include": ["RCT"]},
        }
        result = screen_ta_node(state, screener1, screener2, arbiter)
        assert len(result["included_studies"]) == 1
        assert result["screening_results"][0]["method"] == "dual_include"

    def test_disagreement_union_includes(self):
        """One includes, one excludes → union auto-includes (no arbiter)."""
        screener1 = _mock_agent(ScreenerAgent, {
            "decision": "include", "confidence": 80,
            "reasoning": "Looks relevant", "key_factors": [],
        })
        screener2 = _mock_agent(ScreenerAgent, {
            "decision": "exclude", "confidence": 75,
            "reasoning": "Wrong population", "key_factors": [],
        })
        arbiter = _mock_agent(ArbiterAgent, {
            "decision": "exclude", "confidence": 85,
            "reasoning": "Should not be called",
            "agreed_with": "screener2",
        })

        state = {
            "deduplicated_studies": [
                {"study_id": "S1", "title": "Borderline Study", "abstract": "..."},
            ],
            "pico": {"population": "elderly"},
            "screening_criteria": {},
        }
        result = screen_ta_node(state, screener1, screener2, arbiter)
        # Union: screener1 included → auto-include
        assert len(result["included_studies"]) == 1
        assert result["screening_results"][0]["method"] == "union_include"
        # Arbiter should NOT have been called
        assert result["screening_results"][0]["arbiter"] is None

    def test_both_exclude_uncertain_calls_arbiter(self):
        """Both exclude with low confidence → arbiter reviews."""
        screener1 = _mock_agent(ScreenerAgent, {
            "decision": "exclude", "confidence": 65,
            "reasoning": "Not sure", "key_factors": [],
        })
        screener2 = _mock_agent(ScreenerAgent, {
            "decision": "exclude", "confidence": 60,
            "reasoning": "Maybe irrelevant", "key_factors": [],
        })
        arbiter = _mock_agent(ArbiterAgent, {
            "decision": "include", "confidence": 75,
            "reasoning": "Actually relevant after closer look",
            "agreed_with": "neither",
        })

        state = {
            "deduplicated_studies": [
                {"study_id": "S1", "title": "Ambiguous Study", "abstract": "..."},
            ],
            "pico": {"population": "adults"},
            "screening_criteria": {},
        }
        result = screen_ta_node(state, screener1, screener2, arbiter)
        # Arbiter rescued this study
        assert len(result["included_studies"]) == 1
        assert result["screening_results"][0]["method"] == "arbiter"
        assert result["screening_results"][0]["arbiter"] is not None

    def test_cross_model_tier_override(self):
        """Verify screener1 can be instantiated with FAST tier."""
        s1 = _mock_agent(ScreenerAgent, {
            "decision": "include", "confidence": 80,
            "reasoning": "R", "key_factors": [],
        }, tier_override="fast")
        s2 = _mock_agent(ScreenerAgent, {
            "decision": "include", "confidence": 85,
            "reasoning": "R", "key_factors": [],
        })
        assert s1.tier == "fast"
        assert s2.tier == "smart"
