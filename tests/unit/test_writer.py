"""Tests for writer agent."""

import json
import pytest
from unittest.mock import MagicMock

from lumen.agents.writer import WriterAgent, SECTION_ORDER


def _make_writer(llm_responses: list[dict] | None = None):
    router = MagicMock()
    cost_tracker = MagicMock()
    if llm_responses:
        side_effects = []
        for r in llm_responses:
            side_effects.append((json.dumps(r), {
                "model": "test", "input_tokens": 500,
                "output_tokens": 300, "cost": 0.008, "latency_ms": 600,
            }))
        router.call.side_effect = side_effects
    return WriterAgent(router=router, cost_tracker=cost_tracker, config={})


PICO = {"population": "adults with hypertension", "intervention": "exercise"}

STATS_RESULTS = {
    "systolic_BP": {
        "k": 5,
        "effect_size_type": "SMD",
        "meta": {
            "pooled_effect": -0.55,
            "pooled_se": 0.15,
            "ci_lower": -0.84,
            "ci_upper": -0.26,
            "tau2": 0.02,
            "i2": 35.0,
            "q": 6.15,
            "q_p": 0.19,
            "prediction_interval": (-1.1, 0.0),
            "k": 5,
        },
    },
}

EXTRACTIONS = [
    {"study_id": "S1", "extractions": [{"outcome_name": "SBP", "canonical_outcome": "systolic_BP"}]},
    {"study_id": "S2", "extractions": [{"outcome_name": "SBP", "canonical_outcome": "systolic_BP"}]},
]

QUALITY = {
    "rob2": {"S1": {"overall": "low"}, "S2": {"overall": "some_concerns"}},
    "grade": {"systolic_BP": {"grade": "moderate", "certainty": 3}},
}


class TestSectionOrder:
    def test_order_is_correct(self):
        assert SECTION_ORDER == ["methods", "results", "discussion", "introduction", "abstract"]


class TestFactCheckSection:
    def test_empty_section_returns_unchanged(self):
        agent = _make_writer()
        result = agent._fact_check_section("", "methods", {}, [], {})
        assert result["revised_text"] == ""
        assert result["n_contradicted"] == 0

    def test_placeholder_section_returns_unchanged(self):
        agent = _make_writer()
        result = agent._fact_check_section(
            "[methods section could not be generated]",
            "methods", {}, [], {},
        )
        assert result["revised_text"].startswith("[")

    def test_contradicted_claim_gets_corrected(self):
        """Fact-checker finds wrong number → corrected in revised text."""
        fact_check_response = {
            "claims": [
                {
                    "text": "included 12 studies",
                    "verdict": "CONTRADICTED",
                    "evidence": "Only 10 studies in data",
                    "corrected_text": "included 10 studies",
                },
            ],
            "summary": {"n_supported": 0, "n_contradicted": 1, "n_unsupported": 0},
        }
        agent = _make_writer([fact_check_response])
        original = "This review included 12 studies examining the effect."
        result = agent._fact_check_section(
            original, "results", STATS_RESULTS, EXTRACTIONS, QUALITY,
        )
        assert "included 10 studies" in result["revised_text"]
        assert result["n_contradicted"] == 1

    def test_unsupported_claim_gets_citation_needed(self):
        """Unsupported claim gets [CITATION NEEDED] tag."""
        fact_check_response = {
            "claims": [
                {
                    "text": "Previous meta-analyses found similar results",
                    "verdict": "UNSUPPORTED",
                    "evidence": "No source data for this claim",
                    "corrected_text": None,
                },
            ],
            "summary": {"n_supported": 0, "n_contradicted": 0, "n_unsupported": 1},
        }
        agent = _make_writer([fact_check_response])
        original = "Previous meta-analyses found similar results in older populations."
        result = agent._fact_check_section(
            original, "discussion", STATS_RESULTS, EXTRACTIONS, QUALITY,
        )
        assert "[CITATION NEEDED]" in result["revised_text"]


class TestFullWritePipeline:
    def test_full_write_with_mocked_llm(self):
        """Full pipeline: synthesis + 5 sections + 5 fact-checks = 11 LLM calls."""
        synthesis_response = {
            "key_findings": [
                "Exercise reduces SBP by SMD -0.55 (95% CI -0.84 to -0.26)",
            ],
            "evidence_table": [
                {"study_id": "S1", "design": "RCT", "n": 100,
                 "outcomes_reported": ["SBP"], "risk_of_bias": "low",
                 "key_result": "SBP reduced"},
            ],
            "narrative_skeleton": {
                "methods": "Standard systematic review methods",
                "results": "Meta-analysis shows benefit",
                "discussion": "Findings support exercise",
                "introduction": "Hypertension is a major concern",
                "abstract": "Background, methods, results, conclusions",
            },
            "grade_summary": {"systolic_BP": "Moderate certainty"},
        }

        section_responses = []
        for section in SECTION_ORDER:
            section_responses.append({
                "section_name": section,
                "text": f"This is the {section} section with k=5 studies.",
                "citations_used": ["S1", "S2"],
                "statistics_referenced": ["pooled SMD = -0.55"],
            })

        factcheck_responses = []
        for section in SECTION_ORDER:
            factcheck_responses.append({
                "claims": [
                    {"text": "k=5 studies", "verdict": "SUPPORTED",
                     "evidence": "Data confirms k=5", "corrected_text": None},
                ],
                "summary": {"n_supported": 1, "n_contradicted": 0, "n_unsupported": 0},
            })

        all_responses = [synthesis_response] + section_responses + factcheck_responses
        agent = _make_writer(all_responses)

        result = agent.write(STATS_RESULTS, EXTRACTIONS, QUALITY, PICO)

        assert "evidence_synthesis" in result
        assert "manuscript_sections" in result
        assert "fact_check_log" in result

        # All sections present
        for section in SECTION_ORDER:
            assert section in result["manuscript_sections"]
            assert len(result["manuscript_sections"][section]) > 0

        # Fact-check log has entries
        assert len(result["fact_check_log"]) == 5  # one per section

    def test_write_with_empty_data(self):
        """Empty stats → synthesis still attempts."""
        synthesis_response = {
            "key_findings": ["No data available"],
            "evidence_table": [],
            "narrative_skeleton": {s: "" for s in SECTION_ORDER},
        }
        section_responses = [
            {"section_name": s, "text": f"[{s}]", "citations_used": [], "statistics_referenced": []}
            for s in SECTION_ORDER
        ]
        # Fact-check on short placeholder text returns early (no LLM call)
        # because text starts with "["
        all_responses = [synthesis_response] + section_responses
        agent = _make_writer(all_responses)

        result = agent.write({}, [], {}, PICO)
        assert "manuscript_sections" in result
