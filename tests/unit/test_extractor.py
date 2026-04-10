"""Tests for 4-round IterResearch extractor."""

import json
import pytest
from unittest.mock import MagicMock, patch, call

from lumen.agents.extractor import ExtractorAgent, EVIDENCE_SPAN_THRESHOLD


def _make_extractor(round_responses: list[dict]):
    """Create ExtractorAgent with mocked router returning sequential responses."""
    router = MagicMock()
    cost_tracker = MagicMock()

    responses = []
    for r in round_responses:
        responses.append((json.dumps(r), {
            "model": "test", "input_tokens": 500,
            "output_tokens": 200, "cost": 0.005, "latency_ms": 500,
        }))
    router.call.side_effect = responses

    return ExtractorAgent(router=router, cost_tracker=cost_tracker, config={})


SAMPLE_SKELETON = {
    "design": "RCT",
    "arms": [{"name": "exercise", "description": "aerobic"}, {"name": "control", "description": "usual care"}],
    "n_per_arm": {"exercise": 50, "control": 50},
    "primary_outcomes": ["systolic_bp", "diastolic_bp"],
    "secondary_outcomes": ["bmi"],
    "key_tables_location": ["Table 2"],
    "follow_up_duration": "12 weeks",
    "setting": "community",
    "country": "Taiwan",
}

SAMPLE_EXTRACTIONS = [
    {
        "outcome_name": "systolic_bp",
        "measure": "mean",
        "timepoint": "12 weeks",
        "arm1": {"name": "exercise", "n": 50, "mean": 125.3, "sd": 12.1, "events": None, "total": None},
        "arm2": {"name": "control", "n": 50, "mean": 132.5, "sd": 13.4, "events": None, "total": None},
        "effect_reported": {"type": "MD", "value": -7.2, "ci_lower": -12.1, "ci_upper": -2.3, "p_value": 0.004},
    },
    {
        "outcome_name": "diastolic_bp",
        "measure": "mean",
        "timepoint": "12 weeks",
        "arm1": {"name": "exercise", "n": 50, "mean": 78.2, "sd": 8.5, "events": None, "total": None},
        "arm2": {"name": "control", "n": 50, "mean": 82.1, "sd": 9.1, "events": None, "total": None},
        "effect_reported": {"type": "MD", "value": -3.9, "ci_lower": -7.4, "ci_upper": -0.4, "p_value": 0.03},
    },
]

SAMPLE_CROSSCHECK = {
    "checks_passed": True,
    "issues": [],
}

SAMPLE_SPANS = [
    {"outcome_name": "systolic_bp", "field": "arm1.mean", "value": 125.3,
     "pdf_page": 5, "pdf_text_span": "mean SBP was 125.3 mmHg", "match_confidence": 0.95},
    {"outcome_name": "systolic_bp", "field": "arm2.mean", "value": 132.5,
     "pdf_page": 5, "pdf_text_span": "control group: 132.5 mmHg", "match_confidence": 0.90},
    {"outcome_name": "diastolic_bp", "field": "arm1.mean", "value": 78.2,
     "pdf_page": 5, "pdf_text_span": "DBP 78.2", "match_confidence": 0.85},
    {"outcome_name": "diastolic_bp", "field": "arm2.mean", "value": 82.1,
     "pdf_page": 5, "pdf_text_span": "not clearly found", "match_confidence": 0.5},
]

STUDY = {"study_id": "EXT001", "title": "Exercise and BP"}
PICO = {"population": "adults with hypertension", "intervention": "exercise"}


class TestExtractorAgent:
    def test_full_4round_extraction(self):
        agent = _make_extractor([
            SAMPLE_SKELETON,
            SAMPLE_EXTRACTIONS,
            SAMPLE_CROSSCHECK,
            SAMPLE_SPANS,
        ])
        result = agent.extract(STUDY, "PDF content here...", PICO)
        assert result["rounds_completed"] == 4
        assert result["study_id"] == "EXT001"
        assert result["skeleton"]["design"] == "RCT"
        assert len(result["extractions"]) == 2
        assert result["crosscheck"]["checks_passed"] is True

    def test_evidence_span_threshold(self):
        """v2 audit #8: match_score < 0.7 should be flagged."""
        agent = _make_extractor([
            SAMPLE_SKELETON,
            SAMPLE_EXTRACTIONS,
            SAMPLE_CROSSCHECK,
            SAMPLE_SPANS,
        ])
        result = agent.extract(STUDY, "PDF content...", PICO)
        low = result["low_confidence_spans"]
        # One span has confidence 0.5 < 0.7
        assert len(low) == 1
        assert low[0]["match_confidence"] < EVIDENCE_SPAN_THRESHOLD

    def test_round3_catches_issues(self):
        crosscheck_with_issues = {
            "checks_passed": False,
            "issues": [
                {"field": "arm1.events", "problem": "events > total (60 > 50)",
                 "severity": "critical", "suggestion": "Verify against Table 2"},
            ],
        }
        agent = _make_extractor([
            SAMPLE_SKELETON,
            SAMPLE_EXTRACTIONS,
            crosscheck_with_issues,
            SAMPLE_SPANS,
        ])
        result = agent.extract(STUDY, "PDF content...", PICO)
        assert not result["crosscheck"]["checks_passed"]
        assert len(result["crosscheck"]["issues"]) == 1
        assert result["crosscheck"]["issues"][0]["severity"] == "critical"

    def test_round2_handles_dict_response(self):
        """Round 2 may return {"extractions": [...]} instead of bare list."""
        agent = _make_extractor([
            SAMPLE_SKELETON,
            {"extractions": SAMPLE_EXTRACTIONS},
            SAMPLE_CROSSCHECK,
            SAMPLE_SPANS,
        ])
        result = agent.extract(STUDY, "PDF content...", PICO)
        assert len(result["extractions"]) == 2

    def test_all_4_rounds_called(self):
        agent = _make_extractor([
            SAMPLE_SKELETON,
            SAMPLE_EXTRACTIONS,
            SAMPLE_CROSSCHECK,
            SAMPLE_SPANS,
        ])
        agent.extract(STUDY, "PDF content...", PICO)
        assert agent.router.call.call_count == 4
