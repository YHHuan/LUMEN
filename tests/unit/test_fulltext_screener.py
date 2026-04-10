"""Tests for fulltext screener — fixes v2 audit #15."""

import json
import pytest
from unittest.mock import MagicMock

from lumen.agents.fulltext_screener import (
    FulltextScreenerAgent,
    _truncate_at_paragraph,
    _extract_priority_sections,
)


def _make_ft_screener(llm_response: dict | None = None):
    router = MagicMock()
    cost_tracker = MagicMock()
    if llm_response is not None:
        router.call.return_value = (json.dumps(llm_response), {
            "model": "test", "input_tokens": 200,
            "output_tokens": 80, "cost": 0.003, "latency_ms": 400,
        })
    return FulltextScreenerAgent(router=router, cost_tracker=cost_tracker, config={})


STUDY = {"study_id": "FT001", "title": "Exercise and BP", "year": 2023}
PICO = {"population": "adults"}
CRITERIA = {"include": ["RCT"]}


class TestFulltextScreener:
    def test_screen_include(self):
        agent = _make_ft_screener({
            "decision": "include",
            "confidence": 88,
            "reasoning": "RCT with relevant outcomes",
            "key_sections_reviewed": ["Methods", "Results"],
            "exclusion_reason": None,
        })
        result = agent.screen(STUDY, "Full text here...", PICO, CRITERIA)
        assert result["decision"] == "include"
        assert result["confidence"] == 88
        assert result["study_id"] == "FT001"

    def test_screen_exclude(self):
        agent = _make_ft_screener({
            "decision": "exclude",
            "confidence": 95,
            "reasoning": "Observational study",
            "key_sections_reviewed": ["Methods"],
            "exclusion_reason": "Not an RCT",
        })
        result = agent.screen(STUDY, "Full text...", PICO, CRITERIA)
        assert result["decision"] == "exclude"
        assert result["exclusion_reason"] == "Not an RCT"

    def test_parse_failure_conservative(self):
        router = MagicMock()
        router.call.return_value = ("NOT JSON!!!", {
            "model": "test", "input_tokens": 200,
            "output_tokens": 80, "cost": 0.003, "latency_ms": 400,
        })
        agent = FulltextScreenerAgent(router=router, cost_tracker=MagicMock(), config={})
        result = agent.screen(STUDY, "Full text...", PICO, CRITERIA)
        assert result["decision"] == "include"
        assert result["confidence"] == 0
        assert result.get("parse_error") is True


class TestTruncation:
    """v2 audit #15: truncation at sentence/paragraph boundary."""

    def test_short_text_unchanged(self):
        text = "Short paragraph.\n\nAnother paragraph."
        result = _truncate_at_paragraph(text, max_chars=1000)
        assert result == text

    def test_truncate_at_paragraph_boundary(self):
        para1 = "First paragraph. " * 20  # ~340 chars
        para2 = "Second paragraph. " * 20
        para3 = "Third paragraph. " * 20
        text = f"{para1}\n\n{para2}\n\n{para3}"
        result = _truncate_at_paragraph(text, max_chars=400)
        assert result.endswith("[... truncated at paragraph boundary]")
        assert "Third paragraph" not in result

    def test_truncate_at_sentence_boundary(self):
        # One very long paragraph with no double-newlines until the end
        text = "Sentence one. Sentence two. Sentence three. " * 30
        result = _truncate_at_paragraph(text, max_chars=200)
        # Should end at a sentence, not mid-word
        assert result.rstrip().endswith("]")
        assert "truncated" in result

    def test_never_cuts_mid_sentence_if_possible(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        result = _truncate_at_paragraph(text, max_chars=50)
        # With such a small limit, may have to hard cut, but should try sentence first
        assert "truncated" in result


class TestPrioritySections:
    def test_extracts_methods_and_results(self):
        text = (
            "Introduction\nSome intro text.\n\n"
            "Methods\nStudy design and participants.\n\n"
            "Results\nWe found significant effects.\n\n"
            "Discussion\nOur findings suggest.\n\n"
            "References\n1. Smith et al."
        )
        result = _extract_priority_sections(text)
        assert result is not None
        assert "Study design" in result
        assert "significant effects" in result

    def test_returns_none_when_no_sections(self):
        text = "Just a blob of unstructured text without clear section headings."
        result = _extract_priority_sections(text)
        assert result is None
