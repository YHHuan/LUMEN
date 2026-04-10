"""Tests for context management utilities."""
from lumen.core.context import (
    check_pico_drift,
    compress_context,
    build_agent_context,
)


def test_pico_drift_no_drift():
    """Task with good PICO overlap -> None."""
    pico = {"population": "adults with diabetes", "intervention": "metformin"}
    result = check_pico_drift("screening study about metformin in diabetic adults", pico)
    assert result is None


def test_pico_drift_detected():
    """Task with zero PICO overlap -> warning."""
    pico = {"population": "adults with diabetes", "intervention": "metformin"}
    result = check_pico_drift(
        "analysis of machine learning algorithms for image classification tasks",
        pico,
    )
    assert result is not None
    assert "warning" in result


def test_pico_drift_empty_pico():
    """Empty PICO -> None (can't check)."""
    result = check_pico_drift("some task", {})
    assert result is None


def test_compress_context_short_text():
    """Short text should pass through unchanged."""
    text = "This is a short text."
    assert compress_context(text, max_tokens=100) == text


def test_compress_context_long_text_no_router():
    """Long text without router -> extractive truncation."""
    text = "word " * 5000  # ~5000 words
    result = compress_context(text, max_tokens=100)
    assert "[... compressed ...]" in result
    assert len(result) < len(text)


def test_build_agent_context_screener():
    state = {
        "pico": {"population": "adults"},
        "screening_criteria": {"inclusion": ["RCT"]},
        "extractions": [{"data": "should not see"}],
        "statistics_results": {"should": "not see"},
    }
    ctx = build_agent_context(state, "screener")
    assert "pico" in ctx
    assert "screening_criteria" in ctx
    assert "extractions" not in ctx
    assert "statistics_results" not in ctx


def test_build_agent_context_writer():
    state = {
        "pico": {"population": "adults"},
        "statistics_results": {"outcome1": {}},
        "quality_assessments": {"rob2": {}},
        "evidence_synthesis": {"key_findings": []},
        "included_studies": [{"id": "study1"}],
        "screening_results": [{"should": "not see"}],
    }
    ctx = build_agent_context(state, "writer")
    assert "pico" in ctx
    assert "statistics_results" in ctx
    assert "quality_assessments" in ctx
    assert "evidence_synthesis" in ctx
    assert "included_studies" in ctx
    assert "screening_results" not in ctx


def test_build_agent_context_unknown_agent():
    """Unknown agent gets full state."""
    state = {"pico": {}, "extractions": []}
    ctx = build_agent_context(state, "unknown_agent")
    assert ctx == state
