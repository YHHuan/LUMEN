"""Tests for LumenState schema."""
from lumen.core.state import LumenState


def test_state_can_be_instantiated():
    """LumenState can be created with minimal fields (total=False)."""
    state = LumenState(pico={"population": "adults"}, current_phase="phase1")
    assert state["pico"]["population"] == "adults"
    assert state["current_phase"] == "phase1"


def test_state_accepts_all_fields():
    """LumenState accepts the full set of fields."""
    state = LumenState(
        pico={}, screening_criteria={}, search_strategy={},
        pico_completeness_score=0,
        raw_results=[], deduplicated_studies=[],
        prescreen_results=[], screening_results=[],
        fulltext_results=[], included_studies=[],
        extractions=[], outcome_clusters={}, harmonized_data=[],
        analysis_plan={}, statistics_results={}, anomaly_flags=[],
        quality_assessments={},
        evidence_synthesis={}, manuscript_sections={}, fact_check_log=[],
        current_phase="", cost_tracker={}, human_decisions=[],
        running_summary="",
    )
    assert isinstance(state, dict)
