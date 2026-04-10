"""Tests for LangGraph StateGraph definition."""
from unittest.mock import MagicMock
from lumen.core.graph import build_graph, route_after_quality, route_after_writing
from lumen.core.state import LumenState


def test_graph_compiles():
    router = MagicMock()
    cost_tracker = MagicMock()
    graph = build_graph(router=router, cost_tracker=cost_tracker, config={})
    assert graph is not None


def test_graph_compiles_without_args():
    """Graph can build with no args (deps expected in state at runtime)."""
    graph = build_graph()
    assert graph is not None


def test_route_after_quality_proceed():
    """No critical flags -> proceed."""
    state: LumenState = {"anomaly_flags": []}
    assert route_after_quality(state) == "proceed"


def test_route_after_quality_re_extract():
    """Unresolved critical flag -> re_extract."""
    state: LumenState = {
        "anomaly_flags": [{"severity": "critical", "resolved": False}],
    }
    assert route_after_quality(state) == "re_extract"


def test_route_after_quality_resolved_critical():
    """Resolved critical flag -> proceed."""
    state: LumenState = {
        "anomaly_flags": [{"severity": "critical", "resolved": True}],
    }
    assert route_after_quality(state) == "proceed"


def test_route_after_writing_done():
    """No contradicted claims -> done."""
    state: LumenState = {"fact_check_log": []}
    assert route_after_writing(state) == "done"


def test_route_after_writing_revise():
    """Unresolved contradicted claim -> revise."""
    state: LumenState = {
        "fact_check_log": [{"verdict": "CONTRADICTED", "resolved": False}],
    }
    assert route_after_writing(state) == "revise"


def test_route_after_writing_resolved():
    """Resolved contradicted claim -> done."""
    state: LumenState = {
        "fact_check_log": [{"verdict": "CONTRADICTED", "resolved": True}],
    }
    assert route_after_writing(state) == "done"
