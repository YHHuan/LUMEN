"""Regression guard against the V4 "weight vocabulary" bug class (report item 3).

V4 had a dead guard: a producer emitted the literal ``"important"`` while the
consumer only recognised ``{CRITICAL, HIGH, MEDIUM}`` — so a hard-exclusion
guard never fired. v3's screening has no weighted-question guard at all, so it
cannot have that exact bug. The closest severity-routed guard in v3 is
``route_after_quality`` (critical anomaly → re-extract). This test locks the
invariant that the severity vocabulary the statistician *emits* stays aligned
with the token the router *escalates on*, so an "important"-style mismatch can
never be introduced silently.
"""
import inspect
import re

import lumen.agents.statistician as statistician
from lumen.core.graph import route_after_quality

# The single severity that route_after_quality escalates on.
ESCALATION_SEVERITY = "critical"


class TestRouteAfterQuality:
    def test_critical_unresolved_escalates(self):
        state = {"anomaly_flags": [{"severity": "critical", "resolved": False}]}
        assert route_after_quality(state) == "re_extract"

    def test_warning_does_not_escalate(self):
        state = {"anomaly_flags": [{"severity": "warning"}]}
        assert route_after_quality(state) == "proceed"

    def test_critical_but_resolved_does_not_escalate(self):
        state = {"anomaly_flags": [{"severity": "critical", "resolved": True}]}
        assert route_after_quality(state) == "proceed"

    def test_no_flags_proceeds(self):
        assert route_after_quality({}) == "proceed"


class TestSeverityVocabularyAligned:
    def test_emitted_severities_are_recognized(self):
        """Every severity literal the statistician emits is a known token, and
        the escalation token is actually among them (guard is live, not dead)."""
        src = inspect.getsource(statistician)
        emitted = set(re.findall(r'"severity"\s*:\s*"([a-z_]+)"', src))
        assert emitted, "expected to find severity literals in statistician.py"

        recognized = {"critical", "warning", "info"}
        # No stray vocabulary (an 'important'-style token would fail here).
        assert emitted <= recognized, f"unrecognized severity tokens: {emitted - recognized}"
        # The router's escalation token must be one the statistician can emit,
        # otherwise the critical-anomaly guard would be dead (the V4 failure).
        assert ESCALATION_SEVERITY in emitted
