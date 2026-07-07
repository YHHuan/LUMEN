"""Tests for the human review queue (stabilize P2.3)."""
import tempfile
from pathlib import Path

from lumen.tools.visualization.review_queue import (
    build_review_queue,
    format_review_queue_md,
    write_review_queue,
)


SAMPLE_STATE = {
    "needs_human_review": [
        {"study_id": "S001", "final_decision": "exclude", "method": "arbiter",
         "confidence": 55, "screening_state": "unclear", "review_required": True,
         "flags": ["ARBITER_LOW_CONFIDENCE"]},
        {"study_id": "S002", "final_decision": "include", "method": "union_include",
         "confidence": 45, "screening_state": "include", "review_required": True,
         "flags": ["SCREENER_DISAGREEMENT"]},
    ],
    "fulltext_results": [
        {"study_id": "S010", "decision": "exclude", "confidence": 92,
         "exclusion_reason": "Post-hoc secondary analysis",
         "reason_code": "NOT_PRIMARY_STUDY"},
        {"study_id": "S011", "decision": "include", "confidence": 88},
    ],
    "extractions": [
        {"study_id": "S020", "low_confidence_spans": [
            {"field": "arm2.mean", "match_confidence": 0.5},
        ]},
        {"study_id": "S021", "low_confidence_spans": []},
    ],
    "anomaly_flags": [
        {"type": "trim_fill_direction_flip", "outcome": "mortality",
         "severity": "critical", "description": "Effect direction reversed"},
        {"type": "underpowered", "outcome": "los", "severity": "info",
         "description": "Only k=2"},
    ],
}


class TestBuildReviewQueue:
    def test_collects_all_stages(self):
        rows = build_review_queue(SAMPLE_STATE)
        stages = {r["stage"] for r in rows}
        assert stages == {"title_abstract", "fulltext", "extraction", "statistics"}

    def test_screening_rows(self):
        rows = build_review_queue(SAMPLE_STATE)
        ta = [r for r in rows if r["stage"] == "title_abstract"]
        assert len(ta) == 2
        assert any("ARBITER_LOW_CONFIDENCE" in r["reason_codes"] for r in ta)
        assert any("SCREENER_DISAGREEMENT" in r["reason_codes"] for r in ta)

    def test_only_fulltext_excludes(self):
        rows = build_review_queue(SAMPLE_STATE)
        ft = [r for r in rows if r["stage"] == "fulltext"]
        assert len(ft) == 1
        assert ft[0]["study_id"] == "S010"
        assert ft[0]["reason_codes"] == ["NOT_PRIMARY_STUDY"]

    def test_only_extractions_with_low_spans(self):
        rows = build_review_queue(SAMPLE_STATE)
        ext = [r for r in rows if r["stage"] == "extraction"]
        assert len(ext) == 1
        assert ext[0]["study_id"] == "S020"

    def test_only_serious_anomalies(self):
        rows = build_review_queue(SAMPLE_STATE)
        stat = [r for r in rows if r["stage"] == "statistics"]
        # critical surfaced, info suppressed
        assert len(stat) == 1
        assert stat[0]["verdict"] == "critical"

    def test_legacy_state_fallback(self):
        """No needs_human_review key → derive from screening_results."""
        legacy = {
            "screening_results": [
                {"study_id": "S1", "final_decision": "human_review",
                 "method": "arbiter", "confidence": 30},
                {"study_id": "S2", "final_decision": "exclude", "confidence": 95},
            ]
        }
        rows = build_review_queue(legacy)
        ta = [r for r in rows if r["stage"] == "title_abstract"]
        assert len(ta) == 1
        assert ta[0]["study_id"] == "S1"

    def test_empty_state(self):
        assert build_review_queue({}) == []


class TestFormatReviewQueue:
    def test_markdown_has_sections(self):
        rows = build_review_queue(SAMPLE_STATE)
        md = format_review_queue_md(rows, project_name="demo")
        assert "# LUMEN Review Queue — demo" in md
        assert "Title/Abstract" in md
        assert "NOT_PRIMARY_STUDY" in md
        assert "| Study |" in md

    def test_empty_message(self):
        md = format_review_queue_md([])
        assert "No items require human review" in md

    def test_note_pipe_escaped(self):
        rows = [{"stage": "fulltext", "study_id": "X", "verdict": "exclude",
                 "confidence": 90, "reason_codes": [], "note": "a|b|c"}]
        md = format_review_queue_md(rows)
        assert "a\\|b\\|c" in md  # pipes escaped so the table doesn't break


class TestWriteReviewQueue:
    def test_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review_queue.md"
            written = write_review_queue(SAMPLE_STATE, path, project_name="demo")
            assert written.exists()
            content = written.read_text()
            assert "LUMEN Review Queue" in content
            assert "S010" in content
