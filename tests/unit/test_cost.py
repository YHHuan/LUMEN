"""Tests for CostTracker."""
import json
import tempfile
from pathlib import Path

from lumen.core.cost import CostTracker


def test_record_and_summary():
    tracker = CostTracker()
    tracker.record("phase3", "screener", {
        "input_tokens": 500, "output_tokens": 100, "cost": 0.05,
    })
    tracker.record("phase3", "screener", {
        "input_tokens": 600, "output_tokens": 200, "cost": 0.07,
    })
    tracker.record("phase3", "arbiter", {
        "input_tokens": 300, "output_tokens": 150, "cost": 0.10,
    })

    summary = tracker.summary()
    assert summary["grand_total_calls"] == 3
    assert abs(summary["grand_total_cost"] - 0.22) < 1e-6
    assert summary["grand_total_tokens"] == 500 + 100 + 600 + 200 + 300 + 150
    assert summary["by_phase"]["phase3"]["screener"]["calls"] == 2
    assert summary["by_phase"]["phase3"]["arbiter"]["calls"] == 1


def test_jsonl_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = CostTracker(project_dir=tmpdir)
        tracker.record("phase1", "pico_interviewer", {
            "input_tokens": 200, "output_tokens": 50, "cost": 0.01,
        })
        tracker.record("phase2", "search", {
            "input_tokens": 100, "output_tokens": 30, "cost": 0.005,
        })

        log_path = Path(tmpdir) / "cost_log.jsonl"
        assert log_path.exists()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        entry = json.loads(lines[0])
        assert entry["phase"] == "phase1"
        assert entry["agent"] == "pico_interviewer"
        assert "timestamp" in entry


def test_from_jsonl_reconstruction():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write some entries
        tracker1 = CostTracker(project_dir=tmpdir)
        tracker1.record("phase3", "screener", {
            "input_tokens": 500, "output_tokens": 100, "cost": 0.05,
        })
        tracker1.record("phase5", "statistician", {
            "input_tokens": 800, "output_tokens": 400, "cost": 0.20,
        })

        # Reconstruct from JSONL
        log_path = Path(tmpdir) / "cost_log.jsonl"
        tracker2 = CostTracker.from_jsonl(log_path)

        summary = tracker2.summary()
        assert summary["grand_total_calls"] == 2
        assert abs(summary["grand_total_cost"] - 0.25) < 1e-6


def test_estimate_remaining():
    tracker = CostTracker()
    tracker.record("phase1", "pico_interviewer", {
        "input_tokens": 1000, "output_tokens": 500, "cost": 0.10,
    })
    tracker.record("phase2", "search", {
        "input_tokens": 500, "output_tokens": 200, "cost": 0.05,
    })

    estimate = tracker.estimate_remaining("phase2", n_studies=10)
    assert estimate["cost_so_far"] == 0.15
    assert estimate["estimated_remaining"] is not None
    assert estimate["remaining_phases"] > 0


def test_estimate_remaining_no_data():
    tracker = CostTracker()
    estimate = tracker.estimate_remaining("phase1", n_studies=0)
    assert estimate["estimated_remaining"] is None
