"""
LUMEN v3 Cost Tracker.

Per-agent, per-phase cost tracking.
Append-only JSONL log + in-memory running totals.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class CostTracker:
    """Track LLM costs per phase and agent with JSONL persistence."""

    def __init__(self, project_dir: str | Path | None = None):
        self.totals: dict[str, dict[str, dict[str, float]]] = {}
        # {phase: {agent: {calls: int, input_tokens: int, output_tokens: int, cost: float}}}
        self._log_path: Path | None = None
        if project_dir:
            self._log_path = Path(project_dir) / "cost_log.jsonl"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, phase: str, agent: str, usage: dict[str, Any]) -> None:
        """Record a single LLM call's usage. Appends to JSONL + updates totals."""
        # Update in-memory totals
        if phase not in self.totals:
            self.totals[phase] = {}
        if agent not in self.totals[phase]:
            self.totals[phase][agent] = {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
            }

        bucket = self.totals[phase][agent]
        bucket["calls"] += 1
        bucket["input_tokens"] += usage.get("input_tokens", 0)
        bucket["output_tokens"] += usage.get("output_tokens", 0)
        bucket["cost"] += usage.get("cost", 0.0)

        # Append to JSONL
        entry = {
            "timestamp": time.time(),
            "phase": phase,
            "agent": agent,
            **usage,
        }
        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        """Return cost breakdown by phase and agent, plus grand total."""
        grand_total = 0.0
        grand_tokens = 0
        grand_calls = 0
        for phase_data in self.totals.values():
            for agent_data in phase_data.values():
                grand_total += agent_data["cost"]
                grand_tokens += agent_data["input_tokens"] + agent_data["output_tokens"]
                grand_calls += agent_data["calls"]

        return {
            "by_phase": self.totals,
            "grand_total_cost": round(grand_total, 6),
            "grand_total_tokens": grand_tokens,
            "grand_total_calls": grand_calls,
        }

    def estimate_remaining(self, current_phase: str, n_studies: int) -> dict[str, Any]:
        """Estimate remaining cost based on per-study cost so far.

        Uses the average cost-per-study from completed phases to project
        remaining phases.
        """
        phase_order = [
            "phase1", "phase2", "phase3", "phase4", "phase4.5",
            "phase5", "phase6",
        ]

        total_cost_so_far = 0.0
        completed_phases = []
        for p in phase_order:
            if p in self.totals:
                phase_cost = sum(a["cost"] for a in self.totals[p].values())
                total_cost_so_far += phase_cost
                completed_phases.append(p)
            if p == current_phase:
                break

        if not completed_phases or n_studies == 0:
            return {"estimated_remaining": None, "reason": "insufficient data"}

        cost_per_study = total_cost_so_far / n_studies

        try:
            current_idx = phase_order.index(current_phase)
        except ValueError:
            return {"estimated_remaining": None, "reason": f"unknown phase: {current_phase}"}

        remaining_phases = len(phase_order) - current_idx - 1
        # Rough heuristic: remaining phases cost ~same per study
        estimated = cost_per_study * n_studies * remaining_phases / max(len(completed_phases), 1)

        return {
            "cost_so_far": round(total_cost_so_far, 4),
            "cost_per_study": round(cost_per_study, 6),
            "completed_phases": completed_phases,
            "remaining_phases": remaining_phases,
            "estimated_remaining": round(estimated, 4),
            "n_studies": n_studies,
        }

    @classmethod
    def from_jsonl(cls, log_path: str | Path) -> CostTracker:
        """Reconstruct a CostTracker from an existing JSONL log file."""
        log_path = Path(log_path)
        # Don't pass project_dir to avoid re-writing to the same file during replay
        tracker = cls()
        tracker._log_path = log_path
        if log_path.exists():
            entries = []
            with open(log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))
            # Replay into memory only (temporarily disable file writes)
            tracker._log_path = None
            for entry in entries:
                phase = entry.pop("phase", "unknown")
                agent = entry.pop("agent", "unknown")
                entry.pop("timestamp", None)
                tracker.record(phase, agent, entry)
            # Restore log path for future writes
            tracker._log_path = log_path
        return tracker
