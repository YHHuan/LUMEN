"""
Human Intervention Logger — LUMEN v2
======================================
Structured logging of all human corrections and interventions.
Records: phase, study_id, action, field, auto_value, human_value,
         time_seconds, reason.

Used for cost analysis papers to quantify human-in-the-loop effort.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.utils.project import get_data_dir

logger = logging.getLogger(__name__)


class HumanInterventionLogger:
    """Log human interventions to a JSONL file for cost analysis."""

    def __init__(self, project_dir: Optional[str] = None):
        base = Path(project_dir) if project_dir else Path(get_data_dir())
        self._log_dir = base / ".audit"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "human_intervention_log.jsonl"
        self._timer_start: Optional[float] = None

    def start_timer(self):
        """Start timing a human intervention."""
        self._timer_start = time.time()

    def log(self, *,
            phase: str,
            study_id: str,
            action: str,
            field: str = "",
            auto_value: Any = None,
            human_value: Any = None,
            time_seconds: Optional[float] = None,
            reason: str = "") -> dict:
        """
        Log a single human intervention.

        Actions: correct_extraction, add_missing_data, resolve_conflict,
                 override_screening, correct_citation, manual_pdf_provision,
                 approve_gate, reject_gate, other.
        """
        if time_seconds is None and self._timer_start is not None:
            time_seconds = round(time.time() - self._timer_start, 1)
            self._timer_start = None

        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "study_id": study_id,
            "action": action,
            "field": field,
            "auto_value": _serialize(auto_value),
            "human_value": _serialize(human_value),
            "time_seconds": time_seconds,
            "reason": reason,
        }

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Human intervention logged: {action} on {study_id}/{field}")
        return entry

    def get_all(self) -> list:
        """Load all logged interventions."""
        if not self._log_file.exists():
            return []
        entries = []
        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def summary(self) -> dict:
        """Generate summary statistics for cost analysis."""
        entries = self.get_all()
        if not entries:
            return {"total_interventions": 0}

        by_phase = {}
        by_action = {}
        total_time = 0.0

        for e in entries:
            phase = e.get("phase", "unknown")
            action = e.get("action", "unknown")
            t = e.get("time_seconds") or 0.0

            by_phase.setdefault(phase, {"count": 0, "time": 0.0})
            by_phase[phase]["count"] += 1
            by_phase[phase]["time"] += t

            by_action.setdefault(action, {"count": 0, "time": 0.0})
            by_action[action]["count"] += 1
            by_action[action]["time"] += t

            total_time += t

        return {
            "total_interventions": len(entries),
            "total_time_seconds": round(total_time, 1),
            "total_time_minutes": round(total_time / 60, 1),
            "by_phase": by_phase,
            "by_action": by_action,
            "unique_studies": len(set(e.get("study_id", "") for e in entries)),
        }

    def reset(self):
        """Clear the intervention log (for fresh recording sessions)."""
        if self._log_file.exists():
            self._log_file.unlink()
        logger.info("Human intervention log reset")


def _serialize(val: Any) -> Any:
    """Make value JSON-serializable."""
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    try:
        json.dumps(val)
        return val
    except (TypeError, ValueError):
        return str(val)
