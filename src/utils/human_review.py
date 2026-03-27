"""
Human Review Overlay — LUMEN v2
================================
Allows human annotators to overrule LLM decisions and tracks
human-AI agreement rates across pipeline phases.

Supports:
- Screening overrides (Phase 3)
- Extraction corrections (Phase 4)
- RoB-2 judgment overrides
- GRADE assessment overrides
- Agreement statistics (per-phase, overall)

Storage: JSON overlay files alongside pipeline outputs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HumanReviewOverlay:
    """
    Manages human review overlays on top of LLM pipeline decisions.

    Each overlay entry records:
    - Which item was reviewed (study_id, section, etc.)
    - The LLM's original decision
    - The human's decision
    - Whether they agree
    - Reviewer ID and timestamp
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.overlay_dir = self.data_dir / "human_review"
        self.overlay_dir.mkdir(parents=True, exist_ok=True)

    # ── Record Overrides ──────────────────────────────

    def record_override(
        self,
        phase: str,
        item_id: str,
        field: str,
        llm_value: Any,
        human_value: Any,
        reviewer: str = "anonymous",
        reason: str = "",
    ) -> dict:
        """Record a single human override of an LLM decision."""
        entry = {
            "item_id": item_id,
            "field": field,
            "llm_value": llm_value,
            "human_value": human_value,
            "agrees": _values_agree(llm_value, human_value),
            "reviewer": reviewer,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }

        # Load existing overlay
        overlay = self._load_overlay(phase)
        overlay.append(entry)
        self._save_overlay(phase, overlay)

        status = "AGREE" if entry["agrees"] else "OVERRIDE"
        logger.info(f"Human review [{phase}] {item_id}.{field}: {status}")

        return entry

    def record_screening_override(
        self,
        study_id: str,
        llm_decision: str,
        human_decision: str,
        reviewer: str = "anonymous",
        reason: str = "",
    ) -> dict:
        """Record human override of a screening decision."""
        return self.record_override(
            phase="screening",
            item_id=study_id,
            field="inclusion_decision",
            llm_value=llm_decision,
            human_value=human_decision,
            reviewer=reviewer,
            reason=reason,
        )

    def record_extraction_override(
        self,
        study_id: str,
        field: str,
        llm_value: Any,
        human_value: Any,
        reviewer: str = "anonymous",
        reason: str = "",
    ) -> dict:
        """Record human correction of an extracted value."""
        return self.record_override(
            phase="extraction",
            item_id=study_id,
            field=field,
            llm_value=llm_value,
            human_value=human_value,
            reviewer=reviewer,
            reason=reason,
        )

    def record_rob2_override(
        self,
        study_id: str,
        domain: str,
        llm_judgment: str,
        human_judgment: str,
        reviewer: str = "anonymous",
        reason: str = "",
    ) -> dict:
        """Record human override of a RoB-2 domain judgment."""
        return self.record_override(
            phase="rob2",
            item_id=study_id,
            field=f"domain_{domain}",
            llm_value=llm_judgment,
            human_value=human_judgment,
            reviewer=reviewer,
            reason=reason,
        )

    # ── Apply Overrides ───────────────────────────────

    def apply_screening_overrides(self, screening_results: list) -> list:
        """Apply human screening overrides to LLM screening results."""
        overlay = self._load_overlay("screening")
        if not overlay:
            return screening_results

        # Build override map: latest override per study
        override_map: Dict[str, dict] = {}
        for entry in overlay:
            override_map[entry["item_id"]] = entry

        applied = 0
        for study in screening_results:
            sid = study.get("study_id", "")
            if sid in override_map:
                override = override_map[sid]
                if not override["agrees"]:
                    study["original_llm_decision"] = study.get("decision")
                    study["decision"] = override["human_value"]
                    study["human_reviewed"] = True
                    study["human_reviewer"] = override["reviewer"]
                    study["override_reason"] = override["reason"]
                    applied += 1

        logger.info(f"Applied {applied} screening overrides")
        return screening_results

    def apply_extraction_overrides(self, extracted_data: list) -> list:
        """Apply human extraction corrections."""
        overlay = self._load_overlay("extraction")
        if not overlay:
            return extracted_data

        # Group by study_id
        overrides_by_study: Dict[str, List[dict]] = {}
        for entry in overlay:
            sid = entry["item_id"]
            if sid not in overrides_by_study:
                overrides_by_study[sid] = []
            overrides_by_study[sid].append(entry)

        applied = 0
        for study in extracted_data:
            sid = study.get("study_id", "")
            if sid in overrides_by_study:
                for override in overrides_by_study[sid]:
                    if not override["agrees"]:
                        field = override["field"]
                        study[f"original_llm_{field}"] = study.get(field)
                        study[field] = override["human_value"]
                        applied += 1

                study["human_reviewed"] = True

        logger.info(f"Applied {applied} extraction overrides")
        return extracted_data

    # ── Agreement Statistics ──────────────────────────

    def compute_agreement(self, phase: str = None) -> dict:
        """
        Compute human-AI agreement statistics.

        Args:
            phase: Specific phase or None for all phases
        """
        if phase:
            phases = [phase]
        else:
            phases = [f.stem.replace("overlay_", "")
                      for f in self.overlay_dir.glob("overlay_*.json")]

        overall = {"total": 0, "agree": 0, "override": 0}
        per_phase = {}

        for p in phases:
            overlay = self._load_overlay(p)
            if not overlay:
                continue

            agree = sum(1 for e in overlay if e.get("agrees", False))
            total = len(overlay)
            override = total - agree

            per_phase[p] = {
                "total_reviews": total,
                "agreements": agree,
                "overrides": override,
                "agreement_rate": round(agree / max(total, 1), 4),
                "override_rate": round(override / max(total, 1), 4),
            }

            overall["total"] += total
            overall["agree"] += agree
            overall["override"] += override

        return {
            "overall": {
                "total_reviews": overall["total"],
                "agreements": overall["agree"],
                "overrides": overall["override"],
                "agreement_rate": round(
                    overall["agree"] / max(overall["total"], 1), 4
                ),
            },
            "per_phase": per_phase,
        }

    def generate_agreement_report(self) -> str:
        """Generate human-readable agreement report."""
        stats = self.compute_agreement()
        ov = stats["overall"]

        lines = [
            "=" * 60,
            "  LUMEN v2 — Human-AI Agreement Report",
            "=" * 60,
            "",
            f"  Total Reviews:     {ov['total_reviews']}",
            f"  Agreements:        {ov['agreements']}",
            f"  Overrides:         {ov['overrides']}",
            f"  Agreement Rate:    {ov['agreement_rate']:.1%}",
            "",
        ]

        if stats["per_phase"]:
            lines.extend(["  Per Phase:", "  " + "-" * 50])
            for phase, data in stats["per_phase"].items():
                lines.append(
                    f"  {phase:<20} {data['agreement_rate']:.1%} agree  "
                    f"({data['agreements']}/{data['total_reviews']})"
                )

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    # ── Storage ───────────────────────────────────────

    def _load_overlay(self, phase: str) -> list:
        path = self.overlay_dir / f"overlay_{phase}.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_overlay(self, phase: str, data: list):
        path = self.overlay_dir / f"overlay_{phase}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _values_agree(llm_value: Any, human_value: Any) -> bool:
    """Check if two values agree (flexible comparison)."""
    if llm_value == human_value:
        return True

    # String comparison (case-insensitive)
    if isinstance(llm_value, str) and isinstance(human_value, str):
        return llm_value.strip().lower() == human_value.strip().lower()

    # Numeric comparison (within 5%)
    if isinstance(llm_value, (int, float)) and isinstance(human_value, (int, float)):
        if human_value == 0:
            return llm_value == 0
        return abs(llm_value - human_value) / abs(human_value) < 0.05

    return False
