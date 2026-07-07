"""
Human review queue (stabilize P2.3).

LUMEN v3 already *generates* human-review signals during screening, full-text
review, extraction span-binding and statistics — but the pipeline never wrote
them anywhere a human would look. This module collects every review-worthy
signal from the final pipeline state and renders a single Markdown file
(``review_queue.md``) that a reviewer actually opens.

Everything here is PURE: it reads a state dict and returns rows / text. No LLM,
no network. ``write_review_queue`` is the only function that touches disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


def _sid(item: dict) -> str:
    return str(item.get("study_id", item.get("id", "unknown")))


def build_review_queue(state: dict) -> list[dict]:
    """Collect all review-worthy items from a final pipeline *state*.

    Returns a flat list of rows, each ``{stage, study_id, verdict, confidence,
    reason_codes, note}``. Robust to partial/legacy state (missing keys are
    skipped, older screening results without the tri-state annotation fall back
    to ``final_decision == 'human_review'``).
    """
    rows: list[dict] = []

    # ── Title/abstract screening: unclear / borderline / flagged ───────
    nhr = state.get("needs_human_review")
    if nhr is None:
        # Legacy fallback: derive from screening_results.
        nhr = [
            r for r in state.get("screening_results", [])
            if r.get("review_required")
            or r.get("final_decision") == "human_review"
        ]
    for r in nhr:
        rows.append({
            "stage": "title_abstract",
            "study_id": _sid(r),
            "verdict": r.get("screening_state", r.get("final_decision", "unclear")),
            "confidence": r.get("confidence"),
            "reason_codes": list(r.get("flags", [])),
            "note": f"auto_decision={r.get('final_decision')} "
                    f"method={r.get('method', '?')}",
        })

    # ── Full-text exclusions: verify the reason code is right ──────────
    for fr in state.get("fulltext_results", []):
        if fr.get("decision") == "exclude":
            code = fr.get("reason_code")
            rows.append({
                "stage": "fulltext",
                "study_id": _sid(fr),
                "verdict": "exclude",
                "confidence": fr.get("confidence"),
                "reason_codes": [code] if code else [],
                "note": (fr.get("exclusion_reason") or fr.get("reason") or "")[:160],
            })

    # ── Extraction: values that could not be located in the source ─────
    for ext in state.get("extractions", []):
        low = ext.get("low_confidence_spans", [])
        if low:
            fields = ", ".join(
                str(s.get("field", s.get("outcome_name", "?"))) for s in low[:6]
            )
            rows.append({
                "stage": "extraction",
                "study_id": _sid(ext),
                "verdict": "verify_source",
                "confidence": None,
                "reason_codes": ["LOW_SOURCE_MATCH"],
                "note": f"{len(low)} value(s) with weak source locator: {fields}",
            })

    # ── Statistics anomalies the human should see ──────────────────────
    for af in state.get("anomaly_flags", []):
        if af.get("severity") in ("critical", "warning") and not af.get("resolved"):
            rows.append({
                "stage": "statistics",
                "study_id": af.get("outcome", "—"),
                "verdict": af.get("severity"),
                "confidence": None,
                "reason_codes": [str(af.get("type", "ANOMALY")).upper()],
                "note": (af.get("description") or "")[:160],
            })

    return rows


def _fmt_conf(c: Any) -> str:
    return "" if c is None else f"{c}"


def format_review_queue_md(rows: list[dict], project_name: str = "") -> str:
    """Render review-queue *rows* as a human-readable Markdown document."""
    title = "# LUMEN Review Queue"
    if project_name:
        title += f" — {project_name}"

    lines = [title, ""]
    if not rows:
        lines += ["_No items require human review._", ""]
        return "\n".join(lines)

    by_stage: dict[str, list[dict]] = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(r)

    lines.append(f"**{len(rows)}** item(s) need a human. "
                 "Each was auto-handled by the pipeline; review to confirm.\n")

    # Summary counts
    lines.append("| Stage | Items |")
    lines.append("| --- | --- |")
    for stage, items in by_stage.items():
        lines.append(f"| {stage} | {len(items)} |")
    lines.append("")

    stage_titles = {
        "title_abstract": "Title/Abstract screening (unclear / borderline / disagreement)",
        "fulltext": "Full-text exclusions (verify reason code)",
        "extraction": "Extraction (values with weak source locator)",
        "statistics": "Statistics anomalies",
    }
    for stage, items in by_stage.items():
        lines.append(f"## {stage_titles.get(stage, stage)}")
        lines.append("")
        lines.append("| Study | Verdict | Confidence | Reason codes | Note |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in items:
            codes = ", ".join(r.get("reason_codes") or []) or "—"
            note = str(r.get("note", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {r['study_id']} | {r['verdict']} | "
                f"{_fmt_conf(r.get('confidence'))} | {codes} | {note} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_review_queue(state: dict, output_path: str | Path,
                       project_name: str = "") -> Path:
    """Build the queue from *state* and write ``review_queue.md``. Returns path."""
    rows = build_review_queue(state)
    md = format_review_queue_md(rows, project_name=project_name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    logger.info("review_queue_written", path=str(output_path), n_items=len(rows))
    return output_path
