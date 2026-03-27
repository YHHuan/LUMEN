"""
Human Review Card Generator — LUMEN v2
=========================================
Generates HTML review document with collapsible study cards,
evidence spans, and confidence flags.
"""

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

REVIEW_HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LUMEN v2 — Human Review</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
.study-card { background: white; border-radius: 8px; margin: 15px 0;
              box-shadow: 0 2px 5px rgba(0,0,0,0.1); overflow: hidden; }
.card-header { padding: 15px 20px; cursor: pointer; display: flex;
               justify-content: space-between; align-items: center;
               background: #ecf0f1; border-bottom: 1px solid #ddd; }
.card-header:hover { background: #d5dbdb; }
.card-header h2 { margin: 0; font-size: 16px; color: #2c3e50; }
.card-body { padding: 20px; display: none; }
.card-body.active { display: block; }
.flag-count { font-size: 14px; }
.study-info { width: 100%; border-collapse: collapse; margin-bottom: 15px; }
.study-info th { text-align: left; padding: 6px 12px; background: #f8f9fa;
                 width: 150px; font-weight: 600; }
.study-info td { padding: 6px 12px; }
.outcomes-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
.outcomes-table th, .outcomes-table td { padding: 6px 10px; border: 1px solid #ddd;
                                          text-align: center; font-size: 13px; }
.outcomes-table th { background: #3498db; color: white; }
.evidence-span { background: #fffde7; padding: 8px 12px; margin: 5px 0;
                 border-left: 3px solid #f39c12; font-size: 13px; }
.flag-warning { color: #e67e22; }
.flag-error { color: #e74c3c; }
.confidence-high { color: #27ae60; }
.confidence-medium { color: #f39c12; }
.confidence-low { color: #e74c3c; }
.summary { background: #eaf2f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
</style>
<script>
function toggleCard(header) {
  const body = header.nextElementSibling;
  body.classList.toggle('active');
}
function expandAll() {
  document.querySelectorAll('.card-body').forEach(b => b.classList.add('active'));
}
function collapseAll() {
  document.querySelectorAll('.card-body').forEach(b => b.classList.remove('active'));
}
</script>
</head>
<body>
<h1>LUMEN v2 — Human Review Document</h1>
<div class="summary">
<button onclick="expandAll()">Expand All</button>
<button onclick="collapseAll()">Collapse All</button>
</div>
"""

REVIEW_HTML_FOOTER = """
</body>
</html>
"""


def generate_review_html(extracted_data: list, output_path: str) -> str:
    """Generate HTML review document with collapsible study cards."""
    parts = [REVIEW_HTML_HEADER]

    # Summary
    total = len(extracted_data)
    flagged = sum(1 for s in extracted_data if _count_flags(s) > 0)
    parts.append(
        f'<div class="summary">'
        f'<p><strong>Total studies:</strong> {total} | '
        f'<strong>Flagged for review:</strong> {flagged}</p>'
        f'</div>'
    )

    for study in sorted(extracted_data, key=lambda s: s.get("canonical_citation", "")):
        parts.append(_generate_study_card(study))

    parts.append(REVIEW_HTML_FOOTER)

    html = "\n".join(parts)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Generated review HTML: {output_path}")
    return output_path


def _generate_study_card(study: dict) -> str:
    flags = _get_flags(study)
    flag_icons = "".join(
        '<span class="flag-error">!</span>' if f["level"] == "error"
        else '<span class="flag-warning">?</span>'
        for f in flags
    )

    citation = study.get("canonical_citation", study.get("study_id", "Unknown"))

    outcomes_html = _generate_outcomes_table(study)
    evidence_html = _generate_evidence_list(study)
    flags_html = ""
    if flags:
        flags_html = "<h3>Flags</h3><ul>"
        for f in flags:
            cls = "flag-error" if f["level"] == "error" else "flag-warning"
            flags_html += f'<li class="{cls}">{f["message"]}</li>'
        flags_html += "</ul>"

    return f"""
    <div class="study-card" id="{study.get('study_id', '')}">
        <div class="card-header" onclick="toggleCard(this)">
            <h2>{citation}</h2>
            <span class="flag-count">{flag_icons} [{len(flags)} flags]</span>
        </div>
        <div class="card-body">
            <table class="study-info">
                <tr><th>Study ID</th><td>{study.get('study_id', 'N/A')}</td></tr>
                <tr><th>Design</th><td>{study.get('study_design', 'N/A')}</td></tr>
                <tr><th>Population</th><td>{study.get('population_description', 'N/A')} (N={study.get('total_n', '?')})</td></tr>
                <tr><th>Intervention</th><td>{study.get('intervention_description', 'N/A')}</td></tr>
                <tr><th>Control</th><td>{study.get('control_description', 'N/A')}</td></tr>
            </table>
            <h3>Outcomes</h3>
            {outcomes_html}
            {flags_html}
            <h3>Evidence Spans</h3>
            {evidence_html}
        </div>
    </div>
    """


def _generate_outcomes_table(study: dict) -> str:
    outcomes = study.get("outcomes", [])
    if not outcomes:
        return "<p>No outcomes extracted.</p>"

    rows = ""
    for o in outcomes:
        ig = o.get("intervention_group", {})
        cg = o.get("control_group", {})
        conf_class = f"confidence-{ig.get('confidence', 'medium')}"

        rows += f"""
        <tr>
            <td>{o.get('measure', 'N/A')}</td>
            <td>{o.get('timepoint', 'N/A')}</td>
            <td>{ig.get('mean', '?')} +/- {ig.get('sd', '?')} (n={ig.get('n', '?')})</td>
            <td>{cg.get('mean', '?')} +/- {cg.get('sd', '?')} (n={cg.get('n', '?')})</td>
            <td class="{conf_class}">{ig.get('confidence', '?')}</td>
            <td>{ig.get('evidence_type', '?')}</td>
        </tr>
        """

    return f"""
    <table class="outcomes-table">
        <tr><th>Measure</th><th>Timepoint</th><th>Intervention M+/-SD (n)</th>
            <th>Control M+/-SD (n)</th><th>Confidence</th><th>Source</th></tr>
        {rows}
    </table>
    """


def _generate_evidence_list(study: dict) -> str:
    parts = []
    for o in study.get("outcomes", []):
        for grp_name in ["intervention_group", "control_group"]:
            grp = o.get(grp_name, {})
            span = grp.get("evidence_span", "")
            if span and span != "NOT FOUND IN SOURCE":
                parts.append(
                    f'<div class="evidence-span">'
                    f'<strong>{o.get("measure", "?")} ({grp_name}):</strong> '
                    f'{span} [p.{grp.get("evidence_page", "?")}]</div>'
                )
    return "\n".join(parts) if parts else "<p>No evidence spans recorded.</p>"


def _get_flags(study: dict) -> list:
    flags = []
    for o in study.get("outcomes", []):
        for grp_name in ["intervention_group", "control_group"]:
            grp = o.get(grp_name, {})
            measure = o.get("measure", "?")

            if grp.get("confidence") == "low":
                flags.append({"level": "warning", "message": f"Low confidence: {measure} {grp_name}"})
            if grp.get("evidence_span", "").startswith("NOT FOUND"):
                flags.append({"level": "error", "message": f"Evidence not found: {measure} {grp_name}"})
            if grp.get("sd") and grp.get("mean") and grp["mean"] != 0:
                cv = abs(grp["sd"] / grp["mean"])
                if cv > 2.0:
                    flags.append({"level": "warning", "message": f"Large SD: {measure} {grp_name} (CV={cv:.1f})"})
    return flags


def _count_flags(study: dict) -> int:
    return len(_get_flags(study))
