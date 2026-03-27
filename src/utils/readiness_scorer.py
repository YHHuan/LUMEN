"""
Publication Readiness Scorer — LUMEN v2
========================================
Evaluates manuscript and pipeline output readiness for publication.
Checks PRISMA 2020 compliance, statistical rigor, citation grounding,
data completeness, and transparency metrics.

Produces a structured readiness report with:
- Overall score (0-100)
- Per-dimension scores
- Actionable checklist of issues to fix
- Full cost/transparency audit
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Scoring Dimensions
# ======================================================================

class ReadinessDimension:
    """A single evaluation dimension."""

    def __init__(self, name: str, weight: float, max_score: float = 100.0):
        self.name = name
        self.weight = weight
        self.max_score = max_score
        self.score: float = 0.0
        self.issues: List[dict] = []
        self.checks_passed: int = 0
        self.checks_total: int = 0

    def add_check(self, label: str, passed: bool, severity: str = "warning",
                  detail: str = ""):
        self.checks_total += 1
        if passed:
            self.checks_passed += 1
        else:
            self.issues.append({
                "check": label,
                "severity": severity,    # "critical" | "warning" | "info"
                "detail": detail,
            })

    def compute_score(self):
        if self.checks_total == 0:
            self.score = 0.0
            return
        self.score = round(
            (self.checks_passed / self.checks_total) * self.max_score, 1
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "score": self.score,
            "max_score": self.max_score,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "issues": self.issues,
        }


# ======================================================================
# Main Scorer
# ======================================================================

class PublicationReadinessScorer:
    """Evaluate pipeline outputs for publication readiness."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def score(self) -> dict:
        """Run all readiness checks and produce a scored report."""
        dimensions = [
            self._check_prisma_compliance(),
            self._check_data_completeness(),
            self._check_statistical_rigor(),
            self._check_quality_assessment(),
            self._check_citation_grounding(),
            self._check_manuscript_quality(),
            self._check_transparency_audit(),
        ]

        for dim in dimensions:
            dim.compute_score()

        # Weighted overall score
        total_weight = sum(d.weight for d in dimensions)
        overall = sum(d.score * d.weight for d in dimensions) / max(total_weight, 1)
        overall = round(overall, 1)

        # Readiness grade
        if overall >= 90:
            grade = "A"
            verdict = "Ready for submission"
        elif overall >= 75:
            grade = "B"
            verdict = "Minor revisions needed"
        elif overall >= 60:
            grade = "C"
            verdict = "Significant issues to address"
        elif overall >= 40:
            grade = "D"
            verdict = "Major gaps — not ready"
        else:
            grade = "F"
            verdict = "Incomplete — pipeline phases missing"

        all_issues = []
        for dim in dimensions:
            for issue in dim.issues:
                issue["dimension"] = dim.name
                all_issues.append(issue)

        critical = [i for i in all_issues if i["severity"] == "critical"]
        warnings = [i for i in all_issues if i["severity"] == "warning"]

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall,
            "grade": grade,
            "verdict": verdict,
            "critical_issues": len(critical),
            "warnings": len(warnings),
            "dimensions": [d.to_dict() for d in dimensions],
            "all_issues": all_issues,
            "cost_summary": self._build_cost_summary(),
        }

        return report

    # ── PRISMA 2020 Compliance ────────────────────────

    def _check_prisma_compliance(self) -> ReadinessDimension:
        dim = ReadinessDimension("PRISMA 2020 Compliance", weight=0.25)

        # Check PICO defined
        pico = self._load_json("input/pico.yaml")
        dim.add_check(
            "PICO components defined",
            pico is not None and bool(pico.get("population")),
            severity="critical",
            detail="pico.yaml must define P, I, C, O",
        )

        # Check search strategy documented
        strategy = self._load_json("phase1_strategy/search_strategy.json")
        dim.add_check(
            "Search strategy documented",
            strategy is not None,
            severity="critical",
        )

        # Check screening documented
        screening = self._load_json("phase3_screening/screening_results.json")
        dim.add_check(
            "Title/abstract screening completed",
            screening is not None,
            severity="critical",
        )

        # Check PRISMA flow data available
        has_flow = (
            self._file_exists("phase2_search/deduplicated_studies.json") and
            self._file_exists("phase3_screening/screening_results.json")
        )
        dim.add_check(
            "PRISMA flow diagram data available",
            has_flow,
            severity="critical",
            detail="Need search + screening counts for PRISMA diagram",
        )

        # Prescreen rescue documented
        prescreen = self._load_json("phase2_5_prescreen/prescreen_results.json")
        dim.add_check(
            "Pre-screening documented",
            prescreen is not None,
            severity="warning",
            detail="PRISMA 2020 recommends documenting automation in screening",
        )

        # Extraction completed
        extracted = self._load_json("phase4_extraction/extracted_data.json")
        dim.add_check(
            "Data extraction completed",
            extracted is not None and isinstance(extracted, list) and len(extracted) > 0,
            severity="critical",
        )

        # Registration / protocol
        dim.add_check(
            "Protocol/registration number documented",
            pico is not None and bool(pico.get("registration")),
            severity="warning",
            detail="PRISMA item 24a: protocol registration recommended",
        )

        return dim

    # ── Data Completeness ─────────────────────────────

    def _check_data_completeness(self) -> ReadinessDimension:
        dim = ReadinessDimension("Data Completeness", weight=0.20)

        extracted = self._load_json("phase4_extraction/extracted_data.json")
        if not extracted or not isinstance(extracted, list):
            dim.add_check("Extracted data exists", False, severity="critical")
            return dim

        dim.add_check("Extracted data exists", True)

        k = len(extracted)
        dim.add_check(
            f"Sufficient studies (k={k})",
            k >= 3,
            severity="critical",
            detail=f"Only {k} studies; meta-analysis typically needs k>=3",
        )

        # Check required fields
        required_fields = ["study_id", "title", "year"]
        for field in required_fields:
            n_with = sum(1 for s in extracted if s.get(field))
            dim.add_check(
                f"Field '{field}' present ({n_with}/{k})",
                n_with == k,
                severity="warning",
            )

        # Evidence spans
        n_evidence = sum(
            1 for s in extracted
            if s.get("evidence_spans") or s.get("evidence_text")
        )
        dim.add_check(
            f"Evidence spans present ({n_evidence}/{k})",
            n_evidence >= k * 0.8,
            severity="warning",
            detail="Claim-grounded extraction requires evidence spans",
        )

        # Effect sizes
        n_effect = sum(
            1 for s in extracted
            if s.get("effect_size") is not None or s.get("mean_diff") is not None
        )
        dim.add_check(
            f"Effect sizes extractable ({n_effect}/{k})",
            n_effect >= k * 0.7,
            severity="critical" if n_effect < k * 0.5 else "warning",
        )

        return dim

    # ── Statistical Rigor ─────────────────────────────

    def _check_statistical_rigor(self) -> ReadinessDimension:
        dim = ReadinessDimension("Statistical Rigor", weight=0.20)

        stats = self._load_json("phase5_analysis/statistical_results.json")
        if not stats:
            dim.add_check("Statistical analysis completed", False, severity="critical")
            return dim

        dim.add_check("Statistical analysis completed", True)

        # Core results
        ma = stats.get("meta_analysis", stats)
        dim.add_check(
            "Overall effect estimate present",
            ma.get("overall_effect") is not None or ma.get("pooled_estimate") is not None,
            severity="critical",
        )

        dim.add_check(
            "Heterogeneity reported (I2/tau2)",
            ma.get("I2") is not None or ma.get("heterogeneity", {}).get("I2") is not None,
            severity="critical",
        )

        dim.add_check(
            "Confidence interval reported",
            ma.get("ci_lower") is not None or ma.get("CI") is not None,
            severity="critical",
        )

        # Sensitivity analyses
        dim.add_check(
            "Leave-one-out analysis performed",
            stats.get("leave_one_out") is not None or
            stats.get("sensitivity", {}).get("leave_one_out") is not None,
            severity="warning",
        )

        # Publication bias
        dim.add_check(
            "Publication bias assessed",
            stats.get("publication_bias") is not None or
            stats.get("egger_test") is not None,
            severity="warning",
        )

        # HKSJ adjustment
        dim.add_check(
            "Hartung-Knapp adjustment applied",
            ma.get("hksj") is not None or ma.get("hartung_knapp") is not None,
            severity="warning",
            detail="HKSJ recommended when k < 20",
        )

        # Subgroup analyses
        dim.add_check(
            "Subgroup analyses conducted",
            stats.get("subgroup_analyses") is not None,
            severity="info",
        )

        # Meta-regression
        dim.add_check(
            "Meta-regression performed",
            stats.get("meta_regression") is not None,
            severity="info",
        )

        return dim

    # ── Quality Assessment (RoB-2 + GRADE) ──────────

    def _check_quality_assessment(self) -> ReadinessDimension:
        dim = ReadinessDimension("Quality Assessment (RoB-2/GRADE)", weight=0.15)

        # RoB-2
        rob2 = self._load_json("quality_assessment/rob2_summary.json")
        dim.add_check(
            "RoB-2 assessment completed",
            rob2 is not None,
            severity="warning",
            detail="Cochrane RoB-2 risk of bias assessment",
        )

        if rob2:
            overall = rob2.get("overall_counts", {})
            total = sum(overall.values())
            high = overall.get("High risk", 0)
            dim.add_check(
                "RoB-2 judgments recorded for all studies",
                total > 0,
                severity="warning",
            )
            if total > 0:
                dim.add_check(
                    f"Majority not high risk ({high}/{total})",
                    high < total * 0.5,
                    severity="info",
                    detail=f"{high}/{total} high risk studies",
                )

        # GRADE
        grade = self._load_json("quality_assessment/grade_assessments.json")
        dim.add_check(
            "GRADE evidence certainty assessed",
            grade is not None and isinstance(grade, list) and len(grade) > 0,
            severity="warning",
            detail="GRADE framework for rating certainty of evidence",
        )

        if grade and isinstance(grade, list):
            for g in grade:
                certainty = g.get("final_certainty", "")
                dim.add_check(
                    f"GRADE '{g.get('outcome', '?')}' assessed",
                    certainty in ("High", "Moderate", "Low", "Very low"),
                    severity="info",
                )

        # GRADE evidence profile
        profile = self._load_json("quality_assessment/grade_evidence_profile.json")
        dim.add_check(
            "GRADE Summary of Findings table generated",
            profile is not None,
            severity="warning",
        )

        return dim

    # ── Citation Grounding ────────────────────────────

    def _check_citation_grounding(self) -> ReadinessDimension:
        dim = ReadinessDimension("Citation Grounding", weight=0.15)

        drafts_dir = self.data_dir / "phase6_manuscript" / "drafts"
        if not drafts_dir.exists():
            dim.add_check("Manuscript sections exist", False, severity="critical")
            return dim

        sections = ["introduction", "methods", "results", "discussion", "conclusion"]
        for section in sections:
            md_path = drafts_dir / f"{section}.md"
            dim.add_check(
                f"Section '{section}' written",
                md_path.exists(),
                severity="critical" if section in ("methods", "results") else "warning",
            )

            # Check for unresolved citations
            if md_path.exists():
                text = md_path.read_text(encoding="utf-8")
                unresolved = text.count("[CITATION NEEDED")
                raw_markers = text.count("[REF:")
                dim.add_check(
                    f"'{section}': no unresolved citations",
                    unresolved == 0 and raw_markers == 0,
                    severity="warning",
                    detail=f"{unresolved} CITATION NEEDED, {raw_markers} unresolved REF markers",
                )

            # Check verification report
            verify_path = drafts_dir / f"{section}_verification.json"
            if verify_path.exists():
                verify = json.loads(verify_path.read_text(encoding="utf-8"))
                rate = verify.get("summary", {}).get("verification_rate", 0)
                dim.add_check(
                    f"'{section}': assertion verification rate >= 80%",
                    rate >= 0.8,
                    severity="warning" if rate >= 0.5 else "critical",
                    detail=f"Verification rate: {rate:.0%}",
                )

        return dim

    # ── Manuscript Quality ────────────────────────────

    def _check_manuscript_quality(self) -> ReadinessDimension:
        dim = ReadinessDimension("Manuscript Quality", weight=0.10)

        drafts_dir = self.data_dir / "phase6_manuscript" / "drafts"
        if not drafts_dir.exists():
            dim.add_check("Manuscript exists", False, severity="critical")
            return dim

        total_chars = 0
        for section in ["introduction", "methods", "results", "discussion", "conclusion"]:
            md_path = drafts_dir / f"{section}.md"
            if md_path.exists():
                text = md_path.read_text(encoding="utf-8")
                total_chars += len(text)

                # Minimum length checks
                min_chars = {
                    "introduction": 1000, "methods": 2000,
                    "results": 1500, "discussion": 1500,
                    "conclusion": 400,
                }
                expected = min_chars.get(section, 500)
                dim.add_check(
                    f"'{section}' length adequate (>{expected} chars)",
                    len(text) >= expected,
                    severity="warning",
                    detail=f"Actual: {len(text)} chars",
                )

        dim.add_check(
            "Total manuscript length adequate (>6000 chars)",
            total_chars >= 6000,
            severity="warning",
            detail=f"Total: {total_chars} chars",
        )

        # Check visualizations
        viz_dir = self.data_dir / "phase5_analysis" / "figures"
        if viz_dir.exists():
            fig_types = ["forest", "funnel"]
            for fig in fig_types:
                has_fig = any(viz_dir.glob(f"*{fig}*"))
                dim.add_check(
                    f"{fig.title()} plot generated",
                    has_fig,
                    severity="warning",
                )

        return dim

    # ── Transparency Audit ────────────────────────────

    def _check_transparency_audit(self) -> ReadinessDimension:
        dim = ReadinessDimension("Transparency & Reproducibility", weight=0.10)

        # Prompt audit log
        audit_path = self.data_dir / ".audit" / "prompt_log.jsonl"
        dim.add_check(
            "Prompt audit trail exists",
            audit_path.exists(),
            severity="warning",
            detail="All LLM calls should be logged for reproducibility",
        )

        if audit_path.exists():
            n_entries = sum(1 for _ in open(audit_path, "r"))
            dim.add_check(
                f"Audit log has entries ({n_entries})",
                n_entries > 0,
                severity="warning",
            )

        # Model version pinning
        import yaml
        models_path = Path("config/models.yaml")
        if models_path.exists():
            with open(models_path, "r") as f:
                models = yaml.safe_load(f)
            pinned = sum(
                1 for m in models.get("models", {}).values()
                if m.get("pinned_at")
            )
            total = len(models.get("models", {}))
            dim.add_check(
                f"Model versions pinned ({pinned}/{total})",
                pinned == total,
                severity="warning",
                detail="Pin model versions for reproducibility",
            )

        # Cache/checkpoint data
        dim.add_check(
            "Screening checkpoints saved",
            self._file_exists("phase3_screening/screening_results.json"),
            severity="info",
        )

        # Cost tracking
        cost = self._build_cost_summary()
        dim.add_check(
            "Cost tracking data available",
            cost.get("total_cost_usd", 0) > 0 or audit_path.exists(),
            severity="warning",
        )

        return dim

    # ── Cost Summary Builder ──────────────────────────

    def _build_cost_summary(self) -> dict:
        """Aggregate costs from audit log across all phases."""
        audit_path = self.data_dir / ".audit" / "prompt_log.jsonl"
        if not audit_path.exists():
            return {"total_cost_usd": 0, "phases": {}, "models": {}}

        phase_costs: Dict[str, float] = {}
        model_costs: Dict[str, float] = {}
        total_input = 0
        total_output = 0
        total_cost = 0.0
        n_calls = 0

        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    cost = entry.get("estimated_cost_usd", 0)
                    role = entry.get("role", "unknown")
                    model = entry.get("actual_model", entry.get("model_id", "unknown"))
                    inp = entry.get("input_tokens", 0)
                    out = entry.get("output_tokens", 0)

                    # Map role -> phase
                    phase = _role_to_phase(role)
                    phase_costs[phase] = phase_costs.get(phase, 0) + cost
                    model_costs[model] = model_costs.get(model, 0) + cost
                    total_input += inp
                    total_output += out
                    total_cost += cost
                    n_calls += 1
        except Exception as e:
            logger.warning(f"Error reading audit log: {e}")

        return {
            "total_cost_usd": round(total_cost, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_api_calls": n_calls,
            "cost_per_phase": {k: round(v, 4) for k, v in sorted(phase_costs.items())},
            "cost_per_model": {k: round(v, 4) for k, v in sorted(
                model_costs.items(), key=lambda x: x[1], reverse=True
            )},
        }

    # ── Helpers ───────────────────────────────────────

    def _load_json(self, relative_path: str) -> Optional[Any]:
        path = self.data_dir / relative_path
        if not path.exists():
            # Try .yaml as well
            if relative_path.endswith(".yaml"):
                import yaml
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return yaml.safe_load(f)
                except Exception:
                    return None
            return None

        try:
            if path.suffix in (".yaml", ".yml"):
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _file_exists(self, relative_path: str) -> bool:
        return (self.data_dir / relative_path).exists()


# ======================================================================
# Role -> Phase Mapping (centralized in phase_mapping.py)
# ======================================================================

def _role_to_phase(role: str) -> str:
    from src.utils.phase_mapping import role_to_phase
    return role_to_phase(role, display=False)


# ======================================================================
# Report Formatting
# ======================================================================

def format_readiness_report(report: dict) -> str:
    """Format readiness report as human-readable text."""
    lines = [
        "=" * 60,
        "  LUMEN v2 — Publication Readiness Report",
        "=" * 60,
        "",
        f"  Overall Score: {report['overall_score']}/100  (Grade: {report['grade']})",
        f"  Verdict: {report['verdict']}",
        f"  Critical Issues: {report['critical_issues']}",
        f"  Warnings: {report['warnings']}",
        "",
    ]

    # Per-dimension scores
    lines.append("  Dimension Scores:")
    lines.append("  " + "-" * 50)
    for dim in report["dimensions"]:
        bar_len = int(dim["score"] / 5)
        bar = "#" * bar_len + "." * (20 - bar_len)
        lines.append(
            f"  {dim['name']:<35} {dim['score']:>5.1f}/100  [{bar}]"
        )
    lines.append("")

    # Cost summary
    cost = report.get("cost_summary", {})
    if cost.get("total_cost_usd", 0) > 0:
        lines.append("  Cost Summary:")
        lines.append("  " + "-" * 50)
        lines.append(f"  Total API Cost:     ${cost['total_cost_usd']:.4f}")
        lines.append(f"  Total API Calls:    {cost.get('total_api_calls', 0)}")
        lines.append(f"  Input Tokens:       {cost.get('total_input_tokens', 0):,}")
        lines.append(f"  Output Tokens:      {cost.get('total_output_tokens', 0):,}")
        lines.append("")

        if cost.get("cost_per_phase"):
            lines.append("  Cost by Phase:")
            for phase, c in cost["cost_per_phase"].items():
                lines.append(f"    {phase:<20} ${c:.4f}")
            lines.append("")

        if cost.get("cost_per_model"):
            lines.append("  Cost by Model:")
            for model, c in cost["cost_per_model"].items():
                lines.append(f"    {model:<40} ${c:.4f}")
            lines.append("")

    # Issues
    if report["all_issues"]:
        lines.append("  Issues to Address:")
        lines.append("  " + "-" * 50)

        for issue in sorted(report["all_issues"],
                            key=lambda x: {"critical": 0, "warning": 1, "info": 2}.get(
                                x["severity"], 3)):
            icon = {"critical": "[!]", "warning": "[?]", "info": "[-]"}.get(
                issue["severity"], "[ ]"
            )
            lines.append(f"  {icon} [{issue['dimension']}] {issue['check']}")
            if issue.get("detail"):
                lines.append(f"      {issue['detail']}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
