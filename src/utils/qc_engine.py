"""
QC Checkpoint Engine — LUMEN v2
=================================
Automated quality control layer that runs after each pipeline phase.
Detects issues, flags them, and optionally pauses for human review.

Dual-mode:
  - Claude Code mode: interactive Q&A in conversation
  - Standalone mode: generates QC report + pauses for input()

Cumulative knowledge: learns from past corrections via qc_knowledge.yaml

Usage:
    from src.utils.qc_engine import QCEngine
    qc = QCEngine(dm, pico)
    issues = qc.run_phase_qc("phase3_screening")
    qc.present_issues(issues)  # interactive or report
"""

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.file_handlers import DataManager

logger = logging.getLogger(__name__)

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class QCIssue:
    """A single QC issue found during checkpoint."""

    SEVERITY_CRITICAL = "critical"   # Must fix before proceeding
    SEVERITY_WARNING = "warning"     # Should review, may proceed
    SEVERITY_INFO = "info"           # FYI, no action needed

    def __init__(self, phase: str, category: str, severity: str,
                 message: str, details: dict = None, suggestion: str = ""):
        self.phase = phase
        self.category = category
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion

    def to_dict(self):
        return {
            "phase": self.phase,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
        }

    def __repr__(self):
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}[self.severity]
        return f"{icon} [{self.category}] {self.message}"


class QCKnowledgeBase:
    """Cumulative knowledge from past QC corrections."""

    def __init__(self, project_root: str = "."):
        self.path = Path(project_root) / ".lumen" / "qc_knowledge.yaml"
        self.rules = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {"known_pitfalls": {}, "corrections_log": []}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.dump(self.rules, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)

    def add_correction(self, phase: str, category: str, description: str,
                       pattern: str = "", domain: str = ""):
        """Log a correction for future learning."""
        entry = {
            "phase": phase,
            "category": category,
            "description": description,
            "pattern": pattern,
            "domain": domain,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        self.rules.setdefault("corrections_log", []).append(entry)
        self.save()
        logger.info(f"QC knowledge: logged correction — {description}")

    def get_rules_for_phase(self, phase: str) -> list:
        """Get known pitfall rules for a specific phase."""
        return [
            r for r in self.rules.get("known_pitfalls", {}).get(phase, [])
        ]


class QCEngine:
    """Main QC engine — runs checks after each pipeline phase."""

    def __init__(self, dm: DataManager, pico: dict = None):
        self.dm = dm
        self.pico = pico or {}
        self.kb = QCKnowledgeBase()
        self.interactive = self._detect_interactive_mode()

    @staticmethod
    def _detect_interactive_mode() -> bool:
        """Detect if running inside Claude Code or interactive terminal."""
        # Claude Code sets specific env vars; also check if stdin is a TTY
        if os.environ.get("CLAUDE_CODE"):
            return True
        return sys.stdin.isatty()

    def run_all_checks(self, phase: str) -> List[QCIssue]:
        """Run all QC checks for a given phase."""
        checkers = {
            "phase2_5_prescreen": self._qc_prescreen,
            "phase3_screening": self._qc_screening,
            "phase3_3_fulltext": self._qc_fulltext,
            "phase4_extraction": self._qc_extraction,
            "phase5_nma": self._qc_nma,
        }
        checker = checkers.get(phase)
        if not checker:
            logger.warning(f"No QC checker for phase: {phase}")
            return []
        return checker()

    # ==================================================================
    # Phase-specific QC checks
    # ==================================================================

    def _qc_prescreen(self) -> List[QCIssue]:
        """QC after Phase 2.5/3.0 pre-screening."""
        issues = []

        excluded_path = self.dm.base_dir / "phase2_search" / "prescreened" / "prescreen_excluded.json"
        if not excluded_path.exists():
            return issues

        excluded = json.loads(excluded_path.read_text(encoding="utf-8"))

        # Check 1: RCT-looking studies excluded by keyword
        rct_signals = ["randomized", "randomised", "placebo-controlled",
                       "double-blind", "open-label", "pilot", "clinical trial"]
        rct_excluded = []
        for s in excluded:
            if s.get("exclusion_reason") != "keyword_exclusion":
                continue
            title = s.get("title", "").lower()
            if any(sig in title for sig in rct_signals):
                rct_excluded.append(s)

        if rct_excluded:
            issues.append(QCIssue(
                phase="prescreen",
                category="false_exclusion",
                severity=QCIssue.SEVERITY_WARNING,
                message=f"{len(rct_excluded)} studies with RCT-like titles were "
                        f"excluded by keyword matching",
                details={"studies": [
                    {"pmid": s.get("pmid", "?"),
                     "title": s.get("title", "")[:120],
                     "reason": s.get("exclusion_reason")}
                    for s in rct_excluded[:10]
                ]},
                suggestion="Review these — they may be primary RCTs that mention "
                           "'animal model' or 'review' in abstract background only",
            ))

        # Check 2: no_abstract exclusion count
        no_abs = [s for s in excluded if s.get("exclusion_reason") == "no_abstract"]
        if len(no_abs) > len(excluded) * 0.2:
            issues.append(QCIssue(
                phase="prescreen",
                category="high_no_abstract",
                severity=QCIssue.SEVERITY_INFO,
                message=f"{len(no_abs)}/{len(excluded)} excluded for no abstract "
                        f"({len(no_abs)/len(excluded)*100:.0f}%)",
                suggestion="Consider if exclude_no_abstract is too aggressive for this domain",
            ))

        return issues

    def _qc_screening(self) -> List[QCIssue]:
        """QC after Phase 3.1 dual screening."""
        issues = []

        results_path = self.dm.base_dir / "phase3_screening" / "stage1_title_abstract" / "screening_results.json"
        if not results_path.exists():
            return issues

        data = json.loads(results_path.read_text(encoding="utf-8"))
        stats = data.get("stats", {})

        # Check 1: Very low inclusion rate
        total = stats.get("total_screened", 0)
        included = len([r for r in data.get("results", [])
                        if r.get("final_decision") == "include"])
        if total > 0 and included / total < 0.02:
            issues.append(QCIssue(
                phase="screening",
                category="low_inclusion_rate",
                severity=QCIssue.SEVERITY_WARNING,
                message=f"Very low inclusion rate: {included}/{total} "
                        f"({included/total*100:.1f}%)",
                suggestion="Check if PICO criteria or screening prompt is too strict",
            ))

        # Check 2: Low kappa
        kappa = stats.get("cohens_kappa", 1.0)
        if kappa < 0.6:
            issues.append(QCIssue(
                phase="screening",
                category="low_agreement",
                severity=QCIssue.SEVERITY_WARNING,
                message=f"Low inter-screener agreement: κ={kappa:.3f}",
                suggestion="Screening criteria may be ambiguous — review exclusion rules",
            ))

        return issues

    def _qc_fulltext(self) -> List[QCIssue]:
        """QC after Phase 3.3 full-text screening."""
        issues = []

        results_path = self.dm.base_dir / "phase3_screening" / "stage2_fulltext" / "fulltext_screening_results.json"
        if not results_path.exists():
            return issues

        results = json.loads(results_path.read_text(encoding="utf-8"))

        # Check 1: PDF ↔ title sanity check
        rct_title_signals = ["randomized", "randomised", "double-blind",
                             "placebo-controlled", "open-label", "clinical trial",
                             "pilot"]
        non_rct_reasons = ["animal", "rat", "mouse", "review", "narrative",
                           "scoping", "meta-analysis"]

        mismatches = []
        for r in results:
            if r.get("confidence_score", 5) >= 3:
                continue  # Only check excluded
            title = r.get("title", "").lower()
            reasoning = r.get("reasoning", "").lower()
            title_is_rct = any(sig in title for sig in rct_title_signals)
            reason_is_nonrct = any(nr in reasoning for nr in non_rct_reasons)

            if title_is_rct and reason_is_nonrct:
                mismatches.append(r)

        if mismatches:
            issues.append(QCIssue(
                phase="fulltext",
                category="pdf_title_mismatch",
                severity=QCIssue.SEVERITY_CRITICAL,
                message=f"{len(mismatches)} studies have RCT-like titles but "
                        f"were excluded as non-RCT/review/animal — possible "
                        f"wrong PDF downloaded",
                details={"studies": [
                    {"study_id": r.get("study_id", "?"),
                     "title": r.get("title", "")[:100],
                     "reason": r.get("reasoning", "")[:100]}
                    for r in mismatches
                ]},
                suggestion="Verify PDF files match the metadata. Re-download "
                           "if PDF content doesn't match the expected article.",
            ))

        # Check 2: Borderline exclusions (score=2)
        borderline = [r for r in results if r.get("confidence_score") == 2]
        if borderline:
            issues.append(QCIssue(
                phase="fulltext",
                category="borderline_exclusions",
                severity=QCIssue.SEVERITY_WARNING,
                message=f"{len(borderline)} studies excluded with borderline "
                        f"score (likely_exclude) — consider manual review",
                details={"studies": [
                    {"study_id": r.get("study_id", "?"),
                     "title": r.get("title", "")[:100],
                     "reason": r.get("reasoning", "")[:80]}
                    for r in borderline
                ]},
                suggestion="These are not clearly excluded — review reasoning "
                           "and check if they have relevant outcomes.",
            ))

        return issues

    def _qc_extraction(self) -> List[QCIssue]:
        """QC after Phase 4 extraction."""
        issues = []

        extracted_path = self.dm.base_dir / "phase4_extraction" / "extracted_data.json"
        if not extracted_path.exists():
            return issues

        extracted = json.loads(extracted_path.read_text(encoding="utf-8"))

        # Check 1: PICO outcome coverage
        pico_outcomes = []
        outcome_pico = self.pico.get("outcome", {})
        if isinstance(outcome_pico, dict):
            if outcome_pico.get("primary"):
                pico_outcomes.append(("primary", str(outcome_pico["primary"])[:60]))
            for sec in outcome_pico.get("secondary", []):
                pico_outcomes.append(("secondary", str(sec)[:60]))

        if pico_outcomes:
            all_extracted_measures = set()
            for s in extracted:
                for arm in s.get("arms", []):
                    for o in arm.get("outcomes", []):
                        m = o.get("measure", "")
                        if m:
                            all_extracted_measures.add(m.lower())
                for o in s.get("outcomes", []):
                    m = o.get("measure", "")
                    if m:
                        all_extracted_measures.add(m.lower())

            missing = []
            for level, desc in pico_outcomes:
                # Fuzzy check: any extracted measure contains key words from PICO outcome
                keywords = [w.lower() for w in desc.split() if len(w) > 3]
                found = any(
                    any(kw in em for kw in keywords[:3])
                    for em in all_extracted_measures
                )
                if not found:
                    missing.append(f"[{level}] {desc}")

            if missing:
                issues.append(QCIssue(
                    phase="extraction",
                    category="pico_outcome_gap",
                    severity=QCIssue.SEVERITY_CRITICAL,
                    message=f"{len(missing)} PICO-defined outcomes not found "
                            f"in extracted data",
                    details={"missing_outcomes": missing,
                             "extracted_measures": sorted(all_extracted_measures)[:20]},
                    suggestion="Check if studies actually report these outcomes. "
                               "If not, this is a domain limitation, not a pipeline bug.",
                ))

        # Check 2: SE/SD corrections applied
        contrasts_path = self.dm.base_dir / "phase4_extraction" / "nma_contrasts.json"
        if contrasts_path.exists():
            contrasts = json.loads(contrasts_path.read_text(encoding="utf-8"))
            corrected = [c for c in contrasts if c.get("corrections")]
            if corrected:
                issues.append(QCIssue(
                    phase="extraction",
                    category="se_sd_corrections",
                    severity=QCIssue.SEVERITY_WARNING,
                    message=f"{len(corrected)} contrasts had SE/SD auto-correction "
                            f"applied — verify these are correct",
                    details={"corrections": [
                        {"study": c["studlab"],
                         "outcome": c.get("outcome", ""),
                         "TE": c["TE"],
                         "msgs": c["corrections"][:2]}
                        for c in corrected[:5]
                    ]},
                    suggestion="Check original papers to confirm whether "
                               "reported values are SD or SE.",
                ))

        # Check 3: Extreme SMD values
        if contrasts_path.exists():
            extreme = [c for c in contrasts
                       if c.get("effect_measure") == "SMD" and abs(c.get("TE", 0)) > 3.0]
            if extreme:
                issues.append(QCIssue(
                    phase="extraction",
                    category="extreme_smd",
                    severity=QCIssue.SEVERITY_CRITICAL,
                    message=f"{len(extreme)} contrasts have |SMD| > 3.0 — likely "
                            f"data error or SE/SD confusion",
                    details={"contrasts": [
                        {"study": c["studlab"], "outcome": c.get("outcome", ""),
                         "TE": round(c["TE"], 3)}
                        for c in extreme
                    ]},
                    suggestion="SMD > 3 is almost always a data error. Check "
                               "original paper for these studies.",
                ))

        # Check 4: Duplicate study IDs
        ids = [s.get("study_id") for s in extracted]
        from collections import Counter
        dupes = {k: v for k, v in Counter(ids).items() if v > 1}
        if dupes:
            issues.append(QCIssue(
                phase="extraction",
                category="duplicate_studies",
                severity=QCIssue.SEVERITY_WARNING,
                message=f"{len(dupes)} duplicate study IDs in extracted data",
                details={"duplicates": dupes},
                suggestion="Remove duplicates before running NMA.",
            ))

        return issues

    def _qc_nma(self) -> List[QCIssue]:
        """QC after Phase 5 NMA."""
        issues = []

        results_path = self.dm.base_dir / "phase5_analysis" / "nma_results.json"
        if not results_path.exists():
            return issues

        results = json.loads(results_path.read_text(encoding="utf-8"))
        per_outcome = results.get("per_outcome", results.get("nma", {}).get("per_outcome", {}))

        # Check 1: High I²
        for outcome, r in per_outcome.items():
            i2 = r.get("I2")
            if i2 is not None and i2 != "NA" and float(i2) > 75:
                issues.append(QCIssue(
                    phase="nma",
                    category="high_heterogeneity",
                    severity=QCIssue.SEVERITY_WARNING,
                    message=f"'{outcome}': I²={i2}% — substantial heterogeneity",
                    details={"outcome": outcome, "I2": i2,
                             "tau2": r.get("tau2"), "k": r.get("n_studies")},
                    suggestion="Consider subgroup analysis by drug type or duration, "
                               "or sensitivity analysis excluding outliers.",
                ))

        # Check 2: Failed outcomes
        skipped = results.get("outcomes_skipped", [])
        if skipped:
            issues.append(QCIssue(
                phase="nma",
                category="failed_outcomes",
                severity=QCIssue.SEVERITY_INFO,
                message=f"{len(skipped)} outcomes failed NMA "
                        f"(disconnected network or too few treatments)",
                details={"skipped": skipped},
                suggestion="Consider pairwise MA for these outcomes instead of NMA.",
            ))

        # Check 3: Inconsistency detected
        for outcome, r in per_outcome.items():
            cons = r.get("consistency", {})
            if cons.get("inconsistency_detected"):
                issues.append(QCIssue(
                    phase="nma",
                    category="inconsistency",
                    severity=QCIssue.SEVERITY_WARNING,
                    message=f"'{outcome}': significant inconsistency detected "
                            f"(p={cons.get('pval_between', '?')})",
                    details={"outcome": outcome, "consistency": cons},
                    suggestion="Check node-splitting results. Consider if "
                               "transitivity assumption holds for this outcome.",
                ))

        return issues

    # ==================================================================
    # Presentation
    # ==================================================================

    def present_issues(self, issues: List[QCIssue], phase: str = ""):
        """Present QC issues — interactive or report mode."""
        if not issues:
            print(f"\n  {GREEN}✓ QC checkpoint passed — no issues found.{RESET}")
            return

        # Group by severity
        critical = [i for i in issues if i.severity == QCIssue.SEVERITY_CRITICAL]
        warnings = [i for i in issues if i.severity == QCIssue.SEVERITY_WARNING]
        info = [i for i in issues if i.severity == QCIssue.SEVERITY_INFO]

        phase_label = phase or issues[0].phase
        print(f"\n{'=' * 60}")
        print(f"  {BOLD}QC Checkpoint: {phase_label}{RESET}")
        print(f"{'=' * 60}")
        print(f"  {RED}{len(critical)} critical{RESET}  "
              f"{YELLOW}{len(warnings)} warnings{RESET}  "
              f"{DIM}{len(info)} info{RESET}")

        for issue in issues:
            icon = {"critical": f"{RED}✖", "warning": f"{YELLOW}⚠",
                    "info": f"{DIM}ℹ"}[issue.severity]
            print(f"\n  {icon} [{issue.category}]{RESET} {issue.message}")
            if issue.details:
                # Show first few items
                for key, val in issue.details.items():
                    if isinstance(val, list) and val:
                        for item in val[:3]:
                            if isinstance(item, dict):
                                summary = " | ".join(f"{k}={v}" for k, v in
                                                     list(item.items())[:3])
                                print(f"    {DIM}→ {summary}{RESET}")
                            else:
                                print(f"    {DIM}→ {item}{RESET}")
                        if len(val) > 3:
                            print(f"    {DIM}  ... and {len(val)-3} more{RESET}")
            if issue.suggestion:
                print(f"    {CYAN}💡 {issue.suggestion}{RESET}")

        print()

        # Save QC report
        report = {
            "phase": phase_label,
            "n_critical": len(critical),
            "n_warning": len(warnings),
            "n_info": len(info),
            "issues": [i.to_dict() for i in issues],
        }
        qc_dir = Path(self.dm.base_dir) / "qc_reports"
        qc_dir.mkdir(parents=True, exist_ok=True)
        report_path = qc_dir / f"qc_{phase_label}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  {DIM}QC report saved: {report_path}{RESET}")

        # Interactive mode: pause on critical issues
        if critical and self.interactive:
            print(f"\n  {RED}{BOLD}⚠ {len(critical)} critical issues found. "
                  f"Review before proceeding.{RESET}")
            if sys.stdin.isatty():
                input(f"  {CYAN}Press Enter to continue...{RESET}")

    def run_and_present(self, phase: str) -> List[QCIssue]:
        """Convenience: run checks and present results."""
        issues = self.run_all_checks(phase)
        self.present_issues(issues, phase)
        return issues
