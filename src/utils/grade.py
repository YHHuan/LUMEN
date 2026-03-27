"""
GRADE Evidence Certainty Assessment — LUMEN v2
================================================
Implements the GRADE (Grading of Recommendations, Assessment,
Development and Evaluations) framework for rating certainty of evidence.

5 Downgrade Domains:
  1. Risk of bias (from RoB-2 assessments)
  2. Inconsistency (heterogeneity)
  3. Indirectness (population/intervention/outcome match)
  4. Imprecision (confidence interval width, sample size)
  5. Publication bias (funnel plot asymmetry, statistical tests)

3 Upgrade Domains (for observational studies):
  1. Large effect
  2. Dose-response gradient
  3. All plausible confounders would reduce effect

Certainty levels: High -> Moderate -> Low -> Very low

References:
  Guyatt et al. (2008) BMJ 336:924-926
  Schunemann et al. (2013) Cochrane Handbook Chapter 14
"""

import json
import logging
import math
from typing import Optional, List, Dict

from src.agents.base_agent import BaseAgent
from src.utils.cache import TokenBudget

logger = logging.getLogger(__name__)


# ======================================================================
# GRADE Certainty Levels
# ======================================================================

CERTAINTY_LEVELS = ["High", "Moderate", "Low", "Very low"]

DOWNGRADE_DOMAINS = [
    "risk_of_bias",
    "inconsistency",
    "indirectness",
    "imprecision",
    "publication_bias",
]

UPGRADE_DOMAINS = [
    "large_effect",
    "dose_response",
    "plausible_confounding",
]

VALID_RATINGS = {
    "no_concern": 0,       # No downgrade/upgrade
    "serious": -1,          # Downgrade by 1
    "very_serious": -2,     # Downgrade by 2
    "upgrade_1": 1,         # Upgrade by 1
    "upgrade_2": 2,         # Upgrade by 2
}


# ======================================================================
# GRADE Assessment Structure
# ======================================================================

def create_empty_grade(outcome_name: str, study_design: str = "RCT") -> dict:
    """Create an empty GRADE evidence profile for an outcome."""
    starting_level = 4 if study_design.upper() in ("RCT", "RANDOMIZED") else 2

    domains = {}
    for d in DOWNGRADE_DOMAINS:
        domains[d] = {
            "rating": "no_concern",
            "adjustment": 0,
            "justification": "",
        }
    for d in UPGRADE_DOMAINS:
        domains[d] = {
            "rating": "no_concern",
            "adjustment": 0,
            "justification": "",
        }

    return {
        "outcome": outcome_name,
        "study_design": study_design,
        "starting_level": starting_level,  # 4=High (RCT), 2=Low (obs)
        "domains": domains,
        "final_certainty": None,
        "footnotes": [],
    }


def compute_certainty(grade_assessment: dict) -> str:
    """Compute final certainty level from domain adjustments."""
    level = grade_assessment["starting_level"]

    for domain_name, domain in grade_assessment["domains"].items():
        level += domain["adjustment"]

    # Clamp to valid range
    level = max(1, min(4, level))

    certainty_map = {4: "High", 3: "Moderate", 2: "Low", 1: "Very low"}
    return certainty_map[level]


# ======================================================================
# Automated GRADE Assessor
# ======================================================================

class GRADEAssessor:
    """
    Semi-automated GRADE assessment using statistical results and RoB-2 data.

    Auto-populates what can be determined from data:
    - Risk of bias: from RoB-2 summary
    - Inconsistency: from I2, tau2, prediction interval
    - Imprecision: from CI width, sample size, OIS
    - Publication bias: from Egger test, trim-and-fill

    Indirectness requires LLM judgment (PICO match).
    """

    def __init__(self, llm_agent: Optional[BaseAgent] = None):
        self.llm_agent = llm_agent

    def assess_outcome(
        self,
        outcome_name: str,
        statistical_results: dict,
        rob2_summary: dict = None,
        pico: dict = None,
        n_studies: int = 0,
        study_design: str = "RCT",
    ) -> dict:
        """
        Perform GRADE assessment for a single outcome.

        Args:
            outcome_name: Name of the outcome being assessed
            statistical_results: Phase 5 statistical results
            rob2_summary: RoB-2 summary across studies
            pico: PICO components for indirectness check
            n_studies: Number of studies in the analysis
            study_design: "RCT" or "observational"
        """
        grade = create_empty_grade(outcome_name, study_design)

        # Auto-assess each domain
        self._assess_risk_of_bias(grade, rob2_summary, n_studies)
        self._assess_inconsistency(grade, statistical_results, n_studies)
        self._assess_imprecision(grade, statistical_results, n_studies)
        self._assess_publication_bias(grade, statistical_results)

        # Indirectness needs LLM or manual
        if self.llm_agent and pico:
            self._assess_indirectness_llm(grade, pico, statistical_results)

        grade["final_certainty"] = compute_certainty(grade)
        return grade

    def assess_all_outcomes(
        self,
        outcomes: List[str],
        statistical_results: dict,
        rob2_summary: dict = None,
        pico: dict = None,
        n_studies: int = 0,
        study_design: str = "RCT",
    ) -> List[dict]:
        """Assess multiple outcomes."""
        results = []
        for outcome in outcomes:
            grade = self.assess_outcome(
                outcome, statistical_results, rob2_summary,
                pico, n_studies, study_design,
            )
            results.append(grade)
        return results

    # ── Domain Assessors ──────────────────────────────

    def _assess_risk_of_bias(self, grade: dict, rob2_summary: dict,
                             n_studies: int):
        """Assess risk of bias from RoB-2 results."""
        domain = grade["domains"]["risk_of_bias"]

        if not rob2_summary:
            domain["rating"] = "no_concern"
            domain["justification"] = "RoB-2 assessment not available"
            return

        overall_counts = rob2_summary.get("overall_counts", {})
        high_risk = overall_counts.get("High risk", 0)
        some_concerns = overall_counts.get("Some concerns", 0)
        total = max(n_studies, sum(overall_counts.values()), 1)

        high_pct = high_risk / total
        concern_pct = (high_risk + some_concerns) / total

        if high_pct > 0.5:
            domain["rating"] = "very_serious"
            domain["adjustment"] = -2
        elif high_pct > 0.25 or concern_pct > 0.5:
            domain["rating"] = "serious"
            domain["adjustment"] = -1
        else:
            domain["rating"] = "no_concern"
            domain["adjustment"] = 0

        domain["justification"] = (
            f"{high_risk}/{total} high risk, "
            f"{some_concerns}/{total} some concerns"
        )

    def _assess_inconsistency(self, grade: dict, stats: dict,
                              n_studies: int):
        """Assess inconsistency from heterogeneity statistics."""
        domain = grade["domains"]["inconsistency"]

        ma = stats.get("meta_analysis", stats)
        i2 = ma.get("I2") or ma.get("heterogeneity", {}).get("I2")

        if i2 is None:
            domain["rating"] = "no_concern"
            domain["justification"] = "I2 not available"
            return

        # Convert string percentage if needed
        if isinstance(i2, str):
            i2 = float(i2.replace("%", ""))

        if i2 > 75:
            domain["rating"] = "very_serious"
            domain["adjustment"] = -2
            domain["justification"] = f"I2={i2:.1f}% (substantial heterogeneity)"
        elif i2 > 50:
            domain["rating"] = "serious"
            domain["adjustment"] = -1
            domain["justification"] = f"I2={i2:.1f}% (moderate heterogeneity)"
        else:
            domain["rating"] = "no_concern"
            domain["adjustment"] = 0
            domain["justification"] = f"I2={i2:.1f}% (low/moderate heterogeneity)"

        # Add footnote if prediction interval crosses null
        pi = ma.get("prediction_interval")
        if pi and len(pi) == 2:
            if pi[0] < 0 < pi[1]:
                domain["justification"] += "; prediction interval crosses null"
                if domain["adjustment"] == 0:
                    domain["rating"] = "serious"
                    domain["adjustment"] = -1
                    grade["footnotes"].append(
                        "Downgraded for inconsistency: prediction interval crosses null"
                    )

    def _assess_imprecision(self, grade: dict, stats: dict,
                            n_studies: int):
        """Assess imprecision from CI width and sample size."""
        domain = grade["domains"]["imprecision"]

        ma = stats.get("meta_analysis", stats)

        ci_lower = ma.get("ci_lower") or (ma.get("CI", [None, None])[0] if isinstance(ma.get("CI"), list) else None)
        ci_upper = ma.get("ci_upper") or (ma.get("CI", [None, None])[1] if isinstance(ma.get("CI"), list) else None)

        total_n = ma.get("total_n") or ma.get("total_sample_size", 0)

        reasons = []

        # Check CI width
        if ci_lower is not None and ci_upper is not None:
            ci_width = ci_upper - ci_lower

            # For SMD: CI width > 1.0 is wide
            if ci_width > 1.5:
                domain["adjustment"] = -2
                domain["rating"] = "very_serious"
                reasons.append(f"wide CI ({ci_lower:.2f} to {ci_upper:.2f})")
            elif ci_width > 1.0:
                domain["adjustment"] = -1
                domain["rating"] = "serious"
                reasons.append(f"moderately wide CI ({ci_lower:.2f} to {ci_upper:.2f})")

            # CI crosses null (for effect measures where 0 is null)
            if ci_lower < 0 < ci_upper:
                if domain["adjustment"] == 0:
                    domain["adjustment"] = -1
                    domain["rating"] = "serious"
                reasons.append("CI crosses null")

        # Optimal Information Size (OIS) check
        # Rule of thumb: total N < 400 for continuous, < 300 events for binary
        if total_n and total_n < 400:
            if domain["adjustment"] == 0:
                domain["adjustment"] = -1
                domain["rating"] = "serious"
            reasons.append(f"total N={total_n} below OIS threshold")

        # Few studies
        if n_studies < 3:
            if domain["adjustment"] == 0:
                domain["adjustment"] = -1
                domain["rating"] = "serious"
            reasons.append(f"few studies (k={n_studies})")

        domain["justification"] = "; ".join(reasons) if reasons else "Adequate precision"

    def _assess_publication_bias(self, grade: dict, stats: dict):
        """Assess publication bias from statistical tests."""
        domain = grade["domains"]["publication_bias"]

        pub_bias = stats.get("publication_bias", {})
        egger = pub_bias.get("egger_test", {})
        trim_fill = pub_bias.get("trim_and_fill", {})

        reasons = []

        # Egger test
        egger_p = egger.get("p_value") or egger.get("p")
        if egger_p is not None:
            if egger_p < 0.05:
                domain["adjustment"] = -1
                domain["rating"] = "serious"
                reasons.append(f"Egger test significant (p={egger_p:.3f})")
            elif egger_p < 0.10:
                reasons.append(f"Egger test marginally significant (p={egger_p:.3f})")

        # Trim and fill
        n_imputed = trim_fill.get("n_imputed") or trim_fill.get("studies_imputed", 0)
        if n_imputed and n_imputed > 0:
            if domain["adjustment"] == 0:
                domain["rating"] = "serious"
                domain["adjustment"] = -1
            reasons.append(f"Trim-and-fill imputed {n_imputed} studies")

        domain["justification"] = (
            "; ".join(reasons) if reasons else "No evidence of publication bias"
        )

    def _assess_indirectness_llm(self, grade: dict, pico: dict,
                                 stats: dict):
        """Use LLM to assess indirectness (PICO match)."""
        domain = grade["domains"]["indirectness"]

        if not self.llm_agent:
            return

        prompt = (
            "Assess GRADE indirectness for this systematic review.\n\n"
            f"Review PICO:\n"
            f"  Population: {pico.get('population', 'N/A')}\n"
            f"  Intervention: {pico.get('intervention', 'N/A')}\n"
            f"  Comparator: {pico.get('comparator', 'N/A')}\n"
            f"  Outcome: {pico.get('outcome', 'N/A')}\n\n"
            f"Number of included studies: {stats.get('meta_analysis', {}).get('k', 'N/A')}\n\n"
            "Assess whether the included studies directly address the review question "
            "in terms of population, intervention, comparator, and outcomes.\n\n"
            "Return JSON:\n"
            '{"rating": "no_concern"|"serious"|"very_serious", '
            '"justification": "brief reason"}'
        )

        result = self.llm_agent.call_llm(
            prompt=prompt,
            expect_json=True,
            cache_namespace="grade_indirectness",
            description="GRADE indirectness",
        )

        parsed = result.get("parsed", {})
        if parsed:
            rating = parsed.get("rating", "no_concern")
            if rating in ("serious", "very_serious"):
                domain["rating"] = rating
                domain["adjustment"] = VALID_RATINGS.get(rating, -1)
            domain["justification"] = parsed.get("justification", "")


# ======================================================================
# GRADE Summary / Evidence Profile Table
# ======================================================================

def build_grade_evidence_profile(
    grade_assessments: List[dict],
    statistical_results: dict,
) -> dict:
    """
    Build GRADE evidence profile (Summary of Findings table).

    Returns structured data suitable for rendering as SoF table.
    """
    rows = []

    for grade in grade_assessments:
        ma = statistical_results.get("meta_analysis", statistical_results)

        row = {
            "outcome": grade["outcome"],
            "n_studies": ma.get("k", "?"),
            "study_design": grade["study_design"],
            "risk_of_bias": grade["domains"]["risk_of_bias"]["rating"],
            "inconsistency": grade["domains"]["inconsistency"]["rating"],
            "indirectness": grade["domains"]["indirectness"]["rating"],
            "imprecision": grade["domains"]["imprecision"]["rating"],
            "publication_bias": grade["domains"]["publication_bias"]["rating"],
            "effect_estimate": ma.get("overall_effect") or ma.get("pooled_estimate"),
            "ci": [ma.get("ci_lower"), ma.get("ci_upper")],
            "certainty": grade["final_certainty"],
            "footnotes": grade.get("footnotes", []),
        }

        # Add domain justifications as footnotes
        for d_name, d_data in grade["domains"].items():
            if d_data["adjustment"] != 0 and d_data["justification"]:
                row["footnotes"].append(
                    f"{d_name}: {d_data['justification']}"
                )

        rows.append(row)

    return {
        "title": "GRADE Summary of Findings",
        "rows": rows,
    }


def format_grade_table(profile: dict) -> str:
    """Format GRADE evidence profile as a markdown table."""
    lines = [
        "## GRADE Summary of Findings",
        "",
        "| Outcome | Studies | RoB | Inconsistency | Indirectness | "
        "Imprecision | Pub Bias | Effect (95% CI) | Certainty |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    rating_icons = {
        "no_concern": "-",
        "serious": "(-1)",
        "very_serious": "(-2)",
    }

    certainty_icons = {
        "High": "++++ High",
        "Moderate": "+++ Moderate",
        "Low": "++ Low",
        "Very low": "+ Very low",
    }

    for row in profile["rows"]:
        effect = row.get("effect_estimate")
        ci = row.get("ci", [None, None])
        if effect is not None and ci[0] is not None:
            effect_str = f"{effect:.2f} ({ci[0]:.2f}, {ci[1]:.2f})"
        else:
            effect_str = "N/A"

        lines.append(
            f"| {row['outcome']} "
            f"| {row['n_studies']} "
            f"| {rating_icons.get(row['risk_of_bias'], '?')} "
            f"| {rating_icons.get(row['inconsistency'], '?')} "
            f"| {rating_icons.get(row['indirectness'], '?')} "
            f"| {rating_icons.get(row['imprecision'], '?')} "
            f"| {rating_icons.get(row['publication_bias'], '?')} "
            f"| {effect_str} "
            f"| {certainty_icons.get(row['certainty'], '?')} |"
        )

    # Footnotes
    all_footnotes = []
    for row in profile["rows"]:
        for fn in row.get("footnotes", []):
            if fn not in all_footnotes:
                all_footnotes.append(fn)

    if all_footnotes:
        lines.append("")
        lines.append("**Footnotes:**")
        for i, fn in enumerate(all_footnotes, 1):
            lines.append(f"{i}. {fn}")

    return "\n".join(lines)
