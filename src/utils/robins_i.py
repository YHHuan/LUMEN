"""
ROBINS-I Risk of Bias Assessment — LUMEN v2
=============================================
Implements the ROBINS-I tool for non-randomized studies of interventions.

7 Domains:
  D1: Bias due to confounding
  D2: Bias in selection of participants into the study
  D3: Bias in classification of interventions
  D4: Bias due to deviations from intended interventions
  D5: Bias due to missing data
  D6: Bias in measurement of outcomes
  D7: Bias in selection of the reported result

Each domain scored: "Low" | "Moderate" | "Serious" | "Critical" | "No information"
Overall judgment derived from domain scores.

References:
  Sterne et al. (2016) BMJ 355:i4919
  https://www.riskofbias.info/welcome/home/current-version-of-robins-i
"""

import json
import logging
from typing import Optional, List, Dict

from src.agents.base_agent import BaseAgent
from src.utils.cache import TokenBudget

logger = logging.getLogger(__name__)


# ======================================================================
# ROBINS-I Domain Definitions
# ======================================================================

ROBINS_I_DOMAINS = {
    "D1": {
        "name": "Bias due to confounding",
        "signaling_questions": [
            "Is there potential for confounding of the effect of intervention?",
            "Was the analysis based on splitting participants by intervention received?",
            "Were confounding domains measured validly and reliably?",
            "Did the authors control for confounding using appropriate methods?",
        ],
    },
    "D2": {
        "name": "Bias in selection of participants",
        "signaling_questions": [
            "Was selection into the study related to intervention and outcome?",
            "Was selection related to intervention and outcome (time-varying)?",
            "Do start of follow-up and start of intervention coincide for most?",
        ],
    },
    "D3": {
        "name": "Bias in classification of interventions",
        "signaling_questions": [
            "Were intervention groups clearly defined?",
            "Was the information used to define intervention recorded at start?",
            "Could classification of intervention status have been affected by outcome?",
        ],
    },
    "D4": {
        "name": "Bias due to deviations from intended interventions",
        "signaling_questions": [
            "Were there deviations from the intended intervention beyond what would be expected?",
            "Were important co-interventions balanced across intervention groups?",
            "Was an appropriate analysis used to estimate the effect of starting intervention?",
        ],
    },
    "D5": {
        "name": "Bias due to missing data",
        "signaling_questions": [
            "Were outcome data available for all or nearly all participants?",
            "Were participants excluded due to missing data on intervention status?",
            "Were participants excluded due to missing data on other variables?",
            "Are the proportions of missing data similar across interventions?",
        ],
    },
    "D6": {
        "name": "Bias in measurement of outcomes",
        "signaling_questions": [
            "Could the outcome measure have been influenced by knowledge of intervention?",
            "Were outcome assessors aware of the intervention received by participants?",
            "Were assessment methods comparable across intervention groups?",
        ],
    },
    "D7": {
        "name": "Bias in selection of the reported result",
        "signaling_questions": [
            "Is the reported result likely to have been selected from multiple outcome measurements?",
            "Is the reported result likely to have been selected from multiple analyses?",
            "Is the reported result likely to have been selected from different subgroups?",
        ],
    },
}

VALID_JUDGMENTS = {"Low", "Moderate", "Serious", "Critical", "No information"}


# ======================================================================
# Study Design Classification
# ======================================================================

_RCT_PATTERNS = [
    "rct", "randomized", "randomised", "random assignment", "random allocation",
    "randomly assigned", "randomly allocated", "randomization", "randomisation",
    "double-blind", "double blind", "triple-blind", "single-blind",
    "placebo-controlled randomized", "cluster-randomized", "crossover randomized",
]

_NON_RCT_PATTERNS = [
    "cohort", "case-control", "case control", "cross-sectional", "cross sectional",
    "observational", "retrospective", "prospective cohort", "registry",
    "before-after", "before and after", "interrupted time series",
    "quasi-experimental", "quasi experimental", "non-randomized", "non-randomised",
    "natural experiment", "propensity", "matched", "uncontrolled",
]


def classify_study_design(study: dict) -> str:
    """
    Classify a study as 'RCT' or 'non-RCT' using keyword matching.

    Checks study_design field, title, and population_description.
    Uses pattern matching — not exact string comparison — to handle
    variations like 'randomized controlled trial', 'Randomised Clinical
    Trial', 'RCT', etc.

    Returns: 'RCT' or 'non-RCT'
    """
    # Build searchable text from multiple fields
    fields = [
        study.get("study_design", ""),
        study.get("title", ""),
        study.get("population_description", ""),
        study.get("intervention_description", ""),
    ]
    text = " ".join(str(f) for f in fields).lower()

    # Check RCT patterns first (higher priority)
    for pattern in _RCT_PATTERNS:
        if pattern in text:
            return "RCT"

    # Check non-RCT patterns
    for pattern in _NON_RCT_PATTERNS:
        if pattern in text:
            return "non-RCT"

    # Default: if study_design field is empty or unrecognized, return non-RCT
    # (conservative — ROBINS-I is more general)
    design = study.get("study_design", "").strip()
    if not design:
        return "non-RCT"

    return "non-RCT"


# ======================================================================
# ROBINS-I Assessment Data Structure
# ======================================================================

def create_empty_assessment(study_id: str) -> dict:
    """Create an empty ROBINS-I assessment template for a study."""
    domains = {}
    for domain_id, domain_def in ROBINS_I_DOMAINS.items():
        domains[domain_id] = {
            "name": domain_def["name"],
            "signaling_questions": [
                {"question": q, "answer": None, "support": ""}
                for q in domain_def["signaling_questions"]
            ],
            "judgment": None,
            "justification": "",
        }

    return {
        "study_id": study_id,
        "tool": "ROBINS-I",
        "domains": domains,
        "overall_judgment": None,
        "overall_justification": "",
    }


def derive_overall_judgment(assessment: dict) -> str:
    """Derive overall ROBINS-I judgment from domain judgments."""
    judgments = []
    for domain_id in ROBINS_I_DOMAINS:
        j = assessment["domains"][domain_id].get("judgment")
        if j:
            judgments.append(j)

    if not judgments:
        return "No information"

    if any(j == "Critical" for j in judgments):
        return "Critical"

    if any(j == "Serious" for j in judgments):
        return "Serious"

    if any(j == "Moderate" for j in judgments):
        return "Moderate"

    if all(j == "Low" for j in judgments):
        return "Low"

    return "Moderate"


# ======================================================================
# LLM-Assisted ROBINS-I Assessor
# ======================================================================

ROBINS_I_SYSTEM_PROMPT = """\
You are a systematic review methodologist performing ROBINS-I risk of bias
assessments for non-randomized studies of interventions.

RULES:
1. Assess each domain based ONLY on the information provided.
2. For each signaling question, answer: "Y" (Yes), "PY" (Probably Yes),
   "PN" (Probably No), "N" (No), or "NI" (No Information).
3. Domain judgment: "Low" (comparable to well-performed RCT),
   "Moderate" (sound but not comparable to RCT),
   "Serious" (important problems), "Critical" (too problematic),
   or "No information".
4. If information is insufficient, use "No information" for the question
   and "Moderate" or "Serious" for the domain judgment.
5. Be conservative — observational studies generally have at least "Moderate"
   risk of bias due to confounding.
6. Provide brief justifications citing specific study details.

Return valid JSON only."""


class RobinsIAssessor(BaseAgent):
    """LLM-assisted ROBINS-I assessment for non-randomized studies."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        super().__init__(role_name="extractor", budget=budget)

    def assess_study(self, study_data: dict, study_context: str = "") -> dict:
        study_id = study_data.get("study_id", "unknown")
        assessment = create_empty_assessment(study_id)

        prompt = self._build_prompt(study_data, study_context)

        result = self.call_llm(
            prompt=prompt,
            system_prompt=ROBINS_I_SYSTEM_PROMPT,
            expect_json=True,
            cache_namespace=f"robins_i_{study_id}",
            description=f"ROBINS-I assess {study_id}",
        )

        parsed = result.get("parsed")
        if parsed:
            assessment = self._merge_llm_assessment(assessment, parsed)

        assessment["overall_judgment"] = derive_overall_judgment(assessment)
        return assessment

    def assess_batch(self, studies: List[dict],
                     contexts: Dict[str, str] = None) -> List[dict]:
        contexts = contexts or {}
        assessments = []
        for study in studies:
            sid = study.get("study_id", "")
            ctx = contexts.get(sid, "")
            assessment = self.assess_study(study, ctx)
            assessments.append(assessment)
        return assessments

    def _build_prompt(self, study_data: dict, context: str) -> str:
        study_info = (
            f"Study: {study_data.get('study_id', 'unknown')}\n"
            f"Title: {study_data.get('title', 'N/A')}\n"
            f"Design: {study_data.get('study_design', 'N/A')}\n"
            f"Population: {study_data.get('population_description', 'N/A')}\n"
            f"N: {study_data.get('total_n', 'N/A')}\n"
            f"Intervention: {study_data.get('intervention_description', 'N/A')}\n"
            f"Control: {study_data.get('control_description', 'N/A')}\n"
        )

        if context:
            study_info += f"\nFull text context (excerpt):\n{context[:3000]}\n"

        domain_instructions = []
        for domain_id, domain_def in ROBINS_I_DOMAINS.items():
            qs = "\n".join(f"  - {q}" for q in domain_def["signaling_questions"])
            domain_instructions.append(
                f"{domain_id} ({domain_def['name']}):\n"
                f"  Signaling questions:\n{qs}"
            )

        return (
            f"Perform a ROBINS-I risk of bias assessment for this "
            f"non-randomized study.\n\n"
            f"STUDY INFORMATION:\n{study_info}\n\n"
            f"DOMAINS TO ASSESS:\n" +
            "\n\n".join(domain_instructions) +
            "\n\nReturn JSON with this structure:\n"
            "{\n"
            '  "domains": {\n'
            '    "D1": {\n'
            '      "answers": ["Y"/"PY"/"PN"/"N"/"NI", ...],\n'
            '      "judgment": "Low" | "Moderate" | "Serious" | "Critical",\n'
            '      "justification": "brief reason"\n'
            '    },\n'
            "    ... D2 through D7 ...\n"
            "  }\n"
            "}"
        )

    def _merge_llm_assessment(self, assessment: dict, parsed: dict) -> dict:
        llm_domains = parsed.get("domains", {})

        for domain_id in ROBINS_I_DOMAINS:
            if domain_id not in llm_domains:
                continue

            llm_d = llm_domains[domain_id]

            answers = llm_d.get("answers", [])
            sq_list = assessment["domains"][domain_id]["signaling_questions"]
            for i, ans in enumerate(answers):
                if i < len(sq_list):
                    sq_list[i]["answer"] = ans

            judgment = llm_d.get("judgment", "")
            if judgment in VALID_JUDGMENTS:
                assessment["domains"][domain_id]["judgment"] = judgment

            assessment["domains"][domain_id]["justification"] = (
                llm_d.get("justification", "")
            )

        return assessment


# ======================================================================
# Summary
# ======================================================================

def build_robins_i_summary(assessments: List[dict]) -> dict:
    """Build a summary table of ROBINS-I results across studies."""
    summary = {
        "tool": "ROBINS-I",
        "studies": [],
        "domain_counts": {},
        "overall_counts": {
            "Low": 0, "Moderate": 0, "Serious": 0, "Critical": 0,
        },
    }

    for domain_id in ROBINS_I_DOMAINS:
        summary["domain_counts"][domain_id] = {
            "Low": 0, "Moderate": 0, "Serious": 0, "Critical": 0,
        }

    for a in assessments:
        study_entry = {
            "study_id": a["study_id"],
            "overall": a.get("overall_judgment", "No information"),
        }

        for domain_id in ROBINS_I_DOMAINS:
            j = a["domains"][domain_id].get("judgment", "No information")
            study_entry[domain_id] = j
            if j in summary["domain_counts"][domain_id]:
                summary["domain_counts"][domain_id][j] += 1

        overall = a.get("overall_judgment", "No information")
        if overall in summary["overall_counts"]:
            summary["overall_counts"][overall] += 1

        summary["studies"].append(study_entry)

    return summary
