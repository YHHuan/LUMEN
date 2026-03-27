"""
Risk of Bias 2 (RoB-2) Assessment — LUMEN v2
===============================================
Implements the Cochrane RoB-2 tool for randomized controlled trials.

5 Domains:
  D1: Bias from randomization process
  D2: Bias due to deviations from intended interventions
  D3: Bias due to missing outcome data
  D4: Bias in measurement of the outcome
  D5: Bias in selection of the reported result

Each domain scored: "Low risk" | "Some concerns" | "High risk"
Overall judgment derived from domain scores.

References:
  Sterne et al. (2019) BMJ 366:l4898
  https://www.riskofbias.info
"""

import json
import logging
from typing import Optional, List, Dict

from src.agents.base_agent import BaseAgent
from src.utils.cache import TokenBudget

logger = logging.getLogger(__name__)


# ======================================================================
# RoB-2 Domain Definitions
# ======================================================================

ROB2_DOMAINS = {
    "D1": {
        "name": "Randomization process",
        "signaling_questions": [
            "Was the allocation sequence random?",
            "Was the allocation sequence concealed until participants were enrolled?",
            "Did baseline differences suggest a problem with randomization?",
        ],
    },
    "D2": {
        "name": "Deviations from intended interventions",
        "signaling_questions": [
            "Were participants aware of their assigned intervention?",
            "Were carers/people delivering the interventions aware of the assignment?",
            "Were there deviations from the intended intervention beyond what would be expected?",
            "Were these deviations likely to have affected the outcome?",
            "Was an appropriate analysis used to estimate the effect of assignment to intervention?",
        ],
    },
    "D3": {
        "name": "Missing outcome data",
        "signaling_questions": [
            "Were data for this outcome available for all/nearly all participants?",
            "Is there evidence that the result was not biased by missing outcome data?",
            "Could missingness depend on the true value of the outcome?",
        ],
    },
    "D4": {
        "name": "Measurement of the outcome",
        "signaling_questions": [
            "Was the method of measuring the outcome inappropriate?",
            "Could measurement/ascertainment have differed between groups?",
            "Were outcome assessors aware of the intervention received?",
            "Could assessment of the outcome have been influenced by knowledge of intervention?",
        ],
    },
    "D5": {
        "name": "Selection of the reported result",
        "signaling_questions": [
            "Were the data analysed in accordance with a pre-specified analysis plan?",
            "Is the numerical result being assessed likely to have been selected from multiple outcome measurements?",
            "Is the numerical result being assessed likely to have been selected from multiple analyses?",
        ],
    },
}

VALID_JUDGMENTS = {"Low risk", "Some concerns", "High risk", "No information"}


# ======================================================================
# RoB-2 Assessment Data Structure
# ======================================================================

def create_empty_assessment(study_id: str) -> dict:
    """Create an empty RoB-2 assessment template for a study."""
    domains = {}
    for domain_id, domain_def in ROB2_DOMAINS.items():
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
        "domains": domains,
        "overall_judgment": None,
        "overall_justification": "",
    }


def derive_overall_judgment(assessment: dict) -> str:
    """Derive overall RoB-2 judgment from domain judgments (Cochrane algorithm)."""
    judgments = []
    for domain_id in ROB2_DOMAINS:
        j = assessment["domains"][domain_id].get("judgment")
        if j:
            judgments.append(j)

    if not judgments:
        return "No information"

    if any(j == "High risk" for j in judgments):
        return "High risk"

    if sum(1 for j in judgments if j == "Some concerns") > 1:
        return "High risk"

    if any(j == "Some concerns" for j in judgments):
        return "Some concerns"

    if all(j == "Low risk" for j in judgments):
        return "Low risk"

    return "Some concerns"


# ======================================================================
# LLM-Assisted RoB-2 Assessor
# ======================================================================

class RoB2Assessor(BaseAgent):
    """LLM-assisted RoB-2 assessment for extracted studies."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        # Use extractor model for RoB-2 assessment
        super().__init__(role_name="extractor", budget=budget)

    def assess_study(self, study_data: dict, study_context: str = "") -> dict:
        """
        Perform RoB-2 assessment on a single study.

        Args:
            study_data: Extracted study data (from Phase 4)
            study_context: Optional full text context

        Returns: RoB-2 assessment dict
        """
        study_id = study_data.get("study_id", "unknown")
        assessment = create_empty_assessment(study_id)

        prompt = self._build_rob2_prompt(study_data, study_context)

        result = self.call_llm(
            prompt=prompt,
            system_prompt=ROB2_SYSTEM_PROMPT,
            expect_json=True,
            cache_namespace=f"rob2_{study_id}",
            description=f"RoB-2 assess {study_id}",
        )

        parsed = result.get("parsed")
        if parsed:
            assessment = self._merge_llm_assessment(assessment, parsed)

        assessment["overall_judgment"] = derive_overall_judgment(assessment)
        return assessment

    def assess_batch(self, studies: List[dict],
                     contexts: Dict[str, str] = None) -> List[dict]:
        """Assess multiple studies."""
        contexts = contexts or {}
        assessments = []

        for study in studies:
            sid = study.get("study_id", "")
            ctx = contexts.get(sid, "")
            assessment = self.assess_study(study, ctx)
            assessments.append(assessment)

        return assessments

    def _build_rob2_prompt(self, study_data: dict, context: str) -> str:
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
        for domain_id, domain_def in ROB2_DOMAINS.items():
            qs = "\n".join(f"  - {q}" for q in domain_def["signaling_questions"])
            domain_instructions.append(
                f"{domain_id} ({domain_def['name']}):\n"
                f"  Signaling questions:\n{qs}"
            )

        return (
            f"Perform a Cochrane RoB-2 risk of bias assessment.\n\n"
            f"STUDY INFORMATION:\n{study_info}\n\n"
            f"DOMAINS TO ASSESS:\n" +
            "\n\n".join(domain_instructions) +
            "\n\nReturn JSON with this structure:\n"
            "{\n"
            '  "domains": {\n'
            '    "D1": {\n'
            '      "answers": ["Yes"/"No"/"Probably yes"/"Probably no"/"No information", ...],\n'
            '      "judgment": "Low risk" | "Some concerns" | "High risk",\n'
            '      "justification": "brief reason"\n'
            '    },\n'
            "    ... D2 through D5 ...\n"
            "  }\n"
            "}"
        )

    def _merge_llm_assessment(self, assessment: dict, parsed: dict) -> dict:
        """Merge LLM output into assessment structure."""
        llm_domains = parsed.get("domains", {})

        for domain_id in ROB2_DOMAINS:
            if domain_id not in llm_domains:
                continue

            llm_d = llm_domains[domain_id]

            # Update answers
            answers = llm_d.get("answers", [])
            sq_list = assessment["domains"][domain_id]["signaling_questions"]
            for i, ans in enumerate(answers):
                if i < len(sq_list):
                    sq_list[i]["answer"] = ans

            # Update judgment
            judgment = llm_d.get("judgment", "")
            if judgment in VALID_JUDGMENTS:
                assessment["domains"][domain_id]["judgment"] = judgment

            assessment["domains"][domain_id]["justification"] = (
                llm_d.get("justification", "")
            )

        return assessment


ROB2_SYSTEM_PROMPT = """You are a systematic review methodologist performing Cochrane Risk of Bias 2 (RoB-2) assessments for randomized controlled trials.

RULES:
1. Assess each domain based ONLY on the information provided.
2. For each signaling question, answer: "Yes", "No", "Probably yes", "Probably no", or "No information".
3. Domain judgment: "Low risk" (all favorable), "Some concerns" (some uncertainty), "High risk" (serious concerns).
4. If information is insufficient, use "No information" and note this as "Some concerns".
5. Be conservative — when in doubt, use "Some concerns" rather than "Low risk".
6. Provide brief justifications citing specific study details.

Return valid JSON only."""


# ======================================================================
# Summary Visualization Data
# ======================================================================

def build_rob2_summary(assessments: List[dict]) -> dict:
    """
    Build a summary table of RoB-2 results across studies.
    Suitable for traffic-light plots.
    """
    summary = {
        "studies": [],
        "domain_counts": {},
        "overall_counts": {"Low risk": 0, "Some concerns": 0, "High risk": 0},
    }

    for domain_id in ROB2_DOMAINS:
        summary["domain_counts"][domain_id] = {
            "Low risk": 0, "Some concerns": 0, "High risk": 0,
        }

    for a in assessments:
        study_entry = {
            "study_id": a["study_id"],
            "overall": a.get("overall_judgment", "No information"),
        }

        for domain_id in ROB2_DOMAINS:
            j = a["domains"][domain_id].get("judgment", "No information")
            study_entry[domain_id] = j
            if j in summary["domain_counts"][domain_id]:
                summary["domain_counts"][domain_id][j] += 1

        overall = a.get("overall_judgment", "No information")
        if overall in summary["overall_counts"]:
            summary["overall_counts"][overall] += 1

        summary["studies"].append(study_entry)

    return summary
