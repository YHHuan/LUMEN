"""
PICO Interviewer agent.

Headless mode: completeness score >= 80 → skip elicitation, return as-is.
Interactive mode: structured multi-round Q&A to refine incomplete PICOs.
"""
from __future__ import annotations

import json
import structlog

from lumen.agents.base import BaseAgent, LumenParseError

logger = structlog.get_logger()

# Scoring weights for PICO completeness
SCORING_RUBRIC = {
    "population": 20,
    "intervention": 20,
    "comparator": 15,
    "outcome": 15,
    "inclusion_exclusion": 15,
    "study_design": 10,
    "timing": 5,
}

COMPLETENESS_THRESHOLD = 80


class PICOInterviewerAgent(BaseAgent):
    tier = "strategic"
    agent_name = "pico_interviewer"
    prompt_file = "pico_interviewer.yaml"

    def assess_completeness(self, pico: dict) -> int:
        """
        Score 0-100 based on PICO component presence and specificity.

        Deterministic scoring — no LLM call.
        """
        score = 0

        # Population: +20 if non-empty and has more than one word
        pop = pico.get("population", "")
        if pop and len(pop.split()) >= 2:
            score += SCORING_RUBRIC["population"]
        elif pop:
            score += SCORING_RUBRIC["population"] // 2

        # Intervention: +20 if specific
        interv = pico.get("intervention", "")
        if interv and len(interv.split()) >= 2:
            score += SCORING_RUBRIC["intervention"]
        elif interv:
            score += SCORING_RUBRIC["intervention"] // 2

        # Comparator: +15 if explicit
        comp = pico.get("comparator", "")
        if comp:
            score += SCORING_RUBRIC["comparator"]

        # Outcome: +15 if defined
        outcome = pico.get("outcome", "")
        if isinstance(outcome, list):
            outcome = ", ".join(outcome)
        if outcome:
            score += SCORING_RUBRIC["outcome"]

        # Inclusion/exclusion: +15 if either present
        inclusion = pico.get("inclusion_criteria", [])
        exclusion = pico.get("exclusion_criteria", [])
        if inclusion or exclusion:
            score += SCORING_RUBRIC["inclusion_exclusion"]

        # Study design: +10
        design = pico.get("study_design", "")
        if design:
            score += SCORING_RUBRIC["study_design"]

        # Timing: +5
        timing = pico.get("timing", "") or pico.get("follow_up", "")
        if timing:
            score += SCORING_RUBRIC["timing"]

        return min(score, 100)

    def elicit(self, initial_pico: dict | None = None) -> dict:
        """
        Refine PICO via LLM.

        Headless mode: if score >= 80, return enriched PICO without interaction.
        """
        pico = initial_pico or {}
        score = self.assess_completeness(pico)

        logger.info("pico_completeness", score=score,
                     threshold=COMPLETENESS_THRESHOLD)

        if score >= COMPLETENESS_THRESHOLD:
            # Already complete — do a single LLM pass to standardize format
            return self._refine_pico(pico, score)

        # Below threshold — LLM refines and generates questions
        return self._refine_pico(pico, score)

    def _refine_pico(self, pico: dict, score: int) -> dict:
        """Single LLM pass to refine and standardize the PICO."""
        user_content = (
            f"## Current PICO\n{json.dumps(pico, indent=2)}\n\n"
            f"## Current Completeness Score\n{score}/100\n\n"
            "Please refine this PICO definition. If it's already complete "
            "(score >= 80), standardize the format and confirm."
        )
        messages = self._build_messages(user_content)

        try:
            response = self._call_llm(
                messages, response_format={"type": "json_object"},
                phase="pico_elicitation",
            )
            result = self._parse_json(response, retry_messages=messages,
                                      phase="pico_elicitation")
            refined = result.get("refined_pico", pico)
            # Re-score with refined PICO
            new_score = self.assess_completeness(refined)
            return {
                "pico": refined,
                "completeness_score": max(new_score, result.get("completeness_score", score)),
                "questions": result.get("questions", []),
                "reasoning": result.get("reasoning", ""),
            }
        except LumenParseError:
            logger.warning("pico_refinement_failed")
            return {
                "pico": pico,
                "completeness_score": score,
                "questions": ["Unable to refine — please provide more details"],
                "reasoning": "LLM refinement failed",
            }
