"""
Arbiter agent for screening conflict resolution.

Union-hybrid strategy (literature-informed):
  - Either screener includes → auto-include (union principle, max sensitivity)
  - Both exclude + both confidence ≥ 80 → auto-exclude
  - Both exclude + either confidence < 80 → arbiter reviews
  - Arbiter confidence < 60 → human_review

Fixes v2 audit #7: parse failure → human_review, NOT include.

Literature basis:
  - Sanghera et al. JAMIA 2025: LLM-LLM ensembles → perfect sensitivity
  - medRxiv 2025: dual-model union → 99.7% sensitivity
  - Oami et al. RSM 2025: cross-provider bias complementarity
"""
from __future__ import annotations

import json
import structlog

from lumen.agents.base import BaseAgent, LumenParseError

logger = structlog.get_logger()

# Routing thresholds
CONFIDENCE_AUTO_ACCEPT = 80   # both screeners above this + agree → auto-decide
CONFIDENCE_ARBITER_MIN = 50   # below this → must call arbiter
CONFIDENCE_HUMAN_REVIEW = 60  # arbiter below this → human_review


class ArbiterAgent(BaseAgent):
    tier = "strategic"
    agent_name = "arbiter"
    prompt_file = "arbiter.yaml"

    def resolve(self, study: dict, screen1: dict, screen2: dict,
                pico: dict, criteria: dict) -> dict:
        """
        Resolve disagreement between two screeners.

        Returns {decision, confidence, reasoning, agreed_with}.
        CRITICAL: parse failure → human_review (v2 audit #7).
        """
        user_content = (
            f"## PICO Criteria\n{json.dumps(pico, indent=2)}\n\n"
            f"## Screening Criteria\n{json.dumps(criteria, indent=2)}\n\n"
            f"## Study\n"
            f"Title: {study.get('title', 'N/A')}\n"
            f"Abstract: {study.get('abstract', 'N/A')}\n\n"
            f"## Screener 1 Assessment\n"
            f"Decision: {screen1.get('decision')}\n"
            f"Confidence: {screen1.get('confidence')}\n"
            f"Reasoning: {screen1.get('reasoning')}\n\n"
            f"## Screener 2 Assessment\n"
            f"Decision: {screen2.get('decision')}\n"
            f"Confidence: {screen2.get('confidence')}\n"
            f"Reasoning: {screen2.get('reasoning')}\n"
        )
        messages = self._build_messages(user_content)

        try:
            response = self._call_llm(
                messages,
                response_format={"type": "json_object"},
                phase="screening",
            )
            result = self._parse_json(response, retry_messages=messages,
                                      phase="screening")
        except LumenParseError:
            # v2 audit #7: parse failure → human_review, NOT include
            logger.error("arbiter_parse_failure",
                         study_id=study.get("study_id", "unknown"))
            return {
                "decision": "human_review",
                "confidence": 0,
                "reasoning": "Arbiter failed to produce valid output — routing to human review",
                "agreed_with": "neither",
                "study_id": study.get("study_id", study.get("id", "unknown")),
                "parse_error": True,
            }

        result = self._validate_arbiter_result(result)
        result["study_id"] = study.get("study_id", study.get("id", "unknown"))
        return result

    @staticmethod
    def _validate_arbiter_result(result: dict) -> dict:
        """Validate and enforce confidence-based routing."""
        decision = result.get("decision", "").lower().strip()
        confidence = max(0, min(100, int(result.get("confidence", 50))))

        # Enforce human_review for low confidence
        if confidence < CONFIDENCE_HUMAN_REVIEW:
            decision = "human_review"

        if decision not in ("include", "exclude", "human_review"):
            decision = "human_review"
            confidence = 0

        result["decision"] = decision
        result["confidence"] = confidence
        result.setdefault("reasoning", "")
        result.setdefault("agreed_with", "neither")
        return result


def needs_arbiter(screen1: dict, screen2: dict) -> bool:
    """
    Determine if arbiter is needed (union-hybrid strategy).

    Union principle: if EITHER screener says include → auto-include (no arbiter).
    Arbiter only needed when both exclude but confidence is uncertain.
    """
    d1, d2 = screen1["decision"], screen2["decision"]
    c1, c2 = screen1["confidence"], screen2["confidence"]

    # Either includes → auto-include via union (no arbiter needed)
    if d1 == "include" or d2 == "include":
        return False

    # Both exclude + both high confidence → auto-exclude (no arbiter)
    if c1 >= CONFIDENCE_AUTO_ACCEPT and c2 >= CONFIDENCE_AUTO_ACCEPT:
        return False

    # Both exclude but at least one low confidence → arbiter reviews
    # This catches potential false negatives from uncertain exclusions
    return True


def resolve_screening(screen1: dict, screen2: dict,
                      arbiter_result: dict | None = None) -> dict:
    """
    Produce final screening decision (union-hybrid strategy).

    Logic:
    1. Either includes → include (union, maximizes sensitivity)
    2. Both exclude + high confidence → exclude
    3. Both exclude + low confidence → arbiter decides
    4. Arbiter low confidence → human_review

    Returns {final_decision, method, confidence, arbiter_used}.
    """
    d1, d2 = screen1["decision"], screen2["decision"]
    c1, c2 = screen1["confidence"], screen2["confidence"]

    # Union: either includes → include
    if d1 == "include" or d2 == "include":
        # Use confidence of the includer (or higher if both include)
        if d1 == "include" and d2 == "include":
            conf = max(c1, c2)
            method = "dual_include"
        else:
            conf = c1 if d1 == "include" else c2
            method = "union_include"
        return {
            "final_decision": "include",
            "method": method,
            "confidence": conf,
            "arbiter_used": False,
        }

    # Both exclude — check if arbiter needed
    if not needs_arbiter(screen1, screen2):
        # High confidence dual exclude
        return {
            "final_decision": "exclude",
            "method": "dual_exclude_high_confidence",
            "confidence": min(c1, c2),
            "arbiter_used": False,
        }

    # Arbiter was needed
    if arbiter_result is None:
        return {
            "final_decision": "human_review",
            "method": "arbiter_missing",
            "confidence": 0,
            "arbiter_used": False,
        }

    return {
        "final_decision": arbiter_result["decision"],
        "method": "arbiter",
        "confidence": arbiter_result["confidence"],
        "arbiter_used": True,
        "agreed_with": arbiter_result.get("agreed_with", "neither"),
    }
