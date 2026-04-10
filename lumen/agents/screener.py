"""
Cross-model dual screener with confidence scores.

v3: forced binary (include/exclude) + 0-100 confidence. No undecided.
Supports tier override for cross-model ensemble (e.g., FAST + SMART).

Literature basis: Sanghera et al. JAMIA 2025 — LLM-LLM ensembles
achieve perfect sensitivity. Oami et al. Research Synthesis Methods
2025 — different LLM providers have complementary bias profiles.
"""
from __future__ import annotations

import json
import structlog

from lumen.agents.base import BaseAgent, LumenParseError

logger = structlog.get_logger()


class ScreenerAgent(BaseAgent):
    tier = "smart"  # default; override via __init__(tier_override=...)
    agent_name = "screener"
    prompt_file = "screener.yaml"

    def __init__(self, router, cost_tracker, config, tier_override: str | None = None):
        super().__init__(router=router, cost_tracker=cost_tracker, config=config)
        if tier_override:
            self.tier = tier_override

    def screen_single(self, study: dict, pico: dict,
                      criteria: dict, screener_id: str = "1") -> dict:
        """
        Screen a single study based on title/abstract.

        Returns {decision, confidence, reasoning, key_factors}.
        """
        user_content = (
            f"## PICO Criteria\n{json.dumps(pico, indent=2)}\n\n"
            f"## Screening Criteria\n{json.dumps(criteria, indent=2)}\n\n"
            f"## Study to Screen\n"
            f"Title: {study.get('title', 'N/A')}\n"
            f"Abstract: {study.get('abstract', 'N/A')}\n"
            f"Authors: {study.get('authors', 'N/A')}\n"
            f"Year: {study.get('year', 'N/A')}\n"
        )
        messages = self._build_messages(user_content)
        response = self._call_llm(
            messages,
            response_format={"type": "json_object"},
            phase="screening",
        )
        result = self._parse_json(response, retry_messages=messages, phase="screening")
        result = self._validate_screening_result(result)
        result["screener_id"] = screener_id
        result["study_id"] = study.get("study_id", study.get("id", "unknown"))
        return result

    def screen_batch(self, studies: list[dict], pico: dict,
                     criteria: dict, screener_id: str = "1") -> list[dict]:
        """Screen multiple studies sequentially."""
        results = []
        for study in studies:
            try:
                result = self.screen_single(study, pico, criteria, screener_id)
            except LumenParseError:
                logger.error("screening_parse_failure",
                             study_id=study.get("study_id", "unknown"),
                             screener_id=screener_id)
                result = {
                    "decision": "include",  # conservative: don't lose studies
                    "confidence": 0,
                    "reasoning": "Parse failure — included conservatively",
                    "key_factors": [],
                    "screener_id": screener_id,
                    "study_id": study.get("study_id", study.get("id", "unknown")),
                    "parse_error": True,
                }
            results.append(result)
        return results

    @staticmethod
    def _validate_screening_result(result: dict) -> dict:
        """Ensure decision is binary and confidence is 0-100."""
        decision = result.get("decision", "").lower().strip()
        if decision not in ("include", "exclude"):
            # Force to include if ambiguous (conservative)
            logger.warning("invalid_screening_decision", decision=decision)
            decision = "include"
            result["confidence"] = max(0, result.get("confidence", 0) - 20)

        result["decision"] = decision
        result["confidence"] = max(0, min(100, int(result.get("confidence", 50))))
        result.setdefault("reasoning", "")
        result.setdefault("key_factors", [])
        return result
