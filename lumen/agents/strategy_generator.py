"""
Strategy generator agent: PICO → search strategy + screening criteria.
"""
from __future__ import annotations

import json
import structlog

from lumen.agents.base import BaseAgent, LumenParseError

logger = structlog.get_logger()


class StrategyGeneratorAgent(BaseAgent):
    tier = "smart"
    agent_name = "strategy_generator"
    prompt_file = "strategy_generator.yaml"

    def generate(self, pico: dict) -> dict:
        """
        Generate search strategy and screening criteria from PICO.

        Returns {search_strategy, screening_criteria}.
        """
        user_content = (
            f"## PICO Definition\n{json.dumps(pico, indent=2)}\n\n"
            "Generate a comprehensive search strategy and screening criteria."
        )
        messages = self._build_messages(user_content)

        try:
            response = self._call_llm(
                messages, response_format={"type": "json_object"},
                phase="strategy_generation",
            )
            result = self._parse_json(response, retry_messages=messages,
                                      phase="strategy_generation")
            return {
                "search_strategy": {
                    "queries": result.get("search_queries", []),
                    "mesh_terms": result.get("mesh_terms", []),
                    "expected_yield": result.get("expected_yield", {}),
                },
                "screening_criteria": result.get("screening_criteria", {}),
            }
        except LumenParseError:
            logger.warning("strategy_generation_failed")
            return {
                "search_strategy": {"queries": [], "mesh_terms": []},
                "screening_criteria": {
                    "inclusion": [],
                    "exclusion": [],
                    "required_keywords": [],
                    "exclusion_keywords": [],
                },
            }
