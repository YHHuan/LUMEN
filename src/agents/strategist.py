"""
Strategist Agent — LUMEN v2
==============================
Phase 1: PICO -> search strategy (MeSH terms, queries, screening criteria)
v2: Also generates positive_rescue_keywords for pre-screening rescue pipeline.
"""

import json
import logging
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.utils.cache import TokenBudget
from src.utils.file_handlers import DataManager

logger = logging.getLogger(__name__)


class StrategistAgent(BaseAgent):
    def __init__(self, budget: Optional[TokenBudget] = None):
        super().__init__(role_name="strategist", budget=budget)

    def generate_strategy(self, pico: dict) -> dict:
        """Generate complete search strategy from PICO."""
        system_prompt = self._prompt_config.get("system_prompt", "")
        user_prompt = self._build_strategy_prompt(pico)

        result = self.call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            expect_json=True,
            cache_namespace="phase1_strategy",
            description="Strategy generation from PICO",
        )

        parsed = result.get("parsed")
        if not parsed:
            logger.error("Failed to parse strategy output")
            return {}

        return parsed

    def generate_rescue_keywords(self, pico: dict, strategy: dict) -> dict:
        """Generate positive rescue keywords for pre-screening rescue pipeline."""
        prompt = self._build_rescue_keywords_prompt(pico, strategy)
        system_prompt = self._prompt_config.get("rescue_keywords_system_prompt",
            "You generate keyword lists for rescuing potentially relevant "
            "studies from pre-screening exclusion. Output JSON only."
        )

        result = self.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            expect_json=True,
            cache_namespace="phase1_rescue_keywords",
            description="Rescue keywords generation",
        )

        parsed = result.get("parsed")
        if not parsed:
            # Fallback: build from PICO directly
            return self._fallback_rescue_keywords(pico)

        return parsed

    def _build_strategy_prompt(self, pico: dict) -> str:
        template = self._prompt_config.get("user_prompt_template", "")
        if template:
            return template.format(**pico)

        return (
            f"Generate a comprehensive systematic review search strategy "
            f"for the following PICO:\n\n"
            f"Population: {pico.get('population', '')}\n"
            f"Intervention: {pico.get('intervention', '')}\n"
            f"Comparison: {pico.get('comparison', '')}\n"
            f"Outcome: {pico.get('outcome', '')}\n"
            f"Study Design: {pico.get('study_design', 'RCT')}\n\n"
            f"Return a JSON object with:\n"
            f"1. mesh_terms: list of MeSH terms\n"
            f"2. search_queries: dict of database-specific queries "
            f"   (pubmed, europepmc, scopus, crossref)\n"
            f"3. screening_criteria: {{inclusion: [...], exclusion: [...]}}\n"
            f"4. pico_summary: one-paragraph summary\n"
        )

    def _build_rescue_keywords_prompt(self, pico: dict, strategy: dict) -> str:
        return (
            f"Based on the PICO and search strategy below, generate positive "
            f"keyword signals that indicate a study MIGHT be relevant.\n\n"
            f"PICO:\n{json.dumps(pico, indent=2)}\n\n"
            f"Search strategy MeSH terms: {strategy.get('mesh_terms', [])}\n\n"
            f"Return JSON with:\n"
            f'{{"intervention": ["keyword1", ...], '
            f'"population": ["keyword1", ...], '
            f'"study_design": ["keyword1", ...]}}\n\n'
            f"Include synonyms, abbreviations, and common variants. "
            f"Be comprehensive — these keywords rescue studies from false exclusion."
        )

    def _fallback_rescue_keywords(self, pico: dict) -> dict:
        """Build rescue keywords directly from PICO if LLM fails."""
        import re
        keywords = {"intervention": [], "population": [], "study_design": []}

        for field in ["intervention", "comparison"]:
            val = pico.get(field, "")
            if val:
                keywords["intervention"].extend(
                    kw.strip() for kw in re.split(r"[,;/]", val) if kw.strip()
                )

        for field in ["population"]:
            val = pico.get(field, "")
            if val:
                keywords["population"].extend(
                    kw.strip() for kw in re.split(r"[,;/]", val) if kw.strip()
                )

        keywords["study_design"] = [
            "randomized", "RCT", "controlled trial", "sham",
            "placebo", "double-blind",
        ]

        return keywords
