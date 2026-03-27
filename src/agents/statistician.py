"""
Statistician Agent — LUMEN v2
================================
Phase 5: LLM-assisted statistical analysis code generation.
The heavy math lives in src/utils/statistics.py (pure Python).
This agent supplements with interpretation and custom analysis code.
"""

import json
import logging
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.utils.cache import TokenBudget

logger = logging.getLogger(__name__)


class StatisticianAgent(BaseAgent):
    """LLM agent for supplementary statistical analysis and interpretation."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        super().__init__(role_name="statistician", budget=budget)

    def interpret_results(self, statistical_results: dict,
                          study_context: dict) -> dict:
        """Generate interpretation of meta-analysis results."""
        system_prompt = self._prompt_config.get("system_prompt", "")
        prompt = self._build_interpretation_prompt(
            statistical_results, study_context
        )

        result = self.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            expect_json=True,
            cache_namespace="stats_interpretation",
            description="Interpret meta-analysis results",
        )

        return result.get("parsed") or {"interpretation": result.get("content", "")}

    def generate_supplementary_analysis(self, data_summary: dict,
                                        analysis_request: str) -> str:
        """Generate Python code for a custom supplementary analysis."""
        prompt = (
            f"Write Python code for the following meta-analysis:\n\n"
            f"Data summary:\n{json.dumps(data_summary, indent=2, default=str)}\n\n"
            f"Analysis request: {analysis_request}\n\n"
            f"Use only numpy, scipy, and matplotlib. "
            f"The code should be self-contained and print results."
        )

        result = self.call_llm(
            prompt=prompt,
            cache_namespace="stats_code",
            description=f"Generate code: {analysis_request[:50]}",
        )

        return result.get("content", "")

    def _build_interpretation_prompt(self, stats: dict, context: dict) -> str:
        return (
            f"Interpret these meta-analysis results for a manuscript:\n\n"
            f"Statistical results:\n{json.dumps(stats, indent=2, default=str)}\n\n"
            f"Study context:\n{json.dumps(context, indent=2, default=str)}\n\n"
            f"Return JSON with:\n"
            f"- summary: 2-3 sentence summary of main findings\n"
            f"- heterogeneity_interpretation: assessment of I2/Q\n"
            f"- clinical_significance: practical implications\n"
            f"- limitations: statistical limitations to note\n"
            f"- grade_certainty: GRADE certainty assessment (high/moderate/low/very_low)\n"
        )
