"""
Statistician Agent (Phase 5)
=============================
生成並執行統計分析代碼。
- Random-effects meta-analysis
- Forest plots, funnel plots
- Heterogeneity (I², Q, tau²)
- Subgroup analyses
- Sensitivity analyses
- Publication bias tests
"""

import json
import logging
from src.agents.base_agent import BaseAgent
from src.utils.project import get_data_dir

logger = logging.getLogger(__name__)


class StatisticianAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(role_name="statistician", **kwargs)
        # Load system prompt from YAML; fall back to inline default if file missing
        self._system_prompt = self._prompt_config.get(
            "system_prompt", _DEFAULT_STATISTICIAN_SYSTEM
        )

    def generate_analysis_code(self, extracted_data: list,
                                pico: dict,
                                subgroups: list = None) -> str:
        """
        生成完整的 Python 統計分析代碼。

        Returns:
            Python code as string (ready to execute)
        """
        system_prompt = self._system_prompt
        
        # Summarize the data structure for the LLM
        data_summary = []
        for study in extracted_data[:3]:  # Show first 3 as examples
            data_summary.append({
                "study_id": study.get("study_id"),
                "citation": study.get("citation"),
                "outcomes": {k: v for k, v in study.get("outcomes", {}).items()},
            })
        
        data_dir = get_data_dir()
        prompt = f"""Generate a complete Python meta-analysis script.

DATA STRUCTURE (first 3 of {len(extracted_data)} studies):
{json.dumps(data_summary, indent=2)}

ANALYSIS REQUIREMENTS:
1. Primary analysis: Random-effects meta-analysis (DerSimonian-Laird) for each primary outcome
2. Effect measure: Standardized Mean Difference (SMD / Hedges' g) for continuous outcomes
3. Heterogeneity: I², Q statistic, tau², prediction interval
4. Forest plot for each primary outcome
5. Funnel plot + Egger's test for publication bias
6. Subgroup analyses: {json.dumps(subgroups or [], indent=2)}
7. Sensitivity analysis: leave-one-out
8. Risk of Bias summary figure

INPUT: The script should load data from '{data_dir}/phase4_extraction/extracted_data.json'
OUTPUT: Save all results and figures to '{data_dir}/phase5_analysis/'

Key requirements:
- Handle missing data gracefully (skip studies with insufficient data for specific analyses)
- Calculate 95% confidence intervals
- Use inverse-variance weighting
- Generate publication-quality figures (300 dpi, clear labels)
- Save statistical results as JSON
- Print a summary to console

Generate the complete Python script:"""
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            expect_json=False,
            cache_namespace="phase5_analysis_code",
            description="Generate meta-analysis code",
        )
        
        code = result["content"]
        # Clean up markdown fences if present
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:])  # Remove first line
            if code.rstrip().endswith("```"):
                code = code.rstrip()[:-3]
        
        return code


# ── Inline fallback (used when config/prompts/statistician.yaml is missing) ──
_DEFAULT_STATISTICIAN_SYSTEM = (
    "You are a biostatistician specializing in meta-analysis. "
    "Generate complete, executable Python code for the analysis. "
    "Use scipy, numpy, matplotlib. Do NOT use R. "
    "The code must be self-contained and save all outputs to specified paths. "
    "Respond with ONLY the Python code, no markdown fences."
)
