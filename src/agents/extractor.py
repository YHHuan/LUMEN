"""
Data Extractor Agent (Phase 4) — v5
=====================================
v5: Generalized — outcome keys are now STANDARDIZED by using
    the measure name as the key (e.g., "MMSE", "MoCA", "ADAS-Cog").
    Population/intervention fields are read from pico, not hardcoded.
"""

import json
import logging
import numpy as np
from typing import List

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ExtractorAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(role_name="extractor", **kwargs)
        self._system_prompt = self._prompt_config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)

    def extract_data(self, study: dict, fulltext: str,
                     outcome_definitions: dict,
                     tables: list = None,
                     extraction_guidance: dict = None) -> dict:
        """
        v5: Generalized prompt. The outcome JSON template is built
        dynamically from pico.yaml outcome definitions.
        """
        system_prompt = self._system_prompt
        
        table_ctx = ""
        if tables:
            table_ctx = "\n\nEXTRACTED TABLES:\n"
            for t in tables[:10]:
                table_ctx += f"\n--- Table (Page {t['page']}) ---\n{t.get('markdown','')}\n"
        
        # Build outcome examples from pico definitions
        outcome_examples = self._build_outcome_template(outcome_definitions)
        
        # Build extraction guidance context (from Phase 1 Strategist)
        guidance_ctx = ""
        if extraction_guidance:
            measures = extraction_guidance.get("outcome_measures", [])
            if measures:
                guidance_ctx += "\nOUTCOME MEASURE REFERENCE (from study protocol):\n"
                for m in measures:
                    abbr = m.get("abbreviation", "")
                    full = m.get("full_name", "")
                    rng = m.get("scale_range", "")
                    dirn = m.get("direction", "")
                    guidance_ctx += f"  - {abbr} ({full}): scale {rng}, {dirn}\n"
                guidance_ctx += "  IMPORTANT: Use these scale ranges to distinguish between similarly-named measures.\n"
                guidance_ctx += "  If a reported value falls outside the expected range, it may be a different scale.\n"

            tp = extraction_guidance.get("preferred_timepoint", "")
            tp_list = extraction_guidance.get("timepoint_priority", [])
            if tp:
                guidance_ctx += f"\nPREFERRED TIMEPOINT: {tp}\n"
            if tp_list:
                guidance_ctx += f"TIMEPOINT PRIORITY (in order): {', '.join(tp_list)}\n"

            crossover = extraction_guidance.get("crossover_handling", "")
            if crossover:
                guidance_ctx += f"CROSSOVER STUDIES: {crossover}\n"

            multi_arm = extraction_guidance.get("multi_arm_handling", "")
            if multi_arm:
                guidance_ctx += f"MULTI-ARM STUDIES: {multi_arm}\n"

            special = extraction_guidance.get("special_instructions", [])
            if special:
                guidance_ctx += "SPECIAL INSTRUCTIONS:\n"
                for s in special:
                    guidance_ctx += f"  - {s}\n"

        prompt = f"""Extract data from this study for a meta-analysis.

STUDY: {study.get('title','')} ({study.get('year','')})
ID: {study.get('study_id','')}

OUTCOMES TO EXTRACT:
{json.dumps(outcome_definitions, indent=2)}
{guidance_ctx}
TEXT:
{fulltext}
{table_ctx}

CRITICAL INSTRUCTIONS:
- Use the EXACT MEASURE NAME as the outcome key (e.g., "MMSE", "MoCA", "ADAS-Cog")
- If a study reports multiple timepoints, extract the PREFERRED TIMEPOINT specified above (default: post-treatment immediate). Do NOT mix timepoints within the same outcome.
- If the paper reports SE instead of SD, convert: SD = SE × sqrt(n)
- For change scores: report mean change and SD of change
- For post-treatment scores: report post-treatment mean and SD
- For crossover studies: follow the crossover handling rule above (default: first period only, before crossover)
- If a value falls outside the expected scale range for a measure, verify it is the correct measure. Report it under the correct measure name.

Respond with JSON:
{{
  "study_id": "{study.get('study_id','')}",
  "citation": "<First Author> et al., <Journal> <Year>",
  "characteristics": {{
    "design": "RCT/crossover/etc",
    "blinding": "double-blind/single-blind/open-label",
    "n_total": null,
    "n_randomized": null,
    "n_analyzed_itt": null,
    "duration_weeks": null,
    "setting": "",
    "country": "",
    "registration": "",
    "funding": ""
  }},
  "population": {{
    "diagnosis": "",
    "diagnostic_criteria": "",
    "age_mean": null, "age_sd": null,
    "sex_male_percent": null,
    "education_years_mean": null,
    "baseline_severity": ""
  }},
  "intervention": {{
    "type": "",
    "protocol": "",
    "target_area": "",
    "intensity": "",
    "session_duration_min": null,
    "sessions_total": null,
    "sessions_per_week": null,
    "total_duration_weeks": null,
    "combined_with": "none",
    "n_randomized": null, "n_analyzed": null
  }},
  "comparison": {{
    "type": "sham/active/waitlist/placebo",
    "description": "",
    "n_randomized": null, "n_analyzed": null
  }},
  "outcomes": {{
{outcome_examples}
  }},
  "adverse_events": {{
    "any_ae_intervention": null, "any_ae_control": null,
    "any_ae_intervention_total": null, "any_ae_control_total": null,
    "serious_ae_intervention": null, "serious_ae_control": null,
    "dropout_intervention": null, "dropout_control": null,
    "dropout_intervention_total": null, "dropout_control_total": null,
    "specific_aes": ""
  }},
  "extraction_notes": "",
  "data_completeness": "complete/partial/minimal"
}}"""
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            expect_json=True,
            cache_namespace="phase4_extraction_v5",
            description=f"Extract: {study.get('study_id','')}",
        )
        
        extracted = result.get("parsed") or {
            "study_id": study.get("study_id", ""), "error": "extraction_failed"
        }
        
        if "error" not in extracted:
            extracted = self._back_calculate_se(extracted)
        
        return extracted
    
    def _build_outcome_template(self, outcome_defs: dict) -> str:
        """
        Build a concise outcome template. Lists all measures but only shows
        the full field structure once as an example.
        """
        # Collect all measure names
        all_measures = []
        for category in ["primary", "secondary"]:
            for outcome in outcome_defs.get(category, []):
                for m in outcome.get("measures", []):
                    if m not in all_measures:
                        all_measures.append(m)
        
        # Build concise template: one full example + list the rest
        example_measure = all_measures[0] if all_measures else "outcome_measure"
        other_measures = all_measures[1:6]  # limit to avoid bloat
        
        lines = []
        lines.append(f'    "{example_measure}": {{')
        lines.append(f'      "measure": "{example_measure}",')
        lines.append(f'      "timepoint": "post-treatment",')
        lines.append(f'      "intervention_mean_post": null, "intervention_sd_post": null,')
        lines.append(f'      "intervention_mean_change": null, "intervention_sd_change": null,')
        lines.append(f'      "intervention_n": null,')
        lines.append(f'      "control_mean_post": null, "control_sd_post": null,')
        lines.append(f'      "control_mean_change": null, "control_sd_change": null,')
        lines.append(f'      "control_n": null,')
        lines.append(f'      "mean_difference": null, "md_95ci_lower": null, "md_95ci_upper": null,')
        lines.append(f'      "p_value": null, "spread_type_reported": "SD/SE/95CI",')
        lines.append(f'      "source_location": {{"section": "Table 2 / Results / etc.", "quote": "exact sentence or data cell verbatim from PDF", "page": null}}')
        lines.append(f'    }}')
        
        if other_measures:
            comment = "    // Use the SAME structure for: " + ", ".join(other_measures)
            lines.append(comment)
        
        return "\n".join(lines)
    
    def _back_calculate_se(self, data: dict) -> dict:
        from src.utils.effect_sizes import se_from_ci, se_from_p_value
        
        for name, od in data.get("outcomes", {}).items():
            if od.get("md_se") is None and od.get("mean_difference") is not None:
                ci_l, ci_u = od.get("md_95ci_lower"), od.get("md_95ci_upper")
                if ci_l is not None and ci_u is not None:
                    od["md_se"] = round(se_from_ci(ci_l, ci_u), 4)
                    od["md_se_method"] = "back_calculated_from_ci"
                elif od.get("p_value") and od["p_value"] > 0:
                    se = se_from_p_value(od["mean_difference"], od["p_value"])
                    if se:
                        od["md_se"] = round(se, 4)
                        od["md_se_method"] = "back_calculated_from_p"
            
            for suffix in ["_change", "_post"]:
                m1 = od.get(f"intervention_mean{suffix}")
                s1 = od.get(f"intervention_sd{suffix}")
                n1 = od.get("intervention_n")
                m2 = od.get(f"control_mean{suffix}")
                s2 = od.get(f"control_sd{suffix}")
                n2 = od.get("control_n")
                
                if all(v is not None for v in [m1, s1, n1, m2, s2, n2]):
                    if od.get("mean_difference") is None:
                        od["mean_difference"] = round(m1 - m2, 4)
                        od["md_computed_from"] = f"group_means{suffix}"
                    if od.get("md_se") is None:
                        # Guard against n=0 (extraction error) to avoid ZeroDivisionError
                        if n1 > 0 and n2 > 0:
                            od["md_se"] = round(float(np.sqrt(s1**2/n1 + s2**2/n2)), 4)
                            od["md_se_method"] = f"computed_from_group_sd{suffix}"
                        else:
                            logger.warning(
                                f"Skipping SE calculation for {name}: "
                                f"n1={n1}, n2={n2} (zero sample size extracted)"
                            )
                    break
        
        return data
    
    def assess_risk_of_bias(self, study: dict, fulltext: str) -> dict:
        system_prompt = (
            "You are an expert in Cochrane RoB 2 assessment. "
            "Answer signaling questions and cite evidence. "
            "Respond ONLY with valid JSON."
        )
        
        prompt = f"""Assess Risk of Bias (Cochrane RoB 2) for this RCT.

STUDY: {study.get('title','')} ({study.get('year','')})

TEXT:
{fulltext[:8000]}

Respond with JSON:
{{
  "study_id": "{study.get('study_id','')}",
  "rob2_domains": {{
    "D1_randomization": {{
      "judgement": "low/some_concerns/high",
      "support": "justification with quoted evidence"
    }},
    "D2_deviations": {{
      "judgement": "low/some_concerns/high",
      "support": "justification"
    }},
    "D3_missing_data": {{
      "judgement": "low/some_concerns/high",
      "support": "justification"
    }},
    "D4_measurement": {{
      "judgement": "low/some_concerns/high",
      "support": "justification"
    }},
    "D5_selection_reporting": {{
      "judgement": "low/some_concerns/high",
      "support": "justification"
    }}
  }},
  "overall_rob": "low/some_concerns/high",
  "overall_justification": ""
}}"""
        
        result = self.call_llm(
            prompt=prompt, system_prompt=system_prompt,
            expect_json=True,
            cache_namespace="phase4_rob_v5",
            description=f"RoB: {study.get('study_id','')}",
        )
        
        return result.get("parsed") or {
            "study_id": study.get("study_id", ""), "error": "rob_failed"
        }


# ── Inline fallback system prompt (mirrors config/prompts/extractor.yaml) ────

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert data extractor for systematic reviews. "
    "RULES:\n"
    "1) Copy numbers EXACTLY as reported in the paper.\n"
    "2) Use null if not reported or not found in the text.\n"
    "3) For each outcome, use the MEASURE NAME as the JSON key "
    "(e.g., 'MMSE', 'MoCA', 'ADAS-Cog', 'NPI').\n"
    "4) Report SD, not SE. If the paper reports SE, convert: SD = SE × √n.\n"
    "5) For source_location, provide a structured object: "
    "{\"section\": \"Table 2\", \"quote\": \"exact sentence verbatim\", \"page\": 4}. "
    "Do NOT use a plain string.\n"
    "6) Respond ONLY with valid JSON, no extra text."
)
