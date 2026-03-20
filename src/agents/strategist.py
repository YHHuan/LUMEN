"""
Research Strategist Agent (Phase 1) — v5
==========================================
LLM 負責:
  1. 生成 MeSH terms + free text terms
  2. 為每個 free text term 標記 scope: "broad" / "specific"
  3. 生成 prescreen_exclusion_keywords (Phase 2.5 用)

Python query_syntax 負責:
  組裝正確的 database-specific query syntax
"""

import json
import logging
from src.agents.base_agent import BaseAgent
from src.utils.query_syntax import generate_all_queries

logger = logging.getLogger(__name__)


class StrategistAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(role_name="strategist", **kwargs)
        # Load system prompt from YAML; fall back to inline default if file missing
        self._system_prompt = self._prompt_config.get(
            "system_prompt", _DEFAULT_STRATEGIST_SYSTEM
        )

    def generate_search_strategy(self, pico: dict) -> dict:
        """
        1. Ask LLM for terms with scope labels + prescreen keywords.
        2. Use Python to compile valid database queries.
        """
        system_prompt = self._system_prompt

        prompt = f"""Based on this PICO framework, generate comprehensive search terms.

PICO:
Population: {pico.get('population', '')}
Intervention: {pico.get('intervention', '')}
Comparison: {pico.get('comparison', '')}
Outcome: {json.dumps(pico.get('outcome', {}), indent=2)}

Study designs: {pico.get('study_design', {}).get('include', [])}
Date range: {pico.get('date_range', {})}
Languages: {pico.get('language', [])}

IMPORTANT:
- Generate BROAD, INCLUSIVE terms
- Include all common abbreviations, full spelled-out terms, alternative spellings
- For each free text term, label scope as "broad" or "specific"
- Provide prescreen_exclusion_keywords for obvious non-relevant studies

Respond with JSON:
{{
  "mesh_terms": {{
    "population": ["Valid MeSH Heading 1", "Valid MeSH Heading 2"],
    "intervention": ["Valid MeSH Heading 1", "Valid MeSH Heading 2"]
  }},
  "free_text_terms": {{
    "population": [
      {{"term": "example specific term", "scope": "specific"}},
      {{"term": "example broad umbrella term", "scope": "broad"}}
    ],
    "intervention": [
      {{"term": "exact technique name", "scope": "specific"}},
      {{"term": "general category", "scope": "broad"}}
    ]
  }},
  "study_design_filter": {{
    "required_designs": ["RCT", "CCT"],
    "positive_keywords": ["randomized", "controlled trial", "sham"],
    "design_exclusion_keywords": ["non-interventional", "observational"],
    "filter_mode": "strict"
  }},
  "extraction_guidance": {{
    "outcome_measures": [
      {{"abbreviation": "MMSE", "full_name": "Mini-Mental State Examination", "scale_range": "0-30", "direction": "higher_is_better"}},
      {{"abbreviation": "ADAS-Cog", "full_name": "Alzheimer's Disease Assessment Scale-Cognitive", "scale_range": "0-70", "direction": "lower_is_better"}}
    ],
    "preferred_timepoint": "post-treatment immediate",
    "timepoint_priority": ["post-treatment", "end-of-intervention", "4-week follow-up"],
    "crossover_handling": "first_period_only",
    "multi_arm_handling": "extract_all_arms",
    "special_instructions": []
  }},
  "prescreen_exclusion_keywords": [
    "rat", "rats", "mouse", "mice", "animal", "animal model",
    "review", "meta-analysis", "systematic review",
    "protocol", "editorial", "commentary", "letter", "erratum",
    "case report", "case series", "in vitro", "cell culture"
  ],
  "inclusion_criteria": ["criterion 1", "criterion 2"],
  "exclusion_criteria": ["criterion 1", "criterion 2"],
  "prisma_protocol": {{
    "title": "",
    "objectives": "",
    "eligibility": "",
    "information_sources": "",
    "search_strategy_description": "",
    "study_selection_process": "",
    "data_extraction_plan": "",
    "risk_of_bias_tool": "Cochrane RoB 2",
    "statistical_methods": ""
  }},
  "subgroup_analyses": [
    {{"variable": "...", "groups": ["...", "..."]}}
  ],
  "sensitivity_analyses": ["..."]
}}

IMPORTANT for study_design_filter:
- "required_designs": List of acceptable study designs based on the PICO. Use ["any"] if the review includes all study types.
- "positive_keywords": Keywords that indicate an acceptable study design (searched in title+abstract). Only include if filter_mode is "strict".
- "design_exclusion_keywords": Study design terms to exclude. Only include domain-specific ones that are definitely out of scope for THIS review. Do NOT include generic terms like "observational" unless the PICO explicitly restricts to RCTs.
- "filter_mode": "strict" (only include studies matching positive_keywords) or "loose" (only exclude studies matching design_exclusion_keywords). Use "strict" for RCT-only reviews, "loose" for broader reviews.

IMPORTANT for extraction_guidance:
- "outcome_measures": List ALL expected outcome measures from the PICO. For each, provide the standard abbreviation, full name, scale range (if applicable), and scoring direction. This helps the downstream extractor distinguish between similarly-named scales.
- "preferred_timepoint": Which timepoint to prioritize when a study reports multiple (e.g., "post-treatment immediate", "end-of-intervention", "longest follow-up").
- "timepoint_priority": Ordered list of timepoint preferences for fallback.
- "crossover_handling": How to handle crossover studies: "first_period_only" (extract before crossover), "pooled_periods" (use pooled if available), or "all_periods_separate".
- "multi_arm_handling": How to handle multi-arm studies: "extract_all_arms" (extract each arm vs control separately) or "primary_comparison_only".
- "special_instructions": Any domain-specific extraction notes (e.g., "distinguish between 3MS (0-100) and MMSE (0-30)", "NPI subscale scores are not needed")."""
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            expect_json=True,
            cache_namespace="phase1_strategy_v5",
            description="Generate search terms with scope labels",
        )
        
        parsed = result.get("parsed")
        if not parsed:
            logger.error("LLM failed to return valid JSON.")
            return {"error": "Failed to parse strategy", "raw": result.get("content")}
        
        # === Extract terms and broad set from LLM response ===
        free_text = parsed.get("free_text_terms", {})
        mesh = parsed.get("mesh_terms", {})
        
        pop_terms, pop_broad = self._parse_scoped_terms(free_text.get("population", []))
        int_terms, int_broad = self._parse_scoped_terms(free_text.get("intervention", []))
        broad_set = pop_broad | int_broad
        
        logger.info(f"📋 Population terms: {len(pop_terms)} ({len(pop_broad)} broad)")
        logger.info(f"📋 Intervention terms: {len(int_terms)} ({len(int_broad - pop_broad)} broad)")
        if broad_set:
            logger.info(f"📋 Broad terms (title-only): {sorted(broad_set)}")
        
        prescreen_kw = parsed.get("prescreen_exclusion_keywords", [])
        logger.info(f"📋 Prescreen exclusion keywords: {len(prescreen_kw)}")
        
        # === Build queries ===
        date_range = pico.get("date_range", {})
        languages = pico.get("language", ["English"])
        study_types = pico.get("study_design", {}).get("include", [])
        
        queries = generate_all_queries(
            population_terms=pop_terms,
            intervention_terms=int_terms,
            mesh_population=mesh.get("population", []),
            mesh_intervention=mesh.get("intervention", []),
            date_start=str(date_range.get("start", "2000"))[:4],
            date_end=str(date_range.get("end", "2025"))[:4],
            languages=languages,
            study_types=study_types,
            broad_terms=broad_set,
        )
        
        parsed["search_queries"] = queries
        parsed["_flat_terms"] = {
            "population": pop_terms,
            "intervention": int_terms,
            "broad_terms": sorted(broad_set),
        }
        
        return parsed
    
    @staticmethod
    def _parse_scoped_terms(term_list) -> tuple:
        """
        Parse LLM output:
          - New: [{"term": "...", "scope": "broad"}, ...]
          - Old: ["term1", "term2", ...]  (fallback, all specific)
        Returns: (list_of_strings, set_of_broad_terms)
        """
        terms = []
        broad = set()
        
        for item in term_list:
            if isinstance(item, dict):
                t = item.get("term", "")
                scope = item.get("scope", "specific")
                if t:
                    terms.append(t)
                    if scope == "broad":
                        broad.add(t.lower())
            elif isinstance(item, str):
                terms.append(item)
        
        return terms, broad
    
    def validate_mesh_terms(self, mesh_terms: dict) -> dict:
        """Validate MeSH terms"""
        system_prompt = self._prompt_config.get(
            "mesh_validation_system_prompt",
            "You are a MeSH terminology expert. "
            "Validate and correct the given MeSH terms. "
            "Respond ONLY with valid JSON."
        )
        
        prompt = f"""Review these MeSH terms for a systematic review search strategy.
For each term, confirm if it's a valid MeSH heading, and suggest corrections or alternatives.

MeSH terms to validate:
{json.dumps(mesh_terms, indent=2)}

Respond with JSON:
{{
  "validated_terms": {{
    "population": [
      {{"term": "...", "status": "valid|corrected|suggested", "note": "..."}}
    ],
    "intervention": [...]
  }},
  "additional_suggestions": ["..."]
}}"""
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            expect_json=True,
            cache_namespace="phase1_mesh_validation",
            description="Validate MeSH terms",
        )
        
        return result.get("parsed") or {}


# ── Inline fallback (used when config/prompts/strategist.yaml is missing) ────
_DEFAULT_STRATEGIST_SYSTEM = (
    "You are an expert systematic review methodologist and information specialist. "
    "Your task is to identify comprehensive search TERMS (not full queries). "
    "The actual query syntax for each database will be handled separately.\n\n"
    "CRITICAL: For each free text term, you MUST classify its 'scope':\n"
    "  - 'broad': Very general terms (searched in TITLE ONLY to reduce noise).\n"
    "  - 'specific': Precise terms (searched in TITLE + ABSTRACT).\n\n"
    "When in doubt, label as 'specific'.\n\n"
    "Respond ONLY with valid JSON, no markdown fences."
)
