"""
Manuscript Writer Agent (Phase 6) — v5
=========================================
Multi-round writing with deep data context:
1. Initial draft per section (rich prompts with actual data)
2. Citation cross-validation against study abstracts
3. Final polish

Key design:
- Discussion/Introduction prompts include ALL statistical results,
  study characteristics, intervention breakdowns, and PRISMA numbers
- Citation Guardian validates claims against original abstracts
- Multi-call for long sections (split into subsections)
"""

import json
import logging
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# ── Utility: truncate large JSON for prompt ──

def _compact_json(obj, max_chars=3000):
    """JSON dump with size limit."""
    s = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    if len(s) > max_chars:
        return s[:max_chars] + "\n... [truncated]"
    return s


class WriterAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(role_name="writer", **kwargs)
        # Load system prompt from YAML; fall back to inline default if file missing
        self._cached_system_prompt = self._prompt_config.get(
            "system_prompt", self._system_prompt()
        )

    # ── Public API ──

    def write_section(self, section_name: str, context: dict) -> str:
        system_prompt = self._cached_system_prompt
        prompt = self._build_prompt(section_name, context)
        if not prompt:
            raise ValueError(f"Unknown section: {section_name}")

        result = self.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            expect_json=False,
            cache_namespace=f"phase6_{section_name}",
            description=f"Write {section_name}",
        )
        return result["content"]

    # ── System prompt ──

    def _system_prompt(self):
        return (
            "You are an expert academic medical writer specializing in systematic reviews "
            "and meta-analyses. You write following PRISMA 2020 guidelines.\n\n"
            "RULES:\n"
            "1. Use ONLY the data provided in the context. NEVER fabricate statistics.\n"
            "2. When referencing claims that need external literature support, write "
            "   [CITATION NEEDED: topic description] so the Citation Guardian can fill them.\n"
            "3. When referencing included studies, use the study_id (e.g., PMID_33581283).\n"
            "4. Use precise scientific language. Hedging when evidence is limited.\n"
            "5. Write in English. Use Hedges' g (not Cohen's d) for standardised effect sizes.\n"
            "6. Report exact numbers: k, g, 95% CI, p, I² from the data provided.\n"
            "7. The Discussion and Introduction should each be 1500-2500 words, with clearly "
            "   labeled subsection headings (### Markdown)."
        )

    # ── Prompt router ──

    def _build_prompt(self, section, ctx):
        builders = {
            "title": self._prompt_title,
            "abstract": self._prompt_abstract,
            "introduction": self._prompt_introduction,
            "methods": self._prompt_methods,
            "results": self._prompt_results,
            "discussion": self._prompt_discussion,
        }
        fn = builders.get(section)
        return fn(ctx) if fn else None

    # ════════════════════════════════════════
    #  TITLE
    # ════════════════════════════════════════

    def _prompt_title(self, ctx):
        return f"""Write a concise title for this systematic review and meta-analysis.

Topic: {ctx['pico_summary']}
Primary finding: {ctx.get('primary_finding', '')}

Format: "[Intervention] for [Population]: A Systematic Review and Meta-Analysis"
Return ONLY the title text, no quotes."""

    # ════════════════════════════════════════
    #  ABSTRACT
    # ════════════════════════════════════════

    def _prompt_abstract(self, ctx):
        return f"""Write a structured abstract (max 350 words) with sections:
Background, Objectives, Methods, Results, Conclusions.

{self._core_data_block(ctx)}

Be precise with numbers. Report exact k, effect sizes, CIs, p-values."""

    # ════════════════════════════════════════
    #  INTRODUCTION
    # ════════════════════════════════════════

    def _prompt_introduction(self, ctx):
        # Extract PICO elements for structured prompt
        pico_raw = ctx.get('pico_raw', {})
        pico = pico_raw.get('pico', pico_raw)
        population = pico.get('population', '')
        intervention = pico.get('intervention', '')
        comparison = pico.get('comparison', '')
        outcomes = pico.get('outcome', {})
        primary_outcomes = outcomes.get('primary', [])
        secondary_outcomes = outcomes.get('secondary', [])

        # Build outcome list
        primary_names = ", ".join(o.get('name', '') for o in primary_outcomes if isinstance(o, dict))
        secondary_names = ", ".join(o.get('name', '') for o in secondary_outcomes if isinstance(o, dict))

        return f"""Write the Introduction section (1500-2000 words, 5-6 paragraphs with subsection headings).

=== REVIEW TOPIC ===
{ctx['pico_summary']}

=== YOUR DATA CONTEXT (use to motivate the review) ===
- Studies included: {ctx['n_studies']}
- Total participants: {ctx['study_chars'].get('total_participants', '?')}
- Intervention types studied: {_compact_json(ctx['study_chars'].get('intervention_types', {}))}
- Population types: {_compact_json(ctx['study_chars'].get('population_types', {}))}
- Primary outcomes: {primary_names}
- Secondary outcomes: {secondary_names}
- Databases searched: {ctx.get('databases', [])}
- Primary finding preview: {ctx.get('primary_finding', 'see statistical results')}

=== REQUIRED STRUCTURE ===

### Disease Burden and Clinical Context
- Prevalence and impact of the target population/condition
- Current standard treatments and their limitations
- Unmet clinical need that motivates exploring the intervention
[Use [CITATION NEEDED: topic] for epidemiological and clinical claims]

### The Intervention: Rationale and Mechanisms
- What is the intervention: describe the approach and variants
- Theoretical rationale: why this intervention might work for this population
- Key parameters and targets studied
- Emerging or innovative approaches within this intervention class
[Use [CITATION NEEDED: topic] for mechanistic claims]

### Existing Evidence and Its Limitations
- Previous systematic reviews and meta-analyses on this topic
- What they found: general direction of evidence
- Key limitations of prior reviews:
  * Population scope (too broad or narrow?)
  * Intervention scope (single modality vs. comprehensive?)
  * Recency (missing recent trials?)
  * Methodological issues
[Use [CITATION NEEDED: specific review name] when referencing prior reviews]

### Knowledge Gap and Objectives
- What specific gap this review addresses
- Why a new/updated review is needed now
- Objectives: (1) synthesize evidence on [intervention] for [population],
  (2) assess effects on [primary outcomes] and [secondary outcomes],
  (3) explore sources of heterogeneity through subgroup analyses
- Protocol registration if applicable: [CITATION NEEDED: PROSPERO if registered]

=== POPULATION-SPECIFIC CONTEXT ===
Population: {population[:300]}
Intervention: {intervention[:300]}

=== WRITING GUIDELINES ===
- Build a logical narrative arc: burden → rationale → prior evidence → gap → objectives
- Each paragraph should flow naturally into the next
- Do NOT include any results from this review (save for Results/Discussion)
- Mark ALL factual claims about prevalence, mechanisms, or prior reviews with [CITATION NEEDED: description]
- Aim for authoritative but accessible scientific prose
- Tailor the content specifically to the population and intervention described above"""

    # ════════════════════════════════════════
    #  METHODS
    # ════════════════════════════════════════

    def _prompt_methods(self, ctx):
        return f"""Write the Methods section following PRISMA 2020.

Search strategy: {_compact_json(ctx.get('search_strategy', {}))}
Databases: {ctx.get('databases', [])}
Inclusion: {ctx.get('inclusion_criteria', [])}
Exclusion: {ctx.get('exclusion_criteria', [])}
PRISMA flow: {_compact_json(ctx.get('prisma_flow', {}))}
Screening: Dual independent screening with conflict resolution
RoB tool: Cochrane RoB 2
Statistical: Random-effects (DerSimonian-Laird), Hedges' g, I²/Q/tau²
Subgroups: {ctx.get('subgroups', [])}

Subsections: Protocol, Eligibility, Information sources, Search strategy,
Selection process, Data collection, RoB assessment, Effect measures, Synthesis methods."""

    # ════════════════════════════════════════
    #  RESULTS
    # ════════════════════════════════════════

    def _prompt_results(self, ctx):
        return f"""Write the Results section.

PRISMA flow: {_compact_json(ctx.get('prisma_flow', {}))}
Study characteristics: {_compact_json(ctx['study_chars'])}
All statistical results: {_compact_json(ctx.get('all_analyses', {}))}
RoB summary: {_compact_json(ctx.get('rob_summary', {}))}

Include: Study selection, Study characteristics, RoB, Synthesis results (per outcome),
Subgroup analyses, Sensitivity analyses, Publication bias.
Reference: Figure 1 (PRISMA), Figure 2 (Forest), Figure 3 (Funnel), Figure 4 (RoB).
Use EXACT numbers from data."""

    # ════════════════════════════════════════
    #  DISCUSSION  — the key upgrade
    # ════════════════════════════════════════

    def _prompt_discussion(self, ctx):
        # Build dynamic analysis summary for the prompt
        sig_analyses = []
        nonsig_analyses = []
        for k, v in ctx.get('all_analyses', {}).items():
            entry = f"{k}: k={v['k']}, g={v['pooled_g']:.3f} [{v['ci_lower']:.3f},{v['ci_upper']:.3f}], p={v['p_value']:.4f}, I²={v['I2']}"
            if v.get('significant'):
                sig_analyses.append(entry)
            else:
                nonsig_analyses.append(entry)

        sig_summary = "\n  ".join(sig_analyses) if sig_analyses else "(none reached significance)"
        nonsig_summary = "\n  ".join(nonsig_analyses) if nonsig_analyses else "(all significant)"

        # Extract intervention/population from PICO for context
        pico_raw = ctx.get('pico_raw', {})
        pico = pico_raw.get('pico', pico_raw)
        population = pico.get('population', '')
        intervention = pico.get('intervention', '')

        return f"""Write the Discussion section (2000-2500 words, 7-8 subsections with ### headings).

=== REVIEW TOPIC ===
Population: {population[:300]}
Intervention: {intervention[:300]}

=== ALL META-ANALYSIS RESULTS ===
{_compact_json(ctx.get('all_analyses', {}))}

Significant analyses:
  {sig_summary}
Non-significant analyses:
  {nonsig_summary}

=== STUDY CHARACTERISTICS ===
- Total studies: {ctx['n_studies']}
- Total participants: {ctx['study_chars'].get('total_participants', '?')}
- Study designs: {_compact_json(ctx['study_chars'].get('study_designs', {}))}
- Intervention types: {_compact_json(ctx['study_chars'].get('intervention_types', {}))}
- Population types: {_compact_json(ctx['study_chars'].get('population_types', {}))}
- Session range: {ctx['study_chars'].get('session_range', {})}

=== RISK OF BIAS ===
{_compact_json(ctx.get('rob_summary', {}))}

=== PRISMA FLOW ===
{_compact_json(ctx.get('prisma_flow', {}))}

=== STUDIES IN PRIMARY ANALYSES ===
{_compact_json(ctx.get('analysis_study_details', {}))}

=== REQUIRED STRUCTURE ===

### Summary of Main Findings
- Lead with the strongest (significant) finding — report exact g, CI, p, k
- Report non-significant results and explain why (small k, high I², etc.)
- Report all secondary outcomes with exact numbers
- Note the overall pattern of evidence (promising but heterogeneous, robust, mixed, etc.)

### Comparison with Existing Literature
- Compare your effect sizes with prior meta-analyses and systematic reviews on this topic
  [CITATION NEEDED: prior meta-analysis 1]
  [CITATION NEEDED: prior meta-analysis 2]
- Explain if your effects are larger or smaller than previously reported, and why
  (different population scope, inclusion of newer trials, broader intervention coverage, etc.)
- Discuss any null results in context of prior positive/negative findings

### Possible Mechanisms
- Explain the theoretical mechanisms of action for each intervention type
- Discuss why the target population may be particularly responsive (or resistant)
- If multiple intervention modalities show benefit, discuss shared downstream mechanisms
  [CITATION NEEDED: key mechanistic references]

### Strengths
- Comprehensive search: {len(ctx.get('databases', []))} databases, {ctx.get('prisma_flow', {}).get('total_identified', '?')} records identified
- Rigorous screening: dual independent + conflict resolution
- Broad scope of intervention types included
- Standardized effect sizes: Hedges' g with random-effects modeling
- PRISMA 2020 adherence
- RoB assessment with Cochrane RoB 2

### Limitations
- **Heterogeneity**: discuss I² values and their likely sources
  (different intervention parameters, populations, outcome timepoints)
- **Limited k**: identify which analyses were underpowered
- **PDF availability**: {ctx.get('prisma_flow', {}).get('fulltext_no_pdf', 0)} of {ctx.get('prisma_flow', {}).get('fulltext_assessed', 0)} studies had no retrievable full text
- **Data completeness**: Only {ctx.get('n_computable', '?')} of {ctx['n_studies']} studies provided
  fully computable data (means, SDs, Ns) — rest were abstract-only or incomplete reporting
- **Blinding adequacy**: discuss whether sham/placebo blinding was convincing
  [CITATION NEEDED: blinding adequacy in this intervention type]
- **RoB**: {ctx.get('rob_summary', {}).get('n_assessed', '?')} studies assessed — report distribution
- **Publication bias**: if k < 10 for most analyses, note formal testing was precluded

### Clinical Implications
- Translate the significant effect size(s) into clinically meaningful terms
  [CITATION NEEDED: standard-of-care effect size for comparison]
- Discuss flexibility in intervention selection if multiple types showed benefit
- Comment on dosing/protocol implications from the session range data
- Discuss whether benefits extend beyond primary outcomes (functional, quality of life)

### Future Research Directions
- Standardized protocols: consensus on optimal parameters
- Better reporting: complete statistical data (means, SDs, Ns) in all trials
- Biomarker integration: neuroimaging or lab markers as predictors of response
- Long-term durability: most included studies only assessed short-term outcomes
- Individual patient data (IPD) meta-analysis when more data available
- Head-to-head comparisons between intervention subtypes
  [CITATION NEEDED: relevant reporting guidelines]

### Conclusion
- 2-3 sentence conclusion summarizing the key finding
- Temper with limitations (heterogeneity, limited k)
- Call for standardized protocols and comprehensive reporting

=== WRITING GUIDELINES ===
- Use exact numbers from the data provided. Do NOT round or approximate.
- Each subsection should be 200-400 words.
- Build argumentative flow: findings → context → mechanisms → balance → implications → future
- Be balanced: acknowledge both significant findings AND limitations
- Mark external claims with [CITATION NEEDED: description]
- Reference included studies by study_id when discussing specific findings
- The tone should be scholarly but accessible, suitable for a clinical journal
- Tailor all content to the specific population and intervention described above"""

    # ── Helper: core data block for abstract/short sections ──

    def _core_data_block(self, ctx):
        return f"""PICO: {ctx['pico_summary']}
Studies: {ctx['n_studies']}, Participants: {ctx['study_chars'].get('total_participants', '?')}
PRISMA: {_compact_json(ctx.get('prisma_flow', {}))}
Results: {_compact_json(ctx.get('all_analyses', {}))}
RoB: {_compact_json(ctx.get('rob_summary', {}))}"""


# ═══════════════════════════════════════════
#  CITATION GUARDIAN — Cross-validates claims
# ═══════════════════════════════════════════

class CitationGuardianAgent(BaseAgent):
    """
    Validates claims in the manuscript against original study abstracts.
    
    Two modes:
    1. validate_internal() — check claims about included studies against their abstracts
    2. flag_external() — identify [CITATION NEEDED] markers and suggest specific references
    """

    def __init__(self, **kwargs):
        super().__init__(role_name="citation_guardian", **kwargs)

    def validate_internal_citations(self, manuscript_text: str,
                                     study_abstracts: dict) -> list:
        """
        Find study_id references in text and verify claims against abstracts.
        
        Args:
            manuscript_text: Full manuscript text
            study_abstracts: {study_id: abstract_text}
        
        Returns:
            List of validation results
        """
        import re
        # Find all study_id mentions
        study_ids = set(re.findall(
            r'(PMID_\d+|SCOPUS_[\d\w-]+|RIS_cochrane_\d+|DOI_[\d\w._-]+)',
            manuscript_text
        ))

        results = []
        for sid in study_ids:
            abstract = study_abstracts.get(sid, "")
            if not abstract:
                results.append({
                    "study_id": sid, "status": "no_abstract",
                    "message": "No abstract available for validation"
                })
                continue

            # Extract the sentences that mention this study_id
            claims = self._extract_claims_for_study(manuscript_text, sid)
            if not claims:
                continue

            prompt = f"""You are a citation accuracy checker for a systematic review manuscript.

The manuscript makes these claims about study {sid}:
{json.dumps(claims, indent=2)}

Here is the original study abstract:
{abstract[:2500]}

For EACH claim, verify if the manuscript accurately represents the original study.
Respond with JSON array:
[
  {{
    "claim": "the claim text",
    "accurate": true/false,
    "issue": "description of inaccuracy, or null if accurate",
    "suggestion": "corrected text if inaccurate, or null"
  }}
]

Be strict. Flag any misrepresentation of findings, sample size, intervention type,
or outcome measures. Return ONLY the JSON array."""

            result = self.call_llm(
                prompt=prompt,
                system_prompt="Respond ONLY with a JSON array. Be strict about accuracy.",
                expect_json=True,
                cache_namespace="phase6_citation_validate",
                description=f"Validate citations for {sid}",
            )

            parsed = result.get("parsed")
            if isinstance(parsed, list):
                for item in parsed:
                    item["study_id"] = sid
                results.extend(parsed)
            else:
                results.append({
                    "study_id": sid, "status": "parse_error",
                    "raw": result.get("content", "")[:200]
                })

        return results

    def collect_citation_needed(self, manuscript_text: str) -> list:
        """
        Extract all [CITATION NEEDED: ...] markers from the manuscript.
        Returns list of {"marker": str, "topic": str, "location": str}
        """
        import re
        markers = []
        for m in re.finditer(r'\[CITATION NEEDED:?\s*([^\]]*)\]', manuscript_text):
            # Get surrounding context
            start = max(0, m.start() - 100)
            end = min(len(manuscript_text), m.end() + 100)
            context = manuscript_text[start:end].replace('\n', ' ')
            markers.append({
                "marker": m.group(0),
                "topic": m.group(1).strip(),
                "context": context,
            })
        return markers

    def suggest_references(self, citation_markers: list,
                           included_studies: list) -> list:
        """
        For each [CITATION NEEDED], suggest whether any included study
        can serve as the reference, or mark as needing external reference.
        """
        # Build a compact reference of included studies
        study_refs = []
        for s in included_studies[:60]:
            study_refs.append({
                "id": s.get("study_id", ""),
                "title": (s.get("title") or "")[:100],
                "year": s.get("year", ""),
                "journal": (s.get("journal") or "")[:50],
            })

        prompt = f"""You are helping fill citation placeholders in a systematic review manuscript.

Here are [CITATION NEEDED] markers that need references:
{json.dumps(citation_markers[:20], indent=2)}

Here are the included studies in this review (potential internal references):
{_compact_json(study_refs, max_chars=4000)}

For each marker, determine:
1. Can an included study serve as the reference? If yes, suggest the study_id.
2. If no included study fits, suggest what TYPE of external reference is needed
   (e.g., "epidemiological review", "WHO report", "prior meta-analysis by Dong et al. 2018").

Respond with JSON array:
[
  {{
    "marker": "[CITATION NEEDED: ...]",
    "type": "internal" or "external",
    "suggested_id": "PMID_... (if internal, else null)",
    "external_suggestion": "description of needed reference (if external, else null)"
  }}
]"""

        result = self.call_llm(
            prompt=prompt,
            system_prompt="Respond ONLY with JSON array.",
            expect_json=True,
            cache_namespace="phase6_citation_suggest",
            description="Suggest references for CITATION NEEDED markers",
        )

        return result.get("parsed") or []

    @staticmethod
    def _extract_claims_for_study(text, study_id):
        """Extract sentences that reference a specific study_id."""
        sentences = []
        for line in text.split('\n'):
            if study_id in line:
                # Get the full sentence(s) containing the ID
                for sent in line.split('. '):
                    if study_id in sent:
                        sentences.append(sent.strip())
        return sentences[:5]  # Max 5 claims per study
