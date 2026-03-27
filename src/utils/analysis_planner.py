"""
Analysis Planner — LUMEN v2
============================
Phase 4.5: Sits between extraction (Phase 4) and statistics (Phase 5).
Profiles extracted data, proposes analysis plan via LLM, supports human review.

The core problem this solves: without analysis planning, Phase 5 blindly
pools all studies together, producing meaningless overall estimates with
I²=100% when studies span different interventions, outcomes, or designs.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Data Profiling (pure Python, no LLM)
# ---------------------------------------------------------------------------

# Common normalization patterns for intervention names
_INTERVENTION_PATTERNS = [
    (r'\b(PCV[\s-]?13|Prevnar\s*13|Prevenar\s*13|13[\s-]?valent\s+.*conjugate)\b', 'PCV13'),
    (r'\b(PCV[\s-]?15|V114|15[\s-]?valent\s+.*conjugate)\b', 'PCV15'),
    (r'\b(PCV[\s-]?20|Prevnar\s*20|20[\s-]?valent\s+.*conjugate)\b', 'PCV20'),
    (r'\b(PPSV[\s-]?23|PPV[\s-]?23|Pneumovax|23[\s-]?valent\s+.*polysaccharide)\b', 'PPSV23'),
    (r'\b(PCV[\s-]?7|Prevnar(?!\s*1[35])|7[\s-]?valent\s+.*conjugate)\b', 'PCV7'),
    (r'\b(PCV[\s-]?10|Synflorix|10[\s-]?valent\s+.*conjugate)\b', 'PCV10'),
    (r'\bplacebo\b', 'Placebo'),
    (r'\b(no\s+vaccin|unvaccinat)', 'No vaccination'),
]

_OUTCOME_PATTERNS = [
    (r'\b(VT[\s-]?IPD|vaccine[\s-]?type\s+invasive\s+pneumococcal\s+disease)\b', 'VT-IPD'),
    (r'\b(IPD|invasive\s+pneumococcal\s+disease)\b', 'IPD'),
    (r'\b(VT[\s-]?PP|vaccine[\s-]?type\s+pneumococcal\s+pneumonia)\b', 'VT-PP'),
    (r'\b(CAP|community[\s-]?acquired\s+pneumonia)\b', 'CAP'),
    (r'\b(pneumococcal\s+pneumonia|pneumonia)\b', 'Pneumonia'),
    (r'\b(all[\s-]?cause\s+mortality|mortality)\b', 'Mortality'),
    (r'\b(serotype\s*3)\b', 'Serotype 3'),
    (r'\b(nasopharyngeal|carriage)\b', 'Carriage'),
    (r'\b(immunogenicity|antibod|OPA|GMT|GMC)\b', 'Immunogenicity'),
]


def normalize_intervention(text: str) -> str:
    """Normalize intervention description to a canonical name."""
    if not text:
        return "Unknown"
    for pattern, canonical in _INTERVENTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return canonical
    return text[:50].strip()


def normalize_outcome(text: str) -> str:
    """Normalize outcome measure to a canonical name."""
    if not text:
        return "Unknown"
    for pattern, canonical in _OUTCOME_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return canonical
    return text[:50].strip()


# ---------------------------------------------------------------------------
# Step 0.5: LLM-based Outcome & Intervention Harmonization
# ---------------------------------------------------------------------------

_HARMONIZE_SYSTEM_PROMPT = """\
You are a systematic review methodologist. Your task is to harmonize outcome measure names and intervention names extracted from multiple studies into TWO levels of grouping:

LEVEL 1 — "canonical": Same instrument/same drug-dose (fine-grained)
LEVEL 2 — "broad": Same clinical construct/same drug class (for meta-analysis pooling)

RULES FOR OUTCOMES:
1. "canonical" groups outcomes using the SAME instrument (e.g., HAM-D variants → "Depression (HAM-D)")
2. "broad" groups ALL outcomes measuring the SAME construct regardless of instrument
   - e.g., HAM-D, BDI-II, PHQ-9, MADRS, CES-D all measure depression → broad: "Depression"
   - e.g., ISI, PSQI-sleep, diary-based insomnia → broad: "Insomnia severity"
   - e.g., Sleep efficiency (diary), Sleep efficiency (PSG) → broad: "Sleep efficiency"
3. Do NOT merge DIFFERENT constructs (depression ≠ insomnia ≠ anxiety)

RULES FOR INTERVENTIONS:
4. "canonical" groups name variants of the same specific intervention (e.g., "CBT-I 4 sessions face-to-face" variants)
5. "broad" groups interventions of the same CLASS for pooled analysis
   - e.g., face-to-face CBT-I, digital CBT-I, brief CBT-I, CBT-I + pharmacotherapy → broad: "CBT-I (any format)"
   - e.g., sertraline 50mg, sertraline 100mg → broad: "Sertraline"
6. Keep "Placebo", "No treatment", "TAU", "Waitlist" as-is for both levels

Return valid JSON:
```json
{
  "outcome_groups": [
    {
      "canonical": "Depression (HAM-D)",
      "broad": "Depression",
      "members": ["HAM-D", "HDRS", "Hamilton Depression Rating Scale", ...]
    }
  ],
  "intervention_groups": [
    {
      "canonical": "CBT-I (face-to-face, standard)",
      "broad": "CBT-I (any format)",
      "members": ["Cognitive behavioral therapy for insomnia, 6 weekly sessions", ...]
    }
  ]
}
```

Every item must appear in exactly one group. Items with no matches should still appear as a single-member group."""


def harmonize_outcomes_and_interventions(
    extracted_data: list, pico: dict, agent
) -> dict:
    """
    Use LLM to cluster raw outcome/intervention names into canonical groups.
    Returns mapping dict and applies it in-place to extracted_data.
    """
    # Collect unique raw names
    outcome_names = set()
    intervention_names = set()
    for study in extracted_data:
        interv_study = study.get("intervention_description", "")
        for outcome in study.get("outcomes", []):
            m = outcome.get("measure", "") or outcome.get("outcome_measure", "")
            if m:
                outcome_names.add(m)
            interv = outcome.get("intervention_description", "") or interv_study
            if interv:
                intervention_names.add(interv)

    if not outcome_names:
        logger.warning("No outcome names found, skipping harmonization")
        return {"outcome_map": {}, "intervention_map": {}, "raw_groups": {}}

    prompt_lines = [
        "## PICO Context",
        f"Population: {pico.get('population', 'N/A')}",
        f"Intervention: {pico.get('intervention', 'N/A')}",
        f"Comparison: {pico.get('comparison', 'N/A')}",
    ]
    outcome_pico = pico.get("outcome", {})
    if isinstance(outcome_pico, dict):
        prompt_lines.append(f"Primary outcome: {outcome_pico.get('primary', 'N/A')}")
    else:
        prompt_lines.append(f"Outcome: {outcome_pico}")

    prompt_lines.append("")
    prompt_lines.append(f"## Outcome measures to harmonize ({len(outcome_names)} unique):")
    for name in sorted(outcome_names):
        prompt_lines.append(f"  - {name}")

    prompt_lines.append("")
    prompt_lines.append(f"## Intervention names to harmonize ({len(intervention_names)} unique):")
    for name in sorted(intervention_names):
        prompt_lines.append(f"  - {name}")

    prompt_lines.append("")
    prompt_lines.append("Group these into canonical names following the JSON structure in your instructions.")

    result = agent.call_llm(
        prompt="\n".join(prompt_lines),
        system_prompt=_HARMONIZE_SYSTEM_PROMPT,
        expect_json=True,
        cache_namespace="outcome_harmonization",
        description="Harmonize outcome and intervention names",
    )

    # call_llm returns {"content", "parsed", "tokens"}; parsed has the JSON
    raw_groups = {}
    if isinstance(result, dict):
        parsed = result.get("parsed")
        if isinstance(parsed, dict) and ("outcome_groups" in parsed or "intervention_groups" in parsed):
            raw_groups = parsed
        elif "outcome_groups" in result:
            raw_groups = result

    # Build reverse maps: raw_name → canonical, raw_name → broad
    outcome_map = {}
    outcome_broad_map = {}
    for group in raw_groups.get("outcome_groups", []):
        canonical = group.get("canonical", "")
        broad = group.get("broad", canonical)
        for member in group.get("members", []):
            outcome_map[member] = canonical
            outcome_broad_map[member] = broad

    intervention_map = {}
    intervention_broad_map = {}
    for group in raw_groups.get("intervention_groups", []):
        canonical = group.get("canonical", "")
        broad = group.get("broad", canonical)
        for member in group.get("members", []):
            intervention_map[member] = canonical
            intervention_broad_map[member] = broad

    # Also build canonical → broad maps (for profiler)
    outcome_canonical_to_broad = {}
    for group in raw_groups.get("outcome_groups", []):
        outcome_canonical_to_broad[group.get("canonical", "")] = group.get("broad", group.get("canonical", ""))
    intervention_canonical_to_broad = {}
    for group in raw_groups.get("intervention_groups", []):
        intervention_canonical_to_broad[group.get("canonical", "")] = group.get("broad", group.get("canonical", ""))

    # Apply mapping in-place to extracted data
    n_outcome_mapped = 0
    n_outcome_regex = 0
    n_interv_mapped = 0
    for study in extracted_data:
        interv_study = study.get("intervention_description", "")
        if interv_study in intervention_map:
            study["_original_intervention"] = interv_study
            study["intervention_description"] = intervention_map[interv_study]
            study["intervention_broad"] = intervention_broad_map.get(interv_study, "")
            n_interv_mapped += 1

        for outcome in study.get("outcomes", []):
            raw_measure = outcome.get("measure", "") or outcome.get("outcome_measure", "")
            if raw_measure in outcome_map:
                outcome["_original_measure"] = raw_measure
                outcome["measure"] = outcome_map[raw_measure]
                outcome["measure_broad"] = outcome_broad_map.get(raw_measure, "")
                n_outcome_mapped += 1
            elif raw_measure and not outcome.get("measure_broad"):
                # Regex fallback for outcomes the LLM missed
                broad = normalize_outcome(raw_measure)
                if broad != raw_measure[:50].strip():
                    outcome["measure_broad"] = broad
                    n_outcome_regex += 1

            raw_interv = outcome.get("intervention_description", "")
            if raw_interv in intervention_map:
                outcome["_original_intervention"] = raw_interv
                outcome["intervention_description"] = intervention_map[raw_interv]
                outcome["intervention_broad"] = intervention_broad_map.get(raw_interv, "")

    logger.info(
        f"Harmonization applied: {n_outcome_mapped} outcome entries remapped "
        f"({len(outcome_map)} unique), {n_outcome_regex} regex fallback, "
        f"{n_interv_mapped} intervention entries remapped "
        f"({len(intervention_map)} unique)"
    )

    return {
        "outcome_map": outcome_map,
        "outcome_broad_map": outcome_broad_map,
        "intervention_map": intervention_map,
        "intervention_broad_map": intervention_broad_map,
        "outcome_canonical_to_broad": outcome_canonical_to_broad,
        "intervention_canonical_to_broad": intervention_canonical_to_broad,
        "raw_groups": raw_groups,
    }


def profile_extracted_data(extracted_data: list, pico: dict = None) -> dict:
    """
    Scan all extracted outcomes and build intervention × outcome matrix.
    Uses harmonized names if available (from harmonize_outcomes_and_interventions).
    Builds both fine-grained (canonical) and broad matrices.
    """
    interventions = Counter()
    outcomes = Counter()
    broad_interventions = Counter()
    broad_outcomes = Counter()
    study_designs = Counter()
    matrix = defaultdict(list)  # (intervention, outcome) → [study_ids]
    broad_matrix = defaultdict(list)  # (broad_interv, broad_outcome) → [study_ids]
    study_details = {}

    for study in extracted_data:
        study_id = study.get("study_id", "unknown")
        design = study.get("study_design", "unknown")
        study_designs[design] += 1

        for outcome in study.get("outcomes", []):
            # Use harmonized names directly (already applied in-place)
            interv_raw = outcome.get("intervention_description", "") or study.get("intervention_description", "")
            outcome_raw = outcome.get("measure", "") or outcome.get("outcome_measure", "")

            # Fine-grained: use canonical (already set by harmonization)
            interv = interv_raw[:60].strip() if interv_raw else "Unknown"
            outc = outcome_raw[:60].strip() if outcome_raw else "Unknown"

            # Broad: use broad category if set by harmonization
            interv_broad = outcome.get("intervention_broad") or study.get("intervention_broad") or interv
            outc_broad = outcome.get("measure_broad") or outc

            interventions[interv] += 1
            outcomes[outc] += 1
            broad_interventions[interv_broad] += 1
            broad_outcomes[outc_broad] += 1

            # Check if this outcome has usable effect data
            has_ve = outcome.get("ve_pct") is not None
            has_or = outcome.get("effect_size") is not None or outcome.get("odds_ratio") is not None
            has_rr = outcome.get("risk_ratio") is not None
            has_hr = outcome.get("hr") is not None or outcome.get("hazard_ratio") is not None
            ig = outcome.get("intervention_group", {}) or {}
            has_continuous = ig.get("mean") is not None
            has_binary = (ig.get("events") is not None and ig.get("total") is not None)

            if has_ve or has_or or has_rr or has_hr or has_continuous or has_binary:
                matrix[(interv, outc)].append(study_id)
                broad_matrix[(interv_broad, outc_broad)].append(study_id)

                if study_id not in study_details:
                    study_details[study_id] = []
                study_details[study_id].append({
                    "intervention": interv,
                    "intervention_broad": interv_broad,
                    "outcome": outc,
                    "outcome_broad": outc_broad,
                    "design": design,
                    "outcome_type": outcome.get("outcome_type", "continuous"),
                    "has_ve": has_ve,
                    "has_or": has_or,
                    "has_rr": has_rr,
                    "has_binary": has_binary,
                    "has_continuous": has_continuous,
                    "ve_pct": outcome.get("ve_pct"),
                    "ve_ci_lower": outcome.get("ve_ci_lower"),
                    "ve_ci_upper": outcome.get("ve_ci_upper"),
                })

    # Deduplicate study_ids per cell
    for key in matrix:
        matrix[key] = sorted(set(matrix[key]))
    for key in broad_matrix:
        broad_matrix[key] = sorted(set(broad_matrix[key]))

    # Build feasible analyses from BROAD matrix (k >= 2)
    feasible = []
    for (interv, outc), study_ids in sorted(broad_matrix.items(), key=lambda x: -len(x[1])):
        feasible.append({
            "intervention": interv,
            "outcome": outc,
            "k": len(study_ids),
            "study_ids": study_ids,
        })

    # Also keep fine-grained for reference
    feasible_fine = []
    for (interv, outc), study_ids in sorted(matrix.items(), key=lambda x: -len(x[1])):
        feasible_fine.append({
            "intervention": interv,
            "outcome": outc,
            "k": len(study_ids),
            "study_ids": study_ids,
        })

    return {
        "n_studies": len(extracted_data),
        "n_with_outcomes": len(study_details),
        "interventions": dict(interventions.most_common()),
        "outcomes": dict(outcomes.most_common()),
        "broad_interventions": dict(broad_interventions.most_common()),
        "broad_outcomes": dict(broad_outcomes.most_common()),
        "study_designs": dict(study_designs.most_common()),
        "matrix": {f"{k[0]} × {k[1]}": v for k, v in matrix.items()},
        "broad_matrix": {f"{k[0]} × {k[1]}": v for k, v in broad_matrix.items()},
        "feasible_analyses": feasible,
        "feasible_analyses_fine": feasible_fine,
        "study_details": study_details,
    }


# ---------------------------------------------------------------------------
# Step 2: LLM Proposal
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are a meta-analysis methodologist. Given a data profile from extracted studies, propose a structured analysis plan.

RULES:
1. NEVER pool different interventions together (e.g., PCV13 and PPSV23 must be separate analyses)
2. NEVER pool different outcome types together (e.g., IPD and pneumonia must be separate)
3. Only propose analyses with k >= 3 studies
4. Separate efficacy (RCT) from effectiveness (observational) when both exist with k >= 3
5. Propose subgroup analyses only when the parent analysis has enough studies (k >= 6)
6. Every primary analysis needs a forest plot; overall funnel plot if total k >= 10
7. EFFECT MEASURE SELECTION:
   - SMD: continuous outcomes (scores, scales) across different instruments
   - MD: continuous outcomes all using the same scale
   - OR: binary outcomes (remission, response, mortality rates)
   - RR: binary outcomes when relative risk is more appropriate
   - VE_OR: vaccine effectiveness derived from odds ratio
   - HR: time-to-event / survival outcomes
   Choose the measure that matches the outcome data type. Do NOT default to VE_OR for non-vaccine topics.

Return valid YAML with this structure:

```yaml
analyses:
  - id: <unique_id>
    label: <human-readable label>
    intervention: <normalized name>
    outcome: <normalized name>
    study_ids: [list of study_ids]
    k: <count>
    effect_measure: <SMD|MD|OR|RR|HR|VE_OR|VE_RR>
    rationale: <why this grouping>

subgroup_analyses:
  - parent: <analysis_id>
    by: <variable name>
    rationale: <why>

sensitivity_analyses:
  - parent: <analysis_id>
    type: <leave_one_out|exclude_high_rob|trim_and_fill>

figures:
  - type: <forest|funnel|subgroup_forest>
    analysis: <analysis_id>
    label: <figure caption>
```"""


def generate_analysis_plan_prompt(profile: dict, pico: dict) -> str:
    """Build the user prompt for the analysis planner LLM."""
    # Summarize profile concisely
    lines = [
        "## Extracted Data Profile",
        f"Total studies: {profile['n_studies']}",
        f"Studies with computable outcomes: {profile['n_with_outcomes']}",
        "",
        "### Interventions found:",
    ]
    for interv, count in profile["interventions"].items():
        lines.append(f"  - {interv}: {count} outcome entries")

    lines.append("")
    lines.append("### Outcomes found:")
    for outc, count in profile["outcomes"].items():
        lines.append(f"  - {outc}: {count} outcome entries")

    # Show broad groupings if available
    if profile.get("broad_interventions"):
        lines.append("")
        lines.append("### Interventions (broad grouping for pooling):")
        for name, count in profile["broad_interventions"].items():
            lines.append(f"  - {name}: {count}")
    if profile.get("broad_outcomes"):
        lines.append("")
        lines.append("### Outcomes (broad grouping for pooling):")
        for name, count in profile["broad_outcomes"].items():
            lines.append(f"  - {name}: {count}")

    lines.append("")
    lines.append("### Feasible analysis combinations (intervention × outcome, k >= 2):")
    lines.append("Use the BROAD intervention and outcome names below for your analysis plan:")
    for fa in profile["feasible_analyses"]:
        if fa["k"] >= 2:
            lines.append(f"  - {fa['intervention']} × {fa['outcome']}: k={fa['k']} {fa['study_ids'][:5]}{'...' if len(fa['study_ids']) > 5 else ''}")

    lines.append("")
    lines.append("### Study designs:")
    for design, count in profile["study_designs"].items():
        lines.append(f"  - {design}: {count}")

    # Add PICO context
    lines.append("")
    lines.append("## PICO Definition")
    lines.append(f"Population: {pico.get('population', 'N/A')}")
    lines.append(f"Intervention: {pico.get('intervention', 'N/A')}")
    lines.append(f"Comparison: {pico.get('comparison', 'N/A')}")
    outcome = pico.get("outcome", {})
    if isinstance(outcome, dict):
        lines.append(f"Primary outcome: {outcome.get('primary', 'N/A')}")
    else:
        lines.append(f"Outcome: {outcome}")

    # Gold standard if available
    gs = pico.get("gold_standard", {})
    if gs:
        lines.append("")
        lines.append("## Reference (gold standard) analysis structure:")
        for key, val in gs.items():
            if isinstance(val, dict):
                lines.append(f"  - {key}: {val.get('effect', '')} {val.get('ci', '')} (n={val.get('n_studies', '?')})")
            elif isinstance(val, str):
                lines.append(f"  - {key}: {val}")

    lines.append("")
    lines.append("Propose an analysis plan following the YAML structure in your instructions.")

    return "\n".join(lines)


def propose_analysis_plan(profile: dict, pico: dict, agent) -> dict:
    """Call LLM to generate analysis plan proposal."""
    prompt = generate_analysis_plan_prompt(profile, pico)

    result = agent.call_llm(
        prompt=prompt,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        expect_json=False,
        cache_namespace="analysis_planner",
        description="Propose analysis plan",
    )

    raw_text = result.get("content", "") or result.get("text", "")

    # Extract YAML from response
    plan = _parse_yaml_from_response(raw_text)
    if not plan:
        logger.warning("Failed to parse analysis plan from LLM response")
        plan = _build_fallback_plan(profile, pico)

    plan["_raw_response"] = raw_text
    plan["_profile"] = {
        "n_studies": profile["n_studies"],
        "n_with_outcomes": profile["n_with_outcomes"],
        "n_feasible": len([f for f in profile["feasible_analyses"] if f["k"] >= 3]),
    }

    return plan


def _parse_yaml_from_response(text: str) -> dict:
    """Extract YAML block from LLM response."""
    # Try to find ```yaml ... ``` block
    yaml_match = re.search(r'```ya?ml\s*\n(.*?)```', text, re.DOTALL)
    if yaml_match:
        try:
            return yaml.safe_load(yaml_match.group(1))
        except yaml.YAMLError:
            pass

    # Try parsing entire text as YAML
    try:
        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict) and "analyses" in parsed:
            return parsed
    except yaml.YAMLError:
        pass

    return {}


def _detect_default_measure(pico: dict, outcome_name: str = "") -> str:
    """Detect appropriate effect measure from PICO and outcome name."""
    # Check PICO for explicit preference
    em = pico.get("effect_measure") or pico.get("preferred_measure", "")
    if em:
        return em

    # Check if vaccine-related
    interv = (pico.get("intervention") or "").lower()
    if any(v in interv for v in ["vaccin", "immuniz"]):
        return "VE_OR"

    # Check outcome name for binary indicators
    outcome_lower = outcome_name.lower()
    binary_keywords = ["remission", "response", "mortality", "survival", "relapse",
                       "recovery", "cure", "incidence", "event"]
    if any(kw in outcome_lower for kw in binary_keywords):
        return "OR"

    # Default to SMD for continuous outcomes
    return "SMD"


def _build_fallback_plan(profile: dict, pico: dict) -> dict:
    """Build a simple plan from profile when LLM fails."""
    analyses = []
    for fa in profile["feasible_analyses"]:
        if fa["k"] >= 3:
            aid = f"{fa['intervention']}_{fa['outcome']}".lower().replace(" ", "_").replace("-", "_")
            measure = _detect_default_measure(pico, fa["outcome"])
            analyses.append({
                "id": aid,
                "label": f"{fa['intervention']} vs {fa['outcome']}",
                "intervention": fa["intervention"],
                "outcome": fa["outcome"],
                "study_ids": fa["study_ids"],
                "k": fa["k"],
                "effect_measure": measure,
                "rationale": "Auto-generated from feasible combinations (k>=3)",
            })

    return {
        "analyses": analyses,
        "subgroup_analyses": [],
        "sensitivity_analyses": [
            {"parent": a["id"], "type": "leave_one_out"}
            for a in analyses
        ],
        "figures": [
            {"type": "forest", "analysis": a["id"], "label": f"Forest: {a['label']}"}
            for a in analyses
        ],
    }


# ---------------------------------------------------------------------------
# Step 3: Human Review (Terminal Mode)
# ---------------------------------------------------------------------------

def display_plan_terminal(plan: dict) -> dict:
    """
    Display analysis plan in terminal for human review.
    Returns the (potentially modified) plan with human_approved flag.
    """
    analyses = plan.get("analyses", [])
    subgroups = plan.get("subgroup_analyses", [])
    sensitivity = plan.get("sensitivity_analyses", [])
    figures = plan.get("figures", [])

    print("\n" + "=" * 60)
    print("  Phase 4.5: Analysis Plan Review")
    print("=" * 60)
    print()

    if not analyses:
        print("  No analyses proposed. Check data profile.")
        plan["human_approved"] = False
        return plan

    print(f"  {len(analyses)} primary analyses proposed:")
    print()

    for i, a in enumerate(analyses, 1):
        k = a.get("k", len(a.get("study_ids", [])))
        feasible = "OK" if k >= 3 else "BORDERLINE" if k == 2 else "SKIP"
        print(f"  [{i}] {a.get('label', a.get('id', '?'))}")
        print(f"      Intervention: {a.get('intervention', '?')}")
        print(f"      Outcome:      {a.get('outcome', '?')}")
        print(f"      k={k}  measure={a.get('effect_measure', '?')}  [{feasible}]")
        print(f"      {a.get('rationale', '')[:80]}")
        print()

    if subgroups:
        print(f"  Subgroup analyses: {len(subgroups)}")
        for s in subgroups:
            print(f"    - {s.get('parent', '?')} by {s.get('by', '?')}")
        print()

    if sensitivity:
        print(f"  Sensitivity analyses: {len(sensitivity)}")
        for s in sensitivity:
            print(f"    - {s.get('parent', '?')}: {s.get('type', '?')}")
        print()

    print(f"  Figures: {len(figures)}")
    for f in figures:
        print(f"    - {f.get('type', '?')}: {f.get('label', f.get('analysis', '?'))}")
    print()

    print("-" * 60)
    choice = input("  [A]pprove  [S]kip analysis by number  [Q]uit: ").strip().upper()

    if choice == "A":
        plan["human_approved"] = True
        print("  Plan approved.")
    elif choice == "Q":
        plan["human_approved"] = False
        print("  Plan rejected. Phase 5 will not run.")
    elif choice.startswith("S"):
        # Skip specific analyses
        try:
            skip_nums = [int(x.strip()) for x in choice[1:].split(",")]
            remaining = []
            for i, a in enumerate(analyses, 1):
                if i not in skip_nums:
                    remaining.append(a)
                else:
                    print(f"  Skipped: {a.get('label', a.get('id', '?'))}")
            plan["analyses"] = remaining
            plan["human_approved"] = True
            print(f"  Plan approved with {len(remaining)} analyses.")
        except ValueError:
            plan["human_approved"] = True
            print("  Could not parse skip numbers. Approving all.")
    else:
        plan["human_approved"] = True
        print("  Defaulting to approve.")

    print()
    return plan


# ---------------------------------------------------------------------------
# Step 4: Save & Load
# ---------------------------------------------------------------------------

def save_analysis_plan(plan: dict, output_path: str) -> None:
    """Save analysis plan as YAML."""
    # Remove internal fields for clean output
    clean = {k: v for k, v in plan.items() if not k.startswith("_")}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(clean, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info(f"Analysis plan saved: {output_path}")


def load_analysis_plan(plan_path: str) -> Optional[dict]:
    """Load analysis plan from YAML."""
    p = Path(plan_path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)
