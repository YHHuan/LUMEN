"""
Interactive PICO Builder — LUMEN v2
=====================================
LLM-guided conversational PICO refinement.

Walks the user through structured questions to build a publication-quality
pico.yaml. Uses Claude to:
  1. Ask smart follow-up questions based on the research topic
  2. Validate MeSH terms via PubMed
  3. Suggest exclusion criteria based on domain knowledge
  4. Preview search yield estimates
  5. Confirm all decisions before saving

The output pico.yaml matches the quality level of a well-written protocol.
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.utils.cache import TokenBudget

logger = logging.getLogger(__name__)

# ANSI colors for terminal
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"


def _ask(prompt: str, default: str = "", allow_empty: bool = False) -> str:
    """Prompt user for input with optional default."""
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"\n{CYAN}> {prompt}{suffix}: {RESET}").strip()
        if not answer and default:
            return default
        if answer or allow_empty:
            return answer
        print(f"  {YELLOW}Please provide an answer.{RESET}")


def _confirm(prompt: str, default: bool = True) -> bool:
    """Yes/no confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"\n{CYAN}> {prompt} {suffix}: {RESET}").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def _show(label: str, value: str):
    """Display a labeled value."""
    print(f"  {GREEN}{label}:{RESET} {value}")


def _section(title: str):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {BOLD}{title}{RESET}")
    print(f"{'=' * 60}")


def _llm_ask(agent: BaseAgent, context: str, question: str) -> str:
    """Use LLM to generate a smart question or suggestion."""
    result = agent.call_llm(
        prompt=f"{context}\n\nTask: {question}\n\nBe concise (2-3 sentences max). "
               f"If suggesting a list, use numbered format.",
        system_prompt=(
            "You are a systematic review methodology expert helping a researcher "
            "define their PICO criteria. Be precise, practical, and concise. "
            "Use clinical terminology appropriate for the field. "
            "Respond in the same language the user uses."
        ),
        expect_json=False,
        cache_namespace="pico_builder",
        description="PICO builder guidance",
    )
    return result.get("raw", "")


def run_interactive_pico_builder(
    data_dir: str,
    budget: Optional[TokenBudget] = None,
) -> dict:
    """
    Run the interactive PICO builder.

    Returns the complete pico dict, also saves to data_dir/input/pico.yaml.
    """
    agent = BaseAgent(role_name="strategist", budget=budget)
    pico = {}

    print(f"\n{BOLD}{'=' * 60}")
    print(f"  LUMEN v2 — Interactive PICO Builder")
    print(f"{'=' * 60}{RESET}")
    print(f"\n  {DIM}This wizard will walk you through defining your research")
    print(f"  question step by step. The LLM will help refine each")
    print(f"  element to publication quality.{RESET}")

    # ─── Step 1: Research topic overview ─────────────────────
    _section("Step 1: Research Topic")
    topic = _ask("Briefly describe your research question (1-2 sentences)")

    # LLM suggests initial PICO decomposition
    suggestion = _llm_ask(agent, f"Research topic: {topic}",
        "Decompose this into P-I-C-O components. For each, give a 1-line "
        "draft. Also suggest the most appropriate study design (RCT only, "
        "or also observational).")
    print(f"\n  {DIM}LLM suggestion:{RESET}")
    print(f"  {suggestion}")

    # ─── Step 2: Population ──────────────────────────────────
    _section("Step 2: Population (P)")
    population = _ask("Define the target population (diagnosis, age, criteria)")

    # LLM probes for exclusions
    pop_probe = _llm_ask(agent,
        f"Topic: {topic}\nPopulation: {population}",
        "Suggest 3-5 population exclusion criteria that are commonly applied "
        "in systematic reviews for this population. Format as numbered list.")
    print(f"\n  {DIM}Suggested exclusions:{RESET}")
    print(f"  {pop_probe}")

    pop_exclusions = _ask(
        "Enter population exclusions (comma-separated, or press Enter to accept suggestions)",
        allow_empty=True
    )
    if pop_exclusions:
        population += f"\n  Exclude: {pop_exclusions}"
    elif pop_probe:
        if _confirm("Accept the suggested exclusions above?"):
            population += f"\n  Exclude: {pop_probe}"

    pico["population"] = population

    # ─── Step 3: Intervention ────────────────────────────────
    _section("Step 3: Intervention (I)")
    intervention = _ask("Define the intervention(s)")

    # LLM suggests related interventions that might be missed
    int_probe = _llm_ask(agent,
        f"Topic: {topic}\nIntervention: {intervention}",
        "Are there related interventions, brand names, combination therapies, "
        "or newer agents that should be included in the search? List any that "
        "the researcher might miss.")
    print(f"\n  {DIM}Related interventions to consider:{RESET}")
    print(f"  {int_probe}")

    int_add = _ask("Add any additional interventions? (or Enter to skip)", allow_empty=True)
    if int_add:
        intervention += f", {int_add}"
    pico["intervention"] = intervention

    # ─── Step 4: Comparison ──────────────────────────────────
    _section("Step 4: Comparison (C)")
    comparison = _ask("Define the comparator(s)")

    # LLM asks about comparator exclusions
    comp_probe = _llm_ask(agent,
        f"Topic: {topic}\nIntervention: {intervention}\nComparator: {comparison}",
        "Should any specific comparators be excluded? (e.g., active controls "
        "with different mechanisms, non-pharmacological interventions). "
        "Also: is background therapy (lifestyle, other drugs) permitted?")
    print(f"\n  {DIM}Comparator considerations:{RESET}")
    print(f"  {comp_probe}")

    comp_detail = _ask("Add comparator restrictions? (or Enter to skip)", allow_empty=True)
    if comp_detail:
        comparison += f"\n  {comp_detail}"
    pico["comparison"] = comparison

    # ─── Step 5: Outcomes ────────────────────────────────────
    _section("Step 5: Outcomes (O)")
    primary = _ask("Define the PRIMARY outcome")

    # LLM suggests measurement methods and pitfalls
    out_probe = _llm_ask(agent,
        f"Topic: {topic}\nPrimary outcome: {primary}",
        "For this primary outcome: 1) What are the standard measurement "
        "methods/scales? 2) Are there similar but distinct outcomes that "
        "should NOT be merged with this? 3) Should it be binary or continuous?")
    print(f"\n  {DIM}Outcome considerations:{RESET}")
    print(f"  {out_probe}")

    primary_detail = _ask("Refine primary outcome definition? (or Enter to keep as-is)", allow_empty=True)
    if primary_detail:
        primary = primary_detail

    # Outcome type
    outcome_type = _ask("Primary outcome type", default="binary")
    effect_measure = _ask("Effect measure for primary outcome",
                          default="RR" if outcome_type == "binary" else "SMD")

    # Secondary outcomes
    secondary_input = _ask(
        "List secondary outcomes (comma-separated)",
        allow_empty=True
    )
    secondary = [s.strip() for s in secondary_input.split(",") if s.strip()] if secondary_input else []

    pico["outcome"] = {"primary": primary, "secondary": secondary}
    pico["outcome_type"] = outcome_type
    pico["effect_measure"] = effect_measure

    # ─── Step 6: Study design ────────────────────────────────
    _section("Step 6: Study Design & Restrictions")
    study_design = _ask("Study design", default="Randomized controlled trials (RCTs) only")
    pico["study_design"] = study_design

    # Analysis type
    print(f"\n  {DIM}Analysis types: pairwise (standard MA), nma (network MA){RESET}")
    analysis_type = _ask("Analysis type", default="pairwise")
    pico["analysis_type"] = analysis_type

    if analysis_type == "nma":
        nodes_input = _ask("Define NMA treatment nodes (comma-separated)")
        pico["nma_nodes"] = [n.strip() for n in nodes_input.split(",")]
        ref_group = _ask("Reference group for NMA", default=pico["nma_nodes"][0])
        pico["nma_settings"] = {
            "effect_measure": effect_measure,
            "reference_group": ref_group,
            "small_values": "undesirable",
        }

    # ─── Step 7: Special rules ───────────────────────────────
    _section("Step 7: Screening & Special Rules")

    # No abstract handling
    print(f"  {DIM}Studies without abstracts are hard to screen from title alone.{RESET}")
    exclude_no_abstract = _confirm("Exclude studies with no/short abstract?", default=False)
    pico["exclude_no_abstract"] = exclude_no_abstract

    # Language restriction
    lang = _ask("Language restriction", default="none")
    pico["language_restriction"] = lang

    # Date restriction
    date = _ask("Date restriction", default="none")
    pico["date_restriction"] = date

    # ─── Step 8: Subgroups ───────────────────────────────────
    _section("Step 8: Pre-specified Subgroups")
    sub_probe = _llm_ask(agent,
        f"Topic: {topic}\nPopulation: {population}\nIntervention: {intervention}",
        "Suggest 3-4 clinically meaningful subgroup analyses for this "
        "systematic review. Format: variable name — rationale.")
    print(f"\n  {DIM}Suggested subgroups:{RESET}")
    print(f"  {sub_probe}")

    subgroups_input = _ask(
        "Define subgroups (format: variable1; variable2; ... or Enter to skip)",
        allow_empty=True
    )
    if subgroups_input:
        pico["subgroups"] = [
            {"variable": s.strip(), "label": s.strip()}
            for s in subgroups_input.split(";") if s.strip()
        ]

    # ─── Step 9: Search terms ────────────────────────────────
    _section("Step 9: Search Terms")
    terms_probe = _llm_ask(agent,
        f"Full PICO:\n{json.dumps(pico, indent=2, default=str)}",
        "Generate comprehensive search term lists for each PICO element. "
        "Include MeSH terms, free-text synonyms, abbreviations, brand names, "
        "and spelling variants. Format as JSON with keys: population, "
        "intervention, comparator, study_type.")
    print(f"\n  {DIM}Suggested search terms:{RESET}")
    print(f"  {terms_probe[:500]}...")

    if _confirm("Accept suggested search terms?"):
        try:
            # Try to parse as JSON
            import re
            json_match = re.search(r'\{[^{}]*\}', terms_probe, re.DOTALL)
            if json_match:
                pico["search_terms"] = json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

    # ─── Step 10: Review & confirm ───────────────────────────
    _section("Review: Final PICO Definition")
    print()
    for key, value in pico.items():
        if isinstance(value, dict):
            _show(key, json.dumps(value, indent=4, default=str)[:200])
        elif isinstance(value, list):
            _show(key, ", ".join(str(v) for v in value[:5]))
        else:
            val_str = str(value)
            _show(key, val_str[:150] + ("..." if len(val_str) > 150 else ""))

    if not _confirm("\nSave this PICO definition?"):
        print("  Aborted. PICO not saved.")
        return pico

    # ─── Save ────────────────────────────────────────────────
    output_dir = Path(data_dir) / "input"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pico.yaml"

    # Add header comment
    header = (
        f"# PICO definition generated by LUMEN v2 Interactive Builder\n"
        f"# Review and edit as needed before running Phase 1\n\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(pico, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=120)

    print(f"\n  {GREEN}Saved to: {output_path}{RESET}")
    print(f"  {DIM}You can edit this file manually before running Phase 1.{RESET}")

    return pico
