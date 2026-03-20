"""
Screener Agent (Phase 3)
=========================
Title/Abstract and Full-text screening with a 5-point confidence scale.

Decision scale (both screeners + arbiter use the same set):
  most_likely_include  → include-side
  likely_include       → include-side
  undecided            → include-side (bias toward recall)
  likely_exclude       → exclude-side
  most_likely_exclude  → exclude-side

Conflict resolution:
  - Sides agree              → final decision from agreed side
  - Sides disagree, at least one "undecided" → human_review_queue.json
  - Sides disagree, both firm → Arbiter

Token-saving strategies:
  1. Batch processing for title/abstract screening
  2. Only call Arbiter on firm disagreements (not undecided conflicts)
  3. Checkpoint-based resumption
"""

import json
import logging
from typing import List

from src.agents.base_agent import BaseAgent
from src.utils.cache import TokenBudget, Checkpoint

logger = logging.getLogger(__name__)

# ── Side classification ──────────────────────────────────────────────────────

INCLUDE_SIDE = {"most_likely_include", "likely_include", "undecided"}
EXCLUDE_SIDE = {"likely_exclude", "most_likely_exclude"}
VALID_DECISIONS = INCLUDE_SIDE | EXCLUDE_SIDE

FALLBACK_DECISION = "likely_include"   # on parse/API failure → bias toward recall


def _side(decision: str) -> str:
    """Returns 'include' or 'exclude' for a 5-point decision value."""
    return "include" if decision in INCLUDE_SIDE else "exclude"


def _is_undecided(decision: str) -> bool:
    return decision == "undecided"


# ── Screener Agent ────────────────────────────────────────────────────────────

class ScreenerAgent(BaseAgent):
    """Single screener — configurable as screener1, screener2, or arbiter."""

    def __init__(self, role_name: str = "screener1", **kwargs):
        super().__init__(role_name=role_name, **kwargs)
        # Both screener1 and screener2 share config/prompts/screener.yaml
        prompts = self.load_prompt_config("screener")
        self._system_prompt = prompts.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        self._ta_template = prompts.get("title_abstract_template", _DEFAULT_TA_TEMPLATE)
        self._ft_template = prompts.get("fulltext_template", _DEFAULT_FT_TEMPLATE)

    def screen_title_abstract_batch(self, studies: List[dict],
                                    criteria: dict) -> List[dict]:
        """Batch title/abstract screening. ~300 tokens per study."""
        criteria_text = self._format_criteria(criteria)

        def build_prompt(study):
            raw_abstract = study.get("abstract", "N/A") or "N/A"
            abstract = raw_abstract[:2500]
            if len(raw_abstract) > 2500:
                logger.debug(
                    f"Abstract truncated for {study.get('study_id', '')}: "
                    f"{len(raw_abstract)} → 2500 chars"
                )
            return self._ta_template.format(
                criteria_text=criteria_text,
                study_id=study.get("study_id", ""),
                title=study.get("title", "N/A"),
                abstract=abstract,
                year=study.get("year", "N/A"),
                journal=study.get("journal", "N/A"),
                publication_types=", ".join(study.get("publication_types", [])),
            )

        results = self.call_llm_batch(
            items=studies,
            build_prompt_fn=build_prompt,
            system_prompt=self._system_prompt,
            cache_namespace=f"screening_ta_{self.role_name}",
            description_fn=lambda s: f"Screen T/A: {s.get('study_id', '')}",
        )

        parsed_results = []
        for r in results:
            raw = r["parsed"] if r else None
            # Normalise: list → take first dict element; non-dict → treat as failure
            if isinstance(raw, list):
                raw = raw[0] if raw and isinstance(raw[0], dict) else None
            result = raw if isinstance(raw, dict) else None

            if result:
                # Normalise overall decision value
                decision = result.get("decision", FALLBACK_DECISION)
                if decision not in VALID_DECISIONS:
                    decision = "likely_include" if decision == "include" else "likely_exclude"
                    result["decision"] = decision
                # Normalise per-criterion fields (population / intervention / comparison_design)
                for field in ("population", "intervention", "comparison_design"):
                    val = result.get(field, "")
                    if val and val not in VALID_DECISIONS:
                        result[field] = "likely_include" if val == "include" else "likely_exclude"
                parsed_results.append(result)
            else:
                # API/parse failure → bias toward inclusion
                sid = (r.get("parsed", {}) or {}).get("study_id", "unknown") if r and isinstance(r.get("parsed"), dict) else "unknown"
                parsed_results.append({
                    "study_id": sid,
                    "decision": FALLBACK_DECISION,
                    "population": FALLBACK_DECISION,
                    "intervention": FALLBACK_DECISION,
                    "comparison_design": FALLBACK_DECISION,
                    "confidence": 0.0,
                    "reason": "Parse/API failure — defaulting to likely_include to preserve recall",
                })

        return parsed_results

    def screen_fulltext(self, study: dict, fulltext: str,
                        criteria: dict, study_design: str = "") -> dict:
        """Full-text screening for a single study."""
        criteria_text = self._format_criteria(criteria)

        prompt = self._ft_template.format(
            criteria_text=criteria_text,
            study_id=study.get("study_id", ""),
            title=study.get("title", ""),
            fulltext=fulltext,
            study_design=study_design,
        )

        result = self.call_llm(
            prompt=prompt,
            system_prompt=self._system_prompt,
            expect_json=True,
            cache_namespace=f"screening_ft_{self.role_name}",
            description=f"Full-text screen: {study.get('study_id', '')}",
        )

        parsed = result.get("parsed")
        if parsed:
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}
            if not isinstance(parsed, dict):
                parsed = {}
            decision = parsed.get("decision", FALLBACK_DECISION)
            if decision not in VALID_DECISIONS:
                parsed["decision"] = "likely_include" if decision == "include" else "likely_exclude"
            return parsed

        return {
            "study_id": study.get("study_id", ""),
            "decision": FALLBACK_DECISION,
            "confidence": 0.0,
            "reason": "Parse failed — defaulting to likely_include",
        }

    def _format_criteria(self, criteria: dict) -> str:
        parts = []
        inc = criteria.get("inclusion_criteria", [])
        if inc:
            parts.append("INCLUDE if ALL of:\n" + "\n".join(f"  - {c}" for c in inc))
        exc = criteria.get("exclusion_criteria", [])
        if exc:
            parts.append("EXCLUDE if ANY of:\n" + "\n".join(f"  - {c}" for c in exc))
        return "\n\n".join(parts)


# ── Arbiter Agent ─────────────────────────────────────────────────────────────

class ArbiterAgent(BaseAgent):
    """Resolves firm screener disagreements (not undecided conflicts)."""

    def __init__(self, **kwargs):
        super().__init__(role_name="arbiter", **kwargs)
        prompts = self.load_prompt_config("arbiter")
        self._system_prompt = prompts.get("system_prompt", _DEFAULT_ARBITER_SYSTEM)
        self._conflict_template = prompts.get("conflict_template", _DEFAULT_CONFLICT_TEMPLATE)

    def resolve_conflict(self, study: dict,
                         screener1_result: dict,
                         screener2_result: dict,
                         criteria: dict) -> dict:
        """Called only for firm (non-undecided) disagreements."""
        inc = criteria.get("inclusion_criteria", [])
        exc = criteria.get("exclusion_criteria", [])
        criteria_text = ""
        if inc:
            criteria_text += "Include: " + "; ".join(inc)
        if exc:
            criteria_text += "\nExclude: " + "; ".join(exc)

        prompt = self._conflict_template.format(
            title=study.get("title", "N/A"),
            abstract=(study.get("abstract", "N/A") or "N/A")[:1500],
            criteria_text=criteria_text,
            study_id=study.get("study_id", ""),
            s1_decision=screener1_result.get("decision", "?"),
            s1_confidence=screener1_result.get("confidence", "?"),
            s1_population=screener1_result.get("population", "N/A"),
            s1_intervention=screener1_result.get("intervention", "N/A"),
            s1_comparison_design=screener1_result.get("comparison_design", "N/A"),
            s1_reason=screener1_result.get("reason", "N/A"),
            s2_decision=screener2_result.get("decision", "?"),
            s2_confidence=screener2_result.get("confidence", "?"),
            s2_population=screener2_result.get("population", "N/A"),
            s2_intervention=screener2_result.get("intervention", "N/A"),
            s2_comparison_design=screener2_result.get("comparison_design", "N/A"),
            s2_reason=screener2_result.get("reason", "N/A"),
        )

        result = self.call_llm(
            prompt=prompt,
            system_prompt=self._system_prompt,
            expect_json=True,
            cache_namespace="screening_arbiter",
            description=f"Arbiter: {study.get('study_id', '')}",
        )

        parsed = result.get("parsed")
        if parsed:
            decision = parsed.get("decision", FALLBACK_DECISION)
            if decision not in VALID_DECISIONS:
                parsed["decision"] = "likely_include" if decision == "include" else "likely_exclude"
            for field in ("population", "intervention", "comparison_design"):
                val = parsed.get(field, "")
                if val and val not in VALID_DECISIONS:
                    parsed[field] = "likely_include" if val == "include" else "likely_exclude"
            return parsed

        return {
            "study_id": study.get("study_id", ""),
            "decision": FALLBACK_DECISION,
            "reason": "Arbiter parse failed — defaulting to likely_include",
        }


# ── Dual-screening orchestration ──────────────────────────────────────────────

def run_dual_screening(studies: List[dict], criteria: dict,
                       budget: TokenBudget = None) -> dict:
    """
    Full dual-screening pipeline with 5-point confidence scale.

    Conflict resolution:
      - Both on same side             → agreed decision
      - Sides differ, ≥1 undecided   → human_review_queue (temporarily included)
      - Sides differ, both firm       → Arbiter

    Returns:
        {
            "included": [study_ids],
            "excluded": [study_ids],
            "human_review_queue": [study_ids],   # NEW: undecided conflicts
            "screener1_results": [...],
            "screener2_results": [...],
            "conflicts_firm": [...],              # sent to Arbiter
            "conflicts_undecided": [...],         # sent to human queue
            "arbiter_decisions": [...],
            "agreement_rate": float,              # side-level agreement
            "exact_agreement_rate": float,        # exact 5-point match
            "cohens_kappa": float,                # binary (include/exclude side)
            "total_screened": int,
        }
    """
    logger.info(f"Starting dual screening on {len(studies)} studies")

    screener1 = ScreenerAgent(role_name="screener1", budget=budget)
    results1 = screener1.screen_title_abstract_batch(studies, criteria)

    screener2 = ScreenerAgent(role_name="screener2", budget=budget)
    results2 = screener2.screen_title_abstract_batch(studies, criteria)

    results1_map = {r["study_id"]: r for r in results1 if r and r.get("study_id")}
    results2_map = {r["study_id"]: r for r in results2 if r and r.get("study_id")}

    conflicts_firm = []
    conflicts_undecided = []
    side_agreements = 0
    exact_agreements = 0

    # For Cohen's κ (binary sides)
    all_sides = []   # each element: (side_s1, side_s2)

    for study in studies:
        sid = study["study_id"]
        r1 = results1_map.get(sid, {})
        r2 = results2_map.get(sid, {})
        d1 = r1.get("decision", FALLBACK_DECISION)
        d2 = r2.get("decision", FALLBACK_DECISION)
        s1 = _side(d1)
        s2 = _side(d2)
        all_sides.append((s1, s2))

        if d1 == d2:
            exact_agreements += 1
        if s1 == s2:
            side_agreements += 1
        else:
            # Conflict — classify type
            if _is_undecided(d1) or _is_undecided(d2):
                conflicts_undecided.append({
                    "study_id": sid,
                    "screener1": d1,
                    "screener2": d2,
                    "note": "At least one screener was undecided — routed to human review",
                })
            else:
                conflicts_firm.append({
                    "study_id": sid,
                    "screener1": d1,
                    "screener2": d2,
                })

    n = len(studies) or 1
    agreement_rate = side_agreements / n
    exact_agreement_rate = exact_agreements / n

    # Cohen's κ (binary side classification)
    cohens_kappa = _compute_kappa(all_sides)

    logger.info(
        f"Side agreement: {agreement_rate:.1%} | "
        f"Exact agreement: {exact_agreement_rate:.1%} | "
        f"κ={cohens_kappa:.3f} | "
        f"Firm conflicts: {len(conflicts_firm)} | "
        f"Undecided conflicts → human queue: {len(conflicts_undecided)}"
    )

    # === Arbiter for firm conflicts only ===
    arbiter = ArbiterAgent(budget=budget)
    arbiter_decisions = []

    for conflict in conflicts_firm:
        sid = conflict["study_id"]
        study = next((s for s in studies if s["study_id"] == sid), {})
        r1 = results1_map.get(sid, {})
        r2 = results2_map.get(sid, {})
        decision = arbiter.resolve_conflict(study, r1, r2, criteria)
        arbiter_decisions.append(decision)

    # === Compile final decisions ===
    human_review_ids = {c["study_id"] for c in conflicts_undecided}
    arbiter_map = {d.get("study_id"): d for d in arbiter_decisions}

    included = []
    excluded = []
    human_review_queue = []

    for study in studies:
        sid = study["study_id"]

        if sid in human_review_ids:
            # Undecided conflict → human queue (temporarily included to preserve recall)
            human_review_queue.append(sid)
            included.append(sid)
            continue

        arbiter_dec = arbiter_map.get(sid)
        if arbiter_dec:
            decision_val = arbiter_dec["decision"]
        else:
            r1 = results1_map.get(sid, {})
            decision_val = r1.get("decision", FALLBACK_DECISION)

        if _side(decision_val) == "include":
            included.append(sid)
        else:
            excluded.append(sid)

    return {
        "included": included,
        "excluded": excluded,
        "human_review_queue": human_review_queue,
        "screener1_results": results1,
        "screener2_results": results2,
        "conflicts_firm": conflicts_firm,
        "conflicts_undecided": conflicts_undecided,
        "arbiter_decisions": arbiter_decisions,
        "agreement_rate": round(agreement_rate, 4),
        "exact_agreement_rate": round(exact_agreement_rate, 4),
        "cohens_kappa": round(cohens_kappa, 4),
        "total_screened": len(studies),
    }


def _compute_kappa(all_sides: list) -> float:
    """Binary Cohen's κ on (side_s1, side_s2) pairs."""
    if not all_sides:
        return 0.0
    n = len(all_sides)
    po = sum(s1 == s2 for s1, s2 in all_sides) / n

    # Marginal probabilities
    p_inc_s1 = sum(s1 == "include" for s1, _ in all_sides) / n
    p_inc_s2 = sum(s2 == "include" for _, s2 in all_sides) / n
    p_exc_s1 = 1 - p_inc_s1
    p_exc_s2 = 1 - p_inc_s2

    pe = p_inc_s1 * p_inc_s2 + p_exc_s1 * p_exc_s2
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ── Inline fallback prompts (used when yaml is missing) ──────────────────────
# These mirror the content of config/prompts/screener.yaml and arbiter.yaml.

_DEFAULT_SYSTEM_PROMPT = (
    "You are a systematic review screening expert using a 5-point decision scale:\n"
    "  most_likely_include, likely_include, undecided, likely_exclude, most_likely_exclude\n"
    "Bias toward inclusion — false negatives are worse than false positives.\n"
    "Respond ONLY with valid JSON."
)

_DEFAULT_TA_TEMPLATE = (
    "Evaluate this study for inclusion in a systematic review using a P-I-C criterion-by-criterion assessment.\n\n"
    "CRITERIA:\n{criteria_text}\n\n"
    "STUDY:\nTitle: {title}\nAbstract: {abstract}\n"
    "Year: {year}\nJournal: {journal}\nPublication Type: {publication_types}\n\n"
    "Score each criterion independently. IMPORTANT: If study design (RCT) cannot be determined "
    "from the abstract, use 'undecided' for comparison_design — do NOT exclude on this basis alone.\n\n"
    'Respond with JSON:\n{{"study_id": "{study_id}", '
    '"population": "most_likely_include|likely_include|undecided|likely_exclude|most_likely_exclude", '
    '"intervention": "most_likely_include|likely_include|undecided|likely_exclude|most_likely_exclude", '
    '"comparison_design": "most_likely_include|likely_include|undecided|likely_exclude|most_likely_exclude", '
    '"decision": "most_likely_include|likely_include|undecided|likely_exclude|most_likely_exclude", '
    '"confidence": 0.0-1.0, "reason": "brief reason noting which criterion drove the decision"}}'
)

_DEFAULT_FT_TEMPLATE = (
    "Evaluate this study's FULL TEXT for inclusion in a systematic review.\n\n"
    "STUDY ID: {study_id}\nTitle: {title}\n"
    "Study design / publication type: {study_design}\n\n"
    "CRITERIA:\n{criteria_text}\n\n"
    "FULL TEXT (key sections):\n{fulltext}\n\n"
    "Reason through the following steps before giving your decision:\n\n"
    "STEP 1 — Exclusion check:\n"
    "  Does any exclusion criterion clearly and unambiguously match this study?\n"
    "  If yes, identify it and set decision to likely_exclude or most_likely_exclude.\n\n"
    "STEP 2 — Inclusion checklist (only if Step 1 did not exclude):\n"
    "  2.1 Population   — Does the studied population strictly match the inclusion criteria?\n"
    "  2.2 Intervention — Does the intervention/vaccine match?\n"
    "  2.3 Comparison   — Is there an appropriate comparator?\n"
    "  2.4 Outcome      — Are the required outcomes reported?\n"
    "  2.5 Study design — Does the study design match (consider the stated design above)?\n\n"
    "STEP 3 — Final decision:\n"
    "  If Step 1 matched an exclusion criterion → likely_exclude / most_likely_exclude.\n"
    "  If all of 2.1–2.5 are clearly met → most_likely_include / likely_include.\n"
    "  If insufficient information to judge any sub-criterion → undecided.\n"
    "  Otherwise → likely_exclude.\n\n"
    "Respond with JSON:\n"
    '{{"study_id": "{study_id}", '
    '"decision": "most_likely_include|likely_include|undecided|likely_exclude|most_likely_exclude", '
    '"confidence": 0.0-1.0, '
    '"meets_criteria": {{"population": true, "intervention": true, '
    '"comparison": true, "outcome": true, "study_design": true}}, '
    '"reason": "detailed reason referencing the steps above", '
    '"exclusion_category": "population|intervention|comparison|outcome|design|data|other|none"}}'
)

_DEFAULT_ARBITER_SYSTEM = (
    "You are a senior systematic review arbiter. "
    "When in doubt, INCLUDE the study. Respond ONLY with valid JSON."
)

_DEFAULT_CONFLICT_TEMPLATE = (
    "Two independent screeners disagreed on this study. Make the final decision.\n\n"
    "STUDY:\nTitle: {title}\nAbstract: {abstract}\n\nCRITERIA: {criteria_text}\n\n"
    "SCREENER 1 OVERALL: {s1_decision} (confidence: {s1_confidence})\n"
    "  Criterion breakdown — Population: {s1_population} | Intervention: {s1_intervention} | Comparison/Design: {s1_comparison_design}\n"
    "  Reason: {s1_reason}\n\n"
    "SCREENER 2 OVERALL: {s2_decision} (confidence: {s2_confidence})\n"
    "  Criterion breakdown — Population: {s2_population} | Intervention: {s2_intervention} | Comparison/Design: {s2_comparison_design}\n"
    "  Reason: {s2_reason}\n\n"
    "Remember: if study design (RCT) cannot be determined from abstract, do NOT exclude on that basis.\n\n"
    "Respond with JSON:\n"
    '{{"study_id": "{study_id}", '
    '"population": "most_likely_include|likely_include|undecided|likely_exclude|most_likely_exclude", '
    '"intervention": "most_likely_include|likely_include|undecided|likely_exclude|most_likely_exclude", '
    '"comparison_design": "most_likely_include|likely_include|undecided|likely_exclude|most_likely_exclude", '
    '"decision": "most_likely_include|likely_include|undecided|likely_exclude|most_likely_exclude", '
    '"confidence": 0.0-1.0, '
    '"reason": "your reasoning considering both screeners and each criterion", '
    '"agreed_with": "screener1|screener2|neither"}}'
)
