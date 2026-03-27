"""
Screener & Arbiter Agents — LUMEN v2
=======================================
Phase 3.1: Dual-screener with 5-point confidence scale + Arbiter for conflicts.
Phase 3.2: Full-text screening with chunk-based retrieval.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, List

import numpy as np

from src.agents.base_agent import BaseAgent
from src.utils.cache import TokenBudget

logger = logging.getLogger(__name__)

# 5-point confidence scale
CONFIDENCE_LEVELS = [
    "most_likely_include",
    "likely_include",
    "undecided",
    "likely_exclude",
    "most_likely_exclude",
]

INCLUDE_SIDE = {"most_likely_include", "likely_include"}
EXCLUDE_SIDE = {"likely_exclude", "most_likely_exclude"}
UNDECIDED = {"undecided"}
FIRM_INCLUDE = {"most_likely_include", "likely_include"}
FIRM_EXCLUDE = {"likely_exclude", "most_likely_exclude"}


class ScreenerAgent(BaseAgent):
    """Title/abstract screener using 5-point confidence scale."""

    def __init__(self, role_name: str = "screener_1",
                 budget: Optional[TokenBudget] = None):
        super().__init__(role_name=role_name, budget=budget)

    def screen_study(self, study: dict, criteria: dict) -> dict:
        """Screen a single study. Returns decision + reasoning."""
        system_prompt = self._build_system_prompt(criteria)
        user_prompt = self._build_screening_prompt(study, criteria)

        result = self.call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            expect_json=True,
            cache_namespace=f"screening_{self.role_name}",
            description=f"Screen {study.get('study_id', 'unknown')}",
        )

        parsed = result.get("parsed", {})
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if not isinstance(parsed, dict) or not parsed:
            return {
                "decision": "human_review",
                "confidence": "undecided",
                "reasoning": "Parse failure — requires human review",
                "parse_error": True,
            }

        # Normalize confidence to valid level
        confidence = parsed.get("confidence", parsed.get("decision", "undecided"))
        if confidence not in CONFIDENCE_LEVELS:
            confidence = "undecided"

        if confidence in UNDECIDED:
            decision = "human_review"
        elif confidence in INCLUDE_SIDE:
            decision = "include"
        else:
            decision = "exclude"

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", ""),
        }

    def _build_system_prompt(self, criteria: dict) -> str:
        """Build system prompt with static + dynamic exclusion rules."""
        base = self._prompt_config.get("system_prompt", "")

        # Build dynamic exclusion rules from screening_criteria.json
        exclusions = criteria.get("exclusion", [])
        if exclusions:
            dynamic = "The following PICO-specific exclusion criteria apply:\n"
            for i, exc in enumerate(exclusions, 1):
                dynamic += f"  {i}. {exc}\n"
        else:
            dynamic = "(No additional PICO-specific exclusion criteria provided)\n"

        return base.replace("{dynamic_exclusion_rules}", dynamic)

    def _build_screening_prompt(self, study: dict, criteria: dict) -> str:
        template = self._prompt_config.get("user_prompt_template", "")
        if template:
            return template.format(
                title=study.get("title", ""),
                abstract=study.get("abstract", ""),
                inclusion=json.dumps(criteria.get("inclusion", [])),
                exclusion=json.dumps(criteria.get("exclusion", [])),
            )

        return (
            f"Title: {study.get('title', 'N/A')}\n"
            f"Abstract: {study.get('abstract', 'N/A')}\n\n"
            f"Inclusion criteria: {json.dumps(criteria.get('inclusion', []))}\n"
            f"Exclusion criteria: {json.dumps(criteria.get('exclusion', []))}\n\n"
            f"Rate this study on a 5-point scale:\n"
            f"- most_likely_include\n- likely_include\n- undecided\n"
            f"- likely_exclude\n- most_likely_exclude\n\n"
            f"Return JSON: {{\"confidence\": \"<level>\", \"reasoning\": \"<brief>\"}}"
        )


class ArbiterAgent(BaseAgent):
    """Resolves firm conflicts between dual screeners."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        super().__init__(role_name="arbiter", budget=budget)

    def arbitrate(self, study: dict, screener1_result: dict,
                  screener2_result: dict, criteria: dict) -> dict:
        """Make final decision on a conflicting study."""
        system_prompt = self._prompt_config.get("system_prompt", "")
        user_prompt = self._build_arbiter_prompt(
            study, screener1_result, screener2_result, criteria
        )

        result = self.call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            expect_json=True,
            cache_namespace="screening_arbiter",
            description=f"Arbitrate {study.get('study_id', 'unknown')}",
        )

        parsed = result.get("parsed", {})
        if not parsed:
            return {
                "decision": "include",
                "confidence": "likely_include",
                "reasoning": "Arbiter parse failure — defaulting to include",
            }

        confidence = parsed.get("confidence", "likely_include")
        if confidence not in CONFIDENCE_LEVELS:
            confidence = "likely_include"

        return {
            "decision": "include" if confidence in INCLUDE_SIDE else "exclude",
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", ""),
        }

    def _build_arbiter_prompt(self, study, s1, s2, criteria):
        return (
            f"Two screeners disagreed on this study.\n\n"
            f"Title: {study.get('title', 'N/A')}\n"
            f"Abstract: {study.get('abstract', 'N/A')}\n\n"
            f"Screener 1: {s1.get('confidence')} — {s1.get('reasoning', '')}\n"
            f"Screener 2: {s2.get('confidence')} — {s2.get('reasoning', '')}\n\n"
            f"Criteria: {json.dumps(criteria)}\n\n"
            f"Make the final decision. Return JSON: "
            f"{{\"confidence\": \"<level>\", \"reasoning\": \"<brief>\"}}"
        )


# ======================================================================
# Dual Screening Orchestrator
# ======================================================================

def run_dual_screening(
    studies: list,
    criteria: dict,
    screener1: ScreenerAgent,
    screener2: ScreenerAgent,
    arbiter: ArbiterAgent,
    checkpoint_path: str = "",
) -> dict:
    """
    Run dual-screening pipeline with conflict resolution.
    Supports checkpointing: saves progress every 50 studies and resumes from checkpoint.

    Returns:
        {
            "included": [...],
            "excluded": [...],
            "human_review_queue": [...],
            "screening_results": [...],
            "stats": {...},
        }
    """
    from tqdm import tqdm

    included = []
    excluded = []
    human_review = []
    all_results = []

    # Resume from checkpoint if available
    done_ids: set = set()
    if checkpoint_path:
        _cp = Path(checkpoint_path)
        if _cp.exists():
            import json as _json
            _cp_data = _json.loads(_cp.read_text(encoding="utf-8"))
            all_results = _cp_data.get("results", [])
            done_ids = {r["study_id"] for r in all_results}
            # Rebuild lists from checkpoint
            for study in studies:
                sid = study.get("study_id", "unknown")
                if sid in done_ids:
                    res = next((r for r in all_results if r["study_id"] == sid), None)
                    if res:
                        study["screening_result"] = res
                        if res["final_decision"] == "include":
                            included.append(study)
                        elif res["final_decision"] == "human_review":
                            human_review.append(study)
                        else:
                            excluded.append(study)
            logger.info(f"Resumed from checkpoint: {len(done_ids)} already screened")

    for study in tqdm(studies, desc="Dual screening"):
        study_id = study.get("study_id", "unknown")

        # Skip already-screened studies (resume support)
        if study_id in done_ids:
            continue

        # Both screeners evaluate
        s1 = screener1.screen_study(study, criteria)
        s2 = screener2.screen_study(study, criteria)

        # Map decisions: undecided/parse_error → human_review
        s1_side = s1["decision"]  # "include", "exclude", or "human_review"
        s2_side = s2["decision"]

        # Either screener uncertain → human review (no arbiter waste)
        if s1_side == "human_review" or s2_side == "human_review":
            final_decision = "human_review"
            final_confidence = "undecided"
            method = "undecided_conflict"
        # Agreement
        elif s1_side == s2_side:
            final_decision = s1_side
            final_confidence = s1["confidence"]
            method = "agreement"
        # Both firm but disagree -> arbiter
        else:
            arb = arbiter.arbitrate(study, s1, s2, criteria)
            final_decision = arb["decision"]
            final_confidence = arb["confidence"]
            method = "arbiter"

        result = {
            "study_id": study_id,
            "screener1": s1,
            "screener2": s2,
            "final_decision": final_decision,
            "final_confidence": final_confidence,
            "resolution_method": method,
        }

        if final_decision == "include":
            study["screening_result"] = result
            included.append(study)
        elif final_decision == "human_review":
            study["screening_result"] = result
            human_review.append(study)
        else:
            study["screening_result"] = result
            excluded.append(study)

        all_results.append(result)

        # Checkpoint every 50 studies
        if checkpoint_path and len(all_results) % 50 == 0:
            _cp = Path(checkpoint_path)
            _cp.parent.mkdir(parents=True, exist_ok=True)
            _cp.write_text(json.dumps({"results": all_results}, default=str), encoding="utf-8")
            logger.info(f"Checkpoint saved: {len(all_results)} screened")

    # Final checkpoint save
    if checkpoint_path:
        _cp = Path(checkpoint_path)
        _cp.parent.mkdir(parents=True, exist_ok=True)
        _cp.write_text(json.dumps({"results": all_results}, default=str), encoding="utf-8")

    # Compute inter-rater stats
    stats = _compute_screening_stats(all_results)

    logger.info(
        f"Dual screening complete: {len(included)} included, "
        f"{len(excluded)} excluded, {len(human_review)} for human review"
    )

    return {
        "included": included,
        "excluded": excluded,
        "human_review_queue": human_review,
        "screening_results": all_results,
        "stats": stats,
    }


# ======================================================================
# Single-Agent Screening (benchmark / ablation mode)
# ======================================================================

def run_single_screening(
    studies: list,
    criteria: dict,
    screener: ScreenerAgent,
    checkpoint_path: str = "",
) -> dict:
    """
    Run single-agent screening pipeline.
    Hidden mode for benchmarking — activated via --single flag.
    Mirrors otto-SR's standalone reviewer approach.

    Returns same structure as run_dual_screening for comparability.
    """
    from tqdm import tqdm

    included = []
    excluded = []
    human_review = []
    all_results = []

    # Resume from checkpoint
    done_ids: set = set()
    if checkpoint_path:
        _cp = Path(checkpoint_path)
        if _cp.exists():
            import json as _json
            _cp_data = _json.loads(_cp.read_text(encoding="utf-8"))
            all_results = _cp_data.get("results", [])
            done_ids = {r["study_id"] for r in all_results}
            for study in studies:
                sid = study.get("study_id", "unknown")
                if sid in done_ids:
                    res = next((r for r in all_results if r["study_id"] == sid), None)
                    if res:
                        study["screening_result"] = res
                        if res["final_decision"] == "include":
                            included.append(study)
                        elif res["final_decision"] == "human_review":
                            human_review.append(study)
                        else:
                            excluded.append(study)
            logger.info(f"Resumed from checkpoint: {len(done_ids)} already screened")

    for study in tqdm(studies, desc="Single-agent screening"):
        study_id = study.get("study_id", "unknown")
        if study_id in done_ids:
            continue

        s1 = screener.screen_study(study, criteria)

        final_decision = s1["decision"]
        final_confidence = s1["confidence"]

        result = {
            "study_id": study_id,
            "screener1": s1,
            "screener2": None,
            "final_decision": final_decision,
            "final_confidence": final_confidence,
            "resolution_method": "single_agent",
        }

        if final_decision == "include":
            study["screening_result"] = result
            included.append(study)
        elif final_decision == "human_review":
            study["screening_result"] = result
            human_review.append(study)
        else:
            study["screening_result"] = result
            excluded.append(study)

        all_results.append(result)

        if checkpoint_path and len(all_results) % 50 == 0:
            _cp = Path(checkpoint_path)
            _cp.parent.mkdir(parents=True, exist_ok=True)
            _cp.write_text(json.dumps({"results": all_results}, default=str), encoding="utf-8")
            logger.info(f"Checkpoint saved: {len(all_results)} screened")

    if checkpoint_path:
        _cp = Path(checkpoint_path)
        _cp.parent.mkdir(parents=True, exist_ok=True)
        _cp.write_text(json.dumps({"results": all_results}, default=str), encoding="utf-8")

    # Stats for single agent (no kappa — just distribution)
    confidence_dist = Counter(r["screener1"]["confidence"] for r in all_results)
    decision_dist = Counter(r["final_decision"] for r in all_results)
    stats = {
        "total_screened": len(all_results),
        "mode": "single_agent",
        "screener_role": screener.role_name,
        "decision_distribution": dict(decision_dist),
        "confidence_distribution": dict(confidence_dist),
    }

    logger.info(
        f"Single-agent screening complete: {len(included)} included, "
        f"{len(excluded)} excluded, {len(human_review)} for human review"
    )

    return {
        "included": included,
        "excluded": excluded,
        "human_review_queue": human_review,
        "screening_results": all_results,
        "stats": stats,
    }


def _compute_screening_stats(results: list) -> dict:
    """Compute Cohen's kappa and agreement statistics."""
    if not results:
        return {}

    agreements = 0
    total = len(results)

    s1_decisions = []
    s2_decisions = []

    for r in results:
        s1d = r["screener1"]["decision"]  # include, exclude, or human_review
        s2d = r["screener2"]["decision"]
        # For kappa: map to binary (human_review treated as separate, excluded from kappa)
        s1_binary = 1 if s1d == "include" else 0
        s2_binary = 1 if s2d == "include" else 0

        s1_decisions.append(s1_binary)
        s2_decisions.append(s2_binary)

        if s1_binary == s2_binary:
            agreements += 1

    # Cohen's kappa
    po = agreements / total
    s1_pos = sum(s1_decisions) / total
    s2_pos = sum(s2_decisions) / total
    pe = s1_pos * s2_pos + (1 - s1_pos) * (1 - s2_pos)

    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    # PABAK (Prevalence-Adjusted Bias-Adjusted Kappa)
    pabak = 2 * po - 1

    return {
        "total_screened": total,
        "agreement_rate": round(po, 4),
        "cohens_kappa": round(kappa, 4),
        "pabak": round(pabak, 4),
        "resolution_methods": dict(Counter(r["resolution_method"] for r in results)),
        "confidence_distribution": {
            "screener1": dict(Counter(r["screener1"]["confidence"] for r in results)),
            "screener2": dict(Counter(r["screener2"]["confidence"] for r in results)),
        },
    }
