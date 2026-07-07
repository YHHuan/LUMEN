"""
LangGraph nodes for screening pipeline.

Integrates: prescreen → dual screen → route → arbiter.
"""
from __future__ import annotations

import re
import structlog

from lumen.agents.screener import ScreenerAgent
from lumen.agents.arbiter import ArbiterAgent, needs_arbiter, resolve_screening
from lumen.agents.fulltext_screener import FulltextScreenerAgent
from lumen.agents.screening_signals import ReasonCode, annotate_screening_result

logger = structlog.get_logger()

# Prescreen exclusion reasons → canonical reason codes.
_PRESCREEN_REASON_CODES = {
    "exclusion_keyword_match": ReasonCode.EXCLUSION_KEYWORD,
    "no_required_keyword": ReasonCode.NO_REQUIRED_KEYWORD,
}


def prescreen_node(state: dict) -> dict:
    """
    Deterministic keyword filter (no LLM). Fast exclusion of obviously
    irrelevant studies.

    Quarantine rescue: randomly sample 5% of excluded for FAST-tier check.
    """
    studies = state.get("deduplicated_studies", [])
    criteria = state.get("screening_criteria", {})
    exclusion_keywords = criteria.get("exclusion_keywords", [])
    required_keywords = criteria.get("required_keywords", [])

    passed = []
    excluded = []

    for study in studies:
        text = (
            f"{study.get('title', '')} {study.get('abstract', '')}"
        ).lower()

        # Exclude if any exclusion keyword is present
        if any(kw.lower() in text for kw in exclusion_keywords):
            study["prescreen"] = "excluded"
            study["prescreen_reason"] = "exclusion_keyword_match"
            excluded.append(study)
            continue

        # Exclude if none of the required keywords are present (if specified)
        if required_keywords and not any(kw.lower() in text for kw in required_keywords):
            study["prescreen"] = "excluded"
            study["prescreen_reason"] = "no_required_keyword"
            excluded.append(study)
            continue

        study["prescreen"] = "passed"
        passed.append(study)

    logger.info("prescreen_complete",
                total=len(studies), passed=len(passed), excluded=len(excluded))

    return {
        "prescreen_results": [
            {"study_id": s.get("study_id", s.get("id")),
             "prescreen": s["prescreen"],
             "prescreen_reason": s.get("prescreen_reason", ""),
             "reason_code": _PRESCREEN_REASON_CODES.get(
                 s.get("prescreen_reason", ""))}
            for s in studies
        ],
        # Only studies that passed go forward
        "deduplicated_studies": passed,
    }


def screen_ta_node(state: dict, screener1: ScreenerAgent,
                   screener2: ScreenerAgent, arbiter: ArbiterAgent,
                   config: dict | None = None) -> dict:
    """
    Title/Abstract screening node.

    1. Dual screen each study
    2. Route: agreement + high confidence → auto-decide
    3. Disagreement or low confidence → arbiter
    4. Arbiter low confidence → human_review

    Every result is annotated with an explicit three-state signal
    (``tri_state`` / ``screening_state`` / ``review_required`` / ``flags``) so
    borderline decisions surface to the review queue instead of being silently
    absorbed by the confidence threshold. Annotation never changes
    ``final_decision`` — the pipeline routing is unchanged.
    """
    studies = state.get("deduplicated_studies", [])
    pico = state.get("pico", {})
    criteria = state.get("screening_criteria", {})

    screening_cfg = (config or {}).get("screening", {})
    auto_threshold = screening_cfg.get("confidence_auto_threshold", 80)
    unclear_margin = screening_cfg.get("unclear_margin", 5)

    screening_results = []

    for study in studies:
        # Dual screening
        s1 = screener1.screen_single(study, pico, criteria, screener_id="1")
        s2 = screener2.screen_single(study, pico, criteria, screener_id="2")

        arbiter_result = None
        if needs_arbiter(s1, s2):
            arbiter_result = arbiter.resolve(study, s1, s2, pico, criteria)

        resolution = resolve_screening(s1, s2, arbiter_result)

        entry = {
            "study_id": study.get("study_id", study.get("id", "unknown")),
            "screener1": s1,
            "screener2": s2,
            "arbiter": arbiter_result,
            "final_decision": resolution["final_decision"],
            "method": resolution["method"],
            "confidence": resolution["confidence"],
            "parse_error": bool(s1.get("parse_error") or s2.get("parse_error")
                                or (arbiter_result or {}).get("parse_error")),
        }
        annotate_screening_result(
            entry, auto_threshold=auto_threshold, unclear_margin=unclear_margin)
        screening_results.append(entry)

    # Filter included studies
    included = [
        s for s in studies
        if any(
            r["study_id"] == s.get("study_id", s.get("id"))
            and r["final_decision"] == "include"
            for r in screening_results
        )
    ]

    # Everything a human should actually look at: human_review + borderline
    # excludes + flagged (disagreement) includes. Previously computed and then
    # discarded — now surfaced to state and, downstream, to review_queue.md.
    needs_human_review = [r for r in screening_results if r["review_required"]]

    logger.info("screening_complete",
                total=len(studies),
                included=len(included),
                excluded=sum(1 for r in screening_results
                             if r["final_decision"] == "exclude"),
                needs_human_review=len(needs_human_review))

    return {
        "screening_results": screening_results,
        "included_studies": included,
        "needs_human_review": needs_human_review,
    }
