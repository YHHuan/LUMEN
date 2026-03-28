"""
Context-Aware Pre-screening — LUMEN v2
========================================
Replaces v1's naive unigram keyword exclusion with:
1. Context-aware bigram matching (prevents false exclusions like "stimulation protocol")
2. Quarantine pool for ambiguous matches
3. Rescue pipeline: regex positive signal + LLM-lite rescue
"""

import re
import json
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ======================================================================
# Context-Aware Keyword Configuration
# ======================================================================

EXCLUSION_BIGRAMS = {
    "protocol": {
        "exclude_contexts": [
            "study protocol", "trial protocol", "protocol registration",
            "published protocol", "protocol for a", "protocol of a",
            "systematic review protocol", "registered protocol",
            "protocol paper", "protocol design", "protocol article",
        ],
        "safe_contexts": [
            "stimulation protocol", "treatment protocol", "plasticity protocol",
            "experimental protocol", "dosing protocol", "intervention protocol",
            "therapeutic protocol", "training protocol", "exercise protocol",
            "rehabilitation protocol", "session protocol", "clinical protocol",
        ],
        "bare_action": "quarantine",
    },
    "review": {
        "exclude_contexts": [
            "systematic review", "literature review", "narrative review",
            "scoping review", "umbrella review", "review article",
            "review of reviews", "meta-review", "rapid review",
        ],
        "safe_contexts": [
            "review board", "ethics review", "institutional review",
            "peer review", "reviewed by", "under review",
        ],
        "bare_action": "quarantine",
    },
    "case report": {
        "exclude_contexts": [
            "case report", "case series", "single case",
        ],
        "safe_contexts": [
            "case report form", "case report forms",
        ],
        "bare_action": "exclude",
    },
    "commentary": {
        "exclude_contexts": [
            "commentary", "editorial commentary", "invited commentary",
        ],
        "safe_contexts": [],
        "bare_action": "exclude",
    },
    "editorial": {
        "exclude_contexts": [
            "editorial", "editor's note", "letter to editor",
        ],
        "safe_contexts": [],
        "bare_action": "exclude",
    },
    "erratum": {
        "exclude_contexts": [
            "erratum", "corrigendum", "retraction", "correction notice",
        ],
        "safe_contexts": [],
        "bare_action": "exclude",
    },
    "meta-analysis": {
        "exclude_contexts": [
            "meta-analysis of", "meta-analysis and systematic review",
            "a meta-analysis", "systematic review and meta-analysis",
        ],
        "safe_contexts": [
            "included in meta-analysis", "for meta-analysis",
            "meta-analysis showed", "previous meta-analysis",
        ],
        "bare_action": "quarantine",
    },
    "animal": {
        "exclude_contexts": [
            "animal model", "animal study", "rat model", "mouse model",
            "in vivo animal", "animal experiment",
        ],
        "safe_contexts": [
            "animal naming", "animal fluency",
            # Background references to animal work — not the study itself
            "in animal models", "animal models of", "animal studies have",
            "animal data", "preclinical", "animal experiments have",
            "from animal", "previous animal",
        ],
        "bare_action": "quarantine",
    },
}

# Simple unigram exclusions (no context needed)
SIMPLE_EXCLUSION_KEYWORDS = [
    "retracted", "withdrawn", "conference abstract only",
]


def context_aware_keyword_check(
    title: str,
    abstract: str,
    keyword_config: dict = None,
    pico: dict = None,
) -> str:
    """
    Check a study against context-aware exclusion rules.

    PICO-aware: if pico specifies RCT study design AND the title contains
    strong RCT signals, exclusion keywords in the abstract background
    (e.g., "animal model" in introduction) are overridden.

    Returns:
        "include" — no exclusion keyword found
        "exclude" — clear exclusion context matched
        "quarantine" — ambiguous match, needs rescue pipeline
    """
    if keyword_config is None:
        keyword_config = EXCLUSION_BIGRAMS

    text = f"{title} {abstract}".lower()
    title_lower = title.lower()

    # PICO-aware RCT override: if title has strong RCT signals AND the
    # exclusion keyword only appears in abstract (not title), quarantine
    # instead of excluding — the study is likely a human RCT that references
    # prior animal/review work in its abstract background.
    _RCT_TITLE_SIGNALS = [
        "randomized", "randomised", "double-blind", "placebo-controlled",
        "open-label", "parallel-group", "crossover", "a pilot",
        "clinical trial", "rct",
    ]
    title_has_rct = any(sig in title_lower for sig in _RCT_TITLE_SIGNALS)

    # Simple unigram exclusions first
    for kw in SIMPLE_EXCLUSION_KEYWORDS:
        if kw.lower() in text:
            return "exclude"

    # Context-aware bigram matching
    for keyword, config in keyword_config.items():
        keyword_lower = keyword.lower()

        if keyword_lower not in text:
            continue

        # Check safe contexts first (higher priority)
        safe_hit = any(ctx.lower() in text for ctx in config["safe_contexts"])
        if safe_hit:
            continue  # Safe context found — do not exclude

        # Check exclude contexts
        exclude_hit = any(ctx.lower() in text for ctx in config["exclude_contexts"])
        if exclude_hit:
            # PICO-aware override: if keyword is only in abstract (not title)
            # AND title has RCT signals → quarantine instead of hard exclude
            keyword_in_title = keyword_lower in title_lower
            if title_has_rct and not keyword_in_title:
                logger.info(
                    f"RCT override: '{keyword}' in abstract but title is RCT-like, "
                    f"quarantining instead of excluding"
                )
                return "quarantine"
            return "exclude"

        # Bare keyword with no clear context
        return config.get("bare_action", "quarantine")

    return "include"


# ======================================================================
# Four-Layer Pre-screening Filter (v1 logic, upgraded)
# ======================================================================

def run_prescreen(
    studies: list,
    keyword_config: dict = None,
    *,
    pico: dict = None,
) -> dict:
    """
    Run the full pre-screening pipeline.

    Args:
        pico: Optional PICO config dict. If pico.get("exclude_no_abstract")
              is True, studies with blank/short abstracts are excluded
              instead of flagged.

    Returns:
        {
            "passed": [...],
            "excluded": [...],
            "quarantined": [...],
            "stats": {...},
        }
    """
    exclude_no_abstract = (pico or {}).get("exclude_no_abstract", False)
    passed = []
    excluded = []
    quarantined = []

    for study in studies:
        title = study.get("title", "")
        abstract = study.get("abstract", "")

        # Layer 1: Missing title
        if not title or title.strip() == "":
            study["exclusion_reason"] = "missing_title"
            excluded.append(study)
            continue

        # Layer 2: Duplicate title (handled elsewhere in dedup)

        # Layer 3: Context-aware keyword check
        result = context_aware_keyword_check(title, abstract, keyword_config)

        if result == "exclude":
            study["exclusion_reason"] = "keyword_exclusion"
            excluded.append(study)
        elif result == "quarantine":
            study["quarantine_reason"] = "ambiguous_keyword"
            quarantined.append(study)
        else:
            # Layer 4: Blank abstract check
            if not abstract or abstract.strip() == "" or len(abstract.strip()) < 50:
                if exclude_no_abstract:
                    study["exclusion_reason"] = "no_abstract"
                    excluded.append(study)
                    continue
                study["flag"] = "blank_or_short_abstract"
            passed.append(study)

    stats = {
        "total_input": len(studies),
        "passed": len(passed),
        "excluded": len(excluded),
        "quarantined": len(quarantined),
    }

    logger.info(
        f"Pre-screening: {stats['passed']} passed, "
        f"{stats['excluded']} excluded, "
        f"{stats['quarantined']} quarantined"
    )

    return {
        "passed": passed,
        "excluded": excluded,
        "quarantined": quarantined,
        "stats": stats,
    }


# ======================================================================
# Rescue Pipeline
# ======================================================================

def build_positive_signals(pico_config: dict) -> dict:
    """Build positive keyword signals from PICO config for rescue."""
    signals = {"intervention": [], "population": [], "study_design": []}

    # From PICO fields
    for field in ["intervention", "population", "comparison", "outcome"]:
        value = pico_config.get("pico", {}).get(field, "")
        if value:
            # Split on common delimiters
            keywords = re.split(r"[,;/]", value)
            target = "intervention" if field in ("intervention", "comparison") else "population"
            signals[target].extend(kw.strip() for kw in keywords if kw.strip())

    # Default study design signals
    signals["study_design"] = [
        "randomized", "RCT", "controlled trial", "sham", "placebo",
        "double-blind", "single-blind",
    ]

    return signals


def regex_rescue(quarantined_studies: list, pico_config: dict) -> tuple:
    """
    Stage A: Rescue quarantined studies matching positive PICO signals.
    Zero LLM cost.

    Returns: (rescued, still_quarantined)
    """
    positive_signals = build_positive_signals(pico_config)

    # Also load rescue_keywords.json from Phase 1 if available
    rescued = []
    still_quarantined = []

    for study in quarantined_studies:
        text = f"{study.get('title', '')} {study.get('abstract', '')}".lower()

        intervention_hit = any(
            kw.lower() in text for kw in positive_signals.get("intervention", [])
        )
        population_hit = any(
            kw.lower() in text for kw in positive_signals.get("population", [])
        )

        if intervention_hit and population_hit:
            study["rescue_reason"] = "positive_signal_match"
            study["rescue_stage"] = "regex"
            rescued.append(study)
        else:
            still_quarantined.append(study)

    logger.info(
        f"Regex rescue: {len(rescued)} rescued, "
        f"{len(still_quarantined)} still quarantined"
    )

    return rescued, still_quarantined


def llm_lite_rescue(
    quarantined_studies: list,
    pico_summary: str,
    agent,
    batch_size: int = 20,
) -> tuple:
    """
    Stage B: LLM-lite rescue for remaining quarantined studies.
    Uses cheapest model (e.g. Gemini Flash Lite) with temperature=0.

    Returns: (rescued, final_excluded)
    """
    if not quarantined_studies:
        return [], []

    rescued = []
    final_excluded = []

    # Process in batches
    for i in range(0, len(quarantined_studies), batch_size):
        batch = quarantined_studies[i:i + batch_size]

        studies_text = ""
        for j, study in enumerate(batch):
            title = study.get("title", "N/A")
            abstract = study.get("abstract", "")[:300]
            studies_text += f"\n{j + 1}. Title: {title}\n   Abstract: {abstract}\n"

        prompt = (
            f"You are screening studies for a systematic review.\n"
            f"PICO: {pico_summary}\n\n"
            f"For each study below, answer ONLY 'RESCUE' or 'EXCLUDE' "
            f"followed by a 1-sentence reason.\n"
            f"A study should be RESCUED if it MIGHT be relevant "
            f"(err on the side of inclusion).\n"
            f"{studies_text}"
        )

        result = agent.call_llm(
            prompt=prompt,
            cache_namespace="rescue_screen",
            description=f"LLM-lite rescue batch {i // batch_size + 1}",
        )

        # Parse results
        content = result.get("content", "")
        lines = content.strip().split("\n")

        for j, study in enumerate(batch):
            line_idx = j
            if line_idx < len(lines):
                line = lines[line_idx].upper()
                if "RESCUE" in line:
                    study["rescue_reason"] = "llm_lite_rescue"
                    study["rescue_stage"] = "llm_lite"
                    rescued.append(study)
                else:
                    study["exclusion_reason"] = "llm_lite_excluded"
                    final_excluded.append(study)
            else:
                # Default to rescue (err on inclusion)
                study["rescue_reason"] = "llm_lite_default_rescue"
                study["rescue_stage"] = "llm_lite"
                rescued.append(study)

    logger.info(
        f"LLM-lite rescue: {len(rescued)} rescued, "
        f"{len(final_excluded)} excluded"
    )

    return rescued, final_excluded
