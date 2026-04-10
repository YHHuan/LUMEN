"""
LUMEN v3 Context Management utilities.

Not an agent — utility functions called by graph nodes to manage context.
- PICO drift detection (keyword heuristic, no LLM)
- Context compression (FAST tier LLM)
- Scoped context builder (each agent sees only what it needs)
"""
from __future__ import annotations

import re
from typing import Any

from lumen.core.state import LumenState


def check_pico_drift(current_task: str, pico: dict) -> dict | None:
    """Check if the current task still aligns with the PICO definition.

    Uses keyword overlap heuristic (fast, free, no LLM).
    Returns None if OK, or {warning, suggestion} if drift detected.
    """
    if not pico:
        return None

    # Collect PICO keywords
    pico_keywords: set[str] = set()
    for key in ("population", "intervention", "comparator", "outcome",
                "study_design", "condition", "disease"):
        val = pico.get(key, "")
        if isinstance(val, str):
            pico_keywords.update(_extract_keywords(val))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    pico_keywords.update(_extract_keywords(item))

    if not pico_keywords:
        return None

    task_keywords = _extract_keywords(current_task)
    if not task_keywords:
        return None

    overlap = pico_keywords & task_keywords
    overlap_ratio = len(overlap) / len(task_keywords) if task_keywords else 0

    if overlap_ratio < 0.1 and len(task_keywords) > 3:
        return {
            "warning": f"Low PICO keyword overlap ({overlap_ratio:.0%}). "
                       f"Task may have drifted from original research question.",
            "suggestion": "Review whether the current analysis still addresses the PICO.",
            "overlap_ratio": overlap_ratio,
            "task_keywords": sorted(task_keywords),
            "pico_keywords": sorted(pico_keywords)[:20],
        }

    return None


def compress_context(text: str, max_tokens: int = 2000,
                     router: Any | None = None) -> str:
    """Compress text if it exceeds max_tokens.

    If a router is provided and text is long, uses FAST tier LLM to compress.
    Otherwise, falls back to extractive truncation preserving key information.
    """
    # Rough token estimate: 1 token ~ 4 chars for English, ~2 chars for CJK
    estimated_tokens = len(text) // 3

    if estimated_tokens <= max_tokens:
        return text

    if router is not None:
        try:
            messages = [
                {"role": "system", "content": (
                    "Compress the following text. Preserve: numbers, study IDs, "
                    "effect sizes, confidence intervals, key decisions. "
                    "Discard: reasoning process, repeated descriptions."
                )},
                {"role": "user", "content": text},
            ]
            compressed, _ = router.call(tier="fast", messages=messages,
                                        agent_name="context_compressor")
            return compressed
        except Exception:
            pass  # Fall through to extractive truncation

    # Extractive fallback: keep first and last portions
    char_budget = max_tokens * 3
    if len(text) <= char_budget:
        return text
    half = char_budget // 2
    return text[:half] + "\n\n[... compressed ...]\n\n" + text[-half:]


def build_agent_context(state: LumenState, agent_name: str) -> dict:
    """Build a scoped context slice for a specific agent.

    Each agent only sees the state keys it needs, reducing noise and tokens.
    """
    CONTEXT_SCOPES: dict[str, list[str]] = {
        "screener": ["pico", "screening_criteria"],
        "arbiter": ["pico", "screening_criteria"],
        "fulltext_screener": ["pico", "screening_criteria"],
        "extractor": ["pico", "included_studies"],
        "harmonizer": ["pico", "extractions"],
        "statistician": ["pico", "harmonized_data", "analysis_plan"],
        "writer": [
            "pico", "statistics_results", "quality_assessments",
            "evidence_synthesis", "included_studies",
        ],
        "pico_interviewer": ["pico"],
        "strategy_generator": ["pico", "screening_criteria"],
    }

    scope_keys = CONTEXT_SCOPES.get(agent_name, list(state.keys()))
    return {k: state[k] for k in scope_keys if k in state}


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase alphabetic keywords (>= 3 chars) from text."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    # Filter common stop words
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "are", "was",
        "were", "been", "have", "has", "had", "will", "would", "could",
        "should", "not", "but", "they", "their", "which", "when", "what",
        "than", "more", "also", "between", "after", "before", "each",
        "other", "into", "through", "during", "about", "these", "those",
        "some", "such", "only", "over", "both", "using", "used",
    }
    return set(words) - stop
