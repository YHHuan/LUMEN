"""
LUMEN v3 — HITL screening signal layer.

Adds an explicit three-state screening outcome and structured exclusion
reason codes on top of v3's existing confidence-based routing, so that
borderline / uncertain decisions surface to a human review queue instead of
being silently absorbed by a confidence threshold.

Three-state convention (otto-SR / TrialMind, cited by LUMEN) — {-1, 0, +1}:
    +1  include
    -1  exclude
     0  unclear   (needs human review)

Everything here is PURE and deterministic: no LLM calls, no I/O. This is the
"reminder layer" — it never changes the automated ``final_decision`` (so the
pipeline keeps flowing exactly as before); it only *annotates* each result so
a human can see what the automation was unsure about.
"""
from __future__ import annotations

import re

# ── Decision labels ─────────────────────────────────────────────────
INCLUDE = "include"
EXCLUDE = "exclude"
UNCLEAR = "unclear"
HUMAN_REVIEW = "human_review"  # v3 arbiter routing label; treated as unclear (0)

_TRI_STATE = {INCLUDE: 1, EXCLUDE: -1, UNCLEAR: 0, HUMAN_REVIEW: 0}


def tri_state(decision: str) -> int:
    """Map a screening decision string to {-1, 0, +1}. Unknown → 0 (unclear)."""
    return _TRI_STATE.get((decision or "").lower().strip(), 0)


# ── Structured exclusion / flag reason codes ────────────────────────
class ReasonCode:
    """Canonical, machine-readable reasons for exclusion or human-review flags."""

    # Domain (SRMA) exclusion reasons
    NOT_PRIMARY_STUDY = "NOT_PRIMARY_STUDY"
    DUPLICATE_COHORT = "DUPLICATE_COHORT"
    MIXED_IV_TO_ORAL = "MIXED_IV_TO_ORAL"
    BIOMARKER_GUIDED = "BIOMARKER_GUIDED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    WRONG_POPULATION = "WRONG_POPULATION"
    WRONG_INTERVENTION = "WRONG_INTERVENTION"
    WRONG_STUDY_DESIGN = "WRONG_STUDY_DESIGN"

    # Prescreen (deterministic keyword filter) reasons
    EXCLUSION_KEYWORD = "EXCLUSION_KEYWORD"
    NO_REQUIRED_KEYWORD = "NO_REQUIRED_KEYWORD"

    # Routing / uncertainty flags (not exclusion reasons per se)
    SCREENER_DISAGREEMENT = "SCREENER_DISAGREEMENT"
    BORDERLINE_CONFIDENCE = "BORDERLINE_CONFIDENCE"
    ARBITER_LOW_CONFIDENCE = "ARBITER_LOW_CONFIDENCE"
    PARSE_FAILURE = "PARSE_FAILURE"

    UNSPECIFIED = "UNSPECIFIED"


ALL_REASON_CODES = frozenset(
    v for k, v in vars(ReasonCode).items()
    if k.isupper() and isinstance(v, str)
)


def is_valid_reason_code(code: str) -> bool:
    """True if *code* is one of the canonical reason codes."""
    return code in ALL_REASON_CODES


# Ordered (first match wins) — most specific / decisive reasons first.
# Patterns are matched case-insensitively against the free-text rationale.
_REASON_PATTERNS: list[tuple[str, str]] = [
    (ReasonCode.BIOMARKER_GUIDED,
     r"procalcitonin|biomarker[- ]guided|crp[- ]guided|pct[- ]guided|"
     r"biomarker[- ]driven|guided by (a )?biomarker"),
    (ReasonCode.NOT_PRIMARY_STUDY,
     r"post[- ]?hoc|secondary analysis|sub[- ]?stud(y|ies)|satellite|"
     r"editorial|commentary|narrative review|systematic review|meta[- ]analysis|"
     r"protocol (paper|only)|conference abstract|no original data|"
     r"pooled analysis of"),
    (ReasonCode.DUPLICATE_COHORT,
     r"duplicate|same cohort|overlapping (population|cohort|sample)|"
     r"already included|companion (paper|report)"),
    (ReasonCode.MIXED_IV_TO_ORAL,
     r"iv[- ]to[- ]oral|intravenous[- ]to[- ]oral|switch therapy|"
     r"step[- ]down therapy|oral switch|sequential (iv|antibiotic)"),
    (ReasonCode.WRONG_POPULATION,
     r"typhoid|enteric fever|paediatric|pediatric|children|neonat|"
     r"animal|in vitro|healthy volunteer"),
    (ReasonCode.WRONG_INTERVENTION,
     r"wrong (drug|intervention|comparator)|not.{0,20}antibiotic|"
     r"no comparator"),
    (ReasonCode.WRONG_STUDY_DESIGN,
     r"observational|not (an )?rct|cohort study|case[- ]control|"
     r"case (report|series)|cross[- ]sectional|non[- ]randomi[sz]ed"),
    (ReasonCode.INSUFFICIENT_DATA,
     r"insufficient data|no (extractable|usable|reported) (outcome )?data|"
     r"outcomes? not reported|no outcome data|incomplete data"),
]


def classify_exclusion_reason(text: str,
                              default: str = ReasonCode.UNSPECIFIED) -> str:
    """Map a free-text exclusion rationale to a canonical :class:`ReasonCode`.

    Deterministic keyword heuristic (first match wins). This is the bridge that
    turns an LLM's prose reason into a structured code for the review queue and
    the PRISMA exclusion tally. Returns *default* when nothing matches.
    """
    if not text:
        return default
    t = str(text).lower()
    for code, pattern in _REASON_PATTERNS:
        if re.search(pattern, t):
            return code
    return default


# ── Borderline detection + result annotation ────────────────────────
def is_borderline(confidence: float, threshold: float, margin: float) -> bool:
    """True when *confidence* sits within *margin* above a decision *threshold*.

    Used to catch decisions that only *just* cleared the confidence bar — the
    ones most likely to be silently-wrong exclusions.
    """
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return True  # unparseable confidence → treat as uncertain
    return threshold <= c < (threshold + margin)


def annotate_screening_result(
    result: dict,
    *,
    auto_threshold: float = 80,
    unclear_margin: float = 5,
    disagreement_conf: float = 60,
) -> dict:
    """Attach HITL signals to a resolved screening result **in place**.

    Adds four keys without touching ``final_decision`` / ``method`` (so existing
    routing and the 258-test baseline are unaffected):

    - ``tri_state``      : {-1, 0, +1}; 0 whenever the case needs human review.
    - ``screening_state``: "include" | "exclude" | "unclear".
    - ``review_required``: bool — should this land in the review queue?
    - ``flags``          : list[ReasonCode] explaining *why* it was flagged.
    """
    decision = (result.get("final_decision") or "").lower().strip()
    method = result.get("method", "")
    confidence = result.get("confidence", 0)
    flags = list(result.get("flags", []))
    review_required = False

    if decision == HUMAN_REVIEW:
        review_required = True
        flags.append(ReasonCode.ARBITER_LOW_CONFIDENCE)
    elif decision == EXCLUDE:
        # An exclusion that only just cleared the confidence bar → surface it,
        # because a false exclusion silently drops a potentially eligible study.
        if is_borderline(confidence, auto_threshold, unclear_margin) \
                or confidence < auto_threshold:
            review_required = True
            flags.append(ReasonCode.BORDERLINE_CONFIDENCE)
    elif decision == INCLUDE:
        # Union include born from screener disagreement with a low-confidence
        # includer → still included (sensitivity), but flagged for a spot-check.
        if method == "union_include" and confidence < disagreement_conf:
            review_required = True
            flags.append(ReasonCode.SCREENER_DISAGREEMENT)

    if result.get("parse_error"):
        review_required = True
        if ReasonCode.PARSE_FAILURE not in flags:
            flags.append(ReasonCode.PARSE_FAILURE)

    # Includes stay +1 even when flagged (they ARE included); everything else
    # that needs review becomes an explicit unclear(0) state.
    if review_required and decision != INCLUDE:
        result["tri_state"] = 0
        result["screening_state"] = UNCLEAR
    else:
        result["tri_state"] = tri_state(decision)
        result["screening_state"] = decision or UNCLEAR

    result["review_required"] = review_required
    result["flags"] = flags
    return result
