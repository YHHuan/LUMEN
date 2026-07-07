"""
Deterministic quality bucketing (stabilize P0 — ported from V4 in spirit).

V4 assesses non-randomised studies with ROBINS-I in addition to v3's RoB2, then
buckets each study's overall risk-of-bias into a quality tier and maps the
proportion of high-RoB studies to a GRADE risk-of-bias downgrade
(lumen-code ``nodes/quality_node.py:152``). v3 already has RoB2 + GRADE, but no
ROBINS-I path and no standalone bucketing helper.

This module ports the deterministic parts only — the closed-form ROBINS-I
overall algorithm and the bucketing / downgrade maths. It is pure and
LLM-free (the per-domain judgements are produced upstream). It is intentionally
*standalone*: wiring it into the live GRADE path would change published GRADE
output and so is left to the user (see PORT_NOTES.md).

Vocabulary is normalised so it can never suffer the V4 "important vs CRITICAL"
mismatch: labels are matched case-insensitively against explicit sets.
"""
from __future__ import annotations

# ── ROBINS-I (Sterne et al. BMJ 2016) ───────────────────────────────
ROBINS_I_DOMAINS = [
    "confounding",
    "selection_of_participants",
    "classification_of_interventions",
    "deviations_from_intervention",
    "missing_data",
    "measurement_of_outcome",
    "selection_of_reported_result",
]
_ROBINS_RANK = {"low": 0, "moderate": 1, "serious": 2, "critical": 3}
_ROBINS_LABEL = {0: "Low", 1: "Moderate", 2: "Serious", 3: "Critical"}

# ── RoB2 (Sterne et al. BMJ 2019) overall labels ────────────────────
_ROB2_LEVELS = {"low", "some_concerns", "high"}

# Overall labels that count as "high risk of bias" for GRADE bucketing.
_HIGH_ROB = {"high", "serious", "critical"}
_MODERATE_ROB = {"some_concerns", "moderate"}
_LOW_ROB = {"low"}

# Canonical study-quality buckets.
BUCKET_LOW = "LOW_ROB"
BUCKET_MODERATE = "MODERATE_ROB"
BUCKET_HIGH = "HIGH_ROB"


def robins_i_overall(domains: dict[str, str]) -> str:
    """Closed-form ROBINS-I overall judgement (worst-domain rule).

    Overall = the worst domain rating; three or more ``Serious`` domains
    escalate to ``Critical`` (V4 rule). Returns one of
    ``Low | Moderate | Serious | Critical``.
    """
    missing = [d for d in ROBINS_I_DOMAINS if d not in domains]
    if missing:
        raise ValueError(f"ROBINS-I requires all 7 domains. Missing: {missing}")

    ranks = []
    for d in ROBINS_I_DOMAINS:
        key = str(domains[d]).strip().lower()
        if key not in _ROBINS_RANK:
            raise ValueError(
                f"Invalid ROBINS-I judgement for {d!r}: {domains[d]!r} "
                f"(must be Low/Moderate/Serious/Critical)")
        ranks.append(_ROBINS_RANK[key])

    worst = max(ranks)
    if ranks.count(_ROBINS_RANK["serious"]) >= 3 and worst < _ROBINS_RANK["critical"]:
        worst = _ROBINS_RANK["critical"]
    return _ROBINS_LABEL[worst]


def quality_bucket(overall: str) -> str:
    """Map any RoB2 or ROBINS-I overall label to a canonical quality bucket.

    Case-insensitive; unknown labels fall back to MODERATE (conservative).
    """
    key = str(overall).strip().lower()
    if key in _HIGH_ROB:
        return BUCKET_HIGH
    if key in _MODERATE_ROB:
        return BUCKET_MODERATE
    if key in _LOW_ROB:
        return BUCKET_LOW
    return BUCKET_MODERATE


def bucket_studies(overalls: list[str]) -> dict:
    """Tally a list of per-study overall judgements into quality buckets."""
    counts = {BUCKET_LOW: 0, BUCKET_MODERATE: 0, BUCKET_HIGH: 0}
    for o in overalls:
        counts[quality_bucket(o)] += 1
    k = len(overalls)
    high = counts[BUCKET_HIGH]
    return {
        "k": k,
        "buckets": counts,
        "proportion_high": (high / k) if k else 0.0,
        "any_high": high > 0,
    }


def grade_rob_downgrade(proportion_high: float, any_high: bool = False) -> dict:
    """Map high-RoB proportion to a GRADE risk-of-bias downgrade (V4 rule).

    - proportion > 0.5           → downgrade 2 (very serious)
    - proportion ≥ 0.25 or any high → downgrade 1 (serious)
    - otherwise                  → no downgrade

    Note: the "any high → at least 1" clause is stricter than v3's current
    inline GRADE thresholds; wiring this in would change GRADE output, so it is
    provided as a tool, not auto-applied (see PORT_NOTES.md).
    """
    if proportion_high > 0.5:
        return {"downgrade": 2, "level": "very_serious"}
    if proportion_high >= 0.25 or any_high:
        return {"downgrade": 1, "level": "serious"}
    return {"downgrade": 0, "level": "no_concern"}
