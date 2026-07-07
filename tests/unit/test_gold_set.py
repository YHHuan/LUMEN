"""Gold-set decision tests (stabilize P1.8).

Small, hand-curated "is the inclusion decision correct?" fixtures for canonical
cases from the bacteremia treatment-duration SRMA domain. These exercise the
*deterministic* decision logic only — ``resolve_screening`` (union-hybrid) plus
the reason-code classifier — so they need NO LLM and NO API key. They act as a
regression net for the include/exclude verdict and its structured reason code.

Each fixture simulates what the two screeners *said* (their decision +
confidence + rationale); the test asserts the pipeline resolves to the correct
verdict and, for exclusions, the correct canonical reason code.
"""
import pytest

from lumen.agents.arbiter import resolve_screening
from lumen.agents.screening_signals import (
    annotate_screening_result,
    classify_exclusion_reason,
    ReasonCode,
)


# (name, screener1, screener2, expected_final, expected_tri, rationale, expected_code)
GOLD_CASES = [
    (
        "BALANCE main trial (7 vs 14 days) — primary RCT",
        {"decision": "include", "confidence": 95},
        {"decision": "include", "confidence": 92},
        "include", 1,
        "Randomized non-inferiority trial of antibiotic duration in bacteremia",
        None,
    ),
    (
        "Procalcitonin-guided discontinuation — biomarker strategy, not fixed duration",
        {"decision": "exclude", "confidence": 90},
        {"decision": "exclude", "confidence": 88},
        "exclude", -1,
        "Compares procalcitonin-guided antibiotic discontinuation, not a fixed duration",
        ReasonCode.BIOMARKER_GUIDED,
    ),
    (
        "Typhoid / enteric fever duration trial — out-of-scope infection",
        {"decision": "exclude", "confidence": 93},
        {"decision": "exclude", "confidence": 91},
        "exclude", -1,
        "Population is typhoid / enteric fever, not the bacteremia of interest",
        ReasonCode.WRONG_POPULATION,
    ),
    (
        "Post-hoc satellite paper of a prior trial — not a primary study",
        {"decision": "exclude", "confidence": 89},
        {"decision": "exclude", "confidence": 90},
        "exclude", -1,
        "Post-hoc secondary analysis of an already-included trial cohort",
        ReasonCode.NOT_PRIMARY_STUDY,
    ),
]


@pytest.mark.parametrize(
    "name,s1,s2,expected_final,expected_tri,rationale,expected_code",
    GOLD_CASES,
    ids=[c[0] for c in GOLD_CASES],
)
def test_gold_case_decision(name, s1, s2, expected_final, expected_tri,
                            rationale, expected_code):
    result = resolve_screening(s1, s2)
    assert result["final_decision"] == expected_final, name

    annotate_screening_result(result, auto_threshold=80, unclear_margin=5)
    # High-confidence gold cases should be decided automatically, not flagged.
    assert result["review_required"] is False, name
    assert result["tri_state"] == expected_tri, name

    if expected_code is not None:
        assert classify_exclusion_reason(rationale) == expected_code, name


def test_balance_trial_is_included():
    """The canonical positive control must be included."""
    result = resolve_screening(
        {"decision": "include", "confidence": 95},
        {"decision": "include", "confidence": 92},
    )
    assert result["final_decision"] == "include"


def test_biomarker_guided_reason_code():
    assert classify_exclusion_reason(
        "Antibiotic stopping guided by procalcitonin levels"
    ) == ReasonCode.BIOMARKER_GUIDED


def test_union_rescues_disagreement_toward_inclusion():
    """Sensitivity guard: if either screener includes, the study is included."""
    result = resolve_screening(
        {"decision": "exclude", "confidence": 85},
        {"decision": "include", "confidence": 55},
    )
    assert result["final_decision"] == "include"
    annotate_screening_result(result, disagreement_conf=60)
    # ...but the borderline include is flagged for a human spot-check.
    assert result["review_required"] is True
    assert ReasonCode.SCREENER_DISAGREEMENT in result["flags"]
