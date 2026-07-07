"""Tests for the HITL screening signal layer (stabilize P2.1 / P2.2)."""

from lumen.agents.screening_signals import (
    tri_state,
    ReasonCode,
    ALL_REASON_CODES,
    is_valid_reason_code,
    classify_exclusion_reason,
    is_borderline,
    annotate_screening_result,
    INCLUDE, EXCLUDE, UNCLEAR, HUMAN_REVIEW,
)


class TestTriState:
    def test_include_is_plus_one(self):
        assert tri_state("include") == 1

    def test_exclude_is_minus_one(self):
        assert tri_state("exclude") == -1

    def test_unclear_is_zero(self):
        assert tri_state("unclear") == 0

    def test_human_review_is_zero(self):
        assert tri_state("human_review") == 0

    def test_unknown_defaults_to_zero(self):
        assert tri_state("maybe") == 0
        assert tri_state("") == 0
        assert tri_state(None) == 0

    def test_case_insensitive(self):
        assert tri_state("INCLUDE") == 1
        assert tri_state(" Exclude ") == -1


class TestReasonCodes:
    def test_all_codes_are_strings(self):
        assert ReasonCode.BIOMARKER_GUIDED in ALL_REASON_CODES
        assert all(isinstance(c, str) for c in ALL_REASON_CODES)

    def test_validity_check(self):
        assert is_valid_reason_code("NOT_PRIMARY_STUDY")
        assert not is_valid_reason_code("NOT_A_REAL_CODE")


class TestClassifyExclusionReason:
    def test_biomarker_guided(self):
        assert classify_exclusion_reason(
            "Procalcitonin-guided antibiotic discontinuation"
        ) == ReasonCode.BIOMARKER_GUIDED
        assert classify_exclusion_reason(
            "CRP-guided stopping rule"
        ) == ReasonCode.BIOMARKER_GUIDED

    def test_not_primary_study(self):
        assert classify_exclusion_reason(
            "Post-hoc secondary analysis of a prior trial"
        ) == ReasonCode.NOT_PRIMARY_STUDY
        assert classify_exclusion_reason(
            "This is a narrative review article"
        ) == ReasonCode.NOT_PRIMARY_STUDY

    def test_wrong_population(self):
        assert classify_exclusion_reason(
            "Typhoid / enteric fever cohort"
        ) == ReasonCode.WRONG_POPULATION

    def test_wrong_study_design(self):
        assert classify_exclusion_reason(
            "Observational cohort study, not an RCT"
        ) == ReasonCode.WRONG_STUDY_DESIGN

    def test_mixed_iv_to_oral(self):
        assert classify_exclusion_reason(
            "Evaluates IV-to-oral switch therapy"
        ) == ReasonCode.MIXED_IV_TO_ORAL

    def test_insufficient_data(self):
        assert classify_exclusion_reason(
            "Outcomes not reported; no extractable data"
        ) == ReasonCode.INSUFFICIENT_DATA

    def test_biomarker_wins_over_design(self):
        """A biomarker-guided RCT is excluded for the biomarker reason."""
        assert classify_exclusion_reason(
            "Randomized trial of procalcitonin-guided therapy"
        ) == ReasonCode.BIOMARKER_GUIDED

    def test_default_when_no_match(self):
        assert classify_exclusion_reason("something unclassifiable") == ReasonCode.UNSPECIFIED
        assert classify_exclusion_reason("") == ReasonCode.UNSPECIFIED
        assert classify_exclusion_reason(None) == ReasonCode.UNSPECIFIED


class TestIsBorderline:
    def test_within_margin_above_threshold(self):
        assert is_borderline(82, threshold=80, margin=5)

    def test_at_threshold(self):
        assert is_borderline(80, threshold=80, margin=5)

    def test_outside_margin(self):
        assert not is_borderline(90, threshold=80, margin=5)

    def test_below_threshold_not_borderline_band(self):
        # below threshold is handled separately; the band is [t, t+margin)
        assert not is_borderline(70, threshold=80, margin=5)

    def test_unparseable_is_uncertain(self):
        assert is_borderline("n/a", threshold=80, margin=5)


class TestAnnotate:
    def test_confident_exclude_not_flagged(self):
        r = {"final_decision": "exclude", "method": "dual_exclude_high_confidence",
             "confidence": 90}
        annotate_screening_result(r, auto_threshold=80, unclear_margin=5)
        assert r["tri_state"] == -1
        assert r["screening_state"] == "exclude"
        assert r["review_required"] is False

    def test_borderline_exclude_becomes_unclear(self):
        r = {"final_decision": "exclude", "method": "dual_exclude_high_confidence",
             "confidence": 81}
        annotate_screening_result(r, auto_threshold=80, unclear_margin=5)
        assert r["tri_state"] == 0
        assert r["screening_state"] == UNCLEAR
        assert r["review_required"] is True
        assert ReasonCode.BORDERLINE_CONFIDENCE in r["flags"]

    def test_human_review_is_unclear(self):
        r = {"final_decision": "human_review", "method": "arbiter", "confidence": 40}
        annotate_screening_result(r)
        assert r["tri_state"] == 0
        assert r["screening_state"] == UNCLEAR
        assert r["review_required"] is True
        assert ReasonCode.ARBITER_LOW_CONFIDENCE in r["flags"]

    def test_confident_include_not_flagged(self):
        r = {"final_decision": "include", "method": "dual_include", "confidence": 90}
        annotate_screening_result(r)
        assert r["tri_state"] == 1
        assert r["screening_state"] == "include"
        assert r["review_required"] is False

    def test_disagreement_include_is_flagged_but_still_included(self):
        r = {"final_decision": "include", "method": "union_include", "confidence": 45}
        annotate_screening_result(r, disagreement_conf=60)
        assert r["tri_state"] == 1  # still an include (union sensitivity)
        assert r["screening_state"] == "include"
        assert r["review_required"] is True
        assert ReasonCode.SCREENER_DISAGREEMENT in r["flags"]

    def test_parse_error_forces_review(self):
        r = {"final_decision": "include", "method": "dual_include",
             "confidence": 0, "parse_error": True}
        annotate_screening_result(r)
        assert r["review_required"] is True
        assert ReasonCode.PARSE_FAILURE in r["flags"]
