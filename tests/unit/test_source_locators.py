"""Extraction source-locator tests (stabilize P2.4)."""
import json
from unittest.mock import MagicMock

from lumen.agents.extractor import ExtractorAgent, attach_source_locators


EXTRACTIONS = [
    {"outcome_name": "systolic_bp", "arm1": {"mean": 125.3}, "arm2": {"mean": 132.5}},
    {"outcome_name": "diastolic_bp", "arm1": {"mean": 78.2}, "arm2": {"mean": 82.1}},
]
SPANS = [
    {"outcome_name": "systolic_bp", "field": "arm1.mean", "value": 125.3,
     "pdf_page": 5, "pdf_text_span": "mean SBP was 125.3 mmHg", "match_confidence": 0.95},
    {"outcome_name": "systolic_bp", "field": "arm2.mean", "value": 132.5,
     "pdf_page": 5, "pdf_text_span": "control: 132.5 mmHg", "match_confidence": 0.90},
    {"outcome_name": "diastolic_bp", "field": "arm1.mean", "value": 78.2,
     "pdf_page": 6, "pdf_text_span": "DBP 78.2", "match_confidence": 0.85},
]


class TestAttachSourceLocators:
    def test_locator_attached_to_each_value(self):
        exts = json.loads(json.dumps(EXTRACTIONS))  # deep copy
        attach_source_locators(exts, SPANS)
        loc = exts[0]["source_locators"]["arm1.mean"]
        assert loc["page"] == 5
        assert "125.3" in loc["quote"]
        assert loc["match_confidence"] == 0.95

    def test_all_rows_get_skeleton_even_without_spans(self):
        exts = json.loads(json.dumps(EXTRACTIONS))
        attach_source_locators(exts, [])
        assert exts[0]["source_locators"] == {}
        assert "source_locators" in exts[1]

    def test_matched_by_outcome_name(self):
        exts = json.loads(json.dumps(EXTRACTIONS))
        attach_source_locators(exts, SPANS)
        # diastolic only had arm1 bound
        assert set(exts[1]["source_locators"].keys()) == {"arm1.mean"}
        assert exts[1]["source_locators"]["arm1.mean"]["page"] == 6


def _make_extractor(responses):
    router = MagicMock()
    router.call.side_effect = [
        (json.dumps(r), {"model": "t", "input_tokens": 10, "output_tokens": 5,
                         "cost": 0.001, "latency_ms": 1})
        for r in responses
    ]
    return ExtractorAgent(router=router, cost_tracker=MagicMock(), config={})


def test_extract_end_to_end_populates_locators():
    """A full 4-round extraction leaves each value with an inline locator."""
    skeleton = {"design": "RCT", "primary_outcomes": ["systolic_bp"],
                "secondary_outcomes": []}
    agent = _make_extractor([skeleton, EXTRACTIONS, {"checks_passed": True, "issues": []}, SPANS])
    result = agent.extract({"study_id": "S1"}, "PDF text", {"population": "adults"})
    sbp = next(e for e in result["extractions"] if e["outcome_name"] == "systolic_bp")
    assert sbp["source_locators"]["arm1.mean"]["page"] == 5
