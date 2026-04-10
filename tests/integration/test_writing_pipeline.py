"""Integration tests for writing pipeline."""

import json
import pytest
from unittest.mock import MagicMock

from lumen.agents.writer import WriterAgent, SECTION_ORDER


def _mock_router(*responses):
    router = MagicMock()
    side_effects = []
    for r in responses:
        side_effects.append((json.dumps(r), {
            "model": "test", "input_tokens": 500,
            "output_tokens": 300, "cost": 0.008, "latency_ms": 600,
        }))
    router.call.side_effect = side_effects
    return router


PICO = {"population": "adults with hypertension", "intervention": "exercise"}

STATS = {
    "systolic_BP": {
        "k": 10,
        "meta": {
            "pooled_effect": -0.55,
            "ci_lower": -0.84,
            "ci_upper": -0.26,
            "i2": 35.0,
            "tau2": 0.02,
        },
    },
}

EXTRACTIONS = [
    {"study_id": f"S{i}", "extractions": [
        {"outcome_name": "SBP", "canonical_outcome": "systolic_BP"}
    ]}
    for i in range(10)
]

QUALITY = {
    "grade": {"systolic_BP": {"grade": "moderate", "certainty": 3}},
}


class TestSequentialWritingOrder:
    def test_methods_written_before_results(self):
        """Methods section is written first, Results sees Methods as context."""
        call_order = []

        synthesis = {
            "key_findings": ["Exercise reduces SBP"],
            "evidence_table": [],
            "narrative_skeleton": {s: f"Thesis for {s}" for s in SECTION_ORDER},
        }

        section_texts = {}
        all_responses = [synthesis]
        for section in SECTION_ORDER:
            text = f"This is the {section} section. k=10 studies."
            section_texts[section] = text
            all_responses.append({
                "section_name": section,
                "text": text,
                "citations_used": [],
                "statistics_referenced": [],
            })

        # Fact-check responses
        for section in SECTION_ORDER:
            all_responses.append({
                "claims": [{"text": "k=10 studies", "verdict": "SUPPORTED",
                            "evidence": "Confirmed", "corrected_text": None}],
                "summary": {"n_supported": 1, "n_contradicted": 0, "n_unsupported": 0},
            })

        router = _mock_router(*all_responses)

        # Track what user_content is passed for each section
        original_call = router.call

        def tracking_call(*args, **kwargs):
            messages = kwargs.get("messages") or (args[0] if args else [])
            # Find the user message
            for msg in (messages if isinstance(messages, list) else []):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if "## Section to Write" in content:
                        call_order.append(content)
            return original_call(*args, **kwargs)

        # Can't easily intercept mock side_effect, so verify order via output
        agent = WriterAgent(router=router, cost_tracker=MagicMock(), config={})
        result = agent.write(STATS, EXTRACTIONS, QUALITY, PICO)

        # All sections present in order
        sections = result["manuscript_sections"]
        assert list(sections.keys()) == SECTION_ORDER


class TestFactCheckCatchesWrongNumber:
    def test_wrong_study_count_is_contradicted(self):
        """Manuscript says 'k=12' but data has 10 studies → CONTRADICTED."""
        synthesis = {
            "key_findings": ["Exercise reduces SBP"],
            "evidence_table": [],
            "narrative_skeleton": {s: "" for s in SECTION_ORDER},
        }

        all_responses = [synthesis]
        # Sections — results section has wrong number
        for section in SECTION_ORDER:
            if section == "results":
                text = "A total of 12 studies were included in the meta-analysis."
            else:
                text = f"[{section}]"
            all_responses.append({
                "section_name": section,
                "text": text,
                "citations_used": [],
                "statistics_referenced": [],
            })

        # Only results section gets fact-checked (others start with "[")
        all_responses.append({
            "claims": [
                {
                    "text": "12 studies were included",
                    "verdict": "CONTRADICTED",
                    "evidence": "Data shows 10 studies, not 12",
                    "corrected_text": "10 studies were included",
                },
            ],
            "summary": {"n_supported": 0, "n_contradicted": 1, "n_unsupported": 0},
        })

        router = _mock_router(*all_responses)
        agent = WriterAgent(router=router, cost_tracker=MagicMock(), config={})
        result = agent.write(STATS, EXTRACTIONS, QUALITY, PICO)

        # Check fact_check_log has the contradiction
        contradicted = [c for c in result["fact_check_log"]
                        if c.get("verdict") == "CONTRADICTED"]
        assert len(contradicted) >= 1


class TestFactCheckAutoRevises:
    def test_contradicted_claim_gets_corrected_in_output(self):
        """CONTRADICTED claim gets corrected in revised text."""
        synthesis = {
            "key_findings": ["Finding"],
            "evidence_table": [],
            "narrative_skeleton": {s: "" for s in SECTION_ORDER},
        }

        all_responses = [synthesis]
        for section in SECTION_ORDER:
            if section == "results":
                text = "The pooled effect was SMD = -0.99 (95% CI -1.5 to -0.5)."
            else:
                text = f"[{section}]"
            all_responses.append({
                "section_name": section,
                "text": text,
                "citations_used": [],
                "statistics_referenced": [],
            })

        # Fact-check for results section
        all_responses.append({
            "claims": [
                {
                    "text": "SMD = -0.99",
                    "verdict": "CONTRADICTED",
                    "evidence": "Actual pooled SMD = -0.55",
                    "corrected_text": "SMD = -0.55",
                },
            ],
            "summary": {"n_supported": 0, "n_contradicted": 1, "n_unsupported": 0},
        })

        router = _mock_router(*all_responses)
        agent = WriterAgent(router=router, cost_tracker=MagicMock(), config={})
        result = agent.write(STATS, EXTRACTIONS, QUALITY, PICO)

        # The revised results section should have the corrected number
        results_text = result["manuscript_sections"]["results"]
        assert "SMD = -0.55" in results_text
        assert "SMD = -0.99" not in results_text

        # Raw version should still have the original
        results_raw = result["manuscript_sections_raw"]["results"]
        assert "SMD = -0.99" in results_raw
