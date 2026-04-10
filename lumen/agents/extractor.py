"""
4-round IterResearch extractor.

Each round rebuilds workspace from distilled output — no raw PDF carried forward.
Fixes v2 audit #8: evidence span threshold raised from 0.3 to 0.7.

Dynamic max_tokens: output budget is estimated from the data size (number of
outcomes, PDF length) rather than a fixed tier default, preventing JSON
truncation on studies with many outcomes.
"""
from __future__ import annotations

import json
import structlog

from lumen.agents.base import BaseAgent, LumenParseError

logger = structlog.get_logger()

EVIDENCE_SPAN_THRESHOLD = 0.7  # v2 audit #8: raised from 0.3

# ── Dynamic token budget constants ──────────────────────────────
# Heuristic: base tokens + per-item tokens × count, clamped to [MIN, MAX].
_MIN_TOKENS = 2048
_MAX_TOKENS = 32768
_R1_BASE, _R1_PER_PAGE = 2048, 80        # skeleton: grows with PDF pages
_R2_BASE, _R2_PER_OUTCOME = 1024, 600    # extraction: ~600 tok per outcome row
_R3_BASE, _R3_PER_OUTCOME = 1024, 200    # cross-check: lighter
_R4_BASE, _R4_PER_FIELD = 512, 250       # evidence spans: per extracted field


def _estimate_pages(pdf_text: str) -> int:
    """Count page markers or estimate from char count."""
    n = pdf_text.count("<!-- PAGE")
    if n > 0:
        return n
    return max(1, len(pdf_text) // 3500)  # ~3.5k chars/page


def _estimate_tokens(base: int, per_item: int, n_items: int) -> int:
    return max(_MIN_TOKENS, min(_MAX_TOKENS, base + per_item * n_items))


_MAX_EXTRACT_RETRIES = 2  # total attempts per study (1 original + 1 retry)


class ExtractorAgent(BaseAgent):
    tier = "smart"
    agent_name = "extractor"
    prompt_file = "extractor_round1.yaml"  # default; rounds use specific prompts

    @staticmethod
    def _prepare_pdf(pdf_content: str) -> str:
        """Strip reference section from PDF to save tokens."""
        from lumen.tools.pdf.reader import _strip_reference_section
        return _strip_reference_section(pdf_content)

    def extract(self, study: dict, pdf_content: str, pico: dict) -> dict:
        """
        Full 4-round IterResearch extraction with automatic retry.

        Round 1: Study skeleton (design, arms, outcomes)
        Round 2: Outcome-specific numerical extraction
        Round 3: Cross-check (no PDF, just data consistency)
        Round 4: Evidence span binding (value → PDF location)

        On LumenParseError, retries the entire 4-round sequence once
        (parse failures are often transient LLM formatting issues).
        """
        study_id = study.get("study_id", study.get("id", "unknown"))
        pdf_content = self._prepare_pdf(pdf_content)

        last_err: Exception | None = None
        for attempt in range(1, _MAX_EXTRACT_RETRIES + 1):
            try:
                return self._extract_inner(study_id, pdf_content, pico)
            except (LumenParseError, TypeError) as e:
                last_err = e
                if attempt < _MAX_EXTRACT_RETRIES:
                    logger.warning("extraction_retry",
                                   study_id=study_id, attempt=attempt,
                                   error=str(e)[:200])
                    continue
        # All attempts failed
        raise last_err  # type: ignore[misc]

    def _extract_inner(self, study_id: str, pdf_content: str,
                       pico: dict) -> dict:
        """Single attempt of the 4-round extraction."""
        # Round 1: Skeleton
        skeleton = self._round1_skeleton(pdf_content, pico)
        # Defensive: LLM may return None for outcome lists
        for key in ("primary_outcomes", "secondary_outcomes"):
            if not isinstance(skeleton.get(key), list):
                skeleton[key] = []
        logger.info("extraction_round1_done", study_id=study_id,
                     outcomes=len(skeleton.get("primary_outcomes", [])))

        # Round 2: Outcome extraction (uses skeleton + relevant PDF sections)
        extractions = self._round2_extract(skeleton, pdf_content, pico)
        logger.info("extraction_round2_done", study_id=study_id,
                     n_extractions=len(extractions))

        # Round 3: Cross-check (NO raw PDF — only skeleton + extracted data)
        crosscheck = self._round3_crosscheck(skeleton, extractions)
        logger.info("extraction_round3_done", study_id=study_id,
                     checks_passed=crosscheck.get("checks_passed", False),
                     n_issues=len(crosscheck.get("issues", [])))

        # Round 4: Evidence span binding
        spans = self._round4_bind_spans(extractions, pdf_content)
        low_confidence_spans = [
            s for s in spans
            if s.get("match_confidence", 0) < EVIDENCE_SPAN_THRESHOLD
        ]
        logger.info("extraction_round4_done", study_id=study_id,
                     n_spans=len(spans),
                     n_low_confidence=len(low_confidence_spans))

        return {
            "study_id": study_id,
            "skeleton": skeleton,
            "extractions": extractions,
            "crosscheck": crosscheck,
            "evidence_spans": spans,
            "low_confidence_spans": low_confidence_spans,
            "rounds_completed": 4,
        }

    def _round1_skeleton(self, pdf_content: str, pico: dict) -> dict:
        """Round 1: Extract study skeleton from full PDF."""
        user_content = (
            f"## PICO\n{json.dumps(pico, indent=2)}\n\n"
            f"## PDF Content\n{pdf_content}\n"
        )
        messages = self._build_messages(
            user_content,
            system_override=self._load_round_prompt("extractor_round1.yaml"),
        )
        budget = _estimate_tokens(_R1_BASE, _R1_PER_PAGE,
                                  _estimate_pages(pdf_content))
        response = self._call_llm(
            messages, response_format={"type": "json_object"},
            phase="extraction_round1", max_tokens=budget,
        )
        return self._parse_json(response, retry_messages=messages,
                                phase="extraction_round1")

    def _round2_extract(self, skeleton: dict, pdf_content: str,
                        pico: dict) -> list[dict]:
        """Round 2: Extract numerical data for each outcome."""
        outcomes = (
            skeleton.get("primary_outcomes", [])
            + skeleton.get("secondary_outcomes", [])
        )
        user_content = (
            f"## PICO\n{json.dumps(pico, indent=2)}\n\n"
            f"## Study Skeleton\n{json.dumps(skeleton, indent=2)}\n\n"
            f"## Outcomes to Extract\n{json.dumps(outcomes)}\n\n"
            f"## Relevant PDF Sections\n{pdf_content}\n"
        )
        messages = self._build_messages(
            user_content,
            system_override=self._load_round_prompt("extractor_round2.yaml"),
        )
        budget = _estimate_tokens(_R2_BASE, _R2_PER_OUTCOME, len(outcomes))
        response = self._call_llm(
            messages, response_format={"type": "json_object"},
            phase="extraction_round2", max_tokens=budget,
        )
        result = self._parse_json(response, retry_messages=messages,
                                  phase="extraction_round2")
        # Handle both list and dict-wrapped-list responses
        if isinstance(result, dict):
            return result.get("extractions", result.get("outcomes", [result]))
        return result if isinstance(result, list) else [result]

    def _round3_crosscheck(self, skeleton: dict,
                           extractions: list[dict]) -> dict:
        """Round 3: Cross-check extracted data (NO raw PDF)."""
        user_content = (
            f"## Study Skeleton\n{json.dumps(skeleton, indent=2)}\n\n"
            f"## Extracted Data\n{json.dumps(extractions, indent=2)}\n"
        )
        messages = self._build_messages(
            user_content,
            system_override=self._load_round_prompt("extractor_round3.yaml"),
        )
        budget = _estimate_tokens(_R3_BASE, _R3_PER_OUTCOME, len(extractions))
        response = self._call_llm(
            messages, response_format={"type": "json_object"},
            phase="extraction_round3", max_tokens=budget,
        )
        return self._parse_json(response, retry_messages=messages,
                                phase="extraction_round3")

    def _round4_bind_spans(self, extractions: list[dict],
                           pdf_content: str) -> list[dict]:
        """Round 4: Bind extracted values to PDF text spans."""
        # Count total fields to bind: each extraction has ~6-8 numeric fields
        n_fields = sum(
            sum(1 for k, v in ex.get("arm1", {}).items() if v is not None)
            + sum(1 for k, v in ex.get("arm2", {}).items() if v is not None)
            for ex in extractions
        ) if extractions else 0
        user_content = (
            f"## Extracted Data\n{json.dumps(extractions, indent=2)}\n\n"
            f"## PDF Content\n{pdf_content}\n"
        )
        messages = self._build_messages(
            user_content,
            system_override=self._load_round_prompt("extractor_round4.yaml"),
        )
        budget = _estimate_tokens(_R4_BASE, _R4_PER_FIELD, max(n_fields, len(extractions)))
        response = self._call_llm(
            messages, response_format={"type": "json_object"},
            phase="extraction_round4", max_tokens=budget,
        )
        result = self._parse_json(response, retry_messages=messages,
                                  phase="extraction_round4")
        if isinstance(result, dict):
            return result.get("spans", result.get("bindings", [result]))
        return result if isinstance(result, list) else [result]

    def _load_round_prompt(self, filename: str) -> str:
        """Load a round-specific prompt."""
        from pathlib import Path
        import yaml as _yaml
        path = Path(__file__).resolve().parent.parent.parent / "prompts" / filename
        if not path.exists():
            return ""
        with open(path, "r", encoding="utf-8") as fh:
            data = _yaml.safe_load(fh)
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return data.get("system_prompt", data.get("prompt", ""))
        return ""
