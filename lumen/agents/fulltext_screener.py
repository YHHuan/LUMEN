"""
Fulltext screener agent.

Fixes v2 audit #15: PDF content truncated at sentence/paragraph boundary,
not byte boundary.
"""
from __future__ import annotations

import json
import re
import structlog

from lumen.agents.base import BaseAgent, LumenParseError

logger = structlog.get_logger()

# Max tokens for PDF content sent to LLM
MAX_PDF_TOKENS_ESTIMATE = 6000  # ~24K chars assuming ~4 chars/token


class FulltextScreenerAgent(BaseAgent):
    tier = "smart"
    agent_name = "fulltext_screener"
    prompt_file = "fulltext_screener.yaml"

    def screen(self, study: dict, pdf_content: str,
               pico: dict, criteria: dict) -> dict:
        """
        Screen a study based on its full text.

        PDF content is pre-processed:
        - Truncated at paragraph boundary (not byte/mid-sentence)
        - Methods + Results sections prioritized
        """
        processed = self._prepare_pdf_content(pdf_content)

        user_content = (
            f"## PICO Criteria\n{json.dumps(pico, indent=2)}\n\n"
            f"## Screening Criteria\n{json.dumps(criteria, indent=2)}\n\n"
            f"## Study Metadata\n"
            f"Title: {study.get('title', 'N/A')}\n"
            f"Authors: {study.get('authors', 'N/A')}\n"
            f"Year: {study.get('year', 'N/A')}\n\n"
            f"## Full Text Content\n{processed}\n"
        )
        messages = self._build_messages(user_content)

        try:
            response = self._call_llm(
                messages,
                response_format={"type": "json_object"},
                phase="fulltext_screening",
            )
            result = self._parse_json(response, retry_messages=messages,
                                      phase="fulltext_screening")
        except LumenParseError:
            logger.error("fulltext_screening_parse_failure",
                         study_id=study.get("study_id", "unknown"))
            return {
                "decision": "include",
                "confidence": 0,
                "reasoning": "Parse failure — included conservatively",
                "key_sections_reviewed": [],
                "exclusion_reason": None,
                "study_id": study.get("study_id", study.get("id", "unknown")),
                "parse_error": True,
            }

        result = self._validate_result(result)
        result["study_id"] = study.get("study_id", study.get("id", "unknown"))
        return result

    @staticmethod
    def _prepare_pdf_content(text: str, max_chars: int = MAX_PDF_TOKENS_ESTIMATE * 4) -> str:
        """
        Prepare PDF content for LLM input.

        Fixes v2 audit #15: truncate at paragraph boundary, not byte boundary.
        Prioritizes Methods and Results sections.
        """
        if len(text) <= max_chars:
            return text

        # Try to extract Methods and Results sections first
        sections = _extract_priority_sections(text)
        if sections and len(sections) <= max_chars:
            return sections

        # Fallback: truncate at paragraph boundary
        return _truncate_at_paragraph(text, max_chars)

    @staticmethod
    def _validate_result(result: dict) -> dict:
        decision = result.get("decision", "").lower().strip()
        if decision not in ("include", "exclude"):
            decision = "include"
            result["confidence"] = 0

        result["decision"] = decision
        result["confidence"] = max(0, min(100, int(result.get("confidence", 50))))
        result.setdefault("reasoning", "")
        result.setdefault("key_sections_reviewed", [])
        result.setdefault("exclusion_reason", None)
        return result


def _extract_priority_sections(text: str) -> str | None:
    """Extract Methods and Results sections if identifiable."""
    # Common section headings
    patterns = [
        r"(?i)(methods?\b.*?)(?=\n\s*(?:results?|discussion|references)\b)",
        r"(?i)(results?\b.*?)(?=\n\s*(?:discussion|conclusion|references)\b)",
    ]
    sections = []
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections.append(match.group(1).strip())

    if not sections:
        return None

    return "\n\n---\n\n".join(sections)


def _truncate_at_paragraph(text: str, max_chars: int) -> str:
    """Truncate text at the last paragraph boundary before max_chars."""
    if len(text) <= max_chars:
        return text

    # Find the last double-newline (paragraph break) before the limit
    truncated = text[:max_chars]
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars * 0.5:  # at least halfway through
        return truncated[:last_para].rstrip() + "\n\n[... truncated at paragraph boundary]"

    # Fallback: last sentence boundary
    last_sentence = max(
        truncated.rfind(". "),
        truncated.rfind(".\n"),
        truncated.rfind("? "),
        truncated.rfind("! "),
    )
    if last_sentence > max_chars * 0.5:
        return truncated[:last_sentence + 1].rstrip() + "\n\n[... truncated at sentence boundary]"

    # Last resort: hard cut
    return truncated.rstrip() + "\n\n[... truncated]"
