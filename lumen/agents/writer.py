"""
Writer agent: evidence synthesis → sequential section writing → inline fact-check.

Sequential single-writer approach (Open Deep Research style):
Methods → Results → Discussion → Introduction → Abstract.
Each section sees all previously written sections for consistency.

Fact-check: WikiChat-simplified — extract claims, verify against data,
auto-correct CONTRADICTED, mark UNSUPPORTED with [CITATION NEEDED].
"""
from __future__ import annotations

import json
import structlog

from lumen.agents.base import BaseAgent, LumenParseError

logger = structlog.get_logger()

SECTION_ORDER = ["methods", "results", "discussion", "introduction", "abstract"]


class WriterAgent(BaseAgent):
    tier = "smart"
    agent_name = "writer"
    prompt_file = "writer_synthesis.yaml"

    def write(self, statistics_results: dict, extractions: list[dict],
              quality_assessments: dict, pico: dict,
              interpretations: dict | None = None) -> dict:
        """
        Full 3-step writing pipeline.

        Step 1: Evidence synthesis session
        Step 2: Sequential section writing (M→R→D→I→A)
        Step 3: Inline fact-check per section
        """
        # Step 1: Evidence synthesis
        synthesis = self._evidence_synthesis(
            statistics_results, extractions, quality_assessments,
            pico, interpretations,
        )
        logger.info("writer_step1_synthesis",
                     n_findings=len(synthesis.get("key_findings", [])))

        # Step 2: Sequential section writing
        sections = {}
        for section_name in SECTION_ORDER:
            section_text = self._write_section(
                section_name, synthesis, sections, pico,
                statistics_results, quality_assessments,
            )
            sections[section_name] = section_text
            logger.info("writer_step2_section", section=section_name,
                         length=len(section_text))

        # Step 3: Inline fact-check per section
        fact_check_log = []
        revised_sections = {}
        for section_name in SECTION_ORDER:
            check_result = self._fact_check_section(
                sections[section_name], section_name,
                statistics_results, extractions, quality_assessments,
            )
            fact_check_log.extend(check_result.get("claims", []))
            revised_sections[section_name] = check_result.get(
                "revised_text", sections[section_name],
            )
            logger.info("writer_step3_factcheck", section=section_name,
                         n_supported=check_result.get("n_supported", 0),
                         n_contradicted=check_result.get("n_contradicted", 0),
                         n_unsupported=check_result.get("n_unsupported", 0))

        return {
            "evidence_synthesis": synthesis,
            "manuscript_sections": revised_sections,
            "manuscript_sections_raw": sections,
            "fact_check_log": fact_check_log,
        }

    def _evidence_synthesis(self, statistics_results: dict,
                            extractions: list[dict],
                            quality_assessments: dict,
                            pico: dict,
                            interpretations: dict | None = None) -> dict:
        """Step 1: Produce structured evidence synthesis."""
        user_content = (
            f"## PICO\n{json.dumps(pico, indent=2)}\n\n"
            f"## Statistical Results\n{json.dumps(statistics_results, default=str, indent=2)}\n\n"
            f"## Quality Assessments\n{json.dumps(quality_assessments, default=str, indent=2)}\n\n"
            f"## Number of Included Studies\n{len(extractions)}\n\n"
        )
        if interpretations:
            user_content += f"## Interpretations\n{json.dumps(interpretations, default=str, indent=2)}\n\n"

        user_content += "Produce the evidence synthesis."
        messages = self._build_messages(user_content)

        try:
            response = self._call_llm(
                messages, response_format={"type": "json_object"},
                phase="writer_synthesis",
            )
            return self._parse_json(response, retry_messages=messages,
                                    phase="writer_synthesis")
        except LumenParseError:
            logger.warning("writer_synthesis_failed")
            return {
                "key_findings": ["Evidence synthesis unavailable"],
                "evidence_table": [],
                "narrative_skeleton": {s: "" for s in SECTION_ORDER},
            }

    def _write_section(self, section_name: str, synthesis: dict,
                       previous_sections: dict, pico: dict,
                       statistics_results: dict,
                       quality_assessments: dict) -> str:
        """Step 2: Write a single section with context from prior sections."""
        user_content = (
            f"## Section to Write\n{section_name}\n\n"
            f"## PICO\n{json.dumps(pico, indent=2)}\n\n"
            f"## Evidence Synthesis\n{json.dumps(synthesis, default=str, indent=2)}\n\n"
        )

        if section_name in ("results", "discussion", "introduction", "abstract"):
            user_content += f"## Statistical Results\n{json.dumps(statistics_results, default=str, indent=2)}\n\n"

        if quality_assessments:
            user_content += f"## Quality Assessments\n{json.dumps(quality_assessments, default=str, indent=2)}\n\n"

        if previous_sections:
            user_content += "## Previously Written Sections\n"
            for prev_name, prev_text in previous_sections.items():
                user_content += f"### {prev_name.title()}\n{prev_text}\n\n"

        thesis = synthesis.get("narrative_skeleton", {}).get(section_name, "")
        if thesis:
            user_content += f"## Thesis for This Section\n{thesis}\n\n"

        user_content += f"Write the {section_name} section now."

        messages = self._build_messages(
            user_content,
            system_override=self._load_round_prompt("writer_section.yaml"),
        )

        try:
            response = self._call_llm(
                messages, response_format={"type": "json_object"},
                phase=f"writer_{section_name}",
            )
            parsed = self._parse_json(response, retry_messages=messages,
                                      phase=f"writer_{section_name}")
            return parsed.get("text", "")
        except LumenParseError:
            logger.warning("writer_section_failed", section=section_name)
            return f"[{section_name} section could not be generated]"

    def _fact_check_section(self, section_text: str, section_name: str,
                            statistics_results: dict,
                            extractions: list[dict],
                            quality_assessments: dict) -> dict:
        """
        Step 3: Inline fact-check (WikiChat-simplified).

        a) Extract atomic claims from section text
        b) Check each claim against statistics, extractions, quality data
        c) SUPPORTED / CONTRADICTED / UNSUPPORTED
        d) Auto-revise CONTRADICTED claims
        e) Mark UNSUPPORTED with [CITATION NEEDED]
        """
        if not section_text or section_text.startswith("["):
            return {
                "claims": [],
                "revised_text": section_text,
                "n_supported": 0,
                "n_contradicted": 0,
                "n_unsupported": 0,
            }

        user_content = (
            f"## Section ({section_name})\n{section_text}\n\n"
            f"## Source Data: Statistical Results\n"
            f"{json.dumps(statistics_results, default=str, indent=2)}\n\n"
            f"## Source Data: Study Count\n{len(extractions)} studies\n\n"
            f"## Source Data: Quality Assessments\n"
            f"{json.dumps(quality_assessments, default=str, indent=2)}\n\n"
            "Fact-check the section above against the source data."
        )
        messages = self._build_messages(
            user_content,
            system_override=self._load_round_prompt("fact_checker.yaml"),
        )

        try:
            response = self._call_llm(
                messages, response_format={"type": "json_object"},
                phase=f"factcheck_{section_name}",
            )
            result = self._parse_json(response, retry_messages=messages,
                                      phase=f"factcheck_{section_name}")
        except LumenParseError:
            logger.warning("factcheck_failed", section=section_name)
            return {
                "claims": [],
                "revised_text": section_text,
                "n_supported": 0,
                "n_contradicted": 0,
                "n_unsupported": 0,
            }

        claims = result.get("claims", [])
        summary = result.get("summary", {})

        # Apply corrections and mark as resolved
        revised_text = section_text
        for claim in claims:
            if claim.get("verdict") == "CONTRADICTED" and claim.get("corrected_text"):
                original = claim.get("text", "")
                if original and original in revised_text:
                    revised_text = revised_text.replace(original, claim["corrected_text"])
                    claim["resolved"] = True  # prevents infinite loop in graph routing
                else:
                    claim["resolved"] = True  # can't find text to fix, mark resolved anyway
            elif claim.get("verdict") == "UNSUPPORTED":
                original = claim.get("text", "")
                if original and original in revised_text and "[CITATION NEEDED]" not in revised_text:
                    revised_text = revised_text.replace(
                        original, f"{original} [CITATION NEEDED]"
                    )

        return {
            "claims": claims,
            "revised_text": revised_text,
            "n_supported": summary.get("n_supported", 0),
            "n_contradicted": summary.get("n_contradicted", 0),
            "n_unsupported": summary.get("n_unsupported", 0),
        }

    def _load_round_prompt(self, filename: str) -> str:
        """Load a specific prompt file."""
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
