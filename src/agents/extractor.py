"""
Extractor Agent — LUMEN v2
=============================
Phase 4: PDF -> structured data extraction with:
- 3-pass self-consistency check
- Claim-grounded extraction (evidence spans)
- Vector-indexed retrieval for smart context
- gmft table pre-parsing
"""

import json
import logging
from typing import Optional, List

from src.agents.base_agent import BaseAgent
from src.utils.cache import TokenBudget

logger = logging.getLogger(__name__)


class ExtractorAgent(BaseAgent):
    """Data extraction agent with 3-pass self-consistency."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        super().__init__(role_name="extractor", budget=budget)

    def extract(self, context: str, extraction_schema: dict,
                study_id: str = "") -> dict:
        """Single extraction pass."""
        system_prompt = self._prompt_config.get("system_prompt", "")
        user_prompt = self._build_extraction_prompt(context, extraction_schema)

        result = self.call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            expect_json=True,
            cache_namespace=f"extraction_{study_id}",
            description=f"Extract {study_id}",
        )

        return result.get("parsed") or {}

    def extract_with_consistency(
        self,
        context: str,
        extraction_schema: dict,
        study_id: str = "",
        n_passes: int = 3,
    ) -> dict:
        """
        Run n-pass extraction with self-consistency check.
        If passes disagree, tiebreaker model resolves.
        """
        passes = []
        for i in range(n_passes):
            # Vary cache namespace per pass to get independent results
            result = self.extract(
                context, extraction_schema,
                study_id=f"{study_id}_pass{i + 1}",
            )
            passes.append(result)

        # Check consistency
        if n_passes == 1 or all(self._results_match(passes[0], p) for p in passes[1:]):
            final = passes[0]
            final["_consistency"] = {"method": "unanimous", "passes": n_passes}
            return final

        # Disagreement — use majority vote or tiebreaker
        final = self._resolve_inconsistency(passes, context, study_id)
        return final

    def _build_extraction_prompt(self, context: str, schema: dict) -> str:
        template = self._prompt_config.get("user_prompt_template", "")
        if template:
            return template.format(
                context=context,
                schema=json.dumps(schema, indent=2),
            )

        evidence_instruction = (
            "\n\nIMPORTANT: For EVERY numeric value you extract, "
            "you MUST provide the evidence_span — the exact text or table cell "
            "content from the source that contains this value.\n"
            "If you cannot find a value in the source, set the field to null "
            "and evidence_span to 'NOT FOUND IN SOURCE'.\n"
            "NEVER estimate, calculate, or infer values not explicitly stated.\n"
            "For table-sourced data, reference as 'Table X, Row Y: [exact content]'.\n"
            "For text-sourced data, quote the exact sentence (max 30 words)."
        )

        return (
            f"Extract structured data from this study.\n\n"
            f"=== STUDY CONTENT ===\n{context}\n\n"
            f"=== EXTRACTION SCHEMA ===\n{json.dumps(schema, indent=2)}\n"
            f"{evidence_instruction}\n\n"
            f"Return the extracted data as JSON matching the schema."
        )

    def _results_match(self, a: dict, b: dict, tolerance: float = 0.01) -> bool:
        """Check if two extraction results are consistent."""
        try:
            for key in a:
                if key.startswith("_"):
                    continue
                if key not in b:
                    return False
                va, vb = a[key], b[key]
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    if abs(va - vb) > tolerance:
                        return False
                elif isinstance(va, dict) and isinstance(vb, dict):
                    if not self._results_match(va, vb, tolerance):
                        return False
                elif va != vb:
                    # Allow minor string differences
                    if isinstance(va, str) and isinstance(vb, str):
                        if va.strip().lower() != vb.strip().lower():
                            return False
                    else:
                        return False
            return True
        except Exception:
            return False

    def _resolve_inconsistency(self, passes: list, context: str,
                               study_id: str) -> dict:
        """Resolve inconsistencies via majority vote + tiebreaker."""
        # Try majority vote on each field
        merged = {}
        disagreements = []

        # Flatten and compare field by field
        all_keys = set()
        for p in passes:
            all_keys.update(k for k in p.keys() if not k.startswith("_"))

        for key in all_keys:
            values = [p.get(key) for p in passes if key in p]
            if len(values) <= 1:
                merged[key] = values[0] if values else None
                continue

            # Check if majority agrees
            from collections import Counter
            value_strs = [json.dumps(v, sort_keys=True) for v in values]
            counts = Counter(value_strs)
            most_common_str, count = counts.most_common(1)[0]

            if count > len(passes) / 2:
                merged[key] = json.loads(most_common_str)
            else:
                disagreements.append(key)
                merged[key] = values[0]  # Placeholder

        if disagreements:
            # Use tiebreaker for unresolved fields
            try:
                tiebreaker_result = self._call_tiebreaker(
                    passes, disagreements, context, study_id
                )
                if tiebreaker_result:
                    for key in disagreements:
                        if key in tiebreaker_result:
                            merged[key] = tiebreaker_result[key]
            except Exception as e:
                logger.warning(f"Tiebreaker failed for {study_id}: {e}")

        merged["_consistency"] = {
            "method": "majority_vote" if not disagreements else "tiebreaker",
            "passes": len(passes),
            "disagreement_fields": disagreements,
        }

        return merged

    def _call_tiebreaker(self, passes: list, disagreement_fields: list,
                         context: str, study_id: str) -> Optional[dict]:
        """Call tiebreaker model (GPT-5.4) for unresolved disagreements."""
        tiebreaker = TiebreakerAgent()

        fields_info = {}
        for field in disagreement_fields:
            field_values = []
            for i, p in enumerate(passes):
                val = p.get(field)
                evidence = None
                if isinstance(val, dict):
                    evidence = val.get("evidence_span", "")
                field_values.append({
                    "pass": i + 1,
                    "value": val,
                    "evidence": evidence,
                })
            fields_info[field] = field_values

        prompt = (
            f"Three independent extraction passes produced different values "
            f"for these fields:\n\n"
            f"{json.dumps(fields_info, indent=2)}\n\n"
            f"Source context (relevant excerpt):\n"
            f"{context[:3000]}\n\n"
            f"For each field, determine which pass is correct. "
            f"Return JSON with the correct value for each field."
        )

        result = tiebreaker.call_llm(
            prompt=prompt,
            expect_json=True,
            cache_namespace=f"tiebreaker_{study_id}",
            description=f"Tiebreak {study_id}",
        )

        return result.get("parsed")


class TiebreakerAgent(BaseAgent):
    """Tiebreaker agent for resolving extraction disagreements."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        super().__init__(role_name="extractor_tiebreaker", budget=budget)


# ======================================================================
# Evidence Span Validation
# ======================================================================

def validate_evidence_spans(extraction: dict, segments: list) -> dict:
    """
    Cross-check: does the evidence_span actually exist in the source?
    """
    from difflib import SequenceMatcher

    validation = {}

    def _check_field(path: str, value_info):
        if not isinstance(value_info, dict):
            return

        span = value_info.get("evidence_span", "")
        page = value_info.get("evidence_page")

        if not span or span == "NOT FOUND IN SOURCE":
            validation[path] = {"status": "not_reported", "verified": True}
            return

        # Find matching segments
        if page:
            matching = [s for s in segments if s.page_number == page]
        else:
            matching = segments

        full_text = " ".join(s.content for s in matching)

        ratio = SequenceMatcher(None, span.lower()[:200], full_text.lower()[:2000]).ratio()

        # Also check for the numeric value itself
        value_str = str(value_info.get("mean") or value_info.get("value", ""))
        value_present = value_str and value_str in full_text

        if ratio > 0.3 or value_present:
            validation[path] = {"status": "verified", "match_ratio": round(ratio, 3)}
        else:
            validation[path] = {"status": "evidence_not_found", "match_ratio": round(ratio, 3)}

    def _walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith("_"):
                    continue
                new_prefix = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict) and "evidence_span" in v:
                    _check_field(new_prefix, v)
                else:
                    _walk(v, new_prefix)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{prefix}[{i}]")

    _walk(extraction)
    return validation
