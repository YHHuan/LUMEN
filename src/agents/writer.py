"""
Writer & Citation Guardian Agents — LUMEN v2
==============================================
Phase 6: Manuscript writing with citation-grounded references.
- Writer outputs [REF:keyword] markers instead of raw citations
- Citation Guardian resolves markers against a curated reference pool
"""

import json
import re
import logging
from typing import Optional, List

from src.agents.base_agent import BaseAgent
from src.utils.cache import TokenBudget

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Manuscript section writer with reference markers."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        super().__init__(role_name="writer", budget=budget)

    def write_section(self, section_name: str, context: dict) -> str:
        """Write a single manuscript section."""
        system_prompt = self._prompt_config.get("system_prompt", "")
        user_prompt = self._build_section_prompt(section_name, context)

        result = self.call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            cache_namespace=f"manuscript_{section_name}",
            description=f"Write {section_name}",
        )

        return result.get("content", "")

    def _build_section_prompt(self, section_name: str, context: dict) -> str:
        section_prompts = self._prompt_config.get("section_prompts", {})
        template = section_prompts.get(section_name, "")

        if template:
            return template.format(**context)

        return (
            f"Write the {section_name} section of a systematic review "
            f"and meta-analysis manuscript.\n\n"
            f"Context:\n{json.dumps(context, indent=2, default=str)}\n\n"
            f"CITATION INSTRUCTIONS:\n"
            f"- Do NOT generate specific citation numbers or author names from memory.\n"
            f"- Insert reference markers: [REF:keyword_phrase]\n"
            f"- Example: 'rTMS shows promise [REF:rTMS cognitive MCI meta-analysis]'\n"
            f"- Use [CITATION NEEDED] only if you cannot describe the claim.\n\n"
            f"Write in academic style following PRISMA 2020 guidelines."
        )


class CitationGuardianAgent(BaseAgent):
    """Grounded citation resolution: [REF:keyword] markers -> verified references."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        super().__init__(role_name="citation_guardian", budget=budget)

    def resolve_citations(self, manuscript_text: str,
                          reference_pool,
                          hybrid_verifier=None) -> tuple:
        """
        Resolve all [REF:keyword] markers in the manuscript.

        Args:
            hybrid_verifier: Optional HybridCitationVerifier for BM25+vector search

        Returns: (resolved_text, citation_log)
        """
        markers = re.findall(r"\[REF:(.*?)\]", manuscript_text)

        if not markers:
            return manuscript_text, []

        citation_log = []
        resolved_text = manuscript_text

        for marker_text in set(markers):
            # Use hybrid verifier (BM25+vector) if available, else vector-only
            if hybrid_verifier:
                candidates = hybrid_verifier.find_best_reference(
                    marker_text, top_k=3
                )
            else:
                candidates = reference_pool.find_references_for_claim(
                    marker_text, top_k=3
                )

            resolved = False
            for ref, similarity in candidates:
                if similarity < 0.3:
                    continue

                # LLM verification
                verification = self._verify_citation(
                    claim_context=_get_surrounding_sentence(
                        manuscript_text, marker_text
                    ),
                    reference_title=ref.get("title", ""),
                    reference_abstract=ref.get("abstract", ""),
                )

                if verification.get("supported"):
                    citation_str = f"[{ref.get('citation_number', '?')}]"
                    resolved_text = resolved_text.replace(
                        f"[REF:{marker_text}]", citation_str
                    )
                    citation_log.append({
                        "marker": marker_text,
                        "resolved_to": ref.get("citation", ""),
                        "citation_number": ref.get("citation_number"),
                        "similarity": round(similarity, 3),
                        "verified": True,
                    })
                    resolved = True
                    break

            if not resolved:
                resolved_text = resolved_text.replace(
                    f"[REF:{marker_text}]",
                    f"[CITATION NEEDED: {marker_text}]",
                )
                citation_log.append({
                    "marker": marker_text,
                    "resolved_to": None,
                    "verified": False,
                })

        logger.info(
            f"Citation resolution: {sum(1 for c in citation_log if c['verified'])}"
            f"/{len(citation_log)} resolved"
        )

        return resolved_text, citation_log

    def _verify_citation(self, claim_context: str, reference_title: str,
                         reference_abstract: str) -> dict:
        """LLM call: does this reference support this claim?"""
        prompt = (
            f"Does this reference support the claim in the sentence?\n\n"
            f"Sentence: \"{claim_context}\"\n"
            f"Reference title: \"{reference_title}\"\n"
            f"Reference abstract: \"{reference_abstract[:500]}\"\n\n"
            f"Answer ONLY with JSON: "
            f"{{\"supported\": true/false, \"reason\": \"brief explanation\"}}"
        )

        result = self.call_llm(
            prompt=prompt,
            expect_json=True,
            cache_namespace="citation_verify",
            description="Verify citation",
        )

        parsed = result.get("parsed")
        if parsed:
            return parsed
        return {"supported": False, "reason": "parse_error"}


def _get_surrounding_sentence(text: str, marker: str) -> str:
    """Get the sentence containing a [REF:marker] reference."""
    idx = text.find(f"[REF:{marker}]")
    if idx == -1:
        return ""

    # Find sentence boundaries
    start = max(0, idx - 200)
    end = min(len(text), idx + 200)
    context = text[start:end]

    # Try to find sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", context)
    for sent in sentences:
        if marker in sent:
            return sent

    return context


# ======================================================================
# Reference Pool
# ======================================================================

class ReferencePool:
    """Curated pool of citable references for manuscript writing."""

    def __init__(self, vector_model: str = "all-MiniLM-L6-v2"):
        self.references = []
        self._model_name = vector_model
        self._model = None
        self.index = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def load_from_extraction(self, extracted_data: list):
        """Load included studies as citable references."""
        for i, study in enumerate(extracted_data):
            self.references.append({
                "id": study.get("study_id", f"study_{i}"),
                "citation": study.get("canonical_citation", ""),
                "citation_number": i + 1,
                "title": study.get("title", ""),
                "abstract": study.get("abstract", ""),
                "year": study.get("year", ""),
                "source": "included_study",
            })

    def load_from_yaml(self, references_path: str):
        """Load curated methodology/guideline references from YAML."""
        import yaml
        from pathlib import Path

        path = Path(references_path)
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for ref in data.get("references", []):
            ref["source"] = "curated"
            self.references.append(ref)

    def build_index(self):
        """Build HNSW index over reference titles/abstracts."""
        import hnswlib
        import numpy as np

        if not self.references:
            return

        texts = [
            f"{r.get('title', '')}. {r.get('abstract', '')}"
            for r in self.references
        ]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        dim = embeddings.shape[1]
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(
            max_elements=len(texts), ef_construction=200, M=16
        )
        self.index.add_items(embeddings, list(range(len(texts))))
        self.index.set_ef(50)

    def find_references_for_claim(self, claim: str, top_k: int = 5) -> list:
        """Find the most relevant references for a given claim."""
        if not self.index or not self.references:
            return []

        query_emb = self.model.encode([claim], normalize_embeddings=True)
        k = min(top_k, len(self.references))
        labels, distances = self.index.knn_query(query_emb, k=k)

        return [
            (self.references[i], float(1 - distances[0][j]))
            for j, i in enumerate(labels[0])
        ]
