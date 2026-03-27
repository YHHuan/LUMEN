"""
Vector-Indexed Retrieval — LUMEN v2
=====================================
HNSW index over document segments for retrieval-augmented extraction.
Reduces Phase 4 input tokens by ~40-55% by retrieving only relevant chunks.
"""

import logging
from typing import List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DocumentVectorIndex:
    """HNSW index over document segments for retrieval-augmented extraction."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self.dim = 384  # MiniLM-L6-v2 dimension
        self.index = None
        self.segments = []

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def build_index(self, segments: list):
        """Index all segments from a single study's PDF."""
        import hnswlib

        self.segments = segments
        texts = [seg.content for seg in segments]

        if not texts:
            return

        embeddings = self.model.encode(texts, normalize_embeddings=True)

        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.init_index(max_elements=len(texts), ef_construction=200, M=16)
        self.index.add_items(embeddings, list(range(len(texts))))
        self.index.set_ef(50)

        logger.debug(f"Built vector index: {len(texts)} segments")

    def retrieve(self, query: str, top_k: int = 15) -> list:
        """Retrieve top-k most relevant segments for a query."""
        if not self.index or not self.segments:
            return self.segments

        query_emb = self.model.encode([query], normalize_embeddings=True)
        k = min(top_k, len(self.segments))
        labels, distances = self.index.knn_query(query_emb, k=k)

        # Return in document order
        indices = sorted(labels[0])
        return [self.segments[i] for i in indices]

    def retrieve_for_extraction_fields(
        self,
        field_queries: Dict[str, str],
        top_k: int = 15,
    ) -> Dict[str, list]:
        """
        Retrieve relevant chunks for multiple extraction fields at once.

        Returns dict with per-field segments + "_merged_context" (union in doc order).
        """
        if not self.index or not self.segments:
            return {"_merged_context": self.segments}

        results = {}
        all_retrieved_indices = set()

        for field_name, query in field_queries.items():
            query_emb = self.model.encode([query], normalize_embeddings=True)
            k = min(top_k, len(self.segments))
            labels, distances = self.index.knn_query(query_emb, k=k)
            results[field_name] = [self.segments[i] for i in sorted(labels[0])]
            all_retrieved_indices.update(labels[0])

        # Always include ALL table segments (tables are high-value)
        for i, seg in enumerate(self.segments):
            if seg.segment_type == "table":
                all_retrieved_indices.add(i)

        merged_indices = sorted(all_retrieved_indices)
        results["_merged_context"] = [self.segments[i] for i in merged_indices]

        return results


# ======================================================================
# Extraction Field Queries
# ======================================================================

DEFAULT_FIELD_QUERIES = {
    "study_design": "study design randomized controlled trial methodology blinding allocation",
    "population": "participants patients sample size age diagnosis inclusion criteria population",
    "intervention": "intervention treatment stimulation parameters protocol sessions frequency duration",
    "outcomes_primary": "primary outcome measure results mean SD sample size score change",
    "outcomes_secondary": "secondary outcome memory executive function attention depression",
    "risk_of_bias": "randomization blinding allocation concealment attrition dropout",
}


def build_field_queries(extraction_guidance: dict = None) -> Dict[str, str]:
    """Build extraction field queries, optionally enriched by PICO."""
    queries = dict(DEFAULT_FIELD_QUERIES)

    if extraction_guidance:
        pico = extraction_guidance.get("pico", {})
        if pico.get("intervention"):
            queries["intervention"] += f" {pico['intervention']}"
        if pico.get("population"):
            queries["population"] += f" {pico['population']}"
        if pico.get("outcome"):
            queries["outcomes_primary"] += f" {pico['outcome']}"

    return queries
