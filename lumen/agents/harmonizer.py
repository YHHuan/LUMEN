"""
Outcome harmonization: raw outcome names → canonical clusters.

Two-stage approach:
1. Embedding-based clustering (sentence-transformers)
2. LLM refinement (merge/split/rename)
"""
from __future__ import annotations

import json
import structlog
import numpy as np

from lumen.agents.base import BaseAgent, LumenParseError

logger = structlog.get_logger()

# Cosine similarity threshold for initial clustering
SIMILARITY_THRESHOLD = 0.80


class HarmonizerAgent(BaseAgent):
    tier = "smart"
    agent_name = "harmonizer"
    prompt_file = "harmonizer.yaml"

    def harmonize(self, extractions: list[dict], pico: dict) -> dict:
        """
        Harmonize outcome names across studies.

        Step 1: Collect all raw outcome names
        Step 2: Embedding-based clustering
        Step 3: LLM refinement → canonical names
        Step 4: Apply mapping to extractions

        Returns {outcome_clusters, harmonized_data, unmapped}.
        """
        # Step 1: Collect raw names
        raw_names = self._collect_outcome_names(extractions)
        if not raw_names:
            return {
                "outcome_clusters": {},
                "harmonized_data": extractions,
                "unmapped": [],
            }

        logger.info("harmonizer_start", n_unique_outcomes=len(set(raw_names)))

        # Step 2: Embedding clustering
        candidate_clusters = self._cluster_by_embedding(list(set(raw_names)))

        # Step 3: LLM refinement
        refined = self._llm_refine_clusters(candidate_clusters, pico)
        clusters = refined.get("clusters", candidate_clusters)
        unmapped = refined.get("unmapped", [])

        # Step 4: Apply mapping
        harmonized = self._apply_mapping(extractions, clusters)

        logger.info("harmonizer_done",
                     n_clusters=len(clusters),
                     n_unmapped=len(unmapped))

        return {
            "outcome_clusters": clusters,
            "harmonized_data": harmonized,
            "unmapped": unmapped,
        }

    @staticmethod
    def _collect_outcome_names(extractions: list[dict]) -> list[str]:
        """Gather all outcome names from extraction records."""
        names = []
        for ext in extractions:
            # Handle both flat and nested structures
            if "extractions" in ext:
                for item in ext["extractions"]:
                    name = item.get("outcome_name", "")
                    if name:
                        names.append(name)
            elif "outcome_name" in ext:
                names.append(ext["outcome_name"])
        return names

    def _cluster_by_embedding(self, unique_names: list[str]) -> dict[str, list[str]]:
        """
        Stage 1: Agglomerative clustering using sentence-transformers.
        Falls back to exact-match grouping if embeddings unavailable.
        """
        if len(unique_names) <= 1:
            return {unique_names[0]: unique_names} if unique_names else {}

        try:
            embeddings = self._get_embeddings(unique_names)
            return self._agglomerative_cluster(unique_names, embeddings)
        except Exception as e:
            logger.warning("embedding_clustering_failed", error=str(e))
            # Fallback: lowercased exact match
            return self._fallback_cluster(unique_names)

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get embeddings using sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, normalize_embeddings=True)

    @staticmethod
    def _agglomerative_cluster(names: list[str],
                               embeddings: np.ndarray) -> dict[str, list[str]]:
        """Cluster names by cosine similarity."""
        from sklearn.cluster import AgglomerativeClustering

        n = len(names)
        if n == 1:
            return {names[0]: names}

        # Cosine distance matrix
        sim_matrix = embeddings @ embeddings.T
        dist_matrix = 1 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = np.clip(dist_matrix, 0, 2)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - SIMILARITY_THRESHOLD,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(dist_matrix)

        clusters: dict[str, list[str]] = {}
        for name, label in zip(names, labels):
            key = f"cluster_{label}"
            clusters.setdefault(key, []).append(name)

        # Use the shortest name in each cluster as the key
        renamed = {}
        for _, members in clusters.items():
            canonical = min(members, key=len)
            renamed[canonical] = members

        return renamed

    @staticmethod
    def _fallback_cluster(names: list[str]) -> dict[str, list[str]]:
        """Simple lowercased grouping fallback."""
        clusters: dict[str, list[str]] = {}
        for name in names:
            key = name.lower().strip()
            found = False
            for canonical in clusters:
                if canonical.lower().strip() == key:
                    clusters[canonical].append(name)
                    found = True
                    break
            if not found:
                clusters[name] = [name]
        return clusters

    def _llm_refine_clusters(self, candidate_clusters: dict,
                             pico: dict) -> dict:
        """Stage 2: LLM reviews and refines clusters."""
        user_content = (
            f"## PICO\n{json.dumps(pico, indent=2)}\n\n"
            f"## Candidate Clusters\n{json.dumps(candidate_clusters, indent=2)}\n\n"
            "Please review these clusters and refine them. "
            "Merge clusters that measure the same construct, "
            "split clusters that were incorrectly grouped, "
            "and assign canonical names aligned with the PICO."
        )
        messages = self._build_messages(user_content)

        try:
            response = self._call_llm(
                messages, response_format={"type": "json_object"},
                phase="harmonization",
            )
            return self._parse_json(response, retry_messages=messages,
                                    phase="harmonization")
        except LumenParseError:
            logger.warning("harmonizer_llm_refinement_failed",
                           msg="Using embedding clusters as-is")
            return {"clusters": candidate_clusters, "unmapped": []}

    @staticmethod
    def _apply_mapping(extractions: list[dict],
                       clusters: dict[str, list[str]]) -> list[dict]:
        """Apply canonical name mapping to all extraction records."""
        # Build reverse lookup: raw_name → canonical
        mapping = {}
        for canonical, raw_names in clusters.items():
            for raw in raw_names:
                mapping[raw.lower()] = canonical

        updated = []
        for ext in extractions:
            ext = dict(ext)  # shallow copy
            if "extractions" in ext:
                new_items = []
                for item in ext["extractions"]:
                    item = dict(item)
                    raw = item.get("outcome_name", "").lower()
                    item["canonical_outcome"] = mapping.get(raw, item.get("outcome_name"))
                    new_items.append(item)
                ext["extractions"] = new_items
            elif "outcome_name" in ext:
                raw = ext["outcome_name"].lower()
                ext["canonical_outcome"] = mapping.get(raw, ext["outcome_name"])
            updated.append(ext)

        return updated
