"""
Citation Verification Engine — LUMEN v2
========================================
BM25-based citation retrieval + LLM assertion extraction for grounded
citation verification. Supplements the vector-based ReferencePool with
lexical matching and decomposes claims into verifiable assertions.

Pipeline:
1. Extract assertions from manuscript sentences
2. BM25 retrieval over reference corpus (title + abstract)
3. Hybrid scoring: BM25 rank-score + vector similarity
4. LLM verification of top candidates per assertion
"""

import math
import re
import logging
from collections import Counter
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# BM25 Index
# ======================================================================

class BM25Index:
    """Okapi BM25 index over reference documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_len: List[int] = []
        self.avg_dl: float = 0.0
        self.n_docs: int = 0
        self.df: Counter = Counter()       # document frequency per term
        self.idf: Dict[str, float] = {}
        self._raw_docs: List[dict] = []

    def build(self, references: List[dict]):
        """Build BM25 index from reference dicts with title/abstract."""
        self._raw_docs = references
        self.corpus = []
        self.doc_len = []
        self.df = Counter()

        for ref in references:
            text = f"{ref.get('title', '')} {ref.get('abstract', '')}"
            tokens = _tokenize(text)
            self.corpus.append(tokens)
            self.doc_len.append(len(tokens))
            seen = set(tokens)
            for t in seen:
                self.df[t] += 1

        self.n_docs = len(self.corpus)
        self.avg_dl = sum(self.doc_len) / max(1, self.n_docs)

        # Pre-compute IDF
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(
                (self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_index, score) tuples sorted by BM25 score."""
        query_tokens = _tokenize(query_text)
        scores = []

        for idx, doc_tokens in enumerate(self.corpus):
            score = 0.0
            dl = self.doc_len[idx]
            tf_doc = Counter(doc_tokens)

            for qt in query_tokens:
                if qt not in self.idf:
                    continue
                tf = tf_doc.get(qt, 0)
                idf = self.idf[qt]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / self.avg_dl
                )
                score += idf * numerator / denominator

            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ======================================================================
# Assertion Extractor
# ======================================================================

class AssertionExtractor:
    """
    Decompose manuscript sentences into atomic, verifiable assertions.

    Uses rule-based decomposition for speed, with optional LLM fallback
    for complex sentences.
    """

    # Claim signal patterns
    _CLAIM_PATTERNS = [
        re.compile(r"\b(?:found|showed|demonstrated|reported|observed|indicated)\b", re.I),
        re.compile(r"\b(?:significant|effective|improved|reduced|increased|associated)\b", re.I),
        re.compile(r"\b(?:meta-analysis|systematic review|RCT|trial|study|studies)\b", re.I),
        re.compile(r"\b(?:effect size|Cohen|Hedges|odds ratio|risk ratio|SMD|MD|WMD)\b", re.I),
        re.compile(r"\b(?:p\s*[<=<]\s*0?\.\d|CI\s*[\[:(]|confidence interval)\b", re.I),
    ]

    def extract_assertions(self, text: str) -> List[dict]:
        """
        Extract factual assertions from manuscript text.

        Returns list of dicts:
            {
                "assertion": str,           # The factual claim
                "source_sentence": str,      # Original sentence
                "claim_type": str,           # "empirical" | "methodological" | "general"
                "confidence": float,         # How likely this is a verifiable claim
            }
        """
        sentences = _split_sentences(text)
        assertions = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue

            claim_type, confidence = self._classify_sentence(sent)

            if confidence < 0.3:
                continue

            # For compound sentences, try splitting on conjunctions
            sub_claims = self._split_compound(sent)

            for sub in sub_claims:
                sub = sub.strip()
                if len(sub) < 15:
                    continue
                assertions.append({
                    "assertion": sub,
                    "source_sentence": sent,
                    "claim_type": claim_type,
                    "confidence": round(confidence, 3),
                })

        return assertions

    def _classify_sentence(self, sentence: str) -> Tuple[str, float]:
        """Classify sentence type and estimate verifiability confidence."""
        score = 0.0

        for pattern in self._CLAIM_PATTERNS:
            if pattern.search(sentence):
                score += 0.2

        # Has a citation marker already? Higher confidence it's a claim
        if re.search(r"\[REF:", sentence) or re.search(r"\[CITATION", sentence):
            score += 0.3

        # Statistical results
        if re.search(r"\bp\s*[<=<]\s*0?\.\d", sentence):
            score += 0.2
            return "empirical", min(score, 1.0)

        if re.search(r"\b(?:PRISMA|Cochrane|GRADE|RoB|Jadad)\b", sentence, re.I):
            return "methodological", max(score, 0.5)

        if score >= 0.4:
            return "empirical", min(score, 1.0)

        return "general", score

    def _split_compound(self, sentence: str) -> List[str]:
        """Split compound sentences at conjunctions if both parts are claims."""
        # Don't split short sentences
        if len(sentence) < 80:
            return [sentence]

        # Split on "; " or ", and " or ", while "
        parts = re.split(r";\s+|,\s+and\s+|,\s+while\s+|,\s+whereas\s+", sentence)

        if len(parts) == 1:
            return [sentence]

        # Only keep parts that look like claims
        result = []
        for part in parts:
            part = part.strip()
            if len(part) > 15:
                result.append(part)

        return result if result else [sentence]


# ======================================================================
# Hybrid Citation Verifier
# ======================================================================

class HybridCitationVerifier:
    """
    Combines BM25 + vector similarity for citation verification.

    Steps:
    1. Build BM25 index over references
    2. For each assertion, retrieve candidates via BM25
    3. Re-rank with vector similarity (if available)
    4. Verify top candidate with LLM
    """

    def __init__(self, references: List[dict],
                 vector_pool=None,
                 bm25_weight: float = 0.4,
                 vector_weight: float = 0.6):
        self.references = references
        self.vector_pool = vector_pool
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        self.bm25 = BM25Index()
        self.bm25.build(references)
        self.assertion_extractor = AssertionExtractor()

    def verify_manuscript(self, manuscript_text: str,
                          guardian_agent=None) -> dict:
        """
        Full citation verification pipeline on a manuscript.

        Returns:
            {
                "assertions": [...],
                "verification_results": [...],
                "summary": {
                    "total_assertions": int,
                    "verified": int,
                    "unverified": int,
                    "verification_rate": float,
                },
            }
        """
        # 1. Extract assertions
        assertions = self.assertion_extractor.extract_assertions(manuscript_text)
        logger.info(f"Extracted {len(assertions)} assertions from manuscript")

        # 2. Verify each assertion
        results = []
        for assertion in assertions:
            result = self._verify_single(assertion, guardian_agent)
            results.append(result)

        verified = sum(1 for r in results if r["verified"])
        total = len(results)

        return {
            "assertions": assertions,
            "verification_results": results,
            "summary": {
                "total_assertions": total,
                "verified": verified,
                "unverified": total - verified,
                "verification_rate": round(verified / max(1, total), 3),
            },
        }

    def _verify_single(self, assertion: dict,
                       guardian_agent=None) -> dict:
        """Verify a single assertion against the reference corpus."""
        query = assertion["assertion"]

        # BM25 retrieval
        bm25_hits = self.bm25.query(query, top_k=5)

        # Hybrid scoring
        candidates = []
        for doc_idx, bm25_score in bm25_hits:
            ref = self.references[doc_idx]

            # Normalize BM25 score (approximate)
            norm_bm25 = min(bm25_score / 30.0, 1.0)

            # Vector similarity if available
            vec_sim = 0.0
            if self.vector_pool:
                vec_results = self.vector_pool.find_references_for_claim(
                    query, top_k=10
                )
                for vref, vsim in vec_results:
                    if vref.get("id") == ref.get("id") or \
                       vref.get("title", "").lower() == ref.get("title", "").lower():
                        vec_sim = vsim
                        break

            hybrid_score = (
                self.bm25_weight * norm_bm25 +
                self.vector_weight * vec_sim
            )

            candidates.append({
                "reference": ref,
                "bm25_score": round(norm_bm25, 4),
                "vector_score": round(vec_sim, 4),
                "hybrid_score": round(hybrid_score, 4),
            })

        candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)

        # LLM verification on top candidate
        verified = False
        best_ref = None
        verification_reason = "no_candidates"

        if candidates and candidates[0]["hybrid_score"] >= 0.15:
            top = candidates[0]
            best_ref = top["reference"]

            if guardian_agent:
                llm_result = guardian_agent._verify_citation(
                    claim_context=assertion["source_sentence"],
                    reference_title=best_ref.get("title", ""),
                    reference_abstract=best_ref.get("abstract", ""),
                )
                verified = llm_result.get("supported", False)
                verification_reason = llm_result.get("reason", "")
            else:
                # Without LLM, accept if hybrid score is strong
                verified = top["hybrid_score"] >= 0.5
                verification_reason = "threshold_only"

        return {
            "assertion": assertion["assertion"],
            "claim_type": assertion["claim_type"],
            "verified": verified,
            "best_reference": {
                "title": best_ref.get("title", "") if best_ref else None,
                "citation": best_ref.get("citation", "") if best_ref else None,
                "citation_number": best_ref.get("citation_number") if best_ref else None,
            },
            "hybrid_score": candidates[0]["hybrid_score"] if candidates else 0.0,
            "verification_reason": verification_reason,
            "n_candidates": len(candidates),
        }

    def find_best_reference(self, claim: str,
                            top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        Hybrid BM25 + vector search for a claim.
        Returns: [(reference_dict, hybrid_score), ...]
        """
        bm25_hits = self.bm25.query(claim, top_k=top_k * 2)

        scored = []
        seen_titles = set()

        for doc_idx, bm25_score in bm25_hits:
            ref = self.references[doc_idx]
            title_key = ref.get("title", "").lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            norm_bm25 = min(bm25_score / 30.0, 1.0)

            vec_sim = 0.0
            if self.vector_pool:
                vec_results = self.vector_pool.find_references_for_claim(
                    claim, top_k=10
                )
                for vref, vsim in vec_results:
                    if vref.get("title", "").lower() == title_key:
                        vec_sim = vsim
                        break

            hybrid = (self.bm25_weight * norm_bm25 +
                      self.vector_weight * vec_sim)
            scored.append((ref, hybrid))

        # Also check vector-only hits
        if self.vector_pool:
            vec_results = self.vector_pool.find_references_for_claim(
                claim, top_k=top_k
            )
            for vref, vsim in vec_results:
                title_key = vref.get("title", "").lower()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    scored.append((vref, self.vector_weight * vsim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ======================================================================
# Utilities
# ======================================================================

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "this", "that", "these", "those", "it", "its", "as", "if", "than",
    "so", "such", "very", "too", "also", "each", "all", "any", "both",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "about", "which", "who", "whom",
    "what", "where", "when", "how", "there", "here", "then", "more",
    "most", "other", "some", "only", "just", "own", "same",
})


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization with stop-word removal."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Handle common abbreviations
    text = re.sub(r"(et al)\.", r"\1<DOT>", text)
    text = re.sub(r"(Fig|fig|Tab|tab|vs|Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr)\.", r"\1<DOT>", text)
    text = re.sub(r"(\d)\.", r"\1<DOT>", text)

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    # Restore dots
    return [s.replace("<DOT>", ".") for s in sentences]
