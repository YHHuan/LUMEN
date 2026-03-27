"""
Deduplication Utilities — LUMEN v2
====================================
DOI/PMID exact dedup + fuzzy title matching + canonical citation normalization.
"""

import re
import logging
from collections import defaultdict
from typing import List, Tuple, Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


def deduplicate_studies(studies: list, threshold: float = 90.0) -> Tuple[list, list]:
    """
    Deduplicate studies by DOI, PMID, and fuzzy title matching.

    Returns: (unique_studies, dedup_log)
    """
    seen_doi = {}
    seen_pmid = {}
    seen_titles = []
    unique = []
    dedup_log = []

    for study in studies:
        study_id = study.get("study_id", "")
        doi = _normalize_doi(study.get("doi", ""))
        pmid = study.get("pmid", "")
        title = _normalize_title(study.get("title", ""))

        is_dup = False
        dup_reason = ""
        dup_of = ""

        # DOI match
        if doi and doi in seen_doi:
            is_dup = True
            dup_reason = "doi_match"
            dup_of = seen_doi[doi]

        # PMID match
        elif pmid and pmid in seen_pmid:
            is_dup = True
            dup_reason = "pmid_match"
            dup_of = seen_pmid[pmid]

        # Fuzzy title match
        elif title:
            for prev_title, prev_id in seen_titles:
                score = fuzz.ratio(title, prev_title)
                if score >= threshold:
                    is_dup = True
                    dup_reason = f"fuzzy_title_match (score={score})"
                    dup_of = prev_id
                    break

        if is_dup:
            dedup_log.append({
                "study_id": study_id,
                "duplicate_of": dup_of,
                "reason": dup_reason,
            })
        else:
            if doi:
                seen_doi[doi] = study_id
            if pmid:
                seen_pmid[pmid] = study_id
            if title:
                seen_titles.append((title, study_id))
            unique.append(study)

    logger.info(
        f"Deduplication: {len(studies)} -> {len(unique)} "
        f"({len(dedup_log)} duplicates removed)"
    )

    return unique, dedup_log


def _normalize_doi(doi: str) -> str:
    if not doi:
        return ""
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi


def _normalize_title(title: str) -> str:
    if not title:
        return ""
    title = title.strip().lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title


# ======================================================================
# Canonical Citation Normalization
# ======================================================================

def normalize_citation(citation: str) -> str:
    """Normalize to 'FirstAuthor Year' format."""
    if not citation:
        return ""

    # Try "Author et al. Year" pattern
    m = re.match(r"^([A-Z][a-z]+).*?(\d{4})", citation)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    # Try "Author, X.-Y. (Year)" pattern
    m = re.match(r"^([A-Z][a-z]+),?\s.*?(\d{4})", citation)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    return citation.strip()


def generate_canonical_citation(study: dict) -> str:
    """Generate canonical citation from study fields."""
    authors = study.get("authors", "")
    year = study.get("year", study.get("publication_year", ""))

    if isinstance(authors, list):
        first_author = authors[0] if authors else "Unknown"
    elif isinstance(authors, str):
        first_author = authors.split(",")[0].split(" and ")[0].strip()
    else:
        first_author = "Unknown"

    # Extract surname
    parts = first_author.split()
    surname = parts[-1] if parts else first_author

    return f"{surname} {year}" if year else surname


def deduplicate_for_meta_analysis(extracted_data: list) -> Tuple[list, list]:
    """
    Ensure no duplicate studies in meta-analysis input.
    Groups by canonical citation, keeps the one with most complete data.
    """
    citation_groups = defaultdict(list)

    for study in extracted_data:
        canon = normalize_citation(study.get("canonical_citation", ""))
        if not canon:
            canon = generate_canonical_citation(study)
        citation_groups[canon].append(study)

    deduped = []
    dedup_log = []

    for canon, studies in citation_groups.items():
        if len(studies) == 1:
            deduped.append(studies[0])
        else:
            best = max(studies, key=lambda s: _count_complete_outcomes(s))
            deduped.append(best)
            removed = [s for s in studies if s is not best]
            dedup_log.append({
                "canonical_citation": canon,
                "kept": best.get("study_id", ""),
                "removed": [s.get("study_id", "") for s in removed],
                "reason": "duplicate_citation",
            })

    return deduped, dedup_log


def _count_complete_outcomes(study: dict) -> int:
    """Count how many outcomes have complete data."""
    count = 0
    for outcome in study.get("outcomes", []):
        for group in ["intervention_group", "control_group"]:
            g = outcome.get(group, {})
            if g.get("mean") is not None and g.get("sd") is not None and g.get("n") is not None:
                count += 1
    return count


# ======================================================================
# RIS File Parser
# ======================================================================

def parse_ris_file(filepath: str) -> list:
    """Parse a .ris file into a list of study dicts."""
    try:
        import rispy
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            entries = rispy.load(f)

        studies = []
        for entry in entries:
            study = {
                "title": entry.get("title", entry.get("primary_title", "")),
                "abstract": entry.get("abstract", ""),
                "authors": entry.get("authors", entry.get("first_authors", [])),
                "year": entry.get("year", entry.get("publication_year", "")),
                "doi": entry.get("doi", ""),
                "pmid": entry.get("pmid", ""),
                "journal": entry.get("journal_name", entry.get("secondary_title", "")),
                "source_file": filepath,
            }
            studies.append(study)

        return studies

    except Exception as e:
        logger.error(f"Failed to parse RIS file {filepath}: {e}")
        return []
