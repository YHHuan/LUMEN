"""
Query Syntax Converter — LUMEN v2
====================================
Converts generic search queries to database-specific syntax.
"""

import re
import logging

logger = logging.getLogger(__name__)


def to_pubmed_syntax(query: str) -> str:
    """Convert generic query to PubMed/MEDLINE syntax."""
    query = query.replace(" AND ", " AND ")
    query = query.replace(" OR ", " OR ")
    return query


def to_scopus_syntax(query: str) -> str:
    """Convert generic query to Scopus syntax."""
    q = query.replace("[tiab]", "TITLE-ABS")
    q = q.replace("[mh]", "KEY")
    return q


def to_europepmc_syntax(query: str) -> str:
    """Convert generic query to Europe PMC syntax."""
    return query


def to_crossref_syntax(query: str) -> str:
    """Simplify query for CrossRef (only supports simple keyword search)."""
    # Strip field tags and boolean operators
    clean = re.sub(r"\[.*?\]", "", query)
    clean = re.sub(r"\b(AND|OR|NOT)\b", " ", clean)
    clean = re.sub(r"[()]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean
