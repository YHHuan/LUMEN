"""
OpenAlex API — LUMEN v2
========================
Free, no API key required. Covers ~250M works.
https://docs.openalex.org/

Used for:
1. Literature search (Phase 2)
2. PDF URL lookup (pdf_downloader cascade)
"""

import time
import logging
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"


def search_openalex(query: str, max_results: int = 2000,
                    email: str = "lumen@example.com") -> List[dict]:
    """
    Search OpenAlex works and return study records.

    OpenAlex supports full-text search via the 'search' filter.
    Free, no key needed (polite pool with email in User-Agent).
    """
    studies = []
    per_page = min(200, max_results)
    cursor = "*"

    headers = {
        "User-Agent": f"LUMEN/2.0 (mailto:{email})",
    }

    with httpx.Client(timeout=30.0, headers=headers) as client:
        while len(studies) < max_results:
            params = {
                "search": query,
                "per_page": per_page,
                "cursor": cursor,
                "select": (
                    "id,doi,title,publication_year,authorships,"
                    "primary_location,open_access,cited_by_count,"
                    "abstract_inverted_index,ids"
                ),
            }

            resp = client.get(f"{BASE_URL}/works", params=params)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                # Extract authors
                authors = []
                for authorship in work.get("authorships", [])[:10]:
                    name = authorship.get("author", {}).get("display_name", "")
                    if name:
                        authors.append(name)

                # Extract journal
                primary_loc = work.get("primary_location", {}) or {}
                source = primary_loc.get("source", {}) or {}
                journal = source.get("display_name", "")

                # Extract IDs
                ids = work.get("ids", {}) or {}
                doi = (work.get("doi") or "").replace("https://doi.org/", "")
                pmid = (ids.get("pmid") or "").replace("https://pubmed.ncbi.nlm.nih.gov/", "")
                pmcid = ids.get("pmcid", "")

                # Reconstruct abstract from inverted index
                abstract = _reconstruct_abstract(
                    work.get("abstract_inverted_index")
                )

                study = {
                    "study_id": f"OA_{work.get('id', '').split('/')[-1]}",
                    "title": work.get("title", "") or "",
                    "abstract": abstract,
                    "authors": authors,
                    "year": str(work.get("publication_year", "")),
                    "doi": doi,
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "journal": journal,
                    "cited_by_count": work.get("cited_by_count", 0),
                    "is_oa": work.get("open_access", {}).get("is_oa", False),
                    "oa_url": work.get("open_access", {}).get("oa_url", ""),
                    "source": "openalex",
                }
                studies.append(study)

            # Pagination
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

            time.sleep(0.2)  # polite pool: 10 req/s

    total = data.get("meta", {}).get("count", len(studies))
    logger.info(f"OpenAlex search: {total} total, fetched {len(studies)}")
    return studies[:max_results]


def get_pdf_url(doi: str = "", pmid: str = "",
                email: str = "lumen@example.com") -> Optional[str]:
    """
    Look up open access PDF URL for a work via OpenAlex.
    Try by DOI first, then by PMID.
    """
    if not doi and not pmid:
        return None

    headers = {"User-Agent": f"LUMEN/2.0 (mailto:{email})"}

    try:
        with httpx.Client(timeout=15.0, headers=headers) as client:
            # Build lookup URL
            if doi:
                url = f"{BASE_URL}/works/doi:{doi}"
            else:
                url = f"{BASE_URL}/works/pmid:{pmid}"

            resp = client.get(
                url,
                params={"select": "open_access,primary_location,best_oa_location"},
            )
            if resp.status_code != 200:
                return None

            data = resp.json()

            # Try open_access.oa_url first
            oa = data.get("open_access", {})
            if oa.get("is_oa") and oa.get("oa_url"):
                return oa["oa_url"]

            # Try best_oa_location
            best = data.get("best_oa_location", {})
            if best:
                pdf_url = best.get("pdf_url") or best.get("landing_page_url")
                if pdf_url:
                    return pdf_url

    except Exception as e:
        logger.debug(f"OpenAlex PDF lookup failed: {e}")

    return None


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""

    try:
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(w for _, w in word_positions)
    except Exception:
        return ""
