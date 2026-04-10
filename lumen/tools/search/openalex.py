"""
OpenAlex search -- LUMEN v3

Free scholarly metadata API covering ~250M works.
https://docs.openalex.org/
"""

from __future__ import annotations

import os
import time

import httpx
import structlog

logger = structlog.get_logger()

BASE_URL = "https://api.openalex.org"


def search_openalex(query: str, max_results: int = 2000) -> list[dict]:
    """Search OpenAlex works and return study records.

    Uses cursor-based pagination and the polite pool (email in User-Agent).

    Parameters
    ----------
    query : str
        Free-text search string.
    max_results : int
        Cap on number of records returned (default 2000).

    Returns
    -------
    list[dict]
        One dict per work with keys: study_id, title, abstract, authors,
        year, doi, pmid, pmcid, journal, cited_by_count, is_oa, oa_url,
        source.
    """
    email = os.getenv("NCBI_EMAIL", "lumen@example.com")

    studies: list[dict] = []
    per_page = min(200, max_results)
    cursor = "*"
    data: dict = {}

    headers = {"User-Agent": f"LUMEN/3.0 (mailto:{email})"}

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
                authors: list[str] = []
                for authorship in work.get("authorships", [])[:10]:
                    name = authorship.get("author", {}).get("display_name", "")
                    if name:
                        authors.append(name)

                primary_loc = work.get("primary_location", {}) or {}
                source_obj = primary_loc.get("source", {}) or {}
                journal = source_obj.get("display_name", "")

                ids = work.get("ids", {}) or {}
                doi = (work.get("doi") or "").replace("https://doi.org/", "")
                pmid = (ids.get("pmid") or "").replace(
                    "https://pubmed.ncbi.nlm.nih.gov/", ""
                )
                pmcid = ids.get("pmcid", "")

                abstract = _reconstruct_abstract(
                    work.get("abstract_inverted_index")
                )

                studies.append(
                    {
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
                )

            # Cursor-based pagination
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

            time.sleep(0.2)  # polite pool: ~10 req/s

    total = data.get("meta", {}).get("count", len(studies)) if data else len(studies)
    logger.info(
        "openalex.search",
        total_count=total,
        fetched=len(studies),
        query=query[:120],
    )
    return studies[:max_results]


def get_pdf_url(doi: str = "", pmid: str = "") -> str | None:
    """Look up an open-access PDF URL for a work via OpenAlex.

    Tries DOI first, then PMID.
    """
    if not doi and not pmid:
        return None

    email = os.getenv("NCBI_EMAIL", "lumen@example.com")
    headers = {"User-Agent": f"LUMEN/3.0 (mailto:{email})"}

    try:
        with httpx.Client(timeout=15.0, headers=headers) as client:
            if doi:
                url = f"{BASE_URL}/works/doi:{doi}"
            else:
                url = f"{BASE_URL}/works/pmid:{pmid}"

            resp = client.get(
                url,
                params={
                    "select": "open_access,primary_location,best_oa_location"
                },
            )
            if resp.status_code != 200:
                return None

            data = resp.json()

            oa = data.get("open_access", {})
            if oa.get("is_oa") and oa.get("oa_url"):
                return oa["oa_url"]

            best = data.get("best_oa_location", {})
            if best:
                pdf_url = best.get("pdf_url") or best.get("landing_page_url")
                if pdf_url:
                    return pdf_url

    except Exception as exc:
        logger.debug("openalex.pdf_lookup_failed", error=str(exc))

    return None


# ── internal helpers ─────────────────────────────────────────────────────


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct plain-text abstract from OpenAlex inverted-index format."""
    if not inverted_index:
        return ""
    try:
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(w for _, w in word_positions)
    except Exception:
        return ""
