"""
PubMed / NCBI E-utilities search -- LUMEN v3
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET

import httpx
import structlog

logger = structlog.get_logger()

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str, max_results: int = 5000) -> list[dict]:
    """Search PubMed via E-utilities and return study records.

    Parameters
    ----------
    query : str
        PubMed search string (supports MeSH, boolean operators, etc.).
    max_results : int
        Maximum number of records to retrieve (default 5000).

    Returns
    -------
    list[dict]
        One dict per article with keys: study_id, pmid, title, abstract,
        authors, year, doi, journal, source.
    """
    api_key = os.getenv("NCBI_API_KEY", "")
    email = os.getenv("NCBI_EMAIL", "lumen@example.com")

    # -- Step 1: ESearch to collect PMIDs ---------------------------------
    params: dict = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "usehistory": "y",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email

    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{BASE_URL}/esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()

    result = data.get("esearchresult", {})
    pmids: list[str] = result.get("idlist", [])
    total_count = int(result.get("count", 0))

    logger.info(
        "pubmed.esearch",
        total_count=total_count,
        fetching=len(pmids),
        query=query[:120],
    )

    if not pmids:
        return []

    # -- Step 2: EFetch in batches ----------------------------------------
    studies: list[dict] = []
    batch_size = 200
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        studies.extend(_fetch_details(batch, api_key))
        time.sleep(0.5 if api_key else 1.0)

    return studies


# ── internal helpers ─────────────────────────────────────────────────────


def _fetch_details(pmids: list[str], api_key: str = "") -> list[dict]:
    """Fetch article details via EFetch XML and parse them."""
    params: dict = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if api_key:
        params["api_key"] = api_key

    with httpx.Client(timeout=60.0) as client:
        resp = client.get(f"{BASE_URL}/efetch.fcgi", params=params)
        resp.raise_for_status()

    root = ET.fromstring(resp.text)

    studies: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        study = _parse_pubmed_article(article)
        if study:
            studies.append(study)
    return studies


def _parse_pubmed_article(article: ET.Element) -> dict | None:
    """Parse a single ``<PubmedArticle>`` XML element into a dict."""
    try:
        medline = article.find(".//MedlineCitation")
        if medline is None:
            return None

        pmid = medline.findtext(".//PMID", "")
        art = medline.find(".//Article")
        if art is None:
            return None

        title = art.findtext(".//ArticleTitle", "")

        # Abstract
        abstract_parts: list[str] = []
        for abs_text in art.findall(".//Abstract/AbstractText"):
            label = abs_text.get("Label", "")
            text = abs_text.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors: list[str] = []
        for author in art.findall(".//AuthorList/Author"):
            last = author.findtext("LastName", "")
            first = author.findtext("ForeName", "")
            if last:
                authors.append(f"{last}, {first}" if first else last)

        # Year
        pub_date = art.find(".//Journal/JournalIssue/PubDate")
        year = ""
        if pub_date is not None:
            year = pub_date.findtext("Year", "")
            if not year:
                medline_date = pub_date.findtext("MedlineDate", "")
                if medline_date:
                    m = re.search(r"(\d{4})", medline_date)
                    year = m.group(1) if m else ""

        # DOI
        doi = ""
        for eid in art.findall(".//ELocationID"):
            if eid.get("EIdType") == "doi":
                doi = eid.text or ""

        # Journal
        journal = art.findtext(".//Journal/Title", "")

        return {
            "study_id": f"PMID_{pmid}",
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "doi": doi,
            "journal": journal,
            "source": "pubmed",
        }
    except Exception as exc:
        logger.debug("pubmed.parse_failed", error=str(exc))
        return None
