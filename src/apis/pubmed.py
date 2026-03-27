"""
PubMed / NCBI E-utilities API — LUMEN v2
"""

import time
import logging
from typing import List, Optional

import httpx

from src.config import cfg

logger = logging.getLogger(__name__)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str, max_results: int = 5000) -> List[dict]:
    """Search PubMed and return study records."""
    api_key = cfg.ncbi_api_key
    email = cfg.ncbi_email

    # Step 1: ESearch to get PMIDs
    params = {
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
    pmids = result.get("idlist", [])
    total_count = int(result.get("count", 0))

    logger.info(f"PubMed search: {total_count} total, fetching {len(pmids)}")

    if not pmids:
        return []

    # Step 2: EFetch to get details
    studies = []
    batch_size = 200
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        batch_studies = _fetch_details(batch, api_key)
        studies.extend(batch_studies)
        time.sleep(0.5 if api_key else 1.0)

    return studies


def _fetch_details(pmids: list, api_key: str = "") -> List[dict]:
    """Fetch study details from PubMed via EFetch."""
    params = {
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

    # Parse XML
    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.text)

    studies = []
    for article in root.findall(".//PubmedArticle"):
        study = _parse_pubmed_article(article)
        if study:
            studies.append(study)

    return studies


def _parse_pubmed_article(article) -> Optional[dict]:
    """Parse a single PubmedArticle XML element."""
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
        abstract_parts = []
        for abs_text in art.findall(".//Abstract/AbstractText"):
            label = abs_text.get("Label", "")
            text = abs_text.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors = []
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
                    import re
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
    except Exception as e:
        logger.debug(f"Failed to parse PubMed article: {e}")
        return None
