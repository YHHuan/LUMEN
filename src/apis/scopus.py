"""
Scopus / Elsevier API — LUMEN v2
"""

import time
import logging
from typing import List

import httpx

from src.config import cfg

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elsevier.com/content/search/scopus"


def search_scopus(query: str, max_results: int = 5000) -> List[dict]:
    """Search Scopus and return study records."""
    api_key = cfg.elsevier_api_key
    if not api_key:
        logger.warning("Scopus API key not configured")
        return []

    studies = []
    start = 0
    count = min(25, max_results)

    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }
    inst_token = cfg.elsevier_inst_token
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

    with httpx.Client(timeout=30.0, headers=headers) as client:
        while start < max_results:
            params = {
                "query": query,
                "start": start,
                "count": count,
            }

            resp = client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("search-results", {}).get("entry", [])
            if not results:
                break

            for r in results:
                if r.get("@_fa") == "true":
                    study = {
                        "study_id": f"SCOPUS_{r.get('dc:identifier', '').replace('SCOPUS_ID:', '')}",
                        "title": r.get("dc:title", ""),
                        "abstract": r.get("dc:description", ""),
                        "authors": r.get("dc:creator", ""),
                        "year": r.get("prism:coverDate", "")[:4],
                        "doi": r.get("prism:doi", ""),
                        "journal": r.get("prism:publicationName", ""),
                        "source": "scopus",
                    }
                    studies.append(study)

            total = int(data.get("search-results", {}).get("opensearch:totalResults", 0))
            start += count

            if start >= total:
                break
            time.sleep(0.5)

    logger.info(f"Scopus search: fetched {len(studies)}")
    return studies[:max_results]
