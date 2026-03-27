"""
CrossRef API — LUMEN v2
"""

import time
import logging
from typing import List

import httpx

from src.config import cfg

logger = logging.getLogger(__name__)

BASE_URL = "https://api.crossref.org/works"


def search_crossref(query: str, max_results: int = 1000) -> List[dict]:
    """Search CrossRef and return study records."""
    studies = []
    offset = 0
    rows = min(100, max_results)

    headers = {}
    email = cfg.crossref_email
    if email:
        headers["User-Agent"] = f"LUMEN/2.0 (mailto:{email})"

    with httpx.Client(timeout=30.0, headers=headers) as client:
        while offset < max_results:
            params = {
                "query": query,
                "rows": rows,
                "offset": offset,
                "select": "DOI,title,author,published-print,container-title,abstract",
            }

            resp = client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("message", {}).get("items", [])
            if not items:
                break

            for item in items:
                title_list = item.get("title", [])
                title = title_list[0] if title_list else ""

                authors = []
                for a in item.get("author", []):
                    name = f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                    if name:
                        authors.append(name)

                date_parts = item.get("published-print", {}).get("date-parts", [[]])
                year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""

                journal_list = item.get("container-title", [])
                journal = journal_list[0] if journal_list else ""

                study = {
                    "study_id": f"CR_{item.get('DOI', '').replace('/', '_')}",
                    "title": title,
                    "abstract": item.get("abstract", ""),
                    "authors": authors,
                    "year": year,
                    "doi": item.get("DOI", ""),
                    "journal": journal,
                    "source": "crossref",
                }
                studies.append(study)

            offset += rows
            time.sleep(0.5)

    logger.info(f"CrossRef search: fetched {len(studies)}")
    return studies[:max_results]
