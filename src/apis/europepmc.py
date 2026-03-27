"""
Europe PMC API — LUMEN v2
"""

import time
import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def search_europepmc(query: str, max_results: int = 5000) -> List[dict]:
    """Search Europe PMC and return study records."""
    studies = []
    cursor_mark = "*"
    page_size = min(1000, max_results)

    with httpx.Client(timeout=30.0) as client:
        while len(studies) < max_results:
            params = {
                "query": query,
                "format": "json",
                "pageSize": page_size,
                "cursorMark": cursor_mark,
                "resultType": "core",
            }

            resp = client.get(f"{BASE_URL}/search", params=params)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("resultList", {}).get("result", [])
            if not results:
                break

            for r in results:
                study = {
                    "study_id": f"EPMC_{r.get('id', '')}",
                    "pmid": r.get("pmid", ""),
                    "title": r.get("title", ""),
                    "abstract": r.get("abstractText", ""),
                    "authors": [
                        a.get("fullName", "")
                        for a in (r.get("authorList", {}).get("author", []))
                    ],
                    "year": r.get("pubYear", ""),
                    "doi": r.get("doi", ""),
                    "journal": r.get("journalTitle", ""),
                    "source": "europepmc",
                }
                studies.append(study)

            next_cursor = data.get("nextCursorMark", "")
            if next_cursor == cursor_mark or not next_cursor:
                break
            cursor_mark = next_cursor

            time.sleep(0.3)

    total = data.get("hitCount", len(studies))
    logger.info(f"Europe PMC search: {total} total, fetched {len(studies)}")

    return studies[:max_results]
