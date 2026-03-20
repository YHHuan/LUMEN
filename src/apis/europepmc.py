"""
Europe PMC API Wrapper — v5
=============================
- synonym=false (關閉 MeSH 同義詞自動展開)
- 第一頁印出 Total Hits
"""

import time
import logging
from typing import Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePMCAPI:
    
    def __init__(self):
        self.min_interval = 0.2
        self._last_request_time = 0
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    def search(self, query: str, max_results: int = 5000) -> list:
        """搜尋 Europe PMC。"""
        all_records = []
        cursor_mark = "*"
        page_size = 1000
        
        while len(all_records) < max_results:
            self._rate_limit()
            
            params = {
                "query": query,
                "format": "json",
                "pageSize": min(page_size, max_results - len(all_records)),
                "cursorMark": cursor_mark,
                "resultType": "core",
                "synonym": "false",
            }
            
            resp = requests.get(f"{EUROPEPMC_BASE}/search",
                              params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            # First page: log total hits
            if cursor_mark == "*":
                total_hits = data.get("hitCount", 0)
                logger.info(f"Europe PMC Total Hits: {total_hits}")
                if total_hits == 0:
                    logger.warning("Europe PMC: 0 hits — check query syntax")
                    break
            
            results = data.get("resultList", {}).get("result", [])
            if not results:
                break
            
            for r in results:
                record = self._parse_result(r)
                if record:
                    all_records.append(record)
            
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor
            
            logger.info(f"Europe PMC: {len(all_records)} records fetched...")
        
        logger.info(f"Europe PMC search completed: {len(all_records)} total records")
        return all_records[:max_results]
    
    def _parse_result(self, r: dict) -> Optional[dict]:
        pmid = r.get("pmid", "")
        return {
            "study_id": f"PMID_{pmid}" if pmid else f"EPMC_{r.get('id', '')}",
            "pmid": pmid,
            "pmc_id": r.get("pmcid", ""),
            "doi": r.get("doi", ""),
            "title": r.get("title", ""),
            "abstract": r.get("abstractText", ""),
            "authors": self._parse_authors(r.get("authorList", {})),
            "first_author": "",
            "journal": r.get("journalTitle", ""),
            "year": str(r.get("pubYear", "")),
            "mesh_terms": [],
            "publication_types": [],
            "source": "europepmc",
            "is_open_access": r.get("isOpenAccess", "N") == "Y",
            "full_text_url": (
                r.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url", "")
                if r.get("fullTextUrlList") else ""
            ),
        }
    
    def _parse_authors(self, author_list: dict) -> list:
        authors = []
        for a in author_list.get("author", []):
            name = a.get("fullName", "")
            if name:
                authors.append(name)
        return authors
