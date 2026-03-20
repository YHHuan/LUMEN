"""
Scopus (Elsevier) API Wrapper
===============================
需要: ELSEVIER_API_KEY (+ 校內 IP 或 ELSEVIER_INST_TOKEN)

Rate limit: ~2 req/s with key
回傳上限: 5000 per query (Elsevier hard limit)
"""

import time
import logging
from typing import Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import cfg

logger = logging.getLogger(__name__)

SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"


class ScopusAPI:
    
    def __init__(self):
        self.api_key = cfg.elsevier_api_key
        self.inst_token = cfg.elsevier_inst_token
        self.min_interval = 0.5
        self._last_request_time = 0
        
        if not self.api_key:
            logger.info("Scopus: No API key — skipping")
    
    @property
    def available(self) -> bool:
        return bool(self.api_key)
    
    def _headers(self) -> dict:
        h = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json",
        }
        if self.inst_token:
            h["X-ELS-Insttoken"] = self.inst_token
        return h
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    def search(self, query: str, max_results: int = 5000) -> list:
        """
        搜尋 Scopus。
        
        Args:
            query: Scopus advanced syntax, e.g. 
                   TITLE-ABS-KEY("brain stimulation") AND PUBYEAR > 1999
        """
        if not self.available:
            return []
        
        all_records = []
        start = 0
        count = 25  # Scopus COMPLETE view max per request
        
        while len(all_records) < max_results:
            self._rate_limit()
            
            params = {
                "query": query,
                "start": start,
                "count": min(count, max_results - len(all_records)),
                "view": "STANDARD",
            }
            
            try:
                resp = requests.get(SCOPUS_SEARCH_URL, headers=self._headers(),
                                   params=params, timeout=30)
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 401:
                    logger.error("Scopus: 401 Unauthorized — check API key and IP/VPN")
                elif resp.status_code == 429:
                    logger.warning("Scopus: rate limited, waiting 5s...")
                    time.sleep(5)
                    continue
                else:
                    logger.error(f"Scopus HTTP error: {e}")
                break
            
            data = resp.json()
            results = data.get("search-results", {})
            total = int(results.get("opensearch:totalResults", 0))
            entries = results.get("entry", [])
            
            if not entries or (len(entries) == 1 and entries[0].get("error")):
                break
            
            for entry in entries:
                record = self._parse_entry(entry)
                if record:
                    all_records.append(record)
            
            start += len(entries)
            
            if start >= total or start >= max_results:
                break
            
            logger.info(f"Scopus: {len(all_records)} / {min(total, max_results)}")
        
        logger.info(f"Scopus search completed: {len(all_records)} records")
        return all_records[:max_results]
    
    def _parse_entry(self, entry: dict) -> Optional[dict]:
        """解析 Scopus search result entry"""
        title = entry.get("dc:title", "")
        if not title:
            return None
        
        # Extract DOI
        doi = entry.get("prism:doi", "")
        
        # Extract PMID from identifier
        pmid = ""
        eid = entry.get("eid", "")
        pubmed_id = entry.get("pubmed-id", "")
        if pubmed_id:
            pmid = pubmed_id
        
        # Authors
        authors = []
        author_str = entry.get("dc:creator", "")
        if author_str:
            authors = [author_str]
        
        # Year
        cover_date = entry.get("prism:coverDate", "")
        year = cover_date[:4] if cover_date else ""
        
        sid = f"PMID_{pmid}" if pmid else f"SCOPUS_{eid}"
        
        return {
            "study_id": sid,
            "pmid": pmid,
            "pmc_id": "",
            "doi": doi,
            "title": title,
            "abstract": entry.get("dc:description", ""),
            "authors": authors,
            "first_author": authors[0] if authors else "",
            "journal": entry.get("prism:publicationName", ""),
            "year": year,
            "mesh_terms": [],
            "publication_types": [entry.get("subtypeDescription", "")],
            "source": "scopus",
            "scopus_eid": eid,
        }
