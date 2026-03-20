"""
CrossRef API Wrapper — v5
==========================
- query param (not query.bibliographic)
- No select param
- mailto for polite pool
- Total results logging
"""

import re
import time
import logging
from typing import Optional
import requests

from src.config import cfg

logger = logging.getLogger(__name__)

CROSSREF_BASE = "https://api.crossref.org"
MAX_ROWS_PER_REQUEST = 200


class CrossRefAPI:
    
    def __init__(self):
        self.email = cfg.crossref_email
        self.min_interval = 0.2 if self.email else 1.5
        self._last_request_time = 0
    
    @property
    def available(self) -> bool:
        return True
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()
    
    def search(self, params: dict, max_results: int = 500) -> list:
        all_records = []
        offset = 0
        batch_size = min(MAX_ROWS_PER_REQUEST, max_results)
        first_batch = True
        
        while len(all_records) < max_results:
            self._rate_limit()
            
            request_params = dict(params)
            request_params["offset"] = offset
            request_params["rows"] = min(batch_size, max_results - len(all_records))
            if self.email:
                request_params["mailto"] = self.email
            
            try:
                resp = requests.get(f"{CROSSREF_BASE}/works",
                                   params=request_params, timeout=45)
                if resp.status_code == 400:
                    logger.warning(f"CrossRef 400. URL: {resp.url[:300]}")
                    break
                resp.raise_for_status()
            except requests.exceptions.Timeout:
                logger.warning(f"CrossRef timeout at offset {offset}")
                break
            except Exception as e:
                logger.warning(f"CrossRef error: {e}")
                break
            
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            total = data.get("message", {}).get("total-results", 0)
            
            # First batch: log total
            if first_batch:
                logger.info(f"CrossRef Total Results: {total}")
                first_batch = False
            
            if not items:
                break
            
            for item in items:
                record = self._parse_item(item)
                if record:
                    all_records.append(record)
            
            offset += len(items)
            if offset >= total or offset >= max_results:
                break
            logger.info(f"CrossRef: {len(all_records)} / {min(total, max_results)}")
        
        logger.info(f"CrossRef search completed: {len(all_records)} records")
        return all_records[:max_results]
    
    def _parse_item(self, item: dict) -> Optional[dict]:
        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""
        if not title:
            return None
        doi = item.get("DOI", "")
        authors = []
        for a in item.get("author", []):
            name = f"{a.get('family', '')} {a.get('given', '')}".strip()
            if name:
                authors.append(name)
        year = ""
        for df in ["published-print", "published-online", "created"]:
            dp = item.get(df, {}).get("date-parts", [[]])
            if dp and dp[0]:
                year = str(dp[0][0])
                break
        jl = item.get("container-title", [])
        abstract = item.get("abstract", "")
        if abstract:
            abstract = re.sub(r'<[^>]+>', '', abstract)
        return {
            "study_id": f"DOI_{doi.replace('/', '_')}" if doi else f"CR_{hash(title) % 100000}",
            "pmid": "", "pmc_id": "", "doi": doi,
            "title": title, "abstract": abstract,
            "authors": authors, "first_author": authors[0] if authors else "",
            "journal": jl[0] if jl else "", "year": year,
            "mesh_terms": [], "publication_types": [item.get("type", "")],
            "source": "crossref", "language": item.get("language", ""),
        }
