"""
Unpaywall & CrossRef API Wrappers
==================================
用於取得開放取用的全文 PDF 連結。

Unpaywall: 透過 DOI 查詢是否有免費全文
CrossRef:  透過 DOI 取得文獻 metadata
"""

import os
import time
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class UnpaywallAPI:
    """
    Unpaywall: 查詢論文是否有免費開放取用版本。
    需要提供 email (不需要 API key)。
    """
    
    BASE_URL = "https://api.unpaywall.org/v2"
    
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        self.email = os.getenv("UNPAYWALL_EMAIL", "")
        self.min_interval = 0.1  # 10 req/s limit
        self._last_request_time = 0
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
    def find_open_access(self, doi: str) -> dict:
        """
        查詢 DOI 的開放取用狀態。
        
        Returns:
            {
                "is_oa": bool,
                "best_oa_url": str or None,
                "oa_status": str,  # "gold", "green", "hybrid", "bronze", "closed"
                "pdf_url": str or None,
            }
        """
        if not doi or not self.email:
            return {"is_oa": False, "best_oa_url": None, "oa_status": "unknown", "pdf_url": None}
        
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{doi}"
        params = {"email": self.email}
        
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 404:
                return {"is_oa": False, "best_oa_url": None, "oa_status": "not_found", "pdf_url": None}
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Unpaywall lookup failed for {doi}: {e}")
            return {"is_oa": False, "best_oa_url": None, "oa_status": "error", "pdf_url": None}
        
        best_location = data.get("best_oa_location") or {}
        
        return {
            "is_oa": data.get("is_oa", False),
            "best_oa_url": best_location.get("url"),
            "oa_status": data.get("oa_status", "closed"),
            "pdf_url": best_location.get("url_for_pdf"),
        }


class CrossRefAPI:
    """
    CrossRef: 透過 DOI 取得文獻 metadata。
    可選提供 email 以獲得更高速率。
    """
    
    BASE_URL = "https://api.crossref.org"
    
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        self.email = os.getenv("CROSSREF_EMAIL", "")
        self.min_interval = 0.2  # ~5 req/s
        self._last_request_time = 0
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _headers(self):
        headers = {"Accept": "application/json"}
        if self.email:
            headers["User-Agent"] = f"MetaAnalysisGenerator/1.0 (mailto:{self.email})"
        return headers
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
    def get_metadata(self, doi: str) -> dict:
        """透過 DOI 取得 metadata"""
        if not doi:
            return {}
        
        self._rate_limit()
        
        url = f"{self.BASE_URL}/works/{doi}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=15)
            if resp.status_code == 404:
                return {}
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"CrossRef lookup failed for {doi}: {e}")
            return {}
        
        message = data.get("message", {})
        
        title_list = message.get("title", [])
        title = title_list[0] if title_list else ""
        
        authors = []
        for a in message.get("author", []):
            name = f"{a.get('family', '')} {a.get('given', '')}".strip()
            if name:
                authors.append(name)
        
        # Year
        year = ""
        published = message.get("published-print") or message.get("published-online") or {}
        date_parts = published.get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = str(date_parts[0][0])
        
        return {
            "doi": doi,
            "title": title,
            "authors": authors,
            "first_author": authors[0] if authors else "",
            "year": year,
            "journal": ", ".join(message.get("container-title", [])),
            "type": message.get("type", ""),
            "references_count": message.get("references-count", 0),
        }
