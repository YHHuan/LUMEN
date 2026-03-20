"""
PubMed E-utilities API Wrapper
===============================
使用 NCBI E-utilities 搜尋 PubMed。
- esearch: 搜尋取得 PMID 列表
- efetch:  用 PMID 取得詳細資料 (title, abstract, MeSH, etc.)
- 自動 rate limiting (有 API key: 10 req/s; 無: 3 req/s)
"""

import os
import time
import logging
from typing import Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedAPI:
    
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_key = os.getenv("NCBI_API_KEY")
        self.email = os.getenv("NCBI_EMAIL", "")
        
        # Rate limit: 10/s with key, 3/s without
        self.min_interval = 0.1 if self.api_key else 0.34
        self._last_request_time = 0
    
    def _rate_limit(self):
        """確保不超過 rate limit"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _base_params(self) -> dict:
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    def search(self, query: str, max_results: int = 5000,
               date_start: str = "", date_end: str = "") -> list:
        """
        搜尋 PubMed，回傳 PMID 列表。
        
        Args:
            query: PubMed search query (支援 boolean operators)
            max_results: 最大結果數
            date_start: "YYYY/MM/DD"
            date_end: "YYYY/MM/DD"
        
        Returns:
            list of PMID strings
        """
        self._rate_limit()
        
        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 10000),
            "retmode": "json",
            "sort": "relevance",
        }
        if date_start:
            params["mindate"] = date_start.replace("-", "/")
        if date_end:
            params["maxdate"] = date_end.replace("-", "/")
        if date_start or date_end:
            params["datetype"] = "pdat"
        
        resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        result = data.get("esearchresult", {})
        pmids = result.get("idlist", [])
        total_count = int(result.get("count", 0))
        
        logger.info(f"PubMed search found {total_count} results, retrieved {len(pmids)} PMIDs")
        
        # If more results exist, paginate
        if total_count > len(pmids) and max_results > len(pmids):
            pmids = self._paginate_search(query, total_count, max_results, 
                                           date_start, date_end, pmids)
        
        return pmids
    
    def _paginate_search(self, query, total_count, max_results, 
                         date_start, date_end, initial_pmids):
        """分頁取得所有 PMID"""
        all_pmids = list(initial_pmids)
        batch_size = 10000
        
        while len(all_pmids) < min(total_count, max_results):
            self._rate_limit()
            
            params = {
                **self._base_params(),
                "db": "pubmed",
                "term": query,
                "retstart": len(all_pmids),
                "retmax": batch_size,
                "retmode": "json",
            }
            if date_start:
                params["mindate"] = date_start.replace("-", "/")
            if date_end:
                params["maxdate"] = date_end.replace("-", "/")
            if date_start or date_end:
                params["datetype"] = "pdat"
            
            resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            new_pmids = data.get("esearchresult", {}).get("idlist", [])
            if not new_pmids:
                break
            all_pmids.extend(new_pmids)
            logger.info(f"PubMed pagination: {len(all_pmids)} / {total_count}")
        
        return all_pmids[:max_results]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    def fetch_details(self, pmids: list, batch_size: int = 200) -> list:
        """
        用 PMID 列表取得文獻詳細資料。
        
        Returns:
            list of dict, each containing:
            {study_id, title, abstract, authors, journal, year, doi, 
             mesh_terms, publication_type, pmid, pmc_id}
        """
        all_records = []
        
        for i in range(0, len(pmids), batch_size):
            self._rate_limit()
            batch = pmids[i:i+batch_size]
            
            params = {
                **self._base_params(),
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            }
            
            resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=60)
            resp.raise_for_status()
            
            records = self._parse_xml(resp.text)
            all_records.extend(records)
            
            logger.info(f"Fetched details: {len(all_records)} / {len(pmids)}")
        
        return all_records
    
    def _parse_xml(self, xml_text: str) -> list:
        """解析 PubMed XML 回覆"""
        import xml.etree.ElementTree as ET
        
        records = []
        root = ET.fromstring(xml_text)
        
        for article in root.findall('.//PubmedArticle'):
            try:
                record = self._parse_article(article)
                if record:
                    records.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
                continue
        
        return records
    
    def _parse_article(self, article) -> Optional[dict]:
        """解析單篇文獻"""
        medline = article.find('.//MedlineCitation')
        if medline is None:
            return None
        
        pmid_elem = medline.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else ""
        
        art = medline.find('.//Article')
        if art is None:
            return None
        
        # Title
        title_elem = art.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else ""
        
        # Abstract
        abstract_parts = []
        abstract_elem = art.find('.//Abstract')
        if abstract_elem is not None:
            for abs_text in abstract_elem.findall('.//AbstractText'):
                label = abs_text.get('Label', '')
                text = ''.join(abs_text.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts)
        
        # Authors
        authors = []
        for author in art.findall('.//AuthorList/Author'):
            last = author.find('LastName')
            first = author.find('ForeName')
            if last is not None:
                name = last.text
                if first is not None:
                    name += f" {first.text}"
                authors.append(name)
        
        # Journal
        journal_elem = art.find('.//Journal/Title')
        journal = journal_elem.text if journal_elem is not None else ""
        
        # Year
        year = ""
        year_elem = art.find('.//Journal/JournalIssue/PubDate/Year')
        if year_elem is not None:
            year = year_elem.text
        else:
            medline_date = art.find('.//Journal/JournalIssue/PubDate/MedlineDate')
            if medline_date is not None and medline_date.text:
                year = medline_date.text[:4]
        
        # DOI
        doi = ""
        for id_elem in article.findall('.//PubmedData/ArticleIdList/ArticleId'):
            if id_elem.get('IdType') == 'doi':
                doi = id_elem.text
                break
        
        # PMC ID
        pmc_id = ""
        for id_elem in article.findall('.//PubmedData/ArticleIdList/ArticleId'):
            if id_elem.get('IdType') == 'pmc':
                pmc_id = id_elem.text
                break
        
        # MeSH terms
        mesh_terms = []
        for mesh in medline.findall('.//MeshHeadingList/MeshHeading'):
            descriptor = mesh.find('DescriptorName')
            if descriptor is not None:
                mesh_terms.append(descriptor.text)
        
        # Publication types
        pub_types = []
        for pt in art.findall('.//PublicationTypeList/PublicationType'):
            if pt.text:
                pub_types.append(pt.text)
        
        return {
            "study_id": f"PMID_{pmid}",
            "pmid": pmid,
            "pmc_id": pmc_id,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "first_author": authors[0] if authors else "",
            "journal": journal,
            "year": year,
            "mesh_terms": mesh_terms,
            "publication_types": pub_types,
            "source": "pubmed",
        }
    
    def search_and_fetch(self, query: str, max_results: int = 5000,
                         date_start: str = "", date_end: str = "") -> list:
        """
        一步到位：搜尋 + 取得詳細資料。
        """
        pmids = self.search(query, max_results, date_start, date_end)
        if not pmids:
            return []
        return self.fetch_details(pmids)
