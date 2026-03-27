"""
PDF Downloader — LUMEN v2
============================
Multi-source cascade PDF downloader.
Sources: Unpaywall -> PMC -> EPMC -> OpenAlex -> Semantic Scholar -> Elsevier -> CrossRef -> DOI redirect
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Download sources in priority order
DOWNLOAD_SOURCES = [
    "unpaywall",
    "pmc",
    "europepmc",
    "openalex",
    "semantic_scholar",
    "elsevier",
    "crossref",
    "doi_redirect",
    "web_search",
]


class PDFDownloader:
    """Multi-source PDF downloader with cascade fallback."""

    def __init__(self, output_dir: str, email: str = ""):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.email = email or "lumen@example.com"
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": f"LUMEN/2.0 (mailto:{self.email})"},
        )
        self.stats = {"success": 0, "failed": 0, "cached": 0, "sources": {}}

    def download(self, study: dict) -> Optional[str]:
        """
        Try to download PDF for a study.
        Returns the file path if successful, None otherwise.
        """
        study_id = study.get("study_id", "unknown")
        doi = study.get("doi", "")
        pmid = study.get("pmid", "")
        pmcid = study.get("pmcid", "")

        # Check if already downloaded
        pdf_path = self.output_dir / f"{study_id}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            self.stats["cached"] += 1
            return str(pdf_path)

        for source in DOWNLOAD_SOURCES:
            try:
                # Elsevier needs special handling (direct API download)
                if source == "elsevier" and doi:
                    if self._elsevier_download(doi, pdf_path):
                        self.stats["success"] += 1
                        self.stats["sources"][source] = self.stats["sources"].get(source, 0) + 1
                        logger.info(f"Downloaded {study_id} via {source}")
                        return str(pdf_path)
                    continue

                # Web search needs special handling (title-based search)
                if source == "web_search":
                    title = study.get("title", "")
                    if title and self._web_search_download(title, doi, pdf_path):
                        self.stats["success"] += 1
                        self.stats["sources"][source] = self.stats["sources"].get(source, 0) + 1
                        logger.info(f"Downloaded {study_id} via {source}")
                        return str(pdf_path)
                    continue

                url = self._get_pdf_url(source, doi, pmid, pmcid)
                if not url:
                    continue

                if self._download_from_url(url, pdf_path):
                    self.stats["success"] += 1
                    self.stats["sources"][source] = self.stats["sources"].get(source, 0) + 1
                    logger.info(f"Downloaded {study_id} via {source}")
                    return str(pdf_path)

            except Exception as e:
                logger.debug(f"Failed {source} for {study_id}: {e}")
                continue

            time.sleep(0.5)

        self.stats["failed"] += 1
        logger.warning(f"Could not download PDF for {study_id}")
        return None

    def _download_from_url(self, url: str, pdf_path: Path) -> bool:
        """Download a PDF from a URL, return True if successful."""
        response = self.client.get(url)
        if response.status_code == 200 and len(response.content) > 1000:
            content_type = response.headers.get("content-type", "")
            if "pdf" in content_type or response.content[:5] == b"%PDF-":
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
                return True
        return False

    def _get_pdf_url(self, source: str, doi: str, pmid: str, pmcid: str) -> Optional[str]:
        """Get download URL for a given source."""

        if source == "unpaywall" and doi:
            return self._unpaywall_lookup(doi)

        elif source == "pmc" and pmcid:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

        elif source == "europepmc" and pmid:
            # Try to find PMC PDF via Europe PMC
            return self._europepmc_lookup(pmid)

        elif source == "openalex" and (doi or pmid):
            return self._openalex_lookup(doi, pmid)

        elif source == "semantic_scholar" and doi:
            return self._semantic_scholar_lookup(doi)

        elif source == "doi_redirect" and doi:
            return f"https://doi.org/{doi}"

        return None

    def _unpaywall_lookup(self, doi: str) -> Optional[str]:
        """Get OA PDF URL from Unpaywall."""
        try:
            resp = self.client.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": self.email},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            best = data.get("best_oa_location", {})
            if best:
                return best.get("url_for_pdf") or best.get("url")
            for loc in data.get("oa_locations", []):
                url = loc.get("url_for_pdf") or loc.get("url")
                if url:
                    return url
        except Exception:
            pass
        return None

    def _europepmc_lookup(self, pmid: str) -> Optional[str]:
        """Try Europe PMC for PDF URL."""
        try:
            resp = self.client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": f"ext_id:{pmid}", "format": "json"},
            )
            if resp.status_code != 200:
                return None
            results = resp.json().get("resultList", {}).get("result", [])
            for r in results:
                pmcid = r.get("pmcid", "")
                if pmcid:
                    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        except Exception:
            pass
        return None

    def _openalex_lookup(self, doi: str, pmid: str) -> Optional[str]:
        """Get OA PDF URL from OpenAlex."""
        try:
            from src.apis.openalex import get_pdf_url
            return get_pdf_url(doi=doi, pmid=pmid, email=self.email)
        except Exception:
            pass
        return None

    def _semantic_scholar_lookup(self, doi: str) -> Optional[str]:
        """Get PDF URL from Semantic Scholar."""
        try:
            resp = self.client.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                params={"fields": "openAccessPdf"},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            oa_pdf = data.get("openAccessPdf", {})
            if oa_pdf:
                return oa_pdf.get("url")
        except Exception:
            pass
        return None

    def _elsevier_download(self, doi: str, pdf_path: Path) -> bool:
        """Download PDF directly from Elsevier API (if key available)."""
        api_key = os.getenv("ELSEVIER_API_KEY", "")
        if not api_key:
            return False
        try:
            resp = self.client.get(
                f"https://api.elsevier.com/content/article/doi/{doi}",
                headers={
                    "X-ELS-APIKey": api_key,
                    "Accept": "application/pdf",
                },
            )
            if resp.status_code == 200 and len(resp.content) > 1000:
                content_type = resp.headers.get("content-type", "")
                if "pdf" in content_type or resp.content[:5] == b"%PDF-":
                    with open(pdf_path, "wb") as f:
                        f.write(resp.content)
                    return True
        except Exception:
            pass
        return False

    def _web_search_download(self, title: str, doi: str, pdf_path: Path) -> bool:
        """
        Last-resort: search Google Scholar / web for a PDF link.
        Uses a simple title-based query to find open-access copies.
        """
        import re
        import urllib.parse

        # Clean title for search
        clean_title = re.sub(r'[^\w\s]', '', title)[:120].strip()
        if not clean_title:
            return False

        # Strategy 1: Google Scholar search for PDF links
        queries = []
        if doi:
            queries.append(f"{doi} filetype:pdf")
        queries.append(f'"{clean_title}" filetype:pdf')

        for query in queries:
            try:
                encoded = urllib.parse.quote(query)
                resp = self.client.get(
                    f"https://scholar.google.com/scholar?q={encoded}",
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/120.0.0.0 Safari/537.36",
                    },
                    timeout=15.0,
                )
                if resp.status_code != 200:
                    continue

                # Extract PDF links from Google Scholar HTML
                pdf_urls = re.findall(
                    r'href="(https?://[^"]+\.pdf[^"]*)"', resp.text
                )
                # Also look for [PDF] link patterns
                pdf_urls += re.findall(
                    r'href="(https?://[^"]+)"[^>]*>\[PDF\]', resp.text
                )

                for pdf_url in pdf_urls[:3]:  # Try top 3 results
                    if self._download_from_url(pdf_url, pdf_path):
                        return True
                    time.sleep(0.5)

            except Exception as e:
                logger.debug(f"Web search failed for '{clean_title[:40]}': {e}")
                continue

            time.sleep(1.0)  # Rate limit between queries

        # Strategy 2: Try ResearchGate / Academia.edu via DOI title search
        if doi:
            for base in [
                "https://www.researchgate.net/publication/",
                "https://www.academia.edu/search?q=",
            ]:
                try:
                    resp = self.client.get(
                        f"{base}{urllib.parse.quote(clean_title[:80])}",
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        pdf_urls = re.findall(
                            r'href="(https?://[^"]+\.pdf[^"]*)"', resp.text
                        )
                        for pdf_url in pdf_urls[:2]:
                            if self._download_from_url(pdf_url, pdf_path):
                                return True
                except Exception:
                    pass
                time.sleep(0.5)

        return False

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
