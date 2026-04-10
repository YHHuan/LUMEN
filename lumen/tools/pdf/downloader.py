"""
PDF Downloader — LUMEN v3
=========================
Multi-source cascade PDF downloader.
Sources: Unpaywall → PMC → EuropePMC → Semantic Scholar → DOI redirect
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger()

DOWNLOAD_SOURCES = [
    "unpaywall",
    "pmc",
    "europepmc",
    "semantic_scholar",
    "doi_redirect",
]


class PDFDownloader:
    """Multi-source PDF downloader with cascade fallback."""

    def __init__(self, output_dir: str, email: str = ""):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.email = email or os.getenv("CONTACT_EMAIL", "lumen@example.com")
        self.s2_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": f"LUMEN/3.0 (mailto:{self.email})"},
        )
        self.stats: dict = {"success": 0, "failed": 0, "cached": 0, "sources": {}}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, study: dict) -> Optional[str]:
        """
        Try to download PDF for a study dict.

        Expected keys: study_id, doi, pmid, pmcid (all optional).
        Returns the file path on success, None otherwise.
        """
        study_id = study.get("study_id", "unknown")
        doi = study.get("doi", "")
        pmid = study.get("pmid", "")
        pmcid = study.get("pmcid", "")

        # Check cache
        pdf_path = self.output_dir / f"{study_id}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            self.stats["cached"] += 1
            return str(pdf_path)

        for source in DOWNLOAD_SOURCES:
            try:
                url = self._get_pdf_url(source, doi, pmid, pmcid)
                if not url:
                    continue

                if self._download_from_url(url, pdf_path):
                    self._record_success(study_id, source)
                    return str(pdf_path)

            except Exception as exc:
                logger.debug(
                    "source_failed",
                    source=source,
                    study_id=study_id,
                    error=str(exc),
                )
                continue

            time.sleep(0.5)

        self.stats["failed"] += 1
        logger.warning("pdf_download_failed", study_id=study_id)
        return None

    def download_by_doi(self, doi: str, filename: str | None = None) -> Optional[str]:
        """Convenience: download a single PDF by DOI."""
        safe_name = (filename or doi.replace("/", "_")) + ".pdf"
        study = {"study_id": safe_name.removesuffix(".pdf"), "doi": doi}
        return self.download(study)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record_success(self, study_id: str, source: str) -> None:
        self.stats["success"] += 1
        self.stats["sources"][source] = self.stats["sources"].get(source, 0) + 1
        logger.info("pdf_downloaded", study_id=study_id, source=source)

    def _download_from_url(self, url: str, pdf_path: Path) -> bool:
        """Download a PDF from *url*, return True if the file looks valid."""
        resp = self.client.get(url)
        if resp.status_code == 200 and len(resp.content) > 1000:
            content_type = resp.headers.get("content-type", "")
            if "pdf" in content_type or resp.content[:5] == b"%PDF-":
                pdf_path.write_bytes(resp.content)
                return True
        return False

    def _get_pdf_url(
        self, source: str, doi: str, pmid: str, pmcid: str
    ) -> Optional[str]:
        if source == "unpaywall" and doi:
            return self._unpaywall_lookup(doi)

        if source == "pmc" and pmcid:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

        if source == "europepmc" and pmid:
            return self._europepmc_lookup(pmid)

        if source == "semantic_scholar" and doi:
            return self._semantic_scholar_lookup(doi)

        if source == "doi_redirect" and doi:
            return f"https://doi.org/{doi}"

        return None

    # ------------------------------------------------------------------
    # Source-specific lookups
    # ------------------------------------------------------------------

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
            best = data.get("best_oa_location") or {}
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
        """Get PMC PDF URL via Europe PMC search."""
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

    def _semantic_scholar_lookup(self, doi: str) -> Optional[str]:
        """Get open-access PDF URL from Semantic Scholar."""
        try:
            headers: dict[str, str] = {}
            if self.s2_api_key:
                headers["x-api-key"] = self.s2_api_key
            resp = self.client.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                params={"fields": "openAccessPdf"},
                headers=headers,
            )
            if resp.status_code != 200:
                return None
            oa_pdf = resp.json().get("openAccessPdf") or {}
            return oa_pdf.get("url")
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "PDFDownloader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
