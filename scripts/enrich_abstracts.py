#!/usr/bin/env python3
"""
Abstract Enrichment — fetch missing abstracts for studies in all_studies.json.
===============================================================================
Run after Phase 2 deduplication if you notice blank abstracts.

  python scripts/enrich_abstracts.py               # dry-run (show counts only)
  python scripts/enrich_abstracts.py --fix         # fetch and patch all_studies.json

Sources tried in order (for each study with a blank abstract):
  1. Elsevier Abstract API — if ELSEVIER_API_KEY is set and study has a DOI
  2. PubMed efetch          — if study has a PMID
  3. PubMed DOI search      — if study has a DOI but no PMID
  4. CrossRef /works/{doi}  — if study has a DOI

Also patches data/phase2_search/prescreened/filtered_studies.json if it exists.
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import sys
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.project import select_project, get_data_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEDUP_JSON = PRESCREENED_JSON = Path(".")


def _init_paths():
    global DEDUP_JSON, PRESCREENED_JSON
    dd = get_data_dir()
    DEDUP_JSON       = Path(dd) / "phase2_search" / "deduplicated" / "all_studies.json"
    PRESCREENED_JSON = Path(dd) / "phase2_search" / "prescreened" / "filtered_studies.json"

EUTILS_BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CROSSREF_BASE = "https://api.crossref.org/works"
ELSEVIER_ABS  = "https://api.elsevier.com/content/abstract/doi"


# ── helpers ──────────────────────────────────────────────────────────────────

def _sleep(secs: float):
    time.sleep(secs)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def fetch_abstract_elsevier(doi: str, elsevier_key: str) -> str:
    """Fetch abstract from Elsevier Abstract Retrieval API by DOI."""
    if not elsevier_key or not doi:
        return ""
    try:
        url = f"{ELSEVIER_ABS}/{doi}"
        headers = {
            "X-ELS-APIKey": elsevier_key,
            "Accept": "application/json",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code in (401, 403, 404):
            logger.debug(f"Elsevier API {resp.status_code} for DOI {doi}")
            return ""
        if resp.status_code == 429:
            logger.warning("Elsevier API rate-limited, sleeping 5s...")
            _sleep(5)
            return ""
        resp.raise_for_status()
        data = resp.json()
        # Navigate the Elsevier response structure
        core = data.get("abstracts-retrieval-response", {}).get("coredata", {})
        abstract = core.get("dc:description", "") or core.get("abstract", "")
        return _strip_html(str(abstract)) if abstract else ""
    except Exception as e:
        logger.debug(f"Elsevier fetch failed for {doi}: {e}")
        return ""


def fetch_abstract_pubmed_pmid(pmid: str, ncbi_key: str = "") -> str:
    """Fetch abstract from PubMed by PMID using efetch."""
    try:
        import xml.etree.ElementTree as ET
        params = {
            "db": "pubmed", "id": pmid,
            "retmode": "xml", "rettype": "abstract",
        }
        if ncbi_key:
            params["api_key"] = ncbi_key
        resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=20)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        parts = []
        for art in root.findall(".//Abstract"):
            for at in art.findall(".//AbstractText"):
                label = at.get("Label", "")
                text = "".join(at.itertext())
                parts.append(f"{label}: {text}" if label else text)
        return " ".join(parts).strip()
    except Exception as e:
        logger.debug(f"PubMed PMID fetch failed for {pmid}: {e}")
        return ""


def fetch_pmid_by_doi(doi: str, ncbi_key: str = "") -> str:
    """Search PubMed for a PMID by DOI."""
    try:
        params = {
            "db": "pubmed", "term": f"{doi}[doi]",
            "retmax": 1, "retmode": "json",
        }
        if ncbi_key:
            params["api_key"] = ncbi_key
        resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=15)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        return ids[0] if ids else ""
    except Exception as e:
        logger.debug(f"PubMed DOI search failed for {doi}: {e}")
        return ""


def fetch_abstract_crossref(doi: str, email: str = "") -> str:
    """Fetch abstract from CrossRef by DOI."""
    try:
        params = {}
        if email:
            params["mailto"] = email
        resp = requests.get(f"{CROSSREF_BASE}/{doi}", params=params, timeout=20)
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        abstract = resp.json().get("message", {}).get("abstract", "")
        return _strip_html(abstract)
    except Exception as e:
        logger.debug(f"CrossRef fetch failed for {doi}: {e}")
        return ""


# ── env ──────────────────────────────────────────────────────────────────────

def load_env_vars():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import os
    return (
        os.getenv("NCBI_API_KEY", ""),
        os.getenv("CROSSREF_EMAIL", ""),
        os.getenv("ELSEVIER_API_KEY", ""),
    )


# ── enrichment logic ─────────────────────────────────────────────────────────

def enrich(studies: list, ncbi_key: str, email: str, elsevier_key: str,
           fix: bool) -> tuple[list, int]:
    """Enrich abstracts in place. Returns (updated_studies, n_filled)."""
    blank = [s for s in studies if not (s.get("abstract") or "").strip()]
    logger.info(f"Studies with blank abstract: {len(blank)} / {len(studies)}")

    if not fix:
        by_src: dict[str, int] = {}
        for s in blank:
            src = s.get("source") or s.get("sources") or "unknown"
            if isinstance(src, list):
                src = src[0] if src else "unknown"
            by_src[src] = by_src.get(src, 0) + 1
        for src, n in sorted(by_src.items(), key=lambda x: -x[1]):
            logger.info(f"  {src}: {n} blank")
        if elsevier_key:
            logger.info(f"Elsevier API key found — will be used as first source.")
        else:
            logger.info("No ELSEVIER_API_KEY — will skip Elsevier source.")
        logger.info("Run with --fix to fetch missing abstracts.")
        return studies, 0

    study_map = {s["study_id"]: s for s in studies}
    n_filled = 0
    pubmed_interval = 0.11 if ncbi_key else 0.35

    # Track Elsevier quota: be conservative — 1 req/s is safe for the free tier
    elsevier_interval = 1.1

    for i, study in enumerate(blank):
        pmid = study.get("pmid", "").strip()
        doi  = study.get("doi", "").strip()
        sid  = study.get("study_id", "")
        abstract = ""

        # 1. Elsevier Abstract API (best for Scopus/Elsevier articles)
        if elsevier_key and doi:
            abstract = fetch_abstract_elsevier(doi, elsevier_key)
            _sleep(elsevier_interval)

        # 2. PubMed by PMID
        if not abstract and pmid:
            abstract = fetch_abstract_pubmed_pmid(pmid, ncbi_key)
            _sleep(pubmed_interval)

        # 3. PubMed by DOI → PMID → abstract
        if not abstract and doi:
            found_pmid = fetch_pmid_by_doi(doi, ncbi_key)
            _sleep(pubmed_interval)
            if found_pmid:
                abstract = fetch_abstract_pubmed_pmid(found_pmid, ncbi_key)
                _sleep(pubmed_interval)
                if abstract and not study_map[sid].get("pmid"):
                    study_map[sid]["pmid"] = found_pmid

        # 4. CrossRef by DOI
        if not abstract and doi:
            abstract = fetch_abstract_crossref(doi, email)
            _sleep(0.25)

        if abstract:
            study_map[sid]["abstract"] = abstract
            n_filled += 1
            if n_filled % 10 == 0:
                logger.info(f"  [{i+1}/{len(blank)}] Enriched {n_filled} abstracts so far...")
        else:
            logger.debug(f"  No abstract found for {sid} (DOI: {doi}, PMID: {pmid})")

    logger.info(f"Abstract enrichment complete: {n_filled} / {len(blank)} filled")
    return list(study_map.values()), n_filled


def patch_file(path: Path, enriched_map: dict):
    """Patch a JSON list file in-place using the enriched map."""
    if not path.exists():
        return
    studies = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for s in studies:
        enriched = enriched_map.get(s.get("study_id"))
        if enriched and enriched.get("abstract") and not (s.get("abstract") or "").strip():
            s["abstract"] = enriched["abstract"]
            changed += 1
    path.write_text(json.dumps(studies, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Patched {changed} abstracts in {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enrich blank abstracts via Elsevier, PubMed & CrossRef")
    parser.add_argument("--fix", action="store_true",
                        help="Fetch and write abstracts (default: dry-run)")
    args = parser.parse_args()
    select_project()
    _init_paths()

    if not DEDUP_JSON.exists():
        logger.error(f"Not found: {DEDUP_JSON}")
        logger.error("Run Phase 2 first: python scripts/run_phase2.py")
        return

    ncbi_key, email, elsevier_key = load_env_vars()
    studies = json.loads(DEDUP_JSON.read_text(encoding="utf-8"))

    enriched, n_filled = enrich(studies, ncbi_key, email, elsevier_key, fix=args.fix)

    if args.fix and n_filled > 0:
        DEDUP_JSON.write_text(
            json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Saved enriched data to {DEDUP_JSON}")

        enriched_map = {s["study_id"]: s for s in enriched}
        patch_file(PRESCREENED_JSON, enriched_map)

        still_blank = sum(1 for s in enriched if not (s.get("abstract") or "").strip())
        logger.info(f"Remaining blank abstracts: {still_blank} / {len(enriched)}")
        if still_blank > 0:
            logger.info(
                "Remaining blanks likely have no abstract in any source "
                "(conference papers, protocols, etc.)."
            )


if __name__ == "__main__":
    main()
