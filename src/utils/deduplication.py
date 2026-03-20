"""
Study Deduplication
====================
多層去重策略:
1. DOI 完全匹配
2. PMID 完全匹配  
3. Title + First Author + Year 模糊匹配

也處理 .ris 檔案匯入 (Cochrane / Embase 手動匯出)
"""

import re
import logging
from typing import List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    """正規化標題用於比對"""
    if not title:
        return ""
    # Lowercase, remove punctuation, collapse whitespace
    t = title.lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def fuzzy_title_match(title1: str, title2: str, threshold: float = 0.90) -> bool:
    """
    模糊標題比對。
    使用 SequenceMatcher (內建) 或 thefuzz (如果可用)。
    
    threshold: 0.90 = 90% 相似度
    """
    t1 = normalize_title(title1)
    t2 = normalize_title(title2)
    
    if not t1 or not t2:
        return False
    
    # Exact match after normalization
    if t1 == t2:
        return True
    
    # Try thefuzz if available (faster, better)
    try:
        from thefuzz import fuzz
        ratio = fuzz.ratio(t1, t2) / 100.0
        return ratio >= threshold
    except ImportError:
        pass
    
    # Fallback: SequenceMatcher (stdlib)
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, t1, t2).ratio()
    return ratio >= threshold


def normalize_doi(doi: str) -> str:
    """正規化 DOI"""
    if not doi:
        return ""
    doi = doi.strip().lower()
    # Remove URL prefix if present
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def deduplicate_studies(studies: List[dict]) -> Tuple[List[dict], dict]:
    """
    去重並回傳結果。
    
    Returns:
        (deduplicated_list, dedup_report)
    """
    seen_dois = {}      # doi -> index
    seen_pmids = {}     # pmid -> index
    seen_titles = {}    # normalized_title -> index
    
    deduplicated = []
    duplicates_found = 0
    merge_log = []
    
    skipped_empty = 0

    for study in studies:
        doi = normalize_doi(study.get("doi", ""))
        pmid = study.get("pmid", "")
        title = normalize_title(study.get("title", ""))
        year = study.get("year", "")
        first_author_last = ""
        if study.get("authors"):
            first_author_last = study["authors"][0].split()[0].lower() if study["authors"] else ""

        # Skip records that have no usable identifier at all — they cannot be
        # deduplicated and will cause problems in downstream phases.
        if not doi and not pmid and not title:
            skipped_empty += 1
            logger.debug(f"Skipping empty record (no doi/pmid/title): {study.get('study_id','?')}")
            continue

        is_duplicate = False
        duplicate_of = None
        
        # Check DOI
        if doi and doi in seen_dois:
            is_duplicate = True
            duplicate_of = seen_dois[doi]
        
        # Check PMID
        elif pmid and pmid in seen_pmids:
            is_duplicate = True
            duplicate_of = seen_pmids[pmid]
        
        # Check Title + Author + Year
        elif title:
            title_key = f"{title}|{first_author_last}|{year}"
            if title_key in seen_titles:
                is_duplicate = True
                duplicate_of = seen_titles[title_key]
            # Also check with fuzzy: just title if very long (>10 words)
            elif len(title.split()) > 10:
                # Check if title alone matches
                for existing_key, idx in seen_titles.items():
                    existing_title = existing_key.split("|")[0]
                    if title == existing_title:
                        is_duplicate = True
                        duplicate_of = idx
                        break
        
        if is_duplicate:
            duplicates_found += 1
            # Merge: keep the record with more information
            existing = deduplicated[duplicate_of]
            merged = _merge_records(existing, study)
            deduplicated[duplicate_of] = merged
            merge_log.append({
                "kept": existing.get("study_id"),
                "merged_from": study.get("study_id"),
                "source": study.get("source"),
            })
        else:
            idx = len(deduplicated)
            if doi:
                seen_dois[doi] = idx
            if pmid:
                seen_pmids[pmid] = idx
            if title:
                title_key = f"{title}|{first_author_last}|{year}"
                seen_titles[title_key] = idx
            deduplicated.append(study)
    
    report = {
        "total_input": len(studies),
        "after_dedup": len(deduplicated),
        "duplicates_removed": duplicates_found,
        "skipped_empty_records": skipped_empty,
        "merge_log": merge_log[:50],  # keep first 50 for reference
    }

    if skipped_empty:
        logger.warning(
            f"Skipped {skipped_empty} records with no doi/pmid/title "
            f"(likely malformed imports)"
        )
    logger.info(
        f"Deduplication: {len(studies)} → {len(deduplicated)} "
        f"({duplicates_found} duplicates removed)"
    )
    
    return deduplicated, report


def _merge_records(existing: dict, new: dict) -> dict:
    """
    合併兩筆記錄，保留更完整的資料。
    """
    merged = dict(existing)
    
    # Fill in missing fields
    for key in ["doi", "pmid", "pmc_id", "abstract"]:
        if not merged.get(key) and new.get(key):
            merged[key] = new[key]
    
    # Track all sources
    existing_sources = merged.get("sources", [merged.get("source", "")])
    new_source = new.get("source", "")
    if new_source and new_source not in existing_sources:
        existing_sources.append(new_source)
    merged["sources"] = existing_sources
    
    # Keep longer abstract
    if len(new.get("abstract", "")) > len(merged.get("abstract", "")):
        merged["abstract"] = new["abstract"]
    
    return merged


def parse_ris_file(filepath: str) -> List[dict]:
    """
    解析 .ris 檔案 (Cochrane / Embase 匯出格式)。
    
    Returns:
        list of study dicts (same format as PubMed records)
    """
    try:
        import rispy
    except ImportError:
        logger.error("rispy not installed. Run: pip install rispy")
        return []
    
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"RIS file not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        entries = rispy.load(f)
    
    records = []
    source_name = filepath.stem.split("_")[0]  # e.g., "cochrane" from "cochrane_manual.ris"
    
    for entry in entries:
        # RIS field mapping
        authors = entry.get("authors", entry.get("first_authors", []))
        if isinstance(authors, str):
            authors = [authors]
        
        title = entry.get("title", entry.get("primary_title", ""))
        year = str(entry.get("year", entry.get("publication_year", "")))
        doi = entry.get("doi", "")
        
        # Try to extract DOI from notes or URLs
        if not doi:
            for field in ["notes", "url", "link"]:
                val = entry.get(field, "")
                if isinstance(val, list):
                    val = " ".join(val)
                doi_match = re.search(r'10\.\d{4,}/[^\s]+', str(val))
                if doi_match:
                    doi = doi_match.group()
                    break
        
        record = {
            "study_id": f"RIS_{source_name}_{len(records)}",
            "pmid": "",
            "pmc_id": "",
            "doi": doi,
            "title": title,
            "abstract": entry.get("abstract", entry.get("notes_abstract", "")),
            "authors": authors,
            "first_author": authors[0] if authors else "",
            "journal": entry.get("journal_name", entry.get("secondary_title", "")),
            "year": year,
            "mesh_terms": entry.get("keywords", []),
            "publication_types": [],
            "source": source_name,
        }
        
        records.append(record)
    
    logger.info(f"Parsed {len(records)} records from {filepath.name}")
    return records
