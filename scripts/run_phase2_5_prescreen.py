#!/usr/bin/env python3
"""
Phase 2.5: Keyword-based Pre-screening (Zero Token Cost)
==========================================================
Uses Phase 1 prescreen_exclusion_keywords plus structural rules to filter
obvious non-relevant records before costly LLM screening.
Zero LLM token cost.

Input:  data/phase2_search/deduplicated/all_studies.json
Output: data/phase2_search/prescreened/filtered_studies.json
        data/phase2_search/prescreened/prescreen_excluded_log.json

Usage:
  python scripts/run_phase2_5_prescreen.py
"""

import sys
import json
import logging
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project
from src.utils.file_handlers import DataManager
from src.utils.deduplication import deduplicate_studies

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# Animal/in-vitro keywords: check TITLE only (abstract might say "unlike animal studies")
TITLE_ONLY_KEYWORDS = {
    "rat", "rats", "mouse", "mice", "animal", "animal model",
    "dog", "dogs", "pig", "pigs", "monkey", "monkeys",
    "rabbit", "rabbits", "feline", "canine", "murine",
    "in vitro", "cell culture", "cell line",
}

# ── Structural exclusion rules ────────────────────────────────────────────────

# DOI prefixes that indicate trial registry entries, not publications
_REGISTRY_DOI_PREFIXES = (
    "10.31525/ct1-",   # ClinicalTrials.gov via Crossref
)

# first_author patterns that indicate registry entries (Cochrane imports)
_REGISTRY_AUTHOR_RE = re.compile(
    r'^(NCT|DRKS|ChiCTR|ACTRN|ISRCTN|EUCTR|NL-OMON|UMIN|CTRI|KCT)\d',
    re.IGNORECASE,
)

# Explicit publication_types values that mark non-primary-study records
_EXCLUDE_PUB_TYPES = {
    "clinical trial protocol", "preregistered study",
    "trial registration", "systematic review", "meta-analysis",
    "review", "editorial", "letter", "comment", "news",
    "retracted publication", "erratum", "corrigendum",
    "meeting abstract", "congress", "conference paper",
    "preprint",
}

# Title-level patterns that strongly indicate non-primary-study records
# Checked as ^-anchored patterns (beginning of title) or standalone phrases
_TITLE_EXCLUSION_RE = re.compile(
    r'(?i)^(re:|letter:|letter to|correspondence:|comment on|'
    r'response to|reply to|erratum|corrigendum|correction to|'
    r'retraction of|author.?s reply|commentary[: ])',
)

# Journal/source keywords that indicate conference supplement issues
_CONFERENCE_JOURNAL_RE = re.compile(
    r'\b(supplement|suppl|abstract|poster|proceeding|congress|meeting)\b',
    re.IGNORECASE,
)


def is_structural_exclusion(study: dict) -> tuple[bool, str]:
    """Return (True, reason) if a study can be excluded on structural grounds
    without any keyword matching."""

    doi         = (study.get("doi") or "").strip().lower()
    first_author = (study.get("first_author") or "").strip()
    pub_types   = [pt.lower().strip() for pt in (study.get("publication_types") or [])]
    journal     = (study.get("journal") or "").lower()
    abstract    = (study.get("abstract") or "")
    title       = (study.get("title") or "").strip()

    # 1. DOI prefix → ClinicalTrials.gov registration record
    if any(doi.startswith(p) for p in _REGISTRY_DOI_PREFIXES):
        return True, f"Registry DOI prefix ({doi[:25]})"

    # 2. first_author is a registry ID (common in Cochrane RIS exports)
    if _REGISTRY_AUTHOR_RE.match(first_author):
        return True, f"Registry ID as author ({first_author[:25]})"

    # 3. Explicit publication_type
    for pt in pub_types:
        if pt in _EXCLUDE_PUB_TYPES:
            return True, f"Publication type: '{pt}'"

    # 4. Title-level pattern (letter / commentary / erratum / etc.)
    if title and _TITLE_EXCLUSION_RE.match(title):
        return True, f"Title pattern: '{title[:50]}'"

    # 5. Extremely short title (≤ 10 chars) — registry entry or truncated garbage
    if title and len(title) <= 10:
        return True, f"Title too short ({len(title)} chars): '{title}'"

    # 6. No abstract AND no identifiers → likely registry or incomplete record
    has_abstract   = len(abstract.strip()) > 30
    has_identifier = bool(doi) or bool(study.get("pmid")) or bool(study.get("arxiv_id"))
    if not has_abstract and not has_identifier:
        return True, "No abstract and no DOI/PMID — likely incomplete or registry record"

    # 7. Journal name indicates conference supplement — exclude regardless of abstract length
    #    Previously required abstract < 300 chars, but many conference abstracts have
    #    longer abstracts that still lack full data. Relaxed per validation findings:
    #    31/126 minimal studies were conference abstracts that passed the old threshold.
    if _CONFERENCE_JOURNAL_RE.search(journal):
        return True, f"Conference/supplement journal (journal='{journal[:40]}')"

    return False, ""


# Keep the old name for backwards compatibility
is_registry_or_conference = is_structural_exclusion


def main():
    select_project()
    dm = DataManager()

    # 1. Load data
    if not dm.exists("phase2_search", "all_studies.json", subfolder="deduplicated"):
        logger.error("Phase 2 data not found. Run scripts/run_phase2.py first.")
        sys.exit(1)
    
    studies = dm.load("phase2_search", "all_studies.json", subfolder="deduplicated")
    strategy = dm.load("phase1_strategy", "search_strategy.json")
    
    # 2. Get exclusion keywords
    default_exclusions = [
        "rat", "rats", "mouse", "mice", "animal", "animal model",
        "review", "meta-analysis", "systematic review",
        "editorial", "commentary", "letter", "erratum",
        "protocol", "case report", "case series",
        "in vitro", "cell culture",
    ]
    exclusion_keywords = strategy.get("prescreen_exclusion_keywords", default_exclusions)
    exclusion_keywords = [k.lower().strip() for k in exclusion_keywords if k]

    # 2b. Load study design filter (generated by Phase 1 Strategist)
    design_filter = {}
    if dm.exists("phase1_strategy", "study_design_filter.json"):
        design_filter = dm.load("phase1_strategy", "study_design_filter.json")
        logger.info(f"Study design filter: mode={design_filter.get('filter_mode', 'N/A')}, "
                     f"designs={design_filter.get('required_designs', [])}")
    else:
        # Backward compatibility: check if embedded in search_strategy.json
        design_filter = strategy.get("study_design_filter", {})
        if design_filter:
            logger.info(f"Study design filter (from strategy): mode={design_filter.get('filter_mode', 'N/A')}")

    filter_mode = design_filter.get("filter_mode", "")
    positive_kw = [k.lower() for k in design_filter.get("positive_keywords", [])]
    design_excl_kw = [k.lower() for k in design_filter.get("design_exclusion_keywords", [])]

    logger.info(f"Loaded {len(studies)} studies.")
    logger.info(f"Exclusion keywords ({len(exclusion_keywords)}): {exclusion_keywords}")
    if design_filter:
        logger.info(f"Design positive keywords: {positive_kw}")
        logger.info(f"Design exclusion keywords: {design_excl_kw}")
    
    # 3. Filter
    kept = []
    excluded = []
    
    for study in studies:
        title = study.get("title", "").lower()
        title_abstract = (title + " " + study.get("abstract", "")).lower()

        is_excluded = False
        reason = ""

        # ── Structural checks first (registry / conference / incomplete) ──
        is_excluded, reason = is_structural_exclusion(study)

        # ── Study design filter (from Phase 1 Strategist) ──
        if not is_excluded and design_filter and filter_mode == "strict":
            # In strict mode: exclude studies whose title+abstract contains
            # design_exclusion_keywords (e.g., "non-interventional", "observational")
            for dkw in design_excl_kw:
                pattern = r'\b' + re.escape(dkw) + r'\b'
                if re.search(pattern, title_abstract):
                    is_excluded = True
                    reason = f"Design exclusion keyword '{dkw}' in TITLE/ABSTRACT"
                    break

        if not is_excluded and design_filter and filter_mode == "loose":
            # In loose mode: only exclude studies matching design_exclusion_keywords
            for dkw in design_excl_kw:
                pattern = r'\b' + re.escape(dkw) + r'\b'
                if re.search(pattern, title_abstract):
                    is_excluded = True
                    reason = f"Design exclusion keyword '{dkw}' in TITLE/ABSTRACT"
                    break

        # ── Keyword checks ──
        if not is_excluded:
            for kw in exclusion_keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'

                if kw in TITLE_ONLY_KEYWORDS:
                    # Animal / in-vitro: only check title
                    if re.search(pattern, title):
                        is_excluded = True
                        reason = f"Keyword '{kw}' in TITLE"
                        break
                else:
                    # Publication types: check title + abstract
                    if re.search(pattern, title_abstract):
                        is_excluded = True
                        reason = f"Keyword '{kw}' in TITLE/ABSTRACT"
                        break

        if is_excluded:
            excluded.append({
                "study_id": study.get("study_id"),
                "title": study.get("title"),
                "reason": reason,
            })
        else:
            kept.append(study)
    
    # 4. Cross-database duplicate check (catch any that slipped through Phase 2 dedup)
    #    Runs deduplicate_studies() again on the filtered pool — zero LLM cost.
    #    Value: catches duplicates added via late RIS imports or missed by Phase 2
    #    fuzzy matching. Only reports if new duplicates are found.
    before_dedup = len(kept)
    kept, dedup_report = deduplicate_studies(kept)
    n_dedup = dedup_report.get("duplicates_removed", 0)

    # Append cross-db duplicates to the excluded log
    for merge_entry in dedup_report.get("merge_log", []):
        excluded.append({
            "study_id": merge_entry.get("merged_from"),
            "title": "",      # not available in merge log
            "reason": f"Cross-database duplicate of {merge_entry.get('kept')} "
                      f"(source: {merge_entry.get('source', 'unknown')})",
        })

    # 5. Save
    dm.save("phase2_search", "filtered_studies.json", kept, subfolder="prescreened")
    dm.save("phase2_search", "prescreen_excluded_log.json", excluded, subfolder="prescreened")

    print("\n" + "=" * 60)
    print("✅ Phase 2.5 Pre-screening Complete!")
    print("=" * 60)
    print(f"  Input:    {len(studies)} studies")
    _struct_tags = {"Registry", "Publication type", "Title pattern",
                    "Title too short", "No abstract", "Conference"}
    n_structural = sum(
        1 for e in excluded
        if any(tag in e["reason"] for tag in _struct_tags)
    )
    n_design = sum(
        1 for e in excluded
        if e.get("reason", "").startswith("Design exclusion")
    )
    n_keyword = sum(
        1 for e in excluded
        if e.get("reason", "").startswith("Keyword")
    )
    n_crossdb = n_dedup
    n_other = len(excluded) - n_structural - n_design - n_keyword - n_crossdb
    print(f"  Excluded: {len(excluded)}")
    print(f"    ├─ Structural (registry/conference/type): {n_structural}")
    if n_design > 0:
        print(f"    ├─ Study design filter:                  {n_design}")
    print(f"    ├─ Keyword match:                        {n_keyword}")
    print(f"    └─ Cross-database duplicates:            {n_crossdb}")
    if n_other > 0:
        print(f"    └─ Other:                                {n_other}")
    print(f"  Kept:     {len(kept)}")
    print(f"  Output:   data/phase2_search/prescreened/filtered_studies.json")
    print(f"\n  💰 Token cost: $0.00 (pure Python string matching + dedup)")
    if n_crossdb > 0:
        print(f"  💡 {n_crossdb} cross-db duplicate(s) caught here saves ~{n_crossdb * 2} LLM screening calls.")
    print(f"\nNext: python scripts/run_phase3_stage1.py")


if __name__ == "__main__":
    main()
