#!/usr/bin/env python3
"""
Phase 2.6: Clinical-Trial Publication Lookup
=============================================
Many RIS imports (Embase, Cochrane) contain trial registry entries or conference
abstracts that lack a PMID/DOI.  This step searches PubMed for the published
paper associated with each such record, then back-fills pmid / doi / pmc_id
so that Phase 3 Stage 2 can auto-download the PDF.

Default: runs on phase 3.1 included list (fast, ~168 studies).
Use --dedup to run on the full dedup pool instead (slow, ~1000+ studies).

Workflow:
  python scripts/run_phase2_6_ct_lookup.py            # Enrich phase 3.1 included list
  python scripts/run_phase2_6_ct_lookup.py --dry-run  # Preview matches only
  python scripts/run_phase2_6_ct_lookup.py --dedup    # Run on full dedup pool instead

After running:
  python scripts/run_phase3_stage2.py --integrate
  python scripts/run_phase3_stage2.py --download
"""

import sys
import json
import logging
import re
import time
import argparse
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.apis.pubmed import PubMedAPI
from src.utils.file_handlers import DataManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

NCT_RE = re.compile(r'\b(NCT\d{6,8})\b', re.IGNORECASE)

# ── Paths ────────────────────────────────────────────────────────────────────
def _path(rel: str) -> str:
    return str(Path(get_data_dir()) / rel)

# These are resolved at call time via _path()
_DEDUP_REL     = "phase2_search/deduplicated/all_studies.json"
_PRESCREEN_REL = "phase2_search/prescreened/filtered_studies.json"
_STAGE1_REL    = "phase3_screening/stage1_title_abstract/included_studies.json"
_LOG_REL       = "phase2_search/ct_lookup_log.json"


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_nct(study: dict) -> list[str]:
    """Return all NCT IDs found anywhere in the study record."""
    text = " ".join([
        study.get("title", "") or "",
        study.get("abstract", "") or "",
        " ".join(study.get("authors", []) or []),
        study.get("first_author", "") or "",
        " ".join(study.get("mesh_terms", []) or []),
    ])
    return list({m.upper() for m in NCT_RE.findall(text)})


def needs_lookup(study: dict) -> bool:
    """True if this study is a candidate for PubMed enrichment."""
    if study.get("pmid"):
        return False
    sid = study.get("study_id", "")
    if not sid.startswith("RIS_"):
        return False
    # Skip entries with very short/generic titles — conference fragments, headings
    title = (study.get("title") or "").strip()
    if len(title) < 25:
        return False
    return True


def title_similarity(t1: str, t2: str) -> float:
    t1 = re.sub(r'\W+', ' ', t1.lower()).strip()
    t2 = re.sub(r'\W+', ' ', t2.lower()).strip()
    return SequenceMatcher(None, t1, t2).ratio()


def search_by_doi(api: PubMedAPI, doi: str) -> list[str]:
    """Look up PMID directly from DOI — most precise method."""
    doi = doi.strip()
    if not doi:
        return []
    try:
        # PubMed indexes DOI in the [aid] (Article ID) field
        pmids = api.search(f'"{doi}"[aid]', max_results=3)
        return pmids
    except Exception as e:
        logger.warning(f"DOI search error ({doi}): {e}")
    return []


def search_by_nct(api: PubMedAPI, nct_id: str) -> list[str]:
    """Search PubMed for RESULTS papers linked to this NCT ID.
    Excludes protocol-only papers to avoid registering unpublished trials.
    """
    try:
        # [si] = secondary identifier field — most precise NCT link
        # NOT "protocol"[ti] to avoid trial registration records without results
        query = f"{nct_id}[si] NOT protocol[ti] NOT registry[ti]"
        pmids = api.search(query, max_results=10)
        if pmids:
            return pmids
        # Fallback: all-text search still excluding protocols
        pmids = api.search(f"{nct_id}[tw] NOT protocol[ti] NOT registry[ti]", max_results=10)
        return pmids
    except Exception as e:
        logger.warning(f"NCT search error ({nct_id}): {e}")
    return []


def search_by_title(api: PubMedAPI, title: str, year: str = "") -> list[str]:
    """Search PubMed by title (exact first, then first-8-words fallback)."""
    if not title or len(title) < 15:
        return []
    # Escape quotes
    safe = title.replace('"', '')
    queries = [f'"{safe}"[ti]']
    if year:
        queries.append(f'"{safe}"[ti] AND {year}[dp]')
    # First-8-words fallback (helps with truncated titles)
    words = safe.split()
    if len(words) >= 6:
        partial = " ".join(words[:8])
        if year:
            queries.append(f'"{partial}"[ti] AND {year}[dp]')
        else:
            queries.append(f'"{partial}"[ti]')

    for q in queries:
        try:
            pmids = api.search(q, max_results=5)
            if pmids:
                return pmids
        except Exception as e:
            logger.warning(f"Title search error: {e}")
    return []


def pick_best_match(candidates: list[dict], study: dict,
                    sim_threshold: float = 0.72) -> dict | None:
    """
    From fetched PubMed records, pick the one whose title best matches
    the study title.  Returns None if no candidate exceeds the threshold.
    """
    orig_title = study.get("title", "") or ""
    if not orig_title:
        return candidates[0] if candidates else None

    best, best_score = None, 0.0
    for cand in candidates:
        score = title_similarity(cand.get("title", ""), orig_title)
        if score > best_score:
            best_score, best = score, cand

    if best and best_score >= sim_threshold:
        return best
    # If only one candidate and NCT was used, trust it even at lower similarity
    if len(candidates) == 1 and best_score >= 0.45:
        return best
    return None


def enrich_study(study: dict, pub: dict) -> dict:
    """
    Back-fill publication fields into the study record.
    Never overwrites non-empty values so existing good data is preserved.
    """
    fields = ["pmid", "pmc_id", "doi", "title", "abstract",
              "authors", "first_author", "journal", "year",
              "mesh_terms", "publication_types"]
    for f in fields:
        if not study.get(f) and pub.get(f):
            study[f] = pub[f]
    study["ct_lookup"] = {
        "matched_pmid": pub.get("pmid"),
        "match_score": round(title_similarity(
            pub.get("title", ""), study.get("title", "")), 3),
        "method": pub.get("_lookup_method", "unknown"),
    }
    return study


# ── Main ─────────────────────────────────────────────────────────────────────

def run_lookup(dry_run: bool = False, use_dedup: bool = False,
               sim_threshold: float = 0.72):
    """
    Default: enrich the phase 3.1 included list in-place (fast, targeted).
    use_dedup=True: enrich the full dedup/prescreened pool instead (slower).
    In both cases also patches the stage1 list if it exists and differs.
    """
    select_project()
    dm = DataManager()
    api = PubMedAPI()

    # ── Decide which file to load ─────────────────────────────────────────
    if use_dedup:
        if Path(_path(_PRESCREEN_REL)).exists():
            logger.info(f"Loading prescreened: {_path(_PRESCREEN_REL)}")
            studies = json.loads(Path(_path(_PRESCREEN_REL)).read_text(encoding="utf-8"))
            save_path = _path(_PRESCREEN_REL)
        else:
            logger.info(f"Loading dedup: {_path(_DEDUP_REL)}")
            studies = json.loads(Path(_path(_DEDUP_REL)).read_text(encoding="utf-8"))
            save_path = _path(_DEDUP_REL)
        patch_stage1 = True  # always patch stage1 when running on dedup
    else:
        # Default: work directly on the phase 3.1 included list
        if not Path(_path(_STAGE1_REL)).exists():
            logger.error(f"Phase 3.1 list not found: {_path(_STAGE1_REL)}. Run phase 3 stage 1 first, or use --dedup.")
            sys.exit(1)
        logger.info(f"Loading phase 3.1 included: {_path(_STAGE1_REL)}")
        studies = json.loads(Path(_path(_STAGE1_REL)).read_text(encoding="utf-8"))
        save_path = _path(_STAGE1_REL)
        patch_stage1 = False  # already editing stage1 directly

    candidates = [s for s in studies if needs_lookup(s)]
    logger.info(f"Total studies loaded: {len(studies)} | RIS candidates for lookup: {len(candidates)}")

    log = []
    enriched_count = 0

    for i, study in enumerate(candidates):
        sid = study["study_id"]
        title = (study.get("title") or "").strip()
        year  = str(study.get("year") or "")
        ncts  = extract_nct(study)

        logger.info(f"[{i+1}/{len(candidates)}] {sid} | NCTs: {ncts or 'none'} | {title[:60]}")

        found_pmids = []
        method = ""
        doi = (study.get("doi") or "").strip()

        # Strategy 0: DOI → PMID (most precise; skip other strategies if it works)
        if doi:
            found_pmids = search_by_doi(api, doi)
            if found_pmids:
                method = f"doi:{doi}"

        # Strategy 1: NCT ID search (results papers only, excludes protocols)
        if not found_pmids:
            for nct in ncts:
                pmids = search_by_nct(api, nct)
                if pmids:
                    found_pmids = pmids
                    method = f"nct:{nct}"
                    break

        # Strategy 2: Title search
        if not found_pmids and title:
            found_pmids = search_by_title(api, title, year)
            method = "title"

        if not found_pmids:
            log.append({"study_id": sid, "result": "not_found", "ncts": ncts, "title": title})
            logger.info(f"  ↳ not found")
            continue

        # Fetch details for top candidates (max 5)
        try:
            pubs = api.fetch_details(found_pmids[:5])
        except Exception as e:
            logger.warning(f"  fetch_details failed: {e}")
            log.append({"study_id": sid, "result": "fetch_error", "error": str(e)})
            continue

        for p in pubs:
            p["_lookup_method"] = method

        # DOI match → trust the first result directly (DOI is unique identifier)
        if method.startswith("doi:") and pubs:
            best = pubs[0]
        else:
            best = pick_best_match(pubs, study, sim_threshold)
        if not best:
            log.append({
                "study_id": sid, "result": "low_confidence",
                "candidates": [p.get("pmid") for p in pubs],
                "title": title,
            })
            logger.info(f"  ↳ {len(pubs)} candidates but none confident enough")
            continue

        score = title_similarity(best.get("title", ""), title)
        logger.info(f"  ↳ MATCH  PMID:{best['pmid']}  score:{score:.2f}  [{method}]")
        logger.info(f"          PubMed title: {best.get('title','')[:80]}")

        log.append({
            "study_id": sid,
            "result": "matched",
            "matched_pmid": best["pmid"],
            "matched_doi": best.get("doi", ""),
            "score": round(score, 3),
            "method": method,
            "original_title": title,
            "pubmed_title": best.get("title", ""),
        })

        if not dry_run:
            enrich_study(study, best)
            enriched_count += 1

    # ── Save ─────────────────────────────────────────────────────────────────
    if not dry_run:
        Path(save_path).write_text(
            json.dumps(studies, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"✅ Updated {enriched_count} records in {save_path}")

        # If we enriched the dedup pool, also patch the stage1 list
        stage1_path = Path(_path(_STAGE1_REL))
        if patch_stage1 and stage1_path.exists() and save_path != _path(_STAGE1_REL):
            stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
            enriched_map = {s["study_id"]: s for s in studies}
            patched = 0
            for s in stage1:
                sid = s["study_id"]
                if sid in enriched_map and enriched_map[sid].get("pmid") and not s.get("pmid"):
                    src = enriched_map[sid]
                    for f in ["pmid", "pmc_id", "doi", "ct_lookup"]:
                        if src.get(f):
                            s[f] = src[f]
                    patched += 1
            stage1_path.write_text(
                json.dumps(stage1, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(f"✅ Patched {patched} records in {_path(_STAGE1_REL)}")

    # Save log
    Path(_path(_LOG_REL)).write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    matched   = sum(1 for e in log if e["result"] == "matched")
    not_found = sum(1 for e in log if e["result"] == "not_found")
    low_conf  = sum(1 for e in log if e["result"] == "low_confidence")
    errors    = sum(1 for e in log if e["result"] in ("fetch_error",))

    print("\n" + "="*60)
    print("Phase 2.6 — Clinical Trial Publication Lookup")
    print("="*60)
    print(f"  RIS studies without PMID:  {len(candidates)}")
    print(f"  Matched to publication:    {matched}")
    print(f"  Not found in PubMed:       {not_found}")
    print(f"  Low-confidence (skipped):  {low_conf}")
    print(f"  Errors:                    {errors}")
    if dry_run:
        print(f"\n⚠️  DRY RUN — no files were modified")
    else:
        print(f"\n✅ Enriched {enriched_count} records")
        print(f"   Log: {_path(_LOG_REL)}")
        print(f"\nNext steps:")
        print(f"  python scripts/run_phase3_stage2.py --integrate")
        print(f"  python scripts/run_phase3_stage2.py --download")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.6: Look up published papers for RIS trial/abstract entries"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview matches without modifying any files")
    parser.add_argument("--dedup", action="store_true",
                        help="Run on full dedup/prescreened pool instead of phase 3.1 list")
    parser.add_argument("--threshold", type=float, default=0.72,
                        help="Title similarity threshold for accepting a match (default: 0.72)")
    args = parser.parse_args()

    run_lookup(dry_run=args.dry_run, use_dedup=args.dedup,
               sim_threshold=args.threshold)


if __name__ == "__main__":
    main()
