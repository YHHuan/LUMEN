#!/usr/bin/env python3
"""
Phase 2: Literature Search — v5
=================================
  python scripts/run_phase2.py                    # 搜尋所有啟用的資料庫
  python scripts/run_phase2.py --deduplicate      # 只做去重
  python scripts/run_phase2.py --show-queries     # 顯示 query (不搜尋)

v5:
- Config toggles (ENABLE_PUBMED, ENABLE_EUROPEPMC, etc.)
- 5000 per database upper limit
- Total hits logging
- 支援 .ris / .csv 手動匯入
"""

import sys, json, logging, argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.apis.pubmed import PubMedAPI
from src.apis.europepmc import EuropePMCAPI
from src.apis.scopus import ScopusAPI
from src.apis.crossref import CrossRefAPI
from src.utils.project import select_project, get_data_dir
from src.utils.deduplication import deduplicate_studies, parse_ris_file
from src.utils.file_handlers import DataManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def show_queries(strategy: dict):
    """顯示各資料庫的 query"""
    queries = strategy.get("search_queries", {})
    print("\n" + "=" * 70)
    print("  Generated Queries Per Database")
    print("=" * 70)
    for db, q in queries.items():
        available = db in cfg.available_databases or db in ("pubmed", "europepmc", "crossref")
        status = "✅" if available else "⏸️  (no key / disabled)"
        print(f"\n  {status} {db.upper()}:")
        if isinstance(q, dict):
            print(f"     {json.dumps(q, indent=6)[:500]}")
        else:
            print(f"     {q[:500]}")
    print()


def search_databases(strategy: dict, date_range: dict) -> dict:
    """搜尋所有啟用的 API 資料庫，上限 5000 per database"""
    queries = strategy.get("search_queries", {})
    results = {}
    
    # === PubMed ===
    pubmed_query = queries.get("pubmed", "")
    if pubmed_query and cfg.enable_pubmed:
        logger.info("🔍 Searching PubMed...")
        try:
            pubmed = PubMedAPI()
            pmids = pubmed.search(
                query=pubmed_query, max_results=5000,
                date_start=str(date_range.get("start", "")),
                date_end=str(date_range.get("end", "")),
            )
            records = pubmed.fetch_details(pmids) if pmids else []
            results["pubmed"] = records
            logger.info(f"  PubMed: {len(records)} records")
        except Exception as e:
            logger.error(f"  PubMed failed: {e}")
            results["pubmed"] = []
    else:
        logger.info("⏸️  PubMed: skipped (disabled or empty query)")
    
    # === Europe PMC ===
    epmc_query = queries.get("europepmc", "")
    if epmc_query and cfg.enable_europepmc:
        logger.info("🔍 Searching Europe PMC...")
        try:
            epmc = EuropePMCAPI()
            records = epmc.search(query=epmc_query, max_results=5000)
            results["europepmc"] = records
            logger.info(f"  Europe PMC: {len(records)} records")
        except Exception as e:
            logger.error(f"  Europe PMC failed: {e}")
            results["europepmc"] = []
    else:
        logger.info("⏸️  Europe PMC: skipped (disabled or empty query)")
    
    # === Scopus ===
    if cfg.has_scopus and cfg.enable_scopus:
        scopus_query = queries.get("scopus", "")
        if scopus_query:
            logger.info("🔍 Searching Scopus...")
            try:
                scopus = ScopusAPI()
                records = scopus.search(query=scopus_query, max_results=5000)
                results["scopus"] = records
                logger.info(f"  Scopus: {len(records)} records")
            except Exception as e:
                logger.error(f"  Scopus failed: {e}")
                results["scopus"] = []
    else:
        logger.info("⏸️  Scopus: skipped (no key or disabled)")
    
    # === CrossRef ===
    if cfg.enable_crossref:
        crossref_params = queries.get("crossref", {})
        if crossref_params and isinstance(crossref_params, dict):
            logger.info("🔍 Searching CrossRef...")
            try:
                cr = CrossRefAPI()
                records = cr.search(params=crossref_params, max_results=500)
                results["crossref"] = records
                logger.info(f"  CrossRef: {len(records)} records")
            except Exception as e:
                logger.error(f"  CrossRef failed: {e}")
                results["crossref"] = []
    else:
        logger.info("⏸️  CrossRef: skipped (disabled)")
    
    # === WoS ===
    if cfg.has_wos:
        logger.info("⏸️  WoS: manual .ris import only (API pending)")
    else:
        logger.info("⏸️  WoS: skipped (no key)")
    
    return results


def run_deduplication(dm: DataManager):
    """合併所有來源 + 去重"""
    all_studies = []
    source_counts = {}
    
    raw_dir = Path(get_data_dir()) / "phase2_search" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Load API results
    for json_file in raw_dir.glob("*_results.json"):
        source_name = json_file.stem.replace("_results", "")
        with open(json_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        all_studies.extend(records)
        source_counts[source_name] = len(records)
        logger.info(f"  Loaded {len(records)} from {source_name}")
    
    # Load manual .ris imports
    for ris_file in raw_dir.glob("*.ris"):
        records = parse_ris_file(str(ris_file))
        all_studies.extend(records)
        source_counts[ris_file.stem] = len(records)
        logger.info(f"  Loaded {len(records)} from {ris_file.name}")
    
    # Load .csv imports
    for csv_file in raw_dir.glob("*.csv"):
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            records = []
            for _, row in df.iterrows():
                records.append({
                    "study_id": f"CSV_{csv_file.stem}_{len(records)}",
                    "title": str(row.get("Title", row.get("title", ""))),
                    "abstract": str(row.get("Abstract", row.get("abstract", ""))),
                    "authors": str(row.get("Authors", row.get("authors", ""))).split("; "),
                    "year": str(row.get("Year", row.get("year", ""))),
                    "doi": str(row.get("DOI", row.get("doi", ""))),
                    "journal": str(row.get("Journal", row.get("Source Title", ""))),
                    "source": csv_file.stem,
                })
            all_studies.extend(records)
            source_counts[csv_file.stem] = len(records)
        except Exception as e:
            logger.warning(f"  CSV import failed ({csv_file.name}): {e}")
    
    if not all_studies:
        logger.error("No studies found! Check data/phase2_search/raw/")
        return
    
    logger.info(f"\n📊 Total before dedup: {len(all_studies)}")
    deduped, dedup_report = deduplicate_studies(all_studies)
    
    dm.save("phase2_search", "all_studies.json", deduped, subfolder="deduplicated")
    
    # TRIPOD 5c: record search execution timestamp and model training cutoffs
    import yaml as _yaml
    with open("config/models.yaml", "r", encoding="utf-8") as _mf:
        _models_cfg = _yaml.safe_load(_mf)
    _model_cutoffs = {
        role: m.get("pinned_at", "unknown")
        for role, m in _models_cfg.get("models", {}).items()
    }

    search_log = {
        "search_executed_at": datetime.now(timezone.utc).isoformat(),
        "source_counts": source_counts,
        "total_before_dedup": len(all_studies),
        "total_after_dedup": len(deduped),
        "duplicates_removed": dedup_report["duplicates_removed"],
        "databases_searched": list(source_counts.keys()),
        "databases_skipped": cfg.skipped_databases,
        "model_pinned_at_dates": _model_cutoffs,
    }
    dm.save("phase2_search", "search_log.json", search_log)
    
    print("\n" + "=" * 60)
    print("✅ Phase 2 Complete!")
    print("=" * 60)
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    print(f"\n  Total before dedup: {len(all_studies)}")
    print(f"  Total after dedup:  {len(deduped)}")
    print(f"  Duplicates removed: {dedup_report['duplicates_removed']}")
    if cfg.skipped_databases:
        print(f"\n  ⏸️  Skipped: {', '.join(cfg.skipped_databases)}")
        print(f"     → 手動從 Cochrane/WoS/Embase 匯出 .ris 放到 data/phase2_search/raw/")
    print(f"\nNext: python scripts/run_phase2_5_prescreen.py (recommended)")
    print(f"  Or: python scripts/run_phase3_stage1.py")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Literature Search v5")
    parser.add_argument("--deduplicate", action="store_true", help="Only dedup")
    parser.add_argument("--show-queries", action="store_true", help="Show queries only")
    args = parser.parse_args()
    
    select_project()
    dm = DataManager()
    cfg.print_status()
    
    strategy = dm.load("phase1_strategy", "search_strategy.json")
    pico = dm.load("input", "pico.yaml")
    date_range = pico.get("date_range", {})
    
    if args.show_queries:
        show_queries(strategy)
        return
    
    if args.deduplicate:
        run_deduplication(dm)
        return
    
    # Search
    results = search_databases(strategy, date_range)
    
    # Save raw results
    for source, records in results.items():
        dm.save("phase2_search", f"{source}_results.json", records, subfolder="raw")
    
    # Auto-dedup
    run_deduplication(dm)


if __name__ == "__main__":
    main()
