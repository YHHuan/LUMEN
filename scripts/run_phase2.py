"""
Phase 2: Literature Search + Deduplication — LUMEN v2
=======================================================
Searches multiple databases, imports manual .ris files, deduplicates.

Usage:
    python scripts/run_phase2.py                    # Full search + dedup
    python scripts/run_phase2.py --show-queries     # Preview queries only
    python scripts/run_phase2.py --deduplicate      # Re-dedup after manual imports
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.deduplication import deduplicate_studies, parse_ris_file
from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-queries", action="store_true")
    parser.add_argument("--deduplicate", action="store_true")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    # Load search strategy
    strategy = dm.load("phase1_strategy", "search_strategy.json")
    queries = strategy.get("search_queries", {})

    if args.show_queries:
        print("\n  Search Queries:")
        for db, q in queries.items():
            print(f"\n  [{db}]")
            print(f"  {q}")
        return

    all_studies = []

    if not args.deduplicate:
        # Run searches
        for db_name in cfg.available_databases:
            query = queries.get(db_name, "")
            if not query:
                logger.warning(f"No query for {db_name}, skipping")
                continue

            logger.info(f"Searching {db_name}...")
            try:
                if db_name == "pubmed":
                    from src.apis.pubmed import search_pubmed
                    results = search_pubmed(query)
                elif db_name == "europepmc":
                    from src.apis.europepmc import search_europepmc
                    results = search_europepmc(query)
                elif db_name == "scopus":
                    from src.apis.scopus import search_scopus
                    results = search_scopus(query)
                elif db_name == "crossref":
                    from src.apis.crossref import search_crossref
                    results = search_crossref(query)
                elif db_name == "openalex":
                    from src.apis.openalex import search_openalex
                    results = search_openalex(query)
                else:
                    logger.info(f"No API handler for {db_name}")
                    continue

                logger.info(f"  {db_name}: {len(results)} results")
                dm.save("phase2_search", f"{db_name}_results.json",
                        results, subfolder="raw")
                all_studies.extend(results)

            except Exception as e:
                logger.error(f"  {db_name} search failed: {e}")

    # Import manual .ris files
    raw_dir = Path(get_data_dir()) / "phase2_search" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for ris_file in raw_dir.glob("*.ris"):
        logger.info(f"Importing RIS: {ris_file.name}")
        ris_studies = parse_ris_file(str(ris_file))
        for s in ris_studies:
            s["source"] = ris_file.stem
        all_studies.extend(ris_studies)
        logger.info(f"  {len(ris_studies)} studies from {ris_file.name}")

    if args.deduplicate:
        # Reload existing raw results
        for json_file in raw_dir.glob("*_results.json"):
            import json
            with open(json_file, "r") as f:
                all_studies.extend(json.load(f))

    # Deduplicate
    logger.info(f"Total before dedup: {len(all_studies)}")
    unique, dedup_log = deduplicate_studies(all_studies)

    dm.save("phase2_search", "all_studies.json", unique, subfolder="deduplicated")
    dm.save("phase2_search", "dedup_log.json", dedup_log, subfolder="deduplicated")

    print("\n" + "=" * 50)
    print("  Phase 2 Complete")
    print("=" * 50)
    print(f"  Total records: {len(all_studies)}")
    print(f"  After dedup: {len(unique)}")
    print(f"  Duplicates removed: {len(dedup_log)}")
    print()


if __name__ == "__main__":
    main()
