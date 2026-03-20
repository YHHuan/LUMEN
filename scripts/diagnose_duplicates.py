#!/usr/bin/env python3
"""
Diagnose & fix duplicate studies in Phase 4 extracted data.
============================================================
The same paper often appears multiple times in included_studies.json because
it was retrieved from several databases (PubMed, Cochrane, Embase, …) and
post-dedup RIS entries share no PMID/DOI to trigger deduplication.

Phase 4 extraction then assigns the same citation to all duplicates, causing
inflated study counts in forest plots (e.g. "Cantoni 2025" × 5).

Usage:
  python scripts/diagnose_duplicates.py               # report only
  python scripts/diagnose_duplicates.py --fix         # remove duplicates in-place
  python scripts/diagnose_duplicates.py --fix --dry-run
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.project import select_project, get_data_dir


def _dpath(rel: str) -> str:
    return str(Path(get_data_dir()) / rel)

# Priority order when choosing which duplicate to keep
_SOURCE_RANK = ["PMID_", "EPMC_", "SCOPUS_", "RIS_"]


def source_rank(study_id: str) -> int:
    for i, prefix in enumerate(_SOURCE_RANK):
        if study_id.startswith(prefix):
            return i
    return len(_SOURCE_RANK)


def normalize_citation(cit: str) -> str:
    """Lower-case, strip punctuation noise, normalise whitespace."""
    if not cit:
        return ""
    c = cit.lower()
    c = re.sub(r"[&]", "and", c)
    c = re.sub(r"[^\w\s]", " ", c)
    c = re.sub(r"\s+", " ", c).strip()
    return c


def outcome_field_count(s: dict) -> int:
    """Count non-null numeric outcome values — used to pick richest record."""
    count = 0
    outcomes = s.get("outcomes", {}) or {}
    if isinstance(outcomes, dict):
        for measure_data in outcomes.values():
            if isinstance(measure_data, dict):
                for v in measure_data.values():
                    if v is not None and not isinstance(v, (dict, list)):
                        count += 1
    return count


def build_groups(extracted: list[dict]) -> dict[str, list[dict]]:
    """Group records by normalised citation string."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in extracted:
        key = normalize_citation(s.get("citation", ""))
        if not key:
            key = s["study_id"]          # no citation → treat as singleton
        groups[key].append(s)
    return groups


def pick_canonical(group: list[dict]) -> dict:
    """Return the record to keep: best source rank, then most outcome data."""
    return min(
        group,
        key=lambda s: (source_rank(s["study_id"]), -outcome_field_count(s)),
    )


def run(fix: bool = False, dry_run: bool = False):
    extracted = json.loads(Path(_dpath("phase4_extraction/extracted_data.json")).read_text(encoding="utf-8"))
    stage2 = json.loads(Path(_dpath("phase3_screening/stage2_fulltext/included_studies.json")).read_text(encoding="utf-8"))
    stage2_map = {s["study_id"]: s for s in stage2}

    groups = build_groups(extracted)
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}

    print(f"Total extracted records : {len(extracted)}")
    print(f"Unique citation groups  : {len(groups)}")
    print(f"Duplicate groups (≥2)   : {len(dup_groups)}")
    total_removals = sum(len(v) - 1 for v in dup_groups.values())
    print(f"Records to remove       : {total_removals}")
    print()

    # ── CSV report ────────────────────────────────────────────────────────────
    rows = []
    keep_ids: set[str] = set()
    remove_ids: set[str] = set()

    for key, group in sorted(dup_groups.items(), key=lambda x: -len(x[1])):
        canonical = pick_canonical(group)
        keep_ids.add(canonical["study_id"])
        for s in group:
            action = "KEEP" if s["study_id"] == canonical["study_id"] else "REMOVE"
            if action == "REMOVE":
                remove_ids.add(s["study_id"])
            m = stage2_map.get(s["study_id"], {})
            rows.append({
                "citation":        s.get("citation", ""),
                "study_id":        s["study_id"],
                "action":          action,
                "source_rank":     source_rank(s["study_id"]),
                "outcome_fields":  outcome_field_count(s),
                "has_pmid":        "Y" if m.get("pmid") else "N",
                "doi":             m.get("doi", ""),
                "title":           (m.get("title") or "")[:80],
            })

    Path(_dpath("phase4_extraction/duplicate_report.csv")).parent.mkdir(parents=True, exist_ok=True)
    with open(_dpath("phase4_extraction/duplicate_report.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Report written → {_dpath("phase4_extraction/duplicate_report.csv")}")

    # ── Print summary table ───────────────────────────────────────────────────
    if dup_groups:
        print(f"\n{'Citation':<55} {'Count':>5}  {'Keep'}")
        print("─" * 80)
        for key, group in sorted(dup_groups.items(), key=lambda x: -len(x[1])):
            canonical = pick_canonical(group)
            short = key[:54]
            print(f"{short:<55} {len(group):>5}  {canonical['study_id']}")

    if not fix:
        print("\n⚠️  Run with --fix to apply deduplication.")
        return

    # ── Apply fix ─────────────────────────────────────────────────────────────
    cleaned_extracted = [s for s in extracted if s["study_id"] not in remove_ids]
    cleaned_stage2 = [s for s in stage2 if s["study_id"] not in remove_ids]

    if dry_run:
        print(f"\n🔍 DRY RUN — would keep {len(cleaned_extracted)} / {len(extracted)} extracted records")
        print(f"           would keep {len(cleaned_stage2)} / {len(stage2)} stage2 included studies")
        return

    # Write back
    Path(_dpath("phase4_extraction/extracted_data.json")).write_text(
        json.dumps(cleaned_extracted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(_dpath("phase3_screening/stage2_fulltext/included_studies.json")).write_text(
        json.dumps(cleaned_stage2, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✅ Removed {total_removals} duplicate records.")
    print(f"   extracted_data.json : {len(cleaned_extracted)} records")
    print(f"   stage2 included     : {len(cleaned_stage2)} studies")
    print(f"\nNext steps:")
    print(f"  python scripts/run_phase5.py --builtin-only   # re-run statistics")
    print(f"  python scripts/run_phase6.py                  # re-generate manuscript")


def main():
    parser = argparse.ArgumentParser(description="Diagnose and fix duplicate papers in Phase 4 data")
    parser.add_argument("--fix", action="store_true",
                        help="Remove duplicates from extracted_data.json and stage2 included_studies.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --fix: show what would change without writing files")
    args = parser.parse_args()
    select_project()
    run(fix=args.fix, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
