#!/usr/bin/env python3
"""
Data Extraction Validation — generate review CSV & compute accuracy metrics.
=============================================================================
Step 1 — Export extracted fields for manual verification:
  python scripts/validate_extraction.py --export
  python scripts/validate_extraction.py --export --complete-only   # 12 complete studies only

  Output: data/validation/extraction_validation.csv
  For each row:
    1. Locate the PDF for that study_id in
       data/phase3_screening/stage2_fulltext/fulltext_pdfs/
    2. Find the reported value in the paper
    3. Fill 'actual_value' with exactly what the paper reports
       (e.g. "24.3", "RCT", "MMSE", "double-blind")
    4. Leave 'actual_value' BLANK if the field is genuinely not reported

Step 2 — Check progress:
  python scripts/validate_extraction.py --status

Step 3 — Compute metrics after annotation:
  python scripts/validate_extraction.py --compute

  Output:
    data/validation/extraction_metrics.csv
    data/validation/extraction_metrics_summary.txt

Metrics (Cassell 2023 taxonomy):
  Fields tagged text-based or table-based for comparison by source type.
  Per category (study_design, population, intervention, outcomes):
    k fields, Precision, Recall, F1
  Overall (text-based) and Overall (table-based).

CSV columns:
  study_id        — internal study identifier
  citation        — paper citation string
  has_pdf         — Y/N (whether PDF is in fulltext_pdfs/)
  category        — study_design | population | intervention | outcomes
  field           — field name (e.g. "Study design (RCT/CCT/…)")
  source_type     — text (from prose) or table (from data table)
  extracted_value — what the LLM extracted
  actual_value    — ✏ YOU FILL THIS IN from the PDF
  match           — auto-computed: Y / N / NA (blank actual = NA)
  notes           — ✏ optional comments
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.project import select_project, get_data_dir

EXTRACTED_JSON = STAGE2_JSON = PDF_DIR = EXPORT_CSV = METRICS_CSV = METRICS_TXT = ""


def _init_paths():
    global EXTRACTED_JSON, STAGE2_JSON, PDF_DIR, EXPORT_CSV, METRICS_CSV, METRICS_TXT
    dd = get_data_dir()
    EXTRACTED_JSON = f"{dd}/phase4_extraction/extracted_data.json"
    STAGE2_JSON    = f"{dd}/phase3_screening/stage2_fulltext/included_studies.json"
    PDF_DIR        = f"{dd}/phase3_screening/stage2_fulltext/fulltext_pdfs"
    EXPORT_CSV     = f"{dd}/validation/extraction_validation.csv"
    METRICS_CSV    = f"{dd}/validation/extraction_metrics.csv"
    METRICS_TXT    = f"{dd}/validation/extraction_metrics_summary.txt"
    Path(dd, "validation").mkdir(parents=True, exist_ok=True)

# ── Field definitions ─────────────────────────────────────────────────────────
# (field_path, category, field_label, source_type)
# field_path is dot-notation: 'characteristics.design' or 'outcomes.MMSE.intervention_mean_post'
# source_type: 'text' (typically in Methods/Results prose) or 'table'

STATIC_FIELDS = [
    # Study characteristics — usually text (Methods)
    ("characteristics.design",         "study_design",  "Study design (RCT/CCT/…)",     "text"),
    ("characteristics.blinding",       "study_design",  "Blinding",                      "text"),
    ("characteristics.n_total",        "study_design",  "N total",                       "text"),
    ("characteristics.duration_weeks", "study_design",  "Duration (weeks)",              "text"),
    ("characteristics.registration",   "study_design",  "Trial registration",            "text"),
    # Population
    ("population.diagnosis",           "population",    "Diagnosis",                     "text"),
    ("population.diagnostic_criteria", "population",    "Diagnostic criteria",           "text"),
    ("population.age_mean",            "population",    "Age mean",                      "text"),
    ("population.age_sd",              "population",    "Age SD",                        "text"),
    ("population.sex_male_percent",    "population",    "Sex % male",                    "text"),
    ("population.baseline_severity",   "population",    "Baseline severity",             "text"),
    # Intervention — text
    ("intervention.type",              "intervention",  "Intervention type",             "text"),
    ("intervention.target_area",       "intervention",  "Target area",                   "text"),
    ("intervention.intensity",         "intervention",  "Intensity (% RMT / mA)",        "text"),
    ("intervention.sessions_total",    "intervention",  "Sessions total",                "text"),
    ("intervention.sessions_per_week", "intervention",  "Sessions per week",             "text"),
    ("intervention.total_duration_weeks", "intervention", "Total duration (weeks)",      "text"),
]

# Outcome fields (per measure) — numeric values typically come from tables
OUTCOME_NUMERIC_FIELDS = [
    ("intervention_mean_post",  "outcomes", "Intervention mean post",  "table"),
    ("intervention_sd_post",    "outcomes", "Intervention SD post",    "table"),
    ("intervention_n",          "outcomes", "Intervention N",          "table"),
    ("control_mean_post",       "outcomes", "Control mean post",       "table"),
    ("control_sd_post",         "outcomes", "Control SD post",         "table"),
    ("control_n",               "outcomes", "Control N",               "table"),
    ("p_value",                 "outcomes", "p-value",                 "text"),
]


def get_nested(d: dict, path: str):
    """Retrieve value via dot-notation path from nested dict."""
    keys = path.split(".")
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return None
    return val


def pdf_exists(study_id: str) -> bool:
    safe = study_id.replace("/", "_").replace("\\", "_")
    return (Path(PDF_DIR) / f"{safe}.pdf").exists()


# ── Export ────────────────────────────────────────────────────────────────────

def do_export(complete_only: bool):
    extracted = json.loads(Path(EXTRACTED_JSON).read_text(encoding="utf-8"))
    stage2    = json.loads(Path(STAGE2_JSON).read_text(encoding="utf-8"))
    meta      = {s["study_id"]: s for s in stage2}

    if complete_only:
        studies = [s for s in extracted if s.get("data_completeness") == "complete"]
    else:
        studies = [s for s in extracted
                   if s.get("data_completeness") in ("complete", "partial")]

    print(f"Studies for validation: {len(studies)} "
          f"({'complete only' if complete_only else 'complete + partial'})")

    rows = []

    for s in studies:
        sid     = s["study_id"]
        cit     = s.get("citation", "")
        has_pdf = pdf_exists(sid)

        # Static fields
        for field_path, category, label, src_type in STATIC_FIELDS:
            val = get_nested(s, field_path)
            rows.append({
                "study_id":       sid,
                "citation":       cit,
                "has_pdf":        "Y" if has_pdf else "N",
                "category":       category,
                "field":          label,
                "source_type":    src_type,
                "extracted_value": "" if val is None else str(val),
                "actual_value":   "",   # human fills this
                "match":          "",   # auto-computed on --compute
                "notes":          "",
            })

        # Outcome fields
        outcomes = s.get("outcomes", {}) or {}
        if isinstance(outcomes, dict):
            for measure_name, measure_data in outcomes.items():
                if not isinstance(measure_data, dict):
                    continue
                src_loc = measure_data.get("source_location", {}) or {}
                # Guess source type: if section mentions "table" → table, else text
                section = (src_loc.get("section") or "").lower()
                inferred_src = "table" if "table" in section else "text"
                for sub_field, category, label, _ in OUTCOME_NUMERIC_FIELDS:
                    val = measure_data.get(sub_field)
                    rows.append({
                        "study_id":        sid,
                        "citation":        cit,
                        "has_pdf":         "Y" if has_pdf else "N",
                        "category":        category,
                        "field":           f"{measure_name} — {label}",
                        "source_type":     inferred_src,
                        "extracted_value": "" if val is None else str(val),
                        "actual_value":    "",
                        "match":           "",
                        "notes":           "",
                    })

    fieldnames = ["study_id", "citation", "has_pdf", "category", "field",
                  "source_type", "extracted_value", "actual_value", "match", "notes"]
    with open(EXPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_with_pdf = sum(1 for s in studies if pdf_exists(s["study_id"]))
    print(f"✅ Exported {len(rows)} field-rows across {len(studies)} studies")
    print(f"   PDFs available: {n_with_pdf}/{len(studies)}")
    print(f"   → {EXPORT_CSV}")
    print(f"\n{'─'*60}")
    print(f" HOW TO ANNOTATE")
    print(f"{'─'*60}")
    print(f"  1. Open {EXPORT_CSV} in Excel / LibreOffice / Google Sheets")
    print(f"  2. For each row (grouped by study_id):")
    print(f"     a. Locate the PDF:  data/phase3_screening/stage2_fulltext/")
    print(f"                         fulltext_pdfs/<study_id>.pdf")
    print(f"     b. Find the field value in the paper")
    print(f"     c. Fill 'actual_value' with what is reported")
    print(f"        Examples: '24.3', 'RCT', 'DLPFC', 'double-blind'")
    print(f"     d. Leave 'actual_value' blank if the field is not reported")
    print(f"  3. Save the CSV (keep the same filename)")
    print(f"\nCheck progress: python scripts/validate_extraction.py --status")
    print(f"Compute metrics: python scripts/validate_extraction.py --compute")
    print(f"\nEstimated time: ~4–6 hours for complete studies (~{len(rows)} field-rows)")


# ── Compute ───────────────────────────────────────────────────────────────────

def is_match(extracted: str, actual: str) -> str:
    """
    Returns 'Y', 'N', or 'NA'.
    - NA: actual_value is blank (field not present in paper)
    - Y: values match (numeric: within 1%; text: case-insensitive equality after strip)
    - N: values differ
    """
    actual = actual.strip()
    extracted = extracted.strip()
    if not actual:
        return "NA"
    if not extracted:
        return "N"   # LLM left it blank but paper had a value → miss

    # Try numeric comparison
    try:
        a_float = float(actual.replace(",", ""))
        e_float = float(extracted.replace(",", ""))
        # Within 1% tolerance for floating point
        if a_float == 0:
            return "Y" if e_float == 0 else "N"
        return "Y" if abs(a_float - e_float) / abs(a_float) <= 0.01 else "N"
    except ValueError:
        pass

    # Text comparison — case-insensitive
    return "Y" if actual.lower() == extracted.lower() else "N"


def do_compute():
    if not Path(EXPORT_CSV).exists():
        print(f"❌ Validation CSV not found: {EXPORT_CSV}")
        print(f"   Run --export first.")
        sys.exit(1)

    rows = []
    with open(EXPORT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Fill match column
    annotated = 0
    for row in rows:
        m = is_match(row.get("extracted_value", ""), row.get("actual_value", ""))
        row["match"] = m
        if row.get("actual_value", "").strip():
            annotated += 1

    # Write back with match column filled
    fieldnames = list(rows[0].keys())
    with open(EXPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Compute metrics ───────────────────────────────────────────────────────
    # Only rows where actual_value is known (match != NA)
    eval_rows = [r for r in rows if r["match"] != "NA" and r.get("actual_value","").strip()]
    print(f"Total rows: {len(rows)} | Annotated (non-blank actual): {len(eval_rows)}")

    def metrics_for_subset(subset):
        if not subset: return {}
        k = len(subset)
        # TP: extracted non-blank AND matches
        tp = sum(1 for r in subset if r["match"] == "Y" and r.get("extracted_value","").strip())
        fp = sum(1 for r in subset if r["match"] == "N"  and r.get("extracted_value","").strip())
        fn = sum(1 for r in subset if r["match"] == "N"  and not r.get("extracted_value","").strip())
        tn = sum(1 for r in subset if r["match"] == "Y"  and not r.get("extracted_value","").strip())
        prec   = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
        return dict(k=k, TP=tp, FP=fp, FN=fn, TN=tn,
                    Precision=round(prec,3), Recall=round(recall,3), F1=round(f1,3))

    # Group by category
    categories = sorted(set(r["category"] for r in eval_rows))
    src_types  = sorted(set(r["source_type"] for r in eval_rows))

    results = []
    for cat in categories:
        sub = [r for r in eval_rows if r["category"] == cat]
        m = metrics_for_subset(sub)
        m["group"] = cat
        m["split"] = "by_category"
        results.append(m)

    for st in src_types:
        sub = [r for r in eval_rows if r["source_type"] == st]
        m = metrics_for_subset(sub)
        m["group"] = f"source:{st}"
        m["split"] = "by_source_type"
        results.append(m)

    overall = metrics_for_subset(eval_rows)
    overall["group"] = "OVERALL"
    overall["split"] = "overall"
    results.append(overall)

    # ── CSV ───────────────────────────────────────────────────────────────────
    cols = ["group", "split", "k", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1"]
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in results:
            writer.writerow({c: r.get(c, "") for c in cols})

    # ── Text summary ──────────────────────────────────────────────────────────
    lines = [
        "Data Extraction Validation — Metrics Summary",
        "=" * 65,
        f"Annotated field-rows: {len(eval_rows)}",
        "",
        f"{'Group':<35} {'k':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}",
        "-" * 72,
    ]
    for r in results:
        lines.append(
            f"{r['group']:<35} {r['k']:>5} {r['TP']:>4} {r['FP']:>4} {r['FN']:>4} "
            f"{r['Precision']:>6.3f} {r['Recall']:>6.3f} {r['F1']:>6.3f}"
        )

    summary = "\n".join(lines)
    print("\n" + summary)
    Path(METRICS_TXT).write_text(summary, encoding="utf-8")
    print(f"\n✅ Metrics → {METRICS_CSV}")
    print(f"   Summary → {METRICS_TXT}")
    print(f"   Annotated CSV updated → {EXPORT_CSV}")


# ── Status ────────────────────────────────────────────────────────────────────

def do_status():
    if not Path(EXPORT_CSV).exists():
        print(f"❌ Validation CSV not found: {EXPORT_CSV}")
        print(f"   Run --export first.")
        sys.exit(1)

    rows = []
    with open(EXPORT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    total     = len(rows)
    annotated = sum(1 for r in rows if r.get("actual_value", "").strip())
    remaining = total - annotated
    pct       = annotated / total * 100 if total else 0

    studies   = len(set(r["study_id"] for r in rows))
    ann_std   = len(set(r["study_id"] for r in rows if r.get("actual_value", "").strip()))

    print(f"\n📊 Extraction Validation — Annotation Status")
    print(f"{'─'*45}")
    print(f"  Total field-rows : {total}")
    print(f"  Annotated rows   : {annotated}  ({pct:.1f}%)")
    print(f"  Remaining rows   : {remaining}")
    print(f"\n  Studies in CSV   : {studies}")
    print(f"  Studies started  : {ann_std}")

    if remaining == 0:
        print(f"\n✅ All rows annotated! Run:")
        print(f"   python scripts/validate_extraction.py --compute")
    else:
        print(f"\nKeep going! Run --compute when done.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extraction validation: export fields for human review or compute accuracy"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export",  action="store_true",
                       help="Export extracted fields to CSV for human annotation")
    group.add_argument("--compute", action="store_true",
                       help="Compute precision/recall/F1 from annotated CSV")
    group.add_argument("--status",  action="store_true",
                       help="Show current annotation progress")
    parser.add_argument("--complete-only", action="store_true",
                        help="--export: only include 'complete' studies (default: complete + partial)")
    args = parser.parse_args()
    select_project()
    _init_paths()

    if args.export:
        do_export(complete_only=args.complete_only)
    elif args.status:
        do_status()
    else:
        do_compute()


if __name__ == "__main__":
    main()
