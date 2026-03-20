#!/usr/bin/env python3
"""
Screening Validation — generate human-review CSV & compute metrics.
====================================================================
Step 1 — Export a stratified random sample for human annotation:
  python scripts/validate_screening.py --export --n 100 --seed 42

  Output: data/validation/screening_validation_sample.csv
  Open the CSV, review each row, and fill 'human_decision' with
  'include' or 'exclude'. Use 'human_notes' for comments.
  For two-reviewer mode, also fill 'human_decision_2' / 'human_notes_2'.

Step 2 — Check annotation progress:
  python scripts/validate_screening.py --status

Step 3 — Compute metrics after annotation:
  python scripts/validate_screening.py --compute

  Output:
    data/validation/screening_metrics.csv       — full metrics table
    data/validation/screening_metrics_summary.txt

Metrics computed per screener (Screener1, Screener2, Consensus):
  TP, TN, FP, FN, Accuracy, Precision, Sensitivity, Specificity, F1,
  Cohen's κ, PABAK, WSS@95%

  When both human_decision and human_decision_2 are filled, computes
  Human1, Human2, Consensus vs combined human gold standard, and
  Human1 vs Human2 inter-rater reliability (IRR).

Columns in the CSV:
  study_id           — internal study identifier
  title              — article title (full, up to 500 chars)
  abstract           — article abstract (full, up to 3000 chars)
  year               — publication year
  journal            — journal name
  screener1_raw      — Screener 1 raw 5-point decision
  screener2_raw      — Screener 2 raw 5-point decision
  screener1_binary   — Screener 1 mapped to include/exclude
  screener2_binary   — Screener 2 mapped to include/exclude
  llm_final          — pipeline consensus decision
  human_decision     — ✏ Reviewer 1 (include / exclude)
  human_notes        — ✏ Reviewer 1 optional comments
  human_decision_2   — ✏ Reviewer 2 (include / exclude) — optional
  human_notes_2      — ✏ Reviewer 2 optional comments
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.project import select_project, get_data_dir

# These are set in _init_paths() after project selection
STAGE1_JSON = SCREENING_JSON = SAMPLE_CSV = METRICS_CSV = METRICS_TXT = DEDUP_JSON = ""


def _init_paths():
    global STAGE1_JSON, SCREENING_JSON, SAMPLE_CSV, METRICS_CSV, METRICS_TXT, DEDUP_JSON
    dd = get_data_dir()
    STAGE1_JSON    = f"{dd}/phase3_screening/stage1_title_abstract/included_studies.json"
    SCREENING_JSON = f"{dd}/phase3_screening/stage1_title_abstract/screening_results.json"
    SAMPLE_CSV     = f"{dd}/validation/screening_validation_sample.csv"
    METRICS_CSV    = f"{dd}/validation/screening_metrics.csv"
    METRICS_TXT    = f"{dd}/validation/screening_metrics_summary.txt"
    DEDUP_JSON     = f"{dd}/phase2_search/deduplicated/all_studies.json"
    Path(dd, "validation").mkdir(parents=True, exist_ok=True)


# ── 5-point → binary ─────────────────────────────────────────────────────────

def _side(decision: str) -> str:
    """Map 5-point scale to include/exclude."""
    return "include" if decision in (
        "most_likely_include", "likely_include", "undecided",
        "include",
    ) else "exclude"


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(predictions: list[str], actuals: list[str]) -> dict:
    """predictions and actuals are lists of 'include'/'exclude'."""
    assert len(predictions) == len(actuals)
    n = len(predictions)
    tp = sum(1 for p, a in zip(predictions, actuals) if p == "include" and a == "include")
    tn = sum(1 for p, a in zip(predictions, actuals) if p == "exclude" and a == "exclude")
    fp = sum(1 for p, a in zip(predictions, actuals) if p == "include" and a == "exclude")
    fn = sum(1 for p, a in zip(predictions, actuals) if p == "exclude" and a == "include")

    acc   = (tp + tn) / n if n else 0
    prec  = tp / (tp + fp) if (tp + fp) else 0
    sens  = tp / (tp + fn) if (tp + fn) else 0     # recall / sensitivity
    spec  = tn / (tn + fp) if (tn + fp) else 0
    f1    = 2 * prec * sens / (prec + sens) if (prec + sens) else 0

    # Cohen's κ
    p_e = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n) if n else 0
    kappa = (acc - p_e) / (1 - p_e) if (1 - p_e) else 0

    # PABAK (prevalence-adjusted, bias-adjusted kappa)
    pabak = 2 * acc - 1

    # WSS@95%: work saved over sampling at 95% recall
    # = (TN + FN) / N - 0.05  (only meaningful when sensitivity ≥ 0.95)
    wss95 = (tn + fn) / n - 0.05 if n else 0

    return dict(
        n=n, TP=tp, TN=tn, FP=fp, FN=fn,
        Accuracy=round(acc, 4),
        Precision=round(prec, 4),
        Sensitivity=round(sens, 4),
        Specificity=round(spec, 4),
        F1=round(f1, 4),
        Cohen_kappa=round(kappa, 4),
        PABAK=round(pabak, 4),
        WSS_at_95=round(wss95, 4),
    )


# ── Export ───────────────────────────────────────────────────────────────────

def do_export(n_sample: int, seed: int):
    # Load study metadata
    studies = json.loads(Path(STAGE1_JSON).read_text(encoding="utf-8"))
    meta = {s["study_id"]: s for s in studies}

    # Load screening results
    sr = json.loads(Path(SCREENING_JSON).read_text(encoding="utf-8"))
    included_ids = set(sr.get("included", []))
    excluded_ids = set(sr.get("excluded", []))

    s1_results = {r["study_id"]: r for r in sr.get("screener1_results", [])}
    s2_results = {r["study_id"]: r for r in sr.get("screener2_results", [])}

    # Build full record list (included + excluded, skip human_review_queue for now)
    all_ids = list(included_ids | excluded_ids)
    # Compute consensus binary for each
    records = []
    for sid in all_ids:
        s1 = s1_results.get(sid, {})
        s2 = s2_results.get(sid, {})
        s1_bin = _side(s1.get("decision", "exclude"))
        s2_bin = _side(s2.get("decision", "exclude"))
        # Consensus: included if either screener is include-side
        # (matches pipeline logic: both must be exclude to be excluded)
        final_bin = "include" if sid in included_ids else "exclude"
        m = meta.get(sid, {})
        records.append({
            "study_id":         sid,
            "title":            (m.get("title") or "")[:500],
            "abstract":         (m.get("abstract") or "")[:3000],
            "year":             m.get("year", ""),
            "journal":          m.get("journal", ""),
            "screener1_raw":    s1.get("decision", ""),
            "screener2_raw":    s2.get("decision", ""),
            "screener1_binary": s1_bin,
            "screener2_binary": s2_bin,
            "llm_final":        final_bin,
            "human_decision":   "",   # ✏ Reviewer 1
            "human_notes":      "",
            "human_decision_2": "",   # ✏ Reviewer 2 (optional)
            "human_notes_2":    "",
        })

    # Stratified sample: 50% include, 50% exclude (capped to available)
    inc_pool = [r for r in records if r["llm_final"] == "include"]
    exc_pool = [r for r in records if r["llm_final"] == "exclude"]
    rng = random.Random(seed)
    n_each = n_sample // 2
    sampled = (
        rng.sample(inc_pool, min(n_each, len(inc_pool))) +
        rng.sample(exc_pool, min(n_each, len(exc_pool)))
    )
    rng.shuffle(sampled)

    fieldnames = list(sampled[0].keys())
    with open(SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    n_inc = sum(1 for r in sampled if r["llm_final"] == "include")
    n_exc = sum(1 for r in sampled if r["llm_final"] == "exclude")
    print(f"✅ Exported {len(sampled)} records  (LLM: {n_inc} include, {n_exc} exclude)")
    print(f"   → {SAMPLE_CSV}")
    print(f"\n{'─'*60}")
    print(f" HOW TO ANNOTATE")
    print(f"{'─'*60}")
    print(f"  1. Open {SAMPLE_CSV} in Excel / LibreOffice / Google Sheets")
    print(f"  2. For each row: read the title and abstract")
    print(f"  3. Fill 'human_decision' column with:  include  OR  exclude")
    print(f"     - Include: the study meets your inclusion criteria")
    print(f"     - Exclude: the study does not meet inclusion criteria")
    print(f"  4. Optionally note reasons in 'human_notes'")
    print(f"  5. Save the CSV (keep the same filename)")
    print(f"\nCheck progress: python scripts/validate_screening.py --status")
    print(f"Compute metrics: python scripts/validate_screening.py --compute")
    print(f"\nEstimated time: ~2–4 hours for 100 records (≈ 1–2 min per record)")


# ── Compute ──────────────────────────────────────────────────────────────────

def do_compute():
    if not Path(SAMPLE_CSV).exists():
        print(f"❌ Sample CSV not found: {SAMPLE_CSV}")
        print(f"   Run --export first.")
        sys.exit(1)

    rows = []
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Filter to rows annotated by at least Reviewer 1
    annotated = [r for r in rows if r.get("human_decision", "").strip() in ("include", "exclude")]
    if not annotated:
        print("❌ No annotated rows found. Fill in 'human_decision' column first.")
        sys.exit(1)
    print(f"Annotated rows (Reviewer 1): {len(annotated)} / {len(rows)}")

    # Check for Reviewer 2
    dual = [r for r in annotated if r.get("human_decision_2", "").strip() in ("include", "exclude")]
    has_dual = len(dual) > 0
    if has_dual:
        print(f"Annotated rows (Reviewer 2): {len(dual)} / {len(annotated)}")

    # Use majority vote as gold standard when both reviewers are present
    # For rows with only one reviewer, that reviewer's decision is gold.
    def gold(r):
        h1 = r.get("human_decision", "").strip()
        h2 = r.get("human_decision_2", "").strip()
        if h2 in ("include", "exclude"):
            # Both annotated: agree → use that; disagree → include (conservative)
            return h1 if h1 == h2 else "include"
        return h1

    human_gold = [gold(r) for r in annotated]
    human1 = [r["human_decision"].strip() for r in annotated]

    screeners = {
        "Screener1":  [r["screener1_binary"].strip() for r in annotated],
        "Screener2":  [r["screener2_binary"].strip() for r in annotated],
        "Consensus":  [r["llm_final"].strip()        for r in annotated],
    }

    results = []
    for name, preds in screeners.items():
        m = compute_metrics(preds, human_gold)
        m["Screener"] = name
        results.append(m)

    # LLM S1 vs S2 inter-rater reliability
    s1_preds = screeners["Screener1"]
    s2_preds = screeners["Screener2"]
    irr = compute_metrics(s1_preds, s2_preds)
    irr["Screener"] = "S1_vs_S2 (IRR)"
    results.append(irr)

    # Human vs human IRR (only rows where both reviewers annotated)
    if has_dual:
        dual_h1 = [r["human_decision"].strip() for r in dual]
        dual_h2 = [r["human_decision_2"].strip() for r in dual]
        h_irr = compute_metrics(dual_h1, dual_h2)
        h_irr["Screener"] = "H1_vs_H2 (IRR)"
        results.append(h_irr)
        # Also compute each human vs gold on dual-annotated subset
        dual_gold = [gold(r) for r in dual]
        for label, preds in [("Human1", dual_h1), ("Human2", dual_h2)]:
            m = compute_metrics(preds, dual_gold)
            m["Screener"] = label
            results.append(m)
        # LLM consensus vs gold on dual-annotated subset
        dual_llm = [r["llm_final"].strip() for r in dual]
        m = compute_metrics(dual_llm, dual_gold)
        m["Screener"] = "Consensus (dual subset)"
        results.append(m)

    # ── CSV output ────────────────────────────────────────────────────────────
    cols = ["Screener", "n", "TP", "TN", "FP", "FN",
            "Accuracy", "Precision", "Sensitivity", "Specificity",
            "F1", "Cohen_kappa", "PABAK", "WSS_at_95"]
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in results:
            writer.writerow({c: r.get(c, "") for c in cols})

    # ── Text summary ─────────────────────────────────────────────────────────
    lines = [
        "Screening Validation — Metrics Summary",
        "=" * 72,
        f"Annotated sample: {len(annotated)} studies",
        f"  Include (gold): {sum(1 for h in human_gold if h=='include')}",
        f"  Exclude (gold): {sum(1 for h in human_gold if h=='exclude')}",
        f"  Gold standard: {'majority vote (R1+R2)' if has_dual else 'Reviewer 1 only'}",
        "",
        f"{'Screener':<20} {'k':>4} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} "
        f"{'Acc':>6} {'Prec':>6} {'Sens':>6} {'Spec':>6} "
        f"{'F1':>6} {'κ':>7} {'PABAK':>7} {'WSS95':>7}",
        "-" * 100,
    ]
    for r in results:
        lines.append(
            f"{r['Screener']:<20} {r['n']:>4} {r['TP']:>4} {r['TN']:>4} {r['FP']:>4} {r['FN']:>4} "
            f"{r['Accuracy']:>6.3f} {r['Precision']:>6.3f} {r['Sensitivity']:>6.3f} {r['Specificity']:>6.3f} "
            f"{r['F1']:>6.3f} {r['Cohen_kappa']:>7.3f} {r['PABAK']:>7.3f} {r['WSS_at_95']:>7.3f}"
        )

    summary = "\n".join(lines)
    print("\n" + summary)
    Path(METRICS_TXT).write_text(summary, encoding="utf-8")
    print(f"\n✅ Metrics → {METRICS_CSV}")
    print(f"   Summary → {METRICS_TXT}")


# ── Status ───────────────────────────────────────────────────────────────────

def do_status():
    if not Path(SAMPLE_CSV).exists():
        print(f"❌ Sample CSV not found: {SAMPLE_CSV}")
        print(f"   Run --export first.")
        sys.exit(1)

    rows = []
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    total      = len(rows)
    annotated  = sum(1 for r in rows if r.get("human_decision", "").strip() in ("include", "exclude"))
    annotated2 = sum(1 for r in rows if r.get("human_decision_2", "").strip() in ("include", "exclude"))
    remaining  = total - annotated
    pct        = annotated / total * 100 if total else 0

    print(f"\n📊 Screening Validation — Annotation Status")
    print(f"{'─'*45}")
    print(f"  Total rows     : {total}")
    print(f"  Reviewer 1     : {annotated}  ({pct:.1f}%)")
    print(f"  Reviewer 2     : {annotated2}  ({annotated2/total*100:.1f}%)")
    print(f"  Remaining (R1) : {remaining}")
    if annotated > 0:
        agree = sum(
            1 for r in rows
            if r.get("human_decision", "").strip() == r.get("llm_final", "").strip()
            and r.get("human_decision", "").strip()
        )
        print(f"\n  Preliminary LLM agreement: {agree}/{annotated} ({agree/annotated*100:.1f}%)")
    if annotated2 > 0:
        dual = [(r["human_decision"].strip(), r["human_decision_2"].strip())
                for r in rows
                if r.get("human_decision","").strip() in ("include","exclude")
                and r.get("human_decision_2","").strip() in ("include","exclude")]
        if dual:
            h_agree = sum(1 for h1, h2 in dual if h1 == h2)
            print(f"  Human IRR (R1 vs R2)     : {h_agree}/{len(dual)} ({h_agree/len(dual)*100:.1f}%)")
    if remaining == 0:
        print(f"\n✅ All rows annotated! Run:")
        print(f"   python scripts/validate_screening.py --compute")
    else:
        print(f"\nKeep going! Run --compute when all rows are filled.")


# ── Refresh CSV ──────────────────────────────────────────────────────────────


def do_refresh_csv():
    """
    Refresh title and abstract columns in the existing CSV from source JSON,
    preserving all human annotation columns (human_decision, human_notes, etc.).
    Safe to run after enrich_abstracts.py --fix.

    Metadata priority: all_studies.json (full pool) > included_studies.json
    """
    if not Path(SAMPLE_CSV).exists():
        print(f"❌ Sample CSV not found: {SAMPLE_CSV}")
        sys.exit(1)

    # Build metadata from the full deduplicated pool (has both included + excluded)
    meta: dict = {}
    for json_path in (DEDUP_JSON, STAGE1_JSON):
        p = Path(json_path)
        if p.exists():
            for s in json.loads(p.read_text(encoding="utf-8")):
                sid = s.get("study_id", "")
                if sid and sid not in meta:
                    meta[sid] = s
    if not meta:
        print(f"❌ No source JSON found. Run Phase 2 and Phase 3 first.")
        sys.exit(1)
    print(f"Loaded metadata for {len(meta)} studies from source JSON.")

    rows = []
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # Ensure new columns exist in fieldnames
        for col in ("human_decision_2", "human_notes_2"):
            if col not in fieldnames:
                fieldnames = list(fieldnames) + [col]
        for row in reader:
            rows.append(dict(row))

    updated = 0
    for row in rows:
        sid = row.get("study_id", "")
        m = meta.get(sid, {})
        new_title    = (m.get("title") or "")[:500]
        new_abstract = (m.get("abstract") or "")[:3000]
        if new_title    != row.get("title", ""):
            row["title"] = new_title
            updated += 1
        if new_abstract != row.get("abstract", ""):
            row["abstract"] = new_abstract
            updated += 1
        # Ensure new annotation columns exist
        row.setdefault("human_decision_2", "")
        row.setdefault("human_notes_2", "")

    with open(SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Refreshed {len(rows)} rows ({updated} title/abstract cells updated)")
    print(f"   → {SAMPLE_CSV}")
    print(f"\nAll human_decision / human_notes values preserved.")
    print(f"New columns added: human_decision_2, human_notes_2")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Screening validation: export sample or compute metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export",      action="store_true", help="Export random sample CSV for annotation")
    group.add_argument("--compute",     action="store_true", help="Compute metrics from annotated CSV")
    group.add_argument("--status",      action="store_true", help="Show current annotation progress")
    group.add_argument("--refresh-csv", action="store_true",
                       help="Refresh title/abstract in existing CSV from source JSON (preserves annotations)")
    parser.add_argument("--n",    type=int, default=100, help="Sample size for --export (default: 100)")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed for --export (default: 42)")
    args = parser.parse_args()
    select_project()
    _init_paths()

    if args.export:
        do_export(args.n, args.seed)
    elif args.status:
        do_status()
    elif args.refresh_csv:
        do_refresh_csv()
    else:
        do_compute()


if __name__ == "__main__":
    main()
