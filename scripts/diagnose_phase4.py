#!/usr/bin/env python3
"""
Phase 4 Data Quality Diagnostic
==================================
  python scripts/diagnose_phase4.py          # Full report
  python scripts/diagnose_phase4.py --fix    # Attempt auto-fixes (SE→SD, etc.)

Reports:
  1. How many studies have computable data vs abstract-only
  2. Which studies need re-extraction (PDF was incomplete)
  3. Which studies have fixable issues (SE reported as SD, etc.)
  4. Estimated meta-analysis yield per outcome
"""

import sys, json, re, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir

from src.utils.effect_sizes import auto_compute_effect, _safe_float


def analyze_study(s):
    """Analyze one study's extraction quality."""
    sid = s.get("study_id", "?")
    outcomes = s.get("outcomes", {})
    notes = (s.get("extraction_notes") or "").lower()
    dc = s.get("data_completeness", "unknown")
    
    result = {
        "study_id": sid,
        "completeness": dc,
        "n_outcomes": len(outcomes),
        "computable_outcomes": [],
        "fixable_outcomes": [],
        "empty_outcomes": [],
        "issue": None,
    }
    
    if s.get("error") == "extraction_failed":
        result["issue"] = "extraction_failed"
        return result
    
    if not outcomes:
        if "abstract" in notes or "only" in notes:
            result["issue"] = "abstract_only"
        else:
            result["issue"] = "no_outcomes"
        return result
    
    for key, od in outcomes.items():
        # Try to compute effect
        computed = auto_compute_effect(od, "SMD")
        
        if computed and computed.get("effect") is not None:
            result["computable_outcomes"].append({
                "name": key,
                "measure": od.get("measure", key),
                "method": computed["method"],
            })
            continue
        
        # Check if fixable
        fixable = _check_fixable(od)
        if fixable:
            result["fixable_outcomes"].append({
                "name": key,
                "measure": od.get("measure", key),
                "issue": fixable,
            })
        else:
            result["empty_outcomes"].append(key)
    
    if result["computable_outcomes"]:
        result["issue"] = None  # OK
    elif result["fixable_outcomes"]:
        result["issue"] = "fixable"
    elif "abstract" in notes or "only" in notes or "methods" in notes:
        result["issue"] = "abstract_only"
    elif "p-value" in notes or any(od.get("p_value") for od in outcomes.values()):
        result["issue"] = "p_values_only"
    else:
        result["issue"] = "incomplete_data"
    
    return result


def _check_fixable(od):
    """Check if an outcome has fixable issues."""
    # Case: has mean but SD might be SE
    for suffix in ["_post", "_change"]:
        m1 = _safe_float(od.get(f"intervention_mean{suffix}"))
        s1 = _safe_float(od.get(f"intervention_sd{suffix}"))
        n1 = _safe_float(od.get("intervention_n"))
        m2 = _safe_float(od.get(f"control_mean{suffix}"))
        s2 = _safe_float(od.get(f"control_sd{suffix}"))
        n2 = _safe_float(od.get("control_n"))
        
        if m1 is not None and m2 is not None and n1 is not None and n2 is not None:
            if s1 is None and s2 is None:
                return f"has_mean_n{suffix}_but_no_sd"
            if s1 == 0 or s2 == 0:
                return f"sd_is_zero{suffix}"
    
    # Case: has mean_change but no SD_change, might have SE in text
    for suffix in ["_change", "_post"]:
        mc1 = _safe_float(od.get(f"intervention_mean{suffix}"))
        mc2 = _safe_float(od.get(f"control_mean{suffix}"))
        if mc1 is not None and mc2 is not None:
            n1 = _safe_float(od.get("intervention_n"))
            n2 = _safe_float(od.get("control_n"))
            if n1 and n2:
                return f"has_means_and_n{suffix}_missing_sd"
    
    return None


def apply_auto_fixes(extracted_data):
    """Try to fix common issues."""
    import numpy as np
    fixed_count = 0
    
    for s in extracted_data:
        for key, od in s.get("outcomes", {}).items():
            # Fix: if spread_type_reported == "SE", convert to SD
            spread = (od.get("spread_type_reported") or "").upper()
            if "SE" in spread and "SD" not in spread:
                for suffix in ["_post", "_change"]:
                    for group in ["intervention", "control"]:
                        se = _safe_float(od.get(f"{group}_sd{suffix}"))
                        n = _safe_float(od.get(f"{group}_n"))
                        if se is not None and n is not None and n > 1:
                            sd = se * np.sqrt(n)
                            od[f"{group}_sd{suffix}"] = round(sd, 4)
                            od[f"_fix_applied"] = f"SE→SD: {se}×√{int(n)}={sd:.4f}"
                            fixed_count += 1
    
    return fixed_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Apply auto-fixes (SE→SD)")
    args = parser.parse_args()
    
    select_project()
    data_path = Path(get_data_dir()) / "phase4_extraction" / "extracted_data.json"
    if not data_path.exists():
        print(f"❌ {data_path} not found. Run Phase 4 first.")
        return
    
    with open(data_path) as f:
        data = json.load(f)
    
    print(f"=" * 60)
    print(f"Phase 4 Data Quality Report")
    print(f"=" * 60)
    print(f"Total studies: {len(data)}")
    print()
    
    # Analyze
    results = [analyze_study(s) for s in data]
    
    # Categorize
    categories = {
        None: ("✅ Computable", []),
        "fixable": ("🔧 Fixable", []),
        "abstract_only": ("📄 Abstract Only", []),
        "p_values_only": ("📊 P-values Only", []),
        "incomplete_data": ("⚠️ Incomplete", []),
        "no_outcomes": ("❌ No Outcomes", []),
        "extraction_failed": ("💥 Failed", []),
    }
    
    for r in results:
        if r["issue"] in categories:
            categories[r["issue"]][1].append(r)
        else:
            categories.setdefault(r["issue"], (r["issue"], []))[1].append(r)
    
    for issue_key, (label, items) in categories.items():
        if items:
            print(f"{label}: {len(items)}")
    
    print()
    
    # Detail: computable
    computable = categories[None][1]
    if computable:
        print(f"--- COMPUTABLE STUDIES ({len(computable)}) ---")
        for r in computable:
            measures = ", ".join(o["measure"] for o in r["computable_outcomes"])
            print(f"  {r['study_id']}: {measures}")
    
    # Detail: fixable
    fixable = categories.get("fixable", ("", []))[1]
    if fixable:
        print(f"\n--- FIXABLE STUDIES ({len(fixable)}) ---")
        for r in fixable:
            for fo in r["fixable_outcomes"]:
                print(f"  {r['study_id']} / {fo['measure']}: {fo['issue']}")
    
    # Detail: abstract only
    abstract = categories.get("abstract_only", ("", []))[1]
    if abstract:
        print(f"\n--- ABSTRACT ONLY ({len(abstract)}) — need full-text PDF ---")
        for r in abstract[:10]:
            print(f"  {r['study_id']}")
        if len(abstract) > 10:
            print(f"  ... and {len(abstract)-10} more")
    
    # Estimated yield per measure
    print(f"\n{'='*60}")
    print(f"Estimated Meta-Analysis Yield")
    print(f"{'='*60}")
    
    measure_counts = {}
    for r in computable:
        for o in r["computable_outcomes"]:
            m = o["measure"]
            measure_counts[m] = measure_counts.get(m, 0) + 1
    
    for m, count in sorted(measure_counts.items(), key=lambda x: -x[1]):
        can_meta = "✅ can run" if count >= 2 else "❌ need ≥2"
        print(f"  {m}: {count} studies — {can_meta}")
    
    # Apply fixes
    if args.fix:
        print(f"\n{'='*60}")
        print(f"Applying Auto-Fixes...")
        print(f"{'='*60}")
        n_fixed = apply_auto_fixes(data)
        if n_fixed > 0:
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Applied {n_fixed} fixes. Re-run Phase 5 to see improvements.")
        else:
            print("No auto-fixable issues found.")
    
    # Actionable recommendations
    print(f"\n{'='*60}")
    print(f"Recommendations")
    print(f"{'='*60}")
    
    n_abstract = len(abstract)
    n_failed = len(categories.get("extraction_failed", ("", []))[1])
    n_ok = len(computable)
    
    if n_abstract > 0:
        print(f"\n1. 📄 {n_abstract} studies have abstract-only text.")
        print(f"   → Download full PDFs via Zotero + rename_zotero_pdfs.py")
        print(f"   → Then re-run Phase 4 (delete checkpoint first):")
        print(f"     rm data/.checkpoints/phase4_extraction.json")
        print(f"     python scripts/run_phase4.py")
    
    if n_failed > 0:
        print(f"\n2. 💥 {n_failed} studies failed extraction entirely.")
        print(f"   → Check if their PDFs exist in fulltext_pdfs/")
        print(f"   → Run: python scripts/rename_zotero_pdfs.py --diagnose")
    
    if n_ok >= 2:
        print(f"\n3. ✅ You have {n_ok} computable studies. Phase 5 should work!")
        print(f"   → python scripts/run_phase5.py --builtin-only")


if __name__ == "__main__":
    main()
