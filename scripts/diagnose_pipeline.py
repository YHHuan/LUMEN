#!/usr/bin/env python3
"""
Pipeline Quality Diagnostic — Phase 2 → 3 → 4
=================================================
  python scripts/diagnose_pipeline.py

Checks:
  Phase 2: Search coverage, dedup quality, source balance
  Phase 3 Stage 1: Screening agreement, inclusion/exclusion rates
  Phase 3 Stage 2: PDF coverage, download success, fulltext screening
  PDF Quality: Page count, text extraction, conference abstracts detection
  Phase 4: Extraction quality (delegates to diagnose_phase4.py)
"""

import sys, json, os, re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir


def d(rel_path: str) -> str:
    """Resolve a relative path under the active project data dir."""
    return str(Path(get_data_dir()) / rel_path)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def diagnose_phase2():
    section("Phase 2: Literature Search & Deduplication")
    
    search_log = load_json(d("phase2_search/search_log.json"))
    if not search_log:
        print("  ❌ search_log.json not found — Phase 2 not yet run")
        return None
    
    source_counts = search_log.get("source_counts", {})
    total_before = search_log.get("total_before_dedup", 0)
    total_after = search_log.get("total_after_dedup", 0)
    dupes = search_log.get("duplicates_removed", 0)
    
    print(f"  Sources searched: {len(source_counts)}")
    for src, count in source_counts.items():
        print(f"    {src}: {count}")
    print(f"\n  Total identified: {total_before}")
    print(f"  After dedup: {total_after}")
    print(f"  Duplicates removed: {dupes} ({100*dupes/total_before:.0f}%)" if total_before else "")
    
    # Quality checks
    issues = []
    if len(source_counts) < 3:
        issues.append("⚠️ Less than 3 databases — consider adding more sources for comprehensive review")
    if total_before < 500:
        issues.append("⚠️ <500 records — search might be too narrow")
    if total_before > 10000:
        issues.append("⚠️ >10000 records — search might be too broad")
    if dupes > 0 and dupes / total_before > 0.5:
        issues.append(f"⚠️ High duplication rate ({100*dupes/total_before:.0f}%) — databases may overlap heavily")
    
    skipped = search_log.get("databases_skipped", [])
    if skipped:
        issues.append(f"ℹ️ Skipped databases: {', '.join(skipped)} — add .ris files if available")
    
    if issues:
        print(f"\n  Issues:")
        for i in issues:
            print(f"    {i}")
    else:
        print(f"\n  ✅ Phase 2 looks good")
    
    return total_after


def diagnose_phase3_stage1():
    section("Phase 3 Stage 1: Title/Abstract Screening")
    
    results = load_json(d("phase3_screening/stage1_title_abstract/screening_results.json"))
    if not results:
        print("  ❌ screening_results.json not found — Phase 3 Stage 1 not yet run")
        return None
    
    total = results.get("total_screened", 0)
    included = len(results.get("included", []))
    excluded = len(results.get("excluded", []))
    conflicts = len(results.get("conflicts", []))
    agreement = results.get("agreement_rate", 0)
    
    print(f"  Total screened: {total}")
    print(f"  Included: {included} ({100*included/total:.0f}%)" if total else "")
    print(f"  Excluded: {excluded} ({100*excluded/total:.0f}%)" if total else "")
    print(f"  Conflicts: {conflicts}")
    print(f"  Agreement rate: {agreement:.1%}" if isinstance(agreement, float) else f"  Agreement rate: {agreement}")
    
    # Quality checks
    issues = []
    if included / total > 0.5 and total > 100:
        issues.append(f"⚠️ High inclusion rate ({100*included/total:.0f}%) — screening may be too lenient")
    if included / total < 0.05 and total > 100:
        issues.append(f"⚠️ Very low inclusion rate ({100*included/total:.0f}%) — criteria may be too strict")
    if isinstance(agreement, float) and agreement < 0.7:
        issues.append(f"⚠️ Low agreement ({agreement:.1%}) — screening criteria may be ambiguous")
    if conflicts > included * 0.3:
        issues.append(f"⚠️ Many conflicts ({conflicts}) — consider reviewing criteria")
    
    if issues:
        print(f"\n  Issues:")
        for i in issues:
            print(f"    {i}")
    else:
        print(f"\n  ✅ Stage 1 screening looks good")
    
    return included


def diagnose_phase3_stage2():
    section("Phase 3 Stage 2: Full-text Download & Screening")
    
    # Download log
    download_log = load_json(d("phase3_screening/stage2_fulltext/download_log.json"))
    fulltext_review = load_json(d("phase3_screening/stage2_fulltext/fulltext_review.json"))
    prisma = load_json(d("phase3_screening/prisma_flow.json"))
    included = load_json(d("phase3_screening/stage2_fulltext/included_studies.json"))
    stage1_included = load_json(d("phase3_screening/stage1_title_abstract/included_studies.json"))
    
    if not stage1_included:
        print("  ❌ Stage 1 results not found")
        return None
    
    total_stage1 = len(stage1_included)
    
    # --- PDF Coverage ---
    print(f"\n  📥 PDF Coverage:")
    pdf_dir = Path(d("phase3_screening/stage2_fulltext/fulltext_pdfs"))
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"    PDFs in folder: {len(pdf_files)}")
        
        # Check which included studies have PDFs
        matched = 0
        missing_pdf = []
        for s in stage1_included:
            safe_id = s["study_id"].replace("/", "_").replace("\\", "_")
            pdf_path = pdf_dir / f"{safe_id}.pdf"
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                matched += 1
            else:
                missing_pdf.append(s["study_id"])
        
        print(f"    Matched to studies: {matched}/{total_stage1} ({100*matched/total_stage1:.0f}%)")
        if missing_pdf:
            print(f"    Missing: {len(missing_pdf)}")
    else:
        print(f"    ❌ PDF folder not found: {pdf_dir}")
        return None
    
    # --- PDF Quality Analysis ---
    print(f"\n  📄 PDF Quality Analysis:")
    
    conference_abstracts = []
    tiny_pdfs = []
    normal_pdfs = []
    
    for pdf_file in pdf_files:
        size_kb = pdf_file.stat().st_size / 1024
        name = pdf_file.name
        
        # Detect conference abstracts by size
        if size_kb < 200:
            tiny_pdfs.append((name, size_kb))
        else:
            normal_pdfs.append((name, size_kb))
    
    # Try to check page counts with PyMuPDF if available
    page_counts = {}
    try:
        import fitz
        single_page = []
        multi_page = []
        for pdf_file in pdf_files[:100]:  # Check first 100
            try:
                doc = fitz.open(str(pdf_file))
                pages = len(doc)
                doc.close()
                page_counts[pdf_file.name] = pages
                if pages <= 2:
                    single_page.append((pdf_file.name, pages))
                else:
                    multi_page.append((pdf_file.name, pages))
            except:
                pass
        
        if page_counts:
            print(f"    Full papers (>2 pages): {len(multi_page)}")
            print(f"    Short documents (≤2 pages): {len(single_page)}")
            if single_page:
                print(f"    ⚠️ Short PDFs (likely abstracts/posters):")
                for name, pages in single_page[:10]:
                    print(f"      {name}: {pages} page(s)")
                if len(single_page) > 10:
                    print(f"      ... and {len(single_page)-10} more")
    except ImportError:
        print(f"    (install PyMuPDF for page-level analysis: pip install PyMuPDF)")
    
    print(f"    Small PDFs (<200KB): {len(tiny_pdfs)}")
    print(f"    Normal PDFs (≥200KB): {len(normal_pdfs)}")
    
    # --- Download Log ---
    if download_log:
        print(f"\n  📊 Download Results:")
        status_counts = Counter(entry["status"] for entry in download_log)
        for status, count in status_counts.most_common():
            print(f"    {status}: {count}")
    
    # --- Fulltext Review ---
    if fulltext_review:
        print(f"\n  🔍 Full-text Review:")
        decisions = Counter(r.get("decision") for r in fulltext_review)
        for decision, count in decisions.most_common():
            print(f"    {decision}: {count}")
        
        # Exclusion reasons
        exclusions = [r for r in fulltext_review if r.get("decision") == "exclude"]
        if exclusions:
            reason_counts = Counter()
            for r in exclusions:
                cat = r.get("exclusion_category", r.get("reason", "other"))
                reason_counts[cat] += 1
            print(f"\n  📋 Exclusion Breakdown:")
            for reason, count in reason_counts.most_common():
                print(f"    {reason}: {count}")
    
    # --- PRISMA ---
    if prisma:
        print(f"\n  📐 PRISMA Flow:")
        print(f"    Identified: {prisma.get('total_identified', '?')}")
        print(f"    After dedup: {prisma.get('after_deduplication', '?')}")
        print(f"    T/A screened: {prisma.get('title_abstract_screened', '?')}")
        print(f"    T/A excluded: {prisma.get('title_abstract_excluded', '?')}")
        print(f"    Fulltext assessed: {prisma.get('fulltext_assessed', '?')}")
        print(f"    Fulltext excluded: {prisma.get('fulltext_excluded', '?')}")
        no_pdf = prisma.get('fulltext_no_pdf', 0)
        ft_assessed = prisma.get('fulltext_assessed', 1)
        print(f"    No PDF: {no_pdf} ({100*no_pdf/ft_assessed:.0f}%)" if ft_assessed else "")
        print(f"    Included: {prisma.get('included_in_synthesis', '?')}")
    
    # Issues
    issues = []
    if missing_pdf and len(missing_pdf) > total_stage1 * 0.3:
        issues.append(f"⚠️ {len(missing_pdf)} studies ({100*len(missing_pdf)/total_stage1:.0f}%) missing PDFs")
    if tiny_pdfs and len(tiny_pdfs) > 10:
        issues.append(f"⚠️ {len(tiny_pdfs)} tiny PDFs (<200KB) — likely conference abstracts, not full papers")
    
    # Check for conference abstracts in included studies
    if included:
        n_included = len(included)
        print(f"\n  ✅ Final included: {n_included} studies")
    
    if issues:
        print(f"\n  Issues:")
        for i in issues:
            print(f"    {i}")
    
    return included


def diagnose_pdf_text_quality():
    """Check cached PDF text extraction quality."""
    section("PDF Text Extraction Quality")
    
    cache_dir = Path(d(".cache/pdf_text"))
    if not cache_dir.exists():
        print("  ❌ No PDF text cache found — Phase 3 Stage 2 review not yet run")
        return
    
    cache_files = list(cache_dir.glob("*.json"))
    if not cache_files:
        print("  ❌ Cache empty")
        return
    
    print(f"  Cached extractions: {len(cache_files)}")
    
    has_results = 0
    abstract_only = 0
    has_tables = 0
    avg_tokens = 0
    
    section_coverage = Counter()
    
    for cf in cache_files:
        try:
            with open(cf) as f:
                data = json.load(f)
            
            sections = data.get("sections", {})
            tokens = data.get("tokens_approx", 0)
            tables = data.get("tables", [])
            pages = data.get("pages", 0)
            
            avg_tokens += tokens
            
            for s in ["abstract", "methods", "results", "discussion"]:
                if s in sections:
                    section_coverage[s] += 1
            
            if tables:
                has_tables += 1
            
            if "results" in sections and len(sections.get("results", "")) > 500:
                has_results += 1
            elif pages <= 2:
                abstract_only += 1
            else:
                # Has pages but no results section detected
                abstract_only += 1
                
        except:
            pass
    
    avg_tokens = avg_tokens / len(cache_files) if cache_files else 0
    
    print(f"  Has Results section (>500 chars): {has_results} ({100*has_results/len(cache_files):.0f}%)")
    print(f"  Abstract/short only: {abstract_only}")
    print(f"  Has extracted tables: {has_tables}")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    
    print(f"\n  Section detection:")
    for s in ["abstract", "methods", "results", "discussion"]:
        count = section_coverage.get(s, 0)
        print(f"    {s}: {count}/{len(cache_files)} ({100*count/len(cache_files):.0f}%)")
    
    if has_results / len(cache_files) < 0.5:
        print(f"\n  ⚠️ Less than 50% of PDFs have a detectable Results section!")
        print(f"     Possible causes:")
        print(f"     1. PDFs are conference abstracts (1-2 pages, no Results)")
        print(f"     2. PDFs are scanned images (need OCR)")
        print(f"     3. Results section uses non-standard heading")


def detect_conference_abstracts():
    """Detect which included studies are actually conference abstracts."""
    section("Conference Abstract Detection")
    
    included = load_json(d("phase3_screening/stage2_fulltext/included_studies.json"))
    extracted = load_json(d("phase4_extraction/extracted_data.json"))
    
    if not included:
        print("  ❌ No included studies found")
        return
    
    pdf_dir = Path(d("phase3_screening/stage2_fulltext/fulltext_pdfs"))
    
    conference_keywords = [
        "conference", "congress", "meeting", "symposium", "poster",
        "abstract", "supplement", "proceedings", "annual meeting",
        "world congress", "eposter", "epresentation",
    ]
    
    suspects = []
    confirmed_full = []
    
    for s in included:
        sid = s["study_id"]
        journal = (s.get("journal") or "").lower()
        title = (s.get("title") or "").lower()
        
        is_suspect = False
        reason = []
        
        # Check journal name
        if any(kw in journal for kw in conference_keywords):
            is_suspect = True
            reason.append(f"journal='{s.get('journal','')[:40]}'")
        
        # Check title
        if any(kw in title for kw in ["abstract", "poster", "o1-", "o2-", "o3-", "p1-", "p2-", "p3-"]):
            is_suspect = True
            reason.append("title contains abstract/poster ID")
        
        # Check PDF size
        safe_id = sid.replace("/", "_").replace("\\", "_")
        pdf_path = pdf_dir / f"{safe_id}.pdf"
        if pdf_path.exists():
            size_kb = pdf_path.stat().st_size / 1024
            if size_kb < 200:
                is_suspect = True
                reason.append(f"PDF={size_kb:.0f}KB")
            
            # Check page count
            try:
                import fitz
                doc = fitz.open(str(pdf_path))
                pages = len(doc)
                doc.close()
                if pages <= 2:
                    is_suspect = True
                    reason.append(f"pages={pages}")
            except:
                pass
        
        # Check extracted data quality
        if extracted:
            ext = next((e for e in extracted if e.get("study_id") == sid), None)
            if ext:
                dc = (ext.get("data_completeness") or "").lower()
                notes = (ext.get("extraction_notes") or "").lower()
                if "abstract" in notes or dc == "minimal":
                    is_suspect = True
                    reason.append("extraction=abstract_only")
        
        if is_suspect:
            suspects.append({"study_id": sid, "reasons": reason,
                            "title": (s.get("title") or "")[:60]})
        else:
            confirmed_full.append(sid)
    
    print(f"  Confirmed full papers: {len(confirmed_full)}")
    print(f"  Suspected conference abstracts: {len(suspects)}")
    
    if suspects:
        print(f"\n  🔍 Suspected Conference Abstracts:")
        for s in suspects:
            print(f"    ❓ {s['study_id']}: {', '.join(s['reasons'])}")
            print(f"       {s['title']}")
    
    if suspects:
        print(f"\n  💡 Recommendation:")
        print(f"     These {len(suspects)} studies may be conference abstracts that")
        print(f"     cannot provide full statistical data for meta-analysis.")
        print(f"     Consider:")
        print(f"     1. Checking if full papers were later published")
        print(f"     2. Excluding from quantitative synthesis")
        print(f"     3. Listing as 'identified but data not available'")


def summary_recommendations():
    section("Summary & Recommendations")
    
    prisma = load_json(d("phase3_screening/prisma_flow.json"))
    extracted = load_json(d("phase4_extraction/extracted_data.json"))
    
    if not extracted:
        print("  Run Phase 4 first for full pipeline analysis")
        return
    
    total = len(extracted)
    
    # Count categories
    computable = sum(1 for s in extracted 
                     if any(any(s.get("outcomes", {}).get(k, {}).get(f"intervention_mean{sfx}") is not None
                               for sfx in ["", "_post", "_change"])
                           for k in s.get("outcomes", {})))
    
    no_pdf = prisma.get("fulltext_no_pdf", 0) if prisma else 0
    
    print(f"  Pipeline summary:")
    print(f"    Included studies: {total}")
    print(f"    With computable data: {computable} ({100*computable/total:.0f}%)")
    print(f"    Without full PDF: ~{no_pdf}")
    print(f"    Data gap: {total - computable} studies")
    
    print(f"\n  🎯 To improve data yield:")
    print(f"     1. Run: python scripts/rename_zotero_pdfs.py --diagnose")
    print(f"        → See which studies need PDFs")
    print(f"     2. Download missing PDFs via Zotero / institutional proxy")
    print(f"     3. Re-run Phase 3 Stage 2 review for newly added PDFs:")
    print(f"        python scripts/run_phase3_stage2.py --review")
    print(f"     4. Clear Phase 4 cache and re-extract:")
    
    if os.name == 'nt':
        print(f"        del data\\.checkpoints\\phase4_extraction.json")
    else:
        print(f"        rm data/.checkpoints/phase4_extraction.json")
    
    print(f"        python scripts/run_phase4.py")
    print(f"     5. Re-run Phase 5:")
    print(f"        python scripts/run_phase5.py --builtin-only")


def main():
    select_project()
    print("🔬 Meta-Analysis Pipeline Quality Diagnostic")
    print("=" * 60)

    diagnose_phase2()
    diagnose_phase3_stage1()
    diagnose_phase3_stage2()
    diagnose_pdf_text_quality()
    detect_conference_abstracts()
    summary_recommendations()


if __name__ == "__main__":
    main()
