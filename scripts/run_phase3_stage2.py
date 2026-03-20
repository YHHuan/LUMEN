#!/usr/bin/env python3
"""
Phase 3 Stage 2: Full-text Download & Screening
=================================================
Recommended workflow:
  python scripts/run_phase3_stage2.py --download           # 1. Auto-download (OA sources + Sci-Hub)
  python scripts/run_phase3_stage2.py --integrate          # 2. Integrate Zotero PDFs + export missing list
  # → add PDFs to data/zotero_export/ and re-run --integrate as many times as needed
  python scripts/run_phase3_stage2.py --review             # 3. Full-text screen (skips studies still pending)
  python scripts/run_phase3_stage2.py --finalize-pending   # 4. Gate: exclude no-PDF studies before Phase 4
  python scripts/run_phase3_stage2.py --all                # Steps 1+2+3 in sequence

⚠️ Token 節省設計:
  - PDF 文字只提取一次，快取到 data/.cache/pdf_text/
  - 篩選只看 Methods + Results (truncate to ~5000 tokens)
  - Phase 4 會重用快取，不再重新提取
  - --review 自動跳過已有決定的文獻，只處理新加入的 PDF

前置需求: Phase 3 Stage 1 完成
輸入: data/phase3_screening/stage1_title_abstract/included_studies.json
輸出: data/phase3_screening/stage2_fulltext/
"""

import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.pdf_downloader import PDFDownloader, PDFTextExtractor
from src.utils.langfuse_client import log_phase_start, log_phase_end
from src.agents.screener import ScreenerAgent, ArbiterAgent, _compute_kappa, _side
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget, Checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def download_pdfs(dm: DataManager):
    """下載全文 PDF"""
    studies = dm.load("phase3_screening", "included_studies.json",
                      subfolder="stage1_title_abstract")
    
    downloader = PDFDownloader()
    download_log = []
    
    # 紀錄需要手動下載的文獻清單
    missing_studies = []
    
    for i, study in enumerate(studies):
        logger.info(f"[{i+1}/{len(studies)}] Downloading: {study.get('study_id', '')}")
        pdf_path, status = downloader.download(study)
        
        download_log.append({
            "study_id": study["study_id"],
            "status": status,
            "pdf_path": pdf_path,
            "doi": study.get("doi", ""),
        })
        
        if status == "manual_download_needed" or status == "failed":
            missing_studies.append(study)
    
    dm.save("phase3_screening", "download_log.json", download_log,
            subfolder="stage2_fulltext")
    
    # === 新增：匯出需要手動下載的 CSV 清單 ===
    if missing_studies:
        logger.info("Exporting manual download list...")
        downloader.export_missing_to_csv(missing_studies)

    # Summary
    statuses = {}
    for entry in download_log:
        s = entry["status"]
        statuses[s] = statuses.get(s, 0) + 1

    print("\n📥 Download Summary:")
    for status, count in statuses.items():
        print(f"  {status}: {count}")

    if missing_studies:
        print(f"\n⚠️  {len(missing_studies)} studies need manual sourcing.")
        print(f"\n💡 Next step: run --integrate to match any PDFs you've added to data/zotero_export/")
        print(f"   and get a full missing-PDF list for manual searching:")
        print(f"   python scripts/run_phase3_stage2.py --integrate")


def _export_missing_pdf_list(studies: list, pdf_dir: Path) -> Path:
    """Write missing_pdfs.txt into stage2_fulltext/ and return the path."""
    out_path = pdf_dir.parent / "missing_pdfs.txt"

    missing_doi, missing_no_doi = [], []
    for s in studies:
        sid = s["study_id"]
        safe_id = sid.replace("/", "_").replace("\\", "_")
        p = pdf_dir / f"{safe_id}.pdf"
        if p.exists() and p.stat().st_size > 1000:
            continue
        doi = (s.get("doi") or "").strip()
        year = s.get("year", "")
        title = s.get("title", "")
        first_author = s.get("first_author") or ""
        pmid = (s.get("pmid") or "").strip()
        if doi:
            missing_doi.append((doi, title))
        else:
            id_str = f"PMID:{pmid}" if pmid else sid
            missing_no_doi.append((id_str, year, first_author, title))

    total = len(studies)
    covered = total - len(missing_doi) - len(missing_no_doi)

    lines = [
        f"# Missing PDFs — stage1 studies ({len(missing_doi) + len(missing_no_doi)} missing / {total} total)",
        f"# Coverage: {covered}/{total}  |  updated by --integrate",
        "",
        f"# ── With DOI ({len(missing_doi)}) — paste into Zotero / browser ──",
    ]
    for doi, _ in missing_doi:
        lines.append(doi)
    lines += [
        "",
        f"# ── No DOI — manual search needed ({len(missing_no_doi)}) ──",
    ]
    for id_str, year, author, title in missing_no_doi:
        lines.append(f"{id_str} | {year} | {author} | {title}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def integrate_manual_pdfs(dm: DataManager):
    """
    Integrate manually sourced PDFs from data/zotero_export/ into fulltext_pdfs/,
    then export up-to-date missing_pdfs.txt AND missing_pdfs_action_list.csv.

    Run this step after --download, and again after adding new PDFs to zotero_export/.
    """
    studies = dm.load("phase3_screening", "included_studies.json",
                      subfolder="stage1_title_abstract")
    pdf_dir = Path(get_data_dir()) / "phase3_screening" / "stage2_fulltext" / "fulltext_pdfs"
    zotero_dir = Path(get_data_dir()) / "zotero_export"

    # Step 1: Zotero matching
    if zotero_dir.exists():
        n_before = sum(1 for s in studies
                       if (pdf_dir / (s["study_id"].replace("/", "_").replace("\\", "_") + ".pdf")).exists()
                       and (pdf_dir / (s["study_id"].replace("/", "_").replace("\\", "_") + ".pdf")).stat().st_size > 1000)
        print(f"\n🔄 Integrating Zotero PDFs (coverage before: {n_before}/{len(studies)})...")
        subprocess.run([
            sys.executable, "scripts/rename_zotero_pdfs.py",
            "--source", "stage1", "--threshold", "0.65"
        ], check=False)
    else:
        print(f"ℹ️  No Zotero export folder at {zotero_dir} — skipping Zotero matching.")

    # Step 2: Compute current missing studies (ground truth after Zotero step)
    missing_studies = [
        s for s in studies
        if not (pdf_dir / (s["study_id"].replace("/", "_").replace("\\", "_") + ".pdf")).exists()
        or (pdf_dir / (s["study_id"].replace("/", "_").replace("\\", "_") + ".pdf")).stat().st_size <= 1000
    ]
    covered = len(studies) - len(missing_studies)

    # Step 3: Export missing_pdfs.txt (human-readable, title/author for no-DOI entries)
    out_path = _export_missing_pdf_list(studies, pdf_dir)

    # Step 4: Refresh missing_pdfs_action_list.csv with current missing list
    from src.utils.pdf_downloader import PDFDownloader
    downloader = PDFDownloader()
    if missing_studies:
        downloader.export_missing_to_csv(missing_studies)

    print(f"\n📋 PDF Coverage after integration: {covered}/{len(studies)} have PDFs")
    print(f"   Missing: {len(missing_studies)} studies")
    print(f"   missing_pdfs.txt     → {out_path}  (title/author for no-DOI entries)")
    print(f"   missing_pdfs_action_list.csv → proxy links for manual download")
    if missing_studies:
        print(f"\n💡 To recover more PDFs:")
        print(f"   1. Open missing_pdfs.txt — DOIs are paste-ready for Zotero/browser")
        print(f"   2. Add downloaded PDFs to data/zotero_export/")
        print(f"   3. Re-run: python scripts/run_phase3_stage2.py --integrate")
        print(f"   4. When ready: python scripts/run_phase3_stage2.py --review")


def review_fulltexts(dm: DataManager, reset_budget: bool = False):
    """全文篩選 — dual-screener + arbiter (mirrors Phase 3.1 design)"""
    studies = dm.load("phase3_screening", "included_studies.json",
                      subfolder="stage1_title_abstract")
    criteria = dm.load("phase1_strategy", "screening_criteria.json")

    # Budget is shared across both screeners; 20 USD for dual review.
    budget = TokenBudget(phase="phase3_stage2", limit_usd=20.0, reset=reset_budget)

    # Langfuse: phase start stamp
    import yaml
    from pathlib import Path as _Path
    _models_cfg = yaml.safe_load((_Path(__file__).parent.parent / "config/models.yaml").read_text())
    _lf_span = log_phase_start("phase3_stage2_review", {
        "input_study_count": len(studies),
        "screener1_model":   _models_cfg["models"]["screener1"]["model_id"],
        "screener2_model":   _models_cfg["models"]["screener2"]["model_id"],
        "arbiter_model":     _models_cfg["models"]["arbiter"]["model_id"],
        "budget_usd":        20.0,
        "mode":              "review_only",
    })
    _phase_start = time.time()
    extractor = PDFTextExtractor()
    screener1 = ScreenerAgent(role_name="screener1", budget=budget)
    screener2 = ScreenerAgent(role_name="screener2", budget=budget)
    arbiter   = ArbiterAgent(budget=budget)

    checkpoint = Checkpoint("phase3_fulltext_review")
    checkpoint.set_total(len(studies))

    review_results = []
    exclusion_reasons = []
    no_pdf_count = 0
    conflict_count = 0

    pdf_dir = Path(get_data_dir()) / "phase3_screening" / "stage2_fulltext" / "fulltext_pdfs"

    for i, study in enumerate(studies):
        sid = study["study_id"]

        # Only skip if a definitive decision was already made (not "pending")
        if checkpoint.is_done(sid):
            cached_result = checkpoint.get_result(sid)
            if cached_result.get("decision") in ["include", "exclude"]:
                review_results.append(cached_result)
                if cached_result.get("decision") == "exclude":
                    exclusion_reasons.append(cached_result)
                continue

        # Find PDF
        safe_id = sid.replace("/", "_").replace("\\", "_")
        pdf_path = pdf_dir / f"{safe_id}.pdf"

        if not pdf_path.exists():
            logger.warning(
                f"⚠️  No PDF for {sid} — marked pending. "
                f"Add {safe_id}.pdf and re-run --review to include."
            )
            no_pdf_count += 1
            result = {
                "study_id": sid,
                "decision": "pending",
                "reason": "PDF not available",
            }
            checkpoint.mark_done(sid, result)
            review_results.append(result)
            continue

        # Extract text (CACHED — won't re-extract if already done)
        logger.info(f"[{i+1}/{len(studies)}] Reviewing: {sid}")
        try:
            screening_text = extractor.extract_for_screening(str(pdf_path))
        except Exception as e:
            logger.error(f"PDF extraction failed for {sid}: {e}")
            result = {
                "study_id": sid,
                "decision": "pending",
                "reason": f"PDF extraction failed: {e}",
            }
            checkpoint.mark_done(sid, result)
            review_results.append(result)
            continue

        if not screening_text.strip():
            logger.warning(f"Empty PDF text for {sid} — marked pending")
            result = {
                "study_id": sid,
                "decision": "pending",
                "reason": "PDF text extraction returned empty content (possibly scanned/image PDF)",
            }
            checkpoint.mark_done(sid, result)
            review_results.append(result)
            continue

        # === Dual screening ===
        study_design = ", ".join(study.get("publication_types", [])) or "Not specified"
        r1 = screener1.screen_fulltext(study, screening_text, criteria, study_design=study_design)
        r2 = screener2.screen_fulltext(study, screening_text, criteria, study_design=study_design)

        d1 = r1.get("decision", "include")
        d2 = r2.get("decision", "include")

        if d1 == d2:
            result = r1
            result["screener_agreement"] = True
        else:
            conflict_count += 1
            logger.info(f"  Conflict on {sid}: screener1={d1}, screener2={d2} → arbiter")
            result = arbiter.resolve_conflict(study, r1, r2, criteria)
            result["screener_agreement"] = False

        # Always store both screener decisions so κ can be computed after the fact
        result["screener1_decision"] = d1
        result["screener2_decision"] = d2

        checkpoint.mark_done(sid, result)
        review_results.append(result)

        if result.get("decision") != "pending" and _side(result.get("decision", "")) == "exclude":
            exclusion_reasons.append(result)
    
    checkpoint.finalize()

    # Compute κ from all pairs where both screener decisions are present
    ft_sides = []
    for r in review_results:
        d1 = r.get("screener1_decision")
        d2 = r.get("screener2_decision")
        if d1 and d2:
            ft_sides.append((_side(d1), _side(d2)))

    ft_kappa = _compute_kappa(ft_sides) if ft_sides else None
    ft_agreement = (sum(s1 == s2 for s1, s2 in ft_sides) / len(ft_sides)
                    if ft_sides else None)

    # Save results
    dm.save("phase3_screening", "fulltext_review.json", review_results,
            subfolder="stage2_fulltext")
    dm.save("phase3_screening", "exclusion_reasons.json", exclusion_reasons,
            subfolder="stage2_fulltext")
    dm.save("phase3_screening", "fulltext_review_stats.json", {
        "cohens_kappa": round(ft_kappa, 4) if ft_kappa is not None else None,
        "side_agreement_rate": round(ft_agreement, 4) if ft_agreement is not None else None,
        "n_pairs": len(ft_sides),
        "conflict_count": conflict_count,
    }, subfolder="stage2_fulltext")
    
    # Save included studies for Phase 4
    # Use _side() to map 5-point scale → include/exclude
    included_ids = {r["study_id"] for r in review_results
                    if _side(r.get("decision", "")) == "include"}
    included = [s for s in studies if s["study_id"] in included_ids]
    dm.save("phase3_screening", "included_studies.json", included,
            subfolder="stage2_fulltext")
    
    # Generate PRISMA flow data
    stage1_studies = dm.load("phase2_search", "all_studies.json", 
                             subfolder="deduplicated")
    search_log = dm.load("phase2_search", "search_log.json") if dm.exists("phase2_search", "search_log.json") else {}
    
    prisma_flow = {
        "identification": search_log.get("source_counts", {}),
        "total_identified": search_log.get("total_before_dedup", 0),
        "after_deduplication": search_log.get("total_after_dedup", len(stage1_studies)),
        "title_abstract_screened": len(stage1_studies),
        "title_abstract_excluded": len(stage1_studies) - len(studies),
        "fulltext_assessed": len(studies),
        "fulltext_excluded": len(exclusion_reasons),
        "fulltext_no_pdf": no_pdf_count,
        "included_in_synthesis": len(included),
        "exclusion_breakdown": _count_exclusion_reasons(exclusion_reasons),
    }
    dm.save("phase3_screening", "prisma_flow.json", prisma_flow)
    
    _elapsed = time.time() - _phase_start

    # Save validation metrics (#10)
    _bsum_metrics = budget.summary()
    validation_metrics_ft = {
        "phase":               "phase3_stage2",
        "run_date":            datetime.now().isoformat(),
        "fulltext_assessed":   len(studies),
        "included":            len(included),
        "excluded":            len(exclusion_reasons),
        "pending_no_pdf":      no_pdf_count,
        "conflict_count":      conflict_count,
        "cohens_kappa":        round(ft_kappa, 4) if ft_kappa is not None else None,
        "side_agreement_rate": round(ft_agreement, 4) if ft_agreement is not None else None,
        "duration_seconds":    round(_elapsed, 1),
        "duration_minutes":    round(_elapsed / 60, 1),
        "cost_usd":            _bsum_metrics.get("total_cost_usd", 0),
        "total_llm_calls":     _bsum_metrics.get("total_calls", 0),
    }
    dm.save("phase3_screening", "validation_metrics.json", validation_metrics_ft,
            subfolder="stage2_fulltext")

    # Summary
    print("\n" + "="*60)
    print("✅ Phase 3 Stage 2 Complete!")
    print("="*60)
    print(f"  Full-text assessed:    {len(studies)}")
    print(f"  Included:              {len(included)}")
    print(f"  Excluded:              {len(exclusion_reasons)}")
    print(f"  Screener conflicts:    {conflict_count}")
    print(f"  Pending (no PDF):      {no_pdf_count}")
    if ft_kappa is not None:
        print(f"  Cohen's κ (fulltext):  {ft_kappa:.3f} (side agreement {ft_agreement:.1%})")
    print(f"  ⏱️  Duration:           {_elapsed/60:.1f} min ({_elapsed:.0f}s)")
    if no_pdf_count:
        print(f"\n⚠️  {no_pdf_count} studies have 'pending' status (PDF not available).")
        print(f"   These are NOT in the Phase 4 included list.")
        print(f"   Options:")
        print(f"   1. Add PDFs → fulltext_pdfs/ and re-run: --review")
        print(f"   2. Finalize: --finalize-pending to label them as 'missing' and exclude")
        print(f"   (see missing_pdfs_action_list.csv for details)")
    print(f"\n💰 Token budget: {json.dumps(budget.summary(), indent=2)}")
    if no_pdf_count:
        print(f"\n⚠️  Run --finalize-pending before Phase 4 to formally exclude missing PDFs.")
    else:
        print(f"\nNext step: python scripts/run_phase4.py")

    # Langfuse: phase end stamp
    _bsum = budget.summary()
    log_phase_end(_lf_span, "phase3_stage2_review", {
        "fulltext_assessed":   len(studies),
        "included":            len(included),
        "excluded":            len(exclusion_reasons),
        "pending_no_pdf":      no_pdf_count,
        "conflict_count":      conflict_count,
        "cohens_kappa":        ft_kappa,
        "side_agreement_rate": ft_agreement,
        "total_cost_usd":      _bsum.get("total_cost_usd", 0),
        "cache_savings_usd":   _bsum.get("cache_savings_usd", 0),
    })


def finalize_pending(dm: DataManager):
    """
    Formally exclude all 'pending' (no PDF) studies from the pipeline.
    Labels them as 'excluded' with reason 'full-text not available' in the
    PRISMA flow. This is a gate before Phase 4: ensures no pending studies
    leak into data extraction.
    """
    if not dm.exists("phase3_screening", "fulltext_review.json",
                     subfolder="stage2_fulltext"):
        logger.error("❌ No fulltext_review.json found. Run --review first.")
        sys.exit(1)

    review_results = dm.load("phase3_screening", "fulltext_review.json",
                             subfolder="stage2_fulltext")

    pending = [r for r in review_results if r.get("decision") == "pending"]
    if not pending:
        print("\n✅ No pending studies found. All studies have been reviewed.")
        print(f"Next step: python scripts/run_phase4.py")
        return

    print(f"\n📋 Found {len(pending)} pending studies (PDF not available):")
    for r in pending:
        print(f"   - {r['study_id']}: {r.get('reason', 'unknown')}")

    # Update their status
    for r in review_results:
        if r.get("decision") == "pending":
            r["decision"] = "exclude"
            r["exclusion_category"] = "full-text not available"
            r["reason"] = r.get("reason", "PDF not available") + " → excluded at finalization gate"

    dm.save("phase3_screening", "fulltext_review.json", review_results,
            subfolder="stage2_fulltext")

    # Rebuild included list (exclude the newly finalized)
    studies = dm.load("phase3_screening", "included_studies.json",
                      subfolder="stage1_title_abstract")
    included_ids = {r["study_id"] for r in review_results
                    if r.get("decision") == "include"}
    included = [s for s in studies if s["study_id"] in included_ids]
    dm.save("phase3_screening", "included_studies.json", included,
            subfolder="stage2_fulltext")

    # Count exclusion reasons for PRISMA
    excluded_results = [r for r in review_results if r.get("decision") == "exclude"]
    exclusion_reasons = dm.load("phase3_screening", "exclusion_reasons.json",
                                subfolder="stage2_fulltext") if dm.exists(
        "phase3_screening", "exclusion_reasons.json", subfolder="stage2_fulltext") else []
    # Add newly excluded pending studies
    for r in review_results:
        if r.get("exclusion_category") == "full-text not available":
            if r["study_id"] not in {er["study_id"] for er in exclusion_reasons}:
                exclusion_reasons.append(r)
    dm.save("phase3_screening", "exclusion_reasons.json", exclusion_reasons,
            subfolder="stage2_fulltext")

    n_ft_unavail = sum(1 for r in review_results
                       if r.get("exclusion_category") == "full-text not available")

    print(f"\n✅ Finalized {len(pending)} pending studies → excluded (full-text not available)")
    print(f"   Included for Phase 4: {len(included)} studies")
    print(f"   Total excluded: {len(excluded_results)} (of which {n_ft_unavail} full-text unavailable)")
    print(f"\nNext step: python scripts/run_phase4.py")


def _count_exclusion_reasons(exclusion_reasons):
    counts = {}
    for r in exclusion_reasons:
        cat = r.get("exclusion_category", r.get("reason", "other"))
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Stage 2: Full-text Review")
    parser.add_argument("--download",  action="store_true",
                        help="Auto-download PDFs (Unpaywall / PMC / Sci-Hub / etc.)")
    parser.add_argument("--integrate", action="store_true",
                        help="Integrate Zotero PDFs + export missing_pdfs.txt")
    parser.add_argument("--review",    action="store_true",
                        help="Full-text screen all studies that now have PDFs")
    parser.add_argument("--all",       action="store_true",
                        help="--download + --integrate + --review in sequence")
    parser.add_argument("--finalize-pending", action="store_true",
                        help="Label all pending (no PDF) studies as excluded and gate for Phase 4")
    parser.add_argument("--reset-budget", action="store_true",
                        help="Reset accumulated token spend before running")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    needs_stage1 = args.all or args.download or args.integrate or args.review
    if needs_stage1:
        if not dm.exists("phase3_screening", "included_studies.json",
                         subfolder="stage1_title_abstract"):
            logger.error(
                "❌ Stage 1 included_studies.json not found. "
                "Run Phase 3 Stage 1 first: python scripts/run_phase3_stage1.py"
            )
            sys.exit(1)

    if args.all or args.download:
        download_pdfs(dm)

    if args.all or args.integrate:
        integrate_manual_pdfs(dm)

    if args.all or args.review:
        review_fulltexts(dm, reset_budget=args.reset_budget)

    if args.finalize_pending:
        finalize_pending(dm)

    if not (args.download or args.integrate or args.review or args.all or args.finalize_pending):
        print("Usage: python scripts/run_phase3_stage2.py [--download] [--integrate] [--review] [--all] [--finalize-pending]")
        print()
        print("  --download          Auto-download PDFs (Unpaywall / PMC / Sci-Hub)")
        print("  --integrate         Integrate Zotero PDFs + export missing_pdfs.txt")
        print("                      (run after adding PDFs to data/zotero_export/)")
        print("  --review            Full-text screen studies that now have PDFs")
        print("                      (safe to re-run: skips already-reviewed, picks up new PDFs)")
        print("  --all               Run all three steps in sequence")
        print("  --finalize-pending  Label all pending (no-PDF) studies as excluded")
        print("                      (gate before Phase 4 — ensures clean handoff)")
        print()
        print("Recommended workflow:")
        print("  1. python scripts/run_phase3_stage2.py --download")
        print("  2. python scripts/run_phase3_stage2.py --integrate")
        print("     → check data/phase3_screening/stage2_fulltext/missing_pdfs.txt")
        print("     → add PDFs to data/zotero_export/ and repeat --integrate as needed")
        print("  3. python scripts/run_phase3_stage2.py --review")
        print("  4. python scripts/run_phase3_stage2.py --finalize-pending")
        print("     → formally exclude studies with no PDF before Phase 4")


if __name__ == "__main__":
    main()