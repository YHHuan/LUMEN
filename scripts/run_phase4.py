#!/usr/bin/env python3
"""
Phase 4: Data Extraction
=========================
啟動方式:
  python scripts/run_phase4.py
  python scripts/run_phase4.py --resume           # 從斷點恢復
  python scripts/run_phase4.py --validate-only    # 只做驗證

⚠️ Token 節省設計:
  - 重用 Phase 3 的 PDF 文字快取 (不重新提取!)
  - 提取用更長版本 (~12000 tokens vs 篩選的 5000)
  - Checkpoint: 每提取完一篇就存檔

前置需求: Phase 3 Stage 2 完成
輸入: data/phase3_screening/stage2_fulltext/included_studies.json
輸出: data/phase4_extraction/
"""

import os
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
from src.utils.langfuse_client import log_phase_start, log_phase_end
from src.agents.extractor import ExtractorAgent
from src.utils.pdf_downloader import PDFTextExtractor
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget, Checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Data Extraction")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing data")
    parser.add_argument("--reset-budget", action="store_true",
                        help="Reset accumulated token spend before running (use when re-running a phase from scratch)")
    args = parser.parse_args()

    select_project()
    dm = DataManager()
    _phase_start = time.time()
    from dotenv import load_dotenv
    load_dotenv()
    budget_limit = float(os.getenv("TOKEN_BUDGET_PHASE4", "15.0"))
    budget = TokenBudget(phase="phase4", limit_usd=budget_limit, reset=args.reset_budget)

    # Load included studies
    studies = dm.load("phase3_screening", "included_studies.json",
                      subfolder="stage2_fulltext")
    pico = dm.load("input", "pico.yaml")
    outcome_defs = pico["pico"]["outcome"]

    # Load extraction guidance from Phase 1 (optional — backward compatible)
    extraction_guidance = {}
    if dm.exists("phase1_strategy", "extraction_guidance.json"):
        extraction_guidance = dm.load("phase1_strategy", "extraction_guidance.json")
        n_measures = len(extraction_guidance.get("outcome_measures", []))
        logger.info(f"📋 Extraction guidance loaded: {n_measures} outcome measures, "
                     f"timepoint={extraction_guidance.get('preferred_timepoint', 'N/A')}")
    else:
        logger.info("ℹ️  No extraction_guidance.json found — using default extraction behavior. "
                     "Re-run Phase 1 to generate extraction guidance.")

    logger.info(f"Studies to extract: {len(studies)}")

    # Langfuse: phase start stamp
    import yaml
    from pathlib import Path as _Path
    _models_cfg = yaml.safe_load((_Path(__file__).parent.parent / "config/models.yaml").read_text())
    _lf_span = log_phase_start("phase4_extraction", {
        "input_study_count":   len(studies),
        "extractor_model":     _models_cfg["models"]["extractor"]["model_id"],
        "tiebreaker_model":    _models_cfg["models"]["extractor_tiebreaker"]["model_id"],
        "budget_usd":          15.0,
    })
    
    extractor_agent = ExtractorAgent(budget=budget)
    text_extractor = PDFTextExtractor()  # Uses cache from Phase 3!
    
    checkpoint = Checkpoint("phase4_extraction")
    checkpoint.set_total(len(studies))
    
    extracted_data = []
    rob_data = []
    
    pdf_dir = Path(get_data_dir()) / "phase3_screening" / "stage2_fulltext" / "fulltext_pdfs"
    
    for i, study in enumerate(studies):
        sid = study["study_id"]
        
        if checkpoint.is_done(sid):
            result = checkpoint.get_result(sid)
            extracted_data.append(result.get("extraction", {}))
            rob_data.append(result.get("rob", {}))
            continue
        
        # Get PDF text (CACHED from Phase 3 - no re-extraction!)
        safe_id = sid.replace("/", "_").replace("\\", "_")
        pdf_path = pdf_dir / f"{safe_id}.pdf"
        
        if not pdf_path.exists():
            logger.warning(f"⚠️  No PDF for {sid} — excluded from extraction. "
                           f"Re-run after placing {safe_id}.pdf in fulltext_pdfs/ and "
                           f"deleting data/.checkpoints/phase4_extraction.json")
            checkpoint.mark_failed(sid, "No PDF")
            continue
        
        logger.info(f"[{i+1}/{len(studies)}] Extracting: {sid}")

        # Use longer extraction (12000 tokens vs 5000 for screening)
        # Returns (text, tables) — both are passed to the extractor so the
        # LLM sees pdfplumber-extracted numeric tables alongside the prose.
        fulltext, tables = text_extractor.extract_for_data_extraction(str(pdf_path))
        if not fulltext.strip():
            logger.warning(f"Empty PDF text for {sid} — skipping extraction")
            checkpoint.mark_failed(sid, "Empty PDF text")
            continue

        # Extract structured data (with extraction guidance from Phase 1 if available)
        extraction = extractor_agent.extract_data(
            study, fulltext, outcome_defs,
            tables=tables, extraction_guidance=extraction_guidance
        )
        
        # Assess Risk of Bias
        rob = extractor_agent.assess_risk_of_bias(study, fulltext)
        
        checkpoint.mark_done(sid, {"extraction": extraction, "rob": rob})
        extracted_data.append(extraction)
        rob_data.append(rob)
    
    checkpoint.finalize()

    # Count how many studies were skipped due to missing PDFs
    no_pdf_count = sum(
        1 for sid, err in checkpoint.state.get("failed", {}).items()
        if "No PDF" in str(err) or "Empty PDF" in str(err)
    )

    # Save
    dm.save("phase4_extraction", "extracted_data.json", extracted_data)
    dm.save("phase4_extraction", "risk_of_bias.json", rob_data)

    # Validation report
    validation = _validate_extraction(extracted_data)
    dm.save("phase4_extraction", "validation_report.json", validation)

    _elapsed = time.time() - _phase_start

    print("\n" + "="*60)
    print("✅ Phase 4 Complete!")
    print("="*60)
    print(f"  Studies attempted:          {len(studies)}")
    print(f"  Studies extracted:          {len(extracted_data)}")
    print(f"  Skipped (no/empty PDF):     {no_pdf_count}")
    print(f"  With complete outcome data: {validation['complete_outcomes']}")
    print(f"  With issues:                {validation['with_issues']}")
    print(f"  ⏱️  Duration:               {_elapsed/60:.1f} min ({_elapsed:.0f}s)")
    if no_pdf_count:
        print(f"\n⚠️  {no_pdf_count} studies were skipped because their PDFs are missing.")
        print(f"   Add PDFs to data/phase3_screening/stage2_fulltext/fulltext_pdfs/")
        print(f"   then delete data/.checkpoints/phase4_extraction.json and re-run.")
    print(f"\n💰 Token budget: {json.dumps(budget.summary(), indent=2)}")
    print(f"\nNext step: python scripts/run_phase5.py")

    # Auto-diagnose (#3): run quality check after extraction completes
    print("\n🔍 Running post-extraction quality check (diagnose_phase4.py)...")
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "diagnose_phase4.py")],
            input="1\n",  # auto-select active project
            capture_output=True, text=True, timeout=60
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0 and result.stderr:
            logger.warning(f"diagnose_phase4 warnings: {result.stderr[:500]}")
    except Exception as e:
        logger.warning(f"Auto-diagnose skipped: {e}")

    # Langfuse: phase end stamp
    _bsum = budget.summary()
    log_phase_end(_lf_span, "phase4_extraction", {
        "studies_attempted":       len(studies),
        "studies_extracted":       len(extracted_data),
        "skipped_no_pdf":          no_pdf_count,
        "complete_outcome_data":   validation["complete_outcomes"],
        "with_issues":             validation["with_issues"],
        "total_cost_usd":          _bsum.get("total_cost_usd", 0),
        "cache_savings_usd":       _bsum.get("cache_savings_usd", 0),
    })


def _validate_extraction(data: list) -> dict:
    """驗證提取的資料完整性 (v5: handles _post/_change keys)"""
    complete = 0
    with_issues = 0
    issues = []
    
    for d in data:
        sid = d.get("study_id", "unknown")
        study_issues = []
        
        if not d.get("characteristics", {}).get("n_total"):
            study_issues.append("Missing total N")
        
        if not d.get("intervention", {}).get("type"):
            study_issues.append("Missing intervention type")
        
        outcomes = d.get("outcomes", {})
        if not outcomes:
            study_issues.append("No outcomes extracted")
        else:
            for outcome_name, outcome_data in outcomes.items():
                has_effect = False
                # Check all possible data paths
                for suffix in ["", "_post", "_change"]:
                    m = outcome_data.get(f"intervention_mean{suffix}")
                    s = outcome_data.get(f"intervention_sd{suffix}")
                    n = outcome_data.get("intervention_n")
                    if m is not None and s is not None and n is not None:
                        has_effect = True
                        break
                if not has_effect:
                    # Check pre-computed effects
                    if (outcome_data.get("smd") is not None or
                        (outcome_data.get("mean_difference") is not None and
                         (outcome_data.get("md_95ci_lower") is not None or
                          outcome_data.get("md_se") is not None))):
                        has_effect = True
                
                if not has_effect:
                    study_issues.append(f"No effect data for {outcome_name}")
        
        if study_issues:
            with_issues += 1
            issues.append({"study_id": sid, "issues": study_issues})
        else:
            complete += 1
    
    return {
        "total": len(data),
        "complete_outcomes": complete,
        "with_issues": with_issues,
        "issues_detail": issues,
    }


if __name__ == "__main__":
    main()
