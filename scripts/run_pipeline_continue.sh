#!/bin/bash
# ============================================================
# Pipeline Runner: Phase 3.1 → 3.2 → 4 → 5 → 6
# Runs each phase sequentially, checks outputs, aborts on error.
# ============================================================
set -euo pipefail

PROJECT_DIR="/mnt/c/Users/micha/Downloads/code to upload/v5-package"
LOG="$PROJECT_DIR/pipeline_run.log"
cd "$PROJECT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

run_phase() {
    local label="$1"; shift
    log "========================================"
    log "STARTING: $label"
    log "========================================"
    echo "1" | python3 "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[1]}
    if [ $rc -ne 0 ]; then
        log "ERROR: $label exited with code $rc — aborting."
        exit $rc
    fi
    log "DONE: $label"
    echo ""
}

check_file() {
    local f="$1" label="$2"
    if [ ! -f "$f" ]; then
        log "ERROR: Expected output missing after $label: $f"
        exit 1
    fi
    local n
    n=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
    log "  → $label output: $n records in $(basename $f)"
}

log "========================================"
log "Pipeline started (Phase 3.1 → 3.2 → 4 → 5 → 6)"
log "Input studies (new prescreen): 1947"
log "========================================"

# ── Phase 3 Stage 1: T/A Screening ───────────────────────
run_phase "Phase 3 Stage 1 (T/A screening)" \
    scripts/run_phase3_stage1.py

check_file \
    "data/PSY_MCI_rTMS/phase3_screening/stage1_title_abstract/included_studies.json" \
    "Phase 3.1"

# Stats snapshot
python3 -c "
import json; from pathlib import Path
sr = Path('data/PSY_MCI_rTMS/phase3_screening/stage1_title_abstract/screening_results.json')
if sr.exists():
    d = json.loads(sr.read_text())
    print(f'  Screened: {d.get(\"total_screened\")}  Included: {len(d.get(\"included\",[]))}  '
          f'κ={d.get(\"cohens_kappa\",0):.3f}  Agree={d.get(\"agreement_rate\",0):.1%}')
bf = Path('data/PSY_MCI_rTMS/.budget/phase3_stage1_budget.json')
if bf.exists():
    b = json.loads(bf.read_text())
    print(f'  Cost: \${b.get(\"total_cost_usd\",0):.4f}  '
          f'Cache read: {b.get(\"total_cache_read_tokens\",0):,} tok')
" 2>/dev/null | tee -a "$LOG"

# ── Phase 3 Stage 2: Full-text review (no download) ──────
run_phase "Phase 3 Stage 2 (full-text review)" \
    scripts/run_phase3_stage2.py --review

check_file \
    "data/PSY_MCI_rTMS/phase3_screening/stage2_fulltext/included_studies.json" \
    "Phase 3.2"

# Cross-extraction duplicate check
log "Running cross-extraction duplicate check (report only)..."
echo "1" | python3 scripts/diagnose_duplicates.py 2>&1 | tee -a "$LOG" || true

# ── Phase 4: Data Extraction ──────────────────────────────
run_phase "Phase 4 (data extraction)" \
    scripts/run_phase4.py

check_file \
    "data/PSY_MCI_rTMS/phase4_extraction/extracted_data.json" \
    "Phase 4"

# ── Phase 5: Statistical Analysis ────────────────────────
run_phase "Phase 5 (statistics)" \
    scripts/run_phase5.py

# ── Phase 6: Manuscript Writing ──────────────────────────
run_phase "Phase 6 (manuscript)" \
    scripts/run_phase6.py

# ── Final Summary ─────────────────────────────────────────
log "========================================"
log "ALL PHASES COMPLETE"
log "========================================"
echo "1" | python3 scripts/check_progress.py 2>&1 | tee -a "$LOG"

log "Full log: $LOG"
