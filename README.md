# Meta-Analysis Generator

Python CLI pipeline for automated systematic reviews and meta-analyses.
Uses a chain of LLM agents (OpenRouter) built around the PICO framework.

**Multi-project support:** Each research question lives in its own directory under `data/<project_name>/`. Every script prompts for project selection at startup. Run multiple independent meta-analyses without interference.

---

## Pipeline at a Glance

```
Phase 1    Strategy Generation     → MeSH terms, search queries, Cochrane/Embase instructions
Phase 2    Literature Search       → API search + manual .ris imports + deduplication
Phase 2.5  Pre-screening (FREE)    → structural + keyword filter, zero LLM cost
Phase 3.1  Title/Abstract Screen   → dual LLM screening, 5-point scale, conflict resolution
Phase 3.2  Full-text Review        → PDF download + Zotero import + full-text screening
Phase 4    Data Extraction         → LLM extracts outcomes, characteristics, RoB from PDFs
[Dedup]    Duplicate Check         → detect & remove cross-database duplicate studies
Phase 5    Statistical Analysis    → meta-analysis, forest/funnel plots, PRISMA diagram
Phase 6    Manuscript Writing      → introduction, results, discussion + citation validation
```

---

## Quick Start

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env              # Then edit .env with your keys

# 3. Define your research question
#    Each script will prompt for project selection.
#    For a new project, choose [N] and edit data/<project>/input/pico.yaml

# 4. Run the pipeline (project selector appears at startup)
python scripts/run_phase1.py
python scripts/run_phase2.py
python scripts/run_phase2_5_prescreen.py
python scripts/run_phase3_stage1.py
python scripts/run_phase3_stage2.py --all
python scripts/run_phase3_stage2.py --finalize-pending   # Exclude no-PDF studies
python scripts/run_phase4.py
python scripts/diagnose_duplicates.py --fix   # Remove cross-database duplicates
python scripts/run_phase5.py --builtin-only
python scripts/export_prisma_diagram.py
python scripts/run_phase6.py
```

---

## Phase-by-Phase Commands

### Phase 1: Strategy Generation

Generates search queries, MeSH terms, screening criteria, and manual search instructions for Cochrane and Embase.

```bash
python scripts/run_phase1.py
```

**Input:** `data/input/pico.yaml`

**Output:** `data/phase1_strategy/`

| File | Contents |
|------|----------|
| `search_strategy.json` | Auto-run queries for PubMed, Europe PMC, Scopus |
| `mesh_terms.json` | Validated MeSH headings |
| `mesh_validation.json` | MeSH correction report |
| `screening_criteria.json` | Inclusion/exclusion criteria prose |
| `study_design_filter.json` | Dynamic study design filter for Phase 2.5 pre-screen |
| `extraction_guidance.json` | Outcome measure scales, timepoint rules for Phase 4 |
| `manual_search_cochrane.txt` | ⭐ **Copy-paste strategy for Cochrane Library** |
| `manual_search_embase.txt` | ⭐ **Numbered Ovid strategy for Embase** |

> **MeSH term accuracy:** Phase 1 uses an LLM to generate and validate MeSH terms.
> Always review `mesh_terms.json` before running Phase 2, especially for rare topics.
> Verify Embase EMTREE terms independently — EMTREE differs from MeSH.

If you already ran Phase 1 and just need to regenerate the manual search files:
```bash
python scripts/export_manual_queries.py
```

---

### Phase 2: Literature Search

Searches API databases and imports manual exports, then deduplicates everything.

```bash
python scripts/run_phase2.py                # Search all enabled databases + deduplicate
python scripts/run_phase2.py --show-queries  # Preview queries (no API calls)
python scripts/run_phase2.py --deduplicate   # Only deduplicate (skip search)
```

**Output:** `data/phase2_search/raw/` (per-database), `data/phase2_search/deduplicated/all_studies.json`

#### Manual Database Import (Cochrane, Embase, Web of Science)

These databases have no free API. Export manually using the generated strategies, then place `.ris` files in the raw folder:

```
data/phase2_search/raw/
  ├── pubmed_results.json        ← auto-generated
  ├── europepmc_results.json     ← auto-generated
  ├── scopus_results.json        ← auto-generated
  ├── cochrane_search.ris        ← 🔧 MANUAL (use manual_search_cochrane.txt)
  ├── embase_export.ris          ← 🔧 MANUAL (use manual_search_embase.txt)
  └── wos_export.ris             ← 🔧 MANUAL (Web of Science → Export → RIS)
```

**How to export:**
1. **Cochrane Library** → Advanced Search → paste strategy from `manual_search_cochrane.txt` → Export → RIS
2. **Embase via Ovid** → enter numbered lines from `manual_search_embase.txt` → Export → RIS
   ⚠️ Verify EMTREE terms before running (see notes in the file)
3. **Web of Science** → Search → Export → Other File Formats → RIS

After placing `.ris` files, re-run deduplication:
```bash
python scripts/run_phase2.py --deduplicate
```

**It is safe to add `.ris` files at any time.** `--deduplicate` only re-merges — it does not re-query APIs. Downstream phases use study-level checkpoints, so only newly added studies incur LLM cost.

**Full chain after adding a new .ris file:**
```bash
python scripts/run_phase2.py --deduplicate
python scripts/run_phase2_5_prescreen.py
python scripts/run_phase3_stage1.py           # only NEW studies cost tokens
python scripts/run_phase3_stage2.py --download
python scripts/run_phase3_stage2.py --integrate
python scripts/run_phase3_stage2.py --review
```

> **Note for Methods writing:** `search_log.json` records how many records came from each source, the UTC timestamp of when the search was executed (`search_executed_at`), and all model `pinned_at` dates (`model_pinned_at_dates`) for TRIPOD-LLM Item 5c compliance. Phase 6 uses this to write the "Information Sources" subsection. Use descriptive filenames for `.ris` files (e.g. `cochrane_search.ris`) — the filename becomes the PRISMA source label.

---

### Phase 2.5: Pre-screening (FREE — no LLM cost)

Removes obvious non-relevant records before LLM screening.

```bash
python scripts/run_phase2_5_prescreen.py
```

**Input:** `data/phase2_search/deduplicated/all_studies.json`
**Output:** `data/phase2_search/prescreened/filtered_studies.json`, `prescreen_excluded_log.json`

**Four-layer filtering:**

**Layer 1 — Structural rules** (applied first, exact pattern matching):
- Registry DOIs (`10.31525/ct1-*`) or registry IDs as author (NCT, ChiCTR, DRKS, ACTRN, etc.)
- Explicit publication types: `clinical trial protocol`, `systematic review`, `meta-analysis`, `editorial`, `letter`, `meeting abstract`, `conference paper`, `preprint`, `erratum`, `corrigendum`
- Title-level patterns: titles starting with `Re:`, `Letter to`, `Correspondence:`, `Erratum`, `Retraction of`, etc.
- Title ≤ 10 characters (registry entries or truncated garbage)
- No abstract AND no DOI/PMID (incomplete or registry record)
- Conference/supplement journal name (any abstract length)

**Layer 2 — Study design filter** (dynamic, from Phase 1 Strategist):
- Reads `study_design_filter.json` generated by Phase 1
- `strict` mode: excludes studies matching `design_exclusion_keywords` (e.g., "observational", "retrospective" for RCT-only reviews)
- `loose` mode: only excludes studies matching explicitly out-of-scope designs
- Adapts to different PICO requirements — not hardcoded

**Layer 3 — Keyword rules** (from Phase 1 strategy + built-in defaults):
- Animal/in-vitro terms in **title only**: `rat`, `mice`, `in vitro`, `cell culture`, etc.
- Non-primary publications in title+abstract: `review`, `protocol`, `case report`, etc.

**Layer 4 — Cross-database deduplication** (catch-net after filtering):
- Re-runs `deduplicate_studies()` on the filtered pool to catch duplicates from late RIS imports or missed by Phase 2 fuzzy matching
- Duplicates removed here are logged in `prescreen_excluded_log.json` with reason `"Cross-database duplicate of <study_id>"`
- Saves ~2 LLM screening calls per duplicate caught

> **Re-running is safe and free.** If you add new `.ris` files, just re-run Phase 2.5 before Phase 3.

---

### Phase 3 Stage 1: Title/Abstract Screening

Dual independent LLM screening with 5-point confidence scale and conflict resolution via Arbiter.

```bash
python scripts/run_phase3_stage1.py
```

**Input:** Prescreened studies
**Output:**
- `included_studies.json` — studies passing screening
- `excluded_studies.json`
- `human_review_queue.json` — undecided conflicts (review manually)
- `screening_results.json` — full dual-screening data with Cohen's κ

---

### Phase 3 Stage 2: Full-text Download & Review

```bash
# Recommended workflow
python scripts/run_phase3_stage2.py --download           # 1. Auto-download PDFs (10-source cascade)
python scripts/run_phase3_stage2.py --integrate          # 2. Match Zotero PDFs + show what's missing
# → add PDFs to data/zotero_export/ and repeat --integrate
python scripts/run_phase3_stage2.py --review             # 3. Full-text screening
python scripts/run_phase3_stage2.py --finalize-pending   # 4. Gate: exclude no-PDF studies before Phase 4

# Or all at once (steps 1-3)
python scripts/run_phase3_stage2.py --all
```

**`--finalize-pending`** is a gate before Phase 4: it formally labels all studies with missing PDFs as "excluded (full-text not available)" in the PRISMA flow and removes them from the Phase 4 included list. Run this after you've exhausted all PDF retrieval options.

**`--review` is always safe to re-run.** Already-reviewed studies replay from checkpoint (zero LLM cost); only studies newly with PDFs are screened.

**PDF locations:**
```
data/phase3_screening/stage2_fulltext/
  ├── fulltext_pdfs/        ← PDFs stored here (auto + manual)
  ├── download_log.json
  ├── missing_pdfs.txt      ← DOIs + titles of studies still missing a PDF
  ├── fulltext_review.json  ← screening decisions
  └── included_studies.json ← final included list → Phase 4 input
```

#### Getting missing PDFs

After `--integrate`, `missing_pdfs.txt` has two sections:
- **With DOI** (paste-ready): use with Zotero "Add by Identifier", institutional proxy, or direct download
- **No DOI**: manual search using registry ID, year, author, and title

```bash
# Zotero workflow for each batch:
# 1. Download PDFs via browser or Zotero
# 2. Drop into data/zotero_export/ (subfolders fine)
# 3. Run --integrate to match and copy:
python scripts/run_phase3_stage2.py --integrate
```

#### Zotero PDF filename matching

The `--integrate` step calls `scripts/rename_zotero_pdfs.py` which matches PDFs by:
1. PMID in filename → score 1.0
2. DOI in filename → score 1.0
3. Title similarity (SequenceMatcher + containment + first-words) → score 0.55–0.95

**Handles variable filenames:**
Filenames like `"A Randomized Trial - Author 2023"` or `"Author 2023 - a randomized trial, XXXX"` are matched to the study via multi-strategy title comparison. The matching is not sensitive to word order.

```bash
python scripts/rename_zotero_pdfs.py --dry-run          # preview all matches
python scripts/rename_zotero_pdfs.py --threshold 0.65   # raise threshold for stricter matching
python scripts/rename_zotero_pdfs.py --diagnose         # coverage report
```

Known bad matches (wrong modality, etc.) can be added to `SKIP_FILES` dict at the top of `rename_zotero_pdfs.py`.

---

### Phase 4: Data Extraction

LLM extracts outcome data (means, SDs, Ns), study characteristics, and Risk of Bias from PDFs.

```bash
python scripts/run_phase4.py                  # Run extraction
python scripts/run_phase4.py --validate-only  # Validate without re-extracting
```

**To re-run** (e.g. after adding PDFs):
```bash
rm data/.checkpoints/phase4_extraction.json   # Linux/Mac
del data\.checkpoints\phase4_extraction.json  # Windows CMD
python scripts/run_phase4.py
```

**Input:** `data/phase3_screening/stage2_fulltext/included_studies.json` + PDFs
**Output:** `data/phase4_extraction/extracted_data.json`, `risk_of_bias.json`, `validation_report.json`

---

### Duplicate Detection (After Phase 4)

The same paper may appear in multiple databases (PubMed, Cochrane, Embase) with different IDs but the same content. The LLM will assign them identical citations, causing inflated study counts in forest plots. **Always run this before Phase 5:**

```bash
python scripts/diagnose_duplicates.py          # Report only (safe)
python scripts/diagnose_duplicates.py --fix    # Remove duplicates
python scripts/diagnose_duplicates.py --fix --dry-run   # Preview changes

# Then re-run statistics and manuscript
python scripts/run_phase5.py --builtin-only
python scripts/run_phase6.py
```

Priority for keeping canonical record: `PMID_` > `EPMC_` > `SCOPUS_` > `RIS_`; tiebreaker: richest outcome data.

---

### Phase 5: Statistical Analysis

Random-effects meta-analysis, subgroup analyses, forest/funnel plots, RoB figures.

```bash
python scripts/run_phase5.py                  # Full analysis (includes LLM interpretation)
python scripts/run_phase5.py --builtin-only   # Pure statistics, no LLM (recommended)
```

**Output:**
```
data/phase5_analysis/
  ├── statistical_results.json
  ├── safety_summary.json
  └── figures/
      ├── forest_{outcome}_{measure}.png
      ├── funnel_{outcome}_{measure}.png
      ├── rob_summary.png
      └── rob_domains.png
```

#### PRISMA Flow Diagram

Generate a publication-ready PRISMA 2020 diagram:

```bash
python scripts/export_prisma_diagram.py
# → data/phase5_analysis/figures/prisma_flow.png

# If you ran diagnose_duplicates.py --fix, update the included count:
python scripts/export_prisma_diagram.py --included 173
```

---

### Phase 6: Manuscript Writing

LLM writes each section with full data context, then validates citations.

```bash
python scripts/run_phase6.py                                    # Full manuscript
python scripts/run_phase6.py --section discussion               # Single section
python scripts/run_phase6.py --sections introduction,discussion # Multiple sections
python scripts/run_phase6.py --skip-validation                  # Skip citation checks
python scripts/run_phase6.py --validate-only                    # Only citation validation
```

**Output:**
```
data/phase6_manuscript/drafts/
  ├── title.md
  ├── abstract.md
  ├── introduction.md
  ├── methods.md
  ├── results.md
  ├── discussion.md
  └── manuscript_draft.md          ← combined
```

`[CITATION NEEDED: topic]` markers are collected and the Citation Guardian suggests references.

---

## Diagnostic Tools

```bash
python scripts/check_progress.py              # Overall pipeline progress
python scripts/diagnose_pipeline.py           # Full quality audit (Phase 2→3→4)
python scripts/diagnose_phase4.py             # Phase 4 data quality report
python scripts/diagnose_phase4.py --fix       # Auto-fix SE→SD conversions
python scripts/diagnose_duplicates.py         # Detect cross-database duplicates (Phase 4)
python scripts/diagnose_duplicates.py --fix   # Remove duplicates (then re-run Phase 5 & 6)
python scripts/rename_zotero_pdfs.py --diagnose   # PDF coverage report
python scripts/rename_zotero_pdfs.py --dry-run    # Preview Zotero PDF matching
python scripts/rename_zotero_pdfs.py              # Apply Zotero PDF matching
python scripts/extract_DOI.py                     # Export DOIs for Zotero batch import
python scripts/export_manual_queries.py           # Re-generate Cochrane/Embase search instructions
python scripts/export_prisma_diagram.py           # Generate PRISMA 2020 flow diagram
```

---

## Methodology Validation

For formal methodology validation (required for JAMIA-style publication):

### Screening Validation

```bash
# Step 1: Export 100-abstract sample for human annotation
python scripts/validate_screening.py --export --n 100

# Check progress
python scripts/validate_screening.py --status

# Step 2: After annotating, compute metrics
python scripts/validate_screening.py --compute
```

Open `data/validation/screening_validation_sample.csv`, fill the `human_decision` column (include/exclude) for each row, then run `--compute`.

**Metrics:** Accuracy, Precision, Sensitivity, Specificity, F1, Cohen's κ, PABAK, WSS@95%
**Time:** ~2–4 hours for 100 records

### Extraction Validation

```bash
# Step 1: Export extracted fields for PDF verification
python scripts/validate_extraction.py --export
python scripts/validate_extraction.py --export --complete-only  # complete studies only

# Check progress
python scripts/validate_extraction.py --status

# Step 2: After annotating, compute metrics
python scripts/validate_extraction.py --compute
```

Open `data/validation/extraction_validation.csv`, find each field value in the PDF, fill `actual_value`, then run `--compute`.

**Metrics:** Precision, Recall, F1 — broken down by category (study_design, population, intervention, outcomes) and source type (text-based vs table-based)
**Time:** ~4–6 hours for complete studies

---

## Data Folder Structure

Each project has its own isolated data directory:

```
data/
  .active_project                              ← last selected project name
  PSY_MCI_rTMS/                                ← project directory
  input/
    └── pico.yaml                              ← YOUR RESEARCH QUESTION
  phase1_strategy/
    ├── search_strategy.json                   ← auto-run queries
    ├── mesh_terms.json                        ← validated MeSH headings
    ├── mesh_validation.json
    ├── screening_criteria.json
    ├── study_design_filter.json               ← dynamic design filter for Phase 2.5
    ├── extraction_guidance.json               ← outcome scales + timepoint rules for Phase 4
    ├── manual_search_cochrane.txt             ← ⭐ Cochrane Library strategy
    ├── manual_search_embase.txt               ← ⭐ Embase/Ovid strategy
    └── prisma_protocol.json
  phase2_search/
    ├── raw/                                   ← API results + manual .ris/.csv imports
    │   ├── pubmed_results.json
    │   ├── europepmc_results.json
    │   ├── scopus_results.json
    │   ├── cochrane_search.ris                ← 🔧 MANUAL
    │   └── embase_export.ris                  ← 🔧 MANUAL
    ├── deduplicated/
    │   └── all_studies.json
    ├── prescreened/
    │   ├── filtered_studies.json              ← Phase 2.5 output
    │   └── prescreen_excluded_log.json
    └── search_log.json                        ← source counts for PRISMA
  phase3_screening/
    ├── stage1_title_abstract/
    │   ├── included_studies.json
    │   ├── excluded_studies.json
    │   ├── human_review_queue.json            ← undecided conflicts — review manually
    │   ├── screening_results.json             ← dual-screening data + κ stats
    │   ├── conflicts.json
    │   └── arbiter_decisions.json
    ├── stage2_fulltext/
    │   ├── fulltext_pdfs/                     ← all PDFs (auto-downloaded + manually placed)
    │   ├── included_studies.json              ← FINAL included list → Phase 4 input
    │   ├── fulltext_review.json
    │   ├── exclusion_reasons.json
    │   ├── download_log.json
    │   └── missing_pdfs.txt                   ← updated by --integrate
    └── prisma_flow.json                       ← PRISMA flow data
  phase4_extraction/
    ├── extracted_data.json
    ├── risk_of_bias.json
    ├── validation_report.json
    └── duplicate_report.csv                   ← generated by diagnose_duplicates.py
  phase5_analysis/
    ├── statistical_results.json
    ├── safety_summary.json
    └── figures/
        ├── forest_{outcome}_{measure}.png
        ├── funnel_{outcome}_{measure}.png
        ├── rob_summary.png
        ├── rob_domains.png
        └── prisma_flow.png                    ← generated by export_prisma_diagram.py
  phase6_manuscript/
    └── drafts/
  zotero_export/                               ← 🔧 Drop Zotero PDFs here (subfolders fine)
  validation/
    ├── screening_validation_sample.csv        ← generated by validate_screening.py
    ├── screening_metrics.csv
    ├── extraction_validation.csv              ← generated by validate_extraction.py
    └── extraction_metrics.csv
  .cache/                                      ← LLM cache (do not delete)
  .checkpoints/                                ← resume checkpoints (delete to re-run a phase)
  .audit/
    └── prompt_log.jsonl                       ← PRISMA-trAIce audit trail
```

---

## Configuration (.env)

```ini
# Required
OPENROUTER_API_KEY=sk-or-...

# PubMed (free, but key recommended for higher rate limits)
NCBI_API_KEY=...
NCBI_EMAIL=your@email.com

# Scopus (optional — requires Elsevier API key)
ELSEVIER_API_KEY=...

# CrossRef & Unpaywall (optional — just an email for politeness)
CROSSREF_EMAIL=your@email.com
UNPAYWALL_EMAIL=your@email.com

# Database toggles (set to false to skip)
ENABLE_PUBMED=true
ENABLE_EUROPEPMC=true
ENABLE_SCOPUS=true
ENABLE_CROSSREF=true

# Token budgets (USD per phase)
TOKEN_BUDGET_PHASE1=5.0
TOKEN_BUDGET_PHASE3_TA=10.0
TOKEN_BUDGET_PHASE3_FT=12.0
TOKEN_BUDGET_PHASE4=30.0
```

---

## Windows Notes

| Linux/Mac | Windows CMD | PowerShell |
|-----------|-------------|------------|
| `rm file` | `del file` | `Remove-Item file` |
| `rm -rf dir/` | `rmdir /s /q dir` | `Remove-Item -Recurse dir` |
| `cat file` | `type file` | `Get-Content file` |

---

## Architecture

### Agent Roles

| Agent | Model | Phase | Cache |
|-------|-------|-------|-------|
| Strategist | Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-6`) | 1 — PICO → MeSH + queries | ephemeral (Anthropic) |
| Screener 1 | Gemini 3.1 Flash Lite (`google/gemini-3.1-flash-lite-preview`) | 3.1 — 5-point scale + thinking | internal |
| Screener 2 | GPT-4.1 Mini (`openai/gpt-4.1-mini`) | 3.1 — 5-point confidence scale | auto (OpenAI) |
| Arbiter | Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-6`) | 3.1 — firm conflict resolution | ephemeral (Anthropic) |
| Extractor | Gemini 3.1 Pro (`google/gemini-3.1-pro-preview`) | 4 — PDF → structured data | internal |
| Extractor Tiebreaker | GPT-5.4 (`openai/gpt-5.4`) | 4 — self-consistency check | auto (OpenAI) |
| Statistician | GPT-5.4 (`openai/gpt-5.4`) | 5 — supplementary analysis code | auto (OpenAI) |
| Writer | Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-6`) | 6 — manuscript sections | ephemeral (Anthropic) |
| Citation Guardian | GPT-5.4 (`openai/gpt-5.4`) | 6 — citation validation | auto (OpenAI) |

> Model IDs and pricing are defined in `config/models.yaml`. Update `pinned_at` dates when you want to adopt newer model versions.

### Source Layout

```
src/
  agents/       BaseAgent, strategist, screener, extractor, writer, statistician
  apis/         pubmed, europepmc, scopus, crossref, unpaywall
  utils/        cache, deduplication, effect_sizes, file_handlers,
                normalizer, pdf_downloader (10-source), query_syntax,
                statistics, visualizations (forest/funnel/RoB/PRISMA)
scripts/        Phase runners (run_phase*.py) + diagnostic tools
config/         models.yaml, prompts/{screener,arbiter,extractor}.yaml
data/           All input/output data
```

### Key Design Points

- **Prompt editing:** Edit `config/prompts/<role>.yaml` — not Python source
- **LLM caching:** Two independent caching layers (see below)
- **Checkpointing:** `data/.checkpoints/` — delete a file to re-run that phase
- **Token budgets:** Per-phase USD limits tracked in `data/.budget/`
- **Audit trail:** `data/.audit/prompt_log.jsonl` — one JSON line per real LLM call (PRISMA-trAIce). Each entry records: requested model, actual model returned by API (TRIPOD 6a), prompt version (TRIPOD 6c), API endpoint and method (TRIPOD 7c), token counts, estimated cost, and prompt SHA-256 hashes.
- **Deduplication:** Phase 2 deduplicates by DOI/PMID/title; Phase 2.5 runs a second dedup pass as a catch-net; `diagnose_duplicates.py` handles citation-string-based duplicates that surface after Phase 4 extraction

---

## Prompt Caching & Cost

The pipeline uses two independent caching layers:

### Layer 1 — DiskCache (local, free)
Keyed on `hash(model_id + system_prompt + user_prompt)`. Identical API calls never leave the machine. Stored in `data/<project>/.cache/`. Delete to force fresh LLM calls.

### Layer 2 — API-level Prompt Cache (per-provider)
The system prompt is the same for every abstract in a screening batch. Providers cache it server-side on the first call and charge 10% of input price on subsequent reads — a 90% discount on the repeated portion.

| Provider | Cache type | Min tokens | Discount |
|----------|-----------|------------|---------|
| Anthropic (Claude) | `cache_control: ephemeral` | 2,048 | 90% off read, +25% write |
| OpenAI (GPT-5 series) | Auto (no setup needed) | 1,024 | 90% off cached tokens |
| Google (Gemini) | Internal (no metrics exposed) | — | Estimated ~60–70% off |

**Cache is enabled automatically.** `base_agent.py` pads Anthropic system prompts to meet the 2,048-token minimum. No configuration required.

### Verifying cache status

```bash
# Check cache hit rates from audit log (runs after any phase)
python scripts/check_progress.py

# Run explicit per-model cache test (2 API calls per model)
python scripts/test_cache.py
python scripts/test_cache.py --roles arbiter writer   # test specific roles
```

`check_progress.py` shows a live cache hit rate section under "API Prompt Cache". A ✅ means >50% of recent calls had cache reads. ⬜ on a first run is normal (cache must be written before it can be read).

### Cost savings estimate

Based on PSY_MCI_rTMS run (~5,100 API calls, $26 total):

| Role | Calls | Cache type | Est. savings |
|------|-------|-----------|-------------|
| Screener 1 (Gemini FL) | 2,271 | internal | ~$0.46 (~49%) |
| Screener 2 (GPT-4.1 Mini) | 2,271 | OpenAI auto | ~$0.71 (~64%) |
| Arbiter (Claude) | 210 | Anthropic ephemeral | ~$1.50 (~74%) |
| Extractor (Gemini Pro) | 382 | internal | ~$0.62 (~9%) |
| **Total estimated** | 5,148 | | **~$3.36 (~29%)** |

> Extractor savings are low because user content (PDF text, ~8,000 tokens) dominates over the system prompt.
> Actual savings depend on study count and abstract length.

---

## PSY_MCI_rTMS Run Results (2026-03-15)

This section documents the full pipeline run for the PSY_MCI_rTMS project (rTMS / tDCS / NIBS for MCI and Alzheimer's disease). Use these numbers when writing the methods paper.

### Pipeline Cost & Token Summary

| Phase | Description | Model(s) | Cost (USD) | LLM Calls | Input Tokens | Output Tokens | Cache Read Tokens |
|-------|-------------|----------|------------|-----------|-------------|--------------|-------------------|
| 1 | Strategy generation | Claude Sonnet 4.6 | $0.07 | 2 | 1,552 | 7,036 | 0 |
| 2 | Literature search + dedup | (no LLM) | $0.00 | — | — | — | — |
| 3.1 | Title/abstract screening | Gemini 3.1 FL + GPT-4.1 Mini + Claude Sonnet 4.6 | $6.05 | 2,243 | 4,472,022 | 446,644 | 689,961 |
| 3.2 | Full-text screening | Gemini 3.1 FL + GPT-4.1 Mini | $3.06 | 589 | 1,421,192 | 540,474 | 291,633 |
| 4 | Data extraction + RoB | Gemini 3.1 Pro + GPT-5.4 | $14.11 | 782 | 3,327,948 | 1,383,730 | 102,147 |
| 5 | Statistical analysis | GPT-5.4 | $0.30 | 3 | 6,262 | 24,568 | 0 |
| 6 | Manuscript writing | Claude Sonnet 4.6 + GPT-5.4 | $1.10 | 33 | 108,140 | 58,417 | 12,645 |
| **Total** | | | **$24.68** | **3,652** | **9,337,116** | **2,460,869** | **1,096,386** |

### PRISMA Flow

```
Identification:
  Database records:       5,978 (PubMed 236, Europe PMC 1,864, Scopus 603, Cochrane 780, Embase 2,495)
  After deduplication:    4,045 (removed 1,933 = 32.3%)
  After pre-screen:       1,947

Screening:
  Title/abstract screened: 1,947
  Excluded (T/A):          1,650
  Full-text assessed:      297
  Excluded (full-text):    111

Included:
  Data extraction:         186
  Computable (quantitative synthesis): 48
```

### Screening Agreement Statistics

| Metric | Value |
|--------|-------|
| Total screened | 1,947 |
| Agreement rate (same side) | 84.2% |
| Exact agreement rate (same rating) | 22.8% |
| Cohen's κ | 0.495 (moderate) |
| Firm conflicts → Arbiter | 296 |
| Undecided → Human review queue | 12 |

### Meta-Analysis Results (Random-Effects DerSimonian–Laird)

| Outcome | Measure | k | SMD | 95% CI | p | I² | τ² | Sig. |
|---------|---------|---|-----|--------|---|-----|-----|------|
| Global cognition | ADAS-Cog | 13 | 0.204 | [−0.321, 0.728] | .447 | 85.7% | 0.668 | |
| Global cognition | MMSE | 23 | 0.618 | [0.271, 0.966] | .0005 | 80.0% | 0.531 | **\*** |
| Global cognition | MoCA | 8 | 0.624 | [0.117, 1.130] | .016 | 79.3% | 0.417 | **\*** |
| Cognitive domains | Memory | 14 | 0.650 | [0.335, 0.964] | <.001 | 49.6% | 0.173 | **\*** |
| Cognitive domains | Executive function | 6 | 0.060 | [−0.231, 0.351] | .686 | 0.0% | 0.000 | |
| Cognitive domains | Attention | 5 | 0.541 | [0.045, 1.037] | .033 | 48.9% | 0.152 | **\*** |
| Functional | ADL | 5 | −0.106 | [−0.638, 0.426] | .696 | 80.1% | 0.270 | |
| Functional | IADL | 3 | 0.434 | [−0.679, 1.547] | .445 | 83.3% | 0.798 | |
| Functional | ADCS-ADL | 3 | −0.504 | [−0.735, −0.274] | <.001 | 0.0% | 0.000 | **\*** |
| Neuropsychiatric | NPI | 3 | 0.369 | [−0.116, 0.854] | .136 | 31.4% | 0.058 | |
| Neuropsychiatric | GDS | 3 | 0.242 | [−0.197, 0.680] | .281 | 0.0% | 0.000 | |

### Figures Generated (25 total)

- 11 forest plots (`figures/forest_*.png`)
- 11 funnel plots (`figures/funnel_*.png`)
- PRISMA 2020 flow diagram (`figures/prisma_flow.png`)
- Risk of Bias traffic-light summary (`figures/rob_summary.png`)
- Risk of Bias domain bar chart (`figures/rob_domains.png`)

### Extraction Quality

| Category | Count |
|----------|-------|
| Studies extracted | 186 |
| Fully computable | 48 |
| Fixable (SE→SD) | 2 |
| Abstract-only (need full PDF) | 119 |
| P-values only | 3 |
| Incomplete | 8 |
| No outcomes | 3 |
| Failed extraction | 3 |
| RoB assessed | 186 |

### Manuscript Output

- Sections: title, abstract, introduction, methods, results, discussion
- Combined: `phase6_manuscript/drafts/manuscript_draft.md`
- `[CITATION NEEDED]` markers: 93 (fill manually)
- Citation suggestions: `phase6_manuscript/citation_suggestions.json`

---

## TRIPOD-LLM Compliance

This pipeline implements automated logging for [TRIPOD-LLM](https://tripod-llm.vercel.app/) reporting. The following items are handled by code:

| TRIPOD Item | What is logged | Where |
|-------------|---------------|-------|
| **5c** — Data dates | Search execution timestamp, model `pinned_at` cutoff dates | `search_log.json` |
| **6a** — Model version | Actual model returned by API (`response.model`), not just requested | `prompt_log.jsonl` |
| **6c** — Prompt consistency | Prompt version from `config/prompts/<role>.yaml`, temperature, seed | `prompt_log.jsonl` |
| **7c** — API method | `api_url`, `api_method` (OpenRouter v1/chat/completions, sync) | `prompt_log.jsonl` |
| **12** — Compute cost | Per-call token counts, cache metrics, estimated USD cost | `prompt_log.jsonl` + `.budget/` |

**Items requiring manual action** are tracked in [`投稿相關/TODO_TRIPOD_LLM.md`](投稿相關/TODO_TRIPOD_LLM.md):
- **9a/9b** — Prompt development process documentation
- **14d** — OSF preregistration (upload prompts, models.yaml, pico.yaml)
- **14e/14f** — GitHub public repo + Zenodo DOI, audit log upload

---

## Changelog

### v5.7 — Dynamic Study Design Filter, Pre-screen Improvements, Phase 3→4 Gate

- **Dynamic study design filter (Phase 1 → Phase 2.5):** Phase 1 Strategist now generates a `study_design_filter` with `filter_mode` (strict/loose), `positive_keywords`, and `design_exclusion_keywords` based on the PICO. Phase 2.5 reads this dynamically — no more hardcoded study design exclusions. Different meta-analyses can have different filter strategies.
- **Conference abstract detection relaxed:** Phase 2.5 structural exclusion no longer requires abstract < 300 chars for conference/supplement journals — conference abstracts are now excluded regardless of abstract length. This addresses 31 conference abstracts that previously leaked through to Phase 4 as minimal studies.
- **Extraction guidance (Phase 1 → Phase 4):** Phase 1 Strategist now generates `extraction_guidance.json` with outcome measure definitions (abbreviation, full name, scale range, scoring direction), preferred timepoint, crossover/multi-arm handling rules. Phase 4 Extractor injects this into extraction prompts, preventing scale confusion (e.g., MMSE 0-30 vs 3MS 0-100) and timepoint mismatches. Fully backward compatible — Phase 4 works without guidance but benefits from it.
- **Phase 3.2 `--finalize-pending` gate:** New command formally labels all pending (no-PDF) studies as "excluded (full-text not available)" before Phase 4. Ensures clean handoff and accurate PRISMA flow numbers. Prevents minimal/incomplete studies from wasting Phase 4 extraction tokens.
- **File structure cleanup:** Removed residual files (`y/`, `=2.0`), moved pipeline logs to `data/<project>/.logs/`, moved reference PDFs and TRIPOD checklist to `投稿相關/`.
- **Validation data reorganized:** `validation/` directory restructured into `screening/`, `extraction/source_pdfs/`, `reports/` with computed metrics JSONs.

### v5.6 — TRIPOD-LLM Audit Compliance

- **TRIPOD 5c:** `search_log.json` now includes `search_executed_at` (UTC ISO timestamp) and `model_pinned_at_dates` (all agent pinned_at dates from models.yaml)
- **TRIPOD 6c:** `prompt_log.jsonl` audit entries now include `prompt_version` field (read from `config/prompts/<role>.yaml`)
- **TRIPOD 7c:** `prompt_log.jsonl` audit entries now include `api_url` and `api_method` fields
- **`TODO_TRIPOD_LLM.md`:** Checklist of automated vs manual TRIPOD-LLM compliance items

### v5.4 — Prompt Caching, TRIPOD-LLM Compliance, Cost Optimisation

- **API-level prompt caching:** Anthropic `cache_control: ephemeral` on all system prompts (auto-padded to 2,048-token minimum); OpenAI auto-cache active for GPT-5 models; estimated **~29% cost reduction** on a typical full run
- **Gemini 3.1 Flash Lite for Screener 1** with `reasoning_effort: high` (thinking mode) replacing the previous screener model; lower cost + reasoning trace
- **TRIPOD-LLM compliance:** `actual_model` (item 6a), `temperature/seed/max_tokens` (6c), `inference_date` (7c), `estimated_cost_usd` (12), `prospero_id/run_id` (14d) — all logged per-call to `.audit/prompt_log.jsonl`
- **Langfuse observability:** Optional LLM call tracing via `LANGFUSE_*` env vars; non-fatal if keys absent. Requires `langfuse>=4.0` (installed via `pip install langfuse`). Uses `start_observation(as_type='generation')` API.
- **Cache-aware cost tracking:** `TokenBudget.record()` now accepts `cache_read_tokens` / `cache_write_tokens` and prices them correctly (90% read discount, 25% write surcharge for Anthropic)
- **Enriched prompt YAML files:** All six agent prompts now in `config/prompts/<role>.yaml`; prompts are general-purpose (not project-specific) and versioned
- **`scripts/export_run_report.py`:** Generates per-agent token usage table + Methods-section paragraph + CSV (Supplementary Table S1) — suitable for paper submission
- **`scripts/test_cache.py`:** Explicit cache health test for all model IDs (2 calls each); reports ✅/⬜/❓ per model
- **`check_progress.py`:** Now shows API prompt cache hit rates from audit log alongside token budgets

### v5.5 — Langfuse v4, Dedup-in-Prescreen, Prompt Cache

- **Langfuse v4 compatibility:** `langfuse_client.py` updated to use `start_observation(as_type='generation')` API (v4 is OpenTelemetry-based; old `lf.generation()` / `gen.end(output=...)` pattern removed)
- **Dedup integrated into Phase 2.5:** `run_phase2_5_prescreen.py` now runs `deduplicate_studies()` after keyword filtering as a catch-net for cross-database duplicates that survive Phase 2; cross-db duplicates reported separately in `prescreen_excluded_log.json`
- **API prompt caching enabled:** All Anthropic models auto-padded to ≥2,048 tokens in `base_agent.py`; confirmed cache hits (Claude ✅, GPT-4.1 Mini ✅); estimated ~29% cost saving on re-runs
- **General-purpose prompts:** `arbiter.yaml`, `strategist.yaml`, `writer.yaml` rewritten to be project-agnostic (no PSY/TMS-specific content); dynamic PICO criteria injected per-call via user message only
- **`check_progress.py` cache reporting:** Now reads `.audit/prompt_log.jsonl` and shows per-role API cache hit rates (⬜/🟡/✅)

### v5.3 — Multi-Project Support

- **Multi-project isolation:** Each research question now lives in `data/<project_name>/` with its own input, phases, cache, checkpoints, and budget. Run independent meta-analyses without interference.
- **Interactive project selector:** Every script prompts for project selection at startup (or auto-selects the last active project). Create new projects on the fly.
- **Project state persisted:** `data/.active_project` remembers the last selected project across sessions.
- **Zero data loss:** Existing data automatically migrated to `data/PSY_MCI_rTMS/`.

### v5.2 — Code Cleanup & Stability

- **Cochrane & Embase manual search instructions:** Phase 1 now outputs `manual_search_cochrane.txt` and `manual_search_embase.txt` with numbered search strategies ready to paste into Cochrane Library Advanced Search and Ovid; `export_manual_queries.py` regenerates them from existing Phase 1 data
- **Phase 2.5 improvements:** Additional structural exclusions — titles ≤ 10 chars, no-abstract-no-identifier records, title-level letter/commentary/erratum patterns (`Re:`, `Letter to`, `Erratum`, `Corrigendum`, etc.)
- **PRISMA 2020 diagram:** New `prisma_flow_diagram()` in `visualizations.py` + standalone `export_prisma_diagram.py` script; `--included` flag to set correct count after dedup
- **Validation UX:** `--status` flag added to both `validate_screening.py` and `validate_extraction.py` for annotation progress tracking; extended abstract display (400 → 600 chars); clearer instructions
- **Post-extraction duplicate detection:** `diagnose_duplicates.py` now part of the standard workflow (after Phase 4, before Phase 5); documented in README and CLAUDE.md
- **README reorganized:** Added Methodology Validation section, PRISMA diagram section, Duplicate Detection as a pipeline step; cleaner structure throughout

### v5.1 — PDF Pipeline & Pre-screening Improvements

- **PDF downloader extended to 10 sources:** Sci-Hub (domain rotation + HTML parsing) and LibGen scimag added as strategies 8 and 9
- **`scripts/doi_hunter.py`:** Standalone batch DOI→PDF tool (4-source waterfall); reads `dois.txt`/`dois.csv`; Ctrl+C safe
- **Phase 2.5 structural filters:** Registry/conference detection layer — DOI prefixes, registry ID as author, publication types, conference supplement detection
- **Phase 3 Stage 2 `--integrate` step:** Runs Zotero PDF matching and exports `missing_pdfs.txt`; safe to re-run
- **`rename_zotero_pdfs.py` fixes:** Minimum title-length guard, `_BAD_PDF_PATTERNS` list, `SKIP_FILES` dict
- **New dependency:** `beautifulsoup4>=4.12`, `fake-useragent>=1.5`

### v5 — Initial Release

- LLM-driven broad/specific term classification → title-only vs. title+abstract search
- Europe PMC syntax fix: `(TITLE:"x" OR ABSTRACT:"x")`
- Scopus fix: `DOCTYPE(ar)` + RCT keyword filter
- Database toggles via `.env`
- Phase 2.5 zero-cost pre-screening
- Zotero PDF matcher with PMID/DOI/title matching and `--diagnose` mode
- Phase 5 rewrite: all measures × all outcomes, safety summary
- Phase 6 rewrite: rich data-driven prompts, citation cross-validation, `[CITATION NEEDED]` collection
- Diagnostic tools: `diagnose_pipeline.py`, `diagnose_phase4.py`
