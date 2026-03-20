# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meta-Analysis Generator v5 — a Python CLI pipeline that automates systematic reviews and meta-analyses using a chain of LLM agents. The primary LLM provider is **OpenRouter** (OpenAI-compatible client).

**Multi-project support**: Each research question lives in its own directory under `data/<project_name>/`. Every script prompts for project selection at startup. The active project is persisted in `data/.active_project`.

Current project: `PSY_MCI_rTMS` (Psychotherapy/MCI/rTMS meta-analysis).

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
cp .env.example .env              # then fill in API keys
```

Required: `OPENROUTER_API_KEY`. Optional: `NCBI_API_KEY`, `ELSEVIER_API_KEY`, and per-phase token budgets (see `.env.example`).

## Running the Pipeline

Every script starts with a **project selector** — choose an existing project or create a new one. The PICO research question is defined in `data/<project>/input/pico.yaml`.

Each phase is a standalone script. Phases auto-resume from `.checkpoints/` if interrupted.

```bash
python scripts/run_phase1.py                                   # Strategy generation
python scripts/run_phase2.py                                   # Literature search + dedup
python scripts/run_phase2.py --show-queries                    # Preview queries only
python scripts/run_phase2.py --deduplicate                     # Re-dedup after manual imports
python scripts/run_phase2_5_prescreen.py                       # Free keyword pre-screen
python scripts/run_phase2_6_ct_lookup.py                       # CT/RIS → PubMed lookup, patches phase 3.1 list
python scripts/run_phase2_6_ct_lookup.py --dry-run             # Preview matches only (no file changes)
python scripts/run_phase2_6_ct_lookup.py --dedup               # Run on full dedup pool instead
python scripts/run_phase3_stage1.py                            # Title/abstract screening
python scripts/run_phase3_stage2.py --download                 # PDF download only
python scripts/run_phase3_stage2.py --review                   # Full-text screening only
python scripts/run_phase3_stage2.py --finalize-pending         # Gate: exclude no-PDF studies before Phase 4
python scripts/run_phase3_stage2.py --all                      # Download + integrate + review
python scripts/run_phase4.py                                   # Data extraction
python scripts/run_phase4.py --validate-only                   # Validate without re-extracting
python scripts/run_phase5.py                                   # Statistical analysis
python scripts/run_phase5.py --builtin-only                    # Skip LLM interpretation
python scripts/run_phase6.py                                   # Full manuscript writing
python scripts/run_phase6.py --section discussion              # Single section
python scripts/run_phase6.py --sections introduction,methods   # Multiple sections
python scripts/run_phase6.py --skip-validation                 # Skip citation checks
python scripts/run_phase6.py --validate-only                   # Only citation validation
```

**To re-run a phase from scratch**, delete its checkpoint:
```bash
rm data/<project>/.checkpoints/phase4_extraction.json   # Linux/Mac
del data\<project>\.checkpoints\phase4_extraction.json  # Windows CMD
```

## Diagnostic Tools

```bash
python scripts/check_progress.py              # Overall pipeline progress
python scripts/diagnose_pipeline.py           # Quality check Phase 2→3→4
python scripts/diagnose_phase4.py             # Phase 4 data quality report
python scripts/diagnose_phase4.py --fix       # Auto-fix SE→SD conversions
python scripts/diagnose_duplicates.py         # Detect cross-database duplicate papers in Phase 4
python scripts/diagnose_duplicates.py --fix   # Remove duplicates (then re-run Phase 5 & 6)
python scripts/rename_zotero_pdfs.py --diagnose  # Missing PDF report
python scripts/rename_zotero_pdfs.py --dry-run   # Preview Zotero PDF matching
python scripts/rename_zotero_pdfs.py             # Copy & rename Zotero PDFs
python scripts/extract_DOI.py                    # Export DOIs for Zotero batch import
python scripts/export_manual_queries.py          # Re-generate Cochrane/Embase search instructions
python scripts/export_prisma_diagram.py          # Generate PRISMA 2020 flow diagram PNG
python scripts/export_prisma_diagram.py --included 173  # Override included count after dedup

# Validation (methodology validation for publication)
python scripts/validate_screening.py --export --n 100   # Export 100 abstracts for human annotation
python scripts/validate_screening.py --status           # Check annotation progress
python scripts/validate_screening.py --compute          # Compute sensitivity/specificity/κ/PABAK
python scripts/validate_extraction.py --export          # Export extracted fields (complete+partial studies)
python scripts/validate_extraction.py --export --complete-only  # Export complete studies only
python scripts/validate_extraction.py --status          # Check annotation progress
python scripts/validate_extraction.py --compute         # Compute precision/recall/F1 by field type

# Abstract enrichment (run after Phase 2 if blank abstracts found)
python scripts/enrich_abstracts.py                       # Dry-run: show blank abstract counts by source
python scripts/enrich_abstracts.py --fix                 # Fetch missing abstracts via PubMed + CrossRef

# ScienceDirect / Elsevier access check
python scripts/check_sciencedirect.py                    # Test if your IP can access ScienceDirect
```

## Architecture

### Multi-Project Data Layout

```
data/
  .active_project              ← last selected project name
  PSY_MCI_rTMS/                ← project 1 (current)
    input/pico.yaml
    phase1_strategy/
    phase2_search/
    phase3_screening/
    phase4_extraction/
    phase5_analysis/
    phase6_manuscript/
    .cache/ .checkpoints/ .budget/ .audit/
  NEW_PROJECT/                 ← project 2 (independent)
    input/pico.yaml
    ...
```

Project selection is handled by `src/utils/project.py`. All paths resolve through `get_data_dir()`.

### Agent Chain (LLM Roles)

Each agent maps to a model defined in `config/models.yaml`. All agents extend `src/agents/base_agent.py`, which provides: OpenRouter API calls, DiskCache-based response caching (prevents duplicate LLM calls for identical prompts), token budget enforcement, retry logic, and prompt audit logging.

Prompts for all agents live in **`config/prompts/<role>.yaml`** — edit there, not in source code.

| Agent | Model | Phase |
|-------|-------|-------|
| Strategist | Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-6`) | 1 — PICO → MeSH + queries |
| Screener 1 | Gemini 3.1 Pro (`google/gemini-3.1-pro-preview`) | 3.1 — 5-point confidence scale |
| Screener 2 | GPT-4.1 Mini (`openai/gpt-4.1-mini`) | 3.1 — 5-point confidence scale |
| Arbiter | Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-6`) | 3.1 (firm conflicts only) |
| Extractor | Gemini 3.1 Pro (`google/gemini-3.1-pro-preview`) | 4 — PDF → structured data |
| Extractor Tiebreaker | GPT-5.4 (`openai/gpt-5.4`) | 4 (self-consistency 3rd-model check) |
| Statistician | GPT-5.4 (`openai/gpt-5.4`) | 5 — supplementary analysis code |
| Writer | Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-6`) | 6 — manuscript sections |
| Citation Guardian | GPT-5.4 (`openai/gpt-5.4`) | 6 — citation validation |

#### Phase 3 Screening — 5-point confidence scale

Screeners return one of: `most_likely_include`, `likely_include`, `undecided`, `likely_exclude`, `most_likely_exclude`.

- Include-side: `most_likely_include`, `likely_include`, `undecided`
- Exclude-side: `likely_exclude`, `most_likely_exclude`
- Both agree on same side → final decision
- Disagree + ≥1 undecided → `human_review_queue.json` (temporarily included)
- Disagree + both firm → Arbiter
- API/parse failure → defaults to `likely_include` (preserves recall)

### Data Flow (per project)

```
data/<project>/input/pico.yaml
  → phase1_strategy/   (search_strategy.json, screening_criteria.json)
  → phase2_search/raw/ (per-database JSON + manual .ris/.csv imports)
  → phase2_search/deduplicated/all_studies.json
  → phase2_search/prescreened/filtered_studies.json
  → phase3_screening/stage1_title_abstract/
      included_studies.json
      excluded_studies.json
      human_review_queue.json   ← undecided conflicts; review manually
      screening_results.json    ← full dual-screening data incl. κ stats
  → phase3_screening/stage2_fulltext/ (PDFs + fulltext_review.json)
  → phase4_extraction/ (extracted_data.json, risk_of_bias.json)
  → phase5_analysis/   (statistical_results.json, figures/)
  → phase6_manuscript/drafts/ (per-section .md files)
  → .audit/prompt_log.jsonl  ← PRISMA-trAIce prompt audit trail
```

### Key Source Modules

- **`src/utils/project.py`** — multi-project selector; `select_project()` + `get_data_dir()`
- **`src/agents/base_agent.py`** — base class; all LLM I/O goes through here; provides `load_prompt_config()` and `_log_prompt_audit()`
- **`src/agents/screener.py`** — `ScreenerAgent`, `ArbiterAgent`, `run_dual_screening`; 5-point scale + Cohen's κ
- **`src/utils/cache.py`** — DiskCache wrapper + `TokenBudget` class (tracks spend per phase)
- **`src/utils/file_handlers.py`** — `DataManager`: reads/writes phase JSON artifacts
- **`src/utils/deduplication.py`** — DOI/PMID/fuzzy-title dedup; `.ris` file parsing
- **`src/utils/pdf_downloader.py`** — 10-source cascade: Unpaywall → PMC → EPMC → Semantic Scholar → OpenAlex → CrossRef → Sci-Hub → LibGen → DOI redirect
- **`src/utils/statistics.py`** — random-effects meta-analysis math
- **`src/utils/effect_sizes.py`** — standardized effect size calculations (including SE→SD conversion)
- **`src/apis/`** — one module per database (pubmed, europepmc, scopus, crossref, unpaywall)

### Configuration

- **`.env`** — API keys and database toggles (`ENABLE_PUBMED`, `ENABLE_SCOPUS`, etc.) and per-phase USD token budgets (`TOKEN_BUDGET_PHASE1`, `TOKEN_BUDGET_PHASE3_TA`, etc.)
- **`config/models.yaml`** — model IDs, `pinned_at` dates, temperatures, max_tokens, pricing per agent role; also `batch_settings`
- **`config/prompts/<role>.yaml`** — externalized prompts; edit here instead of in agent source code
- **`data/<project>/input/pico.yaml`** — the research question; edit this before running Phase 1
- **`data/<project>/.audit/prompt_log.jsonl`** — auto-generated audit log; one line per real LLM call (PRISMA-trAIce / TRIPOD-LLM). Fields: `timestamp`, `role`, `model_id`, `actual_model` (6a), `prompt_version` (6c), `api_url`/`api_method` (7c), `pinned_at`, `temperature`, `seed`, token counts, `estimated_cost_usd` (12), prompt SHA-256 hashes

### Manual Database Imports

Cochrane, Embase, and Web of Science require manual export. Place `.ris` files in `data/<project>/phase2_search/raw/` with descriptive names (the filename becomes the PRISMA source label), then run `python scripts/run_phase2.py --deduplicate`.

### Checkpointing & Caching

- **Checkpoints** (`data/<project>/.checkpoints/phase_X.json`): track which studies have been processed; delete to re-run a phase
- **LLM cache** (`data/<project>/.cache/`): DiskCache keyed on prompt hash; do not delete unless you want to force fresh LLM calls and incur new costs
