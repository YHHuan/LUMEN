# CLAUDE.md

## Project Overview

**LUMEN v2** (LLM-based Unified Meta-analysis Extraction Network) — a Python CLI pipeline that automates systematic reviews and meta-analyses using a multi-agent LLM chain via **OpenRouter** (OpenAI-compatible API).

**Multi-project support:** Each research question lives under `data/<project_name>/`. Project selection via interactive selector (`src/utils/project.py`).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
```

Required: `OPENROUTER_API_KEY`. Optional: `NCBI_API_KEY`, `ELSEVIER_API_KEY`, per-phase budgets.

R (optional): Install R + `metafor` package for robust meta-analysis. Falls back to built-in Python REML engine if R is unavailable.

## Pipeline Execution Order

### Core Pipeline

```bash
# Phase 1: Strategy — generate search strategy + screening criteria from PICO
python scripts/run_phase1.py

# Phase 2: Search — literature search across PubMed, Scopus, OpenAlex, EPMC
python scripts/run_phase2.py
python scripts/run_phase2.py --deduplicate              # Re-dedup after manual imports

# Phase 3.0: Pre-screening — context-aware rescue of quarantined studies
python scripts/run_phase2_5_prescreen.py

# Phase 3.1: Dual T/A screening — two models + arbiter, 5-point confidence scale
python scripts/run_phase3_stage1.py

# Phase 3.2: PDF acquisition — download + finalize
python scripts/run_phase3_stage2.py --download
python scripts/run_phase3_stage2.py --finalize-pending

# Phase 3.3: Full-text screening — Sonnet verifies PICO against full PDF
python scripts/run_phase3_3_fulltext_screen.py

# Phase 4: Extraction — 3-pass Gemini + GPT tiebreaker, evidence spans
python scripts/run_phase4.py
python scripts/run_phase4.py --incremental               # Append new, skip existing

# Phase 4.5: Analysis planning — data profile + LLM plan proposal + human review
python scripts/run_phase4_5.py
python scripts/run_phase4_5.py --auto-approve             # Skip human review

# Phase 5: Statistics — REML + HKSJ per analysis plan
python scripts/run_phase5.py --planned

# Phase 6: Manuscript — citation-grounded drafting
python scripts/run_phase6.py

# Quality: RoB-2/ROBINS-I + GRADE (auto-routes by study design)
python scripts/run_quality_assessment.py
```

### Screening Benchmark (Phase 3.1)

5-arm ROC comparison: single Gemini / GPT / Claude + dual + ASReview.

```bash
python scripts/run_phase3_stage1.py --single              # Single-agent screening
python scripts/run_phase3_stage1.py --single --model gpt  # GPT-4.1 Mini
python scripts/run_phase3_stage1.py --single --model claude # Claude Sonnet 4.6
python scripts/run_screening_benchmark.py                  # ROC benchmark
python scripts/run_screening_benchmark.py --asreview X.csv # Add ASReview arm
```

### Extraction Ablation (Phase 4-5)

3-arm comparison: full pipeline vs single Sonnet vs single Gemini.

```bash
python scripts/run_extraction_ablation.py --arm C          # Sonnet single-pass
python scripts/run_extraction_ablation.py --arm D          # Gemini single-pass
python scripts/run_extraction_ablation.py --compare        # Comparison table
```

### Report Collection

```bash
python scripts/collect_domain_report.py                    # Current domain → reports/
python scripts/collect_domain_report.py --all              # All domains
```

### Diagnostics & Utilities

```bash
python scripts/check_progress.py                           # Pipeline progress check
python scripts/diagnose_phase4.py                          # Extraction diagnostics
python scripts/run_cost_report.py                          # Cost dashboard
python scripts/run_readiness_check.py                      # Publication readiness
python scripts/run_transparency_report.py                  # Full transparency report
python scripts/generate_review.py                          # HTML review cards
python scripts/export_prisma_diagram.py                    # PRISMA flow diagram
python scripts/run_pdf_to_markdown.py                      # Gemini PDF→Markdown
python scripts/run_nma.py                                  # Network meta-analysis (R netmeta)
```

## Architecture

### Agent Chain

| Agent | Model | Phase | Role |
|-------|-------|-------|------|
| Strategist | Claude Sonnet 4.6 | 1 | PICO → search strategy + screening criteria |
| Rescue Screener | Gemini Flash Lite | 3.0 | LLM-lite rescue of quarantined studies |
| Screener 1 | Gemini 3.1 Pro | 3.1 | 5-point T/A screening (static + PICO-dynamic prompt) |
| Screener 2 | GPT-4.1 Mini | 3.1 | 5-point T/A screening (same prompt, model diversity) |
| Arbiter | Claude Sonnet 4.6 | 3.1 | Firm conflict resolution |
| Full-text Screener | Claude Sonnet 4.6 | 3.3 | PDF-level PICO verification |
| Extractor | Gemini 3.1 Pro | 4 | 3-pass claim-grounded extraction |
| Tiebreaker | GPT-5.4 | 4 | Disagreement resolution across passes |
| Statistician | GPT-5.4 | 5 | Interpretation + code gen |
| Writer | Claude Sonnet 4.6 | 6 | [REF:keyword] citation markers |
| Citation Guardian | GPT-5.4 | 6 | Verify markers against reference pool |

### Screening Prompt Architecture

The Phase 3.1 screener prompt has two layers:
- **Static rules** (in `screener_1.yaml` / `screener_2.yaml`): Always-apply exclusions for publication types (editorials, protocols) and study types (cost-effectiveness, modelling, guidelines, surveillance)
- **Dynamic rules** (injected at runtime): PICO-specific exclusion criteria from `screening_criteria.json` (generated by Phase 1)

This ensures screening quality is consistent across domains while adapting to each research question.

### Data Flow

```
Phase 1: screening_criteria.json (PICO → search strategy)
Phase 2: deduplicated/all_studies.json
Phase 3.0: prescreened/filtered_studies.json
Phase 3.1: stage1_title_abstract/included_studies.json
Phase 3.2: stage2_fulltext/included_with_pdf.json
Phase 3.3: stage2_fulltext/included_fulltext.json
Phase 4: phase4_extraction/extracted_data.json
Phase 4.5: phase4_5_planning/analysis_plan.yaml
Phase 5: phase5_analysis/planned_results.json
Phase 6: phase6_manuscript/manuscript.md
```

Phase 4 auto-selects the best available included studies via `DataManager.load_best_included()` (priority: 3.3 > 3.2 > 3.1).

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/utils/prescreen.py` | Context-aware bigram matching, quarantine, rescue pipeline |
| `src/utils/pdf_decomposer.py` | gmft table detection + pdfplumber fallback |
| `src/utils/vector_index.py` | sentence-transformers + hnswlib retrieval |
| `src/utils/extraction_context.py` | Smart context builder for extraction |
| `src/utils/analysis_planner.py` | Phase 4.5: data profiling → LLM plan → human review |
| `src/utils/statistics.py` | Pure Python REML (scipy L-BFGS-B), HKSJ, meta-regression, Egger, trim-and-fill |
| `src/utils/effect_sizes.py` | Effect size computation (OR, RR, SMD, VE) |
| `src/utils/visualizations.py` | Forest, funnel, LOO, cumulative, subgroup plots |
| `src/utils/nma.py` | NMA orchestrator: R netmeta subprocess, P-score, consistency |
| `src/agents/writer.py` | ReferencePool + CitationGuardian grounding |
| `src/utils/citation_verifier.py` | BM25 + assertion extraction for citation verification |
| `src/utils/rob2.py` | Cochrane RoB-2 risk of bias (5 domains, RCTs) |
| `src/utils/robins_i.py` | ROBINS-I risk of bias (7 domains, non-RCTs) |
| `src/utils/grade.py` | GRADE evidence certainty (5 downgrade + 3 upgrade) |
| `src/utils/extraction_validator.py` | Per-field accuracy, TOST equivalence, Bland-Altman, error taxonomy |
| `src/utils/concordance_checker.py` | RoB agreement (Cohen's κ), synthesis concordance (ES, CI overlap) |
| `src/utils/screening_benchmark.py` | ROC curve, WSS@95%, calibration, 5-arm benchmark |
| `src/utils/agreement.py` | Cohen's/weighted/Fleiss' kappa, PABAK |
| `src/utils/cost_tracker.py` | Cost aggregation + cost curve visualization |
| `src/utils/reproducibility.py` | SHA-256 config fingerprint |
| `src/utils/prisma_s.py` | PRISMA-S 2021 search reporting checklist |
| `src/utils/human_review.py` | Human override overlay + agreement tracking |

### Configuration

| File | Purpose |
|------|---------|
| `.env` | API keys, per-phase USD budgets |
| `config/models.yaml` | Model IDs, pricing, temperatures per agent |
| `config/v2_settings.yaml` | Algorithmic settings (rescue, vector retrieval, stats, NMA) |
| `config/prompts/<role>.yaml` | Externalized prompts per agent |
| `data/<project>/input/pico.yaml` | Research question definition |

### Report Structure

`collect_domain_report.py` generates per-domain output:

```
reports/<domain>/
├── 00_manifest.md              ← Index with source paths
├── 01_pipeline_costs/          ← Per-phase token usage + USD
├── 02_screening/               ← Results, kappa, distributions
├── 03_extraction/              ← Extracted data, evidence spans
├── 04_analysis_plan/           ← Phase 4.5 profile + plan
├── 05_statistics/              ← Pooled estimates, R output
├── 06_figures/                 ← Forest plots, funnel plots
├── 07_manuscript/              ← Drafts + citation verification
├── 08_quality/                 ← RoB-2/ROBINS-I, GRADE
├── 09_ground_truth_comparison/ ← Extraction validation, synthesis concordance, RoB agreement
├── 10_human_validation/        ← Items requiring human review
├── 11_benchmark/               ← Ablation results
└── 12_supplement/              ← PRISMA, search strategy, PICO
```

### Statistics Notes

- REML: `scipy.optimize.minimize` with L-BFGS-B, log-space parameterization, DL starting value
- Falls back to Nelder-Mead if L-BFGS-B fails
- Meta-regression: WLS with Knapp-Hartung adjusted variance
- R `metafor` preferred when available; pure Python fallback
