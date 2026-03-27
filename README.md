# LUMEN v2

**LLM-based Unified Meta-analysis Extraction Network**

A fully automated systematic review and meta-analysis pipeline powered by multi-agent LLM orchestration. LUMEN v2 covers the complete PRISMA 2020 workflow — from PICO-driven search strategy generation through dual-model screening, claim-grounded data extraction, statistical synthesis, and citation-verified manuscript drafting.

---

## Highlights

- **End-to-end automation** — 10 pipeline phases from search strategy to manuscript, with defined human intervention breakpoints
- **Multi-agent architecture** — 11 specialized LLM agents across 4 model families (Claude, Gemini, GPT) via OpenRouter
- **Dual screening with arbiter** — Two independent screeners + conflict resolution, 5-point confidence scale
- **3-pass extraction** — Claim-grounded extraction with evidence spans and cross-pass tiebreaking
- **Publication-ready statistics** — REML + Knapp-Hartung, with optional R `metafor` backend
- **Quality assessment** — RoB-2 (RCTs) + ROBINS-I (non-RCTs) auto-routed by study design, GRADE certainty
- **Full cost transparency** — Per-phase USD tracking, token budgets, cost curve visualization
- **PRISMA 2020 compliant** — PRISMA-S search reporting, flow diagrams, reproducibility fingerprints

## Pipeline Overview

```
Phase 1       PICO → Search Strategy + Screening Criteria
  ★ BP0       Human review: verify PICO and search logic
Phase 2       Multi-database Literature Search (PubMed, Scopus, OpenAlex, EPMC)
Phase 3.0     Context-aware Pre-screening (quarantine rescue)
Phase 3.1     Dual Title/Abstract Screening (Gemini + GPT + Arbiter)
  ★ BP1       Human review: undecided / parse-failure queue
Phase 3.2     PDF Acquisition
  ★ BP2       Human review: supplement failed downloads
Phase 3.3     Full-text Screening (PDF-level PICO verification)
  ★ BP3       Human review: verify full-text exclusions
Phase 4       3-Pass Data Extraction (Gemini + GPT tiebreaker)
  ★ BP4       Human review: extraction vs ground truth
Phase 4.5     Analysis Planning (data profile → LLM proposal → approval)
  ★ BP5       Human review: approve/modify analysis plan
Phase 5       Statistical Synthesis (REML + HKSJ)
Phase 6       Manuscript Drafting (citation-grounded)
  ★ BP6       Human review: results vs published ground truth
```

> **★ BP** = Human intervention breakpoint. The pipeline pauses at each BP for human review before proceeding.

## Quick Start

### Prerequisites

- Python 3.10+
- [OpenRouter](https://openrouter.ai/) API key
- R + `metafor` package (optional, falls back to pure Python REML)

### Installation

```bash
git clone https://github.com/YHHuan/LUMEN-LLM-based-Unified-Meta-analysis-Extraction-Network.git
cd LUMEN-LLM-based-Unified-Meta-analysis-Extraction-Network

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # Fill in API keys
```

### Configuration

| File | Purpose |
|------|---------|
| `.env` | API keys + per-phase USD budgets |
| `config/models.yaml` | Model IDs, pricing, temperatures per agent |
| `config/v2_settings.yaml` | Algorithmic settings (rescue, vector retrieval, stats) |
| `config/prompts/*.yaml` | Externalized prompts per agent role |
| `data/<project>/input/pico.yaml` | Research question definition (PICO) |

### Define Your Research Question

Create `data/<project_name>/input/pico.yaml`:

```yaml
population: "Adults aged 65+ with major depressive disorder"
intervention: "SSRI antidepressants"
comparator: "Placebo"
outcome: "Depression symptom reduction (HAM-D or equivalent)"
study_design: "Randomized controlled trials"
```

### Run the Pipeline

```bash
# Core pipeline (run sequentially)
python scripts/run_phase1.py                              # Strategy
python scripts/run_phase2.py                              # Search
python scripts/run_phase2_5_prescreen.py                  # Pre-screen
python scripts/run_phase3_stage1.py                       # Dual T/A screening
python scripts/run_phase3_stage2.py --download             # PDF download
python scripts/run_phase3_stage2.py --finalize-pending     # Finalize PDFs
python scripts/run_phase3_3_fulltext_screen.py             # Full-text screening
python scripts/run_phase4.py                               # Data extraction
python scripts/run_phase4_5.py                             # Analysis planning
python scripts/run_phase5.py --planned                     # Statistics
python scripts/run_phase6.py                               # Manuscript
python scripts/run_quality_assessment.py                   # RoB-2/ROBINS-I + GRADE
```

## Agent Architecture

LUMEN v2 orchestrates 11 specialized agents across 4 model families to maximize accuracy through model diversity:

| Agent | Model | Phase | Role |
|-------|-------|-------|------|
| Strategist | Claude Sonnet | 1 | PICO → search strategy + screening criteria |
| Rescue Screener | Gemini Flash Lite | 3.0 | LLM-lite rescue of quarantined studies |
| Screener 1 | Gemini Pro | 3.1 | 5-point T/A screening |
| Screener 2 | GPT-4.1 Mini | 3.1 | 5-point T/A screening (model diversity) |
| Arbiter | Claude Sonnet | 3.1 | Firm conflict resolution |
| Full-text Screener | Claude Sonnet | 3.3 | PDF-level PICO verification |
| Extractor | Gemini Pro | 4 | 3-pass claim-grounded extraction |
| Tiebreaker | GPT | 4 | Cross-pass disagreement resolution |
| Statistician | GPT | 5 | Interpretation + code generation |
| Writer | Claude Sonnet | 6 | `[REF:keyword]` citation markers |
| Citation Guardian | GPT | 6 | Verify markers against reference pool |

### Screening Prompt Architecture

Phase 3.1 uses a two-layer prompt design:
- **Static rules** — Always-apply exclusions for publication types (editorials, protocols) and study types (cost-effectiveness, modelling, guidelines)
- **Dynamic rules** — PICO-specific exclusion criteria from `screening_criteria.json`, injected at runtime

This ensures consistent screening quality across domains while adapting to each research question.

## Data Flow

```
data/<project>/
├── input/
│   └── pico.yaml                                  ← Research question
├── phase1_strategy/
│   ├── search_strategy.json                       ← Boolean search syntax
│   └── screening_criteria.json                    ← Inclusion/exclusion rules
├── phase2_search/
│   └── deduplicated/all_studies.json              ← Deduplicated records
├── phase3_screening/
│   ├── prescreened/filtered_studies.json           ← After pre-screening
│   ├── stage1_title_abstract/
│   │   ├── included_studies.json                  ← T/A screening results
│   │   └── human_review_queue.json                ← Undecided studies
│   └── stage2_fulltext/
│       ├── pdfs/                                  ← Downloaded PDFs
│       ├── included_with_pdf.json                 ← Studies with PDFs
│       └── included_fulltext.json                 ← Full-text screened
├── phase4_extraction/
│   └── extracted_data.json                        ← Structured data + evidence spans
├── phase4_5_planning/
│   ├── data_profile.json                          ← Data profile summary
│   └── analysis_plan.yaml                         ← Approved analysis plan
├── phase5_analysis/
│   └── planned_results.json                       ← Pooled estimates
├── phase6_manuscript/
│   └── manuscript.md                              ← Citation-grounded draft
└── quality_assessment/
    ├── rob2_assessments.json                      ← RoB-2 (RCTs)
    ├── robins_i_assessments.json                  ← ROBINS-I (non-RCTs)
    └── grade_evidence_profile.json                ← GRADE certainty
```

Phase 4 auto-selects the best available included studies via `DataManager.load_best_included()` (priority: Phase 3.3 > 3.2 > 3.1).

## Quality Assessment

LUMEN v2 auto-routes each study to the appropriate risk of bias tool based on study design classification:

| Study Design | Tool | Domains | Scale |
|-------------|------|---------|-------|
| RCT | **RoB-2** (Cochrane) | 5 domains | Low / Some concerns / High risk |
| Non-RCT | **ROBINS-I** | 7 domains | Low / Moderate / Serious / Critical |

Study design is classified using keyword pattern matching across `study_design`, `title`, and `population_description` fields — not exact string comparison.

**GRADE** evidence certainty assessment (5 downgrade + 3 upgrade domains) runs on all outcomes regardless of study design.

## Benchmarking

### Screening Benchmark (Phase 3.1)

5-arm ROC comparison: single Gemini / GPT / Claude + dual-agent + ASReview.

```bash
python scripts/run_phase3_stage1.py --single                # Single Gemini
python scripts/run_phase3_stage1.py --single --model gpt    # Single GPT
python scripts/run_phase3_stage1.py --single --model claude # Single Claude
python scripts/run_screening_benchmark.py                    # ROC curves
python scripts/run_screening_benchmark.py --asreview X.csv  # Add ASReview arm
```

### Extraction Ablation (Phase 4-5)

3-arm comparison: full pipeline vs single Sonnet vs single Gemini.

```bash
python scripts/run_extraction_ablation.py --arm C           # Sonnet single-pass
python scripts/run_extraction_ablation.py --arm D           # Gemini single-pass
python scripts/run_extraction_ablation.py --compare         # Comparison table
```

## Statistics

- **Primary**: REML via `scipy.optimize.minimize` (L-BFGS-B with DL starting value, Nelder-Mead fallback)
- **Confidence intervals**: Knapp-Hartung-Sidik-Jonkman (HKSJ) adjustment
- **Heterogeneity**: Cochran's Q, I², prediction intervals
- **Meta-regression**: Weighted least squares with Knapp-Hartung variance
- **Publication bias**: Egger's test, trim-and-fill
- **Sensitivity**: Leave-one-out, cumulative meta-analysis
- **Effect sizes**: OR, RR, SMD (Hedges' g), MD, VE
- **R backend** (optional): `metafor` package via subprocess for robust REML

### Network Meta-Analysis

NMA is available via R `netmeta` integration:

```bash
python scripts/run_nma.py                                   # Requires R + netmeta
```

Includes network connectivity validation, P-score ranking, and consistency testing.

## Report Collection

```bash
python scripts/collect_domain_report.py                     # Current domain
python scripts/collect_domain_report.py --all               # All domains
python scripts/collect_domain_report.py --paper             # Cross-domain paper figures + tables
```

Generates structured output under `reports/<domain>/`:

```
00_manifest.md              Index with source paths
01_pipeline_costs/          Per-phase token usage + USD
02_screening/               Results, kappa, distributions
03_extraction/              Extracted data, evidence spans
04_analysis_plan/           Phase 4.5 profile + plan
05_statistics/              Pooled estimates, R output
06_figures/                 Forest plots, funnel plots
07_manuscript/              Drafts + citation verification
08_quality/                 RoB-2, ROBINS-I, GRADE
09_ground_truth_comparison/ LUMEN vs published (user fills)
10_human_validation/        Items requiring human review
11_benchmark/               Ablation results
12_supplement/              PRISMA, search strategy, PICO
```

## Diagnostics & Utilities

```bash
python scripts/check_progress.py              # Pipeline progress dashboard
python scripts/diagnose_phase4.py             # Extraction diagnostics
python scripts/run_cost_report.py             # Cost dashboard + curves
python scripts/run_readiness_check.py         # Publication readiness check
python scripts/run_transparency_report.py     # Full transparency report
python scripts/generate_review.py             # HTML review cards
python scripts/export_prisma_diagram.py       # PRISMA flow diagram
python scripts/run_pdf_to_markdown.py         # Gemini PDF → Markdown
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/utils/prescreen.py` | Context-aware bigram matching, quarantine, rescue |
| `src/utils/pdf_decomposer.py` | gmft table detection + pdfplumber fallback |
| `src/utils/vector_index.py` | sentence-transformers + hnswlib retrieval |
| `src/utils/extraction_context.py` | Smart context builder for extraction |
| `src/utils/analysis_planner.py` | Data profiling → LLM plan → human review |
| `src/utils/statistics.py` | Pure Python REML, HKSJ, meta-regression, Egger |
| `src/utils/effect_sizes.py` | Effect size computation (OR, RR, SMD, VE) |
| `src/utils/visualizations.py` | Forest, funnel, LOO, cumulative, subgroup plots |
| `src/utils/nma.py` | NMA orchestrator: R netmeta, P-score, consistency |
| `src/utils/rob2.py` | Cochrane RoB-2 risk of bias (5 domains, RCTs) |
| `src/utils/robins_i.py` | ROBINS-I risk of bias (7 domains, non-RCTs) |
| `src/utils/grade.py` | GRADE evidence certainty (5 downgrade + 3 upgrade) |
| `src/utils/extraction_validator.py` | Per-field accuracy, TOST equivalence, Bland-Altman, error taxonomy |
| `src/utils/concordance_checker.py` | RoB agreement (Cohen's κ), synthesis concordance (ES, CI overlap) |
| `src/utils/citation_verifier.py` | BM25 + assertion extraction for citation verification |
| `src/utils/screening_benchmark.py` | ROC curve, WSS@95%, calibration, 5-arm benchmark |
| `src/utils/agreement.py` | Cohen's / weighted / Fleiss' kappa, PABAK |
| `src/utils/cost_tracker.py` | Cost aggregation + cost curve visualization |
| `src/utils/reproducibility.py` | SHA-256 config fingerprint |
| `src/utils/prisma_s.py` | PRISMA-S 2021 search reporting checklist |
| `src/utils/human_review.py` | Human override overlay + agreement tracking |

## Cost Transparency

LUMEN v2 tracks every API call with per-phase USD budgets:

```
TOKEN_BUDGET_PHASE1=2.00          # Strategy generation
TOKEN_BUDGET_PHASE3_TA=8.00       # Title/abstract screening
TOKEN_BUDGET_PHASE3_FT=5.00       # Full-text screening
TOKEN_BUDGET_PHASE4=15.00         # Data extraction
TOKEN_BUDGET_PHASE5=2.00          # Statistical synthesis
TOKEN_BUDGET_PHASE6=5.00          # Manuscript drafting
TOKEN_BUDGET_QUALITY_ASSESSMENT=5.00
```

Run `python scripts/run_cost_report.py` for a full cost dashboard with per-phase breakdowns and cumulative cost curves.

## License

MIT

## Citation

If you use LUMEN v2 in your research, please cite:

```
@software{lumen_v2,
  author = {Huang, Yen-Hsun and Lin, Yu-Shiou},
  title = {LUMEN: LLM-based Unified Meta-analysis Extraction Network},
  year = {2026},
  url = {https://github.com/YHHuan/LUMEN-LLM-based-Unified-Meta-analysis-Extraction-Network}
}
```
