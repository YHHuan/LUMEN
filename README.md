# LUMEN v3

**LLM-powered Unified Meta-analysis Extraction Network** — an automated systematic review and meta-analysis pipeline using multi-agent LLM orchestration.

LUMEN v3 takes a research question (PICO) and produces a complete systematic review: from literature search through screening, data extraction, meta-analysis, quality assessment, and manuscript generation — with inline fact-checking.

## Quick Start

```bash
# Install
cd lumen-v3
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Set API key (at least one required)
export OPENROUTER_API_KEY=sk-or-...   # recommended: unified access to all models
# or individual keys: ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY

# Run
lumen run --project ./my_review
```

## Pipeline Overview

```
 PICO Definition ──► Search Strategy ──► PubMed / OpenAlex Search
                                              │
                                         Deduplication
                                              │
                                     Keyword Pre-screen
                                              │
                                    ┌─────────┴─────────┐
                                    │  Dual T/A Screen   │
                                    │  (Gemini + Sonnet) │
                                    └─────────┬─────────┘
                                              │
                                  ┌───── Arbiter ──────┐
                                  │  (conflict cases)  │
                                  └────────┬───────────┘
                                           │
                                    PDF Acquisition
                                           │
                                   Full-text Screening
                                           │
                                  4-Round Extraction
                                           │
                                 Outcome Harmonization
                                           │
                                ┌──────────┴──────────┐
                                │   Meta-analysis      │
                                │   (deterministic)    │
                                └──────────┬──────────┘
                                           │
                               Quality Assessment (RoB-2 + GRADE)
                                           │
                                  Manuscript Writing
                                           │
                                  Inline Fact-Check
```

## Architecture

### 9-Agent System

| Agent | Model Tier | Phase | What It Does |
|-------|-----------|-------|--------------|
| **PICO Interviewer** | Strategic | 1 | Scores PICO completeness (0–100), refines via LLM if below threshold |
| **Strategy Generator** | Smart | 1 | Converts PICO into PubMed/OpenAlex queries + screening criteria |
| **Screener** (×2) | Fast + Smart | 3 | Cross-model dual screening — two different LLMs screen each study independently |
| **Arbiter** | Strategic | 3 | Resolves cases where screeners disagree or both have low confidence |
| **Fulltext Screener** | Smart | 3 | Verifies PICO eligibility against full PDF text |
| **Extractor** | Smart | 4 | 4-round IterResearch extraction with evidence span binding |
| **Harmonizer** | Smart | 4.5 | Clusters synonymous outcomes across studies (embedding + LLM) |
| **Statistician** | Smart | 5 | Plans analysis, executes deterministic meta-analysis, interprets results |
| **Writer** | Smart | 6 | Sequential manuscript sections + inline fact-check against source data |

### 3-Tier Model Routing

Every LLM call is routed through one of three tiers. If the primary model fails (rate limit, timeout), the system automatically falls back.

| Tier | Primary | Fallback | Used For |
|------|---------|----------|----------|
| **Fast** | Gemini 2.5 Flash | GPT-4.1 Mini | Screener 1, keyword filtering |
| **Smart** | Claude Sonnet 4.5 | Gemini 2.5 Pro | Most agents — extraction, analysis, writing |
| **Strategic** | Claude Opus 4.1 | GPT-5 | PICO interviewer, arbiter (high-stakes decisions) |

All models are accessed via OpenRouter by default. To use provider APIs directly, set the corresponding `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` and update `configs/models.yaml`.

### Cross-Model Dual Screening

Two LLMs from different providers screen every study independently, exploiting complementary bias profiles:

- **Either includes** → auto-include (union principle — maximizes recall)
- **Both exclude with high confidence (≥80)** → auto-exclude
- **Both exclude but uncertain** → arbiter reviews
- **Arbiter uncertain** → flagged for human review

This design is based on Sanghera et al. (JAMIA 2025) and Oami et al. (Research Synthesis Methods 2025).

### 4-Round IterResearch Extraction

Each study goes through four extraction rounds. Critically, each round only receives distilled output from prior rounds — raw PDF is NOT carried forward after Round 1.

| Round | Input | Output |
|-------|-------|--------|
| 1. **Skeleton** | Full PDF + PICO | Study design, arms, outcome list, key tables |
| 2. **Extract** | Skeleton + PDF | Numerical data per outcome (n, mean, SD, events, total) |
| 3. **Cross-check** | Skeleton + extracted data (no PDF) | Consistency checks (events ≤ total, SD vs SE, etc.) |
| 4. **Evidence spans** | Extracted data + PDF | Maps each value to its source location in the PDF |

**Dynamic token budget**: Output token limits are estimated from data size (number of outcomes, PDF pages) rather than a fixed cap, preventing JSON truncation on studies with many outcomes.

**Reference stripping**: Bibliography sections are automatically detected and removed before extraction, saving ~13% of input tokens per study.

**Auto-retry**: Parse failures (transient LLM formatting issues) trigger an automatic retry of the full 4-round sequence.

### Deterministic Statistics

All statistical calculations use validated, deterministic functions — the LLM plans the analysis but never computes numbers.

- **Random-effects meta-analysis**: REML with DerSimonian-Laird fallback (method tracked in output)
- **HKSJ adjustment**: Hartung-Knapp-Sidik-Jonkman for conservative confidence intervals
- **Prediction intervals**: Computed for k ≥ 3 studies
- **Publication bias**: Egger's test + trim-and-fill (direction flip flagged as critical)
- **Sensitivity**: Leave-one-out analysis for all outcomes
- **Effect sizes**: Hedges' g, mean difference, log risk ratio, odds ratio, risk difference
- **SD imputation**: Automatically recovers SD from SE+n or CI+n when reported data is incomplete

### Quality Assessment

- **RoB-2**: All five domains enforced (randomization, deviations, missing data, measurement, reporting). Incomplete assessments raise an error.
- **GRADE**: Certainty of evidence across five downgrade domains. Indirectness never defaulted to "no concern" (v2 audit fix).

### Inline Fact-Check

Every manuscript section is verified against the pipeline's own source data:

| Verdict | Meaning | Action |
|---------|---------|--------|
| **SUPPORTED** | Claim matches statistics/extractions | None |
| **CONTRADICTED** | Claim conflicts with data | Auto-corrected text provided |
| **UNSUPPORTED** | Claim cannot be verified from pipeline data | Marked for human review |

### Anomaly Detection

| Anomaly | Threshold | Action |
|---------|-----------|--------|
| High heterogeneity | I² > 90% | Leave-one-out analysis triggered |
| Weight dominance | Single study > 40% weight | Influence analysis flagged |
| Underpowered | k < 3 | Warning in interpretation |
| Trim-and-fill direction flip | Effect reverses after adjustment | Critical flag |
| Prediction interval crosses null | PI crosses null but CI doesn't | Noted in interpretation |

## Usage

### Full Pipeline (Non-interactive)

```bash
lumen run --project ./my_review
```

Place input files in the project directory before running:

- `pico.json` — PICO definition (optional; LLM will elicit if missing)
- `studies.json` — Pre-loaded search results (optional; pipeline will search PubMed/OpenAlex if missing)

### Interactive Mode

```bash
lumen interactive --project ./my_review
```

Step-by-step mode with human checkpoints at PICO, strategy, and analysis stages.

### Custom Runner (Recommended for Large Projects)

For projects with pre-curated study lists or when you need checkpoint-resume capability, use a custom runner script (see `met_ovary/run_pipeline.py` for a working example). Key advantages:

- **Stage-by-stage checkpoints**: Each stage saves a `cp{N}_*.json` file. On re-run, completed stages are skipped automatically.
- **Pre-seeded PDFs**: Studies with pre-populated `pdf_content` in `studies.json` skip the download step.
- **Keyword override**: Override auto-generated prescreen keywords when using pre-curated study lists.

### Other Commands

```bash
# Cost summary
lumen cost --project ./my_review

# Validate manuscript against source data
lumen validate --project ./my_review

# Generate visualizations
lumen plot --project ./my_review --type prisma
lumen plot --project ./my_review --type forest
lumen plot --project ./my_review --type funnel
```

## Project Directory Structure

```
my_review/
├── pico.json                    # PICO definition
├── studies.json                 # Search results (pre-loaded or from pipeline)
├── cost_log.jsonl               # Per-call cost tracking (model, tokens, cost, latency)
├── output/
│   ├── abstract.txt             # ┐
│   ├── introduction.txt         # │
│   ├── methods.txt              # ├── Manuscript sections
│   ├── results.txt              # │
│   ├── discussion.txt           # ┘
│   ├── statistics.json          # Pooled estimates, heterogeneity, sensitivity
│   ├── fact_check.json          # Per-claim verdicts + auto-corrections
│   ├── cost_summary.json        # Aggregated cost by agent/phase
│   ├── pipeline_state.json      # Full state snapshot (for PRISMA diagram)
│   └── cp{1..11}_*.json         # Stage checkpoints (for resume)
└── logs/
    └── run.log                  # Full debug log with structlog output
```

## Source Code Structure

```
lumen-v3/
├── configs/
│   ├── default.yaml             # Pipeline thresholds and settings
│   └── models.yaml              # 3-tier model definitions (editable)
├── prompts/                     # 16 externalized YAML prompts (one per agent task)
├── lumen/
│   ├── core/                    # Infrastructure
│   │   ├── config.py            # YAML config loader
│   │   ├── context.py           # PICO drift check + compression
│   │   ├── cost.py              # Per-agent JSONL cost tracking
│   │   ├── graph.py             # LangGraph StateGraph (14 nodes, 2 conditional edges)
│   │   ├── router.py            # 3-tier LiteLLM router with automatic fallback
│   │   └── state.py             # LumenState TypedDict
│   ├── agents/                  # 9 agents + 2 orchestration nodes
│   │   ├── base.py              # BaseAgent: LLM call, JSON parse, retry, prompt loading
│   │   ├── pico_interviewer.py  # Completeness scoring (deterministic) + LLM refinement
│   │   ├── strategy_generator.py
│   │   ├── screener.py          # Single + batch screening (tier-overridable)
│   │   ├── screening_node.py    # Prescreen + dual-screen + arbiter orchestration
│   │   ├── arbiter.py           # Union-hybrid conflict resolution
│   │   ├── fulltext_screener.py
│   │   ├── extractor.py         # 4-round IterResearch with dynamic token budget
│   │   ├── harmonizer.py        # Embedding + LLM outcome clustering
│   │   ├── statistician.py      # 5-step pipeline with SD imputation
│   │   ├── quality_node.py      # RoB-2 (5-domain) + GRADE
│   │   └── writer.py            # Evidence synthesis → sequential sections → fact-check
│   ├── tools/
│   │   ├── search/              # PubMed + OpenAlex search APIs
│   │   ├── pdf/                 # Multi-source downloader + pdfplumber reader + ref stripping
│   │   ├── statistics/          # Effect sizes, meta-analysis, heterogeneity, publication bias
│   │   ├── quality/             # RoB-2, GRADE
│   │   └── visualization/       # PRISMA flow, forest/funnel plots, cost report
│   └── interface/
│       └── cli.py               # CLI: run, interactive, cost, validate, plot
├── tests/
│   ├── unit/                    # 30 test modules
│   └── integration/             # 5 test modules
└── pyproject.toml
```

## Cost Estimate

Typical costs for a systematic review (based on met_ovary pilot: 937 candidate studies → 6 included):

| Phase | Calls | Tokens | Cost |
|-------|------:|-------:|-----:|
| Screening (Gemini Flash + Sonnet dual) | ~1,876 | ~3.7M | ~$9.0 |
| Extraction (4 rounds × included studies) | ~80 | ~1.7M | ~$9.2 |
| Writing + Fact-check | ~30 | ~1.2M | ~$4.9 |
| Fulltext screening | ~35 | ~260K | ~$1.0 |
| Statistics + Quality + PICO + Strategy | ~30 | ~230K | ~$1.9 |
| **Total** | **~2,050** | **~7.1M** | **~$26** |

Cost scales primarily with the number of candidate studies (screening) and included studies (extraction).

## Setup

### Prerequisites

- Python 3.11+
- API key for at least one LLM provider

### Installation

```bash
cd lumen-v3
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"     # include [dev] for pytest
```

### API Keys

```bash
# Recommended: OpenRouter gives access to all models with one key
export OPENROUTER_API_KEY=sk-or-...

# Or use individual provider keys:
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
export OPENAI_API_KEY=sk-...

# Optional: higher PubMed rate limits
export NCBI_API_KEY=...
export NCBI_EMAIL=you@example.com
```

### Verify

```bash
pytest tests/ -q
lumen --help
```

## Testing

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/unit/test_statistician.py -v

# With coverage
pytest tests/ --cov=lumen --cov-report=term-missing
```

## Key Design Decisions

1. **Cross-model screening over single-model**: Two LLMs with different architectures catch different types of errors. Union-based inclusion maximizes recall at moderate cost.

2. **Deterministic statistics**: The LLM plans the analysis (which effect size, which subgroups) but never performs calculations. All math runs through validated Python functions with known edge-case handling.

3. **IterResearch extraction (4 rounds)**: Prevents the "copy the whole table" failure mode. Each round has a focused task with only the context it needs. Round 3 catches internal inconsistencies without PDF access (forces the model to reason from data alone).

4. **Dynamic token budgets**: Output limits scale with the actual data (number of outcomes, PDF pages). A study with 5 outcomes gets ~4K tokens; a study with 25 outcomes gets ~16K. This eliminated the JSON truncation failures that plagued fixed-budget approaches.

5. **Reference stripping**: Bibliography sections contain no extractable data but consume ~13–27% of PDF tokens. Removing them before extraction saves cost and reduces noise.

6. **Fact-check as a pipeline stage**: Every claim in the manuscript is verified against the pipeline's own statistics and extractions. Contradicted claims receive auto-corrected text. This catches the "hallucinated numbers" problem common in LLM-generated scientific text.

## License

MIT
