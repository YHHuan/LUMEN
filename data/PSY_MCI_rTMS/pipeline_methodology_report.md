# Pipeline Methodology Report — PSY_MCI_rTMS

Generated: 2026-03-15

---

## 1. Pipeline Overview

| Phase | Description | Model(s) | Cost (USD) | LLM Calls | Input Tokens | Output Tokens | Cache Read |
|-------|-------------|----------|------------|-----------|-------------|--------------|------------|
| 1 — Strategy | PICO → MeSH + search queries | Claude Sonnet 4.6 | $0.07 | 2 | 1,552 | 7,036 | 0 |
| 2 — Search | Multi-database retrieval + dedup | (no LLM) | $0.00 | 0 | — | — | — |
| 3.1 — Title/Abstract Screening | Dual-screener 5-point scale | Gemini 3.1 Flash Lite + GPT-4.1 Mini + Claude Sonnet 4.6 (arbiter) | $6.05 | 2,243 | 4,472,022 | 446,644 | 689,961 |
| 3.2 — Full-text Screening | PDF-based inclusion/exclusion | Gemini 3.1 Flash Lite + GPT-4.1 Mini | $3.06 | 589 | 1,421,192 | 540,474 | 291,633 |
| 4 — Data Extraction | PDF → structured data + RoB | Gemini 3.1 Pro + GPT-5.4 (tiebreaker) | $14.11 | 782 | 3,327,948 | 1,383,730 | 102,147 |
| 5 — Statistical Analysis | Random-effects meta-analysis | GPT-5.4 (interpretation) | $0.30 | 3 | 6,262 | 24,568 | 0 |
| 6 — Manuscript Writing | Section drafting + citation check | Claude Sonnet 4.6 + GPT-5.4 (citations) | $1.10 | 33 | 108,140 | 58,417 | 12,645 |
| **TOTAL** | | | **$24.68** | **3,652** | **9,337,116** | **2,460,869** | **1,096,386** |

> Note: Phase 4 cost adjusted by −$12.00 (previous run budget overlap not cleared).

---

## 2. Literature Search (Phase 2)

### 2.1 Databases Searched

| Database | Type | Records Retrieved |
|----------|------|-------------------|
| PubMed (NCBI) | API | 236 |
| Europe PMC | API | 1,864 |
| Scopus (Elsevier) | API | 603 |
| Cochrane Library | Manual RIS | 780 |
| Embase (Web of Science) | Manual RIS (5 batches) | 2,495 |
| **Total raw** | | **5,978** |

### 2.2 Deduplication

- After deduplication: **4,045** unique studies
- Duplicates removed: **1,933** (32.3%)
- Method: DOI / PMID exact match + fuzzy title matching

### 2.3 Pre-screening

- Keyword-based pre-screen applied: 4,045 → **1,947** candidates passed to Phase 3

---

## 3. Screening (Phase 3)

### 3.1 Title/Abstract Screening

- **Total screened**: 1,947 studies
- **Dual-screener system**: Gemini 3.1 Flash Lite (Screener 1) + GPT-4.1 Mini (Screener 2)
- **Rating scale**: 5-point confidence (`most_likely_include` → `most_likely_exclude`)
- **Agreement rate** (include/exclude side): 84.2%
- **Exact agreement rate** (same rating): 22.8%
- **Cohen's κ**: 0.495 (moderate agreement)
- **Firm conflicts** (arbiter invoked): 296 studies → arbitrated by Claude Sonnet 4.6
- **Undecided conflicts** (human review queue): 12 studies
- **Included**: 297 studies
- **Excluded**: 1,650 studies

### 3.2 Full-text Screening

- **PDFs reviewed**: 297 studies
- **Included after full-text review**: 186 studies
- **Excluded**: 111 studies

---

## 4. Data Extraction (Phase 4)

- **Studies extracted**: 186 / 186 (0 skipped)
- **Extraction model**: Gemini 3.1 Pro (primary) + GPT-5.4 (self-consistency tiebreaker)
- **Risk of Bias assessed**: 186 studies
- **With complete outcome data**: 24 studies (fully computable)
- **With partial data / issues**: 162 studies
- **Computable for meta-analysis** (after normalization): 48 studies

---

## 5. Statistical Analysis (Phase 5)

### 5.1 Meta-Analysis Results (Random-Effects, DerSimonian–Laird)

| Outcome Category | Measure | k | SMD | 95% CI | p-value | I² (%) | τ² | Significant |
|-----------------|---------|---|-----|--------|---------|--------|-----|-------------|
| Global cognition | ADAS-Cog | 13 | 0.204 | [−0.321, 0.728] | .447 | 85.7 | 0.668 | No |
| Global cognition | MMSE | 23 | 0.618 | [0.271, 0.966] | .0005 | 80.0 | 0.531 | **Yes** |
| Global cognition | MoCA | 8 | 0.624 | [0.117, 1.130] | .016 | 79.3 | 0.417 | **Yes** |
| Cognitive domains | Memory | 14 | 0.650 | [0.335, 0.964] | <.001 | 49.6 | 0.173 | **Yes** |
| Cognitive domains | Executive function | 6 | 0.060 | [−0.231, 0.351] | .686 | 0.0 | 0.000 | No |
| Cognitive domains | Attention | 5 | 0.541 | [0.045, 1.037] | .033 | 48.9 | 0.152 | **Yes** |
| Functional outcomes | ADL | 5 | −0.106 | [−0.638, 0.426] | .696 | 80.1 | 0.270 | No |
| Functional outcomes | IADL | 3 | 0.434 | [−0.679, 1.547] | .445 | 83.3 | 0.798 | No |
| Functional outcomes | ADCS-ADL | 3 | −0.504 | [−0.735, −0.274] | <.001 | 0.0 | 0.000 | **Yes** |
| Neuropsychiatric | NPI | 3 | 0.369 | [−0.116, 0.854] | .136 | 31.4 | 0.058 | No |
| Neuropsychiatric | GDS | 3 | 0.242 | [−0.197, 0.680] | .281 | 0.0 | 0.000 | No |

### 5.2 Figures Generated

| Figure | File | Description |
|--------|------|-------------|
| Forest plot — ADAS-Cog | `figures/forest_Global_cognition_ADAS-Cog.png` | k=13, random-effects |
| Forest plot — MMSE | `figures/forest_Global_cognition_MMSE.png` | k=23, random-effects |
| Forest plot — MoCA | `figures/forest_Global_cognition_MoCA.png` | k=8, random-effects |
| Forest plot — Memory | `figures/forest_Specific_cognitive_domains_memory.png` | k=14, random-effects |
| Forest plot — Executive | `figures/forest_Specific_cognitive_domains_executive_function.png` | k=6, random-effects |
| Forest plot — Attention | `figures/forest_Specific_cognitive_domains_attention.png` | k=5, random-effects |
| Forest plot — ADL | `figures/forest_Functional_outcomes_ADL.png` | k=5, random-effects |
| Forest plot — IADL | `figures/forest_Functional_outcomes_IADL.png` | k=3, random-effects |
| Forest plot — ADCS-ADL | `figures/forest_Functional_outcomes_ADCS-ADL.png` | k=3, random-effects |
| Forest plot — NPI | `figures/forest_Neuropsychiatric_symptoms_NPI.png` | k=3, random-effects |
| Forest plot — GDS | `figures/forest_Neuropsychiatric_symptoms_GDS.png` | k=3, random-effects |
| Funnel plots | `figures/funnel_*.png` | One per outcome (11 total) |
| PRISMA flow diagram | `figures/prisma_flow.png` | PRISMA 2020 compliant |
| Risk of Bias summary | `figures/rob_summary.png` | Traffic-light plot |
| Risk of Bias domains | `figures/rob_domains.png` | Domain-level bar chart |

---

## 6. Manuscript (Phase 6)

### Sections generated:
- `drafts/title.md`
- `drafts/abstract.md`
- `drafts/introduction.md`
- `drafts/methods.md`
- `drafts/results.md`
- `drafts/discussion.md`
- `drafts/manuscript_draft.md` (combined)

### Citation status:
- Internal citations: 0 (none auto-linked)
- `[CITATION NEEDED]` markers: 93 (need manual filling)

---

## 7. LLM Models Used

| Role | Model ID | Provider |
|------|----------|----------|
| Strategist | `anthropic/claude-sonnet-4-6` | Anthropic via OpenRouter |
| Screener 1 | `google/gemini-3.1-flash-lite-preview` | Google via OpenRouter |
| Screener 2 | `openai/gpt-4.1-mini` | OpenAI via OpenRouter |
| Arbiter | `anthropic/claude-sonnet-4-6` | Anthropic via OpenRouter |
| Extractor | `google/gemini-3.1-pro-preview` | Google via OpenRouter |
| Extractor Tiebreaker | `openai/gpt-5.4` | OpenAI via OpenRouter |
| Statistician | `openai/gpt-5.4` | OpenAI via OpenRouter |
| Writer | `anthropic/claude-sonnet-4-6` | Anthropic via OpenRouter |
| Citation Guardian | `openai/gpt-5.4` | OpenAI via OpenRouter |

---

## 8. Token Economy Summary

| Metric | Value |
|--------|-------|
| Total LLM cost | **$24.68 USD** |
| Total LLM calls | 3,652 |
| Total input tokens | 9,337,116 |
| Total output tokens | 2,460,869 |
| Total cache read tokens | 1,096,386 |
| Cache savings estimate | ~$0.99 (90% discount on cached reads) |
| Most expensive phase | Phase 4 — Data Extraction ($14.11, 57%) |
| Cheapest phase | Phase 1 — Strategy ($0.07, 0.3%) |

---

## 9. Key Pipeline Features (for Methods Paper)

1. **Multi-agent architecture**: 9 specialized LLM agents, each with role-specific prompts and model selection
2. **Dual-screening with arbitration**: Two independent screeners + arbiter for firm conflicts; 5-point confidence scale (not binary)
3. **Self-consistency extraction**: Primary extractor + tiebreaker model for quality assurance
4. **Prompt-based caching**: Content-hash based LLM response cache prevents duplicate API calls; saved ~1,096K tokens
5. **PDF text caching**: Extract once, reuse across phases (Phase 3 → Phase 4)
6. **Checkpoint system**: Every study processed is checkpointed; pipeline can resume from any interruption
7. **Token budget enforcement**: Per-phase USD limits with automatic stopping
8. **Prompt audit trail**: Every LLM call logged to `.audit/prompt_log.jsonl` (PRISMA-trAIce compliant)
9. **Multi-database search**: 5 databases (PubMed, Europe PMC, Scopus, Cochrane, Embase) with automated + manual import
10. **Keyword pre-screening**: Free (no LLM) pre-filter before costly LLM screening

---

## 10. PRISMA Flow Numbers

```
Identification:
  Database search: 5,978 records
  After dedup: 4,045 unique
  After pre-screen: 1,947 candidates

Screening:
  Title/abstract screened: 1,947
  Excluded: 1,650
  Full-text assessed: 297
  Excluded at full-text: 111

Included:
  Studies in meta-analysis: 186 (data extraction)
  Computable for quantitative synthesis: 48
```
