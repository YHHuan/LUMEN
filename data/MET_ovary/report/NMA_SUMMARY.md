# MET_ovary NMA Summary Report

## Pipeline Overview
- **Search**: PubMed + Scopus + EPMC = 1,061 records
- **Dedup**: 937 unique
- **Pre-screening**: 543 passed (394 excluded, 115 for no_abstract)
- **Dual T/A Screening**: 19 included, 7 human review (Cohen's kappa = 0.77)
- **Human review**: 3 added (total 22)
- **PDF download**: 25/25 acquired
- **Full-text screening**: 11 included
- **Extraction (NMA)**: 11 studies, 9 with usable arm data
- **NMA**: 4 outcomes feasible (>= 2 studies), all 4 networks connected

## NMA Results

### 1. Weight Change (kg) — 7 studies, 4 treatments
- **tau2 = 0.098, I2 = 54.2%** (moderate heterogeneity)
- **Consistency**: Q_between p=0.035 (borderline); node-splitting all p>0.7
- **Rankings**:
  1. GLP-1RA + Metformin (P=0.996)
  2. Metformin (P=0.532)
  3. GLP-1RA (P=0.472)
  4. Placebo (P=0.000)
- **Key comparison**: GLP-1RA+Met vs Met: SMD=1.87 [1.23, 2.85] (significant)

### 2. BMI Change (kg/m2) — 6 studies, 4 treatments
- **tau2 = 0.173, I2 = 62.6%** (substantial heterogeneity)
- **Consistency**: Q_between p=0.020 (significant inconsistency)
- **Rankings**:
  1. GLP-1RA + Metformin (P=0.975)
  2. Metformin (P=0.561)
  3. GLP-1RA (P=0.464)
  4. Placebo (P=0.000)

### 3. HOMA-IR — 3 studies, 3 treatments
- **tau2 ~ 0, I2 = 0%** (no heterogeneity)
- **Consistency**: p=0.706 (consistent)
- **Rankings**:
  1. GLP-1RA + Metformin (P=0.905)
  2. Metformin (P=0.446)
  3. GLP-1RA (P=0.150)

### 4. Waist Circumference (cm) — 2 studies, 3 treatments
- **tau2 = 0, I2 = NA** (too few studies)
- **Rankings**:
  1. GLP-1RA (P=0.977)
  2. Metformin (P=0.523)
  3. Placebo (P=0.000)

## Key Findings
1. **GLP-1RA + Metformin is consistently the best treatment** across Weight, BMI, and HOMA-IR (P-score > 0.9)
2. **GLP-1RA monotherapy vs Metformin monotherapy**: no significant difference for weight/BMI
3. **HOMA-IR**: perfect consistency (I2=0%), GLP-1RA+Met best
4. **BMI has significant inconsistency** (p=0.02) — needs discussion in manuscript

## Limitations
- Only 8 studies with NMA-usable data (continuous outcomes only)
- No binary outcome (ovulation rate) had sufficient studies for NMA
- Protocol primary outcome (ovulation rate) not analysable via NMA
- Outcome harmonization done post-hoc (similar but differently named measures merged)

## Output Files
- `pdfs/` — 25 included study PDFs
- `nma_figures/` — per-outcome: network graph, forest, funnel, ranking, net heat, node-split
- `nma_tables/` — per-outcome: league table, rankings, node-splitting, LOO, CINeMA template
- `nma_results/` — per-outcome JSON with all numerical results

## Pipeline Cost
- Phase 1 (Strategy): $0.11
- Phase 2.5 (Pre-screening): ~$0.05
- Phase 3.1 (Dual Screening): $3.49
- Phase 3.3 (Full-text): $0.86
- Phase 4 (Extraction): $3.89
- Phase 5 (NMA + Interpretation): ~$0.20
- **Total: ~$8.60**
