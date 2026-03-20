# Validation Summary — PSY_MCI_rTMS

Generated: 2026-03-20

## Screening Validation

- **Sample**: 100 studies (stratified: 43 PMID, 56 RIS, 1 other)
- **Ground truth**: 2 human reviewers, consensus adjudicated, 2 corrections applied
- **Sensitivity**: 1.000 (0 false negatives)
- **Specificity**: 0.862 (8 false positives — all over-inclusions)
- **F1**: 0.913
- **LLM κ**: 0.840 (vs human ground truth)
- **Human inter-rater κ**: 0.709 / PABAK 0.720

**False positive pattern**: 2 review articles, 2 editorials, 1 NIS, 3 unspecified.
All errors favor recall over precision (safe direction for systematic review).

## Extraction Validation

- **Sample**: 10 studies, 324 fields (50 study_design, 60 population, 60 intervention, 154 outcomes)
- **Precision**: 0.997
- **Recall**: 0.974
- **F1**: 0.985
- **Fabrications**: 0

**Error pattern**: 1 value error (Boggio — timepoint confusion), 7 misses (6 Padala — 3MS/MMSE scale confusion, 1 Lu — p-value not extracted). All errors in table-sourced numerical outcomes. Text extraction was perfect.

## Key Takeaways

1. LLM screening achieves perfect sensitivity with specificity comparable to human reviewers
2. LLM agreement with ground truth (κ=0.840) exceeds human inter-rater agreement (κ=0.709)
3. Data extraction is highly accurate (F1=0.985) with zero fabrications
4. Remaining errors are systematic (multi-scale/multi-timepoint studies) and addressable via prompt improvement

## Files

| File | Description |
|------|-------------|
| `screening/screening_validation.csv` | Raw annotations (100 studies, 14 columns) |
| `screening/screening_metrics.json` | Computed metrics + confusion matrix |
| `extraction/extraction_validation.csv` | Raw field-level annotations (324 rows) |
| `extraction/source_pdfs/` | 10 validation study PDFs |
| `extraction/extraction_metrics.json` | Computed metrics + error details |
