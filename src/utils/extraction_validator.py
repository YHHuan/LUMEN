"""
Extraction Validator — LUMEN v2
================================
Compares pipeline-extracted data against ground truth for validation.

Produces:
  - Per-field accuracy (Table 8)
  - TOST equivalence testing for continuous fields
  - Bland-Altman analysis (Figure 6)
  - Error taxonomy classification

Ground truth format (phase4_extraction/ground_truth.json):
[
  {
    "study_id": "Smith_2024_12345678",
    "canonical_citation": "Smith 2024",
    "outcomes": [
      {
        "measure": "HAM-D change",
        "outcome_type": "continuous",
        "intervention_group": {"mean": 12.3, "sd": 4.5, "n": 50},
        "control_group": {"mean": 15.1, "sd": 5.2, "n": 48}
      }
    ]
  }
]
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ======================================================================
# Error Taxonomy
# ======================================================================

ERROR_TYPES = {
    "transcription": "Value copied incorrectly (typo, digit swap)",
    "unit_conversion": "Correct value but wrong unit or scale",
    "arm_mismatch": "Values assigned to wrong intervention/control arm",
    "missing_imputation": "Missing value imputed or hallucinated",
    "outcome_mislabel": "Value from wrong outcome measure",
    "aggregation": "Subgroup values used instead of total, or vice versa",
    "correct": "Extracted value matches ground truth",
}


def classify_error(extracted: float, ground_truth: float,
                   field_name: str = "") -> str:
    """Classify the type of extraction error."""
    if extracted is None or ground_truth is None:
        return "missing_imputation"

    if math.isclose(extracted, ground_truth, rel_tol=0.01, abs_tol=0.01):
        return "correct"

    # Check for unit conversion (factor of 10, 100, 1000)
    if ground_truth != 0:
        ratio = extracted / ground_truth
        for factor in [10, 100, 1000, 0.1, 0.01, 0.001]:
            if math.isclose(ratio, factor, rel_tol=0.05):
                return "unit_conversion"

    # Check for digit transposition (e.g., 12.3 vs 13.2)
    ext_str = f"{extracted:.2f}"
    gt_str = f"{ground_truth:.2f}"
    if len(ext_str) == len(gt_str):
        diffs = sum(1 for a, b in zip(ext_str, gt_str) if a != b)
        if diffs <= 2:
            return "transcription"

    # Check for sign flip or negation
    if math.isclose(abs(extracted), abs(ground_truth), rel_tol=0.01):
        return "transcription"

    return "outcome_mislabel"


# ======================================================================
# Per-Field Accuracy
# ======================================================================

NUMERIC_FIELDS = ["n", "mean", "sd", "events", "total"]


def compute_field_accuracy(extracted_studies: List[dict],
                           ground_truth: List[dict]) -> Dict:
    """
    Compare extracted data against ground truth, field by field.

    Returns:
        overall_accuracy: fraction of fields that match
        per_field: {field_name: {correct, total, accuracy}}
        per_study: {study_id: {correct, total, accuracy}}
        errors: list of individual field errors with classification
    """
    gt_by_id = {s["study_id"]: s for s in ground_truth}

    field_stats = {}
    study_stats = {}
    errors = []
    total_correct = 0
    total_fields = 0

    for ext_study in extracted_studies:
        sid = ext_study.get("study_id", "")
        gt_study = gt_by_id.get(sid)
        if not gt_study:
            continue

        study_correct = 0
        study_total = 0

        ext_outcomes = ext_study.get("outcomes", [])
        gt_outcomes = gt_study.get("outcomes", [])

        for gt_out in gt_outcomes:
            gt_measure = gt_out.get("measure", "")
            # Find matching extracted outcome
            ext_out = _match_outcome(ext_outcomes, gt_measure)
            if not ext_out:
                # All fields missing
                for group in ["intervention_group", "control_group"]:
                    gt_group = gt_out.get(group, {})
                    for field in NUMERIC_FIELDS:
                        if field in gt_group:
                            total_fields += 1
                            study_total += 1
                            _increment(field_stats, field, correct=False)
                            errors.append({
                                "study_id": sid,
                                "outcome": gt_measure,
                                "group": group,
                                "field": field,
                                "extracted": None,
                                "ground_truth": gt_group[field],
                                "error_type": "missing_imputation",
                            })
                continue

            # Compare fields
            for group in ["intervention_group", "control_group"]:
                gt_group = gt_out.get(group, {})
                ext_group = ext_out.get(group, {})

                for field in NUMERIC_FIELDS:
                    if field not in gt_group:
                        continue

                    gt_val = gt_group[field]
                    ext_val = ext_group.get(field)
                    total_fields += 1
                    study_total += 1

                    if ext_val is not None:
                        try:
                            ext_val = float(ext_val)
                            gt_val = float(gt_val)
                        except (ValueError, TypeError):
                            ext_val = None

                    error_type = classify_error(ext_val, gt_val, field)
                    is_correct = error_type == "correct"

                    if is_correct:
                        total_correct += 1
                        study_correct += 1

                    _increment(field_stats, field, correct=is_correct)

                    if not is_correct:
                        errors.append({
                            "study_id": sid,
                            "outcome": gt_measure,
                            "group": group,
                            "field": field,
                            "extracted": ext_val,
                            "ground_truth": gt_val,
                            "error_type": error_type,
                        })

        if study_total > 0:
            study_stats[sid] = {
                "correct": study_correct,
                "total": study_total,
                "accuracy": round(study_correct / study_total, 4),
            }

    # Compute per-field accuracy
    per_field = {}
    for field, stats in field_stats.items():
        per_field[field] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": round(stats["correct"] / max(stats["total"], 1), 4),
        }

    # Error taxonomy breakdown
    error_counts = {}
    for e in errors:
        et = e["error_type"]
        error_counts[et] = error_counts.get(et, 0) + 1

    dominant_error = max(error_counts, key=error_counts.get) if error_counts else "none"

    return {
        "overall_accuracy": round(total_correct / max(total_fields, 1), 4),
        "total_correct": total_correct,
        "total_fields": total_fields,
        "per_field": per_field,
        "per_study": study_stats,
        "errors": errors,
        "error_taxonomy": error_counts,
        "dominant_error_type": dominant_error,
    }


def _match_outcome(outcomes: List[dict], measure: str) -> Optional[dict]:
    """Find the best matching outcome by measure name."""
    measure_lower = measure.lower().strip()
    for o in outcomes:
        if o.get("measure", "").lower().strip() == measure_lower:
            return o
    # Fuzzy: check if measure is substring
    for o in outcomes:
        om = o.get("measure", "").lower()
        if measure_lower in om or om in measure_lower:
            return o
    return None


def _increment(stats: dict, field: str, correct: bool):
    if field not in stats:
        stats[field] = {"correct": 0, "total": 0}
    stats[field]["total"] += 1
    if correct:
        stats[field]["correct"] += 1


# ======================================================================
# TOST Equivalence Testing
# ======================================================================

def tost_equivalence(extracted_values: List[float],
                     ground_truth_values: List[float],
                     margin: float = 0.10) -> Dict:
    """
    Two One-Sided Tests (TOST) for equivalence.

    Tests H0: |mean_diff| >= margin * mean_gt
    vs H1: |mean_diff| < margin * mean_gt

    Args:
        extracted_values: pipeline-extracted numeric values
        ground_truth_values: ground truth numeric values
        margin: equivalence margin as fraction of GT mean (default 10%)

    Returns:
        tost_p: max of two one-sided p-values
        mean_diff: mean(extracted - gt)
        equivalence_margin: absolute margin used
        equivalent: True if tost_p < 0.05
    """
    from scipy import stats as scipy_stats

    pairs = [
        (e, g) for e, g in zip(extracted_values, ground_truth_values)
        if e is not None and g is not None
    ]
    if len(pairs) < 3:
        return {
            "tost_p": None,
            "mean_diff": None,
            "equivalence_margin": None,
            "equivalent": None,
            "n_pairs": len(pairs),
        }

    diffs = [e - g for e, g in pairs]
    gt_vals = [g for _, g in pairs]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    sd_diff = (sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)) ** 0.5
    se = sd_diff / n ** 0.5

    # Equivalence margin based on GT mean
    mean_gt = abs(sum(gt_vals) / len(gt_vals))
    eq_margin = max(margin * mean_gt, 0.5)  # at least 0.5 absolute

    # Two one-sided t-tests
    t_upper = (mean_diff - eq_margin) / max(se, 1e-10)
    t_lower = (mean_diff + eq_margin) / max(se, 1e-10)

    p_upper = scipy_stats.t.cdf(t_upper, df=n - 1)
    p_lower = 1 - scipy_stats.t.cdf(t_lower, df=n - 1)

    tost_p = max(p_upper, p_lower)

    return {
        "tost_p": round(tost_p, 6),
        "mean_diff": round(mean_diff, 4),
        "sd_diff": round(sd_diff, 4),
        "equivalence_margin": round(eq_margin, 4),
        "equivalent": tost_p < 0.05,
        "n_pairs": n,
    }


# ======================================================================
# Bland-Altman Analysis
# ======================================================================

def bland_altman(extracted_values: List[float],
                 ground_truth_values: List[float]) -> Dict:
    """
    Bland-Altman analysis for agreement between extracted and GT values.

    Returns:
        mean_diff: bias (mean of differences)
        sd_diff: standard deviation of differences
        loa_upper: upper limit of agreement (+1.96 SD)
        loa_lower: lower limit of agreement (-1.96 SD)
        points: list of {mean, diff} for plotting
    """
    pairs = [
        (e, g) for e, g in zip(extracted_values, ground_truth_values)
        if e is not None and g is not None
    ]
    if len(pairs) < 2:
        return {"mean_diff": None, "sd_diff": None, "points": []}

    diffs = [e - g for e, g in pairs]
    means = [(e + g) / 2 for e, g in pairs]
    n = len(diffs)

    mean_diff = sum(diffs) / n
    sd_diff = (sum((d - mean_diff) ** 2 for d in diffs) / max(n - 1, 1)) ** 0.5

    return {
        "mean_diff": round(mean_diff, 4),
        "sd_diff": round(sd_diff, 4),
        "loa_upper": round(mean_diff + 1.96 * sd_diff, 4),
        "loa_lower": round(mean_diff - 1.96 * sd_diff, 4),
        "n_pairs": n,
        "points": [
            {"mean": round(m, 4), "diff": round(d, 4)}
            for m, d in zip(means, diffs)
        ],
    }


# ======================================================================
# Full Validation Report
# ======================================================================

def validate_extraction(extracted_path: str,
                        ground_truth_path: str,
                        arm_name: str = "LUMEN") -> Dict:
    """
    Run full extraction validation: accuracy + TOST + Bland-Altman.

    Args:
        extracted_path: path to extracted_data.json
        ground_truth_path: path to ground_truth.json
        arm_name: label for this validation arm

    Returns:
        Complete validation report dict
    """
    with open(extracted_path, encoding="utf-8") as f:
        extracted = json.load(f)
    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Per-field accuracy
    accuracy = compute_field_accuracy(extracted, ground_truth)

    # Collect paired numeric values per field for TOST and Bland-Altman
    gt_by_id = {s["study_id"]: s for s in ground_truth}
    field_pairs = {f: {"extracted": [], "gt": []} for f in NUMERIC_FIELDS}

    for ext_study in extracted:
        sid = ext_study.get("study_id", "")
        gt_study = gt_by_id.get(sid)
        if not gt_study:
            continue

        for gt_out in gt_study.get("outcomes", []):
            ext_out = _match_outcome(
                ext_study.get("outcomes", []),
                gt_out.get("measure", ""),
            )
            if not ext_out:
                continue

            for group in ["intervention_group", "control_group"]:
                gt_g = gt_out.get(group, {})
                ext_g = ext_out.get(group, {})

                for field in NUMERIC_FIELDS:
                    if field in gt_g and field in ext_g:
                        try:
                            ev = float(ext_g[field])
                            gv = float(gt_g[field])
                            field_pairs[field]["extracted"].append(ev)
                            field_pairs[field]["gt"].append(gv)
                        except (ValueError, TypeError):
                            pass

    # TOST per field
    tost_results = {}
    for field in NUMERIC_FIELDS:
        p = field_pairs[field]
        if p["extracted"]:
            tost_results[field] = tost_equivalence(p["extracted"], p["gt"])

    # Bland-Altman per field
    ba_results = {}
    for field in NUMERIC_FIELDS:
        p = field_pairs[field]
        if p["extracted"]:
            ba_results[field] = bland_altman(p["extracted"], p["gt"])

    # Overall TOST (all numeric fields pooled)
    all_ext = []
    all_gt = []
    for field in NUMERIC_FIELDS:
        all_ext.extend(field_pairs[field]["extracted"])
        all_gt.extend(field_pairs[field]["gt"])
    overall_tost = tost_equivalence(all_ext, all_gt) if all_ext else {}

    return {
        "arm": arm_name,
        "n_studies_matched": len(accuracy["per_study"]),
        "accuracy": accuracy,
        "tost": {
            "overall": overall_tost,
            "per_field": tost_results,
        },
        "bland_altman": ba_results,
        "field_pairs": {
            field: {"n": len(field_pairs[field]["extracted"])}
            for field in NUMERIC_FIELDS
        },
    }


def format_validation_table(results: List[Dict]) -> str:
    """
    Format Table 8: Extraction validation across domains/arms.

    Args:
        results: list of validate_extraction() outputs
    """
    lines = [
        "# Table 8: Extraction Validation",
        "",
        "| Arm | Accuracy (%) | MAE (continuous) | TOST p | Dominant Error |",
        "|-----|-------------|-----------------|--------|----------------|",
    ]

    for r in results:
        acc = r["accuracy"]
        tost = r["tost"]["overall"]
        tost_p = f"{tost['tost_p']:.4f}" if tost.get("tost_p") is not None else "N/A"
        mae = abs(tost["mean_diff"]) if tost.get("mean_diff") is not None else None
        mae_str = f"{mae:.3f}" if mae is not None else "N/A"

        lines.append(
            f"| {r['arm']} | {acc['overall_accuracy'] * 100:.1f} | "
            f"{mae_str} | {tost_p} | {acc['dominant_error_type']} |"
        )

    return "\n".join(lines)
