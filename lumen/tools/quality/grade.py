"""
GRADE (Grading of Recommendations, Assessment, Development and Evaluations).

Fixes v2 audit #3: indirectness must be explicitly assessed, not defaulted.
Fixes v2 audit #11: threshold sources documented as comments.
"""

from __future__ import annotations

from typing import Literal

# GRADE starts at "high" for RCTs and downgrades
STARTING_CERTAINTY = {"rct": 4, "observational": 2}

CERTAINTY_LABELS = {4: "high", 3: "moderate", 2: "low", 1: "very_low"}

DOMAIN_NAMES = [
    "risk_of_bias",
    "inconsistency",
    "indirectness",
    "imprecision",
    "publication_bias",
]


def assess_grade(
    rob_data: dict | None,
    inconsistency_data: dict | None,
    indirectness_data: dict | None,
    imprecision_data: dict | None,
    publication_bias_data: dict | None,
    study_design: str = "rct",
) -> dict:
    """
    Assess GRADE certainty of evidence.

    Parameters
    ----------
    rob_data : {"level": "no_concern" | "serious" | "very_serious", "reason": str}
    inconsistency_data : same format
    indirectness_data : same format — MUST NOT be None (v2 audit #3)
    imprecision_data : same format
    publication_bias_data : same format
    study_design : "rct" or "observational"

    Returns
    -------
    dict with grade, downgrade details, or error if indirectness missing.
    """
    # v2 audit #3: indirectness=None is an error, not a default
    if indirectness_data is None:
        return {
            "error": "indirectness not assessed — GRADE cannot default to no concern",
            "grade": None,
            "certainty": None,
        }

    all_data = {
        "risk_of_bias": rob_data,
        "inconsistency": inconsistency_data,
        "indirectness": indirectness_data,
        "imprecision": imprecision_data,
        "publication_bias": publication_bias_data,
    }

    # Check for other missing domains
    missing = [d for d, v in all_data.items() if v is None]
    if missing:
        return {
            "error": f"Missing domain assessments: {missing}",
            "grade": None,
            "certainty": None,
        }

    certainty = STARTING_CERTAINTY.get(study_design, 2)
    downgrades = {}

    for domain, data in all_data.items():
        level = data.get("level", "no_concern")
        reason = data.get("reason", "")

        if level == "serious":
            downgrade = 1
        elif level == "very_serious":
            downgrade = 2
        else:
            downgrade = 0

        if downgrade > 0:
            downgrades[domain] = {
                "level": level,
                "downgrade": downgrade,
                "reason": reason,
            }
            certainty -= downgrade

    certainty = max(1, certainty)
    grade_label = CERTAINTY_LABELS[certainty]

    return {
        "grade": grade_label,
        "certainty": certainty,
        "starting_certainty": STARTING_CERTAINTY.get(study_design, 2),
        "study_design": study_design,
        "total_downgrade": sum(d["downgrade"] for d in downgrades.values()),
        "downgrades": downgrades,
        "domains_assessed": DOMAIN_NAMES,
        "threshold_sources": {
            # v2 audit #11: document threshold provenance
            "imprecision": "OIS-based (Cochrane Handbook 15.5.4)",
            "inconsistency": "I² thresholds (Cochrane Handbook 10.10.2): "
                             "0-40% low, 30-60% moderate, 50-90% substantial, 75-100% considerable",
            "risk_of_bias": "≥50% high risk → very serious (Schünemann 2019)",
            "indirectness": "Population/intervention/comparator/outcome directness "
                            "(Cochrane Handbook 14.2.3)",
            "publication_bias": "Egger's test p<0.10 or funnel asymmetry "
                                "(Cochrane Handbook 13.3.5.3)",
        },
    }
