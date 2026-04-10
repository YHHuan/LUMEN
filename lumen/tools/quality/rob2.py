"""
RoB-2 (Risk of Bias 2) assessment with 5-domain enforcement.

Fixes v2 audit #10: all 5 domains must be assessed — no defaults.
"""

from __future__ import annotations

REQUIRED_DOMAINS = [
    "randomization_process",
    "deviations_from_intervention",
    "missing_outcome_data",
    "measurement_of_outcome",
    "selection_of_reported_result",
]

VALID_JUDGEMENTS = {"low", "some_concerns", "high"}


def assess_rob2(domains: dict[str, str]) -> dict:
    """
    Assess overall RoB-2 from individual domain judgements.

    Parameters
    ----------
    domains : {domain_name: "low" | "some_concerns" | "high"}

    Returns
    -------
    dict with overall judgement, domain details, and any issues.

    Raises
    ------
    ValueError if any required domain is missing or has invalid judgement.
    """
    # Check all required domains present
    missing = [d for d in REQUIRED_DOMAINS if d not in domains]
    if missing:
        raise ValueError(
            f"RoB-2 requires all 5 domains. Missing: {missing}"
        )

    # Validate judgements
    invalid = {d: v for d, v in domains.items()
               if d in REQUIRED_DOMAINS and v not in VALID_JUDGEMENTS}
    if invalid:
        raise ValueError(
            f"Invalid judgements (must be low/some_concerns/high): {invalid}"
        )

    # Overall algorithm (Cochrane RoB-2 tool guidance):
    # - "high" if any domain is "high"
    # - "some_concerns" if any domain is "some_concerns" (and none "high")
    # - "low" only if all domains are "low"
    judgements = [domains[d] for d in REQUIRED_DOMAINS]

    if "high" in judgements:
        overall = "high"
    elif "some_concerns" in judgements:
        overall = "some_concerns"
    else:
        overall = "low"

    return {
        "overall": overall,
        "domains": {d: domains[d] for d in REQUIRED_DOMAINS},
        "missing_domains": [],
        "n_high": judgements.count("high"),
        "n_some_concerns": judgements.count("some_concerns"),
        "n_low": judgements.count("low"),
    }


def summarize_rob2_across_studies(assessments: list[dict]) -> dict:
    """
    Summarize RoB-2 across multiple studies for GRADE input.

    Returns proportion of studies at each risk level per domain
    and overall.
    """
    k = len(assessments)
    if k == 0:
        return {"k": 0, "summary": {}}

    domain_counts = {d: {"low": 0, "some_concerns": 0, "high": 0}
                     for d in REQUIRED_DOMAINS}
    overall_counts = {"low": 0, "some_concerns": 0, "high": 0}

    for a in assessments:
        overall_counts[a["overall"]] += 1
        for d in REQUIRED_DOMAINS:
            domain_counts[d][a["domains"][d]] += 1

    return {
        "k": k,
        "overall": {level: count / k for level, count in overall_counts.items()},
        "by_domain": {
            d: {level: count / k for level, count in counts.items()}
            for d, counts in domain_counts.items()
        },
        "proportion_high_overall": overall_counts["high"] / k,
    }
