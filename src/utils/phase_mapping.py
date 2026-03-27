"""
Phase Mapping — LUMEN v2
=========================
Centralized role-to-phase mapping used across cost tracking,
readiness scoring, and transparency reporting.
"""

# Role name (from models.yaml) -> Phase display name
ROLE_PHASE_MAP = {
    "strategist": "Phase 1",
    "rescue_screener": "Phase 3.0",
    "screener_1": "Phase 3",
    "screener_2": "Phase 3",
    "arbiter": "Phase 3",
    "extractor": "Phase 4",
    "extractor_tiebreaker": "Phase 4",
    "statistician": "Phase 5",
    "writer": "Phase 6",
    "citation_guardian": "Phase 6",
}

# Extended display names (for reports)
ROLE_PHASE_DISPLAY = {
    "strategist": "Phase 1 (Strategy)",
    "rescue_screener": "Phase 3.0 (Prescreen)",
    "screener_1": "Phase 3 (Screening)",
    "screener_2": "Phase 3 (Screening)",
    "arbiter": "Phase 3 (Screening)",
    "extractor": "Phase 4 (Extraction)",
    "extractor_tiebreaker": "Phase 4 (Extraction)",
    "statistician": "Phase 5 (Statistics)",
    "writer": "Phase 6 (Manuscript)",
    "citation_guardian": "Phase 6 (Manuscript)",
}


def role_to_phase(role: str, display: bool = False) -> str:
    """Map agent role name to pipeline phase.

    Args:
        role: Agent role name from models.yaml
        display: If True, return descriptive display name
    """
    if display:
        return ROLE_PHASE_DISPLAY.get(role, f"Other ({role})")
    return ROLE_PHASE_MAP.get(role, f"Other ({role})")
