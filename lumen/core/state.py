"""
LUMEN v3 State Schema.

Central TypedDict defining the entire pipeline state.
LangGraph nodes read from and write to this state.
"""
from __future__ import annotations

from typing import TypedDict


class LumenState(TypedDict, total=False):
    # === Phase 1 outputs (immutable after confirmation) ===
    pico: dict                          # structured PICO definition
    screening_criteria: dict            # inclusion/exclusion rules
    search_strategy: dict               # queries, databases, filters
    pico_completeness_score: int        # 0-100

    # === Phase 2 outputs ===
    raw_results: list[dict]             # search hits before dedup
    deduplicated_studies: list[dict]    # after dedup

    # === Phase 3 outputs ===
    prescreen_results: list[dict]       # keyword filter results
    screening_results: list[dict]       # [{study_id, screener1, screener2, confidence1, confidence2, arbiter?, final_decision}]
    fulltext_results: list[dict]        # [{study_id, decision, reason}]
    included_studies: list[dict]        # final included set

    # === Phase 4 outputs ===
    extractions: list[dict]             # [{study_id, round1_skeleton, round2_data, round3_checked, round4_spans}]

    # === Phase 4.5 outputs ===
    outcome_clusters: dict              # {canonical_name: [raw_names]}
    harmonized_data: list[dict]         # extraction data with canonical outcome names

    # === Phase 5 outputs ===
    analysis_plan: dict                 # {outcomes: [{name, measure, model, subgroups}]}
    statistics_results: dict            # {outcome: {pooled_effect, ci, i2, tau2, ...}}
    anomaly_flags: list[dict]           # [{type, description, severity, action_taken}]
    quality_assessments: dict           # {rob2: {...}, grade: {...}}

    # === Phase 6 outputs ===
    evidence_synthesis: dict            # {key_findings, evidence_table, narrative_skeleton}
    manuscript_sections: dict           # {methods: str, results: str, ...}
    fact_check_log: list[dict]          # [{claim, verdict, evidence_source}]

    # === Cross-cutting ===
    current_phase: str
    cost_tracker: dict                  # {phase: {agent: {calls: int, tokens: int, cost: float}}}
    human_decisions: list[dict]         # [{phase, question, answer, timestamp}]
    running_summary: str                # compressed context for long pipelines
