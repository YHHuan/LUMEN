"""
Data Normalizer & Sanity Checker — v5
=======================================
v5 CRITICAL FIX: prepare_meta_data now uses fuzzy matching to find outcomes.

Problem: LLM generates keys like "global_cognition_mmse", "primary_cdr_sb",
  but pico.yaml says "Global cognition" with measure "MMSE".
Solution: Smart matching by measure name (MMSE, MoCA, etc.) across ALL
  outcome keys, regardless of what the LLM named them.
"""

import re
import numpy as np
import logging
from typing import List, Optional
from src.utils.effect_sizes import (
    auto_compute_effect, correct_direction, get_scale_direction,
    se_from_ci, se_from_ci_log, _safe_float
)

logger = logging.getLogger(__name__)


# ======================================================================
# Data Normalizer
# ======================================================================

class DataNormalizer:
    
    @staticmethod
    def parse_mean_sd(text: str) -> Optional[dict]:
        if text is None:
            return None
        text = str(text).strip()
        patterns = [
            r'([\d.]+)\s*\((\s*[\d.]+)\s*\)',
            r'([\d.]+)\s*±\s*([\d.]+)',
            r'([\d.]+)\s*\[(\s*[\d.]+)\s*\]',
            r'([\d.]+)\s*\+/?-\s*([\d.]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return {"mean": float(match.group(1)), "sd": float(match.group(2))}
        try:
            return {"mean": float(text), "sd": None}
        except ValueError:
            return None
    
    @staticmethod
    def parse_ci(text: str) -> Optional[dict]:
        if text is None:
            return None
        text = str(text).strip()
        patterns = [
            r'([\d.-]+)\s*[-–—to]+\s*([\d.-]+)',
            r'\(?\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\)?',
            r'\[\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\]',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return {"lower": float(match.group(1)), "upper": float(match.group(2))}
        return None
    
    @staticmethod
    def normalize_sex(data: dict) -> dict:
        result = dict(data)
        pop = result.get("population", {})
        if pop.get("sex_male_percent") is not None:
            return result
        if pop.get("sex_female_percent") is not None:
            pop["sex_male_percent"] = round(100 - pop["sex_female_percent"], 1)
        elif pop.get("sex_male_n") is not None and pop.get("n_total") is not None:
            pop["sex_male_percent"] = round(pop["sex_male_n"] / pop["n_total"] * 100, 1)
        elif pop.get("sex_female_n") is not None and pop.get("n_total") is not None:
            pop["sex_male_percent"] = round((pop["n_total"] - pop["sex_female_n"]) / pop["n_total"] * 100, 1)
        result["population"] = pop
        return result
    
    @staticmethod
    def detect_se_vs_sd_confusion(mean: float, reported_spread: float, n: int) -> str:
        if n <= 1 or reported_spread <= 0:
            return "unknown"
        cv = reported_spread / abs(mean) if mean != 0 else float('inf')
        if cv < 0.02 and n > 30:
            return "likely_se_reported_as_sd"
        return "likely_sd"


# ======================================================================
# Sanity Checker
# ======================================================================

class SanityChecker:
    
    @staticmethod
    def check_study(data: dict) -> dict:
        warnings = []
        errors = []
        auto_fixes = []
        sid = data.get("study_id", "unknown")
        
        chars = data.get("characteristics", {})
        pop = data.get("population", {})
        intervention = data.get("intervention", {})
        comparison = data.get("comparison", {})
        outcomes = data.get("outcomes", {})
        
        n_total = chars.get("n_total")
        n_int = intervention.get("n") or intervention.get("n_randomized")
        n_ctrl = comparison.get("n") or comparison.get("n_randomized")
        
        try:
            if all(v is not None for v in [n_total, n_int, n_ctrl]):
                n_total, n_int, n_ctrl = int(n_total), int(n_int), int(n_ctrl)
                if abs(n_int + n_ctrl - n_total) > 5:
                    warnings.append(f"N mismatch: {n_int}+{n_ctrl}={n_int+n_ctrl} ≠ {n_total}")
        except (ValueError, TypeError):
            warnings.append("N values not numeric")
        
        age = _safe_float(pop.get("age_mean"))
        if age is not None:
            if age < 18:
                errors.append(f"Age {age} < 18")
            elif age > 100:
                warnings.append(f"Age {age} > 100")
        
        for outcome_name, outcome_data in outcomes.items():
            prefix = f"'{outcome_name}'"
            for group in ["intervention", "control"]:
                for suffix in ["_sd", "_sd_post", "_sd_change"]:
                    sd_val = _safe_float(outcome_data.get(f"{group}{suffix}"))
                    if sd_val is not None and sd_val <= 0:
                        errors.append(f"{prefix}: {group} SD ({sd_val}) ≤ 0")
            
            p = outcome_data.get("p_value")
            if p is not None:
                pf = _safe_float(p)
                if pf is not None and (pf < 0 or pf > 1):
                    errors.append(f"{prefix}: p-value ({pf}) out of [0,1]")
        
        return {
            "study_id": sid, "passed": len(errors) == 0,
            "warnings": warnings, "errors": errors, "auto_fixes": auto_fixes,
        }
    
    @staticmethod
    def check_all(studies: List[dict]) -> dict:
        results = []
        clean = warn = err = 0
        for s in studies:
            c = SanityChecker.check_study(s)
            results.append(c)
            if c["passed"] and not c["warnings"]:
                clean += 1
            elif c["passed"]:
                warn += 1
            else:
                err += 1
        return {"total": len(studies), "clean": clean, "with_warnings": warn,
                "with_errors": err, "details": results}


# ======================================================================
# Provenance Tracker
# ======================================================================

class ProvenanceTracker:
    def __init__(self):
        self.records = []
    
    def add(self, study_id, field, value, source_page=None, source_section=None,
            source_text=None, confidence="medium"):
        self.records.append({
            "study_id": study_id, "field": field, "value": value,
            "source_page": source_page, "source_section": source_section,
            "source_text": source_text[:200] if source_text else None,
            "confidence": confidence,
        })
    
    def get_for_study(self, study_id):
        return [r for r in self.records if r["study_id"] == study_id]
    
    def to_dict(self):
        return self.records


# ======================================================================
# Smart Outcome Matching (v5 CRITICAL FIX)
# ======================================================================

def _find_outcome_data(outcomes: dict, measure_name: str, outcome_category: str) -> tuple:
    """
    Smart fuzzy matching for LLM-generated outcome keys.
    
    Problem: pico.yaml says measure="MMSE", outcome="Global cognition"
    LLM might produce: {"global_cognition_mmse": {"measure": "MMSE", ...}}
    Or: {"mmse_post_treatment": {"measure": "MMSE", ...}}
    Or: {"MMSE": {"measure": "MMSE", ...}}
    
    Strategy (in priority order):
    1. Check measure field inside each outcome → exact match on measure name
    2. Check if measure name appears in the outcome key itself
    3. Check if outcome category appears in the outcome key
    
    Returns: (outcome_data_dict, measure_name_used) or (None, None)
    """
    if not outcomes:
        return None, None
    
    measure_lower = measure_name.lower().strip()
    category_lower = outcome_category.lower().strip()
    
    # Pass 1: Match by "measure" field inside outcome data
    for key, odata in outcomes.items():
        m = (odata.get("measure") or "").lower().strip()
        if measure_lower in m or m in measure_lower:
            return odata, odata.get("measure", measure_name)
    
    # Pass 2: Match by outcome key name containing measure
    for key, odata in outcomes.items():
        key_lower = key.lower().replace("_", " ").replace("-", " ")
        if measure_lower in key_lower:
            return odata, odata.get("measure", measure_name)
    
    # Pass 3: Match by outcome key containing category
    for key, odata in outcomes.items():
        key_lower = key.lower().replace("_", " ").replace("-", " ")
        cat_words = category_lower.replace("_", " ").split()
        if any(w in key_lower for w in cat_words if len(w) > 3):
            return odata, odata.get("measure", key)
    
    return None, None


def _find_any_computable_outcome(outcomes: dict) -> tuple:
    """
    Last resort: scan ALL outcomes and return the first one that has
    enough data to compute an effect size.
    """
    for key, odata in outcomes.items():
        # Check if it has mean/sd/n (post or change)
        for suffix in ["_post", "_change", ""]:
            has_data = all(
                odata.get(f"intervention_mean{suffix}") is not None and
                odata.get(f"intervention_sd{suffix}") is not None and
                odata.get("intervention_n") is not None and
                odata.get(f"control_mean{suffix}") is not None and
                odata.get(f"control_sd{suffix}") is not None and
                odata.get("control_n") is not None
                for _ in [1]  # just to use 'all' for readability
            )
            if has_data:
                return odata, odata.get("measure", key)
        
        # Check pre-computed effect
        if odata.get("smd") is not None or odata.get("mean_difference") is not None:
            return odata, odata.get("measure", key)
    
    return None, None


# ======================================================================
# Prepare Meta-Analysis Data (v5 — with smart matching)
# ======================================================================

def prepare_meta_data(extracted_studies: List[dict],
                      primary_outcome: str = "global_cognition",
                      preferred_measure: str = None,
                      effect_type: str = "SMD") -> List[dict]:
    """
    v5: Uses smart fuzzy matching to find outcomes regardless of LLM key naming.
    
    Args:
        extracted_studies: List of study dicts from Phase 4
        primary_outcome: Outcome category (e.g., "Global cognition", "Functional outcomes")
        preferred_measure: Specific measure name (e.g., "MMSE", "ADAS-Cog", "PHQ-9").
                          If None, matches any measure in the outcome category.
        effect_type: "SMD" or "MD"
    
    Matching priority:
    1. Match by measure name (if preferred_measure given) in outcome's "measure" field
    2. Match by measure name appearing in outcome key
    3. Match by outcome category appearing in outcome key
    4. Fallback: first outcome with computable data
    """
    meta_ready = []
    
    for study in extracted_studies:
        sid = study.get("study_id", "unknown")
        outcomes = study.get("outcomes", {})
        
        if not outcomes:
            logger.warning(f"[{sid}] No outcomes extracted — skipping")
            continue
        
        # Smart matching (by measure name and category)
        target_data, measure_used = _find_outcome_data(outcomes, preferred_measure, primary_outcome)
        
        if target_data is None:
            logger.debug(f"[{sid}] No '{preferred_measure}' outcome found (keys: {list(outcomes.keys())})")
            continue
        
        # Compute effect size (v5: handles _post/_change keys automatically)
        computed = auto_compute_effect(target_data, preferred_type=effect_type)
        
        if computed is None or computed.get("effect") is None:
            logger.warning(f"[{sid}] Effect computation failed for '{measure_used}'")
            continue
        
        # Direction correction
        direction = get_scale_direction(measure_used)
        if computed["effect_type"] in ("MD", "SMD"):
            computed = correct_direction(computed, direction)
        
        # Validate SE
        if computed.get("se") is not None and computed["se"] <= 0:
            logger.warning(f"[{sid}] Invalid SE: {computed['se']}")
            continue
        
        meta_ready.append({
            "study_id": sid,
            "citation": study.get("citation", sid),
            "measure": measure_used,
            "n_intervention": target_data.get("intervention_n"),
            "n_control": target_data.get("control_n"),
            **computed,
        })
    
    logger.info(f"Prepared {len(meta_ready)}/{len(extracted_studies)} studies for meta-analysis "
                f"(outcome={primary_outcome}, measure={preferred_measure})")
    return meta_ready
