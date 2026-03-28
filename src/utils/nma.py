"""
Network Meta-Analysis (NMA) Module — LUMEN v2
===============================================
Orchestrates NMA via R's netmeta package (frequentist, graph-theoretic).
Uses the same subprocess pattern as statistics.py R bridge.

Key capabilities:
- Network construction and connectivity validation
- Frequentist NMA (netmeta with REML)
- Treatment ranking (P-score)
- Consistency assessment (design decomposition, node-splitting)
- League table, forest plot, network graph, funnel plot
- Leave-one-out sensitivity analysis
- CINeMA template generation
"""

import json
import csv
import io
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NMA_R_TEMPLATE = Path(__file__).parent / "nma_r_template.R"


# ======================================================================
# Availability check
# ======================================================================

def is_netmeta_available() -> bool:
    """Check if Rscript is available and netmeta is installed."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", "library(netmeta); library(jsonlite); cat('OK')"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and "OK" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ======================================================================
# Data preparation
# ======================================================================

def prepare_nma_data(extractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert LUMEN extraction results to NMA contrast-level data.

    Each extraction should have:
    - study_id / study_label
    - arms: list of {treatment, n, mean, sd} or {treatment, events, total}
    - OR legacy 2-arm: {intervention, control, effect_size, se, ...}

    Returns list of contrasts: [{studlab, treat1, treat2, TE, seTE}, ...]
    """
    contrasts = []

    for ext in extractions:
        study_id = ext.get("study_id") or ext.get("study_label", "Unknown")

        # Case 1: Multi-arm data with 'arms' field
        if "arms" in ext and len(ext["arms"]) >= 2:
            arms = ext["arms"]
            contrasts.extend(_arms_to_contrasts(study_id, arms, ext))
            continue

        # Case 2: NMA-ready contrast data
        if all(k in ext for k in ("treat1", "treat2", "TE", "seTE")):
            contrasts.append({
                "studlab": study_id,
                "treat1": ext["treat1"],
                "treat2": ext["treat2"],
                "TE": float(ext["TE"]),
                "seTE": float(ext["seTE"]),
            })
            continue

        # Case 3: Legacy 2-arm pairwise data
        if "effect_size" in ext and "se" in ext:
            treat1 = ext.get("intervention_name", ext.get("intervention", "Intervention"))
            treat2 = ext.get("control_name", ext.get("control", "Control"))
            contrasts.append({
                "studlab": study_id,
                "treat1": str(treat1),
                "treat2": str(treat2),
                "TE": float(ext["effect_size"]),
                "seTE": float(ext["se"]),
            })
            continue

        logger.warning(f"Skipping study {study_id}: no extractable NMA data")

    return contrasts


def _arms_to_contrasts(study_id: str, arms: List[Dict], ext: Dict) -> List[Dict]:
    """
    Convert multi-arm data to all pairwise contrasts within a study.

    For k arms, generates k*(k-1)/2 contrasts per matching outcome.
    Supports both flat arm data (mean/sd/n at root) and nested outcomes[].
    """
    import math
    contrasts = []
    k = len(arms)

    # Treatment name: try treatment_name (NMA schema), then treatment (legacy)
    def _tname(arm, idx):
        return arm.get("treatment_name") or arm.get("treatment") or f"Arm_{idx}"

    # Check if arms have nested outcomes (NMA extraction schema)
    has_nested = any("outcomes" in a and a["outcomes"] for a in arms)

    if has_nested:
        # Generate contrasts per outcome measure across arms
        contrasts.extend(_nested_arms_to_contrasts(study_id, arms, _tname))
    else:
        # Legacy flat format: mean/sd/n or events/total at arm root
        for i in range(k):
            for j in range(i + 1, k):
                a1 = arms[i]
                a2 = arms[j]
                t1 = _tname(a1, i)
                t2 = _tname(a2, j)

                if "TE" in a1 and "TE" in a2:
                    te = float(a2["TE"]) - float(a1["TE"])
                    se = math.sqrt(float(a1.get("seTE", 0))**2 + float(a2.get("seTE", 0))**2)
                elif all(k_ in a1 for k_ in ("mean", "sd", "n")) and \
                     all(k_ in a2 for k_ in ("mean", "sd", "n")):
                    te, se = _compute_smd(
                        float(a1["mean"]), float(a1["sd"]), int(a1["n"]),
                        float(a2["mean"]), float(a2["sd"]), int(a2["n"]),
                    )
                elif all(k_ in a1 for k_ in ("events", "total")) and \
                     all(k_ in a2 for k_ in ("events", "total")):
                    te, se = _compute_log_rr(
                        int(a1["events"]), int(a1["total"]),
                        int(a2["events"]), int(a2["total"]),
                    )
                else:
                    logger.warning(f"Cannot compute effect for {study_id}: {t1} vs {t2}")
                    continue

                if te is not None and se is not None and se > 0:
                    contrasts.append({
                        "studlab": study_id,
                        "treat1": t1,
                        "treat2": t2,
                        "TE": round(te, 6),
                        "seTE": round(se, 6),
                    })

    return contrasts


def _nested_arms_to_contrasts(study_id: str, arms: List[Dict], tname_fn) -> List[Dict]:
    """Generate contrasts from arms with nested outcomes[] arrays."""
    import math
    contrasts = []
    k = len(arms)

    # Build per-outcome lookup: {measure: {arm_idx: outcome_dict}}
    outcome_map: Dict[str, Dict[int, Dict]] = {}
    for idx, arm in enumerate(arms):
        for o in arm.get("outcomes", []):
            measure = o.get("measure", "")
            if not measure:
                continue
            outcome_map.setdefault(measure, {})[idx] = o

    # For each outcome, generate pairwise contrasts
    for measure, arm_outcomes in outcome_map.items():
        for i in range(k):
            for j in range(i + 1, k):
                if i not in arm_outcomes or j not in arm_outcomes:
                    continue
                o1 = arm_outcomes[i]
                o2 = arm_outcomes[j]
                t1 = tname_fn(arms[i], i)
                t2 = tname_fn(arms[j], j)
                n1 = arms[i].get("n") or o1.get("total")
                n2 = arms[j].get("n") or o2.get("total")

                te, se = None, None
                corrections = []

                # Binary: events/total → log-RR
                em = None
                if o1.get("events") is not None and o2.get("events") is not None:
                    tot1 = o1.get("total") or n1
                    tot2 = o2.get("total") or n2
                    if tot1 and tot2:
                        te, se = _compute_log_rr(
                            int(o1["events"]), int(tot1),
                            int(o2["events"]), int(tot2),
                        )
                        em = "RR"
                # Continuous: mean/sd/n → SMD (with SE/SD confusion check)
                elif o1.get("mean") is not None and o2.get("mean") is not None:
                    sd1_raw = o1.get("sd")
                    sd2_raw = o2.get("sd")
                    if sd1_raw and sd2_raw and n1 and n2:
                        m1, m2 = float(o1["mean"]), float(o2["mean"])
                        sd1_raw, sd2_raw = float(sd1_raw), float(sd2_raw)
                        # Check SE/SD confusion per arm
                        sd1, corr1, msg1 = _check_se_sd_confusion(
                            m1, sd1_raw, int(n1), measure)
                        sd2, corr2, msg2 = _check_se_sd_confusion(
                            m2, sd2_raw, int(n2), measure)
                        corrections = []
                        if corr1:
                            corrections.append(msg1)
                        if corr2:
                            corrections.append(msg2)
                        te, se = _compute_smd(m1, sd1, int(n1), m2, sd2, int(n2))
                        # Post-hoc sanity check
                        if te is not None:
                            te, se, extra = _check_smd_sanity(
                                te, se, study_id, t1, t2, measure,
                                sd1, sd2, int(n1), int(n2), m1, m2,
                                corrections)
                            corrections.extend(extra)
                        em = "SMD"

                if te is not None and se is not None and se > 0:
                    entry = {
                        "studlab": study_id,
                        "treat1": t1,
                        "treat2": t2,
                        "TE": round(te, 6),
                        "seTE": round(se, 6),
                        "outcome": measure,
                        "effect_measure": em,
                    }
                    if corrections:
                        entry["corrections"] = corrections
                    contrasts.append(entry)

    return contrasts


def _check_se_sd_confusion(mean_val, reported_sd, n, measure_name=""):
    """
    Detect likely SE/SD confusion in reported data.

    Heuristic: If reported "SD" is suspiciously small relative to n,
    it's likely SE. SD = SE * sqrt(n).

    Returns: (corrected_sd, was_corrected, explanation)
    """
    import math

    if reported_sd is None or reported_sd <= 0 or n is None or n <= 1:
        return reported_sd, False, ""

    sqrt_n = math.sqrt(n)
    corrected_sd = reported_sd * sqrt_n

    # --- Heuristic 1: domain-specific small SD for weight/BMI change ---
    measure_lower = measure_name.lower() if measure_name else ""
    is_weight_bmi = any(kw in measure_lower for kw in ("weight", "bmi", "body mass"))
    if is_weight_bmi and n > 20 and reported_sd < 2.0:
        explanation = (
            f"SE/SD correction applied for '{measure_name}': "
            f"reported SD={reported_sd} too small for n={n} "
            f"(weight/BMI threshold: SD<2.0 with n>20). "
            f"Corrected SD = {reported_sd} * sqrt({n}) = {corrected_sd:.2f}"
        )
        logger.warning(explanation)
        return corrected_sd, True, explanation

    # --- Heuristic 2: SD suspiciously small relative to mean ---
    # Only apply to change scores (negative mean or small absolute mean suggests change)
    abs_mean = abs(mean_val) if mean_val else 0
    is_likely_change = (mean_val is not None and (mean_val < 0 or abs_mean < 20))
    if is_likely_change and abs_mean > 1.0 and n > 20 and reported_sd < abs_mean * 0.15:
        explanation = (
            f"SE/SD correction applied for '{measure_name}': "
            f"reported SD={reported_sd} < 15% of |mean|={abs_mean:.2f} "
            f"with n={n}. Corrected SD = {reported_sd} * sqrt({n}) = {corrected_sd:.2f}"
        )
        logger.warning(explanation)
        return corrected_sd, True, explanation

    # --- Heuristic 3: ratio test — reported value closer to expected SE ---
    # If we assume real SD ~ corrected_sd, then expected SE = corrected_sd / sqrt(n)
    # = reported_sd (tautological). Instead, use the *other arm* logic at caller level.
    # Here we just flag if the coefficient of variation is implausibly low.
    if abs_mean > 0 and n > 20:
        cv = reported_sd / abs_mean
        expected_cv_if_se = cv * sqrt_n
        # If CV < 0.05 and correcting would give a plausible CV (0.2-2.0), likely SE
        if cv < 0.05 and 0.1 < expected_cv_if_se < 3.0:
            explanation = (
                f"SE/SD correction applied for '{measure_name}': "
                f"CV={cv:.3f} implausibly low for n={n}. "
                f"Corrected SD = {reported_sd} * sqrt({n}) = {corrected_sd:.2f} "
                f"(corrected CV={expected_cv_if_se:.3f})"
            )
            logger.warning(explanation)
            return corrected_sd, True, explanation

    return reported_sd, False, ""


def _check_smd_sanity(te, se, study_id, t1, t2, measure, sd1_orig, sd2_orig, n1, n2,
                       m1, m2, corrections):
    """
    Post-hoc sanity check: if |SMD| > 3.0 and neither arm was already corrected,
    attempt SE→SD correction on the arm with the smaller SD and recompute.

    Returns: (te, se, extra_corrections)
    """
    import math

    if te is None or abs(te) <= 3.0:
        return te, se, []

    extra_corrections = []
    new_sd1, new_sd2 = sd1_orig, sd2_orig
    corrected_any = False

    # Try correcting the arm with smaller SD
    for label, sd_val, n_val, mean_val in [("arm1", sd1_orig, n1, m1), ("arm2", sd2_orig, n2, m2)]:
        if n_val > 10 and sd_val > 0:
            candidate = sd_val * math.sqrt(n_val)
            # Check if corrected value is more plausible
            other_sd = sd2_orig if label == "arm1" else sd1_orig
            if other_sd > 0 and sd_val < other_sd * 0.3:
                msg = (
                    f"Post-hoc SMD sanity correction for {study_id} "
                    f"({t1} vs {t2}, '{measure}'): |SMD|={abs(te):.2f} > 3.0. "
                    f"{label} SD={sd_val} -> {candidate:.2f} (SE*sqrt(n))"
                )
                logger.warning(msg)
                extra_corrections.append(msg)
                if label == "arm1":
                    new_sd1 = candidate
                else:
                    new_sd2 = candidate
                corrected_any = True

    if corrected_any:
        te2, se2 = _compute_smd_raw(m1, new_sd1, n1, m2, new_sd2, n2)
        if te2 is not None:
            return te2, se2, extra_corrections

    return te, se, extra_corrections


def _compute_smd_raw(m1, sd1, n1, m2, sd2, n2):
    """Compute Hedges' g (bias-corrected SMD) — raw computation, no SE/SD checks."""
    import math
    sd_pooled = math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    if sd_pooled == 0:
        return None, None
    d = (m2 - m1) / sd_pooled
    df = n1 + n2 - 2
    j = 1 - 3 / (4 * df - 1)  # Hedges' correction
    g = d * j
    se = math.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2 - 2))) * j
    return g, se


def _compute_smd(m1, sd1, n1, m2, sd2, n2):
    """Compute Hedges' g (bias-corrected SMD)."""
    return _compute_smd_raw(m1, sd1, n1, m2, sd2, n2)


def _compute_log_rr(e1, n1, e2, n2):
    """Compute log risk ratio with 0.5 continuity correction."""
    import math
    # Apply continuity correction if any zero cell
    if e1 == 0 or e2 == 0 or e1 == n1 or e2 == n2:
        e1, n1 = e1 + 0.5, n1 + 1
        e2, n2 = e2 + 0.5, n2 + 1
    p1 = e1 / n1
    p2 = e2 / n2
    if p1 <= 0 or p2 <= 0:
        return None, None
    log_rr = math.log(p2 / p1)
    se = math.sqrt(1/e1 - 1/n1 + 1/e2 - 1/n2)
    return log_rr, se


# ======================================================================
# Validation
# ======================================================================

def validate_network(contrasts: List[Dict]) -> Dict[str, Any]:
    """
    Validate the NMA network before running analysis.

    Returns dict with: valid, n_studies, n_treatments, treatments,
    n_contrasts, warnings, errors.
    """
    if not contrasts:
        return {"valid": False, "errors": ["No contrast data provided"]}

    studies = set()
    treatments = set()
    for c in contrasts:
        studies.add(c["studlab"])
        treatments.add(c["treat1"])
        treatments.add(c["treat2"])

    result = {
        "n_studies": len(studies),
        "n_treatments": len(treatments),
        "treatments": sorted(treatments),
        "n_contrasts": len(contrasts),
        "warnings": [],
        "errors": [],
    }

    # Check minimum requirements
    if len(treatments) < 3:
        result["errors"].append(
            f"NMA requires >= 3 treatments, found {len(treatments)}. "
            "Use standard pairwise meta-analysis instead."
        )

    if len(studies) < 3:
        result["warnings"].append(f"Only {len(studies)} studies — results may be unreliable.")

    # Check connectivity via simple graph traversal
    adj = {t: set() for t in treatments}
    for c in contrasts:
        adj[c["treat1"]].add(c["treat2"])
        adj[c["treat2"]].add(c["treat1"])

    visited = set()
    queue = [next(iter(treatments))]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj[node] - visited)

    if visited != treatments:
        disconnected = treatments - visited
        result["errors"].append(
            f"Disconnected network: treatments {disconnected} not reachable. "
            "NMA requires a single connected component."
        )

    # Check for missing data
    n_missing = sum(1 for c in contrasts if c.get("TE") is None or c.get("seTE") is None)
    if n_missing > 0:
        result["errors"].append(f"{n_missing} contrasts have missing TE or seTE.")

    result["valid"] = len(result["errors"]) == 0
    return result


# ======================================================================
# Treatment mapping
# ======================================================================

def build_treatment_mapping(contrasts: List[Dict],
                            canonical_map: Optional[Dict[str, str]] = None) -> List[Dict]:
    """
    Apply canonical treatment name mapping to contrasts.

    canonical_map: {"ice vest" -> "Ice Vest", "cold water" -> "CWI", ...}
    """
    if not canonical_map:
        return contrasts

    mapped = []
    for c in contrasts:
        mc = dict(c)
        mc["treat1"] = canonical_map.get(c["treat1"], c["treat1"])
        mc["treat2"] = canonical_map.get(c["treat2"], c["treat2"])
        mapped.append(mc)

    return mapped


# ======================================================================
# Main NMA execution
# ======================================================================

def run_nma(contrasts: List[Dict[str, Any]],
            output_dir: str,
            *,
            effect_measure: str = "SMD",
            method_tau: str = "REML",
            reference_group: Optional[str] = None,
            small_values: str = "undesirable",
            run_consistency: bool = True,
            run_node_splitting: bool = True,
            run_net_heat: bool = True,
            run_loo: bool = True,
            run_network_graph: bool = True,
            run_forest: bool = True,
            run_league: bool = True,
            run_league_heatmap: bool = True,
            run_funnel: bool = True,
            run_ranking: bool = True,
            timeout: int = 300) -> Dict[str, Any]:
    """
    Run full NMA analysis via R netmeta.

    Args:
        contrasts: List of {studlab, treat1, treat2, TE, seTE}
        output_dir: Directory for figures, tables, and results JSON
        effect_measure: "SMD", "MD", "RR", "OR", "HR"
        method_tau: "REML", "DL", "ML", "EB", "PM", "SJ"
        reference_group: Control treatment name (or None for auto)
        small_values: "undesirable" or "desirable" for ranking
        timeout: R subprocess timeout in seconds

    Returns:
        Dict with NMA results (parsed from R JSON output)

    Raises:
        RuntimeError: If R execution fails or netmeta not available
    """
    # Validate
    validation = validate_network(contrasts)
    if not validation["valid"]:
        raise ValueError(f"Invalid network: {'; '.join(validation['errors'])}")

    if not is_netmeta_available():
        raise RuntimeError(
            "R netmeta not available. Install with: "
            "Rscript -e 'install.packages(c(\"netmeta\", \"jsonlite\"))'"
        )

    # Convert contrasts to CSV string (strip extra fields like 'outcome')
    csv_fields = ["studlab", "treat1", "treat2", "TE", "seTE"]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(contrasts)
    data_csv = csv_buffer.getvalue()

    # Read template and inject values
    template = NMA_R_TEMPLATE.read_text(encoding="utf-8")

    ref_group_r = f'"{reference_group}"' if reference_group else "NULL"

    # Use absolute path so R file.path() doesn't create nested paths
    abs_output = str(Path(output_dir).resolve())
    r_script = template.replace("{{output_dir}}", abs_output.replace("\\", "/"))
    r_script = r_script.replace("{{effect_measure}}", effect_measure)
    r_script = r_script.replace("{{method_tau}}", method_tau)
    r_script = r_script.replace("{{reference_group}}", ref_group_r)
    r_script = r_script.replace("{{small_values}}", small_values)
    r_script = r_script.replace("{{data_csv}}", data_csv.replace('"', '\\"').replace('\n', '\\n'))

    # Boolean replacements
    bool_map = {
        "{{run_consistency}}": run_consistency,
        "{{run_node_splitting}}": run_node_splitting,
        "{{run_net_heat}}": run_net_heat,
        "{{run_loo}}": run_loo,
        "{{run_network_graph}}": run_network_graph,
        "{{run_forest}}": run_forest,
        "{{run_league}}": run_league,
        "{{run_league_heatmap}}": run_league_heatmap,
        "{{run_funnel}}": run_funnel,
        "{{run_ranking}}": run_ranking,
    }
    for placeholder, val in bool_map.items():
        r_script = r_script.replace(placeholder, "TRUE" if val else "FALSE")

    # Write and execute R script
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".R", prefix="lumen_nma_",
        dir=str(output_path), delete=False, encoding="utf-8"
    ) as f:
        f.write(r_script)
        r_script_path = f.name

    logger.info(f"Running NMA R script: {r_script_path}")

    try:
        proc = subprocess.run(
            ["Rscript", r_script_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(output_path),
        )

        if proc.returncode != 0:
            logger.error(f"R NMA stderr: {proc.stderr}")
            raise RuntimeError(f"R NMA failed (rc={proc.returncode}): {proc.stderr[:500]}")

        # Parse results
        results_json = output_path / "nma_results.json"
        if not results_json.exists():
            raise RuntimeError(f"R NMA produced no output. stdout: {proc.stdout[:300]}")

        with open(results_json, encoding="utf-8") as f:
            results = json.load(f)

        results["engine"] = "netmeta"
        results["r_stdout"] = proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout
        results["validation"] = validation

        logger.info(
            f"NMA complete: {results.get('n_treatments', '?')} treatments, "
            f"{results.get('n_studies', '?')} studies, "
            f"tau2={results.get('tau2', '?')}, I2={results.get('I2', '?')}%"
        )

        return results

    finally:
        # Clean up temp R script
        try:
            Path(r_script_path).unlink()
        except OSError:
            pass


# ======================================================================
# Settings integration
# ======================================================================

def load_nma_settings() -> Dict[str, Any]:
    """Load NMA settings from config/v2_settings.yaml."""
    import yaml
    settings_path = Path(__file__).parent.parent.parent / "config" / "v2_settings.yaml"
    if not settings_path.exists():
        return {}

    with open(settings_path, encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    return settings.get("nma", {})


def run_nma_from_settings(contrasts: List[Dict], output_dir: str) -> Dict[str, Any]:
    """Run NMA with settings loaded from v2_settings.yaml."""
    cfg = load_nma_settings()

    if not cfg.get("enabled", False):
        raise ValueError("NMA is disabled in v2_settings.yaml. Set nma.enabled: true")

    plots_cfg = cfg.get("plots", {})
    consistency_cfg = cfg.get("consistency_checks", {})
    sensitivity_cfg = cfg.get("sensitivity_analyses", {})

    return run_nma(
        contrasts,
        output_dir,
        effect_measure=cfg.get("effect_measure", "SMD"),
        method_tau=cfg.get("method_tau", "REML"),
        reference_group=cfg.get("reference_group"),
        small_values=cfg.get("small_values", "undesirable"),
        run_consistency=consistency_cfg.get("design_decomposition", True),
        run_node_splitting=consistency_cfg.get("node_splitting", True),
        run_net_heat=consistency_cfg.get("net_heat_plot", True),
        run_loo=sensitivity_cfg.get("leave_one_out", True),
        run_network_graph=plots_cfg.get("network_graph", True),
        run_forest=plots_cfg.get("forest_plot", True),
        run_league=plots_cfg.get("league_table", True),
        run_league_heatmap=plots_cfg.get("league_heatmap", True),
        run_funnel=plots_cfg.get("funnel_plot", True),
        run_ranking=plots_cfg.get("ranking_plot", True),
    )
