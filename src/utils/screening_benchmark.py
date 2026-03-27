"""
Screening Benchmark Framework — LUMEN v2
==========================================
Computes ROC curves, sensitivity/specificity at multiple thresholds,
and cross-arm comparison for Phase 3.1 screening ablation.

5 Arms:
  A. Single Gemini (screener_1)      — extracted from dual results
  B. Single GPT (screener_2)         — extracted from dual results
  C. Single Claude (arbiter config)  — requires dedicated run
  D. Dual + Arbiter (default)        — existing results
  E. ASReview (external baseline)    — loaded from CSV, binary mapped

Usage:
    from src.utils.screening_benchmark import ScreeningBenchmark
    bench = ScreeningBenchmark(ground_truth, screening_results)
    bench.compute_all()
    bench.plot_roc()
    bench.export_table()
"""

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# 5-point confidence → ordinal score
CONFIDENCE_SCORE = {
    "most_likely_include": 5,
    "likely_include": 4,
    "undecided": 3,
    "likely_exclude": 2,
    "most_likely_exclude": 1,
}

# Standard thresholds for ROC sweep
ROC_THRESHOLDS = [1.5, 2.5, 3.5, 4.5]

# Operating point used for primary comparison with otto-SR
DEFAULT_THRESHOLD = 3.5


@dataclass
class ArmResult:
    """Results for one benchmark arm."""
    arm_name: str
    model: str
    n_studies: int = 0
    # Per-threshold metrics
    roc_points: List[Dict] = field(default_factory=list)
    # At operating point (threshold=3.5)
    sensitivity: float = 0.0
    specificity: float = 0.0
    f1: float = 0.0
    ppv: float = 0.0
    npv: float = 0.0
    # WSS@95% and AUC
    wss_at_95: Optional[float] = None
    auc: float = 0.0
    # Cost
    cost_usd: float = 0.0
    cost_per_100: float = 0.0
    # Distribution
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    decision_distribution: Dict[str, int] = field(default_factory=dict)
    # Calibration (Table 7b)
    mean_score: Optional[float] = None
    score_relevance_correlation: Optional[float] = None
    calibration_slope: Optional[float] = None
    # Multi-agent specific (Table 7b, LUMEN only)
    inter_screener_kappa: Optional[float] = None
    arbitrator_trigger_rate: Optional[float] = None


def compute_binary_metrics(
    predicted: List[bool],
    actual: List[bool],
) -> Dict:
    """Compute sensitivity, specificity, F1, PPV, NPV from binary predictions."""
    tp = sum(1 for p, a in zip(predicted, actual) if p and a)
    fp = sum(1 for p, a in zip(predicted, actual) if p and not a)
    tn = sum(1 for p, a in zip(predicted, actual) if not p and not a)
    fn = sum(1 for p, a in zip(predicted, actual) if not p and a)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
    }


def compute_roc_curve(
    confidences: List[str],
    ground_truth: List[bool],
    thresholds: List[float] = None,
) -> List[Dict]:
    """
    Sweep thresholds on 5-point confidence scores to produce ROC curve points.

    Args:
        confidences: list of 5-point confidence labels
        ground_truth: list of bool (True = should be included)
        thresholds: ordinal thresholds to sweep

    Returns:
        List of dicts with threshold, sensitivity, specificity, fpr, etc.
    """
    if thresholds is None:
        thresholds = ROC_THRESHOLDS

    scores = [CONFIDENCE_SCORE.get(c, 3) for c in confidences]
    roc_points = []

    for threshold in sorted(thresholds):
        predicted = [s >= threshold for s in scores]
        metrics = compute_binary_metrics(predicted, ground_truth)
        metrics["threshold"] = threshold
        metrics["fpr"] = round(1 - metrics["specificity"], 4)
        roc_points.append(metrics)

    return roc_points


def compute_auc(roc_points: List[Dict]) -> float:
    """Compute AUC from ROC points using trapezoidal rule."""
    # Sort by FPR ascending
    points = sorted(roc_points, key=lambda p: p["fpr"])

    # Add (0,0) and (1,1) endpoints if not present
    fprs = [p["fpr"] for p in points]
    tprs = [p["sensitivity"] for p in points]

    if fprs[0] != 0.0:
        fprs.insert(0, 0.0)
        tprs.insert(0, 0.0)
    if fprs[-1] != 1.0:
        fprs.append(1.0)
        tprs.append(1.0)

    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2
    return round(auc, 4)


def compute_wss_at_95(
    confidences: List[str],
    ground_truth: List[bool],
    target_sensitivity: float = 0.95,
) -> Optional[float]:
    """
    Compute Work Saved over Sampling at target sensitivity (WSS@95%).

    WSS@k = (TN + FN) / N - (1 - k)
    where the threshold is chosen to achieve sensitivity >= k.

    Returns None if target sensitivity is unreachable.
    """
    scores = [CONFIDENCE_SCORE.get(c, 3) for c in confidences]
    n = len(scores)
    if n == 0:
        return None

    # Try thresholds from low to high (most inclusive to most exclusive)
    for threshold in sorted(set(scores)):
        predicted = [s >= threshold for s in scores]
        metrics = compute_binary_metrics(predicted, ground_truth)
        if metrics["sensitivity"] >= target_sensitivity:
            tn_fn = metrics["tn"] + metrics["fn"]
            wss = tn_fn / n - (1 - target_sensitivity)
            return round(wss, 4)

    return None


def compute_calibration(
    confidences: List[str],
    ground_truth: List[bool],
) -> Dict:
    """
    Compute calibration statistics for 5-point scores.

    Returns:
        mean_score: average ordinal score
        score_relevance_correlation: point-biserial correlation (score vs inclusion)
        calibration_slope: slope of observed inclusion rate vs score level
        per_level: {score: {"n": int, "inclusion_rate": float}}
    """
    scores = [CONFIDENCE_SCORE.get(c, 3) for c in confidences]
    n = len(scores)
    if n == 0:
        return {}

    # Mean score
    mean_score = round(sum(scores) / n, 3)

    # Per-level inclusion rates (for calibration curve)
    per_level = {}
    for level in sorted(set(scores)):
        indices = [i for i, s in enumerate(scores) if s == level]
        n_level = len(indices)
        n_included = sum(1 for i in indices if ground_truth[i])
        per_level[level] = {
            "n": n_level,
            "inclusion_rate": round(n_included / max(n_level, 1), 4),
        }

    # Point-biserial correlation (score vs binary inclusion)
    gt_numeric = [1.0 if g else 0.0 for g in ground_truth]
    mean_s = sum(scores) / n
    mean_g = sum(gt_numeric) / n
    cov = sum((s - mean_s) * (g - mean_g) for s, g in zip(scores, gt_numeric)) / n
    std_s = (sum((s - mean_s) ** 2 for s in scores) / n) ** 0.5
    std_g = (sum((g - mean_g) ** 2 for g in gt_numeric) / n) ** 0.5
    correlation = round(cov / max(std_s * std_g, 1e-10), 4)

    # Calibration slope: simple linear regression of inclusion_rate on score level
    levels = sorted(per_level.keys())
    if len(levels) >= 2:
        x = levels
        y = [per_level[l]["inclusion_rate"] for l in levels]
        n_l = len(x)
        mean_x = sum(x) / n_l
        mean_y = sum(y) / n_l
        ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        ss_xx = sum((xi - mean_x) ** 2 for xi in x)
        slope = round(ss_xy / max(ss_xx, 1e-10), 4)
    else:
        slope = None

    return {
        "mean_score": mean_score,
        "score_relevance_correlation": correlation,
        "calibration_slope": slope,
        "per_level": per_level,
    }


def compute_arbitrator_stats(dual_results: List[Dict]) -> Dict:
    """
    Compute inter-screener agreement and arbitrator trigger rate from dual results.

    Returns:
        inter_screener_kappa: Cohen's kappa between screener1 and screener2
        arbitrator_trigger_rate: fraction of studies where arbiter was invoked
        n_agreements: number of studies where screeners agreed
        n_disagreements: number where they disagreed
    """
    agreements = 0
    disagreements = 0
    arbiter_calls = 0
    s1_decisions = []
    s2_decisions = []

    for r in dual_results:
        s1 = r.get("screener1", {})
        s2 = r.get("screener2", {})
        if not s1 or not s2:
            continue

        d1 = s1.get("decision", "")
        d2 = s2.get("decision", "")
        s1_decisions.append(d1)
        s2_decisions.append(d2)

        if d1 == d2:
            agreements += 1
        else:
            disagreements += 1

        method = r.get("resolution_method", "")
        if method in ("arbiter", "arbitration"):
            arbiter_calls += 1

    total = agreements + disagreements
    if total == 0:
        return {
            "inter_screener_kappa": None,
            "arbitrator_trigger_rate": None,
            "n_agreements": 0,
            "n_disagreements": 0,
        }

    # Cohen's kappa
    categories = sorted(set(s1_decisions + s2_decisions))
    if len(categories) >= 2:
        from src.utils.agreement import cohens_kappa
        kappa = cohens_kappa(s1_decisions, s2_decisions)
    else:
        kappa = 1.0 if agreements == total else 0.0

    # Arbitrator trigger rate: if not explicitly logged, use disagreement rate
    trigger_rate = arbiter_calls / total if arbiter_calls > 0 else disagreements / total

    return {
        "inter_screener_kappa": round(kappa, 4) if isinstance(kappa, float) else kappa,
        "arbitrator_trigger_rate": round(trigger_rate, 4),
        "n_agreements": agreements,
        "n_disagreements": disagreements,
    }


# ---------------------------------------------------------------------------
# Arm extraction from existing dual screening results
# ---------------------------------------------------------------------------

def extract_single_arm_from_dual(
    dual_results: List[Dict],
    screener_key: str,  # "screener1" or "screener2"
    arm_name: str,
    model_name: str,
) -> List[Dict]:
    """
    Extract single-agent decisions from dual screening results.
    Repackages each study's screener1 or screener2 decision as if it were
    the sole screener (for benchmark comparison).
    """
    single_results = []
    for r in dual_results:
        screener = r.get(screener_key, {})
        if not screener:
            continue
        single_results.append({
            "study_id": r["study_id"],
            "screener1": screener,
            "screener2": None,
            "final_decision": screener.get("decision", "human_review"),
            "final_confidence": screener.get("confidence", "undecided"),
            "resolution_method": "single_agent",
            "_arm": arm_name,
            "_model": model_name,
        })
    return single_results


def load_asreview_results(
    csv_path: str,
    label_col: str = "label_included",
    id_col: str = "study_id",
) -> List[Dict]:
    """
    Load ASReview screening results from CSV.
    ASReview outputs binary include/exclude — map to 5-point scale:
      include → "likely_include" (score=4)
      exclude → "likely_exclude" (score=2)

    This conservative mapping avoids inflating ASReview's ROC curve by
    not using the extreme endpoints (5/1).
    """
    import csv

    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            included = str(row.get(label_col, "0")).strip() in ("1", "true", "True", "include")
            confidence = "likely_include" if included else "likely_exclude"
            results.append({
                "study_id": row.get(id_col, ""),
                "screener1": {
                    "decision": "include" if included else "exclude",
                    "confidence": confidence,
                    "reasoning": "ASReview automated decision",
                },
                "screener2": None,
                "final_decision": "include" if included else "exclude",
                "final_confidence": confidence,
                "resolution_method": "asreview",
                "_arm": "asreview",
                "_model": "ASReview (active learning)",
            })
    return results


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class ScreeningBenchmark:
    """
    Runs multi-arm screening benchmark with ROC curves.

    Usage:
        bench = ScreeningBenchmark()
        bench.add_arm("single_gemini", results_list, model="Gemini 3.1 Pro")
        bench.add_arm("single_gpt", results_list, model="GPT-4.1 Mini")
        bench.add_arm("dual", results_list, model="Gemini+GPT+Claude")
        bench.set_ground_truth(ground_truth_dict)
        report = bench.compute_all()
    """

    def __init__(self):
        self.arms: Dict[str, List[Dict]] = {}
        self.arm_models: Dict[str, str] = {}
        self.arm_costs: Dict[str, float] = {}
        self.ground_truth: Dict[str, bool] = {}  # study_id → should_include

    def add_arm(self, name: str, results: List[Dict], model: str = "",
                cost_usd: float = 0.0):
        self.arms[name] = results
        self.arm_models[name] = model
        self.arm_costs[name] = cost_usd

    def set_ground_truth(self, gt: Dict[str, bool]):
        """Set ground truth: {study_id: True/False for include/exclude}."""
        self.ground_truth = gt

    def load_ground_truth_from_file(self, path: str, id_col: str = "study_id",
                                     label_col: str = "included"):
        """Load ground truth from CSV or JSON."""
        p = Path(path)
        if p.suffix == ".json":
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.ground_truth = {
                    d[id_col]: bool(d.get(label_col, False))
                    for d in data
                }
            else:
                self.ground_truth = data
        elif p.suffix == ".csv":
            import csv
            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row.get(id_col, "")
                    inc = str(row.get(label_col, "0")).strip() in ("1", "true", "True")
                    self.ground_truth[sid] = inc

    def compute_arm(self, name: str) -> ArmResult:
        """Compute metrics for one arm."""
        results = self.arms[name]
        model = self.arm_models.get(name, "")
        cost = self.arm_costs.get(name, 0.0)

        # Extract confidences and ground truth (only for studies with GT)
        confidences = []
        gt_labels = []
        conf_dist = Counter()
        dec_dist = Counter()

        for r in results:
            sid = r["study_id"]
            conf = r.get("final_confidence", "undecided")
            dec = r.get("final_decision", "human_review")
            conf_dist[conf] += 1
            dec_dist[dec] += 1

            if sid in self.ground_truth:
                confidences.append(conf)
                gt_labels.append(self.ground_truth[sid])

        if not gt_labels:
            logger.warning(f"Arm '{name}': no ground truth overlap found")
            return ArmResult(arm_name=name, model=model, n_studies=len(results))

        # ROC curve
        roc_points = compute_roc_curve(confidences, gt_labels)
        auc = compute_auc(roc_points)

        # WSS@95%
        wss = compute_wss_at_95(confidences, gt_labels)

        # Calibration (Table 7b)
        cal = compute_calibration(confidences, gt_labels)

        # Multi-agent stats (only for dual-agent arms)
        is_dual = any(r.get("screener2") is not None for r in results)
        arb_stats = compute_arbitrator_stats(results) if is_dual else {}

        # Metrics at operating point
        op_point = next(
            (p for p in roc_points if p["threshold"] == DEFAULT_THRESHOLD),
            roc_points[0] if roc_points else {},
        )

        n = len(results)
        return ArmResult(
            arm_name=name,
            model=model,
            n_studies=n,
            roc_points=roc_points,
            sensitivity=op_point.get("sensitivity", 0.0),
            specificity=op_point.get("specificity", 0.0),
            f1=op_point.get("f1", 0.0),
            ppv=op_point.get("ppv", 0.0),
            npv=op_point.get("npv", 0.0),
            wss_at_95=wss,
            auc=auc,
            cost_usd=cost,
            cost_per_100=round(cost / n * 100, 4) if n > 0 else 0.0,
            confidence_distribution=dict(conf_dist),
            decision_distribution=dict(dec_dist),
            mean_score=cal.get("mean_score"),
            score_relevance_correlation=cal.get("score_relevance_correlation"),
            calibration_slope=cal.get("calibration_slope"),
            inter_screener_kappa=arb_stats.get("inter_screener_kappa"),
            arbitrator_trigger_rate=arb_stats.get("arbitrator_trigger_rate"),
        )

    def compute_all(self) -> Dict[str, ArmResult]:
        """Compute metrics for all arms."""
        results = {}
        for name in self.arms:
            results[name] = self.compute_arm(name)
        return results

    def export_table(self, arm_results: Dict[str, ArmResult] = None) -> str:
        """Generate markdown comparison table."""
        if arm_results is None:
            arm_results = self.compute_all()

        lines = [
            "# Screening Benchmark Comparison",
            "",
            "## Table 7a: Binary Decision Metrics",
            "",
            "| Arm | Model | n | Sens. | Spec. | WSS@95% | AUC | $/100 |",
            "|-----|-------|---|-------|-------|---------|-----|-------|",
        ]

        for name, r in arm_results.items():
            wss_str = f"{r.wss_at_95:.3f}" if r.wss_at_95 is not None else "N/A"
            lines.append(
                f"| {r.arm_name} | {r.model} | {r.n_studies} | "
                f"{r.sensitivity:.3f} | {r.specificity:.3f} | "
                f"{wss_str} | {r.auc:.3f} | "
                f"${r.cost_per_100:.2f} |"
            )

        # Table 7b: Calibration (LLM arms only)
        llm_arms = {n: r for n, r in arm_results.items()
                    if r.mean_score is not None}
        if llm_arms:
            lines.extend([
                "",
                "## Table 7b: Calibration & Multi-Agent Analysis",
                "",
                "| Arm | Mean Score | Score-Incl. Corr. | Cal. Slope | κ (inter-screener) | Arb. Trigger Rate |",
                "|-----|-----------|-------------------|-----------|-------------------|-------------------|",
            ])
            for name, r in llm_arms.items():
                corr_str = f"{r.score_relevance_correlation:.3f}" if r.score_relevance_correlation is not None else "N/A"
                slope_str = f"{r.calibration_slope:.3f}" if r.calibration_slope is not None else "N/A"
                kappa_str = f"{r.inter_screener_kappa:.3f}" if r.inter_screener_kappa is not None else "N/A"
                arb_str = f"{r.arbitrator_trigger_rate:.1%}" if r.arbitrator_trigger_rate is not None else "N/A"
                lines.append(
                    f"| {r.arm_name} | {r.mean_score:.2f} | "
                    f"{corr_str} | {slope_str} | {kappa_str} | {arb_str} |"
                )

        lines.append("")
        lines.append(f"Operating point: threshold = {DEFAULT_THRESHOLD} "
                      f"(include if confidence >= likely_include)")
        return "\n".join(lines)

    def export_roc_data(self, arm_results: Dict[str, ArmResult] = None) -> Dict:
        """Export ROC data for plotting."""
        if arm_results is None:
            arm_results = self.compute_all()

        roc_data = {}
        for name, r in arm_results.items():
            points = r.roc_points
            auc = compute_auc(points) if points else 0.0
            roc_data[name] = {
                "model": r.model,
                "auc": auc,
                "points": points,
            }
        return roc_data

    def plot_roc(self, arm_results: Dict[str, ArmResult] = None,
                 output_path: str = None):
        """Plot ROC curves for all arms."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping ROC plot")
            return

        if arm_results is None:
            arm_results = self.compute_all()

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
                   "#00BCD4", "#795548"]

        for i, (name, r) in enumerate(arm_results.items()):
            if not r.roc_points:
                continue

            auc = compute_auc(r.roc_points)
            fprs = [p["fpr"] for p in sorted(r.roc_points, key=lambda p: p["fpr"])]
            tprs = [p["sensitivity"] for p in sorted(r.roc_points, key=lambda p: p["fpr"])]

            color = colors[i % len(colors)]
            ax.plot(fprs, tprs, "o-", color=color, linewidth=2, markersize=8,
                    label=f"{name} ({r.model}) AUC={auc:.3f}")

            # Mark operating point (threshold=3.5)
            op = next((p for p in r.roc_points if p["threshold"] == DEFAULT_THRESHOLD), None)
            if op:
                ax.plot(op["fpr"], op["sensitivity"], "*", color=color,
                        markersize=15, markeredgecolor="black", markeredgewidth=1)

        # Add otto-SR reference point
        ax.plot(1 - 0.939, 0.966, "D", color="red", markersize=10,
                markeredgecolor="black", markeredgewidth=1.5,
                label="otto-SR (GPT-4.1) Sens=0.966")

        # Diagonal
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)

        ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
        ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=12)
        ax.set_title("Screening ROC: 5-Point Confidence Threshold Sweep", fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC plot saved: {output_path}")
        else:
            plt.show()

        plt.close(fig)

    def save_results(self, arm_results: Dict[str, ArmResult], output_dir: str):
        """Save all benchmark results to output directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON results
        serializable = {name: asdict(r) for name, r in arm_results.items()}
        with open(out / "screening_benchmark.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

        # Markdown table
        table = self.export_table(arm_results)
        with open(out / "screening_benchmark.md", "w", encoding="utf-8") as f:
            f.write(table)

        # ROC data
        roc_data = self.export_roc_data(arm_results)
        with open(out / "roc_data.json", "w", encoding="utf-8") as f:
            json.dump(roc_data, f, indent=2, default=str)

        # ROC plot
        self.plot_roc(arm_results, output_path=str(out / "roc_curves.png"))

        logger.info(f"Benchmark results saved to {out}")
