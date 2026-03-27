"""
Domain Report Collector — LUMEN v2
=====================================
Copies all analysis artifacts into a structured output folder for
publication, review, and comparison. Reusable across all 5 domains.

Usage:
    python scripts/collect_domain_report.py                    # Current active project
    python scripts/collect_domain_report.py --output reports/  # Custom output dir
    python scripts/collect_domain_report.py --all              # All domains

Output structure:
    reports/<domain>/
    ├── 00_manifest.md              ← What's in each folder, source paths, comments
    ├── 01_pipeline_costs/          ← Per-phase token usage + USD cost
    ├── 02_screening/               ← Screening results, stats, kappa, distributions
    ├── 03_extraction/              ← Extracted data, evidence spans
    ├── 04_analysis_plan/           ← Phase 4.5 profile + plan
    ├── 05_statistics/              ← Pooled estimates, per-analysis results, R output
    ├── 06_figures/                 ← Forest plots, funnel plots, ROC curves
    ├── 07_manuscript/              ← Raw + resolved drafts, verification reports
    ├── 08_quality/                 ← RoB-2, GRADE assessments
    ├── 09_ground_truth_comparison/ ← LUMEN vs published meta-analysis comparison
    ├── 10_human_validation/        ← Items requiring human review/decision
    ├── 11_benchmark/               ← Screening + extraction ablation results
    └── 12_supplement/              ← PRISMA flow, search strategy, dedup log
"""

import sys
import argparse
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


# Source mapping: (source_phase, source_file, subfolder) → destination
ARTIFACT_MAP = [
    # --- 01: Pipeline costs ---
    {
        "dest": "01_pipeline_costs",
        "files": [],
        "globs": [
            (".budget", "*_budget.json"),
            ("benchmark", "cost_*.json"),
        ],
        "description": "Per-phase token usage and USD cost tracking. "
                       "Each file contains input/output tokens, LLM calls, and accumulated cost.",
    },
    # --- 02: Screening ---
    {
        "dest": "02_screening",
        "files": [
            ("phase3_screening", "screening_results.json", "stage1_title_abstract"),
            ("phase3_screening", "included_studies.json", "stage1_title_abstract"),
            ("phase3_screening", "excluded_studies.json", "stage1_title_abstract"),
            ("phase3_screening", "human_review_queue.json", "stage1_title_abstract"),
        ],
        "description": "Dual-screener results with 5-point confidence scale. "
                       "screening_results.json contains per-study decisions from both screeners, "
                       "resolution method, Cohen's kappa, and PABAK.",
    },
    # --- 03: Extraction ---
    {
        "dest": "03_extraction",
        "files": [
            ("phase4_extraction", "extracted_data.json", None),
            ("phase4_extraction", "extraction_log.json", None),
            ("phase4_extraction", "evidence_validation.json", None),
        ],
        "description": "Structured data extracted from PDFs. Each study entry contains outcomes "
                       "with evidence_span fields linking to source text/tables.",
    },
    # --- 04: Analysis plan ---
    {
        "dest": "04_analysis_plan",
        "files": [
            ("phase4_5_planning", "data_profile.json", None),
            ("phase4_5_planning", "analysis_plan.yaml", None),
            ("phase4_5_planning", "analysis_plan.json", None),
        ],
        "description": "Phase 4.5 Analysis Planner output. data_profile.json = intervention×outcome "
                       "matrix. analysis_plan.yaml = approved plan with primary/subgroup/sensitivity analyses.",
    },
    # --- 05: Statistics ---
    {
        "dest": "05_statistics",
        "files": [
            ("phase5_analysis", "statistical_results.json", None),
            ("phase5_analysis", "planned_results.json", None),
            ("phase5_analysis", "meta_dedup_log.json", None),
            ("phase5_analysis", "interpretation.json", None),
        ],
        "description": "Statistical analysis results. planned_results.json = per-analysis pooled "
                       "estimates from Phase 4.5 plan. statistical_results.json = R metafor output.",
    },
    # --- 06: Figures ---
    {
        "dest": "06_figures",
        "copy_dirs": [
            ("phase5_analysis", "figures_*"),
        ],
        "globs": [
            ("benchmark/screening", "roc_curves.png"),
        ],
        "description": "Forest plots, funnel plots, and ROC curves. Each figures_<analysis_id>/ "
                       "subfolder contains plots for one Phase 4.5 analysis.",
    },
    # --- 07: Manuscript ---
    {
        "dest": "07_manuscript",
        "files": [
            ("phase6_manuscript", "manuscript.md", None),
        ],
        "copy_dirs": [
            ("phase6_manuscript", "drafts"),
        ],
        "description": "Generated manuscript sections. drafts/ contains raw (with [REF:] markers) "
                       "and resolved versions, plus citation verification reports.",
    },
    # --- 08: Quality ---
    {
        "dest": "08_quality",
        "files": [
            ("quality_assessment", "rob2_assessments.json", None),
            ("quality_assessment", "rob2_summary.json", None),
            ("quality_assessment", "robins_i_assessments.json", None),
            ("quality_assessment", "robins_i_summary.json", None),
            ("quality_assessment", "grade_assessments.json", None),
            ("quality_assessment", "grade_evidence_profile.json", None),
            ("quality_assessment", "grade_summary.md", None),
        ],
        "description": "RoB-2 (RCTs) / ROBINS-I (non-RCTs) risk of bias and GRADE evidence certainty. "
                       "Auto-routed by study design classification.",
    },
    # --- 09: Ground truth comparison ---
    {
        "dest": "09_ground_truth_comparison",
        "files": [
            ("phase4_extraction", "ground_truth.json", None),
            ("phase4_extraction", "extraction_validation.json", None),
            ("phase5_analysis", "ground_truth_estimates.json", None),
            ("phase5_analysis", "synthesis_concordance.json", None),
            ("quality_assessment", "rob_ground_truth.json", None),
            ("quality_assessment", "rob_agreement.json", None),
        ],
        "description": "Validation data: extraction accuracy (Table 8), synthesis concordance (Figure 7), "
                       "RoB agreement (Table 9). Ground truth JSONs must be manually created at BP4/BP6.",
    },
    # --- 10: Human validation ---
    {
        "dest": "10_human_validation",
        "files": [
            ("phase3_screening", "human_review_queue.json", "stage1_title_abstract"),
            ("phase4_extraction", "evidence_validation.json", None),
        ],
        "description": "Items requiring human review: screening undecided cases, extraction "
                       "low-confidence evidence spans, and analysis plan approval status.",
    },
    # --- 11: Benchmark ---
    {
        "dest": "11_benchmark",
        "copy_dirs": [
            ("benchmark", "screening"),
        ],
        "files": [
            ("benchmark", "extraction_ablation_comparison.md", None),
            ("phase5_analysis", "planned_results_ablation_sonnet.json", None),
            ("phase5_analysis", "planned_results_ablation_gemini.json", None),
        ],
        "description": "Ablation study results. screening/ = 5-arm ROC benchmark. "
                       "extraction_ablation_comparison.md = Phase 4-5 multi-pass vs single-model.",
    },
    # --- 12: Supplement ---
    {
        "dest": "12_supplement",
        "files": [
            ("phase1_strategy", "search_strategy.json", None),
            ("phase1_strategy", "screening_criteria.json", None),
            ("phase2_search", "dedup_report.json", "deduplicated"),
            ("input", "pico.yaml", None),
        ],
        "description": "Supplementary materials: search strategy, PICO definition, "
                       "deduplication log, PRISMA flow data.",
    },
]


def collect_domain_report(data_dir: str, output_dir: str, domain_name: str):
    """Collect all artifacts for one domain into structured output."""
    data_path = Path(data_dir)
    out_path = Path(output_dir) / domain_name
    out_path.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        f"# Domain Report: {domain_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source: {data_dir}",
        "",
    ]

    total_copied = 0

    for section in ARTIFACT_MAP:
        dest_name = section["dest"]
        dest_dir = out_path / dest_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        manifest_lines.append(f"## {dest_name}")
        manifest_lines.append(f"{section.get('description', '')}")
        manifest_lines.append("")

        copied_files = []

        # Copy individual files
        for file_spec in section.get("files", []):
            phase, filename, subfolder = file_spec
            if subfolder:
                src = data_path / phase / subfolder / filename
            else:
                src = data_path / phase / filename

            if src.exists():
                dst = dest_dir / filename
                shutil.copy2(str(src), str(dst))
                copied_files.append(f"  - `{filename}` ← `{src.relative_to(data_path)}`")
                total_copied += 1
            else:
                # Try without phase subfolder (e.g., input/pico.yaml)
                src_alt = data_path / phase / filename
                if src_alt.exists() and src_alt != src:
                    dst = dest_dir / filename
                    shutil.copy2(str(src_alt), str(dst))
                    copied_files.append(f"  - `{filename}` ← `{src_alt.relative_to(data_path)}`")
                    total_copied += 1

        # Copy directory patterns
        for dir_spec in section.get("copy_dirs", []):
            phase, pattern = dir_spec
            phase_path = data_path / phase
            if phase_path.exists():
                for match in sorted(phase_path.glob(pattern)):
                    if match.is_dir():
                        dst = dest_dir / match.name
                        if dst.exists():
                            shutil.rmtree(str(dst))
                        shutil.copytree(str(match), str(dst))
                        n_files = sum(1 for _ in match.rglob("*") if _.is_file())
                        copied_files.append(f"  - `{match.name}/` ({n_files} files) ← `{match.relative_to(data_path)}`")
                        total_copied += n_files

        # Copy glob patterns
        for glob_spec in section.get("globs", []):
            phase, pattern = glob_spec
            search_path = data_path / phase if phase != "." else data_path
            if search_path.exists():
                for match in sorted(search_path.glob(pattern)):
                    if match.is_file():
                        dst = dest_dir / match.name
                        shutil.copy2(str(match), str(dst))
                        copied_files.append(f"  - `{match.name}` ← `{match.relative_to(data_path)}`")
                        total_copied += 1

        if copied_files:
            manifest_lines.extend(copied_files)
        else:
            manifest_lines.append("  - *(no files found)*")
        manifest_lines.append("")

    # Collect cost summary across all phases
    _write_cost_summary(data_path, out_path, manifest_lines)

    # Write manifest
    manifest_path = out_path / "00_manifest.md"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))

    logger.info(f"Collected {total_copied} files → {out_path}")
    return total_copied


def _write_cost_summary(data_path: Path, out_path: Path, manifest_lines: list):
    """Aggregate cost data from all phases into a single summary."""
    cost_data = {}

    # Look for budget/token files in each phase directory
    for phase_dir in sorted(data_path.iterdir()):
        if not phase_dir.is_dir():
            continue
        for cost_file in phase_dir.glob("token_usage*.json"):
            try:
                with open(cost_file, encoding="utf-8") as f:
                    data = json.load(f)
                cost_data[phase_dir.name] = data
            except (json.JSONDecodeError, IOError):
                pass

    # Check .budget/ directory (TokenBudget saves here)
    for budget_dir_name in [".budget", ".cache/budgets"]:
        budget_dir = data_path / budget_dir_name
        if budget_dir.exists():
            for budget_file in budget_dir.glob("*.json"):
                try:
                    with open(budget_file, encoding="utf-8") as f:
                        data = json.load(f)
                    phase = budget_file.stem.replace("_budget", "")
                    if phase not in cost_data:
                        cost_data[phase] = data
                except (json.JSONDecodeError, IOError):
                    pass

    if cost_data:
        summary_path = out_path / "01_pipeline_costs" / "cost_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        total_cost = 0
        total_input = 0
        total_output = 0
        total_calls = 0

        for phase, data in cost_data.items():
            total_cost += data.get("total_cost_usd", 0)
            total_input += data.get("total_input_tokens", 0)
            total_output += data.get("total_output_tokens", 0)
            total_calls += data.get("total_llm_calls", 0)

        summary = {
            "per_phase": cost_data,
            "totals": {
                "total_cost_usd": round(total_cost, 4),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_llm_calls": total_calls,
            },
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        manifest_lines.append("## Cost Summary")
        manifest_lines.append(f"  Total cost: ${total_cost:.2f}")
        manifest_lines.append(f"  Total LLM calls: {total_calls}")
        manifest_lines.append(f"  Total input tokens: {total_input:,}")
        manifest_lines.append(f"  Total output tokens: {total_output:,}")
        manifest_lines.append("")


def collect_paper_figures(output_dir: str):
    """
    Aggregate cross-domain data to produce paper-level figures.

    Reads per-domain audit logs, generates:
      - fig2_cost_by_phase_stacked.png (Figure 2)
      - fig3_cost_by_model_tier.png (Figure 3)
      - fig4_wallclock_by_phase.png (Figure 4)
      - fig9_cross_domain_profile.png (Figure 9)

    Also aggregates Tables 4-6 data into summary JSONs.
    """
    from src.utils.cost_tracker import (
        CostTracker,
        generate_paper_cost_figures,
    )

    base = Path("data")
    paper_dir = Path(output_dir) / "_paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)

    domain_reports = {}
    table4_rows = []

    for domain_dir in sorted(base.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith("."):
            continue

        tracker = CostTracker(str(domain_dir))
        n_entries = tracker.load()
        if n_entries == 0:
            continue

        # Count studies for token efficiency
        dm_path = domain_dir / "phase4_extraction" / "extracted_data.json"
        n_studies = 0
        if dm_path.exists():
            try:
                with open(dm_path, encoding="utf-8") as f:
                    n_studies = len(json.load(f))
            except Exception:
                pass

        report = tracker.full_report(n_studies=n_studies)
        domain_reports[domain_dir.name] = report

        # Table 4 row
        cs = report["cost_summary"]
        table4_rows.append({
            "dataset": domain_dir.name,
            "total_tokens": cs["total_tokens"],
            "total_cost_usd": cs["total_cost_usd"],
            "n_api_calls": report["total_api_calls"],
            "n_extracted": n_studies,
        })

    if not domain_reports:
        logger.warning("No domain data found for paper figures")
        return

    # Generate multi-domain cost figures (Fig 2, 3, 4, 9)
    generate_paper_cost_figures(domain_reports, str(paper_dir))

    # Save aggregated table data
    with open(paper_dir / "table4_pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(table4_rows, f, indent=2)

    # Table 5: detailed breakdown (dataset × phase)
    table5_rows = []
    for domain, report in domain_reports.items():
        for phase, data in report.get("cost_by_phase", {}).items():
            table5_rows.append({
                "dataset": domain,
                "phase": phase,
                "input_tokens": data["input_tokens"],
                "output_tokens": data["output_tokens"],
                "api_calls": data["calls"],
                "cost_usd": data["cost_usd"],
                "wall_clock_min": round(data.get("wall_clock_s", 0) / 60, 1),
            })

    with open(paper_dir / "table5_detailed_cost.json", "w", encoding="utf-8") as f:
        json.dump(table5_rows, f, indent=2)

    # Table 6: operational metrics by phase (aggregated)
    phase_agg = {}
    for report in domain_reports.values():
        for phase, data in report.get("cost_by_phase", {}).items():
            if phase not in phase_agg:
                phase_agg[phase] = {
                    "latencies": [], "retry_rates": [],
                    "failure_rates": [], "input_per_call": [], "output_per_call": [],
                }
            phase_agg[phase]["latencies"].append(data.get("avg_latency_s", 0))
            phase_agg[phase]["retry_rates"].append(data.get("retry_rate_pct", 0))
            phase_agg[phase]["failure_rates"].append(data.get("failure_rate_pct", 0))
            if data["calls"] > 0:
                phase_agg[phase]["input_per_call"].append(data["input_tokens"] / data["calls"])
                phase_agg[phase]["output_per_call"].append(data["output_tokens"] / data["calls"])

    table6_rows = []
    for phase, agg in sorted(phase_agg.items()):
        _mean = lambda lst: round(sum(lst) / max(len(lst), 1), 2)
        table6_rows.append({
            "phase": phase,
            "avg_latency_s": _mean(agg["latencies"]),
            "retry_rate_pct": _mean(agg["retry_rates"]),
            "failure_rate_pct": _mean(agg["failure_rates"]),
            "avg_input_tokens_per_call": round(_mean(agg["input_per_call"])),
            "avg_output_tokens_per_call": round(_mean(agg["output_per_call"])),
        })

    with open(paper_dir / "table6_operational_metrics.json", "w", encoding="utf-8") as f:
        json.dump(table6_rows, f, indent=2)

    logger.info(f"Paper figures and tables saved to {paper_dir}")
    print(f"\n  Paper data → {paper_dir}/")
    print(f"  Domains processed: {len(domain_reports)}")
    print(f"  Generated: fig2, fig3, fig4, fig9 + table4, table5, table6 JSONs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="reports",
                        help="Output directory for collected reports")
    parser.add_argument("--all", action="store_true",
                        help="Collect reports for all domains")
    parser.add_argument("--paper", action="store_true",
                        help="Generate cross-domain paper figures and tables")
    args = parser.parse_args()

    if args.paper:
        collect_paper_figures(args.output)
        return

    if args.all:
        # Find all domain directories
        base = Path("data")
        if not base.exists():
            print("No data/ directory found")
            return
        for domain_dir in sorted(base.iterdir()):
            if domain_dir.is_dir() and not domain_dir.name.startswith("."):
                logger.info(f"Collecting: {domain_dir.name}")
                collect_domain_report(str(domain_dir), args.output, domain_dir.name)
    else:
        select_project()
        data_dir = get_data_dir()
        domain_name = Path(data_dir).name
        collect_domain_report(data_dir, args.output, domain_name)

    print(f"\n  Reports collected → {args.output}/")
    print()


if __name__ == "__main__":
    main()
