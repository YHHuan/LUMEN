"""
LUMEN v3 CLI.

Commands:
  lumen run --project <dir> [--phases 1-6] [--resume]
  lumen interactive --project <dir>
  lumen cost --project <dir>
  lumen validate --project <dir>
  lumen plot --project <dir> --type forest|funnel|prisma
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger()


def _save_pipeline_state(result: dict, output_dir: Path) -> None:
    """Save pipeline state keys needed for PRISMA diagram and audit."""
    state_keys = [
        "raw_results", "deduplicated_studies", "prescreen_results",
        "screening_results", "fulltext_results", "included_studies",
        "outcome_clusters", "anomaly_flags", "quality_assessments",
    ]
    state_snapshot = {}
    for key in state_keys:
        val = result.get(key)
        if val is not None:
            state_snapshot[key] = val
    with open(output_dir / "pipeline_state.json", "w") as f:
        json.dump(state_snapshot, f, indent=2, default=str)


def _check_api_keys() -> None:
    """Validate required API keys are set. Exit with helpful message if not."""
    import os
    missing = []
    # At least one LLM provider key is needed
    has_any_llm = False
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if os.getenv(var):
            has_any_llm = True
            break

    if not has_any_llm:
        print(
            "ERROR: No LLM API key found.\n"
            "Set at least one of:\n"
            "  ANTHROPIC_API_KEY  (for Claude Sonnet/Opus)\n"
            "  GEMINI_API_KEY     (for Gemini Flash/Pro)\n"
            "  OPENAI_API_KEY     (for GPT models)\n"
            "  OPENROUTER_API_KEY (for OpenRouter unified access)\n"
            "\nExample: export ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """Run the LUMEN pipeline."""
    from lumen.core.config import load_config
    from lumen.core.router import ModelRouter
    from lumen.core.cost import CostTracker
    from lumen.core.graph import build_graph

    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()  # loads from default configs/ directory
    _check_api_keys()
    router = ModelRouter(config=config)
    cost_tracker = CostTracker(str(project_dir))

    graph = build_graph(router=router, cost_tracker=cost_tracker, config=config)

    # Build initial state
    initial_state: dict = {"current_phase": "start"}

    # Load PICO if available
    pico_path = project_dir / "pico.json"
    if pico_path.exists():
        with open(pico_path, "r") as f:
            initial_state["pico"] = json.load(f)
        logger.info("pico_loaded", path=str(pico_path))

    # Load studies if available
    studies_path = project_dir / "studies.json"
    if studies_path.exists():
        with open(studies_path, "r") as f:
            initial_state["raw_results"] = json.load(f)
        logger.info("studies_loaded", path=str(studies_path),
                     n=len(initial_state["raw_results"]))

    thread_id = args.project
    config_dict = {"configurable": {"thread_id": thread_id}}

    print(f"LUMEN v3 — Running pipeline for project: {project_dir}")
    print(f"  Phases: {args.phases}")
    print(f"  Resume: {args.resume}")
    print()

    try:
        result = graph.invoke(initial_state, config=config_dict)

        # Save outputs
        output_dir = project_dir / "output"
        output_dir.mkdir(exist_ok=True)

        if "manuscript_sections" in result:
            for section, text in result["manuscript_sections"].items():
                out_path = output_dir / f"{section}.txt"
                with open(out_path, "w") as f:
                    f.write(text)
            print(f"\nManuscript sections saved to {output_dir}/")

        if "statistics_results" in result:
            with open(output_dir / "statistics.json", "w") as f:
                json.dump(result["statistics_results"], f, indent=2, default=str)

        if "fact_check_log" in result:
            with open(output_dir / "fact_check.json", "w") as f:
                json.dump(result["fact_check_log"], f, indent=2, default=str)

        # Save full pipeline state for PRISMA diagram and audit
        _save_pipeline_state(result, output_dir)

        print("\nPipeline complete.")
        summary = cost_tracker.summary()
        print(f"  Total cost: ${summary['grand_total_cost']:.4f}")

    except Exception as e:
        logger.error("pipeline_failed", error=str(e))
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_cost(args: argparse.Namespace) -> None:
    """Show cost summary for a project."""
    from lumen.core.cost import CostTracker

    project_dir = Path(args.project).resolve()
    if not project_dir.exists():
        print(f"Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    log_path = project_dir / "cost_log.jsonl"
    if log_path.exists():
        tracker = CostTracker.from_jsonl(log_path)
    else:
        tracker = CostTracker(str(project_dir))
    summary = tracker.summary()

    print(f"LUMEN v3 — Cost Summary for: {project_dir.name}")
    print("=" * 60)

    by_phase = summary.get("by_phase", {})
    for phase, agents in sorted(by_phase.items()):
        print(f"\n  Phase: {phase}")
        for agent, stats in sorted(agents.items()):
            calls = stats.get("calls", 0)
            tokens = stats.get("input_tokens", 0) + stats.get("output_tokens", 0)
            cost = stats.get("cost", 0.0)
            print(f"    {agent:30s}  calls={calls:4d}  tokens={tokens:8d}  ${cost:.4f}")

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: calls={summary['grand_total_calls']}, "
          f"tokens={summary['grand_total_tokens']}, "
          f"cost=${summary['grand_total_cost']:.4f}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Run fact-checker on existing manuscript."""
    project_dir = Path(args.project).resolve()
    output_dir = project_dir / "output"

    if not output_dir.exists():
        print(f"No output directory found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Load stats and manuscript
    stats_path = output_dir / "statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        print("No statistics.json found — cannot validate", file=sys.stderr)
        sys.exit(1)

    sections = {}
    for section_file in output_dir.glob("*.txt"):
        name = section_file.stem
        sections[name] = section_file.read_text()

    if not sections:
        print("No manuscript sections found", file=sys.stderr)
        sys.exit(1)

    print(f"LUMEN v3 — Validating manuscript for: {project_dir.name}")
    print(f"  Sections found: {', '.join(sections.keys())}")
    print(f"  Running fact-checker...")

    from lumen.core.config import load_config
    from lumen.core.router import ModelRouter
    from lumen.core.cost import CostTracker
    from lumen.agents.writer import WriterAgent

    config = load_config()
    _check_api_keys()
    router = ModelRouter(config=config)
    cost_tracker = CostTracker(str(project_dir))
    writer = WriterAgent(router=router, cost_tracker=cost_tracker, config=config)

    all_claims = []
    for name, text in sections.items():
        result = writer._fact_check_section(text, name, stats, [], {})
        claims = result.get("claims", [])
        all_claims.extend(claims)
        n_c = sum(1 for c in claims if c.get("verdict") == "CONTRADICTED")
        n_u = sum(1 for c in claims if c.get("verdict") == "UNSUPPORTED")
        n_s = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
        print(f"  {name:15s}: {n_s} supported, {n_c} contradicted, {n_u} unsupported")

    # Save validation report
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(all_claims, f, indent=2, default=str)

    total_c = sum(1 for c in all_claims if c.get("verdict") == "CONTRADICTED")
    if total_c > 0:
        print(f"\n  WARNING: {total_c} contradicted claims found. Review validation_report.json")
    else:
        print(f"\n  All claims verified.")


def cmd_interactive(args: argparse.Namespace) -> None:
    """Interactive step-by-step pipeline with human checkpoints."""
    from lumen.core.config import load_config
    from lumen.core.router import ModelRouter
    from lumen.core.cost import CostTracker

    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    _check_api_keys()
    router = ModelRouter(config=config)
    cost_tracker = CostTracker(str(project_dir))

    print("LUMEN v3 — Interactive Mode")
    print("=" * 50)

    # Step 1: PICO
    print("\n[Step 1/6] PICO Definition")
    pico_path = project_dir / "pico.json"
    if pico_path.exists():
        with open(pico_path) as f:
            pico = json.load(f)
        print(f"  Loaded existing PICO from {pico_path}")
    else:
        pico = {}
        print("  No existing PICO found.")
        print("  Enter your research question (or press Enter to skip):")
        question = input("  > ").strip()
        if question:
            pico = {"research_question": question}

    from lumen.agents.pico_interviewer import PICOInterviewerAgent
    interviewer = PICOInterviewerAgent(router=router, cost_tracker=cost_tracker, config=config)
    result = interviewer.elicit(pico)
    pico = result["pico"]
    score = result["completeness_score"]
    print(f"\n  Completeness score: {score}/100")
    print(f"  PICO: {json.dumps(pico, indent=4)}")

    if result.get("questions"):
        print("\n  Suggested refinements:")
        for q in result["questions"]:
            print(f"    - {q}")

    confirm = input("\n  Accept this PICO? [Y/n] ").strip().lower()
    if confirm == "n":
        print("  Please edit pico.json and re-run.")
        with open(pico_path, "w") as f:
            json.dump(pico, f, indent=2)
        return

    with open(pico_path, "w") as f:
        json.dump(pico, f, indent=2)

    # Step 2: Strategy
    print("\n[Step 2/6] Search Strategy Generation")
    from lumen.agents.strategy_generator import StrategyGeneratorAgent
    generator = StrategyGeneratorAgent(router=router, cost_tracker=cost_tracker, config=config)
    strategy_result = generator.generate(pico)
    print(f"  Queries generated: {len(strategy_result['search_strategy'].get('queries', []))}")
    print(f"  Screening criteria: {json.dumps(strategy_result['screening_criteria'], indent=4)}")

    confirm = input("\n  Accept search strategy? [Y/n] ").strip().lower()
    if confirm == "n":
        print("  Strategy rejected. Please refine PICO and re-run.")
        return

    # Step 3: Data loading
    print("\n[Step 3/6] Study Loading")
    studies_path = project_dir / "studies.json"
    if studies_path.exists():
        with open(studies_path) as f:
            studies = json.load(f)
        print(f"  Loaded {len(studies)} studies from {studies_path}")
    else:
        print(f"  No studies.json found at {studies_path}")
        print("  Please prepare your search results and re-run.")
        return

    # Step 4: Analysis plan checkpoint
    print(f"\n[Step 4/6] Pipeline will process {len(studies)} studies through:")
    print("  Prescreen → T/A Screening → Fulltext → Extraction → Harmonization")
    print("  → Statistical Analysis → Quality Assessment → Writing")
    confirm = input("\n  Proceed with full pipeline? [Y/n] ").strip().lower()
    if confirm == "n":
        print("  Pipeline cancelled.")
        return

    # Run full pipeline
    print("\n[Step 5/6] Running pipeline...")
    from lumen.core.graph import build_graph
    graph = build_graph(router=router, cost_tracker=cost_tracker, config=config)

    initial_state = {
        "pico": pico,
        "screening_criteria": strategy_result["screening_criteria"],
        "search_strategy": strategy_result["search_strategy"],
        "raw_results": studies,
        "current_phase": "start",
    }

    result = graph.invoke(initial_state, {"configurable": {"thread_id": args.project}})

    # Step 6: Output
    print("\n[Step 6/6] Saving outputs...")
    output_dir = project_dir / "output"
    output_dir.mkdir(exist_ok=True)

    if "manuscript_sections" in result:
        for section, text in result["manuscript_sections"].items():
            (output_dir / f"{section}.txt").write_text(text)
        print(f"  Manuscript sections → {output_dir}/")

    if "statistics_results" in result:
        with open(output_dir / "statistics.json", "w") as f:
            json.dump(result["statistics_results"], f, indent=2, default=str)

    print("\n  Done! Review outputs in", output_dir)
    summary = cost_tracker.summary()
    total = sum(
        stats.get("cost", 0)
        for agents in summary.values()
        for stats in agents.values()
    )
    print(f"  Total cost: ${total:.4f}")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate visualization plots."""
    project_dir = Path(args.project).resolve()
    output_dir = project_dir / "output"
    output_dir.mkdir(exist_ok=True)

    plot_type = args.type

    if plot_type == "prisma":
        from lumen.tools.visualization.prisma import (
            compute_prisma_counts, generate_prisma_figure, generate_prisma_text,
        )
        state_path = output_dir / "pipeline_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
        else:
            print("No pipeline_state.json found. Using empty state.", file=sys.stderr)
            state = {}

        counts = compute_prisma_counts(state)
        print(generate_prisma_text(counts))

        out_path = output_dir / "prisma_flow.png"
        generate_prisma_figure(counts, output_path=out_path)
        print(f"\nFigure saved: {out_path}")

    elif plot_type in ("forest", "funnel"):
        stats_path = output_dir / "statistics.json"
        if not stats_path.exists():
            print("No statistics.json found", file=sys.stderr)
            sys.exit(1)

        with open(stats_path) as f:
            stats = json.load(f)

        from lumen.tools.visualization.plots import forest_plot, funnel_plot

        for outcome, data in stats.items():
            meta = data.get("meta", {})
            if "error" in data or "error" in meta:
                continue

            # We need per-study data for plots
            loo = data.get("leave_one_out", [])
            if loo:
                effects = [r["pooled_effect"] for r in loo]
                labels = [r["omitted"] for r in loo]
                ci_lo = [r["ci_lower"] for r in loo]
                ci_hi = [r["ci_upper"] for r in loo]
            else:
                continue

            if plot_type == "forest":
                out = output_dir / f"forest_{outcome}.png"
                forest_plot(effects, ci_lo, ci_hi, labels,
                            pooled_effect=meta.get("pooled_effect"),
                            pooled_ci=(meta.get("ci_lower"), meta.get("ci_upper")),
                            title=f"Forest Plot — {outcome}",
                            output_path=out)
                print(f"Forest plot saved: {out}")
            else:
                # Need SEs for funnel
                ses = [abs(hi - lo) / 3.92 for lo, hi in zip(ci_lo, ci_hi)]
                out = output_dir / f"funnel_{outcome}.png"
                funnel_plot(effects, ses,
                            pooled_effect=meta.get("pooled_effect"),
                            title=f"Funnel Plot — {outcome}",
                            output_path=out)
                print(f"Funnel plot saved: {out}")
    else:
        print(f"Unknown plot type: {plot_type}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lumen",
        description="LUMEN v3 — LLM-powered systematic review pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument("--project", required=True, help="Project directory")
    run_parser.add_argument("--phases", default="1-6", help="Phase range (e.g., '1-6', '5-6')")
    run_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # interactive
    int_parser = subparsers.add_parser("interactive", help="Interactive step-by-step mode")
    int_parser.add_argument("--project", required=True, help="Project directory")

    # cost
    cost_parser = subparsers.add_parser("cost", help="Show cost summary")
    cost_parser.add_argument("--project", required=True, help="Project directory")

    # validate
    val_parser = subparsers.add_parser("validate", help="Validate manuscript")
    val_parser.add_argument("--project", required=True, help="Project directory")

    # plot
    plot_parser = subparsers.add_parser("plot", help="Generate visualizations")
    plot_parser.add_argument("--project", required=True, help="Project directory")
    plot_parser.add_argument("--type", required=True,
                             choices=["forest", "funnel", "prisma"],
                             help="Plot type")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "cost":
        cmd_cost(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "plot":
        cmd_plot(args)


if __name__ == "__main__":
    main()
