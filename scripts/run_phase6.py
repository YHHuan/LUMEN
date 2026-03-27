"""
Phase 6: Manuscript Writing — LUMEN v2
=========================================
Citation-grounded manuscript generation with reference markers.

Usage:
    python scripts/run_phase6.py                                # Full manuscript
    python scripts/run_phase6.py --section discussion           # Single section
    python scripts/run_phase6.py --sections introduction,methods # Multiple sections
    python scripts/run_phase6.py --skip-validation              # Skip citation checks
    python scripts/run_phase6.py --validate-only                # Only citation validation
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.config import cfg
from src.agents.writer import WriterAgent, CitationGuardianAgent, ReferencePool
from src.utils.citation_verifier import HybridCitationVerifier
from src.utils.stage_gate import validate_phase5_to_6

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_SECTIONS = ["introduction", "methods", "results", "discussion", "conclusion"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=str)
    parser.add_argument("--sections", type=str)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.section:
        sections = [args.section]
    elif args.sections:
        sections = [s.strip() for s in args.sections.split(",")]
    else:
        sections = ALL_SECTIONS

    select_project()
    dm = DataManager()

    # Stage gate validation (Phase 5 -> 6)
    gate = validate_phase5_to_6(dm)
    if not gate.passed:
        logger.error("Stage gate Phase 5 -> 6 FAILED. Complete Phase 5 before manuscript generation.")
        return

    # Load context
    stats = dm.load_if_exists("phase5_analysis", "statistical_results.json", default={})
    extracted = dm.load_if_exists("phase4_extraction", "extracted_data.json", default=[])
    pico = dm.load_if_exists("input", "pico.yaml", default={})
    interpretation = dm.load_if_exists("phase5_analysis", "interpretation.json", default={})

    # Detect NMA mode from results
    is_nma = stats.get("analysis_type") == "nma"
    nma_results = dm.load_if_exists("phase5_analysis", "nma_results.json", default={})

    context = {
        "pico": pico,
        "statistical_results": stats,
        "extracted_data_summary": {
            "k": len(extracted),
            "studies": [s.get("canonical_citation", s.get("study_id", ""))
                        for s in extracted],
        },
        "interpretation": interpretation,
        "analysis_type": "nma" if is_nma else "pairwise",
    }

    if is_nma and nma_results:
        context["nma"] = {
            "n_treatments": nma_results.get("n_treatments"),
            "treatments": nma_results.get("treatments", []),
            "rankings": nma_results.get("rankings"),
            "pairwise_vs_reference": nma_results.get("pairwise_vs_reference"),
            "reference_group": nma_results.get("reference_group"),
            "consistency": nma_results.get("consistency"),
            "tau2": nma_results.get("tau2"),
            "I2": nma_results.get("I2"),
            "effect_measure": nma_results.get("effect_measure"),
        }
        logger.info("NMA mode: enriching writer context with network results")

    # Enrich extracted data with abstracts from screening (for citation matching)
    screening = dm.load_if_exists(
        "phase3_screening", "included_studies.json",
        subfolder="stage1_title_abstract", default=[])
    abstract_map = {s.get("study_id"): s.get("abstract", "")
                    for s in screening if s.get("abstract")}
    enriched = 0
    for study in extracted:
        if not study.get("abstract") and study.get("study_id") in abstract_map:
            study["abstract"] = abstract_map[study["study_id"]]
            enriched += 1
    if enriched:
        logger.info(f"Enriched {enriched}/{len(extracted)} studies with abstracts for citation matching")

    # Build reference pool
    p6 = cfg.phase6_settings
    reference_pool = ReferencePool()
    reference_pool.load_from_extraction(extracted)

    ref_yaml = Path(get_data_dir()) / "input" / "references.yaml"
    if ref_yaml.exists():
        reference_pool.load_from_yaml(str(ref_yaml))

    reference_pool.build_index()
    logger.info(f"Reference pool: {len(reference_pool.references)} references")

    # Build hybrid verifier (BM25 + vector)
    hybrid_verifier = HybridCitationVerifier(
        references=reference_pool.references,
        vector_pool=reference_pool,
    )
    logger.info("Hybrid citation verifier ready (BM25 + vector)")

    budget = TokenBudget("phase6", limit_usd=cfg.budget("phase6"), reset=True)

    if not args.validate_only:
        # Write sections
        writer = WriterAgent(budget=budget)
        drafts_dir = dm.phase_dir("phase6_manuscript", "drafts")

        for section in sections:
            logger.info(f"Writing {section}...")
            text = writer.write_section(section, context)

            # Save raw draft (with [REF:] markers)
            with open(drafts_dir / f"{section}_raw.md", "w", encoding="utf-8") as f:
                f.write(text)

            # Resolve citations
            if not args.skip_validation and p6.get("citation_mode") == "grounded":
                guardian = CitationGuardianAgent(budget=budget)
                resolved_text, citation_log = guardian.resolve_citations(
                    text, reference_pool, hybrid_verifier=hybrid_verifier
                )

                with open(drafts_dir / f"{section}.md", "w", encoding="utf-8") as f:
                    f.write(resolved_text)

                dm.save("phase6_manuscript", f"{section}_citations.json",
                        citation_log, subfolder="drafts")

                # Assertion-level verification on resolved text
                verify_report = hybrid_verifier.verify_manuscript(
                    resolved_text, guardian_agent=guardian
                )
                dm.save("phase6_manuscript", f"{section}_verification.json",
                        verify_report, subfolder="drafts")
                vsum = verify_report["summary"]
                logger.info(
                    f"  Assertion verification: {vsum['verified']}/{vsum['total_assertions']} "
                    f"({vsum['verification_rate']:.0%})"
                )
            else:
                with open(drafts_dir / f"{section}.md", "w", encoding="utf-8") as f:
                    f.write(text)

            logger.info(f"  {section} complete ({len(text)} chars)")

    # Citation validation only
    if args.validate_only:
        guardian = CitationGuardianAgent(budget=budget)
        drafts_dir = dm.phase_dir("phase6_manuscript", "drafts")

        for section in sections:
            raw_path = drafts_dir / f"{section}_raw.md"
            if raw_path.exists():
                with open(raw_path, "r") as f:
                    text = f.read()
                resolved, log = guardian.resolve_citations(
                    text, reference_pool, hybrid_verifier=hybrid_verifier
                )
                with open(drafts_dir / f"{section}.md", "w") as f:
                    f.write(resolved)
                dm.save("phase6_manuscript", f"{section}_citations.json",
                        log, subfolder="drafts")

                verify_report = hybrid_verifier.verify_manuscript(
                    resolved, guardian_agent=guardian
                )
                dm.save("phase6_manuscript", f"{section}_verification.json",
                        verify_report, subfolder="drafts")

    print(f"\n  Phase 6 complete. Budget: {budget.summary()['total_cost_usd']}")

    # Render combined manuscript via pandoc (if available)
    _render_manuscript(dm)


def _render_manuscript(dm):
    """Combine section drafts into a single manuscript and render via pandoc."""
    import subprocess as sp
    drafts_dir = dm.phase_dir("phase6_manuscript", "drafts")
    out_dir = dm.phase_dir("phase6_manuscript")

    # Combine resolved sections in order
    combined = []
    for section in ALL_SECTIONS:
        path = drafts_dir / f"{section}.md"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                combined.append(f"# {section.title()}\n\n{f.read()}")

    if not combined:
        logger.warning("No manuscript sections to render")
        return

    combined_md = out_dir / "manuscript.md"
    with open(combined_md, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(combined))
    logger.info(f"Combined manuscript: {combined_md}")

    # Try pandoc render
    pandoc_available = False
    try:
        sp.run(["pandoc", "--version"], capture_output=True, timeout=10)
        pandoc_available = True
    except (FileNotFoundError, sp.TimeoutExpired):
        pass

    if pandoc_available:
        for fmt, ext in [("docx", "docx"), ("pdf", "pdf")]:
            out_file = out_dir / f"manuscript.{ext}"
            try:
                cmd = [
                    "pandoc", str(combined_md),
                    "-o", str(out_file),
                    "--from", "markdown",
                    "-V", "geometry:margin=1in",
                    "-V", "fontsize=11pt",
                ]
                result = sp.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and out_file.exists():
                    logger.info(f"Rendered: {out_file}")
                else:
                    logger.warning(f"pandoc {fmt} failed: {result.stderr[:200]}")
            except Exception as e:
                logger.warning(f"pandoc {fmt} render failed: {e}")
    else:
        logger.info("pandoc not found — manuscript.md saved (install pandoc for PDF/DOCX)")

    # Generate ground truth comparison template
    _generate_comparison_template(dm, out_dir)


def _generate_comparison_template(dm, out_dir):
    """Generate a comparison table template: LUMEN output vs ground truth."""
    stats = dm.load_if_exists("phase5_analysis", "statistical_results.json", default={})
    main = stats.get("main", {})

    template = [
        "# LUMEN Output vs Ground Truth Comparison",
        "",
        "| Metric | LUMEN Output | Ground Truth | Match |",
        "|--------|-------------|-------------|-------|",
        f"| k (studies) | {main.get('k', '?')} | | |",
        f"| Pooled effect | {main.get('pooled_effect', '?'):.4f} | | |" if main.get('pooled_effect') else "| Pooled effect | ? | | |",
        f"| 95% CI lower | {main.get('ci_lower', '?'):.4f} | | |" if main.get('ci_lower') else "| 95% CI lower | ? | | |",
        f"| 95% CI upper | {main.get('ci_upper', '?'):.4f} | | |" if main.get('ci_upper') else "| 95% CI upper | ? | | |",
        f"| p-value | {main.get('p_value', '?'):.6f} | | |" if main.get('p_value') else "| p-value | ? | | |",
        f"| I2 (%) | {main.get('I2', '?'):.1f} | | |" if main.get('I2') is not None else "| I2 (%) | ? | | |",
        f"| tau2 | {main.get('tau2', '?'):.4f} | | |" if main.get('tau2') is not None else "| tau2 | ? | | |",
        f"| Estimator | {main.get('estimator', '?')} | | |",
        f"| Adjustment | {main.get('adjustment', '?')} | | |",
        "",
        "## Notes",
        "- Fill 'Ground Truth' column from the published meta-analysis",
        "- Match column: exact / within-1% / different",
    ]

    comp_path = out_dir / "comparison_template.md"
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(template))
    logger.info(f"Comparison template: {comp_path}")


if __name__ == "__main__":
    main()
