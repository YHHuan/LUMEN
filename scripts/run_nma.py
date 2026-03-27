"""
Network Meta-Analysis — LUMEN v2
===================================
Standalone NMA script. Can be run independently after Phase 4 extraction.
Uses R netmeta (frequentist, REML) via subprocess.

Usage:
    python scripts/run_nma.py                      # Run NMA from extracted data
    python scripts/run_nma.py --from-csv data.csv  # Run NMA from a CSV file
    python scripts/run_nma.py --validate-only      # Validate network only
    python scripts/run_nma.py --check-r            # Check R + netmeta availability
"""

import sys
import argparse
import logging
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.nma import (
    is_netmeta_available, prepare_nma_data, validate_network,
    run_nma, run_nma_from_settings, load_nma_settings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LUMEN v2 Network Meta-Analysis")
    parser.add_argument("--from-csv", type=str,
                        help="CSV file with columns: studlab,treat1,treat2,TE,seTE")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate the network, don't run NMA")
    parser.add_argument("--check-r", action="store_true",
                        help="Check R + netmeta availability and exit")
    parser.add_argument("--effect-measure", type=str, default=None,
                        help="Override effect measure (SMD, MD, RR, OR, HR)")
    parser.add_argument("--reference-group", type=str, default=None,
                        help="Override reference treatment group")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Check R availability
    if args.check_r:
        if is_netmeta_available():
            print("R + netmeta: OK")
        else:
            print("R + netmeta: NOT AVAILABLE")
            print("Install with: Rscript -e 'install.packages(c(\"netmeta\", \"jsonlite\"))'")
        return

    # Load contrasts
    if args.from_csv:
        contrasts = _load_csv(args.from_csv)
    else:
        contrasts = _load_from_project()

    if not contrasts:
        logger.error("No contrast data available")
        return

    # Validate
    validation = validate_network(contrasts)
    print(f"\nNetwork: {validation['n_treatments']} treatments, "
          f"{validation['n_studies']} studies, {validation['n_contrasts']} contrasts")
    print(f"Treatments: {', '.join(validation['treatments'])}")

    if not validation["valid"]:
        for err in validation["errors"]:
            print(f"  ERROR: {err}")
        return

    for warn in validation.get("warnings", []):
        print(f"  WARNING: {warn}")

    if args.validate_only:
        print("\nNetwork validation passed.")
        return

    # Check R
    if not is_netmeta_available():
        logger.error(
            "R netmeta not available. Install with:\n"
            "  Rscript -e 'install.packages(c(\"netmeta\", \"jsonlite\"))'"
        )
        return

    # Determine output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from src.utils.project import get_data_dir
        output_dir = str(Path(get_data_dir()) / "phase5_analysis" / "nma")

    # Build kwargs for overrides
    kwargs = {}
    if args.effect_measure:
        kwargs["effect_measure"] = args.effect_measure
    if args.reference_group:
        kwargs["reference_group"] = args.reference_group

    # Run NMA
    logger.info("Starting NMA analysis...")
    try:
        if kwargs:
            nma_cfg = load_nma_settings()
            results = run_nma(
                contrasts, output_dir,
                effect_measure=kwargs.get("effect_measure", nma_cfg.get("effect_measure", "SMD")),
                method_tau=nma_cfg.get("method_tau", "REML"),
                reference_group=kwargs.get("reference_group", nma_cfg.get("reference_group")),
                small_values=nma_cfg.get("small_values", "undesirable"),
            )
        else:
            try:
                results = run_nma_from_settings(contrasts, output_dir)
            except ValueError:
                results = run_nma(contrasts, output_dir)
    except Exception as e:
        logger.error(f"NMA failed: {e}")
        return

    # Save alongside Phase 5 data if in project mode
    if not args.from_csv:
        try:
            from src.utils.file_handlers import DataManager
            dm = DataManager()
            dm.save("phase5_analysis", "nma_results.json", results)
        except Exception:
            pass

    # Print summary
    _print_summary(results, output_dir)


def _load_csv(csv_path: str):
    """Load contrasts from a CSV file."""
    import csv
    contrasts = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            contrasts.append({
                "studlab": row["studlab"],
                "treat1": row["treat1"],
                "treat2": row["treat2"],
                "TE": float(row["TE"]),
                "seTE": float(row["seTE"]),
            })
    logger.info(f"Loaded {len(contrasts)} contrasts from {csv_path}")
    return contrasts


def _load_from_project():
    """Load contrasts from project data directory."""
    from src.utils.project import select_project
    from src.utils.file_handlers import DataManager

    select_project()
    dm = DataManager()

    # Try NMA-ready contrasts first
    if dm.exists("phase4_extraction", "nma_contrasts.json"):
        contrasts = dm.load("phase4_extraction", "nma_contrasts.json")
        logger.info(f"Loaded {len(contrasts)} NMA contrasts from Phase 4")
        return contrasts

    # Fall back: generate from extracted_data.json
    if dm.exists("phase4_extraction", "extracted_data.json"):
        logger.info("Generating NMA contrasts from extracted_data.json")
        extracted = dm.load("phase4_extraction", "extracted_data.json")
        contrasts = prepare_nma_data(extracted)
        if contrasts:
            dm.save("phase4_extraction", "nma_contrasts.json", contrasts)
        return contrasts

    logger.error("No extraction data found. Run Phase 4 first.")
    return []


def _print_summary(results, output_dir):
    """Print NMA results summary."""
    print("\n" + "=" * 50)
    print("  NMA Analysis Complete")
    print("=" * 50)
    print(f"  Studies:         {results.get('n_studies', '?')}")
    print(f"  Treatments:      {results.get('n_treatments', '?')}")
    print(f"  Contrasts:       {results.get('n_contrasts', '?')}")
    print(f"  Effect measure:  {results.get('effect_measure', '?')}")
    print(f"  tau2:            {results.get('tau2', '?')}")
    print(f"  I2:              {results.get('I2', '?')}%")

    # Rankings
    rankings = results.get("rankings")
    if rankings:
        print("\n  Treatment Rankings (P-score):")
        if isinstance(rankings, list):
            for r in rankings:
                print(f"    {r.get('rank', '?')}. {r.get('treatment', '?')} "
                      f"(P={r.get('p_score', '?')})")
        elif isinstance(rankings, dict):
            for t, p in sorted(rankings.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
                print(f"    {t}: P={p}")

    # Pairwise vs reference
    pairwise = results.get("pairwise_vs_reference", {})
    if pairwise:
        ref = results.get("reference_group", "?")
        print(f"\n  Pairwise estimates vs {ref}:")
        for t, est in pairwise.items():
            ci_str = f"[{est['lower']:.3f}, {est['upper']:.3f}]"
            sig = "*" if est.get("pval", 1) < 0.05 else ""
            print(f"    {t}: {est['TE']:.3f} {ci_str} p={est.get('pval', '?')}{sig}")

    # Consistency
    consistency = results.get("consistency", {})
    if consistency:
        pval = consistency.get("pval_between", "?")
        incon = consistency.get("inconsistency_detected", False)
        print(f"\n  Consistency: p={pval} ({'INCONSISTENCY DETECTED' if incon else 'OK'})")

    print(f"\n  Output: {output_dir}/")
    print()


if __name__ == "__main__":
    main()
