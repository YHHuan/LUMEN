#!/usr/bin/env python3
"""
Export PRISMA 2020 Flow Diagram as PNG.
========================================
Reads data/phase3_screening/prisma_flow.json and generates a publication-ready
PRISMA 2020 flow diagram.

Usage:
  python scripts/export_prisma_diagram.py
  python scripts/export_prisma_diagram.py --output my_prisma.png

Output (default):
  data/phase5_analysis/figures/prisma_flow.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.visualizations import prisma_flow_diagram


def main():
    parser = argparse.ArgumentParser(description="Generate PRISMA 2020 flow diagram PNG")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: <project>/phase5_analysis/figures/prisma_flow.png)")
    parser.add_argument("--included", type=int, default=None,
                        help="Override 'included_in_synthesis' count (e.g. after deduplication)")
    args = parser.parse_args()

    select_project()
    dd = get_data_dir()
    PRISMA_JSON = f"{dd}/phase3_screening/prisma_flow.json"
    if args.output is None:
        args.output = f"{dd}/phase5_analysis/figures/prisma_flow.png"

    if not Path(PRISMA_JSON).exists():
        print(f"❌ PRISMA data not found: {PRISMA_JSON}")
        print("   Run Phase 3 Stage 2 first: python scripts/run_phase3_stage2.py --review")
        sys.exit(1)

    data = json.loads(Path(PRISMA_JSON).read_text(encoding="utf-8"))

    if args.included is not None:
        data["included_in_synthesis"] = args.included
        print(f"ℹ️  Overriding included_in_synthesis → {args.included}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = prisma_flow_diagram(data, output_path=args.output)

    print(f"✅ PRISMA flow diagram saved: {out}")
    print(f"\nKey counts:")
    print(f"  Identified  : {data.get('total_identified', '?'):,}")
    print(f"  Deduplicated: {data.get('after_deduplication', '?'):,}")
    print(f"  Screened    : {data.get('title_abstract_screened', '?'):,}")
    print(f"  Full-text   : {data.get('fulltext_assessed', '?'):,}")
    print(f"  Included    : {data.get('included_in_synthesis', '?'):,}")


if __name__ == "__main__":
    main()
