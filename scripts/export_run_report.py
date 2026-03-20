"""
Export LLM Run Report — for paper Methods / Supplementary Table
===============================================================
Reads the prompt audit log (.audit/prompt_log.jsonl) and budget files
(.budget/*.json) and produces:

  1. Console table:  per-agent usage (model, calls, tokens, cache, cost)
  2. CSV export:     machine-readable, for Supplementary Table
  3. Text summary:   copy-paste ready for Methods section

Usage:
    python scripts/export_run_report.py
    python scripts/export_run_report.py --out results/llm_report.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.project import select_project, get_data_dir


# ── helpers ──────────────────────────────────────────────────────────────────

def load_audit_log(data_dir: str) -> list:
    path = Path(data_dir) / ".audit" / "prompt_log.jsonl"
    if not path.exists():
        print(f"  [WARN] Audit log not found: {path}")
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def load_budget_files(data_dir: str) -> list:
    budget_dir = Path(data_dir) / ".budget"
    records = []
    if not budget_dir.exists():
        return records
    for bf in sorted(budget_dir.glob("*.json")):
        try:
            with open(bf, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception:
            pass
    return records


def aggregate_by_role(entries: list) -> dict:
    """Aggregate audit log entries by agent role."""
    by_role = defaultdict(lambda: {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cache_hits": 0,
        "estimated_cost_usd": 0.0,
        "models_seen": set(),
    })

    for e in entries:
        role = e.get("role", "unknown")
        r = by_role[role]
        r["calls"] += 1

        # Support both old format (tokens nested) and new format (top-level)
        r["input_tokens"]       += e.get("input_tokens",  e.get("tokens", {}).get("input", 0))
        r["output_tokens"]      += e.get("output_tokens", e.get("tokens", {}).get("output", 0))
        r["cache_read_tokens"]  += e.get("cache_read_tokens",  e.get("tokens", {}).get("cache_read_tokens", 0))
        r["cache_write_tokens"] += e.get("cache_write_tokens", e.get("tokens", {}).get("cache_write_tokens", 0))
        r["estimated_cost_usd"] += e.get("estimated_cost_usd", e.get("tokens", {}).get("estimated_cost_usd", 0.0))

        if e.get("cache_hit") or e.get("tokens", {}).get("cache_read_tokens", 0) > 0:
            r["cache_hits"] += 1

        actual = e.get("actual_model") or e.get("tokens", {}).get("actual_model") or e.get("model_id", "")
        if actual:
            r["models_seen"].add(actual)

    return dict(by_role)


def print_table(by_role: dict) -> None:
    ROLE_ORDER = [
        "strategist", "screener1", "screener2", "arbiter",
        "extractor", "extractor_tiebreaker",
        "statistician", "writer", "methodologist", "citation_guardian",
    ]
    roles = [r for r in ROLE_ORDER if r in by_role]
    roles += [r for r in by_role if r not in ROLE_ORDER]

    # Header
    col = "{:<22} {:>6} {:>10} {:>10} {:>12} {:>10} {:>10}"
    print("\n" + "="*90)
    print("  LLM Run Report — Per-Agent Usage")
    print("="*90)
    print(col.format("Agent", "Calls", "In-Tok", "Out-Tok", "Cache-Read", "Hit%", "Cost $"))
    print("-"*90)

    totals = {"calls":0,"in":0,"out":0,"cr":0,"cw":0,"cost":0.0,"hits":0}

    for role in roles:
        d = by_role[role]
        hit_pct = f"{d['cache_hits']/d['calls']*100:.0f}%" if d["calls"] else "0%"
        print(col.format(
            role,
            d["calls"],
            f"{d['input_tokens']:,}",
            f"{d['output_tokens']:,}",
            f"{d['cache_read_tokens']:,}",
            hit_pct,
            f"{d['estimated_cost_usd']:.4f}",
        ))
        totals["calls"] += d["calls"]
        totals["in"]    += d["input_tokens"]
        totals["out"]   += d["output_tokens"]
        totals["cr"]    += d["cache_read_tokens"]
        totals["cw"]    += d["cache_write_tokens"]
        totals["cost"]  += d["estimated_cost_usd"]
        totals["hits"]  += d["cache_hits"]

    print("-"*90)
    total_hit_pct = f"{totals['hits']/totals['calls']*100:.0f}%" if totals["calls"] else "0%"
    print(col.format(
        "TOTAL",
        totals["calls"],
        f"{totals['in']:,}",
        f"{totals['out']:,}",
        f"{totals['cr']:,}",
        total_hit_pct,
        f"{totals['cost']:.4f}",
    ))
    print("="*90)

    # Cache savings estimate
    if totals["cr"] > 0:
        # If those tokens had not been cached, they'd have been charged at full price
        # We need average input price — approximate from cost / (in - cr) * in
        avg_inp_price = totals["cost"] / max(1, totals["in"]) * 1_000_000  # $/MTok approx
        savings = totals["cr"] * avg_inp_price / 1_000_000 * 0.90  # 90% saved
        print(f"\n  Estimated cache savings: ~${savings:.4f} "
              f"({totals['cr']:,} tokens read from cache @ ~90% discount)")


def print_methods_text(by_role: dict, entries: list) -> None:
    """Print a Methods-section-ready paragraph."""
    total_calls = sum(d["calls"] for d in by_role.values())
    total_cost  = sum(d["estimated_cost_usd"] for d in by_role.values())
    total_in    = sum(d["input_tokens"] for d in by_role.values())
    total_out   = sum(d["output_tokens"] for d in by_role.values())
    total_cr    = sum(d["cache_read_tokens"] for d in by_role.values())

    # Collect unique actual models seen
    all_models = set()
    for e in entries:
        m = e.get("actual_model") or e.get("tokens", {}).get("actual_model", "")
        if m:
            all_models.add(m)

    model_list = ", ".join(sorted(all_models)) or "see per-agent table"

    print("\n" + "="*90)
    print("  Methods Section Text (copy-paste)")
    print("="*90)
    print(f"""
All LLM inference was performed via the OpenRouter API (v1/chat/completions, synchronous
mode) with a fixed temperature of 0.0 and seed 42 for all agents to ensure
reproducibility (TRIPOD-LLM Items 6c). The pipeline made a total of {total_calls:,} API
calls consuming {total_in:,} input tokens and {total_out:,} output tokens
(of which {total_cr:,} input tokens were served from prompt cache, representing a
{total_cr/max(1,total_in)*100:.1f}% cache utilisation rate). The estimated total
inference cost was USD ${total_cost:.2f}. Models used: {model_list}.
Full per-agent token usage and cost breakdown are provided in Supplementary Table S1.
""")


def export_csv(by_role: dict, out_path: str) -> None:
    rows = []
    for role, d in by_role.items():
        rows.append({
            "agent_role":          role,
            "models":              "; ".join(sorted(d["models_seen"])),
            "api_calls":           d["calls"],
            "input_tokens":        d["input_tokens"],
            "output_tokens":       d["output_tokens"],
            "cache_read_tokens":   d["cache_read_tokens"],
            "cache_write_tokens":  d["cache_write_tokens"],
            "cache_hits":          d["cache_hits"],
            "cache_hit_rate_pct":  round(d["cache_hits"]/d["calls"]*100, 1) if d["calls"] else 0,
            "estimated_cost_usd":  round(d["estimated_cost_usd"], 6),
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export LLM run report for paper writing")
    parser.add_argument("--out", default="", help="Path for CSV export (optional)")
    args = parser.parse_args()

    data_dir = select_project(skip_prompt=True)
    print(f"\nProject: {data_dir}")

    entries = load_audit_log(data_dir)
    if not entries:
        print("  No audit log entries found. Run at least one pipeline phase first.")
        return

    print(f"  Loaded {len(entries)} audit log entries.")

    by_role = aggregate_by_role(entries)

    print_table(by_role)
    print_methods_text(by_role, entries)

    if args.out:
        export_csv(by_role, args.out)
    else:
        default_out = str(Path(data_dir) / "phase6_manuscript" / "llm_usage_report.csv")
        export_csv(by_role, default_out)


if __name__ == "__main__":
    main()
