#!/usr/bin/env python3
"""
Progress Checker
=================
查看整體進度:
  python scripts/check_progress.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project


def _print_kappa(filepath: Path):
    """Print κ line if available; silently skip on any error."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        kappa = data.get("cohens_kappa")
        agreement = data.get("agreement_rate") or data.get("side_agreement_rate")
        if kappa is not None:
            agr_str = f" (side agreement {agreement:.1%})" if agreement is not None else ""
            print(f"       └─ κ = {kappa:.2f}{agr_str}")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        pass


def check_phase(base: Path, phase: str, key_files: list) -> dict:
    """檢查某個 phase 的完成狀態"""
    phase_dir = base / phase
    if not phase_dir.exists():
        return {"status": "NOT_STARTED", "files": []}
    
    found = []
    missing = []
    for f in key_files:
        path = phase_dir / f
        if path.exists():
            size = path.stat().st_size
            found.append(f"{f} ({size:,} bytes)")
        else:
            missing.append(f)
    
    if not missing:
        return {"status": "COMPLETE", "files": found}
    elif found:
        return {"status": "IN_PROGRESS", "files": found, "missing": missing}
    else:
        return {"status": "NOT_STARTED", "files": [], "missing": missing}


def main():
    base = Path(select_project())
    
    phases = [
        ("Phase 1: Strategy", "phase1_strategy", 
         ["search_strategy.json", "mesh_terms.json", "screening_criteria.json"]),
        ("Phase 2: Search", "phase2_search", 
         ["deduplicated/all_studies.json", "search_log.json"]),
        ("Phase 3.1: T/A Screen", "phase3_screening",
         ["stage1_title_abstract/included_studies.json"]),
        ("Phase 3.2: Full-text", "phase3_screening",
         ["stage2_fulltext/fulltext_review.json", "stage2_fulltext/included_studies.json"]),
        ("Phase 4: Extraction", "phase4_extraction",
         ["extracted_data.json", "risk_of_bias.json"]),
        ("Phase 5: Analysis", "phase5_analysis",
         ["statistical_results.json", "analysis_code.py"]),
        ("Phase 6: Manuscript", "phase6_manuscript",
         ["drafts/round1_initial.md"]),
    ]
    
    icons = {"COMPLETE": "✅", "IN_PROGRESS": "🔄", "NOT_STARTED": "⏸️ "}
    
    print("\n📊 Meta-Analysis Pipeline Progress")
    print("=" * 55)
    
    for name, phase_dir, key_files in phases:
        result = check_phase(base, phase_dir, key_files)
        icon = icons.get(result["status"], "❓")
        print(f"  {icon} {name:<28} [{result['status']}]")

        if result.get("files"):
            for f in result["files"][:3]:
                print(f"       └─ {f}")

        if name.startswith("Phase 3.1"):
            _print_kappa(base / "phase3_screening" / "stage1_title_abstract" / "screening_results.json")
        elif name.startswith("Phase 3.2"):
            _print_kappa(base / "phase3_screening" / "stage2_fulltext" / "fulltext_review_stats.json")
    
    # Token budgets
    budget_dir = base / ".budget"
    if budget_dir.exists():
        print(f"\n💰 Token Usage:")
        total_cost = 0
        for bf in sorted(budget_dir.glob("*_budget.json")):
            with open(bf, 'r') as f:
                b = json.load(f)
            cost = b.get("total_cost_usd", 0)
            total_cost += cost
            print(f"  {b['phase']}: ${cost:.4f} / ${b.get('limit_usd', '?')}")
        print(f"  {'─'*30}")
        print(f"  Total: ${total_cost:.4f}")
    
    # Cross-phase consistency checks (#9)
    print(f"\n🔗 Cross-phase Consistency:")
    _issues = []

    # Phase 3.1 included → Phase 3.2 input
    p31_included = base / "phase3_screening" / "stage1_title_abstract" / "included_studies.json"
    p32_input    = base / "phase3_screening" / "stage1_title_abstract" / "included_studies.json"
    p32_output   = base / "phase3_screening" / "stage2_fulltext" / "included_studies.json"
    p4_input     = base / "phase3_screening" / "stage2_fulltext" / "included_studies.json"
    p4_output    = base / "phase4_extraction" / "extracted_data.json"

    def _count(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return len(data) if isinstance(data, list) else None
        except Exception:
            return None

    n31 = _count(p31_included)
    n32 = _count(p32_output)
    n4  = _count(p4_output)

    if n31 is not None and n32 is not None:
        match = "✅" if n31 >= n32 else "⚠️ "
        print(f"  {match} Phase 3.1 included ({n31}) → Phase 3.2 reviewed ({n32})")
        if n31 < n32:
            _issues.append(f"Phase 3.2 reviewed {n32} but Phase 3.1 only included {n31}")
    elif n31 is not None:
        print(f"  ⏸️  Phase 3.1: {n31} included | Phase 3.2: not yet run")
    else:
        print(f"  ⏸️  Phase 3.1 not yet complete")

    if n32 is not None and n4 is not None:
        match = "✅" if n32 >= n4 else "⚠️ "
        print(f"  {match} Phase 3.2 included ({n32}) → Phase 4 extracted ({n4})")
        if n32 < n4:
            _issues.append(f"Phase 4 extracted {n4} but Phase 3.2 only included {n32}")
    elif n32 is not None:
        print(f"  ⏸️  Phase 3.2: {n32} included | Phase 4: not yet run")

    # Pending studies warning
    try:
        ft_stats = base / "phase3_screening" / "stage2_fulltext" / "fulltext_review_stats.json"
        if ft_stats.exists():
            with open(ft_stats) as f:
                fts = json.load(f)
            pending = fts.get("pending_no_pdf", 0)
            if pending:
                print(f"  ⚠️  {pending} studies still pending (no PDF) in Phase 3.2")
    except Exception:
        pass

    if _issues:
        print(f"\n  ❌ Issues found:")
        for issue in _issues:
            print(f"     - {issue}")

    # DiskCache stats (local response deduplication)
    cache_dir = base / ".cache"
    if cache_dir.exists():
        cache_files = list(cache_dir.rglob("*.json"))
        print(f"\n📦 DiskCache (local dedup): {len(cache_files)} entries")

    # API-level prompt cache status (reads audit log)
    audit_path = base / ".audit" / "prompt_log.jsonl"
    if audit_path.exists():
        entries = []
        try:
            entries = [json.loads(l) for l in audit_path.read_text().splitlines() if l.strip()]
        except Exception:
            pass

        if entries:
            # Per-role cache stats
            from collections import defaultdict
            role_stats = defaultdict(lambda: {"calls": 0, "hits": 0})
            for e in entries:
                role = e.get("role", "unknown")
                role_stats[role]["calls"] += 1
                cr = e.get("cache_read_tokens", e.get("tokens", {}).get("cache_read_tokens", 0)) or 0
                if cr > 0:
                    role_stats[role]["hits"] += 1

            total_calls = sum(s["calls"] for s in role_stats.values())
            total_hits  = sum(s["hits"]  for s in role_stats.values())
            hit_rate    = total_hits / total_calls * 100 if total_calls else 0

            if hit_rate > 50:
                status_icon = "✅"
            elif total_hits > 0:
                status_icon = "🟡"
            else:
                status_icon = "⬜"

            print(f"\n{status_icon} API Prompt Cache:  {total_hits}/{total_calls} calls with cache read ({hit_rate:.0f}%)")
            for role, s in sorted(role_stats.items()):
                if s["calls"] > 1:
                    r = s["hits"] / s["calls"] * 100
                    icon = "✅" if r > 50 else ("🟡" if r > 0 else "  ")
                    print(f"     {icon} {role:<22} {s['hits']:>4}/{s['calls']:<6} ({r:.0f}%)")

    print()


if __name__ == "__main__":
    main()
