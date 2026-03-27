"""
PRISMA-S Search Reporting Checklist — LUMEN v2
================================================
Automated compliance check against the PRISMA-S 2021 extension
for reporting literature searches in systematic reviews.

16 items covering:
- Database/platform identification
- Multi-database searching
- Search strategy documentation
- Study registries
- Grey literature
- Limits and restrictions
- Search validation
- Peer review of search
- Documentation of total results
- Search update timing

Reference:
  Rethlefsen et al. (2021) Systematic Reviews 10:39
  https://doi.org/10.1186/s13643-020-01542-z
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


PRISMA_S_ITEMS = [
    {
        "id": 1,
        "section": "Information sources and methods",
        "item": "Name each database searched, giving the platform/system used",
        "auto_check": "databases_named",
    },
    {
        "id": 2,
        "section": "Information sources and methods",
        "item": "State whether multi-database searching was used and how deduplication was handled",
        "auto_check": "multi_db_dedup",
    },
    {
        "id": 3,
        "section": "Information sources and methods",
        "item": "Describe any study registries searched (e.g., ClinicalTrials.gov, WHO ICTRP)",
        "auto_check": "registries",
    },
    {
        "id": 4,
        "section": "Information sources and methods",
        "item": "Report any citation searching done (backward/forward)",
        "auto_check": "citation_search",
    },
    {
        "id": 5,
        "section": "Information sources and methods",
        "item": "List other methods used to identify studies (contacting experts, grey literature)",
        "auto_check": "other_methods",
    },
    {
        "id": 6,
        "section": "Search strategies",
        "item": "Present the full search strategy for ALL databases, including limits/filters",
        "auto_check": "full_strategy",
    },
    {
        "id": 7,
        "section": "Search strategies",
        "item": "Describe any search filters/hedges used (e.g., RCT filter)",
        "auto_check": "filters_described",
    },
    {
        "id": 8,
        "section": "Search strategies",
        "item": "Report language or publication date restrictions",
        "auto_check": "restrictions",
    },
    {
        "id": 9,
        "section": "Search strategies",
        "item": "List any database-specific syntax adaptations",
        "auto_check": "syntax_adaptation",
    },
    {
        "id": 10,
        "section": "Peer review",
        "item": "Report whether the search strategy was peer reviewed (e.g., PRESS guideline)",
        "auto_check": "peer_review",
    },
    {
        "id": 11,
        "section": "Managing records",
        "item": "Report the total records identified from each database",
        "auto_check": "per_db_counts",
    },
    {
        "id": 12,
        "section": "Managing records",
        "item": "Report total records after deduplication",
        "auto_check": "dedup_count",
    },
    {
        "id": 13,
        "section": "Managing records",
        "item": "Report the reference management software used",
        "auto_check": "ref_manager",
    },
    {
        "id": 14,
        "section": "Managing records",
        "item": "Describe the process for selecting studies (screening stages)",
        "auto_check": "screening_process",
    },
    {
        "id": 15,
        "section": "Timing",
        "item": "Report the date of each search or the date of the most recent search",
        "auto_check": "search_date",
    },
    {
        "id": 16,
        "section": "Timing",
        "item": "Report whether any search updates were done and their dates",
        "auto_check": "search_update",
    },
]


class PrismaSChecker:
    """Automated PRISMA-S compliance checker."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def check(self) -> dict:
        """Run all PRISMA-S checks and produce a compliance report."""
        results = []

        for item in PRISMA_S_ITEMS:
            check_fn = getattr(self, f"_check_{item['auto_check']}", None)
            if check_fn:
                status, detail = check_fn()
            else:
                status, detail = "manual", "Requires manual verification"

            results.append({
                "id": item["id"],
                "section": item["section"],
                "item": item["item"],
                "status": status,        # "pass" | "fail" | "partial" | "manual"
                "detail": detail,
            })

        passed = sum(1 for r in results if r["status"] == "pass")
        partial = sum(1 for r in results if r["status"] == "partial")
        total = len(results)

        return {
            "checklist": "PRISMA-S 2021",
            "total_items": total,
            "passed": passed,
            "partial": partial,
            "failed": sum(1 for r in results if r["status"] == "fail"),
            "manual_review": sum(1 for r in results if r["status"] == "manual"),
            "compliance_pct": round((passed + partial * 0.5) / total * 100, 1),
            "items": results,
        }

    # ── Individual Checks ─────────────────────────────

    def _check_databases_named(self):
        strategy = self._load("phase1_strategy/search_strategy.json")
        if not strategy:
            return "fail", "No search strategy found"

        dbs = strategy.get("databases", [])
        if not dbs:
            # Check alternative locations
            dbs = strategy.get("sources", [])

        if dbs and len(dbs) > 0:
            return "pass", f"Databases: {', '.join(str(d) for d in dbs[:5])}"
        return "fail", "No databases listed in search strategy"

    def _check_multi_db_dedup(self):
        dedup = self._load("phase2_search/deduplicated_studies.json")
        search = self._load("phase2_search/search_results.json")

        if dedup is not None and search is not None:
            if isinstance(search, list) and isinstance(dedup, list):
                before = len(search)
                after = len(dedup)
                return "pass", f"Multi-DB: {before} -> {after} after dedup ({before - after} removed)"
            if isinstance(search, dict) and isinstance(dedup, list):
                return "pass", f"{len(dedup)} studies after deduplication"
        if dedup is not None:
            return "partial", "Dedup results found but pre-dedup counts unclear"
        return "fail", "No deduplication data found"

    def _check_registries(self):
        strategy = self._load("phase1_strategy/search_strategy.json")
        if strategy:
            sources = str(strategy).lower()
            if any(reg in sources for reg in [
                "clinicaltrials", "ictrp", "prospero", "registry"
            ]):
                return "pass", "Registry search documented in strategy"
        return "manual", "Check if trial registries were searched"

    def _check_citation_search(self):
        return "manual", "Document any backward/forward citation searching"

    def _check_other_methods(self):
        return "manual", "Document grey literature or expert contacts"

    def _check_full_strategy(self):
        strategy = self._load("phase1_strategy/search_strategy.json")
        if not strategy:
            return "fail", "No search strategy found"

        has_query = bool(
            strategy.get("search_queries") or
            strategy.get("query_string") or
            strategy.get("boolean_query")
        )
        has_terms = bool(strategy.get("search_terms") or strategy.get("keywords"))

        if has_query:
            return "pass", "Full search queries documented"
        if has_terms:
            return "partial", "Search terms present but full query not documented"
        return "fail", "No search queries or terms found"

    def _check_filters_described(self):
        strategy = self._load("phase1_strategy/search_strategy.json")
        if strategy:
            content = json.dumps(strategy).lower()
            if any(w in content for w in ["filter", "hedge", "rct", "clinical trial"]):
                return "pass", "Filters/hedges documented"
        return "manual", "Document any search filters used"

    def _check_restrictions(self):
        strategy = self._load("phase1_strategy/search_strategy.json")
        if strategy:
            content = json.dumps(strategy).lower()
            if any(w in content for w in [
                "language", "date", "year", "restriction", "limit",
                "english", "publication type"
            ]):
                return "pass", "Restrictions documented in strategy"
            return "partial", "Strategy exists but restrictions not explicitly noted"
        return "fail", "No strategy to check restrictions"

    def _check_syntax_adaptation(self):
        strategy = self._load("phase1_strategy/search_strategy.json")
        if not strategy:
            return "fail", "No strategy"

        queries = strategy.get("search_queries", {})
        if isinstance(queries, dict) and len(queries) > 1:
            return "pass", f"Syntax adapted for {len(queries)} databases"
        return "partial", "Check if syntax was adapted per database"

    def _check_peer_review(self):
        return "manual", "Document if search was peer reviewed (PRESS guideline)"

    def _check_per_db_counts(self):
        search = self._load("phase2_search/search_results.json")
        if search and isinstance(search, dict) and search.get("per_database"):
            counts = search["per_database"]
            return "pass", f"Per-DB counts: {json.dumps(counts)}"
        if search and isinstance(search, list):
            # Check if studies have source field
            sources = set()
            for s in search:
                src = s.get("source", s.get("database", ""))
                if src:
                    sources.add(src)
            if sources:
                return "partial", f"Sources found: {', '.join(sources)}"
        return "fail", "Per-database result counts not found"

    def _check_dedup_count(self):
        dedup = self._load("phase2_search/deduplicated_studies.json")
        if dedup is not None:
            if isinstance(dedup, list):
                return "pass", f"{len(dedup)} records after deduplication"
            return "pass", "Deduplication results documented"
        return "fail", "No deduplication results found"

    def _check_ref_manager(self):
        # LUMEN itself is the reference manager
        return "pass", "LUMEN v2 pipeline (automated reference management)"

    def _check_screening_process(self):
        screening = self._file_exists("phase3_screening/screening_results.json")
        prescreen = self._file_exists("phase2_5_prescreen/prescreen_results.json")

        if screening and prescreen:
            return "pass", "Pre-screening + dual screening documented"
        if screening:
            return "pass", "Screening process documented"
        return "fail", "No screening results found"

    def _check_search_date(self):
        strategy = self._load("phase1_strategy/search_strategy.json")
        if strategy:
            content = json.dumps(strategy)
            # Look for date patterns
            import re
            dates = re.findall(r"20\d{2}[-/]\d{2}[-/]\d{2}", content)
            if dates:
                return "pass", f"Search date(s): {', '.join(dates[:3])}"

            if strategy.get("search_date") or strategy.get("date"):
                return "pass", f"Date: {strategy.get('search_date') or strategy.get('date')}"

        # Check file timestamps
        strategy_path = self.data_dir / "phase1_strategy" / "search_strategy.json"
        if strategy_path.exists():
            from datetime import datetime
            mtime = strategy_path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            return "partial", f"Strategy file date: {dt.strftime('%Y-%m-%d')} (from file mtime)"

        return "fail", "Search date not documented"

    def _check_search_update(self):
        return "manual", "Document if search was updated and when"

    # ── Helpers ───────────────────────────────────────

    def _load(self, rel_path: str):
        path = self.data_dir / rel_path
        if not path.exists():
            return None
        try:
            if path.suffix in (".yaml", ".yml"):
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _file_exists(self, rel_path: str) -> bool:
        return (self.data_dir / rel_path).exists()


# ======================================================================
# Formatting
# ======================================================================

def format_prisma_s_report(report: dict) -> str:
    """Format PRISMA-S report as human-readable text."""
    lines = [
        "=" * 65,
        "  LUMEN v2 — PRISMA-S Search Reporting Compliance",
        "=" * 65,
        "",
        f"  Compliance: {report['compliance_pct']:.0f}%  "
        f"({report['passed']}/{report['total_items']} passed, "
        f"{report['partial']} partial, "
        f"{report['manual_review']} need manual review)",
        "",
    ]

    status_icons = {
        "pass": "[+]",
        "fail": "[X]",
        "partial": "[~]",
        "manual": "[?]",
    }

    current_section = ""
    for item in report["items"]:
        if item["section"] != current_section:
            current_section = item["section"]
            lines.extend(["", f"  {current_section}:", "  " + "-" * 55])

        icon = status_icons.get(item["status"], "[ ]")
        lines.append(f"  {icon} {item['id']:>2}. {item['item']}")
        if item["detail"]:
            lines.append(f"       {item['detail']}")

    lines.extend(["", "=" * 65])
    return "\n".join(lines)
