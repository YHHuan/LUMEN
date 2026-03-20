#!/usr/bin/env python3
"""
Phase 6: Manuscript Writing — v5
===================================
  python scripts/run_phase6.py                        # Full pipeline
  python scripts/run_phase6.py --section discussion   # Single section
  python scripts/run_phase6.py --validate-only        # Only validate citations
  python scripts/run_phase6.py --sections intro,discussion  # Multiple sections

v5 improvements:
  - Rich context building: all analyses, study details, intervention breakdown
  - Deep discussion/introduction prompts with structured subsections
  - Citation cross-validation against study abstracts
  - [CITATION NEEDED] marker collection and reference suggestion
"""

import sys, json, logging, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.langfuse_client import log_phase_start, log_phase_end
from src.agents.writer import WriterAgent, CitationGuardianAgent
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
#  Context Building — feed ALL data to prompts
# ═══════════════════════════════════════════

def build_context(dm: DataManager) -> dict:
    """Build comprehensive context from all previous phases."""
    import yaml

    pico = dm.load("input", "pico.yaml")
    strategy = dm.load("phase1_strategy", "search_strategy.json") \
        if dm.exists("phase1_strategy", "search_strategy.json") else {}
    criteria = dm.load("phase1_strategy", "screening_criteria.json") \
        if dm.exists("phase1_strategy", "screening_criteria.json") else {}
    prisma = dm.load("phase3_screening", "prisma_flow.json") \
        if dm.exists("phase3_screening", "prisma_flow.json") else {}
    extracted = dm.load("phase4_extraction", "extracted_data.json") \
        if dm.exists("phase4_extraction", "extracted_data.json") else []
    rob = dm.load("phase4_extraction", "risk_of_bias.json") \
        if dm.exists("phase4_extraction", "risk_of_bias.json") else []
    stats = dm.load("phase5_analysis", "statistical_results.json") \
        if dm.exists("phase5_analysis", "statistical_results.json") else {}

    # ── Study characteristics (deep) ──
    study_chars = _build_study_characteristics(extracted)

    # ── All analysis results (structured) ──
    all_analyses = _build_all_analyses(stats)

    # ── Analysis-specific study details ──
    analysis_study_details = _build_analysis_study_details(stats, extracted)

    # ── Study abstracts for citation validation ──
    # Abstracts live in screening results, not extraction output
    stage1_studies = dm.load("phase3_screening", "included_studies.json",
                              subfolder="stage1_title_abstract") \
        if dm.exists("phase3_screening", "included_studies.json",
                      subfolder="stage1_title_abstract") else []
    stage2_studies = dm.load("phase3_screening", "included_studies.json",
                              subfolder="stage2_fulltext") \
        if dm.exists("phase3_screening", "included_studies.json",
                      subfolder="stage2_fulltext") else []
    screening_studies = stage2_studies or stage1_studies

    study_abstracts = {}
    study_titles = {}
    for s in screening_studies:
        sid = s.get("study_id", "")
        if s.get("abstract"):
            study_abstracts[sid] = s["abstract"]
        if s.get("title"):
            study_titles[sid] = s["title"]

    # ── Computable count ──
    n_computable = sum(1 for s in extracted if _has_computable_data(s))

    # ── PICO summary ──
    pico_data = pico.get("pico", pico)
    pico_summary = (
        f"Population: {pico_data.get('population', '')[:200]}\n"
        f"Intervention: {pico_data.get('intervention', '')[:200]}\n"
        f"Comparison: {pico_data.get('comparison', '')[:200]}"
    )

    return {
        "pico_summary": pico_summary,
        "pico_raw": pico,
        "search_strategy": strategy,
        "inclusion_criteria": criteria.get("inclusion_criteria", []),
        "exclusion_criteria": criteria.get("exclusion_criteria", []),
        "databases": list(prisma.get("identification", {}).keys()),
        "prisma_flow": prisma,
        "n_studies": len(extracted),
        "n_computable": n_computable,
        "study_chars": study_chars,
        "all_analyses": all_analyses,
        "analysis_study_details": analysis_study_details,
        "statistical_results": stats,
        "primary_finding": _build_primary_finding(all_analyses),
        "rob_summary": _build_rob_summary(rob),
        "subgroups": pico.get("subgroup_analyses", []),
        "study_abstracts": study_abstracts,
        "included_studies": extracted,
    }


def _normalize_intervention_type(raw_type: str) -> str:
    """
    Generic intervention type normalization.
    Collapses verbose LLM-generated descriptions into short canonical labels.
    Strategy: lowercase, strip common padding words, collapse known abbreviations.
    """
    if not raw_type or raw_type == "NR":
        return "NR"
    s = raw_type.strip()

    # Step 1: Check if it's already a short abbreviation (≤15 chars)
    if len(s) <= 15:
        return s

    # Step 2: If it contains " combined with " / " + " / " and ", split and shorten each part
    for delim in [" combined with ", " plus ", " + ", " and "]:
        if delim in s.lower():
            idx = s.lower().index(delim)
            left = _shorten_to_core(s[:idx].strip())
            right = _shorten_to_core(s[idx + len(delim):].strip())
            return f"{left} + {right}"

    # Step 3: Shorten long single-term descriptions
    return _shorten_to_core(s)


def _shorten_to_core(s: str) -> str:
    """Extract core term from verbose description."""
    if not s:
        return s
    # If already short, keep it
    if len(s) <= 15:
        return s
    # Take up to 4 meaningful words, skip leading articles/adjectives
    skip_words = {"the", "a", "an", "with", "using", "based", "standard", "conventional"}
    words = [w for w in s.split() if w.lower() not in skip_words]
    return " ".join(words[:4]).rstrip(",.:;")


def _normalize_short_label(raw_label: str, max_words: int = 4) -> str:
    """
    Generic label normalization: collapse verbose descriptions into short labels.
    Groups exact duplicates that differ only in casing/spacing.
    """
    if not raw_label or raw_label == "NR":
        return "NR"
    # Normalize whitespace and strip
    s = " ".join(raw_label.split()).strip().rstrip(".")
    # If already short, just title-case
    words = s.split()
    if len(words) <= max_words:
        return s
    # Truncate to max_words
    return " ".join(words[:max_words]).rstrip(",.:;")


def _build_study_characteristics(extracted):
    """Comprehensive study characteristics summary."""
    n = len(extracted)
    total_n = 0
    intervention_types = Counter()
    population_types = Counter()
    study_designs = Counter()
    targets = Counter()
    session_range = []

    for d in extracted:
        chars = d.get("characteristics", {})
        total_n += chars.get("n_total", 0) or 0

        design = chars.get("design") or "NR"
        study_designs[design] += 1

        interv = d.get("intervention", {})
        itype = interv.get("type") or "NR"
        # Normalize common variants (collapse verbose LLM descriptions)
        itype_normalized = _normalize_intervention_type(itype)
        intervention_types[itype_normalized] += 1

        target = interv.get("target") or "NR"
        targets[target] += 1

        sessions = interv.get("sessions_total")
        if sessions:
            try:
                session_range.append(int(sessions))
            except (ValueError, TypeError):
                pass

        pop = d.get("population", {})
        diag = pop.get("diagnosis") or "NR"
        # Normalize verbose LLM descriptions into short labels
        diag_normalized = _normalize_short_label(diag, max_words=4)
        population_types[diag_normalized] += 1

    return {
        "n_studies": n,
        "total_participants": total_n,
        "study_designs": dict(study_designs.most_common()),
        "intervention_types": dict(intervention_types.most_common()),
        "population_types": dict(population_types.most_common()),
        "stimulation_targets": dict(targets.most_common(10)),
        "session_range": {
            "min": min(session_range) if session_range else None,
            "max": max(session_range) if session_range else None,
            "median": sorted(session_range)[len(session_range)//2] if session_range else None,
        },
    }


def _build_all_analyses(stats):
    """Structure all meta-analysis results into a clean summary."""
    if not stats:
        return {}

    analyses = {}
    # stats may have different structures; handle the Phase 5 output format
    for key, value in stats.items():
        if not isinstance(value, dict):
            continue
        pa = value.get("primary_analysis", value)
        if "pooled_effect" not in pa:
            continue

        het = pa.get("heterogeneity", {})
        analyses[key] = {
            "k": pa.get("k", pa.get("n_studies", "?")),
            "pooled_g": pa.get("pooled_effect"),
            "ci_lower": pa.get("ci_lower"),
            "ci_upper": pa.get("ci_upper"),
            "p_value": pa.get("p_value"),
            "I2": het.get("I2", "N/A"),
            "tau2": het.get("tau2", "N/A"),
            "significant": pa.get("p_value", 1) < 0.05,
        }

    return analyses


def _build_analysis_study_details(stats, extracted):
    """For each analysis, list the contributing studies with their characteristics."""
    details = {}
    for key, value in stats.items():
        if not isinstance(value, dict):
            continue
        studies_in = value.get("studies", [])
        if not studies_in:
            continue

        study_info = []
        for sid in studies_in:
            s = next((x for x in extracted if x.get("study_id") == sid), None)
            if not s:
                continue
            interv = s.get("intervention", {})
            pop = s.get("population", {})
            chars = s.get("characteristics", {})
            study_info.append({
                "study_id": sid,
                "modality": interv.get("type", "?"),
                "target": interv.get("target", "?"),
                "sessions": interv.get("sessions_total", "?"),
                "n_total": chars.get("n_total", "?"),
                "population": pop.get("diagnosis", "?"),
            })
        if study_info:
            details[key] = study_info

    return details


def _build_rob_summary(rob):
    """Summarize Risk of Bias."""
    if not rob:
        return {}
    overall = Counter(r.get("overall_rob", "unknown") for r in rob)
    return {
        "n_assessed": len(rob),
        "overall_distribution": dict(overall.most_common()),
    }


def _build_primary_finding(all_analyses):
    """One-sentence summary of primary finding."""
    sig = [k for k, v in all_analyses.items() if v.get("significant")]
    if sig:
        best = sig[0]
        a = all_analyses[best]
        return (f"{best}: Hedges' g = {a['pooled_g']:.3f}, "
                f"95% CI [{a['ci_lower']:.3f}, {a['ci_upper']:.3f}], "
                f"p = {a['p_value']:.4f}, k = {a['k']}")
    return "No significant primary outcomes"


def _has_computable_data(study):
    """Check if a study has computable outcome data."""
    for key, od in study.get("outcomes", {}).items():
        for suffix in ["", "_post", "_change"]:
            if od.get(f"intervention_mean{suffix}") is not None:
                if od.get(f"intervention_sd{suffix}") is not None:
                    if od.get("intervention_n") is not None:
                        return True
        if od.get("mean_difference") is not None and od.get("md_95ci_lower") is not None:
            return True
        if od.get("smd") is not None:
            return True
    return False


# ═══════════════════════════════════════════
#  Writing Pipeline
# ═══════════════════════════════════════════

def write_manuscript(dm: DataManager, budget: TokenBudget,
                     sections_to_write: list = None):
    """Main writing pipeline."""
    ctx = build_context(dm)
    writer = WriterAgent(budget=budget)

    all_sections = ["title", "abstract", "introduction", "methods", "results", "discussion"]
    sections = sections_to_write or all_sections

    manuscript_parts = {}
    for sec in sections:
        logger.info(f"✍️  Writing: {sec}")
        try:
            text = writer.write_section(sec, ctx)
            manuscript_parts[sec] = text
            dm.save("phase6_manuscript", f"{sec}.md", text, subfolder="drafts")
            logger.info(f"✅ Saved: {sec} ({len(text)} chars)")
        except Exception as e:
            logger.error(f"❌ Failed to write {sec}: {e}")
            manuscript_parts[sec] = f"[ERROR: {e}]"

    # Combine into full manuscript
    if len(sections) > 1:
        full_text = "\n\n---\n\n".join([
            f"# {sec.upper()}\n\n{text}"
            for sec, text in manuscript_parts.items()
        ])
        dm.save("phase6_manuscript", "manuscript_draft.md", full_text,
                subfolder="drafts")
        logger.info(f"📄 Full manuscript saved: manuscript_draft.md")

    return manuscript_parts


# ═══════════════════════════════════════════
#  Citation Validation Pipeline
# ═══════════════════════════════════════════

def validate_citations(dm: DataManager, budget: TokenBudget):
    """Run citation cross-validation on manuscript drafts."""
    guardian = CitationGuardianAgent(budget=budget)
    ctx = build_context(dm)

    # Load all manuscript text
    drafts_dir = Path(get_data_dir()) / "phase6_manuscript" / "drafts"
    full_text = ""
    for f in sorted(drafts_dir.glob("*.md")):
        full_text += f.read_text(encoding="utf-8") + "\n"
    if not full_text:
        for f in sorted(drafts_dir.glob("*.txt")):
            full_text += f.read_text(encoding="utf-8") + "\n"

    if not full_text:
        logger.warning("No manuscript drafts found!")
        return

    # ── Step 1: Validate internal citations ──
    logger.info("🔍 Validating internal citations against study abstracts...")
    internal_results = guardian.validate_internal_citations(
        full_text, ctx["study_abstracts"]
    )

    n_accurate = sum(1 for r in internal_results if r.get("accurate", True))
    n_issues = sum(1 for r in internal_results if not r.get("accurate", True))
    logger.info(f"  Internal citations: {n_accurate} accurate, {n_issues} issues")

    dm.save("phase6_manuscript", "citation_validation_internal.json",
            internal_results)

    # ── Step 2: Collect [CITATION NEEDED] markers ──
    logger.info("📋 Collecting [CITATION NEEDED] markers...")
    markers = guardian.collect_citation_needed(full_text)
    logger.info(f"  Found {len(markers)} [CITATION NEEDED] markers")

    if markers:
        # ── Step 3: Suggest references ──
        logger.info("💡 Suggesting references...")
        suggestions = guardian.suggest_references(markers, ctx["included_studies"])
        dm.save("phase6_manuscript", "citation_suggestions.json", suggestions)

        internal_count = sum(1 for s in suggestions if s.get("type") == "internal")
        external_count = sum(1 for s in suggestions if s.get("type") == "external")
        logger.info(f"  Suggestions: {internal_count} internal, {external_count} external")

    # ── Summary report ──
    report = {
        "internal_citations_checked": len(internal_results),
        "internal_accurate": n_accurate,
        "internal_issues": n_issues,
        "citation_needed_markers": len(markers),
        "issues": [r for r in internal_results if not r.get("accurate", True)],
        "markers": markers,
    }
    dm.save("phase6_manuscript", "citation_report.json", report)

    # Print summary
    print("\n" + "=" * 60)
    print("📝 Citation Validation Report")
    print("=" * 60)
    print(f"  Internal citations checked: {len(internal_results)}")
    print(f"  Accurate: {n_accurate}")
    print(f"  Issues found: {n_issues}")
    if n_issues > 0:
        print(f"\n  ⚠️ Issues:")
        for r in internal_results:
            if not r.get("accurate", True):
                print(f"    {r.get('study_id', '?')}: {r.get('issue', '?')[:80]}")

    print(f"\n  [CITATION NEEDED] markers: {len(markers)}")
    if markers:
        print(f"  Types needed:")
        for m in markers[:10]:
            print(f"    → {m['topic'][:60]}")
        if len(markers) > 10:
            print(f"    ... and {len(markers)-10} more")


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Manuscript Writing v5")
    parser.add_argument("--section", type=str, help="Write single section")
    parser.add_argument("--sections", type=str,
                        help="Comma-separated sections (e.g. introduction,discussion)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate citations")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip citation validation")
    args = parser.parse_args()

    select_project()
    dm = DataManager()
    budget = TokenBudget(phase="phase6", limit_usd=15.0)

    # Langfuse: phase start stamp
    import yaml
    from pathlib import Path as _Path
    _models_cfg = yaml.safe_load((_Path(__file__).parent.parent / "config/models.yaml").read_text())
    _lf_span = log_phase_start("phase6_manuscript", {
        "writer_model":          _models_cfg["models"]["writer"]["model_id"],
        "citation_guardian_model": _models_cfg["models"]["citation_guardian"]["model_id"],
        "budget_usd":            15.0,
        "sections_requested":    args.section or args.sections or "all",
        "validate_only":         args.validate_only,
        "skip_validation":       args.skip_validation,
    })

    if args.validate_only:
        validate_citations(dm, budget)
    else:
        # Determine sections to write
        sections = None
        if args.section:
            sections = [args.section]
        elif args.sections:
            sections = [s.strip() for s in args.sections.split(",")]

        # Write
        parts = write_manuscript(dm, budget, sections_to_write=sections)

        # Validate
        if not args.skip_validation:
            logger.info("\n🔍 Running citation validation...")
            validate_citations(dm, budget)

    print(f"\n💰 Token budget: {json.dumps(budget.summary(), indent=2)}")
    print("\n✅ Phase 6 Complete!")
    print("Check outputs in: data/phase6_manuscript/drafts/")

    # Langfuse: phase end stamp
    _bsum = budget.summary()
    log_phase_end(_lf_span, "phase6_manuscript", {
        "total_cost_usd":    _bsum.get("total_cost_usd", 0),
        "cache_savings_usd": _bsum.get("cache_savings_usd", 0),
        "sections_written":  args.section or args.sections or "all",
    })


if __name__ == "__main__":
    main()
