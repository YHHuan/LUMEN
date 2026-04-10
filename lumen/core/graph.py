"""
LUMEN v3 LangGraph StateGraph definition.

Sprint 6: all placeholder nodes replaced with real agent logic.

Key conditional edges:
- assess_quality -> re_extract (critical anomaly) or proceed
- write_manuscript -> revise (fact-check contradicted) or done
"""
from __future__ import annotations

import json
import structlog

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from lumen.core.state import LumenState

logger = structlog.get_logger()


def build_graph(
    router=None,
    cost_tracker=None,
    config: dict | None = None,
) -> StateGraph:
    """Build and compile the LUMEN v3 pipeline graph.

    If router/cost_tracker/config are provided, they are injected into
    node functions via closure. Otherwise nodes expect them in state
    under _router, _cost_tracker, _config keys.
    """
    _config = config or {}

    def _get_deps(state):
        r = router or state.get("_router")
        c = cost_tracker or state.get("_cost_tracker")
        cfg = _config or state.get("_config", {})
        return r, c, cfg

    # ── Node functions ──────────────────────────────────────────────

    def pico_refine_node(state: LumenState) -> dict:
        from lumen.agents.pico_interviewer import PICOInterviewerAgent
        r, c, cfg = _get_deps(state)
        agent = PICOInterviewerAgent(router=r, cost_tracker=c, config=cfg)
        result = agent.elicit(state.get("pico"))
        return {
            "pico": result["pico"],
            "pico_completeness_score": result["completeness_score"],
            "current_phase": "pico_refinement",
        }

    def strategy_node(state: LumenState) -> dict:
        from lumen.agents.strategy_generator import StrategyGeneratorAgent
        r, c, cfg = _get_deps(state)
        agent = StrategyGeneratorAgent(router=r, cost_tracker=c, config=cfg)
        result = agent.generate(state.get("pico", {}))
        return {
            "search_strategy": result["search_strategy"],
            "screening_criteria": result["screening_criteria"],
            "current_phase": "strategy_generation",
        }

    def search_node(state: LumenState) -> dict:
        """Search PubMed (+ other databases) using generated strategy.

        If raw_results already loaded (e.g., from studies.json), skip search.
        Otherwise, execute PubMed queries from search_strategy.
        """
        if state.get("raw_results"):
            logger.info("search_node", msg="Using pre-loaded studies",
                         n=len(state["raw_results"]))
            return {
                "raw_results": state["raw_results"],
                "current_phase": "search",
            }

        strategy = state.get("search_strategy", {})
        queries = strategy.get("queries", [])
        all_results = []

        for q in queries:
            db = q.get("database", "").lower()
            query_str = q.get("query", "")
            if not query_str:
                continue
            try:
                if "pubmed" in db:
                    from lumen.tools.search.pubmed import search_pubmed
                    results = search_pubmed(query_str)
                    all_results.extend(results)
                    logger.info("search_pubmed_done", n=len(results))
                elif "openalex" in db:
                    from lumen.tools.search.openalex import search_openalex
                    results = search_openalex(query_str)
                    all_results.extend(results)
                    logger.info("search_openalex_done", n=len(results))
                else:
                    logger.warning("search_unsupported_db", db=db)
            except Exception as e:
                logger.error("search_failed", db=db, error=str(e))

        return {
            "raw_results": all_results,
            "current_phase": "search",
        }

    def dedup_node(state: LumenState) -> dict:
        """Deduplicate studies by title similarity."""
        raw = state.get("raw_results", [])
        seen_titles: set[str] = set()
        deduped = []
        for study in raw:
            title = study.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                deduped.append(study)
            elif not title:
                deduped.append(study)

        logger.info("dedup_complete", raw=len(raw), deduped=len(deduped))
        return {
            "deduplicated_studies": deduped,
            "current_phase": "deduplication",
        }

    def prescreen_wrapper(state: LumenState) -> dict:
        from lumen.agents.screening_node import prescreen_node
        result = prescreen_node(state)
        result["current_phase"] = "prescreening"
        return result

    def screen_ta_wrapper(state: LumenState) -> dict:
        from lumen.agents.screening_node import screen_ta_node
        from lumen.agents.screener import ScreenerAgent
        from lumen.agents.arbiter import ArbiterAgent
        r, c, cfg = _get_deps(state)
        # Cross-model dual screening: FAST (Gemini) + SMART (Sonnet)
        # Different providers → complementary bias profiles
        # (Oami et al. Research Synthesis Methods 2025)
        s1 = ScreenerAgent(router=r, cost_tracker=c, config=cfg, tier_override="fast")
        s2 = ScreenerAgent(router=r, cost_tracker=c, config=cfg)  # default: smart
        arb = ArbiterAgent(router=r, cost_tracker=c, config=cfg)
        result = screen_ta_node(state, s1, s2, arb)
        result["current_phase"] = "screening"
        return result

    def pdf_acquire_node(state: LumenState) -> dict:
        """Download PDFs and extract text for included studies.

        Skips studies that already have pdf_content populated.
        Uses multi-source cascade downloader (Unpaywall, PMC, etc.)
        and pdfplumber for text extraction.
        """
        included = state.get("included_studies", [])
        updated = []

        for study in included:
            if study.get("pdf_content"):
                updated.append(study)
                continue

            study = dict(study)  # shallow copy
            try:
                from lumen.tools.pdf.downloader import PDFDownloader
                from lumen.tools.pdf.reader import extract_text
                import tempfile

                with PDFDownloader(tempfile.mkdtemp()) as dl:
                    pdf_path = dl.download(study)
                    if pdf_path:
                        study["pdf_content"] = extract_text(pdf_path)
                        study["pdf_path"] = pdf_path
                        logger.info("pdf_acquired", study_id=study.get("study_id"))
                    else:
                        study["pdf_content"] = ""
                        logger.warning("pdf_not_found", study_id=study.get("study_id"))
            except ImportError:
                logger.warning("pdf_tools_not_available",
                               study_id=study.get("study_id"))
                study["pdf_content"] = ""
            except Exception as e:
                logger.error("pdf_acquire_failed",
                             study_id=study.get("study_id"), error=str(e))
                study["pdf_content"] = ""

            updated.append(study)

        return {
            "included_studies": updated,
            "current_phase": "pdf_acquisition",
        }

    def fulltext_screen_wrapper(state: LumenState) -> dict:
        from lumen.agents.fulltext_screener import FulltextScreenerAgent
        r, c, cfg = _get_deps(state)
        agent = FulltextScreenerAgent(router=r, cost_tracker=c, config=cfg)

        included = state.get("included_studies", [])
        pico = state.get("pico", {})
        criteria = state.get("screening_criteria", {})

        results = []
        final_included = []
        for study in included:
            pdf_content = study.get("pdf_content", "")
            if not pdf_content:
                # No PDF available — include by default
                results.append({
                    "study_id": study.get("study_id", "unknown"),
                    "decision": "include",
                    "reason": "No PDF available for fulltext screening",
                })
                final_included.append(study)
                continue

            result = agent.screen(study, pdf_content, pico, criteria)
            results.append(result)
            if result.get("decision") == "include":
                final_included.append(study)

        return {
            "fulltext_results": results,
            "included_studies": final_included,
            "current_phase": "fulltext_screening",
        }

    def extract_wrapper(state: LumenState) -> dict:
        from lumen.agents.extractor import ExtractorAgent
        r, c, cfg = _get_deps(state)
        agent = ExtractorAgent(router=r, cost_tracker=c, config=cfg)

        included = state.get("included_studies", [])
        pico = state.get("pico", {})
        extractions = []

        for study in included:
            pdf_content = study.get("pdf_content", "")
            result = agent.extract(study, pdf_content, pico)
            extractions.append(result)

        return {
            "extractions": extractions,
            "current_phase": "extraction",
        }

    def harmonize_wrapper(state: LumenState) -> dict:
        from lumen.agents.harmonizer import HarmonizerAgent
        r, c, cfg = _get_deps(state)
        agent = HarmonizerAgent(router=r, cost_tracker=c, config=cfg)
        result = agent.harmonize(state.get("extractions", []), state.get("pico", {}))
        return {
            "outcome_clusters": result["outcome_clusters"],
            "harmonized_data": result["harmonized_data"],
            "current_phase": "harmonization",
        }

    def statistician_wrapper(state: LumenState) -> dict:
        from lumen.agents.statistician import StatisticianAgent
        r, c, cfg = _get_deps(state)
        agent = StatisticianAgent(router=r, cost_tracker=c, config=cfg)
        result = agent.analyze(
            state.get("harmonized_data", []),
            state.get("pico", {}),
            state.get("quality_assessments"),
        )
        return {
            "analysis_plan": result.get("analysis_plan", {}),
            "statistics_results": result.get("statistics_results", {}),
            "anomaly_flags": result.get("anomaly_flags", []),
            "current_phase": "statistics",
        }

    def quality_wrapper(state: LumenState) -> dict:
        from lumen.agents.quality_node import QualityAssessorAgent
        r, c, cfg = _get_deps(state)
        agent = QualityAssessorAgent(router=r, cost_tracker=c, config=cfg)
        result = agent.assess(
            state.get("extractions", []),
            state.get("statistics_results", {}),
            state.get("pico", {}),
        )
        return {
            "quality_assessments": result,
            "current_phase": "quality_assessment",
        }

    def synthesis_wrapper(state: LumenState) -> dict:
        from lumen.agents.writer import WriterAgent
        r, c, cfg = _get_deps(state)
        agent = WriterAgent(router=r, cost_tracker=c, config=cfg)
        # Only do evidence synthesis step
        synthesis = agent._evidence_synthesis(
            state.get("statistics_results", {}),
            state.get("extractions", []),
            state.get("quality_assessments", {}),
            state.get("pico", {}),
        )
        return {
            "evidence_synthesis": synthesis,
            "current_phase": "evidence_synthesis",
        }

    def writing_wrapper(state: LumenState) -> dict:
        from lumen.agents.writer import WriterAgent
        r, c, cfg = _get_deps(state)
        agent = WriterAgent(router=r, cost_tracker=c, config=cfg)
        result = agent.write(
            state.get("statistics_results", {}),
            state.get("extractions", []),
            state.get("quality_assessments", {}),
            state.get("pico", {}),
        )
        return {
            "manuscript_sections": result["manuscript_sections"],
            "fact_check_log": result["fact_check_log"],
            "current_phase": "writing",
        }

    # ── Build graph ─────────────────────────────────────────────────

    graph = StateGraph(LumenState)

    # Phase 1
    graph.add_node("pico_refine", pico_refine_node)
    graph.add_node("generate_strategy", strategy_node)

    # Phase 2
    graph.add_node("search", search_node)
    graph.add_node("deduplicate", dedup_node)

    # Phase 3
    graph.add_node("prescreen", prescreen_wrapper)
    graph.add_node("screen_ta", screen_ta_wrapper)
    graph.add_node("acquire_pdfs", pdf_acquire_node)
    graph.add_node("screen_fulltext", fulltext_screen_wrapper)

    # Phase 4
    graph.add_node("extract", extract_wrapper)

    # Phase 4.5
    graph.add_node("harmonize", harmonize_wrapper)

    # Phase 5
    graph.add_node("plan_and_analyze", statistician_wrapper)
    graph.add_node("assess_quality", quality_wrapper)

    # Phase 6
    graph.add_node("synthesize_evidence", synthesis_wrapper)
    graph.add_node("write_manuscript", writing_wrapper)

    # === Edges ===
    graph.add_edge(START, "pico_refine")
    graph.add_edge("pico_refine", "generate_strategy")
    graph.add_edge("generate_strategy", "search")
    graph.add_edge("search", "deduplicate")
    graph.add_edge("deduplicate", "prescreen")
    graph.add_edge("prescreen", "screen_ta")
    graph.add_edge("screen_ta", "acquire_pdfs")
    graph.add_edge("acquire_pdfs", "screen_fulltext")
    graph.add_edge("screen_fulltext", "extract")
    graph.add_edge("extract", "harmonize")
    graph.add_edge("harmonize", "plan_and_analyze")
    graph.add_edge("plan_and_analyze", "assess_quality")

    # Conditional: critical anomaly -> re-extract or proceed
    graph.add_conditional_edges(
        "assess_quality",
        route_after_quality,
        {"re_extract": "extract", "proceed": "synthesize_evidence"},
    )

    graph.add_edge("synthesize_evidence", "write_manuscript")

    # Conditional: fact-check -> revise or done
    graph.add_conditional_edges(
        "write_manuscript",
        route_after_writing,
        {"revise": "write_manuscript", "done": END},
    )

    return graph.compile(checkpointer=MemorySaver())


def route_after_quality(state: LumenState) -> str:
    """Route based on critical unresolved anomaly flags."""
    flags = state.get("anomaly_flags", [])
    critical = [
        f for f in flags
        if f.get("severity") == "critical" and not f.get("resolved")
    ]
    return "re_extract" if critical else "proceed"


def route_after_writing(state: LumenState) -> str:
    """Route based on unresolved contradicted fact-check claims."""
    log = state.get("fact_check_log", [])
    contradicted = [
        c for c in log
        if c.get("verdict") == "CONTRADICTED" and not c.get("resolved")
    ]
    return "revise" if contradicted else "done"
