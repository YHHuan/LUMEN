"""
Database Query Syntax Reference & Translator — v5
====================================================
完全通用化：broad/specific 由 LLM Strategist 決定。

語法差異速查表 (2025 verified):
╔════════════════╦══════════════════════════╦═══════════════════════════╦══════════════════════════╗
║                ║ PubMed                   ║ Europe PMC                ║ Scopus (Elsevier)        ║
╠════════════════╬══════════════════════════╬═══════════════════════════╬══════════════════════════╣
║ Title only     ║ "x"[ti]                  ║ TITLE:"x"                ║ TITLE("x")               ║
║ Title+Abstract ║ "x"[tiab]                ║ (TITLE:"x" OR ABSTRACT:"x")║ TITLE-ABS-KEY("x")    ║
║ MeSH/Keyword   ║ "term"[Majr]             ║ KW:"term"                ║ KEY("term")              ║
║ Language        ║ eng[la]                  ║ LANG:"eng"               ║ LANGUAGE(english)        ║
║ Year range      ║ API mindate/maxdate      ║ FIRST_PDATE:[YYYY TO YYYY]║ PUBYEAR > N AND < N    ║
║ Source filter   ║ (不需要)                 ║ SRC:MED (只看MEDLINE)     ║ DOCTYPE(ar)             ║
║ Boolean         ║ AND, OR, NOT             ║ AND, OR, NOT             ║ AND, OR, AND NOT        ║
╚════════════════╩══════════════════════════╩═══════════════════════════╩══════════════════════════╝

Broad vs Specific (由 LLM Phase 1 標記):
  broad  → PubMed: [ti], Scopus: TITLE(), Europe PMC: TITLE:"x", WoS: TI=()
  specific → PubMed: [tiab], Scopus: TITLE-ABS-KEY(), Europe PMC: (TITLE:"x" OR ABSTRACT:"x"), WoS: TS=()
"""

import logging
from typing import List, Set

logger = logging.getLogger(__name__)


# ======================================================================
# PubMed
# ======================================================================

def build_pubmed_query(population_terms: List[str],
                       intervention_terms: List[str],
                       mesh_population: List[str] = None,
                       mesh_intervention: List[str] = None,
                       date_start: str = "2000",
                       date_end: str = "2025",
                       languages: List[str] = None,
                       study_types: List[str] = None,
                       broad_terms: Set[str] = None) -> str:
    """
    PubMed E-utilities syntax.
    - MeSH: "term"[Majr] (Major Topic)
    - Broad: "term"[ti]
    - Specific: "term"[tiab]
    """
    _broad = broad_terms or set()
    blocks = []
    
    # Population
    pop_parts = []
    if mesh_population:
        for m in mesh_population:
            # [MeSH Terms] matches any article that mentions the topic (not just
            # articles where it is the primary focus, which [Majr] would enforce).
            pop_parts.append(f'"{m}"[MeSH Terms]')
    for t in population_terms:
        field = "[ti]" if t.lower() in _broad else "[tiab]"
        pop_parts.append(f'"{t.replace(chr(34), "")}"{field}')
    if pop_parts:
        blocks.append("(" + " OR ".join(pop_parts) + ")")

    # Intervention
    int_parts = []
    if mesh_intervention:
        for m in mesh_intervention:
            int_parts.append(f'"{m}"[MeSH Terms]')
    for t in intervention_terms:
        field = "[ti]" if t.lower() in _broad else "[tiab]"
        int_parts.append(f'"{t.replace(chr(34), "")}"{field}')
    if int_parts:
        blocks.append("(" + " OR ".join(int_parts) + ")")
    
    query = " AND ".join(blocks)
    
    # Language
    if languages:
        lang_map = {"english": "eng", "chinese": "chi", "eng": "eng", "chi": "chi"}
        lang_parts = [f"{lang_map.get(l.lower(), l[:3].lower())}[la]" for l in languages]
        query += " AND (" + " OR ".join(lang_parts) + ")"
    
    # Study type
    if study_types:
        type_parts = []
        for st in study_types:
            st_l = st.lower()
            if "rct" in st_l or "random" in st_l:
                type_parts.append('randomized controlled trial[pt]')
            elif "clinical trial" in st_l:
                type_parts.append('clinical trial[pt]')
            elif "controlled" in st_l:
                type_parts.append('controlled clinical trial[pt]')
        if type_parts:
            query += " AND (" + " OR ".join(type_parts) + ")"
    
    return query


# ======================================================================
# Europe PMC
# ======================================================================

def build_europepmc_query(population_terms: List[str],
                          intervention_terms: List[str],
                          date_start: str = "2000",
                          date_end: str = "2025",
                          languages: List[str] = None,
                          mesh_population: List[str] = None,
                          mesh_intervention: List[str] = None,
                          broad_terms: Set[str] = None,
                          study_types: List[str] = None,
                          medline_only: bool = False) -> str:
    """
    Europe PMC syntax — v5 fix:
    - Broad: TITLE:"x" (title only)
    - Specific: (TITLE:"x" OR ABSTRACT:"x") — NOT free text (avoids full-text noise)
    - MeSH: KW:"x"
    - medline_only=False (default): search all of Europe PMC including PMC-only papers.
      Set True to restrict to MEDLINE (SRC:MED) for reproducibility comparisons.
    """
    _broad = broad_terms or set()
    blocks = []
    
    # Population
    pop_parts = []
    if mesh_population:
        for m in mesh_population:
            pop_parts.append(f'KW:"{m}"')
    for t in population_terms:
        clean_t = t.replace('"', '')
        if clean_t.lower() in _broad:
            pop_parts.append(f'TITLE:"{clean_t}"')
        else:
            pop_parts.append(f'(TITLE:"{clean_t}" OR ABSTRACT:"{clean_t}")')
    if pop_parts:
        blocks.append("(" + " OR ".join(pop_parts) + ")")
    
    # Intervention
    int_parts = []
    if mesh_intervention:
        for m in mesh_intervention:
            int_parts.append(f'KW:"{m}"')
    for t in intervention_terms:
        clean_t = t.replace('"', '')
        if clean_t.lower() in _broad:
            int_parts.append(f'TITLE:"{clean_t}"')
        else:
            int_parts.append(f'(TITLE:"{clean_t}" OR ABSTRACT:"{clean_t}")')
    if int_parts:
        blocks.append("(" + " OR ".join(int_parts) + ")")
    
    query = " AND ".join(blocks)

    # Restrict to MEDLINE only when requested. Default is OFF so that PMC-only
    # open-access papers (which Europe PMC's main advantage covers) are included.
    if medline_only:
        query += " AND SRC:MED"

    # Date
    if date_start and date_end:
        query += f" AND (FIRST_PDATE:[{date_start}-01-01 TO {date_end}-12-31])"
    
    # Language
    if languages:
        lang_map = {"english": "eng", "chinese": "chi", "eng": "eng", "chi": "chi"}
        lang_parts = [f'LANG:"{lang_map.get(l.lower(), l[:3].lower())}"' for l in languages]
        query += " AND (" + " OR ".join(lang_parts) + ")"
    
    return query


# ======================================================================
# Scopus
# ======================================================================

def build_scopus_query(population_terms: List[str],
                       intervention_terms: List[str],
                       date_start: str = "2000",
                       date_end: str = "2025",
                       languages: List[str] = None,
                       broad_terms: Set[str] = None,
                       study_types: List[str] = None,
                       mesh_population: List[str] = None,
                       mesh_intervention: List[str] = None) -> str:
    """
    Scopus syntax — v5:
    - Broad free-text: TITLE("x")
    - Specific free-text: TITLE-ABS-KEY("x")
    - MeSH/controlled vocab: KEY("x") — Scopus keyword field includes
      author keywords and controlled vocabulary terms
    - DOCTYPE(ar) — only journal articles
    - RCT filter if study_types includes RCT
    """
    _broad = broad_terms or set()
    blocks = []

    for terms_list, mesh_list in [
        (population_terms, mesh_population or []),
        (intervention_terms, mesh_intervention or []),
    ]:
        if not terms_list and not mesh_list:
            continue
        specific = [f'"{t}"' for t in terms_list if t.lower() not in _broad]
        broad = [f'"{t}"' for t in terms_list if t.lower() in _broad]
        parts = []
        if mesh_list:
            parts.append("KEY(" + " OR ".join(f'"{m}"' for m in mesh_list) + ")")
        if specific:
            parts.append("TITLE-ABS-KEY(" + " OR ".join(specific) + ")")
        if broad:
            parts.append("TITLE(" + " OR ".join(broad) + ")")
        if parts:
            blocks.append("(" + " OR ".join(parts) + ")")
    
    query = " AND ".join(blocks)
    
    # Date
    if date_start:
        query += f" AND PUBYEAR > {int(date_start) - 1}"
    if date_end:
        query += f" AND PUBYEAR < {int(date_end) + 1}"
    
    # Language
    if languages:
        lang_map = {"english": "english", "eng": "english", "chinese": "chinese", "chi": "chinese"}
        lang_parts = [f"LANGUAGE({lang_map.get(l.lower(), l.lower())})" for l in languages]
        query += " AND (" + " OR ".join(lang_parts) + ")"
    
    # Article type only
    query += " AND DOCTYPE(ar)"
    
    # RCT filter
    if study_types:
        has_rct = any("rct" in s.lower() or "random" in s.lower() for s in study_types)
        if has_rct:
            rct_kw = ['"randomized"', '"randomised"', '"placebo"', '"double-blind"', '"trial"', '"crossover"']
            query += " AND TITLE-ABS-KEY(" + " OR ".join(rct_kw) + ")"
    
    return query


# ======================================================================
# Web of Science
# ======================================================================

def build_wos_query(population_terms: List[str],
                    intervention_terms: List[str],
                    date_start: str = "2000",
                    date_end: str = "2025",
                    languages: List[str] = None,
                    broad_terms: Set[str] = None) -> str:
    """WoS: broad → TI=(), specific → TS=()"""
    _broad = broad_terms or set()
    blocks = []
    
    for terms_list in [population_terms, intervention_terms]:
        if not terms_list:
            continue
        specific = [f'"{t}"' for t in terms_list if t.lower() not in _broad]
        broad = [f'"{t}"' for t in terms_list if t.lower() in _broad]
        parts = []
        if specific:
            parts.append("TS=(" + " OR ".join(specific) + ")")
        if broad:
            parts.append("TI=(" + " OR ".join(broad) + ")")
        if parts:
            blocks.append("(" + " OR ".join(parts) + ")")
    
    query = " AND ".join(blocks)
    
    if date_start and date_end:
        query += f" AND PY=({date_start}-{date_end})"
    
    if languages:
        lang_map = {"english": "English", "eng": "English", "chinese": "Chinese", "chi": "Chinese"}
        lang_parts = [lang_map.get(l.lower(), l.capitalize()) for l in languages]
        query += " AND LA=(" + " OR ".join(lang_parts) + ")"
    
    return query


# ======================================================================
# CrossRef
# ======================================================================

def build_crossref_params(population_terms: List[str],
                          intervention_terms: List[str],
                          date_start: str = "2000",
                          date_end: str = "2025",
                          broad_terms: Set[str] = None) -> dict:
    """
    CrossRef: `query` param, no boolean.
    Only use SPECIFIC terms — CrossRef free text is very loose.
    """
    _broad = broad_terms or set()
    
    # Use up to 4 specific terms per PICO component (was 2 — too narrow).
    # CrossRef has no boolean syntax so we send the most specific terms as
    # keyword soup; relevance ranking handles the rest.
    pop_specific = [t for t in population_terms if t.lower() not in _broad and len(t) > 2][:4]
    int_specific = [t for t in intervention_terms if t.lower() not in _broad and len(t) > 2][:4]
    query_text = " ".join(pop_specific + int_specific)
    
    if not query_text.strip():
        query_text = " ".join((population_terms + intervention_terms)[:4])
    
    return {
        "query": query_text,
        "filter": f"from-pub-date:{date_start},until-pub-date:{date_end},type:journal-article",
        "rows": 200,
        "sort": "relevance",
        "order": "desc",
    }


# ======================================================================
# Cochrane Library (CDSR/CENTRAL)
# ======================================================================

def build_cochrane_query(population_terms: List[str],
                         intervention_terms: List[str],
                         mesh_population: List[str] = None,
                         mesh_intervention: List[str] = None,
                         date_start: str = "2000",
                         date_end: str = "2025",
                         broad_terms: Set[str] = None) -> str:
    """
    Cochrane Library Advanced Search strategy.

    Syntax:
    - MeSH descriptor: [mh "Term"]
    - Title/Abstract (specific): "term":ti,ab
    - Title only (broad): "term":ti
    - Boolean: AND, OR
    - Date filter: applied via the Cochrane Library interface Year dropdown

    Returns a numbered search block suitable for the Cochrane Search Manager.
    """
    _broad = broad_terms or set()
    lines: list[str] = []
    n = 1

    def _add(parts: list[str]) -> int | None:
        if not parts:
            return None
        nonlocal n
        lines.append(f"#{n}  " + " OR ".join(parts))
        result = n
        n += 1
        return result

    # Population
    pop_mesh_line = _add([f'[mh "{m}"]' for m in (mesh_population or [])])
    pop_ft_line   = _add([
        f'"{t}"' + (":ti" if t.lower() in _broad else ":ti,ab")
        for t in population_terms
    ])
    pop_refs = " OR ".join(f"#{x}" for x in [pop_mesh_line, pop_ft_line] if x)
    pop_combined  = _add([pop_refs]) if pop_refs else None

    # Intervention
    int_mesh_line = _add([f'[mh "{m}"]' for m in (mesh_intervention or [])])
    int_ft_line   = _add([
        f'"{t}"' + (":ti" if t.lower() in _broad else ":ti,ab")
        for t in intervention_terms
    ])
    int_refs = " OR ".join(f"#{x}" for x in [int_mesh_line, int_ft_line] if x)
    int_combined  = _add([int_refs]) if int_refs else None

    # Final AND combination
    final = " AND ".join(f"#{x}" for x in [pop_combined, int_combined] if x)
    if final:
        lines.append(f"#{n}  {final}   [FINAL — use this result set]")

    header = (
        "Cochrane Library (CDSR / CENTRAL) — Search Strategy\n"
        "======================================================\n"
        "Interface: https://www.cochranelibrary.com/advanced-search\n"
        f"Date range filter: {date_start}–{date_end} (set in Year dropdown)\n\n"
        "NOTE: MeSH terms were validated for PubMed and should work in Cochrane\n"
        "      (Cochrane uses MeSH for indexing). Verify any unusual headings via\n"
        "      Cochrane's MeSH on Demand: https://meshb.nlm.nih.gov/search\n\n"
    )
    footer = (
        "\n\nHow to use:\n"
        "1. Go to https://www.cochranelibrary.com/advanced-search\n"
        "2. Paste each #N line one at a time into the search bar and click 'Add'\n"
        "   (the interface builds a numbered history list automatically)\n"
        "3. Apply Year filter: Publication Year from {start} to {end}\n"
        "4. Run the final line (highest #N) and review results\n"
        "5. Export: Export → Format: RIS → save as cochrane_search.ris\n"
        "6. Place in: data/<project>/phase2_search/raw/cochrane_search.ris\n"
        "7. Run: python scripts/run_phase2.py --deduplicate"
    ).format(start=date_start, end=date_end)

    return header + "\n".join(lines) + footer


# ======================================================================
# Embase via Ovid
# ======================================================================

def build_embase_ovid_query(population_terms: List[str],
                            intervention_terms: List[str],
                            mesh_population: List[str] = None,
                            mesh_intervention: List[str] = None,
                            date_start: str = "2000",
                            date_end: str = "2025",
                            broad_terms: Set[str] = None) -> str:
    """
    Embase search strategy formatted for Ovid interface.

    Syntax:
    - EMTREE (exploded): exp <term>/
    - Title/Abstract (specific): (term$ OR term).ti,ab.
    - Title only (broad): term$.ti.
    - Wildcard: $ (truncation)
    - Boolean: AND, OR
    - Date limit: applied at end with 'limit X to yr="YYYY-YYYY" and article'

    IMPORTANT: Embase uses EMTREE (not MeSH). The MeSH terms provided here
    are approximate EMTREE mappings. Verify at:
      https://www.embase.com/search/quick → Emtree Browser
    Common differences:
      MeSH "Mild Cognitive Impairment" → EMTREE "mild cognitive impairment"
      MeSH "Transcranial Magnetic Stimulation" → EMTREE "transcranial magnetic stimulation"
    """
    _broad = broad_terms or set()
    lines: list[str] = []
    n = 1

    def _add(parts: list[str], comment: str = "") -> int | None:
        if not parts:
            return None
        nonlocal n
        suffix = f"  /* {comment} */" if comment else ""
        lines.append(f"{n}. " + " OR ".join(parts) + suffix)
        result = n
        n += 1
        return result

    def _trunc(term: str) -> str:
        """Add $ truncation to the last word for Ovid."""
        words = term.strip().split()
        if not words:
            return term
        # Don't truncate short words or abbreviations (all-caps)
        if len(words[-1]) <= 3 or words[-1].isupper():
            return term
        words[-1] = words[-1].rstrip("$") + "$"
        return " ".join(words)

    # Population EMTREE (use MeSH as approximate mapping)
    pop_emtree = _add(
        [f"exp {m}/" for m in (mesh_population or [])],
        comment="EMTREE — verify equivalents before running",
    )
    # Population free text
    pop_ft = _add([
        f"({'$ OR '.join(_trunc(t).split()) if False else _trunc(t)}).ti." if t.lower() in _broad
        else f"({_trunc(t)}).ti,ab."
        for t in population_terms
    ])
    pop_combined = _add(
        [f"{x} OR {y}" for x, y in [(pop_emtree, pop_ft)] if x and y]
        or ([f"{pop_emtree}"] if pop_emtree else [f"{pop_ft}"] if pop_ft else [])
    )

    # Intervention EMTREE
    int_emtree = _add(
        [f"exp {m}/" for m in (mesh_intervention or [])],
        comment="EMTREE — verify equivalents before running",
    )
    int_ft = _add([
        f"({_trunc(t)}).ti." if t.lower() in _broad
        else f"({_trunc(t)}).ti,ab."
        for t in intervention_terms
    ])
    int_combined = _add(
        [f"{x} OR {y}" for x, y in [(int_emtree, int_ft)] if x and y]
        or ([f"{int_emtree}"] if int_emtree else [f"{int_ft}"] if int_ft else [])
    )

    # Final AND + date limit
    final_refs = [str(x) for x in [pop_combined, int_combined] if x]
    if final_refs:
        final_line = " AND ".join(final_refs)
        lines.append(f"{n}. {final_line}")
        limit_n = n + 1
        lines.append(
            f"{limit_n}. limit {n} to (yr=\"{date_start} - {date_end}\" and article)"
            "   [FINAL — use this result set]"
        )

    header = (
        "Embase via Ovid — Search Strategy\n"
        "====================================\n"
        "Interface: Ovid (https://ovidsp.ovid.com)\n"
        f"Date range: {date_start}–{date_end} (applied as limit in final step)\n\n"
        "⚠ IMPORTANT: Embase uses EMTREE controlled vocabulary, NOT MeSH.\n"
        "  Lines marked '/* EMTREE */' use MeSH terms as approximate mappings.\n"
        "  Verify and replace with correct EMTREE terms before running:\n"
        "  → Emtree browser: https://www.embase.com/search/quick (click Emtree)\n\n"
    )
    footer = (
        "\n\nHow to use:\n"
        "1. Log in to Ovid Embase\n"
        "2. Enter each numbered line sequentially in the search box\n"
        "3. The interface builds a history list automatically\n"
        "4. Run the final line (highest number) to get the combined result\n"
        "5. Export: Export → RIS format → save as embase_export.ris\n"
        "6. Place in: data/<project>/phase2_search/raw/embase_export.ris\n"
        "7. Run: python scripts/run_phase2.py --deduplicate"
    )

    return header + "\n".join(lines) + footer


# ======================================================================
# Multi-database query generator
# ======================================================================

def generate_all_queries(population_terms: List[str],
                         intervention_terms: List[str],
                         mesh_population: List[str] = None,
                         mesh_intervention: List[str] = None,
                         date_start: str = "2000",
                         date_end: str = "2025",
                         languages: List[str] = None,
                         study_types: List[str] = None,
                         broad_terms: Set[str] = None) -> dict:
    """Generate queries for all databases. broad_terms from LLM Strategist."""
    _broad = broad_terms or set()
    return {
        "pubmed": build_pubmed_query(
            population_terms, intervention_terms,
            mesh_population, mesh_intervention,
            date_start, date_end, languages, study_types, _broad,
        ),
        "europepmc": build_europepmc_query(
            population_terms, intervention_terms,
            date_start, date_end, languages,
            mesh_population, mesh_intervention, _broad, study_types,
            medline_only=False,
        ),
        "scopus": build_scopus_query(
            population_terms, intervention_terms,
            date_start, date_end, languages, _broad, study_types,
            mesh_population=mesh_population,
            mesh_intervention=mesh_intervention,
        ),
        "wos": build_wos_query(
            population_terms, intervention_terms,
            date_start, date_end, languages, _broad,
        ),
        "crossref": build_crossref_params(
            population_terms, intervention_terms,
            date_start, date_end, _broad,
        ),
        "cochrane": build_cochrane_query(
            population_terms, intervention_terms,
            mesh_population, mesh_intervention,
            date_start, date_end, _broad,
        ),
        "embase_ovid": build_embase_ovid_query(
            population_terms, intervention_terms,
            mesh_population, mesh_intervention,
            date_start, date_end, _broad,
        ),
    }
