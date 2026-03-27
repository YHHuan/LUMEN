"""
Phase 4: Data Extraction — LUMEN v2
======================================
PDF -> structured data extraction with:
- gmft table pre-parsing
- Vector-indexed retrieval for smart context
- 3-pass self-consistency check
- Claim-grounded evidence spans
- NMA multi-arm extraction support

Usage:
    python scripts/run_phase4.py                  # Full extraction (pairwise)
    python scripts/run_phase4.py --nma            # NMA multi-arm extraction
    python scripts/run_phase4.py --validate-only  # Validate without re-extracting
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.cache import TokenBudget
from src.utils.pdf_decomposer import get_or_decompose, format_segments_for_llm
from src.utils.vector_index import DocumentVectorIndex, build_field_queries
from src.utils.extraction_context import build_extraction_context
from src.utils.deduplication import generate_canonical_citation
from src.config import cfg
from src.agents.extractor import ExtractorAgent, validate_evidence_spans

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--nma", action="store_true",
                        help="NMA mode: extract multi-arm treatment data")
    parser.add_argument("--incremental", action="store_true",
                        help="Skip already-extracted study_ids, append new results, accumulate cost")
    args = parser.parse_args()

    select_project()
    dm = DataManager()

    # Load included studies (auto-selects best available: 3.3 > 3.2 > 3.1)
    studies = dm.load_best_included()

    logger.info(f"Extracting data from {len(studies)} studies")

    # Settings
    ext_settings = cfg.extraction_settings
    use_vector = ext_settings.get("use_vector_retrieval", True)
    use_tables = ext_settings.get("use_table_transformer", True)
    n_passes = cfg.v2.get("batch_settings", {}).get("extraction_passes", 3)
    cache_dir = str(Path(get_data_dir()) / ".cache" / "decomposed")

    # Budget — incremental mode preserves prior cost data
    budget = TokenBudget("phase4", limit_usd=cfg.budget("phase4"),
                         reset=not args.incremental)

    # Agent
    extractor = ExtractorAgent(budget=budget)

    # Load PICO for field queries
    pico = dm.load_if_exists("input", "pico.yaml", default={})

    # Extraction schema — NMA mode uses multi-arm schema
    nma_mode = args.nma or cfg.v2.get("nma", {}).get("enabled", False)
    if nma_mode:
        import yaml
        nma_prompt_path = Path(__file__).parent.parent / "config" / "prompts" / "extractor_nma.yaml"
        with open(nma_prompt_path, encoding="utf-8") as f:
            nma_prompt_cfg = yaml.safe_load(f)
        extraction_schema = nma_prompt_cfg["extraction_schema"]
        # Override extractor system prompt for NMA
        extractor._prompt_config["system_prompt"] = nma_prompt_cfg["system_prompt"]
        logger.info("NMA mode: using multi-arm extraction schema")
    else:
        extraction_schema = extractor._prompt_config.get("extraction_schema", {
            "study_design": "string",
            "population_description": "string",
            "total_n": "integer",
            "intervention_description": "string",
            "control_description": "string",
            "outcomes": [{
                "measure": "string",
                "timepoint": "string",
                "intervention_group": {
                    "mean": "float", "sd": "float", "n": "integer",
                    "evidence_span": "string", "evidence_page": "integer",
                    "evidence_type": "string", "confidence": "string",
                },
                "control_group": {
                    "mean": "float", "sd": "float", "n": "integer",
                    "evidence_span": "string", "evidence_page": "integer",
                    "evidence_type": "string", "confidence": "string",
                },
            }],
        })

    # Incremental: load existing results to skip and merge later
    existing_extracted = []
    existing_validation = []
    existing_ids = set()
    if args.incremental:
        existing_extracted = dm.load_if_exists(
            "phase4_extraction", "extracted_data.json") or []
        existing_validation = dm.load_if_exists(
            "phase4_extraction", "evidence_validation.json") or []
        existing_ids = {s.get("study_id") for s in existing_extracted}
        logger.info(f"Incremental mode: {len(existing_ids)} already extracted, skipping")

    extracted_data = []
    validation_results = []

    from tqdm import tqdm
    for study in tqdm(studies, desc="Extracting"):
        study_id = study.get("study_id", "unknown")
        pdf_path = study.get("pdf_path", "")

        if args.incremental and study_id in existing_ids:
            continue

        if not pdf_path or not Path(pdf_path).exists():
            logger.warning(f"No PDF for {study_id}, skipping")
            continue

        try:
            # Step 1: Decompose PDF
            segments = get_or_decompose(pdf_path, cache_dir)

            # Step 2: Build context
            if use_vector and len(segments) > 20:
                vector_index = DocumentVectorIndex()
                vector_index.build_index(segments)
                context = build_extraction_context(
                    segments, vector_index,
                    extraction_guidance=pico,
                )
            else:
                context = format_segments_for_llm(segments)

            # Step 3: Extract with self-consistency
            if not args.validate_only:
                result = extractor.extract_with_consistency(
                    context, extraction_schema,
                    study_id=study_id,
                    n_passes=n_passes,
                )
            else:
                # Load existing extraction
                existing = dm.load_if_exists("phase4_extraction", "extracted_data.json")
                if existing:
                    result = next(
                        (s for s in existing if s.get("study_id") == study_id),
                        {}
                    )
                else:
                    continue

            # Step 4: Validate evidence spans
            validation = validate_evidence_spans(result, segments)

            # Add metadata
            result["study_id"] = study_id
            result["canonical_citation"] = generate_canonical_citation(study)
            result["title"] = study.get("title", "")
            result["_evidence_validation"] = validation

            extracted_data.append(result)
            validation_results.append({
                "study_id": study_id,
                "validation": validation,
            })

        except Exception as e:
            logger.error(f"Extraction failed for {study_id}: {e}")
            continue

    # Merge with existing if incremental
    if args.incremental and existing_extracted:
        new_ids = {s.get("study_id") for s in extracted_data}
        merged = [s for s in existing_extracted if s.get("study_id") not in new_ids]
        merged.extend(extracted_data)
        extracted_data = merged

        existing_val_ids = {v.get("study_id") for v in validation_results}
        merged_val = [v for v in existing_validation
                      if v.get("study_id") not in existing_val_ids]
        merged_val.extend(validation_results)
        validation_results = merged_val

        logger.info(f"Incremental merge: {len(existing_ids)} prior + "
                     f"{len(new_ids)} new = {len(extracted_data)} total")

    # Save results
    dm.save("phase4_extraction", "extracted_data.json", extracted_data)
    dm.save("phase4_extraction", "evidence_validation.json", validation_results)

    # Risk of bias
    rob_data = [{
        "study_id": s.get("study_id"),
        "risk_of_bias": s.get("risk_of_bias", {}),
    } for s in extracted_data]
    dm.save("phase4_extraction", "risk_of_bias.json", rob_data)

    # NMA: generate contrast-level data from multi-arm extractions
    if nma_mode:
        from src.utils.nma import prepare_nma_data, validate_network
        contrasts = prepare_nma_data(extracted_data)
        validation = validate_network(contrasts)
        dm.save("phase4_extraction", "nma_contrasts.json", contrasts)
        dm.save("phase4_extraction", "nma_network_validation.json", validation)

        treatments = validation.get("treatments", [])
        logger.info(
            f"NMA contrasts: {len(contrasts)} from {validation.get('n_studies', 0)} studies, "
            f"{len(treatments)} treatments: {', '.join(treatments)}"
        )
        if not validation.get("valid"):
            for err in validation.get("errors", []):
                logger.error(f"NMA validation error: {err}")

    print("\n" + "=" * 50)
    print("  Phase 4 Extraction Complete")
    print("=" * 50)
    print(f"  Studies extracted: {len(extracted_data)}")
    if args.incremental and existing_ids:
        print(f"  Prior (kept):      {len(existing_ids)}")
        print(f"  New (this run):    {len(extracted_data) - len(existing_ids)}")
    if nma_mode:
        print(f"  NMA contrasts: {len(contrasts)}")
        print(f"  Treatments: {', '.join(validation.get('treatments', []))}")
        print(f"  Network valid: {validation.get('valid', False)}")
    print(f"  Budget: {budget.summary()['total_cost_usd']}")
    print()


if __name__ == "__main__":
    main()
