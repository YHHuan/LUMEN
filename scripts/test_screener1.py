"""
Quick smoke test: screener1 (Gemini Flash Lite + thinking mode)
Run from repo root:
    python scripts/test_screener1.py
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_screener1")

# ── Minimal project setup (avoids interactive selector) ──────────────────────
from src.utils.project import set_data_dir
set_data_dir("data/PSY_MCI_rTMS")

from src.agents.screener import ScreenerAgent

DUMMY_STUDY = {
    "study_id": "test_001",
    "title": "Repetitive Transcranial Magnetic Stimulation for Mild Cognitive Impairment: A Randomised Controlled Trial",
    "abstract": (
        "Background: rTMS has shown promise in MCI. "
        "Methods: We randomised 60 patients with MCI (aged 60-85) to active rTMS "
        "(10 Hz, left DLPFC, 20 sessions) or sham. Primary outcome: MMSE at 12 weeks. "
        "Results: Active rTMS improved MMSE by 2.3 points vs 0.1 sham (p=0.03). "
        "Conclusion: rTMS is effective for MCI."
    ),
    "year": "2024",
    "journal": "Brain Stimulation",
    "publication_types": ["Randomized Controlled Trial"],
}

DUMMY_CRITERIA = {
    "inclusion_criteria": [
        "Participants with mild cognitive impairment (MCI) or early dementia",
        "Intervention: repetitive TMS (rTMS) or theta-burst stimulation",
        "RCT or controlled clinical trial design",
    ],
    "exclusion_criteria": [
        "Healthy controls only (no clinical population)",
        "Case reports or case series",
        "Non-human studies",
    ],
}


def main():
    logger.info("Initialising ScreenerAgent (screener1 = Gemini Flash Lite + thinking)...")
    agent = ScreenerAgent(role_name="screener1")

    logger.info(f"Model ID  : {agent.model_config['model_id']}")
    logger.info(f"Thinking  : budget_tokens={agent.model_config.get('thinking_budget', 'N/A')}")
    logger.info(f"Temp      : {agent.model_config.get('temperature')}")

    logger.info("Running screen_title_abstract_batch on 1 dummy study...")
    results = agent.screen_title_abstract_batch([DUMMY_STUDY], DUMMY_CRITERIA)

    print("\n" + "="*60)
    print("RESULT:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("="*60)

    r = results[0] if results else {}
    # screen_title_abstract_batch returns parsed dicts directly
    print(f"\nDecision  : {r.get('decision', 'N/A')}")
    print(f"Confidence: {r.get('confidence', 'N/A')}")
    print(f"Reason    : {r.get('reason', 'N/A')}")
    print("\nPASS" if r.get("decision") else "\nFAIL — no decision field")


if __name__ == "__main__":
    main()
