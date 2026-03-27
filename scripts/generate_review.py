"""
Generate Human Review HTML Document — LUMEN v2
=================================================
Usage: python scripts/generate_review.py
Output: data/<project>/phase4_extraction/human_review.html
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project import select_project, get_data_dir
from src.utils.file_handlers import DataManager
from src.utils.review_generator import generate_review_html

logging.basicConfig(level=logging.INFO)


def main():
    select_project()
    dm = DataManager()

    extracted = dm.load("phase4_extraction", "extracted_data.json")
    output = str(Path(get_data_dir()) / "phase4_extraction" / "human_review.html")

    generate_review_html(extracted, output)
    print(f"\n  Generated: {output}")
    print(f"  Studies: {len(extracted)}")


if __name__ == "__main__":
    main()
