"""
Text Normalization Utilities — LUMEN v2
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize Unicode text for consistent matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_author_name(name: str) -> str:
    """Normalize author name for matching."""
    name = normalize_text(name)
    name = re.sub(r"\b(Jr|Sr|III|IV|II)\b\.?", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_year(text: str) -> str:
    """Extract 4-digit year from text."""
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else ""
