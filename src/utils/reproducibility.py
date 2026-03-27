"""
Reproducibility Hash — LUMEN v2
================================
Generates a deterministic SHA-256 fingerprint over the full pipeline
configuration (models, prompts, settings, PICO) to ensure
reproducibility across runs.

Outputs a reproducibility manifest with:
- Config hash: single hash covering all config files
- Individual file hashes
- Software versions (Python, key packages)
- Timestamp of generation
"""

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Files that define the pipeline behavior
CONFIG_FILES = [
    "config/models.yaml",
    "config/v2_settings.yaml",
    "config/prompts/strategist.yaml",
    "config/prompts/screener.yaml",
    "config/prompts/arbiter.yaml",
    "config/prompts/extractor.yaml",
    "config/prompts/statistician.yaml",
    "config/prompts/writer.yaml",
    "config/prompts/citation_guardian.yaml",
]


def compute_config_hash(project_root: str = ".",
                        pico_path: str = None) -> dict:
    """
    Compute reproducibility hash over all pipeline config files.

    Returns manifest dict with individual + combined hashes.
    """
    root = Path(project_root)
    file_hashes: Dict[str, str] = {}
    hasher = hashlib.sha256()

    # Hash config files
    for rel_path in CONFIG_FILES:
        full_path = root / rel_path
        if full_path.exists():
            content = full_path.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            file_hashes[rel_path] = file_hash
            hasher.update(content)
        else:
            file_hashes[rel_path] = "MISSING"

    # Hash PICO if provided
    if pico_path:
        pico_file = Path(pico_path)
        if pico_file.exists():
            content = pico_file.read_bytes()
            file_hashes["pico.yaml"] = hashlib.sha256(content).hexdigest()
            hasher.update(content)

    combined_hash = hasher.hexdigest()

    # Gather software versions
    versions = _get_software_versions()

    manifest = {
        "config_hash": combined_hash,
        "config_hash_short": combined_hash[:12],
        "generated_at": datetime.now().isoformat(),
        "file_hashes": file_hashes,
        "software_versions": versions,
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    return manifest


def verify_reproducibility(manifest_path: str,
                           project_root: str = ".") -> dict:
    """
    Verify current config against a saved manifest.

    Returns dict with match status and any differences.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    current = compute_config_hash(project_root)

    match = current["config_hash"] == saved["config_hash"]

    diffs = []
    for file_path, saved_hash in saved.get("file_hashes", {}).items():
        current_hash = current["file_hashes"].get(file_path, "MISSING")
        if current_hash != saved_hash:
            diffs.append({
                "file": file_path,
                "saved": saved_hash[:12],
                "current": current_hash[:12],
            })

    return {
        "match": match,
        "saved_hash": saved["config_hash_short"],
        "current_hash": current["config_hash_short"],
        "differences": diffs,
        "saved_at": saved.get("generated_at", "unknown"),
        "verified_at": current["generated_at"],
    }


def _get_software_versions() -> dict:
    """Gather versions of key packages."""
    versions = {}

    packages = [
        "openai", "pyyaml", "scipy", "numpy", "pandas",
        "sentence_transformers", "hnswlib", "pdfplumber",
        "matplotlib", "statsmodels", "gmft",
    ]

    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not_installed"

    return versions


def format_manifest(manifest: dict) -> str:
    """Format reproducibility manifest for display."""
    lines = [
        "=" * 60,
        "  LUMEN v2 — Reproducibility Manifest",
        "=" * 60,
        "",
        f"  Config Hash:   {manifest['config_hash_short']}",
        f"  Generated:     {manifest['generated_at']}",
        f"  Python:        {manifest['python_version'].split()[0]}",
        f"  Platform:      {manifest['platform']}",
        "",
        "  Config File Hashes:",
        "  " + "-" * 50,
    ]

    for path, h in manifest["file_hashes"].items():
        status = "OK" if h != "MISSING" else "MISSING"
        hash_display = h[:12] if h != "MISSING" else "N/A"
        lines.append(f"  {path:<45} {hash_display}  [{status}]")

    lines.extend(["", "  Package Versions:", "  " + "-" * 50])
    for pkg, ver in manifest["software_versions"].items():
        lines.append(f"  {pkg:<30} {ver}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)
