"""
File Handlers & Data Manager — LUMEN v2
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import logging

import numpy as np

from src.utils.project import get_data_dir

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DataManager:
    """Unified data I/O for all pipeline phases."""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(get_data_dir())

    def save(self, phase: str, filename: str, data: Any,
             subfolder: str = "") -> Path:
        if subfolder:
            output_dir = self.base_dir / phase / subfolder
        else:
            output_dir = self.base_dir / phase
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        if filename.endswith(".json"):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        elif filename.endswith((".yaml", ".yml")):
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        elif filename.endswith(".md"):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(data))
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(data))

        self._update_metadata(phase, subfolder, filename, data)
        logger.info(f"Saved: {filepath}")
        return filepath

    def load(self, phase: str, filename: str, subfolder: str = "") -> Any:
        if subfolder:
            filepath = self.base_dir / phase / subfolder / filename
        else:
            filepath = self.base_dir / phase / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")

        if filename.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        elif filename.endswith((".yaml", ".yml")):
            with open(filepath, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

    def exists(self, phase: str, filename: str, subfolder: str = "") -> bool:
        if subfolder:
            return (self.base_dir / phase / subfolder / filename).exists()
        return (self.base_dir / phase / filename).exists()

    def load_if_exists(self, phase: str, filename: str,
                       subfolder: str = "", default: Any = None) -> Any:
        if self.exists(phase, filename, subfolder):
            return self.load(phase, filename, subfolder)
        return default

    def load_best_included(self) -> list:
        """Load best available included studies with priority chain.

        Priority: Phase 3.3 fulltext > Phase 3.2 with PDF > Phase 3.1 T/A.
        """
        for fname, subfolder in [
            ("included_fulltext.json", "stage2_fulltext"),
            ("included_with_pdf.json", "stage2_fulltext"),
            ("included_studies.json", "stage1_title_abstract"),
        ]:
            if self.exists("phase3_screening", fname, subfolder=subfolder):
                studies = self.load("phase3_screening", fname,
                                    subfolder=subfolder)
                logger.info(f"Loaded {len(studies)} studies from {subfolder}/{fname}")
                return studies
        return []

    def phase_dir(self, phase: str, subfolder: str = "") -> Path:
        if subfolder:
            d = self.base_dir / phase / subfolder
        else:
            d = self.base_dir / phase
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _update_metadata(self, phase: str, subfolder: str,
                         filename: str, data: Any):
        """Track file metadata for pipeline progress reporting."""
        meta_dir = self.base_dir / ".meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = meta_dir / "file_log.jsonl"

        count = None
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            for key in ("studies", "results", "items"):
                if key in data and isinstance(data[key], list):
                    count = len(data[key])
                    break

        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "subfolder": subfolder,
            "filename": filename,
            "record_count": count,
        }
        with open(meta_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
