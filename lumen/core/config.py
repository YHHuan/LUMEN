"""
LUMEN v3 Configuration loader.

Loads YAML configs for model tiers, defaults, and project settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and merge default.yaml + models.yaml from *config_dir*.

    Returns a single dict with top-level keys ``models`` and ``defaults``.
    """
    config_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR

    merged: dict[str, Any] = {}

    for name in ("default", "models"):
        path = config_dir / f"{name}.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
                merged.update(data)

    return merged


def get_tier_config(config: dict, tier: str) -> dict[str, Any]:
    """Return the model configuration for a specific tier (fast/smart/strategic)."""
    tiers = config.get("tiers", {})
    if tier not in tiers:
        raise KeyError(f"Unknown model tier: {tier!r}. Available: {list(tiers.keys())}")
    return tiers[tier]
