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
    """Return the model configuration for a specific tier (fast/smart/strategic).

    Note: a tier may name *any* provider/model (e.g. ``anthropic/claude-...``,
    ``gemini/...``, ``openai/...`` or ``openrouter/...``). The router passes the
    string straight to LiteLLM, so tiers are not bound to any single provider.
    """
    tiers = config.get("tiers", {})
    if tier not in tiers:
        raise KeyError(f"Unknown model tier: {tier!r}. Available: {list(tiers.keys())}")
    return tiers[tier]


def _defaults(config: dict) -> dict:
    return config.get("defaults", {}) if isinstance(config, dict) else {}


def is_evidence_run(config: dict | None) -> bool:
    """True when this is an *evidence run*: no caching, fully metered/reproducible.

    Accepts the flag at either the top level (``config['evidence_run']``) or
    under ``defaults``.
    """
    if not config:
        return False
    return bool(config.get("evidence_run") or _defaults(config).get("evidence_run"))


def is_caching_enabled(config: dict | None) -> bool:
    """True only when prompt caching is explicitly on AND this is not an evidence run.

    Evidence runs always win: they must never reuse a cached generation.
    """
    if not config:
        return False
    if is_evidence_run(config):
        return False
    return bool(config.get("prompt_caching",
                           _defaults(config).get("prompt_caching", False)))


def apply_evidence_run(config: dict, enabled: bool = True) -> dict:
    """Mark *config* as an evidence run in place (also forces caching off)."""
    config.setdefault("defaults", {})
    config["evidence_run"] = enabled
    config["defaults"]["evidence_run"] = enabled
    if enabled:
        config["defaults"]["prompt_caching"] = False
    return config
