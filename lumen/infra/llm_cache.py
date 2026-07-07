"""
Content-addressed prompt cache (stabilize P0 port from V4, default OFF).

Ported in spirit from lumen-code (V4) ``infra/llm_cache.py``. Rewritten from
scratch (no code/secrets copied). A disk cache keyed by the SHA-256 of the full
request so identical calls can be replayed without paying again.

**Off by default and evidence-run always disables it** — a published evidence
run must never silently reuse a cached generation (that was a real V4 footgun:
v4.0 defaulted the cache ON and caused silent reuse). Enable only for cheap
iteration via config ``prompt_caching: true`` or ``LUMEN_LLM_CACHE=1``.

This module does no network I/O and imports nothing heavy, so it is fully
unit-testable without an API key.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def cache_enabled_from_env() -> bool:
    """True if ``LUMEN_LLM_CACHE`` is set to a truthy value."""
    return os.environ.get("LUMEN_LLM_CACHE", "").strip().lower() in ("1", "true", "yes", "on")


def make_cache_key(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    response_format: dict | None = None,
    namespace: str = "",
) -> str:
    """Deterministic SHA-256 key for a completion request.

    Includes everything that can change the output, so a hit is a genuine
    identical-request replay. ``sort_keys`` makes it order-stable.
    """
    payload = {
        "namespace": namespace,
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": response_format,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class PromptCache:
    """Tiny JSON-file disk cache. Each entry stores the response text + token
    counts so cost accounting can mark replayed input tokens as ``cached``.
    """

    def __init__(self, cache_dir: str | Path, namespace: str = ""):
        self.cache_dir = Path(cache_dir)
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """Return a cached ``{text, input_tokens, output_tokens}`` or None."""
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.misses += 1
            return None
        self.hits += 1
        return data

    def put(self, key: str, text: str, input_tokens: int = 0,
            output_tokens: int = 0) -> None:
        entry = {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self._path(key).write_text(
            json.dumps(entry, ensure_ascii=False), encoding="utf-8")

    def manifest(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "namespace": self.namespace,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
