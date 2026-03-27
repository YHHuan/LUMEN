"""
Cache & Checkpoint System — LUMEN v2
======================================
1. ContentCache: LLM response cache (same prompt+model = cached result)
2. Checkpoint: batch processing resume from interruption
3. PDFTextCache: PDF text extraction cache
4. TokenBudget: per-phase USD spend tracking with auto-stop
"""

import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import logging

from src.utils.project import get_data_dir

logger = logging.getLogger(__name__)


class ContentCache:
    """Content-hash based LLM response cache."""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(get_data_dir()) / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _hash_key(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, namespace: str, key: str) -> Path:
        ns_dir = self.cache_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{key}.json"

    def get(self, namespace: str, content: str) -> Optional[dict]:
        key = self._hash_key(content)
        path = self._cache_path(namespace, key)
        if path.exists():
            self.hits += 1
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached.get("result")
        self.misses += 1
        return None

    def set(self, namespace: str, content: str, result: dict):
        key = self._hash_key(content)
        path = self._cache_path(namespace, key)
        cached = {
            "key": key,
            "namespace": namespace,
            "timestamp": datetime.now().isoformat(),
            "content_preview": content[:200],
            "result": result,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cached, f, indent=2, ensure_ascii=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / total * 100:.1f}%" if total > 0 else "N/A",
            "total_queries": total,
        }

    def clear_namespace(self, namespace: str):
        ns_dir = self.cache_dir / namespace
        if ns_dir.exists():
            for f in ns_dir.glob("*.json"):
                f.unlink()
            logger.info(f"Cleared cache namespace: {namespace}")


class Checkpoint:
    """Batch processing checkpoint for resume-from-interruption."""

    def __init__(self, task_name: str, checkpoint_dir: str = None):
        self.task_name = task_name
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir
            else Path(get_data_dir()) / ".checkpoints"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"{task_name}.json"

        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                self.state = json.load(f)
            logger.info(
                f"Resumed checkpoint [{task_name}]: "
                f"{len(self.state['completed'])} items already done"
            )
        else:
            self.state = {
                "task_name": task_name,
                "started_at": datetime.now().isoformat(),
                "completed": {},
                "failed": {},
                "total_items": 0,
            }

    def set_total(self, total: int):
        self.state["total_items"] = total
        self._save()

    def is_done(self, item_id: str) -> bool:
        return item_id in self.state["completed"]

    def mark_done(self, item_id: str, result: Any):
        self.state["completed"][item_id] = result
        self._save()

    def mark_failed(self, item_id: str, error: str):
        self.state["failed"][item_id] = error
        self._save()

    def get_result(self, item_id: str) -> Optional[Any]:
        return self.state["completed"].get(item_id)

    def get_all_results(self) -> dict:
        return self.state["completed"]

    def progress(self) -> dict:
        done = len(self.state["completed"])
        failed = len(self.state["failed"])
        total = self.state["total_items"]
        return {
            "completed": done,
            "failed": failed,
            "remaining": max(0, total - done - failed),
            "total": total,
            "percent": f"{done / total * 100:.1f}%" if total > 0 else "N/A",
        }

    def finalize(self):
        self.state["finished_at"] = datetime.now().isoformat()
        self._save()
        logger.info(f"Checkpoint [{self.task_name}] finalized: {self.progress()}")

    def _save(self):
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)


class TokenBudget:
    """Per-phase USD spend tracker with cache-adjusted pricing."""

    def __init__(self, phase: str, limit_usd: float = 10.0,
                 budget_dir: str = None, reset: bool = False):
        self.phase = phase
        self.limit_usd = limit_usd
        self.budget_dir = (
            Path(budget_dir) if budget_dir
            else Path(get_data_dir()) / ".budget"
        )
        self.budget_dir.mkdir(parents=True, exist_ok=True)
        self.budget_file = self.budget_dir / f"{phase}_budget.json"

        fresh = {
            "phase": phase,
            "limit_usd": limit_usd,
            "calls": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_write_tokens": 0,
            "total_cost_usd": 0.0,
        }

        if reset and self.budget_file.exists():
            self.budget_file.unlink()

        if self.budget_file.exists() and not reset:
            with open(self.budget_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for k, v in fresh.items():
                loaded.setdefault(k, v)
            self.records = loaded
        else:
            self.records = fresh

    def record(self, model: str, input_tokens: int, output_tokens: int,
               pricing: dict, description: str = "",
               cache_read_tokens: int = 0, cache_write_tokens: int = 0):
        """Record one LLM call with cache-adjusted pricing."""
        inp_price = pricing["input_per_1m"] / 1_000_000
        out_price = pricing["output_per_1m"] / 1_000_000

        regular_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
        call_cost = (
            regular_input * inp_price
            + cache_read_tokens * inp_price * 0.10
            + cache_write_tokens * inp_price * 1.25
            + output_tokens * out_price
        )

        self.records["calls"].append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": round(call_cost, 8),
            "description": description,
        })

        self.records["total_input_tokens"] += input_tokens
        self.records["total_output_tokens"] += output_tokens
        self.records["total_cache_read_tokens"] += cache_read_tokens
        self.records["total_cache_write_tokens"] += cache_write_tokens
        self.records["total_cost_usd"] = round(
            self.records["total_cost_usd"] + call_cost, 6
        )
        self._save()

        usage_pct = self.records["total_cost_usd"] / self.limit_usd * 100
        if usage_pct > 80:
            logger.warning(
                f"Token budget [{self.phase}]: "
                f"${self.records['total_cost_usd']:.4f} / ${self.limit_usd} "
                f"({usage_pct:.1f}%)"
            )

    def is_over_budget(self) -> bool:
        return self.records["total_cost_usd"] >= self.limit_usd

    def remaining_usd(self) -> float:
        return max(0, self.limit_usd - self.records["total_cost_usd"])

    def summary(self) -> dict:
        """Return summary with raw numeric values (no pre-formatting)."""
        total_in = self.records["total_input_tokens"]
        cr = self.records.get("total_cache_read_tokens", 0)
        return {
            "phase": self.phase,
            "total_cost_usd": round(self.records["total_cost_usd"], 4),
            "limit_usd": self.limit_usd,
            "remaining_usd": round(self.remaining_usd(), 4),
            "total_input_tokens": total_in,
            "total_output_tokens": self.records["total_output_tokens"],
            "total_cache_read_tokens": cr,
            "cache_hit_rate": round(cr / total_in * 100, 1) if total_in > 0 else 0,
            "total_calls": len(self.records["calls"]),
            "total_llm_calls": len(self.records["calls"]),
            "over_budget": self.is_over_budget(),
        }

    def _save(self):
        with open(self.budget_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)


class BudgetExceededError(Exception):
    pass


class PDFTextCache:
    """Cache for PDF text extraction results (keyed by file hash)."""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = (
            Path(cache_dir) if cache_dir
            else Path(get_data_dir()) / ".cache" / "pdf_text"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _file_hash(self, pdf_path: str) -> str:
        h = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def get(self, pdf_path: str, max_tokens: int = 0) -> Optional[dict]:
        file_hash = self._file_hash(pdf_path)
        cache_file = self.cache_dir / f"{file_hash}_t{max_tokens}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set(self, pdf_path: str, extracted: dict, max_tokens: int = 0):
        file_hash = self._file_hash(pdf_path)
        cache_file = self.cache_dir / f"{file_hash}_t{max_tokens}.json"
        extracted["_meta"] = {
            "source_file": str(pdf_path),
            "file_hash": file_hash,
            "max_tokens": max_tokens,
            "extracted_at": datetime.now().isoformat(),
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
