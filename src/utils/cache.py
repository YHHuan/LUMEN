"""
Cache & Checkpoint System
=========================
防止 token 浪費的核心機制：

1. LLM Response Cache: 
   - 相同 prompt + model → 直接返回快取結果
   - 用 content hash 做 key，不怕 prompt 微調後浪費

2. Processing Checkpoint:
   - 批量處理中斷時，從上次斷點恢復
   - 每處理完一筆就寫入 checkpoint

3. PDF Text Cache:
   - PDF 只 extract 一次，結果存在 cache 中
   - 避免重複 parse 同一份 PDF

4. Token Budget Tracker:
   - 即時追蹤花費，超過預算自動停止
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
    """
    基於內容 hash 的快取系統。
    相同的 LLM 請求不會重複送出。
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(get_data_dir()) / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
    
    def _hash_key(self, content: str) -> str:
        """生成 content hash 作為 cache key"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _cache_path(self, namespace: str, key: str) -> Path:
        """快取檔案路徑"""
        ns_dir = self.cache_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{key}.json"
    
    def get(self, namespace: str, content: str) -> Optional[dict]:
        """
        查詢快取。
        
        Args:
            namespace: 快取命名空間 (如 "screening", "extraction")
            content: 用於生成 hash 的內容 (通常是 prompt 全文)
        
        Returns:
            快取的結果 dict，或 None
        """
        key = self._hash_key(content)
        path = self._cache_path(namespace, key)
        
        if path.exists():
            self.hits += 1
            with open(path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            logger.debug(f"Cache HIT [{namespace}]: {key}")
            return cached.get("result")
        
        self.misses += 1
        logger.debug(f"Cache MISS [{namespace}]: {key}")
        return None
    
    def set(self, namespace: str, content: str, result: dict):
        """寫入快取"""
        key = self._hash_key(content)
        path = self._cache_path(namespace, key)
        
        cached = {
            "key": key,
            "namespace": namespace,
            "timestamp": datetime.now().isoformat(),
            "content_preview": content[:200],
            "result": result
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cached, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Cache SET [{namespace}]: {key}")
    
    def stats(self) -> dict:
        """回傳快取命中率統計"""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits/total*100:.1f}%" if total > 0 else "N/A",
            "total_queries": total
        }
    
    def clear_namespace(self, namespace: str):
        """清除特定命名空間的快取"""
        ns_dir = self.cache_dir / namespace
        if ns_dir.exists():
            for f in ns_dir.glob("*.json"):
                f.unlink()
            logger.info(f"Cleared cache namespace: {namespace}")


class Checkpoint:
    """
    批量處理的斷點恢復系統。
    
    使用方式:
        cp = Checkpoint("phase3_screening")
        
        for study in studies:
            if cp.is_done(study['id']):
                continue  # 跳過已完成的
            
            result = process_study(study)
            cp.mark_done(study['id'], result)
        
        cp.finalize()
    """
    
    def __init__(self, task_name: str, checkpoint_dir: str = None):
        self.task_name = task_name
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(get_data_dir()) / ".checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"{task_name}.json"
        
        # Load existing checkpoint
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                self.state = json.load(f)
            logger.info(
                f"Resumed checkpoint [{task_name}]: "
                f"{len(self.state['completed'])} items already done"
            )
        else:
            self.state = {
                "task_name": task_name,
                "started_at": datetime.now().isoformat(),
                "completed": {},   # {item_id: result}
                "failed": {},      # {item_id: error_msg}
                "total_items": 0,
            }
    
    def set_total(self, total: int):
        """設定總項目數"""
        self.state["total_items"] = total
        self._save()
    
    def is_done(self, item_id: str) -> bool:
        """檢查某項是否已完成"""
        return item_id in self.state["completed"]
    
    def mark_done(self, item_id: str, result: Any):
        """標記完成並存儲結果"""
        self.state["completed"][item_id] = result
        self._save()
    
    def mark_failed(self, item_id: str, error: str):
        """標記失敗"""
        self.state["failed"][item_id] = error
        self._save()
    
    def get_result(self, item_id: str) -> Optional[Any]:
        """取得已完成項目的結果"""
        return self.state["completed"].get(item_id)
    
    def get_all_results(self) -> dict:
        """取得所有已完成的結果"""
        return self.state["completed"]
    
    def progress(self) -> dict:
        """回傳進度"""
        done = len(self.state["completed"])
        failed = len(self.state["failed"])
        total = self.state["total_items"]
        return {
            "completed": done,
            "failed": failed,
            "remaining": max(0, total - done - failed),
            "total": total,
            "percent": f"{done/total*100:.1f}%" if total > 0 else "N/A"
        }
    
    def finalize(self):
        """完成時更新時間戳"""
        self.state["finished_at"] = datetime.now().isoformat()
        self._save()
        logger.info(f"Checkpoint [{self.task_name}] finalized: {self.progress()}")
    
    def _save(self):
        """持久化到磁碟"""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)


class TokenBudget:
    """
    Token 花費追蹤器。
    超過預算自動停止，避免意外大筆花費。
    
    使用方式:
        budget = TokenBudget(phase="phase3", limit_usd=10.0)
        
        # 每次 LLM 呼叫後記錄
        budget.record(
            model="claude-sonnet-4",
            input_tokens=500,
            output_tokens=200,
            pricing={"input_per_million": 3.0, "output_per_million": 15.0}
        )
        
        # 檢查是否超標
        if budget.is_over_budget():
            raise BudgetExceededError(budget.summary())
    """
    
    def __init__(self, phase: str, limit_usd: float = 10.0,
                 budget_dir: str = None, reset: bool = False):
        self.phase = phase
        self.limit_usd = limit_usd
        self.budget_dir = Path(budget_dir) if budget_dir else Path(get_data_dir()) / ".budget"
        self.budget_dir.mkdir(parents=True, exist_ok=True)
        self.budget_file = self.budget_dir / f"{phase}_budget.json"

        fresh_records = {
            "phase": phase,
            "limit_usd": limit_usd,
            "calls": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_write_tokens": 0,
            "total_cost_usd": 0.0,
        }

        # reset=True clears accumulated spend from previous runs so that
        # re-running a phase doesn't hit the budget limit prematurely.
        if reset and self.budget_file.exists():
            self.budget_file.unlink()
            logger.info(f"Budget reset for phase [{phase}]")

        if self.budget_file.exists() and not reset:
            with open(self.budget_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            # Backfill any keys added in newer versions (e.g. cache token fields)
            for k, v in fresh_records.items():
                loaded.setdefault(k, v)
            self.records = loaded
        else:
            self.records = fresh_records
    
    def record(self, model: str, input_tokens: int, output_tokens: int,
               pricing: dict, description: str = "",
               cache_read_tokens: int = 0, cache_write_tokens: int = 0):
        """
        記錄一次 LLM 呼叫的 token 消耗（支援 cache 折扣計價）。

        Cache pricing (OpenRouter):
          Anthropic — read: 0.10x, write: 1.25x input price
          OpenAI    — read: 0.10x, write: 1.00x (free write)
        We use the conservative Anthropic factors for all models.
        """
        inp_price = pricing["input_per_million"] / 1_000_000
        out_price = pricing["output_per_million"] / 1_000_000

        regular_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)

        input_cost  = regular_input      * inp_price
        read_cost   = cache_read_tokens  * inp_price * 0.10   # 90% discount
        write_cost  = cache_write_tokens * inp_price * 1.25   # 25% surcharge
        output_cost = output_tokens      * out_price
        call_cost   = input_cost + read_cost + write_cost + output_cost

        self.records["calls"].append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens":  cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": round(call_cost, 8),
            "description": description,
        })
        
        self.records["total_input_tokens"]       += input_tokens
        self.records["total_output_tokens"]      += output_tokens
        self.records["total_cache_read_tokens"]  += cache_read_tokens
        self.records["total_cache_write_tokens"] += cache_write_tokens
        self.records["total_cost_usd"] = round(
            self.records["total_cost_usd"] + call_cost, 6
        )
        
        self._save()
        
        # Log warning if approaching limit
        usage_pct = self.records["total_cost_usd"] / self.limit_usd * 100
        if usage_pct > 80:
            logger.warning(
                f"⚠️  Token budget [{self.phase}]: "
                f"${self.records['total_cost_usd']:.4f} / ${self.limit_usd} "
                f"({usage_pct:.1f}%)"
            )
    
    def is_over_budget(self) -> bool:
        """是否超過預算"""
        return self.records["total_cost_usd"] >= self.limit_usd
    
    def remaining_usd(self) -> float:
        """剩餘預算"""
        return max(0, self.limit_usd - self.records["total_cost_usd"])
    
    def summary(self) -> dict:
        """花費摘要 — includes cache savings for paper reporting."""
        total_in  = self.records["total_input_tokens"]
        cr_tokens = self.records.get("total_cache_read_tokens", 0)
        cw_tokens = self.records.get("total_cache_write_tokens", 0)
        cache_hit_rate = (
            f"{cr_tokens / total_in * 100:.1f}%" if total_in > 0 else "N/A"
        )
        return {
            "phase": self.phase,
            "total_cost_usd":          f"${self.records['total_cost_usd']:.4f}",
            "limit_usd":               f"${self.limit_usd}",
            "remaining_usd":           f"${self.remaining_usd():.4f}",
            "total_input_tokens":      total_in,
            "total_output_tokens":     self.records["total_output_tokens"],
            "total_cache_read_tokens": cr_tokens,
            "total_cache_write_tokens": cw_tokens,
            "cache_hit_rate":          cache_hit_rate,
            "total_calls":             len(self.records["calls"]),
            "over_budget":             self.is_over_budget(),
        }
    
    def _save(self):
        with open(self.budget_file, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)


class BudgetExceededError(Exception):
    """Token 預算超標時拋出"""
    pass


class PDFTextCache:
    """
    PDF 文字提取快取。
    每個 PDF 只做一次 text extraction，之後直接讀快取。
    
    避免場景:
    - Phase 3 全文篩選 extract 一次
    - Phase 4 資料提取又 extract 一次  ← 這就浪費了
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(get_data_dir()) / ".cache" / "pdf_text"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _file_hash(self, pdf_path: str) -> str:
        """計算 PDF 檔案的 hash"""
        h = hashlib.md5()
        with open(pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    
    def get(self, pdf_path: str, max_tokens: int = 0) -> Optional[dict]:
        """
        取得已快取的 PDF 文字。
        max_tokens is included in the cache key so 5000-token (screening)
        and 12000-token (extraction) results are stored separately.

        Returns:
            {"full_text": str, "sections": dict, "pages": int, "tokens_approx": int}
            或 None
        """
        file_hash = self._file_hash(pdf_path)
        cache_file = self.cache_dir / f"{file_hash}_t{max_tokens}.json"

        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def set(self, pdf_path: str, extracted: dict, max_tokens: int = 0):
        """儲存 PDF 文字快取 (keyed on file hash + max_tokens)"""
        file_hash = self._file_hash(pdf_path)
        cache_file = self.cache_dir / f"{file_hash}_t{max_tokens}.json"

        extracted["_meta"] = {
            "source_file": str(pdf_path),
            "file_hash": file_hash,
            "max_tokens": max_tokens,
            "extracted_at": datetime.now().isoformat(),
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
