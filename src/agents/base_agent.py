"""
Base Agent
==========
所有 AI agent 的基底類別。
內建:
- OpenRouter API 呼叫
- 自動快取 (相同 prompt 不重複呼叫)
- Token budget 追蹤
- 重試邏輯
- 結構化 JSON 輸出解析
"""

import hashlib
import json
import time
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.cache import ContentCache, TokenBudget, BudgetExceededError
from src.utils.project import get_data_dir

logger = logging.getLogger(__name__)


def _audit_log_path() -> Path:
    return Path(get_data_dir()) / ".audit" / "prompt_log.jsonl"


class BaseAgent:
    """
    所有 agent 的基底。
    
    子類只需要:
    1. 設定 self.role_name (對應 models.yaml 中的 key)
    2. 實作 build_prompt() 方法
    3. 呼叫 self.call_llm(prompt) 即可
    """
    
    def __init__(self, role_name: str, config_path: str = "config/models.yaml",
                 budget: Optional[TokenBudget] = None):
        self.role_name = role_name
        
        # Load model config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config["models"][role_name]
        self.batch_settings = config.get("batch_settings", {})
        
        # OpenRouter client
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
        # Cache & Budget
        self.cache = ContentCache()
        self.budget = budget

        # Prompt config (loaded from config/prompts/<role>.yaml if present)
        self._prompt_config = self.load_prompt_config(role_name)
    
    @staticmethod
    def load_prompt_config(role_name: str,
                           prompts_dir: str = "config/prompts") -> dict:
        """Load prompt yaml for this role. Returns {} if file absent."""
        path = Path(prompts_dir) / f"{role_name}.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _log_prompt_audit(self, system_prompt: str, user_prompt: str,
                          cache_namespace: str, description: str,
                          tokens: dict) -> None:
        """Append one entry to the JSONL prompt audit log (PRISMA-trAIce §prompt docs)."""
        try:
            audit_path = _audit_log_path()
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "role": self.role_name,
                # model_id = what we requested; actual_model = what OpenRouter used
                "model_id": self.model_config["model_id"],
                "actual_model": tokens.get("actual_model", self.model_config["model_id"]),
                "pinned_at": self.model_config.get("pinned_at", ""),
                "temperature": self.model_config.get("temperature", 0.0),
                "seed": self.model_config.get("seed"),
                # TRIPOD 6c: prompt version from config/prompts/<role>.yaml
                "prompt_version": self._prompt_config.get("version", "unknown"),
                # TRIPOD 7c: API method and endpoint
                "api_url": "https://openrouter.ai/api/v1",
                "api_method": "v1/chat/completions (sync)",
                "cache_namespace": cache_namespace or "",
                "description": description,
                "system_prompt_sha256": hashlib.sha256(
                    system_prompt.encode()).hexdigest()[:16],
                "user_prompt_sha256": hashlib.sha256(
                    user_prompt.encode()).hexdigest()[:16],
                # full token breakdown for paper reporting
                "input_tokens":        tokens.get("input", 0),
                "output_tokens":       tokens.get("output", 0),
                "cache_read_tokens":   tokens.get("cache_read_tokens", 0),
                "cache_write_tokens":  tokens.get("cache_write_tokens", 0),
                "estimated_cost_usd":  tokens.get("estimated_cost_usd", 0.0),
                "cache_hit":           tokens.get("cache_read_tokens", 0) > 0,
                # keep nested tokens dict for backward-compatibility
                "tokens": tokens,
            }
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Audit log write failed (non-fatal): {e}")

    def call_llm(self, prompt: str, system_prompt: str = "",
                 expect_json: bool = False,
                 cache_namespace: Optional[str] = None,
                 description: str = "") -> dict:
        """
        呼叫 LLM，帶有快取和預算控制。
        
        Args:
            prompt: 使用者 prompt
            system_prompt: 系統 prompt
            expect_json: 是否期望 JSON 格式回覆
            cache_namespace: 快取命名空間 (None = 不快取)
            description: 用於 budget 記錄的描述
        
        Returns:
            {"content": str, "parsed": dict|None, "tokens": dict}
        """
        # === 1. Check budget ===
        if self.budget and self.budget.is_over_budget():
            raise BudgetExceededError(
                f"Budget exceeded for {self.budget.phase}: "
                f"{json.dumps(self.budget.summary(), indent=2)}"
            )
        
        # === 2. Check cache ===
        cache_key_content = f"{self.model_config['model_id']}|{system_prompt}|{prompt}"
        
        if cache_namespace:
            cached = self.cache.get(cache_namespace, cache_key_content)
            if cached is not None:
                logger.info(
                    f"[{self.role_name}] Cache hit! Saved tokens for: "
                    f"{description or prompt[:50]}..."
                )
                return cached
        
        # === 3. Build messages ===
        # Anthropic models via OpenRouter support prompt caching via cache_control.
        # Wrapping the system prompt in a content block enables the cache.
        # Min 2,048 tokens for Sonnet; pad with spaces to reach threshold cheaply.
        if system_prompt and self._is_anthropic_model():
            # Pad to ≥2,048 tokens for Anthropic cache activation.
            # Spaces collapse; use actual content density estimate from the prompt.
            _tok_est = max(1, len(system_prompt) // 4)  # rough 4 chars/token
            _pad_tokens_needed = max(0, 2100 - _tok_est)  # 2100 gives headroom
            _padded = system_prompt
            if _pad_tokens_needed > 0:
                # Short separator lines tokenize reliably (~15 tok/line)
                _lines_needed = (_pad_tokens_needed // 15) + 1
                _pad = "\n" + "\n".join(["#" + "─" * 78] * _lines_needed)
                _padded = system_prompt + _pad
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": _padded,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": prompt},
            ]
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        # === 4. Call LLM with retry ===
        response = self._call_with_retry(messages)
        
        # === 5. Parse response ===
        msg = response.choices[0].message

        # content may be None when:
        #   a) model returned only a thinking trace (but we didn't request it)
        #   b) API/SDK parse error
        # Always normalise to str so downstream code never sees None.
        content: str = msg.content or ""

        if not content:
            # Secondary fallback: some OpenRouter models surface the actual
            # text reply in model_extra fields (e.g. older Gemini routing).
            # NOTE: "reasoning" here is the text answer, NOT a thinking trace —
            # we only hit this branch when content is absent entirely.
            extra = getattr(msg, "model_extra", {}) or {}
            content = extra.get("reasoning") or extra.get("content") or ""
            if content:
                logger.debug(
                    f"[{self.role_name}] content was None — extracted from model_extra."
                )

        if not content:
            logger.warning(
                f"[{self.role_name}] Empty response from API "
                f"(model={self.model_config['model_id']}). "
                "Will return empty result."
            )

        parsed = None
        if expect_json and content:
            parsed = self._extract_json(content)
        
        # === 6. Track tokens ===
        usage = response.usage
        input_tokens  = usage.prompt_tokens     if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # TRIPOD Item 6a: actual model version from API response
        actual_model = getattr(response, "model", self.model_config["model_id"])

        # Cache metrics — Anthropic (via OpenRouter)
        cache_read_tokens  = getattr(usage, "cache_read_input_tokens",      0) or 0
        cache_write_tokens = getattr(usage, "cache_creation_input_tokens",  0) or 0
        # Cache metrics — OpenAI (prompt_tokens_details.cached_tokens)
        if cache_read_tokens == 0:
            _details = getattr(usage, "prompt_tokens_details", None)
            if _details:
                cache_read_tokens = getattr(_details, "cached_tokens", 0) or 0

        if cache_read_tokens or cache_write_tokens:
            logger.info(
                f"[{self.role_name}] Cache — read: {cache_read_tokens} tok, "
                f"write: {cache_write_tokens} tok"
            )

        if self.budget:
            self.budget.record(
                model=actual_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                pricing=self.model_config["pricing"],
                description=description,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )

        # Estimate cost (cache-adjusted) for Langfuse metadata
        pricing = self.model_config["pricing"]
        _inp_per_m  = pricing["input_per_million"]
        _out_per_m  = pricing["output_per_million"]
        _regular    = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
        _est_cost   = round(
            _regular          * _inp_per_m   / 1_000_000 +
            cache_read_tokens  * _inp_per_m   / 1_000_000 * 0.10 +
            cache_write_tokens * _inp_per_m   / 1_000_000 * 1.25 +
            output_tokens      * _out_per_m   / 1_000_000,
            8,
        )

        token_summary = {
            "input":              input_tokens,
            "output":             output_tokens,
            "cache_read_tokens":  cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "actual_model":       actual_model,
            "estimated_cost_usd": _est_cost,
        }

        # === 7. Write prompt audit entry ===
        self._log_prompt_audit(
            system_prompt=system_prompt,
            user_prompt=prompt,
            cache_namespace=cache_namespace or "",
            description=description,
            tokens=token_summary,
        )

        # === 7b. Langfuse observability (TRIPOD-LLM) ===
        try:
            from src.utils.langfuse_client import log_generation
            log_generation(
                role_name=self.role_name,
                model_id=self.model_config["model_id"],
                actual_model=actual_model,
                messages=messages,
                content=content,
                tokens=token_summary,
                model_config=self.model_config,
            )
        except Exception as _lf_err:
            logger.debug(f"Langfuse logging skipped: {_lf_err}")

        result = {
            "content": content,
            "parsed": parsed,
            "tokens": token_summary,
        }

        # === 8. Save to cache ===
        if cache_namespace:
            self.cache.set(cache_namespace, cache_key_content, result)
        
        return result
    
    def _is_anthropic_model(self) -> bool:
        """True if the configured model is served by Anthropic (via OpenRouter)."""
        return self.model_config.get("model_id", "").startswith("anthropic/")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
    )
    def _call_with_retry(self, messages: list):
        """帶重試的 API 呼叫，支援 reasoning_effort 和 seed"""
        try:
            kwargs = dict(
                model=self.model_config["model_id"],
                messages=messages,
                max_tokens=self.model_config.get("max_tokens", 4096),
                temperature=self.model_config.get("temperature", 0.0),
            )

            # Seed for reproducibility (TRIPOD 6c)
            seed = self.model_config.get("seed")
            if seed is not None:
                kwargs["seed"] = seed

            # Reasoning/thinking mode: OpenRouter passes this via extra_body
            # Gemini 3.1 Flash Lite supports: minimal | low | medium | high
            reasoning_effort = self.model_config.get("reasoning_effort")
            if reasoning_effort:
                kwargs["extra_body"] = {
                    "reasoning": {"effort": reasoning_effort}
                }
                logger.debug(
                    f"[{self.role_name}] Reasoning mode enabled "
                    f"(effort={reasoning_effort})"
                )

            response = self.client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            logger.warning(f"[{self.role_name}] API call failed: {e}, retrying...")
            raise
    
    def _extract_json(self, content: str) -> Optional[dict]:
        """
        從 LLM 回覆中提取 JSON。
        處理: markdown blocks, 多餘文字, 截斷的 JSON。
        """
        import re
        
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from ```json ... ``` blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON object/array boundaries
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = content.find(start_char)
            end = content.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start:end+1])
                except json.JSONDecodeError:
                    pass
        
        # === v5: Try to repair truncated JSON ===
        # LLM may have hit max_tokens mid-output, leaving unclosed braces
        first_brace = content.find('{')
        if first_brace != -1:
            fragment = content[first_brace:]
            repaired = self._repair_truncated_json(fragment)
            if repaired is not None:
                logger.info(f"[{self.role_name}] Repaired truncated JSON successfully")
                return repaired
        
        logger.warning(f"[{self.role_name}] Could not parse JSON from response "
                       f"(length={len(content)}, first 200 chars: {content[:200]})")
        return None
    
    def _repair_truncated_json(self, fragment: str) -> Optional[dict]:
        """
        Attempt to repair truncated JSON by closing open braces/brackets.
        Common when LLM hits max_tokens mid-output.
        """
        # Count open/close braces
        open_braces = fragment.count('{') - fragment.count('}')
        open_brackets = fragment.count('[') - fragment.count(']')
        
        if open_braces <= 0 and open_brackets <= 0:
            return None  # Not a truncation issue
        
        # Strip trailing comma or incomplete key-value
        repaired = fragment.rstrip()
        # Remove trailing partial strings like: "key": "val...  or  "key":
        repaired = repaired.rstrip(',')
        # Remove incomplete string at end
        if repaired.count('"') % 2 == 1:
            last_quote = repaired.rfind('"')
            repaired = repaired[:last_quote + 1]
        # Remove trailing colon (incomplete key-value pair)
        repaired = repaired.rstrip().rstrip(':').rstrip(',')
        # Remove incomplete key at end like:  , "some_key
        import re
        repaired = re.sub(r',\s*"[^"]*$', '', repaired)
        
        # Close brackets/braces
        repaired += ']' * max(0, open_brackets)
        repaired += '}' * max(0, open_braces)
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Try more aggressive: find the last valid closing point
            for trim in range(min(200, len(repaired)), 0, -10):
                try:
                    candidate = repaired[:len(repaired)-trim].rstrip().rstrip(',')
                    ob = candidate.count('{') - candidate.count('}')
                    olb = candidate.count('[') - candidate.count(']')
                    if candidate.count('"') % 2 == 1:
                        candidate += '"'
                    candidate += ']' * max(0, olb)
                    candidate += '}' * max(0, ob)
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
            return None
    
    def call_llm_batch(self, items: list, build_prompt_fn, 
                       system_prompt: str = "",
                       cache_namespace: str = "",
                       batch_size: int = None,
                       description_fn=None) -> list:
        """
        批量呼叫 LLM，每個 item 會被 build_prompt_fn 轉換為 prompt。
        內建 checkpoint，中斷後可從斷點恢復。
        
        Args:
            items: 待處理的項目列表
            build_prompt_fn: fn(item) -> prompt_str
            system_prompt: 共用的系統 prompt
            cache_namespace: 快取命名空間
            batch_size: 每批幾個 (None = 逐個處理)
            description_fn: fn(item) -> str, 用於 budget 記錄
        
        Returns:
            list of results (same order as items)
        """
        from src.utils.cache import Checkpoint
        from tqdm import tqdm
        
        if batch_size is None:
            batch_size = self.batch_settings.get("screening_batch_size", 10)
        
        checkpoint = Checkpoint(f"{self.role_name}_{cache_namespace}")
        checkpoint.set_total(len(items))
        
        results = []
        
        for i, item in enumerate(tqdm(items, desc=f"[{self.role_name}]")):
            item_id = item.get("study_id", item.get("id", str(i)))
            
            # Skip if already done
            if checkpoint.is_done(item_id):
                results.append(checkpoint.get_result(item_id))
                continue
            
            try:
                prompt = build_prompt_fn(item)
                desc = description_fn(item) if description_fn else f"Item {item_id}"
                
                result = self.call_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    expect_json=True,
                    cache_namespace=cache_namespace,
                    description=desc,
                )
                
                checkpoint.mark_done(item_id, result)
                results.append(result)
                
                # Rate limiting between calls
                delay = self.batch_settings.get("batch_delay_seconds", 2)
                time.sleep(delay)
                
            except BudgetExceededError:
                logger.error(f"Budget exceeded! Stopping at item {i}/{len(items)}")
                checkpoint.finalize()
                raise
            except Exception as e:
                logger.error(f"Failed processing {item_id}: {e}")
                checkpoint.mark_failed(item_id, str(e))
                results.append(None)
        
        checkpoint.finalize()
        return results
