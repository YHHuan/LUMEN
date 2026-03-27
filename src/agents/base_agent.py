"""
Base Agent — LUMEN v2
======================
All AI agents inherit from this class.
Provides: OpenRouter API calls, response caching, token budget tracking,
retry logic, structured JSON parsing, prompt audit logging.
"""

import hashlib
import json
import re
import time
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryCallState

from src.utils.cache import ContentCache, TokenBudget, BudgetExceededError
from src.utils.project import get_data_dir

logger = logging.getLogger(__name__)


def _audit_log_path() -> Path:
    return Path(get_data_dir()) / ".audit" / "prompt_log.jsonl"


class BaseAgent:
    """
    Base class for all LLM agents.

    Subclasses need:
    1. Set self.role_name (matches key in models.yaml)
    2. Implement domain-specific methods
    3. Call self.call_llm(prompt) for LLM interaction
    """

    def __init__(self, role_name: str, config_path: str = None,
                 budget: Optional[TokenBudget] = None):
        self.role_name = role_name

        import os
        from dotenv import load_dotenv
        load_dotenv()

        if config_path is None:
            config_path = os.getenv("LUMEN_MODEL_CONFIG", "config/models.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.model_config = config["models"][role_name]
        self.batch_settings = config.get("batch_settings", {})

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=120.0,
        )

        self.cache = ContentCache()
        self.budget = budget
        self._prompt_config = self.load_prompt_config(role_name)

    @staticmethod
    def load_prompt_config(role_name: str,
                           prompts_dir: str = "config/prompts") -> dict:
        path = Path(prompts_dir) / f"{role_name}.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _log_prompt_audit(self, system_prompt: str, user_prompt: str,
                          cache_namespace: str, description: str,
                          tokens: dict) -> None:
        try:
            audit_path = _audit_log_path()
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "role": self.role_name,
                "model_id": self.model_config["model_id"],
                "actual_model": tokens.get("actual_model", self.model_config["model_id"]),
                "pinned_at": self.model_config.get("pinned_at", ""),
                "temperature": self.model_config.get("temperature", 0.0),
                "seed": self.model_config.get("seed"),
                "prompt_version": self._prompt_config.get("version", "unknown"),
                "api_url": "https://openrouter.ai/api/v1",
                "api_method": "v1/chat/completions (sync)",
                "cache_namespace": cache_namespace or "",
                "description": description,
                "system_prompt_sha256": hashlib.sha256(
                    system_prompt.encode()).hexdigest()[:16],
                "user_prompt_sha256": hashlib.sha256(
                    user_prompt.encode()).hexdigest()[:16],
                "input_tokens": tokens.get("input", 0),
                "output_tokens": tokens.get("output", 0),
                "cache_read_tokens": tokens.get("cache_read_tokens", 0),
                "cache_write_tokens": tokens.get("cache_write_tokens", 0),
                "estimated_cost_usd": tokens.get("estimated_cost_usd", 0.0),
                "latency_seconds": tokens.get("latency_seconds", 0.0),
                "retry_count": tokens.get("retry_count", 0),
                "failed": tokens.get("failed", False),
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
        Call LLM with caching and budget control.

        Returns: {"content": str, "parsed": dict|None, "tokens": dict}
        """
        # 1. Budget check
        if self.budget and self.budget.is_over_budget():
            raise BudgetExceededError(
                f"Budget exceeded for {self.budget.phase}: "
                f"cost=${self.budget.summary().get('total_cost_usd', 0)}, "
                f"limit=${self.budget.summary().get('limit_usd', 0)}"
            )

        # 2. Cache check
        cache_key_content = f"{self.model_config['model_id']}|{system_prompt}|{prompt}"

        if cache_namespace:
            cached = self.cache.get(cache_namespace, cache_key_content)
            if cached is not None:
                logger.info(
                    f"[{self.role_name}] Cache hit: {description or prompt[:50]}..."
                )
                # Record original token usage (cost=$0) so budget file tracks cache hits
                if self.budget and isinstance(cached, dict):
                    ct = cached.get("tokens", {})
                    if ct:
                        # Store original token counts for visibility, but zero-price
                        zero_pricing = {"input_per_1m": 0, "output_per_1m": 0}
                        self.budget.record(
                            model=ct.get("actual_model", "cache_hit"),
                            input_tokens=ct.get("input", 0),
                            output_tokens=ct.get("output", 0),
                            pricing=zero_pricing,
                            description=f"[cache hit, $0] {description}",
                            cache_read_tokens=ct.get("cache_read_tokens", 0),
                            cache_write_tokens=ct.get("cache_write_tokens", 0),
                        )
                return cached

        # 3. Build messages (with Anthropic prompt caching support)
        if system_prompt and self._is_anthropic_model():
            _tok_est = max(1, len(system_prompt) // 4)
            _pad_tokens_needed = max(0, 2100 - _tok_est)
            _padded = system_prompt
            if _pad_tokens_needed > 0:
                _lines_needed = (_pad_tokens_needed // 15) + 1
                _pad = "\n" + "\n".join(["#" + "\u2500" * 78] * _lines_needed)
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

        # 4. Call with retry (with latency + retry tracking)
        _t0 = time.time()
        _retry_count = 0
        _call_failed = False
        try:
            response = self._call_with_retry(messages)
            # tenacity stores attempt number on the retry state
            _stats = getattr(self._call_with_retry, "statistics", {})
            _retry_count = max(0, _stats.get("attempt_number", 1) - 1)
        except Exception:
            _call_failed = True
            raise
        _latency_s = round(time.time() - _t0, 3)

        # 5. Parse response (guard against empty choices from API)
        if not response or not response.choices:
            logger.warning(f"[{self.role_name}] Empty response from API, retrying once...")
            import time as _time
            _time.sleep(2)
            _retry_count += 1
            _t0 = time.time()
            response = self._call_with_retry(messages)
            _latency_s = round(time.time() - _t0, 3)
            if not response or not response.choices:
                raise RuntimeError(f"[{self.role_name}] API returned empty response twice")
        msg = response.choices[0].message
        content: str = msg.content or ""

        if not content:
            extra = getattr(msg, "model_extra", {}) or {}
            content = extra.get("reasoning") or extra.get("content") or ""

        parsed = None
        if expect_json and content:
            parsed = self._extract_json(content)

        # 6. Token tracking
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        actual_model = getattr(response, "model", self.model_config["model_id"])

        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        if cache_read_tokens == 0:
            _details = getattr(usage, "prompt_tokens_details", None)
            if _details:
                cache_read_tokens = getattr(_details, "cached_tokens", 0) or 0

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

        pricing = self.model_config["pricing"]
        _inp_per_m = pricing["input_per_1m"]
        _out_per_m = pricing["output_per_1m"]
        _regular = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
        _est_cost = round(
            _regular * _inp_per_m / 1_000_000
            + cache_read_tokens * _inp_per_m / 1_000_000 * 0.10
            + cache_write_tokens * _inp_per_m / 1_000_000 * 1.25
            + output_tokens * _out_per_m / 1_000_000,
            8,
        )

        token_summary = {
            "input": input_tokens,
            "output": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "actual_model": actual_model,
            "estimated_cost_usd": _est_cost,
            "latency_seconds": _latency_s,
            "retry_count": _retry_count,
            "failed": _call_failed,
        }

        # 7. Audit log
        self._log_prompt_audit(
            system_prompt=system_prompt,
            user_prompt=prompt,
            cache_namespace=cache_namespace or "",
            description=description,
            tokens=token_summary,
        )

        result = {
            "content": content,
            "parsed": parsed,
            "tokens": token_summary,
        }

        # 8. Save to cache
        if cache_namespace:
            self.cache.set(cache_namespace, cache_key_content, result)

        return result

    def _is_anthropic_model(self) -> bool:
        return self.model_config.get("model_id", "").startswith("anthropic/")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
    )
    def _call_with_retry(self, messages: list):
        try:
            kwargs = dict(
                model=self.model_config["model_id"],
                messages=messages,
                max_tokens=self.model_config.get("max_tokens", 4096),
                temperature=self.model_config.get("temperature", 0.0),
            )
            seed = self.model_config.get("seed")
            if seed is not None:
                kwargs["seed"] = seed

            reasoning_effort = self.model_config.get("reasoning_effort")
            if reasoning_effort:
                kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}

            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.warning(f"[{self.role_name}] API call failed: {e}, retrying...")
            raise

    def _extract_json(self, content: str) -> Optional[dict]:
        """Extract JSON from LLM response, handling markdown blocks and truncation."""
        # Direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # From ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Opening code fence without closing (truncated output)
        fence_match = re.match(r"```(?:json)?\s*", content)
        if fence_match:
            content = content[fence_match.end():]

        # Find JSON object/array boundaries
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = content.find(start_char)
            end = content.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start:end + 1])
                except json.JSONDecodeError:
                    pass

        # Repair truncated JSON
        first_brace = content.find("{")
        if first_brace != -1:
            repaired = self._repair_truncated_json(content[first_brace:])
            if repaired is not None:
                logger.info(f"[{self.role_name}] Repaired truncated JSON")
                return repaired

        logger.warning(
            f"[{self.role_name}] Could not parse JSON "
            f"(length={len(content)}, first 200: {content[:200]})"
        )
        return None

    def _repair_truncated_json(self, fragment: str) -> Optional[dict]:
        open_braces = fragment.count("{") - fragment.count("}")
        open_brackets = fragment.count("[") - fragment.count("]")

        if open_braces <= 0 and open_brackets <= 0:
            return None

        repaired = fragment.rstrip().rstrip(",")
        if repaired.count('"') % 2 == 1:
            last_quote = repaired.rfind('"')
            repaired = repaired[:last_quote + 1]
        repaired = repaired.rstrip().rstrip(":").rstrip(",")
        repaired = re.sub(r',\s*"[^"]*$', "", repaired)

        repaired += "]" * max(0, open_brackets)
        repaired += "}" * max(0, open_braces)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Line-by-line rollback: cut trailing lines until parseable
        lines = fragment.split("\n")
        max_cut = min(len(lines) - 1, len(lines) // 2)
        for cut in range(1, max_cut):
            candidate = "\n".join(lines[:-cut]).rstrip().rstrip(",")
            ob = candidate.count("{") - candidate.count("}")
            olb = candidate.count("[") - candidate.count("]")
            if candidate.count('"') % 2 == 1:
                candidate += '"'
            candidate += "]" * max(0, olb)
            candidate += "}" * max(0, ob)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def call_llm_batch(self, items: list, build_prompt_fn,
                       system_prompt: str = "",
                       cache_namespace: str = "",
                       batch_size: int = None,
                       description_fn=None) -> list:
        """Batch LLM calls with checkpoint resume."""
        from src.utils.cache import Checkpoint
        from tqdm import tqdm

        if batch_size is None:
            batch_size = self.batch_settings.get("screening_batch_size", 10)

        checkpoint = Checkpoint(f"{self.role_name}_{cache_namespace}")
        checkpoint.set_total(len(items))

        results = []

        for i, item in enumerate(tqdm(items, desc=f"[{self.role_name}]")):
            item_id = item.get("study_id", item.get("id", str(i)))

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

                delay = self.batch_settings.get("batch_delay_seconds", 2)
                time.sleep(delay)

            except BudgetExceededError:
                logger.error(f"Budget exceeded at item {i}/{len(items)}")
                checkpoint.finalize()
                raise
            except Exception as e:
                logger.error(f"Failed processing {item_id}: {e}")
                checkpoint.mark_failed(item_id, str(e))
                results.append(None)

        checkpoint.finalize()
        return results
