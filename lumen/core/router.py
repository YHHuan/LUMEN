"""
LUMEN v3 Model Router.

Three-tier model routing (fast / smart / strategic) via LiteLLM.
Static mapping for v3.0; future Paper 2 may introduce dynamic routing.
"""
from __future__ import annotations

import time
from typing import Any

import litellm
import structlog

from lumen.core.config import load_config, get_tier_config

logger = structlog.get_logger()

# Silence LiteLLM's verbose logging
litellm.suppress_debug_info = True


class LumenModelError(Exception):
    """Raised when both primary and fallback models fail."""

    def __init__(self, tier: str, primary_error: Exception, fallback_error: Exception | None = None):
        self.tier = tier
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        msg = f"Tier '{tier}' failed. Primary: {primary_error}"
        if fallback_error:
            msg += f" | Fallback: {fallback_error}"
        super().__init__(msg)


class ModelRouter:
    """Route LLM calls through a 3-tier model hierarchy with automatic fallback."""

    def __init__(self, config: dict | None = None, config_path: str | None = None):
        if config is None:
            config = load_config(config_path)
        self.config = config
        self._tiers = config.get("tiers", {})

    def call(
        self,
        tier: str,
        messages: list[dict],
        response_format: dict | None = None,
        agent_name: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> tuple[str, dict]:
        """Call an LLM through the tier routing system.

        Returns (response_text, usage_metadata).
        usage_metadata keys: model, input_tokens, output_tokens, cost, latency_ms.

        Retry logic:
        1. Try primary model
        2. If rate limit / timeout -> try fallback
        3. If both fail -> raise LumenModelError
        """
        tier_cfg = get_tier_config(self.config, tier)
        primary = tier_cfg["primary"]
        fallback = tier_cfg.get("fallback")
        token_limit = max_tokens or tier_cfg.get("max_tokens", 4096)

        # Build kwargs
        call_kwargs: dict[str, Any] = {
            "model": primary,
            "messages": messages,
            "max_tokens": token_limit,
            "temperature": temperature,
        }
        if response_format:
            call_kwargs["response_format"] = response_format

        # --- Try primary ---
        primary_error: Exception | None = None
        try:
            text, usage = self._do_call(call_kwargs, tier_cfg, agent_name)
            return text, usage
        except Exception as e:
            primary_error = e
            logger.warning(
                "primary_model_failed",
                agent=agent_name, tier=tier, model=primary, error=str(e),
            )

        # --- Try fallback ---
        if fallback:
            call_kwargs["model"] = fallback
            try:
                text, usage = self._do_call(call_kwargs, tier_cfg, agent_name)
                usage["fallback_used"] = True
                return text, usage
            except Exception as fallback_err:
                logger.error(
                    "fallback_model_failed",
                    agent=agent_name, tier=tier, model=fallback, error=str(fallback_err),
                )
                raise LumenModelError(tier, primary_error, fallback_err) from fallback_err

        raise LumenModelError(tier, primary_error)

    def _do_call(
        self,
        kwargs: dict[str, Any],
        tier_cfg: dict,
        agent_name: str,
    ) -> tuple[str, dict]:
        """Execute a single LiteLLM completion call and return (text, usage)."""
        start = time.perf_counter()
        response = litellm.completion(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        text = choice.message.content or ""

        usage_raw = response.usage
        input_tokens = usage_raw.prompt_tokens if usage_raw else 0
        output_tokens = usage_raw.completion_tokens if usage_raw else 0

        cost_in = input_tokens / 1000 * tier_cfg.get("cost_per_1k_input", 0)
        cost_out = output_tokens / 1000 * tier_cfg.get("cost_per_1k_output", 0)

        usage = {
            "model": kwargs["model"],
            "agent_name": agent_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": round(cost_in + cost_out, 6),
            "latency_ms": round(latency_ms, 1),
            "fallback_used": False,
        }

        return text, usage
