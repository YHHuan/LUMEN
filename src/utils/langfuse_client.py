"""
Langfuse Observability Client (v4 SDK)
=======================================
Singleton init — reads LANGFUSE_* from environment automatically.
Non-fatal: if keys missing or SDK unavailable, logs once and skips silently.

Supports Langfuse SDK v4 (OpenTelemetry-based). Falls back silently if
the package is not installed or keys are not configured.

TRIPOD-LLM items covered:
  6a  — model_version (actual model string from API response)
  6c  — temperature, seed, max_tokens
  7c  — inference_date, api_method
  12  — estimated_cost_usd (cache-adjusted)
  14d — prospero_id, run_id
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional, Any

logger = logging.getLogger(__name__)

_langfuse = None
_enabled: Optional[bool] = None   # None = not yet attempted


def get_langfuse():
    """Return the Langfuse singleton, or None if unavailable."""
    global _langfuse, _enabled

    if _enabled is not None:          # already initialised (or failed)
        return _langfuse if _enabled else None

    secret = os.getenv("LANGFUSE_SECRET_KEY", "").strip('"').strip("'")
    public = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip('"').strip("'")
    host   = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not secret or not public:
        logger.info("Langfuse keys not set — observability disabled")
        _enabled = False
        return None

    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            secret_key=secret,
            public_key=public,
            host=host,
        )
        _enabled = True
        logger.info(f"Langfuse observability enabled (host={host})")
    except ImportError:
        logger.info("langfuse package not installed — run: pip install langfuse")
        _enabled = False
    except Exception as e:
        logger.warning(f"Langfuse init failed (non-fatal): {e}")
        _enabled = False

    return _langfuse if _enabled else None


def log_generation(
    role_name: str,
    model_id: str,
    actual_model: str,
    messages: list,
    content: str,
    tokens: dict,
    model_config: dict,
) -> None:
    """
    Log one LLM generation to Langfuse (v4 SDK). Completely non-fatal.

    Args:
        role_name    : agent role key (e.g. "screener1")
        model_id     : model_id from config (the requested model)
        actual_model : model string returned by the API (TRIPOD 6a)
        messages     : the messages list sent to the API
        content      : text response from the LLM
        tokens       : token dict from call_llm result["tokens"]
        model_config : the model's config dict from models.yaml
    """
    lf = get_langfuse()
    if lf is None:
        return

    try:
        run_id = os.getenv("RUN_ID", "run_unknown")

        # v4 SDK: use start_observation with as_type='generation'
        # Pass all data at creation time so the generation is complete on .end()
        gen = lf.start_observation(
            name=role_name,
            as_type="generation",
            model=actual_model or model_id,
            model_parameters={
                "temperature": model_config.get("temperature", 0.0),
                "max_tokens":  model_config.get("max_tokens", 4096),
                "seed":        model_config.get("seed"),
            },
            input=messages,
            output=content or "",
            # v4 usage_details: token counts as Dict[str, int]
            usage_details={
                "input":  tokens.get("input", 0),
                "output": tokens.get("output", 0),
                # Anthropic cache fields
                "cache_read_input_tokens":     tokens.get("cache_read_tokens", 0),
                "cache_creation_input_tokens": tokens.get("cache_write_tokens", 0),
            },
            # v4 cost_details: costs as Dict[str, float]
            cost_details={
                "total": tokens.get("estimated_cost_usd", 0.0),
            },
            metadata={
                # ── TRIPOD-LLM ──────────────────────────────
                "model_version":    actual_model or model_id,
                "inference_date":   datetime.now(timezone.utc).isoformat(),
                "api_method":       "openrouter/v1/chat/completions",
                "pipeline_version": os.getenv("PIPELINE_VERSION", "v5"),
                "prospero_id":      os.getenv("PROSPERO_ID", ""),
                "run_id":           run_id,
                "run_date":         os.getenv("RUN_DATE", ""),
                # ── Cache ───────────────────────────────────
                "cache_hit":          tokens.get("cache_read_tokens", 0) > 0,
                "cache_read_tokens":  tokens.get("cache_read_tokens", 0),
                "cache_write_tokens": tokens.get("cache_write_tokens", 0),
                # ── Cost ────────────────────────────────────
                "estimated_cost_usd": tokens.get("estimated_cost_usd", 0.0),
                # ── Agent ───────────────────────────────────
                "agent_role":      role_name,
                "model_pinned_at": model_config.get("pinned_at", ""),
            },
        )
        gen.end()
    except Exception as e:
        logger.debug(f"Langfuse log_generation failed (non-fatal): {e}")


def log_phase_start(phase_name: str, metadata: dict) -> Any:
    """
    Log a phase-run span to Langfuse. Call at the very start of each phase's
    main(). Returns a span handle — pass it to log_phase_end() when the phase
    finishes. Returns None if Langfuse is unavailable (always safe to ignore).

    metadata should include: input_study_count, models used, budget_usd, etc.
    """
    lf = get_langfuse()
    if lf is None:
        return None

    try:
        from langfuse.types import TraceContext

        run_id   = os.getenv("RUN_ID", "run_unknown")
        run_date = os.getenv("RUN_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        span = lf.start_observation(
            name=f"{phase_name}",
            as_type="span",
            trace_context=TraceContext(session_id=run_id),
            input={
                **metadata,
                # ── run identity ───────────────────────────────
                "phase":            phase_name,
                "run_id":           run_id,
                "run_date":         run_date,
                "pipeline_version": os.getenv("PIPELINE_VERSION", "v5"),
                "prospero_id":      os.getenv("PROSPERO_ID", ""),
                "started_at":       datetime.now(timezone.utc).isoformat(),
            },
            metadata={
                "phase":            phase_name,
                "run_id":           run_id,
                "run_date":         run_date,
                "pipeline_version": os.getenv("PIPELINE_VERSION", "v5"),
                "prospero_id":      os.getenv("PROSPERO_ID", ""),
            },
        )
        logger.info(f"Langfuse: phase span started for '{phase_name}' (run={run_id})")
        return span
    except Exception as e:
        logger.debug(f"log_phase_start failed (non-fatal): {e}")
        return None


def log_phase_end(span: Any, phase_name: str, summary: dict) -> None:
    """
    End a phase span logged by log_phase_start(). Call at the very end of
    each phase's main(), after all outputs are saved.

    summary should include: included/excluded counts, total_cost_usd,
    cache_hit_rate, duration_s, etc.
    """
    if span is None:
        return

    lf = get_langfuse()
    if lf is None:
        return

    try:
        span.update(
            output={
                **summary,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        span.end()
        lf.flush()   # ensure data is sent before the process exits
        logger.info(f"Langfuse: phase span ended for '{phase_name}'")
    except Exception as e:
        logger.debug(f"log_phase_end failed (non-fatal): {e}")
