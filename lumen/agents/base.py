"""
LUMEN v3 Base Agent.

Common base for all agents. Handles:
- Model routing via ModelRouter
- Structured output parsing (JSON with retry)
- Cost tracking (automatic per LLM call)
- Structured logging via structlog
- Prompt loading from prompts/*.yaml
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
import structlog

from lumen.core.router import ModelRouter
from lumen.core.cost import CostTracker

logger = structlog.get_logger()

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class LumenParseError(Exception):
    """Raised when JSON parsing fails after retries."""


class BaseAgent:
    """Base class for all LUMEN agents."""

    tier: str = "smart"        # override in subclass
    agent_name: str = ""       # override in subclass
    prompt_file: str = ""      # override in subclass

    def __init__(
        self,
        router: ModelRouter,
        cost_tracker: CostTracker,
        config: dict,
    ):
        self.router = router
        self.cost = cost_tracker
        self.config = config
        self.prompt_template: str = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load the system prompt from prompts/{self.prompt_file}."""
        if not self.prompt_file:
            return ""
        path = _PROMPTS_DIR / self.prompt_file
        if not path.exists():
            logger.warning("prompt_file_missing", path=str(path), agent=self.agent_name)
            return ""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        # Support both plain string and dict with 'system_prompt' key
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return data.get("system_prompt", data.get("prompt", ""))
        return ""

    def _call_llm(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        phase: str = "",
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Central LLM call point. All subclasses should use this.

        Handles: routing -> call -> cost tracking -> logging -> return.
        """
        text, usage = self.router.call(
            tier=self.tier,
            messages=messages,
            response_format=response_format,
            agent_name=self.agent_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.cost.record(phase=phase, agent=self.agent_name, usage=usage)
        logger.info(
            "llm_call",
            agent=self.agent_name,
            phase=phase,
            model=usage["model"],
            cost=usage["cost"],
            tokens=usage["input_tokens"] + usage["output_tokens"],
        )
        return text

    def _parse_json(self, text: str, retry_messages: list[dict] | None = None,
                    phase: str = "") -> dict:
        """Parse LLM response as JSON.

        - Strip markdown fences (```json ... ```)
        - Try json.loads
        - If fail and retry_messages provided: ONE retry asking for valid JSON
        - If still fail: raise LumenParseError
        """
        cleaned = self._strip_markdown_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # One retry if we have messages context
        if retry_messages:
            logger.warning("json_parse_retry", agent=self.agent_name,
                           original_text=text[:200])
            retry_msgs = retry_messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    "Your previous response was not valid JSON. "
                    "Please respond with ONLY valid JSON, no explanation."
                )},
            ]
            retry_text = self._call_llm(retry_msgs, phase=phase)
            cleaned = self._strip_markdown_fences(retry_text)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise LumenParseError(
                    f"Agent '{self.agent_name}' failed to produce valid JSON after retry. "
                    f"Last response: {retry_text[:300]}"
                ) from e

        raise LumenParseError(
            f"Agent '{self.agent_name}' returned invalid JSON: {text[:300]}"
        )

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove ```json ... ```, ```js ... ```, or ``` ... ``` wrappers."""
        text = text.strip()
        pattern = r"^```(?:json|js|javascript)?\s*\n?(.*?)\n?\s*```$"
        match = re.match(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def _build_messages(self, user_content: str,
                        system_override: str | None = None) -> list[dict]:
        """Build a standard [system, user] message list."""
        system = system_override or self.prompt_template
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def run(self, state: dict) -> dict:
        """Override in subclass. Takes state, returns state updates."""
        raise NotImplementedError(f"{self.agent_name}.run() not implemented")
