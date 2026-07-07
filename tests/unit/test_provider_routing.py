"""Tier ↔ provider decoupling tests (stabilize P3.5).

Confirms v3's router is already provider-agnostic: any tier can name any
provider/model (direct anthropic/openai/gemini or openrouter), and tier names
are not fixed to {fast, smart, strategic}. The model string is passed straight
to LiteLLM, which routes by prefix.
"""
from unittest.mock import patch, MagicMock

from lumen.core.config import get_tier_config
from lumen.core.router import ModelRouter


def _mock_response():
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "ok"
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    resp.usage = usage
    return resp


# Every tier names a DIFFERENT provider directly; plus a non-standard tier name.
MIXED_CONFIG = {
    "tiers": {
        "fast": {"primary": "gemini/gemini-2.5-flash", "max_tokens": 4096,
                 "cost_per_1k_input": 0.0001, "cost_per_1k_output": 0.0004},
        "smart": {"primary": "anthropic/claude-sonnet-4-5", "max_tokens": 8192,
                  "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015},
        "strategic": {"primary": "openai/gpt-5", "max_tokens": 16384,
                      "cost_per_1k_input": 0.015, "cost_per_1k_output": 0.075},
        # arbitrary, non-standard tier name pointing at a direct provider
        "ultra": {"primary": "anthropic/claude-opus-4-1", "max_tokens": 16384,
                  "cost_per_1k_input": 0.015, "cost_per_1k_output": 0.075},
    }
}


class TestTierProviderDecoupling:
    def test_custom_tier_name_resolves(self):
        cfg = get_tier_config(MIXED_CONFIG, "ultra")
        assert cfg["primary"] == "anthropic/claude-opus-4-1"

    @patch("lumen.core.router.litellm.completion")
    def test_each_tier_routes_to_its_named_provider(self, mock_completion):
        mock_completion.return_value = _mock_response()
        router = ModelRouter(config=MIXED_CONFIG)
        for tier, expected_model in [
            ("fast", "gemini/gemini-2.5-flash"),
            ("smart", "anthropic/claude-sonnet-4-5"),
            ("strategic", "openai/gpt-5"),
            ("ultra", "anthropic/claude-opus-4-1"),
        ]:
            _, usage = router.call(tier=tier, messages=[{"role": "user", "content": "x"}])
            assert usage["model"] == expected_model
            # the model string is passed straight through to LiteLLM
            assert mock_completion.call_args.kwargs["model"] == expected_model

    @patch("lumen.core.router.litellm.completion")
    def test_openrouter_prefix_also_works(self, mock_completion):
        """A tier may equally name an OpenRouter-routed model."""
        mock_completion.return_value = _mock_response()
        cfg = {"tiers": {"fast": {"primary": "openrouter/google/gemini-2.5-flash",
                                  "max_tokens": 4096}}}
        router = ModelRouter(config=cfg)
        _, usage = router.call(tier="fast", messages=[{"role": "user", "content": "x"}])
        assert usage["model"] == "openrouter/google/gemini-2.5-flash"
