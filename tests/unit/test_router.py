"""Tests for ModelRouter."""
import pytest
from unittest.mock import patch, MagicMock

from lumen.core.router import ModelRouter, LumenModelError


@pytest.fixture
def mock_config():
    return {
        "tiers": {
            "fast": {
                "primary": "gemini/gemini-2.0-flash",
                "fallback": "openai/gpt-4.1-mini",
                "max_tokens": 4096,
                "cost_per_1k_input": 0.00010,
                "cost_per_1k_output": 0.00040,
            },
            "smart": {
                "primary": "anthropic/claude-sonnet-4-20250514",
                "fallback": "gemini/gemini-2.5-pro",
                "max_tokens": 8192,
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.015,
            },
            "strategic": {
                "primary": "anthropic/claude-opus-4-20250514",
                "fallback": "openai/o3",
                "max_tokens": 16384,
                "cost_per_1k_input": 0.015,
                "cost_per_1k_output": 0.075,
            },
        }
    }


@pytest.fixture
def router(mock_config):
    return ModelRouter(config=mock_config)


def _make_mock_response(text="hello", input_tokens=10, output_tokens=5):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    resp.usage = usage
    return resp


def test_router_instantiation(router):
    assert "fast" in router._tiers
    assert "smart" in router._tiers
    assert "strategic" in router._tiers


def test_router_unknown_tier(router):
    with pytest.raises(KeyError, match="Unknown model tier"):
        router.call(tier="ultra", messages=[{"role": "user", "content": "hi"}])


@patch("lumen.core.router.litellm.completion")
def test_router_successful_call(mock_completion, router):
    mock_completion.return_value = _make_mock_response("test response", 100, 50)

    text, usage = router.call(
        tier="fast",
        messages=[{"role": "user", "content": "hello"}],
        agent_name="test_agent",
    )

    assert text == "test response"
    assert usage["model"] == "gemini/gemini-2.0-flash"
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["cost"] > 0
    assert usage["latency_ms"] >= 0
    assert usage["fallback_used"] is False


@patch("lumen.core.router.litellm.completion")
def test_router_fallback_on_primary_failure(mock_completion, router):
    """When primary fails, fallback should be tried."""
    mock_completion.side_effect = [
        Exception("primary down"),
        _make_mock_response("fallback response"),
    ]

    text, usage = router.call(
        tier="fast",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert text == "fallback response"
    assert usage["fallback_used"] is True
    assert mock_completion.call_count == 2


@patch("lumen.core.router.litellm.completion")
def test_router_raises_when_both_fail(mock_completion, router):
    """LumenModelError raised when primary and fallback both fail."""
    mock_completion.side_effect = Exception("all down")

    with pytest.raises(LumenModelError) as exc_info:
        router.call(tier="fast", messages=[{"role": "user", "content": "hi"}])

    assert exc_info.value.tier == "fast"
    assert exc_info.value.primary_error is not None
    assert exc_info.value.fallback_error is not None


@patch("lumen.core.router.litellm.completion")
def test_router_cost_calculation(mock_completion, router):
    """Cost should be calculated from tier pricing."""
    mock_completion.return_value = _make_mock_response("ok", 1000, 500)

    _, usage = router.call(tier="fast", messages=[{"role": "user", "content": "hi"}])

    # fast tier: input 0.00010/1k, output 0.00040/1k
    expected = (1000 / 1000 * 0.00010) + (500 / 1000 * 0.00040)
    assert abs(usage["cost"] - expected) < 1e-6
