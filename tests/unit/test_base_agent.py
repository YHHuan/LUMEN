"""Tests for BaseAgent."""
import pytest
from unittest.mock import MagicMock, patch

from lumen.agents.base import BaseAgent, LumenParseError
from lumen.core.router import ModelRouter
from lumen.core.cost import CostTracker


class DummyAgent(BaseAgent):
    tier = "fast"
    agent_name = "dummy"
    prompt_file = ""  # no prompt file

    def run(self, state: dict) -> dict:
        return {"current_phase": "test"}


@pytest.fixture
def mock_router():
    router = MagicMock(spec=ModelRouter)
    return router


@pytest.fixture
def agent(mock_router):
    cost = CostTracker()
    return DummyAgent(router=mock_router, cost_tracker=cost, config={})


def test_agent_creation(agent):
    assert agent.agent_name == "dummy"
    assert agent.tier == "fast"


def test_agent_run(agent):
    result = agent.run({})
    assert result == {"current_phase": "test"}


def test_agent_not_implemented():
    """BaseAgent.run() should raise NotImplementedError."""
    router = MagicMock(spec=ModelRouter)
    cost = CostTracker()
    base = BaseAgent(router=router, cost_tracker=cost, config={})
    with pytest.raises(NotImplementedError):
        base.run({})


def test_call_llm(agent, mock_router):
    mock_router.call.return_value = ("response text", {
        "model": "test", "input_tokens": 10, "output_tokens": 5,
        "cost": 0.001, "latency_ms": 100,
    })

    result = agent._call_llm(
        messages=[{"role": "user", "content": "hi"}],
        phase="phase1",
    )

    assert result == "response text"
    mock_router.call.assert_called_once()
    # Cost should be recorded
    assert agent.cost.summary()["grand_total_calls"] == 1


def test_parse_json_valid(agent):
    result = agent._parse_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_with_fences(agent):
    result = agent._parse_json('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}


def test_parse_json_invalid_raises(agent):
    with pytest.raises(LumenParseError):
        agent._parse_json("not json at all")


def test_parse_json_retry(agent, mock_router):
    """Invalid JSON triggers a retry call."""
    mock_router.call.return_value = ('{"fixed": true}', {
        "model": "test", "input_tokens": 10, "output_tokens": 5,
        "cost": 0.001, "latency_ms": 50,
    })

    result = agent._parse_json(
        "not valid json",
        retry_messages=[{"role": "user", "content": "give me json"}],
        phase="phase1",
    )
    assert result == {"fixed": True}


def test_strip_markdown_fences():
    assert BaseAgent._strip_markdown_fences('```json\n{"a":1}\n```') == '{"a":1}'
    assert BaseAgent._strip_markdown_fences('```\n{"a":1}\n```') == '{"a":1}'
    assert BaseAgent._strip_markdown_fences('{"a":1}') == '{"a":1}'


def test_build_messages(agent):
    msgs = agent._build_messages("hello world")
    assert len(msgs) == 1  # no prompt file loaded
    assert msgs[0]["role"] == "user"

    msgs = agent._build_messages("hello", system_override="system prompt")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
