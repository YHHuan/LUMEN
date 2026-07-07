"""Prompt-cache skeleton + evidence-run tests (stabilize P0 caching / P1.7)."""
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from lumen.infra.llm_cache import PromptCache, make_cache_key, cache_enabled_from_env
from lumen.core.config import (
    is_caching_enabled, is_evidence_run, apply_evidence_run,
)
from lumen.core.router import ModelRouter


MESSAGES = [{"role": "user", "content": "hello"}]


class TestCacheKey:
    def test_deterministic(self):
        k1 = make_cache_key("m", MESSAGES, max_tokens=100, temperature=0.0)
        k2 = make_cache_key("m", MESSAGES, max_tokens=100, temperature=0.0)
        assert k1 == k2

    def test_changes_with_model(self):
        assert make_cache_key("m1", MESSAGES) != make_cache_key("m2", MESSAGES)

    def test_changes_with_messages(self):
        other = [{"role": "user", "content": "world"}]
        assert make_cache_key("m", MESSAGES) != make_cache_key("m", other)


class TestPromptCache:
    def test_put_get_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = PromptCache(tmp)
            c.put("k1", "cached text", input_tokens=100, output_tokens=20)
            got = c.get("k1")
            assert got["text"] == "cached text"
            assert got["input_tokens"] == 100

    def test_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = PromptCache(tmp)
            assert c.get("nope") is None
            assert c.misses == 1

    def test_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = PromptCache(tmp)
            c.put("k", "t")
            c.get("k")      # hit
            c.get("x")      # miss
            m = c.manifest()
            assert m["hits"] == 1 and m["misses"] == 1
            assert m["hit_rate"] == 0.5

    def test_env_flag(self, monkeypatch):
        monkeypatch.setenv("LUMEN_LLM_CACHE", "1")
        assert cache_enabled_from_env()
        monkeypatch.setenv("LUMEN_LLM_CACHE", "0")
        assert not cache_enabled_from_env()


class TestCachingConfigGuards:
    def test_off_by_default(self):
        assert is_caching_enabled({}) is False
        assert is_caching_enabled({"tiers": {}}) is False

    def test_on_when_flag_set(self):
        assert is_caching_enabled({"prompt_caching": True}) is True
        assert is_caching_enabled({"defaults": {"prompt_caching": True}}) is True

    def test_evidence_run_forces_off(self):
        cfg = {"prompt_caching": True, "evidence_run": True}
        assert is_evidence_run(cfg) is True
        assert is_caching_enabled(cfg) is False  # evidence run wins

    def test_apply_evidence_run(self):
        cfg = {"defaults": {"prompt_caching": True}}
        apply_evidence_run(cfg)
        assert is_evidence_run(cfg) is True
        assert is_caching_enabled(cfg) is False


def _mock_response(text="live text", inp=100, out=20):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    usage = MagicMock()
    usage.prompt_tokens = inp
    usage.completion_tokens = out
    resp.usage = usage
    return resp


TIER_CFG = {
    "tiers": {"fast": {"primary": "gemini/gemini-2.0-flash", "max_tokens": 4096,
                       "cost_per_1k_input": 0.0001, "cost_per_1k_output": 0.0004}}
}


class TestRouterCaching:
    @patch("lumen.core.router.litellm.completion")
    def test_caching_off_calls_llm_every_time(self, mock_completion):
        """Default (no caching) → v3 behaviour: every call hits the LLM."""
        mock_completion.return_value = _mock_response()
        router = ModelRouter(config=dict(TIER_CFG))  # no cache, caching off
        router.call(tier="fast", messages=MESSAGES)
        router.call(tier="fast", messages=MESSAGES)
        assert mock_completion.call_count == 2

    @patch("lumen.core.router.litellm.completion")
    def test_caching_on_serves_second_call_from_cache(self, mock_completion):
        mock_completion.return_value = _mock_response("hi", inp=100, out=20)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(TIER_CFG)
            cfg["prompt_caching"] = True
            router = ModelRouter(config=cfg, cache=PromptCache(tmp))
            text1, u1 = router.call(tier="fast", messages=MESSAGES)
            text2, u2 = router.call(tier="fast", messages=MESSAGES)
            assert text1 == text2 == "hi"
            assert mock_completion.call_count == 1        # second served from cache
            assert u2.get("cache_hit") is True
            assert u2["cached_input_tokens"] == 100
            assert u2["cost"] == 0.0

    @patch("lumen.core.router.litellm.completion")
    def test_evidence_run_disables_cache(self, mock_completion):
        mock_completion.return_value = _mock_response()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(TIER_CFG)
            cfg["prompt_caching"] = True
            apply_evidence_run(cfg)  # evidence run must override caching
            router = ModelRouter(config=cfg, cache=PromptCache(tmp))
            router.call(tier="fast", messages=MESSAGES)
            router.call(tier="fast", messages=MESSAGES)
            assert mock_completion.call_count == 2  # no cache reuse
