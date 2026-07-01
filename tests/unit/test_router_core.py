"""Router execution paths: fallback ladder, budgets, tier order, streaming guards.

All offline - providers are faked, no network.
"""

import asyncio
import time

import pytest

from bkvy.core.router import IntelligentRouter
from bkvy.core.rate_limits import RateLimitManager
from bkvy.core.queues import QueueManager
from bkvy.core.circuit_breaker import CircuitBreakerManager
from bkvy.models.data_classes import CompletionTimeAnalysis
from bkvy.models.schemas import Message
from bkvy.models.circuit_states import CircuitStatus


def run(coro):
    return asyncio.run(coro)


# --- fakes -------------------------------------------------------------------

class FakeModelConfig:
    endpoint = "http://fake"
    version = None
    intelligence_tier = "low"
    supports_thinking = False
    avg_response_time_ms = 1000
    cost_per_1k_tokens = 0.0


class FakeKeyConfig:
    def __init__(self, key_id):
        self.api_key = f"key-{key_id}"
        self.rate_limits = {"m": {"rpm": 100, "rpd": 1000}}


class FakeProviderConfig:
    def __init__(self, key_ids):
        self.keys = {k: FakeKeyConfig(k) for k in key_ids}
        self.models = {"m": FakeModelConfig()}


class FakeConfig:
    def __init__(self, key_ids=("k1", "k2")):
        self.providers = {"p": FakeProviderConfig(key_ids)}


class ScriptedLLMClient:
    """Non-streaming client whose outcome depends on the api_key used."""

    def __init__(self, script):
        # script: api_key -> Exception to raise, or dict response to return
        self.script = script
        self.calls = []

    async def _make_api_call(self, provider, model, api_key, messages, options,
                             endpoint, version=None):
        self.calls.append(api_key)
        outcome = self.script[api_key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def analysis_for(key_id):
    return CompletionTimeAnalysis(
        rate_limit_wait_seconds=0, queue_wait_seconds=0, processing_time_seconds=1,
        total_seconds=1, cost_per_1k_tokens=0.0, combination_key=f"p_{key_id}_m",
        provider="p", model="m", api_key_id=key_id)


def make_router(llm_client, config=None, circuit_breaker=None, tmp_path=None):
    rate_mgr = RateLimitManager(state_dir=str(tmp_path))
    return IntelligentRouter(config or FakeConfig(), rate_mgr, QueueManager(),
                             llm_client, circuit_breaker=circuit_breaker)


class FakePersistence:
    async def save_state(self, circuit):
        pass

    async def load_all_states(self):
        return {}

    async def cleanup_old_states(self, valid):
        return 0


# --- tier order --------------------------------------------------------------

class TestTierOrder:
    def test_requested_then_up_then_down(self, monkeypatch):
        monkeypatch.delenv("TIER_DOWNGRADE_ENABLED", raising=False)
        assert IntelligentRouter._tiers_to_try("low") == ["low", "medium", "high"]
        assert IntelligentRouter._tiers_to_try("medium") == ["medium", "high", "low"]
        assert IntelligentRouter._tiers_to_try("high") == ["high", "medium", "low"]

    def test_downgrade_disabled(self, monkeypatch):
        monkeypatch.setenv("TIER_DOWNGRADE_ENABLED", "false")
        assert IntelligentRouter._tiers_to_try("high") == ["high"]
        assert IntelligentRouter._tiers_to_try("medium") == ["medium", "high"]
        assert IntelligentRouter._tiers_to_try("low") == ["low", "medium", "high"]


# --- synchronous fallback ladder ----------------------------------------------

class TestFallbackLadder:
    def test_rate_limited_key_fails_over(self, tmp_path):
        """First key 429s -> router moves to the second key and succeeds."""
        client = ScriptedLLMClient({
            "key-k1": Exception("OpenAI API error 429: rate limit reached, try again in 20s"),
            "key-k2": {"content": "hello", "usage": {"input_tokens": 1, "output_tokens": 1,
                                                     "total_tokens": 2}},
        })
        router = make_router(client, tmp_path=tmp_path)

        async def scenario():
            return await router._execute_with_retry_and_fallback(
                [analysis_for("k1"), analysis_for("k2")],
                [Message(role="user", content="hi")], None,
                time.time(), "req-1")

        result, used, info = run(scenario())
        assert result["success"]
        assert used.api_key_id == "k2"
        assert client.calls == ["key-k1", "key-k2"]  # no pointless retries on a 429
        assert info["failures"][0]["strategy"] == "skip_alternative"

    def test_model_error_skips_alternative_with_circuit_breaker(self, tmp_path):
        client = ScriptedLLMClient({
            "key-k1": Exception("OpenAI API error 404: The model `m` does not exist "
                                "or you do not have access to it."),
            "key-k2": {"content": "ok", "usage": {"input_tokens": 1, "output_tokens": 1,
                                                  "total_tokens": 2}},
        })
        config = FakeConfig()
        breaker = CircuitBreakerManager(config, persistence=FakePersistence())
        breaker.enabled = True
        router = make_router(client, config=config, circuit_breaker=breaker, tmp_path=tmp_path)

        async def scenario():
            return await router._execute_with_retry_and_fallback(
                [analysis_for("k1"), analysis_for("k2")],
                [Message(role="user", content="hi")], None,
                time.time(), "req-2")

        result, used, info = run(scenario())
        assert result["success"]
        assert used.api_key_id == "k2"
        # Model errors must not be retried against the same combination
        assert client.calls == ["key-k1", "key-k2"]
        circuit = breaker._get_or_create_circuit("p", "m", "k1")
        assert circuit.consecutive_failures == 1

    def test_budget_exhaustion_fails_fast(self, tmp_path):
        """A microscopic budget must fail without burning retry sleeps."""
        client = ScriptedLLMClient({
            "key-k1": Exception("connection reset"),  # classified retryable
            "key-k2": Exception("connection reset"),
        })
        router = make_router(client, tmp_path=tmp_path)

        async def scenario():
            started = time.time()
            result, _, info = await router._execute_with_retry_and_fallback(
                [analysis_for("k1"), analysis_for("k2")],
                [Message(role="user", content="hi")], None,
                started, "req-3", budget_seconds=0.5)
            return result, info, time.time() - started

        result, info, elapsed = run(scenario())
        assert not result["success"]
        assert elapsed < 3  # no exponential backoff sleeps past the budget


# --- streaming ----------------------------------------------------------------

class TruncatingStreamClient:
    """Emits deltas then ends WITHOUT a done event (mid-commit break)."""

    async def _make_api_call_stream(self, **kwargs):
        yield {"type": "delta", "content": "partial "}
        yield {"type": "delta", "content": "output"}


class EmptyStreamClient:
    """Ends immediately: no delta, no done (pre-commit break)."""

    async def _make_api_call_stream(self, **kwargs):
        if False:
            yield


class HealthyStreamClient:
    async def _make_api_call_stream(self, **kwargs):
        yield {"type": "delta", "content": "hi"}
        yield {"type": "done", "finish_reason": "stop",
               "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}


class TestStreaming:
    def _collect(self, router, analyses, budget=None):
        async def scenario():
            events = []
            async for ev in router._execute_with_fallback_stream(
                    analyses, [Message(role="user", content="hi")], None,
                    time.time(), "req-s", budget_seconds=budget):
                events.append(ev)
            return events

        return run(scenario())

    def test_committed_break_is_terminal_never_spliced(self, tmp_path):
        router = make_router(TruncatingStreamClient(), tmp_path=tmp_path)
        events = self._collect(router, [analysis_for("k1"), analysis_for("k2")])
        types = [e["type"] for e in events]
        assert types[-1] == "error"
        assert "without completion" in events[-1]["error"]
        assert types.count("start") == 1  # second alternative never started

    def test_precommit_break_falls_back(self, tmp_path):
        router = make_router(EmptyStreamClient(), tmp_path=tmp_path)
        events = self._collect(router, [analysis_for("k1")])
        assert events[-1]["type"] == "_tier_failed"

    def test_happy_path_single_done(self, tmp_path):
        router = make_router(HealthyStreamClient(), tmp_path=tmp_path)
        events = self._collect(router, [analysis_for("k1")])
        types = [e["type"] for e in events]
        assert types == ["_committed", "start", "delta", "done"]
        assert events[-1]["content"] == "hi"

    def test_concurrency_slot_shortage_is_local_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAX_CONCURRENT_PER_COMBINATION", "1")
        router = make_router(HealthyStreamClient(), tmp_path=tmp_path)

        async def scenario():
            # Occupy the only slot, then try to stream with a tiny budget
            assert await router.queues.try_acquire_slot("p", "m", "k1", timeout=0.05)
            events = []
            async for ev in router._execute_with_fallback_stream(
                    [analysis_for("k1")], [Message(role="user", content="hi")], None,
                    time.time(), "req-s2", budget_seconds=1):
                events.append(ev)
            return events

        events = run(scenario())
        assert events[-1]["type"] == "_tier_failed"
        assert "concurrency slot" in events[-1]["error"]
