"""Sliding-window rate limiting and state persistence."""

import asyncio
import json
from datetime import datetime, timezone, timedelta

import pytest

from bkvy.core.rate_limits import RateLimitManager, RPM_WINDOW_SECONDS


def run(coro):
    return asyncio.run(coro)


class TestTryConsume:
    def test_respects_rpm(self, tmp_path):
        async def scenario():
            mgr = RateLimitManager(state_dir=str(tmp_path))
            results = []
            for _ in range(4):
                ok, _ = await mgr.try_consume("openai", "m", "k", rpm_limit=3, rpd_limit=100)
                results.append(ok)
            return results

        assert run(scenario()) == [True, True, True, False]

    def test_concurrent_no_overshoot(self, tmp_path):
        async def scenario():
            mgr = RateLimitManager(state_dir=str(tmp_path))
            outcomes = await asyncio.gather(*[
                mgr.try_consume("openai", "m", "k", rpm_limit=5, rpd_limit=100)
                for _ in range(20)
            ])
            return sum(1 for ok, _ in outcomes if ok)

        assert run(scenario()) == 5

    def test_zero_limits_mean_unlimited(self, tmp_path):
        async def scenario():
            mgr = RateLimitManager(state_dir=str(tmp_path))
            results = [await mgr.try_consume("openai", "m", "k", rpm_limit=0, rpd_limit=0)
                       for _ in range(50)]
            return all(ok for ok, _ in results)

        assert run(scenario())


class TestSlidingWindow:
    def test_boundary_burst_blocked(self, tmp_path):
        """10 requests 30s ago must still block an 11th now (calendar-minute
        windows would have reset at the minute boundary and allowed 2x RPM)."""
        async def scenario():
            mgr = RateLimitManager(state_dir=str(tmp_path))
            key = mgr._get_combination_key("gemini", "m", "k")
            # Prime: consume once to create state, then rewrite the window
            await mgr.try_consume("gemini", "m", "k", rpm_limit=10, rpd_limit=1000)
            now = datetime.now(timezone.utc)
            mgr.states[key].recent_request_times = [
                now - timedelta(seconds=30) for _ in range(10)
            ]
            ok, wait = await mgr.try_consume("gemini", "m", "k", rpm_limit=10, rpd_limit=1000)
            return ok, wait

        ok, wait = run(scenario())
        assert not ok
        assert 25 <= wait <= RPM_WINDOW_SECONDS  # ~30s until the oldest ages out

    def test_expired_window_admits(self, tmp_path):
        async def scenario():
            mgr = RateLimitManager(state_dir=str(tmp_path))
            key = mgr._get_combination_key("gemini", "m", "k")
            await mgr.try_consume("gemini", "m", "k", rpm_limit=10, rpd_limit=1000)
            now = datetime.now(timezone.utc)
            mgr.states[key].recent_request_times = [
                now - timedelta(seconds=61) for _ in range(10)
            ]
            ok, _ = await mgr.try_consume("gemini", "m", "k", rpm_limit=10, rpd_limit=1000)
            return ok

        assert run(scenario())


class TestDailyReset:
    def test_gemini_resets_at_pacific_midnight(self):
        reset = RateLimitManager._next_day_reset("gemini")
        # Midnight Pacific is 07:00 or 08:00 UTC depending on DST
        assert reset.astimezone(timezone.utc).hour in (7, 8)

    def test_others_reset_at_utc_midnight(self):
        reset = RateLimitManager._next_day_reset("openai")
        assert reset.hour == 0 and reset.minute == 0


class TestPersistence:
    def test_roundtrip_preserves_window(self, tmp_path):
        async def scenario():
            mgr = RateLimitManager(state_dir=str(tmp_path))
            await mgr.try_consume("openai", "m", "k", rpm_limit=10, rpd_limit=100)
            await mgr.try_consume("openai", "m", "k", rpm_limit=10, rpd_limit=100)

            fresh = RateLimitManager(state_dir=str(tmp_path))
            key = fresh._get_combination_key("openai", "m", "k")
            state = await fresh._load_state(key)
            return state

        state = run(scenario())
        assert len(state.recent_request_times) == 2
        assert state.requests_today == 2

    def test_legacy_state_file_loads(self, tmp_path):
        """Old-format files (no recent_request_times) must load cleanly."""
        legacy = {
            "requests_this_minute": 3,
            "requests_today": 42,
            "minute_reset_time": "2026-01-01T00:01:00+00:00",
            "day_reset_time": "2026-01-02T00:00:00+00:00",
            "currently_rate_limited": False,
            "rate_limit_wait_seconds": 0,
            "rpm_limit": 10,
            "rpd_limit": 250,
            "last_request_time": "2026-01-01T00:00:30+00:00",
        }
        (tmp_path / "openai_k_m.json").write_text(json.dumps(legacy))

        async def scenario():
            mgr = RateLimitManager(state_dir=str(tmp_path))
            return await mgr._load_state("openai_k_m")

        state = run(scenario())
        assert state.recent_request_times == []
        assert state.requests_today == 42

    def test_save_is_atomic_no_tmp_left_behind(self, tmp_path):
        async def scenario():
            mgr = RateLimitManager(state_dir=str(tmp_path))
            await mgr.try_consume("openai", "m", "k", rpm_limit=10, rpd_limit=100)

        run(scenario())
        files = [p.name for p in tmp_path.iterdir()]
        assert "openai_k_m.json" in files
        assert not [f for f in files if f.endswith(".tmp")]
