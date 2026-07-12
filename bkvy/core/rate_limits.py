"""
Rate limit management for bkvy
"""

import asyncio
import json
import os
import aiofiles
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import asdict

from ..models.data_classes import RateLimitState
from ..utils.logging import setup_logging

logger = setup_logging()

# RPM is by definition a 60-second window
RPM_WINDOW_SECONDS = 60.0


class RateLimitManager:
    """Manages rate limiting for all (API_KEY, MODEL) combinations.

    RPM is enforced with a true sliding window of request timestamps, so a
    burst straddling a calendar-minute boundary cannot exceed the provider's
    real limit. RPD remains a counter that resets on the provider's actual
    daily boundary (midnight Pacific for Gemini, UTC midnight otherwise).
    """

    def __init__(self, state_dir: str = "rate_states"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.states: Dict[str, RateLimitState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_combination_key(self, provider: str, model: str, api_key_id: str) -> str:
        """Generate unique key for (provider, model, api_key_id) combination"""
        return f"{provider}_{api_key_id}_{model}"

    def _get_lock(self, combination_key: str) -> asyncio.Lock:
        """Get or create lock for a combination"""
        if combination_key not in self._locks:
            self._locks[combination_key] = asyncio.Lock()
        return self._locks[combination_key]

    async def _load_state(self, combination_key: str) -> RateLimitState:
        """Load rate limit state from file"""
        state_file = self.state_dir / f"{combination_key}.json"

        if not state_file.exists():
            return RateLimitState()

        try:
            async with aiofiles.open(state_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            # Parse datetime fields
            state = RateLimitState()
            for field, value in data.items():
                if field == 'recent_request_times':
                    setattr(state, field, [datetime.fromisoformat(v) for v in (value or [])])
                elif field.endswith('_time') and value:
                    setattr(state, field, datetime.fromisoformat(value))
                else:
                    setattr(state, field, value)

            return state

        except Exception as e:
            logger.warning("Failed to load rate limit state",
                         combination=combination_key, error=str(e))
            return RateLimitState()

    async def _save_state(self, combination_key: str, state: RateLimitState):
        """Save rate limit state to file atomically (tmp file + rename)"""
        state_file = self.state_dir / f"{combination_key}.json"
        temp_file = state_file.with_suffix('.json.tmp')

        try:
            # Convert datetime fields to ISO format
            data = {}
            for field, value in asdict(state).items():
                if field == 'recent_request_times':
                    data[field] = [v.isoformat() for v in (value or [])]
                elif field.endswith('_time') and value:
                    data[field] = value.isoformat()
                else:
                    data[field] = value

            async with aiofiles.open(temp_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))

            # Atomic replace so a crash mid-write cannot leave torn JSON
            os.replace(temp_file, state_file)

        except Exception as e:
            logger.error("Failed to save rate limit state",
                        combination=combination_key, error=str(e))

    @staticmethod
    def _next_day_reset(provider: str) -> datetime:
        """Next daily-quota reset time (UTC) for a provider.

        Gemini's free-tier RPD resets at midnight Pacific (same convention the
        failure classifier uses for 429 backoff); everything else uses UTC
        midnight. Keeping the two in agreement matters: an early local reset
        would over-admit requests against a still-exhausted upstream quota.
        """
        now_utc = datetime.now(timezone.utc)

        if provider == "gemini":
            try:
                from zoneinfo import ZoneInfo
                pacific = ZoneInfo("America/Los_Angeles")
            except Exception:
                pacific = timezone(timedelta(hours=-8))
            now_local = now_utc.astimezone(pacific)
            midnight = (now_local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return midnight.astimezone(timezone.utc)

        return (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    def _prune_window(self, state: RateLimitState, now: datetime):
        """Drop window entries older than 60s and sync the derived counters"""
        cutoff = now - timedelta(seconds=RPM_WINDOW_SECONDS)
        state.recent_request_times = [t for t in state.recent_request_times if t > cutoff]
        state.requests_this_minute = len(state.recent_request_times)
        if state.recent_request_times:
            state.minute_reset_time = state.recent_request_times[0] + timedelta(seconds=RPM_WINDOW_SECONDS)
        else:
            state.minute_reset_time = now + timedelta(seconds=RPM_WINDOW_SECONDS)

    async def _update_rate_counters(self, state: RateLimitState, provider: str):
        """Update rate limit counters and reset times"""
        now = datetime.now(timezone.utc)

        self._prune_window(state, now)

        # Reset daily counter if needed
        if now >= state.day_reset_time:
            state.requests_today = 0
            state.day_reset_time = self._next_day_reset(provider)

    def _window_status(self, state: RateLimitState, rpm_limit: int, rpd_limit: int,
                       now: datetime) -> Tuple[bool, float]:
        """Shared limit check. Assumes counters were just updated.

        Non-positive limits mean "unlimited" for that dimension.
        Returns (is_limited, seconds_until_a_slot_frees).
        """
        if rpm_limit > 0 and len(state.recent_request_times) >= rpm_limit:
            oldest = state.recent_request_times[0]
            wait_seconds = max(0.0, (oldest + timedelta(seconds=RPM_WINDOW_SECONDS) - now).total_seconds())
            return True, wait_seconds

        if rpd_limit > 0 and state.requests_today >= rpd_limit:
            wait_seconds = max(0.0, (state.day_reset_time - now).total_seconds())
            return True, wait_seconds

        return False, 0.0

    async def check_rate_limit_status(self, provider: str, model: str, api_key_id: str,
                                    rpm_limit: int, rpd_limit: int) -> Tuple[bool, float]:
        """Check if combination is rate limited and return wait time (read-only)"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        lock = self._get_lock(combination_key)

        async with lock:
            if combination_key not in self.states:
                self.states[combination_key] = await self._load_state(combination_key)

            state = self.states[combination_key]
            state.rpm_limit = rpm_limit
            state.rpd_limit = rpd_limit

            await self._update_rate_counters(state, provider)

            now = datetime.now(timezone.utc)
            is_limited, wait_seconds = self._window_status(state, rpm_limit, rpd_limit, now)

            state.currently_rate_limited = is_limited
            state.rate_limit_wait_seconds = wait_seconds
            return is_limited, wait_seconds

    async def record_request(self, provider: str, model: str, api_key_id: str):
        """Record a request for rate limiting purposes"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        lock = self._get_lock(combination_key)

        async with lock:
            if combination_key not in self.states:
                self.states[combination_key] = await self._load_state(combination_key)

            state = self.states[combination_key]
            await self._update_rate_counters(state, provider)

            now = datetime.now(timezone.utc)
            state.recent_request_times.append(now)
            state.requests_this_minute = len(state.recent_request_times)
            state.requests_today += 1
            state.last_request_time = now

            await self._save_state(combination_key, state)

    async def try_consume(self, provider: str, model: str, api_key_id: str,
                          rpm_limit: int, rpd_limit: int) -> Tuple[bool, float]:
        """Atomically check the rate limit and, if allowed, record the request.

        A separate check-then-record lets two concurrent requests both pass the
        check and overshoot the limit; this holds the combination lock across
        both steps. Returns (True, 0) when a slot was consumed, otherwise
        (False, seconds_until_a_slot_frees).
        """
        combination_key = self._get_combination_key(provider, model, api_key_id)
        lock = self._get_lock(combination_key)

        async with lock:
            if combination_key not in self.states:
                self.states[combination_key] = await self._load_state(combination_key)

            state = self.states[combination_key]
            state.rpm_limit = rpm_limit
            state.rpd_limit = rpd_limit

            await self._update_rate_counters(state, provider)

            now = datetime.now(timezone.utc)
            is_limited, wait_seconds = self._window_status(state, rpm_limit, rpd_limit, now)

            if is_limited:
                state.currently_rate_limited = True
                state.rate_limit_wait_seconds = wait_seconds
                return False, wait_seconds

            state.currently_rate_limited = False
            state.rate_limit_wait_seconds = 0
            state.recent_request_times.append(now)
            state.requests_this_minute = len(state.recent_request_times)
            state.requests_today += 1
            state.last_request_time = now

            await self._save_state(combination_key, state)
            return True, 0

    async def get_all_states(self) -> Dict[str, Dict[str, any]]:
        """Get all rate limit states for monitoring"""
        states = {}
        # list() snapshot: the await below yields to request handlers that may
        # add new combinations, and dict mutation mid-iteration raises.
        for combination_key, state in list(self.states.items()):
            # combination keys are "provider_apikeyid_model"; provider names
            # contain no underscore
            provider = combination_key.split("_", 1)[0]
            await self._update_rate_counters(state, provider)
            state_dict = asdict(state)
            # Keep the JSON response serializable
            state_dict["recent_request_times"] = [
                t.isoformat() for t in (state.recent_request_times or [])
            ]
            states[combination_key] = state_dict
        return states
