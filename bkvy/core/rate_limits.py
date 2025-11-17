"""
Rate limit management for bkvy
"""

import asyncio
import json
import aiofiles
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import asdict

from ..models.data_classes import RateLimitState
from ..utils.logging import setup_logging

logger = setup_logging()


class RateLimitManager:
    """Manages rate limiting for all (API_KEY, MODEL) combinations"""
    
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
                if field.endswith('_time') and value:
                    setattr(state, field, datetime.fromisoformat(value))
                else:
                    setattr(state, field, value)
            
            return state
            
        except Exception as e:
            logger.warning("Failed to load rate limit state", 
                         combination=combination_key, error=str(e))
            return RateLimitState()
    
    async def _save_state(self, combination_key: str, state: RateLimitState):
        """Save rate limit state to file"""
        state_file = self.state_dir / f"{combination_key}.json"
        
        try:
            # Convert datetime fields to ISO format
            data = {}
            for field, value in asdict(state).items():
                if field.endswith('_time') and value:
                    data[field] = value.isoformat()
                else:
                    data[field] = value
            
            async with aiofiles.open(state_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            logger.error("Failed to save rate limit state", 
                        combination=combination_key, error=str(e))
    
    async def _update_rate_counters(self, state: RateLimitState):
        """Update rate limit counters and reset times"""
        now = datetime.now(timezone.utc)
        
        # Reset minute counter if needed
        if now >= state.minute_reset_time:
            state.requests_this_minute = 0
            state.minute_reset_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Reset daily counter if needed
        if now >= state.day_reset_time:
            state.requests_today = 0
            state.day_reset_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    async def check_rate_limit_status(self, provider: str, model: str, api_key_id: str, 
                                    rpm_limit: int, rpd_limit: int) -> Tuple[bool, float]:
        """Check if combination is rate limited and return wait time"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        lock = self._get_lock(combination_key)
        
        async with lock:
            if combination_key not in self.states:
                self.states[combination_key] = await self._load_state(combination_key)
            
            state = self.states[combination_key]
            state.rpm_limit = rpm_limit
            state.rpd_limit = rpd_limit
            
            await self._update_rate_counters(state)
            
            now = datetime.now(timezone.utc)
            
            # Check if rate limited
            if state.requests_this_minute >= rpm_limit:
                wait_seconds = (state.minute_reset_time - now).total_seconds()
                state.currently_rate_limited = True
                state.rate_limit_wait_seconds = max(0, wait_seconds)
                return True, wait_seconds
            
            if state.requests_today >= rpd_limit:
                wait_seconds = (state.day_reset_time - now).total_seconds()
                state.currently_rate_limited = True
                state.rate_limit_wait_seconds = max(0, wait_seconds)
                return True, wait_seconds
            
            state.currently_rate_limited = False
            state.rate_limit_wait_seconds = 0
            return False, 0
    
    async def record_request(self, provider: str, model: str, api_key_id: str):
        """Record a request for rate limiting purposes"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        lock = self._get_lock(combination_key)
        
        async with lock:
            if combination_key not in self.states:
                self.states[combination_key] = await self._load_state(combination_key)
            
            state = self.states[combination_key]
            await self._update_rate_counters(state)
            
            state.requests_this_minute += 1
            state.requests_today += 1
            state.last_request_time = datetime.now(timezone.utc)
            
            await self._save_state(combination_key, state)
    
    async def get_all_states(self) -> Dict[str, Dict[str, any]]:
        """Get all rate limit states for monitoring"""
        states = {}
        for combination_key, state in self.states.items():
            await self._update_rate_counters(state)
            states[combination_key] = asdict(state)
        return states