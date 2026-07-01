"""
Queue management for bkvy
"""

import asyncio
import os
import time
import uuid
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from dataclasses import asdict

from ..models.data_classes import QueueState
from ..utils.logging import setup_logging

logger = setup_logging()


class QueueManager:
    """Tracks in-flight load and executes requests for all (API_KEY, MODEL) combinations"""

    def __init__(self):
        self.states: Dict[str, QueueState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_combination_key(self, provider: str, model: str, api_key_id: str) -> str:
        """Generate unique key for (provider, model, api_key_id) combination"""
        return f"{provider}_{api_key_id}_{model}"

    def _get_lock(self, combination_key: str) -> asyncio.Lock:
        """Get or create lock for a combination"""
        if combination_key not in self._locks:
            self._locks[combination_key] = asyncio.Lock()
        return self._locks[combination_key]

    def _get_state(self, combination_key: str) -> QueueState:
        """Get or create queue state for a combination"""
        if combination_key not in self.states:
            self.states[combination_key] = QueueState()
        return self.states[combination_key]

    def note_inflight_start(self, provider: str, model: str, api_key_id: str):
        """Count a request against a combination (waiting or executing)."""
        state = self._get_state(self._get_combination_key(provider, model, api_key_id))
        state.current_queue_length += 1
        state.last_updated = datetime.now(timezone.utc)

    def note_inflight_end(self, provider: str, model: str, api_key_id: str):
        """Remove a request from a combination's in-flight count."""
        state = self._get_state(self._get_combination_key(provider, model, api_key_id))
        state.current_queue_length = max(0, state.current_queue_length - 1)
        state.last_updated = datetime.now(timezone.utc)

    async def get_queue_wait_time(self, provider: str, model: str, api_key_id: str,
                                avg_response_time_ms: int) -> float:
        """Calculate estimated wait time behind requests already in flight"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        state = self._get_state(combination_key)

        queue_length = state.current_queue_length
        avg_response_time_seconds = max(avg_response_time_ms / 1000, 0.1)  # Prevent division by zero

        estimated_wait = queue_length * avg_response_time_seconds

        # Update state
        state.estimated_queue_wait_seconds = estimated_wait
        state.last_updated = datetime.now(timezone.utc)

        return estimated_wait

    async def execute_request_directly(self, provider: str, model: str, api_key_id: str,
                                     request_data: Dict[str, Any], max_wait_seconds: int,
                                     rate_limit_manager, config_manager, llm_client,
                                     attempt_marker: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute request directly and wait for response.

        attempt_marker, when provided, gets "api_call_started" set to True the
        moment the upstream call begins. The router uses it to tell a provider
        timeout apart from time spent waiting for the local slot - only the
        former should count against the provider's circuit.
        """
        combination_key = self._get_combination_key(provider, model, api_key_id)
        lock = self._get_lock(combination_key)

        request_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info("Executing direct request",
                   combination=combination_key,
                   request_id=request_id)

        self.note_inflight_start(provider, model, api_key_id)
        try:
            async with lock:
                # Acquire a rate-limit slot (atomic check+record), waiting if
                # the window clears soon and failing fast if it does not.
                await self.acquire_rate_slot(provider, model, api_key_id,
                                             rate_limit_manager, config_manager)

                if attempt_marker is not None:
                    attempt_marker["api_call_started"] = True

                # Make the actual API call
                response = await llm_client._make_api_call(
                    provider=provider,
                    model=model,
                    api_key=request_data["api_key"],
                    messages=request_data["messages"],
                    options=request_data.get("options", {}),
                    endpoint=request_data["endpoint"],
                    version=request_data.get("version")
                )

                processing_time = time.time() - start_time

                # Return successful result
                result = {
                    "success": True,
                    "request_id": request_id,
                    "provider_used": provider,
                    "model_used": model,
                    "api_key_used": api_key_id,
                    "processing_time_seconds": processing_time,
                    "response": response,
                    "completed_at": datetime.now(timezone.utc).isoformat()
                }

                logger.info("Request processed successfully",
                           request_id=request_id,
                           provider=provider,
                           model=model,
                           processing_time=processing_time)

                return result

        except Exception as e:
            processing_time = time.time() - start_time

            logger.error("Request processing failed",
                        request_id=request_id,
                        error=str(e),
                        traceback=traceback.format_exc())

            return {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                # Populated for ProviderAPIError; None for local failures.
                "status_code": getattr(e, "status_code", None),
                "headers": getattr(e, "headers", None),
                "processing_time_seconds": processing_time,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
        finally:
            self.note_inflight_end(provider, model, api_key_id)

    async def acquire_rate_slot(self, provider: str, model: str, api_key_id: str,
                                rate_limit_manager, config_manager):
        """Atomically consume a rate-limit slot, waiting for the window if needed.

        Cap how long one request may block on rate-limit clearance. An
        RPD-exhausted key reports a wait of many hours; parking a request
        there guarantees a client-side timeout and starves throughput. When
        the wait exceeds the cap, fail fast so the router fails over to an
        RPM-recoverable key (RPM windows clear within ~60s). 0 disables the cap.
        """
        provider_config = config_manager.providers[provider]
        key_config = provider_config.keys[api_key_id]
        rate_limits = key_config.rate_limits[model]

        max_wait = int(os.getenv("RATE_LIMIT_MAX_WAIT_SECONDS", "120"))
        while True:
            consumed, wait_time = await rate_limit_manager.try_consume(
                provider, model, api_key_id, rate_limits["rpm"], rate_limits["rpd"]
            )

            if consumed:
                return

            if max_wait and wait_time > max_wait:
                # Surface as a rate-limit error so the router benches this key
                # (retry hint) and tries the next alternative immediately.
                raise Exception(
                    f"rate limit: wait {wait_time:.0f}s exceeds cap {max_wait}s; "
                    f"please retry in {int(wait_time)}s"
                )

            logger.info("Waiting for rate limit",
                       provider=provider, model=model, api_key_id=api_key_id,
                       wait_seconds=wait_time)

            await asyncio.sleep(min(wait_time, 1))  # Sleep in chunks of 1 second max

    async def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all queue states for monitoring"""
        states = {}
        for combination_key, state in self.states.items():
            state_dict = asdict(state)
            # Convert datetime to string for JSON serialization
            if state_dict.get("last_updated"):
                state_dict["last_updated"] = state.last_updated.isoformat()
            states[combination_key] = state_dict
        return states
