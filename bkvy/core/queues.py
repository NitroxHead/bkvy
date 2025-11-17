"""
Queue management for bkvy
"""

import asyncio
import time
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict

from ..models.data_classes import QueueState
from ..utils.logging import setup_logging

logger = setup_logging()


class QueueManager:
    """Manages request queues for all (API_KEY, MODEL) combinations"""
    
    def __init__(self, state_file: str = "queue_states.json"):
        self.state_file = Path(state_file)
        self.queues: Dict[str, asyncio.Queue] = {}
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
    
    def _get_queue(self, combination_key: str) -> asyncio.Queue:
        """Get or create queue for a combination"""
        if combination_key not in self.queues:
            self.queues[combination_key] = asyncio.Queue()
            self.states[combination_key] = QueueState()
        return self.queues[combination_key]
    
    async def get_queue_wait_time(self, provider: str, model: str, api_key_id: str, 
                                avg_response_time_ms: int) -> float:
        """Calculate estimated wait time in queue"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        queue = self._get_queue(combination_key)
        state = self.states[combination_key]
        
        queue_length = queue.qsize()
        avg_response_time_seconds = max(avg_response_time_ms / 1000, 0.1)  # Prevent division by zero
        
        estimated_wait = queue_length * avg_response_time_seconds
        
        # Update state
        state.current_queue_length = queue_length
        state.estimated_queue_wait_seconds = estimated_wait
        state.last_updated = datetime.now(timezone.utc)
        
        return estimated_wait
    
    async def execute_request_directly(self, provider: str, model: str, api_key_id: str,
                                     request_data: Dict[str, Any], max_wait_seconds: int, 
                                     rate_limit_manager, config_manager, llm_client) -> Dict[str, Any]:
        """Execute request directly and wait for response (no timeout - completes regardless of time)"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        lock = self._get_lock(combination_key)
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info("Executing direct request", 
                   combination=combination_key, 
                   request_id=request_id)
        
        try:
            async with lock:
                # Wait for rate limit if needed (but don't timeout the request)
                await self._wait_for_rate_limit(provider, model, api_key_id, 
                                              rate_limit_manager, config_manager)
                
                # Record the request for rate limiting
                await rate_limit_manager.record_request(provider, model, api_key_id)
                
                # Make the actual API call (no timeout - complete regardless of time)
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
                "processing_time_seconds": processing_time,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
    
    async def _wait_for_rate_limit(self, provider: str, model: str, api_key_id: str, 
                                 rate_limit_manager, config_manager):
        """Wait for rate limit to clear if needed (no timeout)"""
        
        # Get provider config to check rate limits
        provider_config = config_manager.providers[provider]
        key_config = provider_config.keys[api_key_id]
        rate_limits = key_config.rate_limits[model]
        
        while True:
            is_limited, wait_time = await rate_limit_manager.check_rate_limit_status(
                provider, model, api_key_id, rate_limits["rpm"], rate_limits["rpd"]
            )
            
            if not is_limited:
                break
            
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