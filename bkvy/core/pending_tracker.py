"""
Pending request tracker for active in-flight LLM requests
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PendingRequest:
    request_id: str
    client_id: str
    routing_method: str        # intelligence / scenario / direct
    target: str                # intelligence level, scenario name, or provider/model
    provider: str = ""
    model: str = ""
    api_key_id: str = ""
    start_time: float = field(default_factory=time.time)
    max_wait_seconds: int = 0


class PendingRequestTracker:
    """Thread-safe in-memory registry for active requests"""

    def __init__(self):
        self._requests: Dict[str, PendingRequest] = {}
        self._lock = asyncio.Lock()

    async def add(self, request_id: str, client_id: str, routing_method: str,
                  target: str, max_wait_seconds: int) -> None:
        async with self._lock:
            self._requests[request_id] = PendingRequest(
                request_id=request_id,
                client_id=client_id,
                routing_method=routing_method,
                target=target,
                max_wait_seconds=max_wait_seconds,
            )

    async def update(self, request_id: str, provider: str, model: str, api_key_id: str) -> None:
        async with self._lock:
            req = self._requests.get(request_id)
            if req:
                req.provider = provider
                req.model = model
                req.api_key_id = api_key_id

    async def remove(self, request_id: str) -> None:
        async with self._lock:
            self._requests.pop(request_id, None)

    async def get_all(self) -> List[dict]:
        async with self._lock:
            now = time.time()
            return [
                {
                    "request_id": req.request_id,
                    "client_id": req.client_id,
                    "routing_method": req.routing_method,
                    "target": req.target,
                    "provider": req.provider,
                    "model": req.model,
                    "api_key_id": req.api_key_id,
                    "elapsed_seconds": round(now - req.start_time, 1),
                    "max_wait_seconds": req.max_wait_seconds,
                }
                for req in self._requests.values()
            ]


# Singleton
_pending_tracker: Optional[PendingRequestTracker] = None


def init_pending_tracker() -> PendingRequestTracker:
    global _pending_tracker
    _pending_tracker = PendingRequestTracker()
    return _pending_tracker


def get_pending_tracker() -> Optional[PendingRequestTracker]:
    return _pending_tracker
