"""
Data classes for bkvy application
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any


@dataclass
class RateLimitState:
    """Track rate limit state for a specific (API_KEY, MODEL) combination"""
    requests_this_minute: int = 0
    requests_today: int = 0
    minute_reset_time: datetime = None
    day_reset_time: datetime = None
    currently_rate_limited: bool = False
    rate_limit_wait_seconds: float = 0
    rpm_limit: int = 0
    rpd_limit: int = 0
    last_request_time: datetime = None

    def __post_init__(self):
        if self.minute_reset_time is None:
            now = datetime.now(timezone.utc)
            self.minute_reset_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        if self.day_reset_time is None:
            now = datetime.now(timezone.utc)
            self.day_reset_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass 
class QueueState:
    """Track queue state for a specific (API_KEY, MODEL) combination"""
    current_queue_length: int = 0
    active_requests: List[str] = None
    estimated_queue_wait_seconds: float = 0
    last_updated: datetime = None

    def __post_init__(self):
        if self.active_requests is None:
            self.active_requests = []
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)


@dataclass
class CompletionTimeAnalysis:
    """Analysis of total completion time for a combination"""
    rate_limit_wait_seconds: float
    queue_wait_seconds: float
    processing_time_seconds: float
    total_seconds: float
    cost_per_1k_tokens: float
    combination_key: str
    provider: str
    model: str
    api_key_id: str
    # Circuit breaker annotations (added dynamically)
    priority_penalty: int = 0
    circuit_state: Optional[Any] = None  # CircuitStatus enum


@dataclass
class VisionLimits:
    """Vision capability limits for a model"""
    max_images_per_request: int = 1
    max_image_size_mb: int = 20
    max_dimension: Optional[int] = None
    max_single_image_dimension: Optional[int] = None
    max_multi_image_dimension: Optional[int] = None


@dataclass
class ProviderModel:
    """Model configuration for a provider"""
    endpoint: str
    cost_per_1k_tokens: float
    avg_response_time_ms: int
    intelligence_tier: str
    version: Optional[str] = None
    supports_thinking: Optional[bool] = None
    supports_vision: Optional[bool] = False
    vision_limits: Optional[VisionLimits] = None


@dataclass
class ProviderKey:
    """API key configuration with rate limits"""
    api_key: str
    rate_limits: Dict[str, Dict[str, int]]


@dataclass
class ProviderConfig:
    """Complete provider configuration"""
    keys: Dict[str, ProviderKey]
    models: Dict[str, ProviderModel]