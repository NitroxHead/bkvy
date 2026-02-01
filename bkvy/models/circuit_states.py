"""
Circuit breaker state models for provider health management
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field, asdict
import json


class CircuitStatus(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"          # Healthy, accepting requests
    OPEN = "open"              # Unhealthy, blocking requests
    HALF_OPEN = "half_open"    # Testing recovery


class FailureType(str, Enum):
    """Types of failures that can occur"""
    RATE_LIMIT_429 = "rate_limit_429"
    SERVICE_ERROR_5XX = "service_error_5xx"
    AUTH_ERROR_4XX = "auth_error_4xx"
    TIMEOUT_ERROR = "timeout_error"
    CONTENT_ERROR = "content_error"
    UNKNOWN_ERROR = "unknown_error"


class HealthStatus(str, Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class FailureEvent:
    """Record of a single failure occurrence"""
    timestamp: datetime
    failure_type: FailureType
    error_message: str
    response_time_ms: float = 0.0


@dataclass
class StateChangeEvent:
    """Record of a state transition"""
    timestamp: datetime
    from_state: CircuitStatus
    to_state: CircuitStatus
    reason: str
    failure_type: Optional[FailureType] = None


@dataclass
class CircuitState:
    """Circuit breaker state for a provider/model/key combination"""

    # Identity
    combination_key: str                    # "provider_model_keyid"
    provider: str
    model: str
    api_key_id: str

    # Current state
    state: CircuitStatus = CircuitStatus.CLOSED

    # Failure tracking
    failure_count: int = 0                  # Total failures in sliding window (computed property)
    consecutive_failures: int = 0            # Consecutive failures (resets on success)
    failure_history: List[FailureEvent] = field(default_factory=list)  # Sliding window of failures
    last_failure_time: Optional[datetime] = None
    last_failure_type: Optional[FailureType] = None
    last_error_message: str = ""

    # Success tracking
    last_success_time: Optional[datetime] = None
    consecutive_successes: int = 0

    # Recovery management
    next_test_time: Optional[datetime] = None
    backoff_level: int = 0                  # 0-10 for exponential backoff
    open_count: int = 0                     # How many times opened (anti-flapping)

    # Health probing
    test_probe_in_progress: bool = False
    health_probe_failures: int = 0

    # Performance tracking
    avg_response_time_ms: float = 0.0
    last_response_time_ms: float = 0.0

    # Flapping detection
    is_flapping: bool = False
    flapping_detected_at: Optional[datetime] = None
    priority_penalty: int = 0               # Penalty for prioritization (0 = normal)
    state_changes: List[StateChangeEvent] = field(default_factory=list)

    # Learning
    estimated_recovery_time: float = 0.0    # Learned average recovery duration
    recovery_times: List[float] = field(default_factory=list)  # Historical recovery times

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_failure_to_window(self, failure_type: FailureType, error_message: str,
                              response_time_ms: float = 0.0, window_seconds: int = 600):
        """Add a failure to the sliding window and clean up old entries"""
        now = datetime.now(timezone.utc)

        # Add new failure
        failure_event = FailureEvent(
            timestamp=now,
            failure_type=failure_type,
            error_message=error_message,
            response_time_ms=response_time_ms
        )
        self.failure_history.append(failure_event)

        # Clean up failures outside the window
        cutoff_time = now - timedelta(seconds=window_seconds)
        self.failure_history = [
            event for event in self.failure_history
            if event.timestamp > cutoff_time
        ]

        # Update computed failure_count
        self.failure_count = len(self.failure_history)

    def get_failure_count_in_window(self, window_seconds: int = 600) -> int:
        """Get count of failures in the sliding window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

        count = sum(
            1 for event in self.failure_history
            if event.timestamp > cutoff_time
        )

        return count

    def get_failures_by_type_in_window(self, failure_type: FailureType,
                                        window_seconds: int = 600) -> int:
        """Count failures of a specific type in the sliding window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

        count = sum(
            1 for event in self.failure_history
            if event.timestamp > cutoff_time and event.failure_type == failure_type
        )

        return count

    def clear_failure_window(self):
        """Clear the failure history (e.g., on successful recovery)"""
        self.failure_history.clear()
        self.failure_count = 0

    def record_state_change(self, new_state: CircuitStatus, reason: str,
                           failure_type: Optional[FailureType] = None):
        """Record a state transition"""
        event = StateChangeEvent(
            timestamp=datetime.now(timezone.utc),
            from_state=self.state,
            to_state=new_state,
            reason=reason,
            failure_type=failure_type
        )
        self.state_changes.append(event)

        # Keep only last 50 state changes
        if len(self.state_changes) > 50:
            self.state_changes = self.state_changes[-50:]

        self.state = new_state
        self.updated_at = datetime.now(timezone.utc)

    def get_open_events_in_window(self, window_seconds: int = 300) -> int:
        """Count how many times circuit opened in time window (for flapping detection)"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

        open_events = [
            event for event in self.state_changes
            if event.to_state == CircuitStatus.OPEN and event.timestamp > cutoff_time
        ]

        return len(open_events)

    def calculate_backoff_seconds(self, failure_type: FailureType,
                                  initial_backoff: int = 30) -> int:
        """Calculate backoff time based on failure type and level"""

        # Apply flapping penalty if detected
        if self.is_flapping:
            multiplier = 5
        else:
            multiplier = 1

        # Exponential backoff: initial * (2 ^ level) * multiplier
        backoff = initial_backoff * (2 ** self.backoff_level) * multiplier

        # Cap based on failure type
        if failure_type == FailureType.RATE_LIMIT_429:
            max_backoff = 86400  # 24 hours
        elif failure_type == FailureType.SERVICE_ERROR_5XX:
            max_backoff = 1800   # 30 minutes
        elif failure_type == FailureType.TIMEOUT_ERROR:
            max_backoff = 300    # 5 minutes
        elif failure_type == FailureType.AUTH_ERROR_4XX:
            max_backoff = 999999999  # Effectively infinite
        else:
            max_backoff = 600    # 10 minutes

        return min(backoff, max_backoff)

    def update_recovery_time(self, duration_seconds: float):
        """Update learned recovery time"""
        self.recovery_times.append(duration_seconds)

        # Keep only last 20 recovery times
        if len(self.recovery_times) > 20:
            self.recovery_times = self.recovery_times[-20:]

        # Calculate 90th percentile as estimated recovery time
        if self.recovery_times:
            sorted_times = sorted(self.recovery_times)
            percentile_90_idx = int(len(sorted_times) * 0.9)
            self.estimated_recovery_time = sorted_times[percentile_90_idx]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)

        # Convert datetime objects to ISO format
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
            elif isinstance(value, (CircuitStatus, FailureType, HealthStatus)):
                data[key] = value.value

        # Convert failure_history
        data['failure_history'] = [
            {
                'timestamp': event.timestamp.isoformat(),
                'failure_type': event.failure_type.value,
                'error_message': event.error_message,
                'response_time_ms': event.response_time_ms
            }
            for event in self.failure_history
        ]

        # Convert state_changes
        data['state_changes'] = [
            {
                'timestamp': event.timestamp.isoformat(),
                'from_state': event.from_state.value,
                'to_state': event.to_state.value,
                'reason': event.reason,
                'failure_type': event.failure_type.value if event.failure_type else None
            }
            for event in self.state_changes
        ]

        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'CircuitState':
        """Create from dictionary"""
        # Convert ISO format strings back to datetime
        for key in ['last_failure_time', 'last_success_time', 'next_test_time',
                   'flapping_detected_at', 'created_at', 'updated_at']:
            if key in data and data[key]:
                data[key] = datetime.fromisoformat(data[key])

        # Convert enum strings back to enums
        if 'state' in data:
            data['state'] = CircuitStatus(data['state'])
        if 'last_failure_type' in data and data['last_failure_type']:
            data['last_failure_type'] = FailureType(data['last_failure_type'])

        # Convert failure_history
        if 'failure_history' in data:
            data['failure_history'] = [
                FailureEvent(
                    timestamp=datetime.fromisoformat(event['timestamp']),
                    failure_type=FailureType(event['failure_type']),
                    error_message=event['error_message'],
                    response_time_ms=event.get('response_time_ms', 0.0)
                )
                for event in data['failure_history']
            ]

        # Convert state_changes
        if 'state_changes' in data:
            data['state_changes'] = [
                StateChangeEvent(
                    timestamp=datetime.fromisoformat(event['timestamp']),
                    from_state=CircuitStatus(event['from_state']),
                    to_state=CircuitStatus(event['to_state']),
                    reason=event['reason'],
                    failure_type=FailureType(event['failure_type']) if event.get('failure_type') else None
                )
                for event in data['state_changes']
            ]

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'CircuitState':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def create_new(cls, provider: str, model: str, api_key_id: str) -> 'CircuitState':
        """Create a new circuit state for a combination"""
        combination_key = f"{provider}_{model}_{api_key_id}"
        return cls(
            combination_key=combination_key,
            provider=provider,
            model=model,
            api_key_id=api_key_id
        )


@dataclass
class ProviderHealthState:
    """Aggregated health state for a provider"""

    provider_name: str
    overall_health: HealthStatus
    total_circuits: int
    open_circuits: int
    half_open_circuits: int
    closed_circuits: int
    open_percentage: float

    # Time tracking
    last_success_any_circuit: Optional[datetime] = None
    last_failure_any_circuit: Optional[datetime] = None

    # Failure patterns
    failure_pattern: Optional[str] = None  # "auth" | "widespread" | "isolated" | "rate_limited"
    dominant_failure_type: Optional[FailureType] = None

    # Recommendations
    recommended_action: Optional[str] = None
    should_skip_provider: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'provider_name': self.provider_name,
            'overall_health': self.overall_health.value,
            'total_circuits': self.total_circuits,
            'open_circuits': self.open_circuits,
            'half_open_circuits': self.half_open_circuits,
            'closed_circuits': self.closed_circuits,
            'open_percentage': self.open_percentage,
            'last_success_any_circuit': self.last_success_any_circuit.isoformat() if self.last_success_any_circuit else None,
            'last_failure_any_circuit': self.last_failure_any_circuit.isoformat() if self.last_failure_any_circuit else None,
            'failure_pattern': self.failure_pattern,
            'dominant_failure_type': self.dominant_failure_type.value if self.dominant_failure_type else None,
            'recommended_action': self.recommended_action,
            'should_skip_provider': self.should_skip_provider
        }
