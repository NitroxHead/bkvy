"""
Failure classification and strategy determination for circuit breaker
"""

from dataclasses import dataclass
from typing import Optional
from ..models.circuit_states import FailureType


@dataclass
class FailureStrategy:
    """Strategy for handling a specific failure type"""

    # Circuit breaker behavior
    should_circuit_break: bool          # Should this failure open the circuit?
    retry_count: int                    # How many retries before moving on?
    skip_alternatives: bool             # Skip to next alternative immediately?
    skip_provider: bool                 # Skip entire provider?

    # Backoff configuration
    backoff_schedule: str               # "exponential" | "fixed" | "calculated"
    initial_backoff_seconds: int
    max_backoff_seconds: int

    # Recovery behavior
    requires_health_probe: bool         # Need active probing before recovery?
    auto_recoverable: bool              # Can recover without config change?

    # Additional metadata
    severity: str                       # "low" | "medium" | "high" | "critical"
    description: str


class FailureClassifier:
    """Classifies failures and determines appropriate handling strategy"""

    # Strategy mapping for each failure type
    STRATEGIES = {
        FailureType.RATE_LIMIT_429: FailureStrategy(
            should_circuit_break=True,
            retry_count=0,                      # Don't retry, move to next
            skip_alternatives=True,             # Try different key immediately
            skip_provider=False,
            backoff_schedule="calculated",      # Based on response headers
            initial_backoff_seconds=60,         # Default if no header
            max_backoff_seconds=86400,          # 24 hours
            requires_health_probe=True,         # MUST test before reopening
            auto_recoverable=True,
            severity="medium",
            description="Rate limit exceeded, likely shared key usage"
        ),

        FailureType.SERVICE_ERROR_5XX: FailureStrategy(
            should_circuit_break=True,
            retry_count=3,
            skip_alternatives=False,
            skip_provider=False,
            backoff_schedule="exponential",
            initial_backoff_seconds=30,
            max_backoff_seconds=1800,           # 30 minutes
            requires_health_probe=True,
            auto_recoverable=True,
            severity="high",
            description="Provider service error, temporary outage likely"
        ),

        FailureType.AUTH_ERROR_4XX: FailureStrategy(
            should_circuit_break=True,
            retry_count=0,
            skip_alternatives=True,
            skip_provider=True,                 # Entire provider likely misconfigured
            backoff_schedule="fixed",
            initial_backoff_seconds=999999999,  # Never auto-retry
            max_backoff_seconds=999999999,
            requires_health_probe=False,        # No point probing
            auto_recoverable=False,             # Needs human intervention
            severity="critical",
            description="Authentication failure, requires configuration fix"
        ),

        FailureType.TIMEOUT_ERROR: FailureStrategy(
            should_circuit_break=True,
            retry_count=5,                      # More lenient than service errors
            skip_alternatives=False,
            skip_provider=False,
            backoff_schedule="exponential",
            initial_backoff_seconds=10,
            max_backoff_seconds=300,            # 5 minutes
            requires_health_probe=True,
            auto_recoverable=True,
            severity="medium",
            description="Network timeout, may be transient"
        ),

        FailureType.CONTENT_ERROR: FailureStrategy(
            should_circuit_break=False,         # Request-specific, not provider health
            retry_count=1,
            skip_alternatives=True,
            skip_provider=False,
            backoff_schedule="fixed",
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            requires_health_probe=False,
            auto_recoverable=True,
            severity="low",
            description="Content parsing error, request-specific issue"
        ),

        FailureType.UNKNOWN_ERROR: FailureStrategy(
            should_circuit_break=False,         # Don't circuit break on unknown errors
            retry_count=2,
            skip_alternatives=False,
            skip_provider=False,
            backoff_schedule="exponential",
            initial_backoff_seconds=10,
            max_backoff_seconds=300,
            requires_health_probe=False,
            auto_recoverable=True,
            severity="medium",
            description="Unknown error type, treat conservatively"
        ),
    }

    @classmethod
    def classify_error(cls, error_message: str, status_code: Optional[int] = None) -> FailureType:
        """
        Classify an error based on message and status code

        Args:
            error_message: Error message from API or exception
            status_code: HTTP status code if available

        Returns:
            FailureType enum value
        """
        error_lower = error_message.lower()

        # Check status code first (most reliable)
        if status_code:
            if status_code == 429:
                return FailureType.RATE_LIMIT_429
            elif 500 <= status_code < 600:
                return FailureType.SERVICE_ERROR_5XX
            elif status_code in [401, 403]:
                return FailureType.AUTH_ERROR_4XX

        # Rate limiting patterns
        if any(phrase in error_lower for phrase in [
            "rate limit", "429", "quota", "exceeded", "resource_exhausted",
            "too many requests", "rate_limit_exceeded"
        ]):
            return FailureType.RATE_LIMIT_429

        # Authentication/authorization patterns
        if any(phrase in error_lower for phrase in [
            "401", "403", "unauthorized", "forbidden", "invalid api key",
            "authentication failed", "invalid_api_key", "api key", "invalid key"
        ]):
            return FailureType.AUTH_ERROR_4XX

        # Service error patterns
        if any(phrase in error_lower for phrase in [
            "500", "502", "503", "504", "internal server error",
            "bad gateway", "service unavailable", "gateway timeout",
            "server error", "internal error"
        ]):
            return FailureType.SERVICE_ERROR_5XX

        # Timeout patterns
        if any(phrase in error_lower for phrase in [
            "timeout", "timed out", "connection", "network", "read timeout",
            "connect timeout", "connection error", "connection refused",
            "connection reset", "connection aborted"
        ]):
            return FailureType.TIMEOUT_ERROR

        # Content/response errors
        if any(phrase in error_lower for phrase in [
            "empty content", "could not extract content", "no content",
            "parse error", "json decode", "invalid response",
            "max_tokens", "content filter", "safety"
        ]):
            return FailureType.CONTENT_ERROR

        # Default to unknown
        return FailureType.UNKNOWN_ERROR

    @classmethod
    def get_strategy(cls, failure_type: FailureType) -> FailureStrategy:
        """
        Get handling strategy for a failure type

        Args:
            failure_type: Type of failure

        Returns:
            FailureStrategy with handling instructions
        """
        return cls.STRATEGIES.get(failure_type, cls.STRATEGIES[FailureType.UNKNOWN_ERROR])

    @classmethod
    def extract_rate_limit_reset_time(cls, error_message: str, headers: Optional[dict] = None) -> Optional[int]:
        """
        Extract rate limit reset time from error response

        Args:
            error_message: Error message
            headers: Response headers if available

        Returns:
            Seconds until reset, or None if not available
        """
        # Try to extract from headers first (most reliable)
        if headers:
            # Check common rate limit headers
            reset_time = headers.get('X-RateLimit-Reset') or \
                        headers.get('X-Rate-Limit-Reset') or \
                        headers.get('RateLimit-Reset') or \
                        headers.get('Retry-After')

            if reset_time:
                try:
                    # Could be Unix timestamp or seconds
                    reset_int = int(reset_time)

                    # If it's a Unix timestamp (large number)
                    if reset_int > 1000000000:
                        import time
                        return max(0, reset_int - int(time.time()))
                    else:
                        # It's seconds to wait
                        return reset_int
                except (ValueError, TypeError):
                    pass

        # Try to extract from error message patterns
        error_lower = error_message.lower()

        import re

        # Pattern: "Please retry in 43.217415972s" (Gemini format)
        gemini_pattern = re.search(r'retry in ([\d.]+)s', error_lower)
        if gemini_pattern:
            return int(float(gemini_pattern.group(1)) + 1)  # Round up

        # Pattern: "retry after 60 seconds"
        retry_pattern = re.search(r'retry after (\d+) seconds?', error_lower)
        if retry_pattern:
            return int(retry_pattern.group(1))

        # Pattern: "wait 60 seconds"
        wait_pattern = re.search(r'wait (\d+) seconds?', error_lower)
        if wait_pattern:
            return int(wait_pattern.group(1))

        # Pattern: "try again in 1 minute"
        minute_pattern = re.search(r'try again in (\d+) minutes?', error_lower)
        if minute_pattern:
            return int(minute_pattern.group(1)) * 60

        return None

    @classmethod
    def should_skip_retries(cls, error_message: str, status_code: Optional[int] = None) -> str:
        """
        Determine error handling strategy (for backward compatibility with router)

        Args:
            error_message: Error message
            status_code: HTTP status code if available

        Returns:
            Strategy string: "retry" | "skip_alternative" | "skip_provider"
        """
        failure_type = cls.classify_error(error_message, status_code)
        strategy = cls.get_strategy(failure_type)

        if strategy.skip_provider:
            return "skip_provider"
        elif strategy.skip_alternatives:
            return "skip_alternative"
        else:
            return "retry"

    @classmethod
    def get_failure_severity(cls, failure_type: FailureType) -> str:
        """Get severity level for a failure type"""
        strategy = cls.get_strategy(failure_type)
        return strategy.severity

    @classmethod
    def is_recoverable(cls, failure_type: FailureType) -> bool:
        """Check if failure type can auto-recover"""
        strategy = cls.get_strategy(failure_type)
        return strategy.auto_recoverable

    @classmethod
    def needs_health_probe(cls, failure_type: FailureType) -> bool:
        """Check if failure type requires health probing"""
        strategy = cls.get_strategy(failure_type)
        return strategy.requires_health_probe
