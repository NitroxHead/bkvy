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

        FailureType.MODEL_ERROR: FailureStrategy(
            should_circuit_break=True,
            retry_count=0,                      # Retrying a missing model never helps
            skip_alternatives=True,             # Move to the next alternative immediately
            skip_provider=False,                # Other models on the provider are fine
            backoff_schedule="exponential",
            initial_backoff_seconds=300,
            max_backoff_seconds=86400,          # 24 hours
            requires_health_probe=True,         # Probes report 404 honestly
            auto_recoverable=False,             # Usually needs a config fix
            severity="high",
            description="Model not found, retired, or inaccessible - check config"
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

        # Request-too-large errors first: they are request-specific CONTENT
        # errors, but their wording ("context_length_exceeded") collides with
        # rate-limit vocabulary. Misclassifying them as 429 opens circuits on
        # healthy keys whenever a client sends an oversized prompt.
        if any(phrase in error_lower for phrase in [
            "context_length_exceeded", "context length", "maximum context",
            "prompt is too long", "too many tokens", "input token count",
            "reduce the length"
        ]):
            return FailureType.CONTENT_ERROR

        # Check status code first (most reliable)
        if status_code:
            if status_code == 429:
                return FailureType.RATE_LIMIT_429
            elif 500 <= status_code < 600:
                return FailureType.SERVICE_ERROR_5XX
            elif status_code in [401, 403]:
                return FailureType.AUTH_ERROR_4XX
            elif status_code == 404:
                return FailureType.MODEL_ERROR

        # Rate limiting patterns (no bare "exceeded" - see content check above)
        if any(phrase in error_lower for phrase in [
            "rate limit", "429", "quota", "resource_exhausted",
            "too many requests", "rate_limit_exceeded"
        ]):
            return FailureType.RATE_LIMIT_429

        # Authentication/authorization patterns
        if any(phrase in error_lower for phrase in [
            "401", "403", "unauthorized", "forbidden", "invalid api key",
            "authentication failed", "invalid_api_key", "api key", "invalid key"
        ]):
            return FailureType.AUTH_ERROR_4XX

        # Model errors: missing, retired, or inaccessible models
        if any(phrase in error_lower for phrase in [
            "404", "not_found", "not found", "does not exist", "unknown model",
            "model_not_found", "has been deprecated", "end-of-life"
        ]):
            return FailureType.MODEL_ERROR

        # Service error patterns
        if any(phrase in error_lower for phrase in [
            "500", "502", "503", "504", "internal server error",
            "bad gateway", "service unavailable", "gateway timeout",
            "server error", "internal error", "high demand"
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
            # Check common rate limit headers, case-insensitively
            headers_lower = {str(k).lower(): v for k, v in headers.items()}
            reset_time = headers_lower.get('x-ratelimit-reset') or \
                        headers_lower.get('x-rate-limit-reset') or \
                        headers_lower.get('ratelimit-reset') or \
                        headers_lower.get('retry-after')

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

        # Gemini daily quota exhaustion (free tier RPD) - wait until midnight Pacific
        # Two patterns:
        # 1. "PerDay" in quotaId (e.g. gemini-2.0-flash format)
        # 2. metric "generate_content_free_tier_requests" with limit: 20 (gemini-2.5-flash format)
        #    - "retry in Xs" in this message is the RPM window reset, NOT the daily reset
        error_lower_no_sep = error_lower.replace('_', '').replace('-', '')
        is_daily_quota = (
            ('perday' in error_lower_no_sep and 'free_tier' in error_lower)
            or 'free_tier_requests' in error_lower
        )
        if is_daily_quota:
            return cls._seconds_until_gemini_daily_reset()

        # Pattern: "Please retry in 43.217415972s" (Gemini format)
        gemini_pattern = re.search(r'retry in ([\d.]+)s', error_lower)
        if gemini_pattern:
            return int(float(gemini_pattern.group(1)) + 1)  # Round up

        # Pattern: "Please try again in 20s" / "try again in 6m59.56s" (OpenAI
        # uses Go-style durations)
        openai_pattern = re.search(r'try again in (?:(\d+)m)?([\d.]+)\s*s', error_lower)
        if openai_pattern:
            minutes = int(openai_pattern.group(1)) if openai_pattern.group(1) else 0
            return minutes * 60 + int(float(openai_pattern.group(2)) + 1)  # Round up

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

    @staticmethod
    def _seconds_until_gemini_daily_reset() -> int:
        """Calculate seconds until midnight Pacific Time (Gemini daily quota reset).

        Returns value floored at 1800s (30 min) and capped at 86400s (24h).
        """
        from datetime import datetime, timedelta, timezone

        now_utc = datetime.now(timezone.utc)

        try:
            from zoneinfo import ZoneInfo
            pacific = ZoneInfo("America/Los_Angeles")
            now_pacific = now_utc.astimezone(pacific)
            midnight = (now_pacific + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            seconds = int((midnight - now_pacific).total_seconds())
        except Exception:
            # Fallback: assume UTC-8
            utc_minus_8 = timezone(timedelta(hours=-8))
            now_pacific = now_utc.astimezone(utc_minus_8)
            midnight = (now_pacific + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            seconds = int((midnight - now_pacific).total_seconds())

        return max(1800, min(seconds, 86400))
