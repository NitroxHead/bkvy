"""
Global timeout manager for request execution
"""

import os
import time
from typing import List, Callable, Any, Dict
from ..models.data_classes import CompletionTimeAnalysis
from ..models.circuit_states import CircuitStatus
from ..utils.logging import setup_logging

logger = setup_logging()


class GlobalTimeoutManager:
    """Manages global request timeouts with progressive escalation"""

    def __init__(self):
        """Initialize timeout manager with configurable thresholds"""
        self.soft_timeout_seconds = int(os.getenv("REQUEST_SOFT_TIMEOUT_SECONDS", "30"))
        self.hard_timeout_seconds = int(os.getenv("REQUEST_HARD_TIMEOUT_SECONDS", "120"))
        self.escalation_threshold = int(os.getenv("REQUEST_ESCALATION_THRESHOLD", "10"))

        logger.info(
            "Global timeout manager initialized",
            soft_timeout=self.soft_timeout_seconds,
            hard_timeout=self.hard_timeout_seconds
        )

    def should_escalate(self, start_time: float) -> bool:
        """
        Check if request should escalate to faster strategy

        Args:
            start_time: Request start timestamp

        Returns:
            True if soft timeout exceeded
        """
        elapsed = time.time() - start_time
        return elapsed > self.soft_timeout_seconds

    def should_abort(self, start_time: float) -> bool:
        """
        Check if request should abort completely

        Args:
            start_time: Request start timestamp

        Returns:
            True if hard timeout exceeded
        """
        elapsed = time.time() - start_time
        return elapsed > self.hard_timeout_seconds

    def get_remaining_time(self, start_time: float) -> float:
        """
        Get remaining time until hard timeout

        Args:
            start_time: Request start timestamp

        Returns:
            Seconds remaining (0 if exceeded)
        """
        elapsed = time.time() - start_time
        remaining = self.hard_timeout_seconds - elapsed
        return max(0, remaining)

    def get_request_timeout(self, start_time: float, escalated: bool, default_timeout: int = 300) -> int:
        """
        Calculate per-request timeout based on global timeout state

        Args:
            start_time: Request start timestamp
            escalated: Whether request is in escalated mode
            default_timeout: Default timeout for non-escalated requests

        Returns:
            Timeout in seconds for this specific request
        """
        remaining = self.get_remaining_time(start_time)

        if escalated:
            # Use shorter timeout after escalation
            escalated_timeout = 30
            return int(min(escalated_timeout, remaining))
        else:
            # Use default timeout if enough time remains
            return int(min(default_timeout, remaining))

    def reorder_for_escalation(
        self,
        alternatives: List[CompletionTimeAnalysis],
        current_provider: str
    ) -> List[CompletionTimeAnalysis]:
        """
        Reorder alternatives for escalated mode

        In escalation:
        - Only use CLOSED circuits (skip HALF_OPEN)
        - Prefer different providers for diversity
        - Optimize for speed over cost

        Args:
            alternatives: List of alternatives
            current_provider: Provider that was just tried

        Returns:
            Reordered list optimized for speed
        """
        # Filter to only CLOSED circuits
        closed_only = [
            alt for alt in alternatives
            if alt.circuit_state == CircuitStatus.CLOSED
        ]

        if not closed_only:
            # No CLOSED circuits available, use all
            closed_only = alternatives

        # Sort by:
        # 1. Different provider than current (for diversity)
        # 2. Speed (total_seconds)
        # 3. Cost (tiebreaker)
        closed_only.sort(key=lambda x: (
            0 if x.provider != current_provider else 1,  # Prioritize different provider
            x.total_seconds,                               # Then fastest
            x.cost_per_1k_tokens                           # Then cheapest
        ))

        return closed_only

    def create_timeout_failure(
        self,
        reason: str,
        elapsed_seconds: float,
        alternatives_tried: int,
        request_id: str
    ) -> Dict[str, Any]:
        """
        Create failure response for timeout

        Args:
            reason: Timeout reason ("soft_timeout" or "hard_timeout")
            elapsed_seconds: Total elapsed time
            alternatives_tried: Number of alternatives attempted
            request_id: Request identifier

        Returns:
            Failure response dictionary
        """
        return {
            "success": False,
            "request_id": request_id,
            "error_code": "timeout_exceeded",
            "message": f"Request exceeded {reason} ({elapsed_seconds:.1f}s) after trying {alternatives_tried} alternatives",
            "elapsed_seconds": elapsed_seconds,
            "timeout_type": reason,
            "alternatives_tried": alternatives_tried
        }

    def get_escalation_config(self, escalated: bool) -> Dict[str, Any]:
        """
        Get configuration for current escalation state

        Args:
            escalated: Whether in escalated mode

        Returns:
            Configuration dictionary
        """
        if escalated:
            return {
                "max_retries": 1,              # Reduce retries
                "api_timeout": 30,             # Shorter API timeout
                "use_half_open": False,        # Skip HALF_OPEN circuits
                "optimize_for": "speed",       # Optimize for speed
                "provider_diversity": True     # Try different providers
            }
        else:
            return {
                "max_retries": 3,              # Normal retries
                "api_timeout": 300,            # Normal API timeout
                "use_half_open": True,         # Use all available circuits
                "optimize_for": "cost",        # Optimize for cost
                "provider_diversity": False    # Follow normal priority
            }

    def log_escalation(self, elapsed_seconds: float, alternatives_tried: int):
        """Log timeout escalation event"""
        logger.warning(
            "Request escalated to fast mode",
            elapsed_seconds=elapsed_seconds,
            soft_timeout=self.soft_timeout_seconds,
            alternatives_tried=alternatives_tried,
            reason="soft_timeout_exceeded"
        )

    def log_timeout_abort(self, elapsed_seconds: float, alternatives_tried: int):
        """Log hard timeout abort event"""
        logger.error(
            "Request aborted due to hard timeout",
            elapsed_seconds=elapsed_seconds,
            hard_timeout=self.hard_timeout_seconds,
            alternatives_tried=alternatives_tried,
            reason="hard_timeout_exceeded"
        )

    def get_status_summary(self, start_time: float) -> Dict[str, Any]:
        """
        Get current timeout status

        Args:
            start_time: Request start timestamp

        Returns:
            Status summary dictionary
        """
        elapsed = time.time() - start_time
        remaining = self.get_remaining_time(start_time)
        escalated = self.should_escalate(start_time)
        should_abort = self.should_abort(start_time)

        return {
            "elapsed_seconds": round(elapsed, 2),
            "remaining_seconds": round(remaining, 2),
            "soft_timeout_seconds": self.soft_timeout_seconds,
            "hard_timeout_seconds": self.hard_timeout_seconds,
            "escalated": escalated,
            "should_abort": should_abort,
            "timeout_percentage": round((elapsed / self.hard_timeout_seconds) * 100, 1)
        }
