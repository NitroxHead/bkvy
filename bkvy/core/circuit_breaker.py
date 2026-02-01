"""
Circuit breaker manager for provider health management
"""

import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from ..models.circuit_states import (
    CircuitState, CircuitStatus, FailureType, ProviderHealthState, HealthStatus
)
from ..models.data_classes import CompletionTimeAnalysis
from ..core.failure_classifier import FailureClassifier
from ..utils.circuit_persistence import CircuitStatePersistence
from ..utils.logging import setup_logging

logger = setup_logging()


class CircuitBreakerManager:
    """Manages circuit breaker states for all provider/model/key combinations"""

    def __init__(self, config_manager, persistence: Optional[CircuitStatePersistence] = None):
        """
        Initialize circuit breaker manager

        Args:
            config_manager: Configuration manager instance
            persistence: Optional persistence layer (defaults to new instance)
        """
        self.config = config_manager
        self.persistence = persistence or CircuitStatePersistence()
        self.failure_classifier = FailureClassifier()

        # In-memory circuit states
        self.circuits: Dict[str, CircuitState] = {}

        # Configuration
        self.enabled = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
        self.failure_threshold = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))
        self.backoff_multiplier = float(os.getenv("CIRCUIT_BACKOFF_MULTIPLIER", "2.0"))
        self.max_backoff_seconds = int(os.getenv("CIRCUIT_MAX_BACKOFF_SECONDS", "1800"))
        self.flapping_threshold = int(os.getenv("CIRCUIT_FLAPPING_THRESHOLD", "3"))
        self.flapping_window_seconds = int(os.getenv("CIRCUIT_FLAPPING_WINDOW_SECONDS", "300"))

        # Sliding window configuration
        self.sliding_window_seconds = int(os.getenv("CIRCUIT_SLIDING_WINDOW_SECONDS", "600"))  # 10 minutes
        self.sliding_window_threshold = int(os.getenv("CIRCUIT_SLIDING_WINDOW_THRESHOLD", "5"))  # 5 failures

        logger.info(
            "Circuit breaker manager initialized",
            enabled=self.enabled,
            failure_threshold=self.failure_threshold,
            max_backoff=self.max_backoff_seconds
        )

    async def initialize(self):
        """Load circuit states from disk"""
        if not self.enabled:
            logger.info("Circuit breaker is disabled, skipping initialization")
            return

        # Load all persisted states
        self.circuits = await self.persistence.load_all_states()

        logger.info(
            "Circuit breaker states loaded",
            total_circuits=len(self.circuits),
            open_circuits=sum(1 for c in self.circuits.values() if c.state == CircuitStatus.OPEN),
            half_open_circuits=sum(1 for c in self.circuits.values() if c.state == CircuitStatus.HALF_OPEN)
        )

        # Cleanup old states for combinations no longer in config
        valid_combinations = self._get_valid_combinations()
        await self.persistence.cleanup_old_states(valid_combinations)

    def _get_valid_combinations(self) -> set:
        """Get all valid provider/model/key combinations from config"""
        combinations = set()

        for provider_name, provider_config in self.config.providers.items():
            for model_name in provider_config.models.keys():
                for api_key_id in provider_config.keys.keys():
                    combination_key = f"{provider_name}_{model_name}_{api_key_id}"
                    combinations.add(combination_key)

        return combinations

    def _get_or_create_circuit(self, provider: str, model: str, api_key_id: str) -> CircuitState:
        """Get existing circuit state or create new one"""
        combination_key = f"{provider}_{model}_{api_key_id}"

        if combination_key not in self.circuits:
            self.circuits[combination_key] = CircuitState.create_new(provider, model, api_key_id)
            logger.debug(
                "Created new circuit state",
                provider=provider,
                model=model,
                api_key_id=api_key_id
            )

        return self.circuits[combination_key]

    async def record_failure(
        self,
        provider: str,
        model: str,
        api_key_id: str,
        error_message: str,
        status_code: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        response_headers: Optional[dict] = None
    ) -> Tuple[bool, bool]:
        """
        Record a failure and potentially open circuit

        Args:
            provider: Provider name
            model: Model name
            api_key_id: API key identifier
            error_message: Error message
            status_code: HTTP status code if available
            response_time_ms: Response time in milliseconds
            response_headers: Response headers if available

        Returns:
            Tuple of (should_skip_retries, should_skip_provider)
        """
        if not self.enabled:
            # Circuit breaker disabled, use simple classification
            strategy_str = self.failure_classifier.should_skip_retries(error_message, status_code)
            return (strategy_str != "retry", strategy_str == "skip_provider")

        circuit = self._get_or_create_circuit(provider, model, api_key_id)

        # Classify the failure
        failure_type = self.failure_classifier.classify_error(error_message, status_code)
        strategy = self.failure_classifier.get_strategy(failure_type)

        logger.info(
            "Recording circuit failure",
            provider=provider,
            model=model,
            api_key_id=api_key_id,
            failure_type=failure_type.value,
            current_state=circuit.state.value,
            consecutive_failures=circuit.consecutive_failures + 1
        )

        # Add failure to sliding window
        circuit.add_failure_to_window(
            failure_type=failure_type,
            error_message=error_message[:500],  # Truncate long messages
            response_time_ms=response_time_ms or 0.0,
            window_seconds=self.sliding_window_seconds
        )

        # Update failure counters
        circuit.consecutive_failures += 1
        circuit.consecutive_successes = 0
        circuit.last_failure_time = datetime.now(timezone.utc)
        circuit.last_failure_type = failure_type
        circuit.last_error_message = error_message[:500]  # Truncate long messages

        if response_time_ms:
            circuit.last_response_time_ms = response_time_ms

        # Decide if circuit should open based on:
        # 1. Consecutive failures (immediate pattern)
        # 2. Sliding window failures (sustained issues)
        should_open = False

        if strategy.should_circuit_break:
            # Check consecutive failures
            if circuit.consecutive_failures >= self.failure_threshold:
                should_open = True
            # Or check sliding window
            elif circuit.get_failure_count_in_window(window_seconds=self.sliding_window_seconds) >= self.sliding_window_threshold:
                should_open = True
                logger.info(
                    "Circuit opening due to sliding window threshold",
                    provider=provider,
                    model=model,
                    api_key_id=api_key_id,
                    failures_in_window=circuit.get_failure_count_in_window(self.sliding_window_seconds),
                    window_seconds=self.sliding_window_seconds,
                    threshold=self.sliding_window_threshold
                )

        if should_open and circuit.state != CircuitStatus.OPEN:
            await self._open_circuit(circuit, failure_type, response_headers)

        # Persist state
        await self.persistence.save_state(circuit)

        return (strategy.skip_alternatives, strategy.skip_provider)

    async def _open_circuit(
        self,
        circuit: CircuitState,
        failure_type: FailureType,
        response_headers: Optional[dict] = None
    ):
        """Open a circuit and set recovery time"""
        previous_state = circuit.state

        # Clear probe lock when reopening circuit (failed test probe)
        circuit.test_probe_in_progress = False
        circuit.test_probe_started_at = None

        # Record state change
        circuit.record_state_change(
            CircuitStatus.OPEN,
            f"consecutive_failures={circuit.consecutive_failures}",
            failure_type
        )

        circuit.open_count += 1
        circuit.backoff_level = min(circuit.backoff_level + 1, 10)

        # Calculate next test time
        strategy = self.failure_classifier.get_strategy(failure_type)

        if failure_type == FailureType.RATE_LIMIT_429:
            # Try to extract reset time from headers
            reset_seconds = self.failure_classifier.extract_rate_limit_reset_time(
                circuit.last_error_message,
                response_headers
            )

            if reset_seconds:
                backoff = reset_seconds
            else:
                # Default backoff for rate limits
                backoff = circuit.calculate_backoff_seconds(failure_type, strategy.initial_backoff_seconds)

        elif failure_type == FailureType.AUTH_ERROR_4XX:
            # Never auto-test auth errors
            circuit.next_test_time = None
        else:
            # Exponential backoff for other errors
            backoff = circuit.calculate_backoff_seconds(failure_type, strategy.initial_backoff_seconds)

        if failure_type != FailureType.AUTH_ERROR_4XX:
            circuit.next_test_time = datetime.now(timezone.utc) + timedelta(seconds=backoff)

        # Check for flapping
        await self._check_flapping(circuit)

        logger.warning(
            "Circuit opened",
            provider=circuit.provider,
            model=circuit.model,
            api_key_id=circuit.api_key_id,
            failure_type=failure_type.value,
            consecutive_failures=circuit.consecutive_failures,
            backoff_seconds=backoff,
            next_test_time=circuit.next_test_time.isoformat() if circuit.next_test_time else "never",
            is_flapping=circuit.is_flapping,
            previous_state=previous_state.value
        )

    async def record_success(
        self,
        provider: str,
        model: str,
        api_key_id: str,
        response_time_ms: Optional[float] = None
    ):
        """
        Record a success and potentially close circuit

        Args:
            provider: Provider name
            model: Model name
            api_key_id: API key identifier
            response_time_ms: Response time in milliseconds
        """
        if not self.enabled:
            return

        circuit = self._get_or_create_circuit(provider, model, api_key_id)

        logger.debug(
            "Recording circuit success",
            provider=provider,
            model=model,
            api_key_id=api_key_id,
            current_state=circuit.state.value
        )

        # Update success tracking
        circuit.consecutive_successes += 1
        circuit.consecutive_failures = 0
        circuit.last_success_time = datetime.now(timezone.utc)

        if response_time_ms:
            circuit.last_response_time_ms = response_time_ms
            # Update rolling average
            if circuit.avg_response_time_ms == 0:
                circuit.avg_response_time_ms = response_time_ms
            else:
                # Exponential moving average
                alpha = 0.3
                circuit.avg_response_time_ms = (
                    alpha * response_time_ms + (1 - alpha) * circuit.avg_response_time_ms
                )

        # Handle state transitions
        if circuit.state == CircuitStatus.HALF_OPEN:
            # Success during testing - close the circuit
            await self._close_circuit(circuit)

        elif circuit.state == CircuitStatus.CLOSED:
            # Sliding window automatically expires old failures
            # Just sync the failure_count for consistency
            circuit.failure_count = circuit.get_failure_count_in_window(window_seconds=self.sliding_window_seconds)

        # Persist state
        await self.persistence.save_state(circuit)

    async def _close_circuit(self, circuit: CircuitState):
        """Close a circuit after successful recovery"""
        # Calculate recovery time if we have opening time
        if circuit.last_failure_time and circuit.state == CircuitStatus.HALF_OPEN:
            recovery_duration = (datetime.now(timezone.utc) - circuit.last_failure_time).total_seconds()
            circuit.update_recovery_time(recovery_duration)

        previous_state = circuit.state

        # Record state change
        circuit.record_state_change(
            CircuitStatus.CLOSED,
            "success_after_testing"
        )

        # Reset failure tracking
        circuit.consecutive_failures = 0
        circuit.backoff_level = max(0, circuit.backoff_level - 1)  # Gradual backoff reduction
        circuit.next_test_time = None
        circuit.test_probe_in_progress = False
        circuit.test_probe_started_at = None
        circuit.health_probe_failures = 0

        # Clear sliding window on successful recovery
        circuit.clear_failure_window()

        # Clear flapping if circuit has been stable
        if circuit.is_flapping:
            await self._check_flapping_clear(circuit)

        logger.info(
            "Circuit closed",
            provider=circuit.provider,
            model=circuit.model,
            api_key_id=circuit.api_key_id,
            previous_state=previous_state.value,
            recovery_time_seconds=circuit.estimated_recovery_time if circuit.estimated_recovery_time > 0 else None
        )

    async def should_try_combination(
        self,
        provider: str,
        model: str,
        api_key_id: str
    ) -> Tuple[bool, int, str]:
        """
        Check if a combination should be attempted

        Args:
            provider: Provider name
            model: Model name
            api_key_id: API key identifier

        Returns:
            Tuple of (can_try: bool, wait_time_seconds: int, reason: str)
        """
        if not self.enabled:
            return (True, 0, "circuit_breaker_disabled")

        circuit = self._get_or_create_circuit(provider, model, api_key_id)

        if circuit.state == CircuitStatus.CLOSED:
            return (True, 0, "healthy")

        elif circuit.state == CircuitStatus.OPEN:
            # Check if it's time to test recovery
            now = datetime.now(timezone.utc)

            if circuit.next_test_time is None:
                # Auth errors never auto-recover
                return (False, 999999, "auth_failure_requires_manual_fix")

            if now >= circuit.next_test_time:
                # Time to test - try to transition to HALF_OPEN
                if not circuit.test_probe_in_progress:
                    # Acquire test lock with timestamp
                    circuit.test_probe_in_progress = True
                    circuit.test_probe_started_at = datetime.now(timezone.utc)
                    circuit.record_state_change(
                        CircuitStatus.HALF_OPEN,
                        "testing_recovery"
                    )
                    await self.persistence.save_state(circuit)

                    logger.info(
                        "Circuit transitioned to HALF_OPEN for testing",
                        provider=provider,
                        model=model,
                        api_key_id=api_key_id
                    )

                    return (True, 0, "testing_recovery")
                else:
                    # Someone else is testing
                    return (False, 2, "probe_in_progress")
            else:
                # Not time yet
                wait_seconds = int((circuit.next_test_time - now).total_seconds())
                return (False, wait_seconds, "circuit_open")

        elif circuit.state == CircuitStatus.HALF_OPEN:
            if circuit.test_probe_in_progress:
                # Check if probe lock has timed out
                now = datetime.now(timezone.utc)
                probe_timeout_seconds = int(os.getenv("PROBE_LOCK_TIMEOUT_SECONDS", "300"))  # 5 min

                if circuit.test_probe_started_at:
                    elapsed = (now - circuit.test_probe_started_at).total_seconds()

                    if elapsed > probe_timeout_seconds:
                        # Probe lock timed out - clear it and allow testing
                        logger.warning(
                            "Probe lock timeout - clearing stuck lock",
                            provider=provider,
                            model=model,
                            api_key_id=api_key_id,
                            elapsed_seconds=elapsed,
                            timeout_seconds=probe_timeout_seconds
                        )

                        circuit.test_probe_in_progress = False
                        circuit.test_probe_started_at = None
                        await self.persistence.save_state(circuit)

                        return (True, 0, "testing_recovery_after_timeout")

                # Probe still in progress within timeout
                return (False, 2, "probe_in_progress")
            else:
                # Allow testing
                return (True, 0, "testing_recovery")

        return (False, 0, "unknown_state")

    async def filter_alternatives(
        self,
        analyses: List[CompletionTimeAnalysis]
    ) -> Tuple[List[CompletionTimeAnalysis], List[dict]]:
        """
        Filter alternatives based on circuit states

        Args:
            analyses: List of completion time analyses

        Returns:
            Tuple of (usable_alternatives, blocked_alternatives_with_reasons)
        """
        if not self.enabled:
            # Circuit breaker disabled - return all as usable
            return (analyses, [])

        usable = []
        blocked = []

        for analysis in analyses:
            can_try, wait_time, reason = await self.should_try_combination(
                analysis.provider,
                analysis.model,
                analysis.api_key_id
            )

            if can_try:
                # Annotate with circuit priority
                circuit = self._get_or_create_circuit(
                    analysis.provider,
                    analysis.model,
                    analysis.api_key_id
                )

                # Add priority penalty for flapping circuits
                analysis.priority_penalty = circuit.priority_penalty

                # Add circuit state for sorting
                analysis.circuit_state = circuit.state

                usable.append(analysis)
            else:
                blocked.append({
                    'provider': analysis.provider,
                    'model': analysis.model,
                    'api_key_id': analysis.api_key_id,
                    'reason': reason,
                    'wait_time_seconds': wait_time,
                    'circuit_state': self.circuits.get(
                        f"{analysis.provider}_{analysis.model}_{analysis.api_key_id}"
                    ).state.value if f"{analysis.provider}_{analysis.model}_{analysis.api_key_id}" in self.circuits else 'unknown'
                })

        # Sort usable alternatives by:
        # 1. Circuit state (CLOSED > HALF_OPEN)
        # 2. Priority penalty (lower is better)
        # 3. Cost (lower is better)
        # 4. Speed (lower is better)
        usable.sort(key=lambda x: (
            0 if x.circuit_state == CircuitStatus.CLOSED else 1,
            x.priority_penalty,
            x.cost_per_1k_tokens,
            x.total_seconds
        ))

        logger.debug(
            "Filtered alternatives through circuit breaker",
            total_alternatives=len(analyses),
            usable=len(usable),
            blocked=len(blocked)
        )

        return (usable, blocked)

    async def _check_flapping(self, circuit: CircuitState):
        """Check if circuit is flapping and apply penalty"""
        open_events = circuit.get_open_events_in_window(self.flapping_window_seconds)

        if open_events >= self.flapping_threshold and not circuit.is_flapping:
            circuit.is_flapping = True
            circuit.flapping_detected_at = datetime.now(timezone.utc)
            circuit.priority_penalty = 1000  # Large penalty

            # Increase backoff significantly
            circuit.backoff_level = min(circuit.backoff_level + 3, 10)

            logger.warning(
                "Flapping detected",
                provider=circuit.provider,
                model=circuit.model,
                api_key_id=circuit.api_key_id,
                open_events_in_window=open_events,
                window_seconds=self.flapping_window_seconds
            )

    async def _check_flapping_clear(self, circuit: CircuitState):
        """Check if flapping status should be cleared"""
        if not circuit.is_flapping:
            return

        # Clear flapping if circuit has been CLOSED for 10 minutes
        if circuit.state == CircuitStatus.CLOSED and circuit.last_success_time:
            time_stable = (datetime.now(timezone.utc) - circuit.last_success_time).total_seconds()

            if time_stable > 600:  # 10 minutes
                circuit.is_flapping = False
                circuit.priority_penalty = 0
                circuit.flapping_detected_at = None

                logger.info(
                    "Flapping status cleared",
                    provider=circuit.provider,
                    model=circuit.model,
                    api_key_id=circuit.api_key_id,
                    stable_duration_seconds=int(time_stable)
                )

    async def get_provider_health(self, provider_name: str) -> ProviderHealthState:
        """
        Get aggregated health state for a provider

        Args:
            provider_name: Provider name

        Returns:
            ProviderHealthState with aggregated metrics
        """
        # Get all circuits for this provider
        provider_circuits = [
            circuit for circuit in self.circuits.values()
            if circuit.provider == provider_name
        ]

        if not provider_circuits:
            return ProviderHealthState(
                provider_name=provider_name,
                overall_health=HealthStatus.HEALTHY,
                total_circuits=0,
                open_circuits=0,
                half_open_circuits=0,
                closed_circuits=0,
                open_percentage=0.0
            )

        # Count by state
        total = len(provider_circuits)
        open_count = sum(1 for c in provider_circuits if c.state == CircuitStatus.OPEN)
        half_open_count = sum(1 for c in provider_circuits if c.state == CircuitStatus.HALF_OPEN)
        closed_count = sum(1 for c in provider_circuits if c.state == CircuitStatus.CLOSED)
        open_percentage = open_count / total if total > 0 else 0.0

        # Find most recent success/failure across all circuits
        last_success = max(
            (c.last_success_time for c in provider_circuits if c.last_success_time),
            default=None
        )
        last_failure = max(
            (c.last_failure_time for c in provider_circuits if c.last_failure_time),
            default=None
        )

        # Analyze failure patterns
        open_circuits = [c for c in provider_circuits if c.state == CircuitStatus.OPEN]
        failure_types = [c.last_failure_type for c in open_circuits if c.last_failure_type]

        failure_pattern = None
        dominant_failure_type = None
        recommended_action = None
        should_skip = False

        if failure_types:
            # Find most common failure type
            type_counts = defaultdict(int)
            for ft in failure_types:
                type_counts[ft] += 1
            dominant_failure_type = max(type_counts, key=type_counts.get)

            # Check for systemic issues
            if all(ft == FailureType.AUTH_ERROR_4XX for ft in failure_types):
                failure_pattern = "auth"
                recommended_action = "check_api_keys"
                should_skip = True
            elif all(ft == FailureType.RATE_LIMIT_429 for ft in failure_types):
                failure_pattern = "rate_limited"
                recommended_action = "external_usage_likely"
            elif open_percentage > 0.8:
                failure_pattern = "widespread"
                recommended_action = "provider_outage_likely"
                should_skip = True
            elif open_count > 0 and open_count < total * 0.3:
                failure_pattern = "isolated"
                recommended_action = "specific_keys_or_models_affected"

        # Determine overall health
        if open_percentage < 0.1:
            overall_health = HealthStatus.HEALTHY
        elif open_percentage < 0.5:
            overall_health = HealthStatus.DEGRADED
        else:
            overall_health = HealthStatus.UNHEALTHY

        return ProviderHealthState(
            provider_name=provider_name,
            overall_health=overall_health,
            total_circuits=total,
            open_circuits=open_count,
            half_open_circuits=half_open_count,
            closed_circuits=closed_count,
            open_percentage=open_percentage,
            last_success_any_circuit=last_success,
            last_failure_any_circuit=last_failure,
            failure_pattern=failure_pattern,
            dominant_failure_type=dominant_failure_type,
            recommended_action=recommended_action,
            should_skip_provider=should_skip
        )

    def get_all_circuit_states(self) -> Dict[str, dict]:
        """Get all circuit states as dictionaries"""
        return {
            key: circuit.to_dict()
            for key, circuit in self.circuits.items()
        }

    def get_circuit_state(self, provider: str, model: str, api_key_id: str) -> Optional[dict]:
        """Get specific circuit state"""
        circuit = self._get_or_create_circuit(provider, model, api_key_id)
        return circuit.to_dict() if circuit else None

    async def reset_circuit(self, provider: str, model: str, api_key_id: str) -> bool:
        """Manually reset a circuit (admin action)"""
        combination_key = f"{provider}_{model}_{api_key_id}"

        if combination_key in self.circuits:
            circuit = self.circuits[combination_key]
            circuit.record_state_change(CircuitStatus.CLOSED, "manual_reset")
            circuit.consecutive_failures = 0
            circuit.clear_failure_window()  # Clear sliding window
            circuit.backoff_level = 0
            circuit.is_flapping = False
            circuit.priority_penalty = 0
            circuit.test_probe_in_progress = False
            circuit.test_probe_started_at = None

            await self.persistence.save_state(circuit)

            logger.info(
                "Circuit manually reset",
                provider=provider,
                model=model,
                api_key_id=api_key_id
            )

            return True

        return False

    def get_probe_lock_stats(self) -> dict:
        """Get statistics about probe locks for monitoring"""
        now = datetime.now(timezone.utc)
        probe_timeout = int(os.getenv("PROBE_LOCK_TIMEOUT_SECONDS", "300"))

        stats = {
            "total_circuits": len(self.circuits),
            "half_open_with_locks": 0,
            "stuck_probe_locks": 0,
            "stuck_circuits": []
        }

        for circuit in self.circuits.values():
            if circuit.state == CircuitStatus.HALF_OPEN and circuit.test_probe_in_progress:
                stats["half_open_with_locks"] += 1

                if circuit.test_probe_started_at:
                    elapsed = (now - circuit.test_probe_started_at).total_seconds()

                    if elapsed > probe_timeout:
                        stats["stuck_probe_locks"] += 1
                        stats["stuck_circuits"].append({
                            "provider": circuit.provider,
                            "model": circuit.model,
                            "api_key_id": circuit.api_key_id,
                            "stuck_duration_seconds": elapsed
                        })

        return stats
