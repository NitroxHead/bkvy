"""
Health probe system for testing circuit recovery
"""

import os
import asyncio
from typing import Dict, Optional, Tuple
import aiohttp

from ..models.circuit_states import FailureType, CircuitState, CircuitStatus
from ..core.failure_classifier import FailureClassifier
from ..utils.logging import setup_logging

logger = setup_logging()


class HealthProbe:
    """Lightweight health checking for circuit recovery testing"""

    def __init__(self, llm_client=None):
        """
        Initialize health probe

        Args:
            llm_client: Optional LLM client (if not provided, creates minimal requests)
        """
        self.llm_client = llm_client
        self.failure_classifier = FailureClassifier()

        # Configuration
        self.probe_timeout_seconds = int(os.getenv("HEALTH_PROBE_TIMEOUT_SECONDS", "10"))
        self.enabled = os.getenv("HEALTH_PROBE_ENABLED", "true").lower() == "true"

    async def probe_combination(
        self,
        provider: str,
        model: str,
        api_key_id: str,
        api_key: str,
        endpoint: str,
        version: Optional[str] = None
    ) -> Tuple[bool, Optional[FailureType], str]:
        """
        Send minimal request to test provider health

        Args:
            provider: Provider name
            model: Model name
            api_key_id: API key identifier
            api_key: Actual API key
            endpoint: API endpoint
            version: API version (for Anthropic)

        Returns:
            Tuple of (success: bool, failure_type: FailureType, error_msg: str)
        """
        if not self.enabled:
            return (True, None, "health_probe_disabled")

        logger.info(
            "Sending health probe",
            provider=provider,
            model=model,
            api_key_id=api_key_id
        )

        try:
            if provider == "ollama":
                # Use free version endpoint
                success, error = await self._probe_ollama(endpoint)
            elif provider == "gemini":
                # Use models endpoint (free)
                success, error = await self._probe_gemini(endpoint, api_key, model)
            elif provider == "openai":
                # Minimal completion request
                success, error = await self._probe_openai(endpoint, api_key, model)
            elif provider == "anthropic":
                # Minimal completion request
                success, error = await self._probe_anthropic(endpoint, api_key, model, version)
            else:
                return (False, FailureType.UNKNOWN_ERROR, f"Unsupported provider: {provider}")

            if success:
                logger.info(
                    "Health probe successful",
                    provider=provider,
                    model=model,
                    api_key_id=api_key_id
                )
                return (True, None, "healthy")
            else:
                failure_type = self.failure_classifier.classify_error(error)
                logger.warning(
                    "Health probe failed",
                    provider=provider,
                    model=model,
                    api_key_id=api_key_id,
                    failure_type=failure_type.value,
                    error=error
                )
                return (False, failure_type, error)

        except Exception as e:
            error_msg = str(e)
            failure_type = self.failure_classifier.classify_error(error_msg)

            logger.error(
                "Health probe exception",
                provider=provider,
                model=model,
                api_key_id=api_key_id,
                error=error_msg
            )

            return (False, failure_type, error_msg)

    async def _probe_ollama(self, endpoint: str) -> Tuple[bool, str]:
        """Probe Ollama using free version endpoint"""
        try:
            # Extract base URL
            if "/api/chat" in endpoint:
                base_url = endpoint.replace("/api/chat", "")
            else:
                base_url = endpoint

            health_endpoint = f"{base_url}/api/version"

            timeout = aiohttp.ClientTimeout(total=self.probe_timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_endpoint) as response:
                    if response.status == 200:
                        return (True, "")
                    else:
                        error = f"HTTP {response.status}: {await response.text()}"
                        return (False, error)

        except asyncio.TimeoutError:
            return (False, "Health probe timeout")
        except Exception as e:
            return (False, str(e))

    async def _probe_gemini(self, endpoint: str, api_key: str, model: str) -> Tuple[bool, str]:
        """Probe Gemini using actual generation endpoint (tests rate limits)"""
        try:
            # Use the actual generation endpoint to test rate limits
            # This costs tokens but accurately tests if we can make requests
            payload = {
                "contents": [{"parts": [{"text": "test"}]}],
                "generationConfig": {"maxOutputTokens": 1}
            }

            # Add API key to endpoint
            probe_url = f"{endpoint}?key={api_key}"

            timeout = aiohttp.ClientTimeout(total=self.probe_timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(probe_url, json=payload) as response:
                    if response.status == 200:
                        return (True, "")
                    else:
                        error_text = await response.text()
                        error = f"Gemini API error {response.status}: {error_text}"
                        return (False, error)

        except asyncio.TimeoutError:
            return (False, "Health probe timeout")
        except Exception as e:
            return (False, str(e))

    async def _probe_openai(self, endpoint: str, api_key: str, model: str) -> Tuple[bool, str]:
        """Probe OpenAI with minimal completion request"""
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            timeout = aiohttp.ClientTimeout(total=self.probe_timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, json=payload, headers=headers) as response:
                    if response.status == 200:
                        return (True, "")
                    else:
                        error_text = await response.text()
                        error = f"HTTP {response.status}: {error_text}"
                        return (False, error)

        except asyncio.TimeoutError:
            return (False, "Health probe timeout")
        except Exception as e:
            return (False, str(e))

    async def _probe_anthropic(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        version: Optional[str]
    ) -> Tuple[bool, str]:
        """Probe Anthropic with minimal completion request"""
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }

            headers = {
                "x-api-key": api_key,
                "anthropic-version": version or "2023-06-01",
                "Content-Type": "application/json"
            }

            timeout = aiohttp.ClientTimeout(total=self.probe_timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, json=payload, headers=headers) as response:
                    if response.status == 200:
                        return (True, "")
                    else:
                        error_text = await response.text()
                        error = f"HTTP {response.status}: {error_text}"
                        return (False, error)

        except asyncio.TimeoutError:
            return (False, "Health probe timeout")
        except Exception as e:
            return (False, str(e))

    async def batch_probe(self, circuits: list, config_manager) -> Dict[str, Tuple[bool, Optional[FailureType], str]]:
        """
        Probe multiple circuits concurrently

        Args:
            circuits: List of CircuitState objects to probe
            config_manager: Config manager for getting provider details

        Returns:
            Dictionary mapping combination_key to probe results
        """
        if not self.enabled:
            return {}

        # Group by provider to avoid hammering one provider
        provider_groups = {}
        for circuit in circuits:
            if circuit.provider not in provider_groups:
                provider_groups[circuit.provider] = []
            provider_groups[circuit.provider].append(circuit)

        results = {}

        # Probe each provider group (max 3 concurrent per provider)
        for provider, provider_circuits in provider_groups.items():
            logger.info(
                "Probing provider circuits",
                provider=provider,
                circuit_count=len(provider_circuits)
            )

            # Create probe tasks (max 3 concurrent)
            tasks = []
            for circuit in provider_circuits[:3]:  # Limit to 3 concurrent
                try:
                    provider_config = config_manager.providers[circuit.provider]
                    model_config = provider_config.models[circuit.model]
                    key_config = provider_config.keys[circuit.api_key_id]

                    task = self.probe_combination(
                        provider=circuit.provider,
                        model=circuit.model,
                        api_key_id=circuit.api_key_id,
                        api_key=key_config.api_key,
                        endpoint=model_config.endpoint,
                        version=getattr(model_config, 'version', None)
                    )
                    tasks.append((circuit.combination_key, task))
                except Exception as e:
                    logger.error(
                        "Failed to create probe task",
                        circuit=circuit.combination_key,
                        error=str(e)
                    )

            # Execute probes concurrently
            if tasks:
                task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

                for (combination_key, _), result in zip(tasks, task_results):
                    if isinstance(result, Exception):
                        results[combination_key] = (False, FailureType.UNKNOWN_ERROR, str(result))
                    else:
                        results[combination_key] = result

        return results


class BackgroundProbeWorker:
    """Background worker that periodically probes unhealthy circuits"""

    def __init__(self, circuit_breaker, config_manager, health_probe):
        """
        Initialize background probe worker

        Args:
            circuit_breaker: CircuitBreakerManager instance
            config_manager: Config manager instance
            health_probe: HealthProbe instance
        """
        self.circuit_breaker = circuit_breaker
        self.config = config_manager
        self.health_probe = health_probe

        self.enabled = os.getenv("HEALTH_PROBE_ENABLED", "true").lower() == "true"
        self.interval_seconds = int(os.getenv("HEALTH_PROBE_INTERVAL_SECONDS", "10"))
        self.max_concurrent = int(os.getenv("HEALTH_PROBE_CONCURRENT_MAX", "5"))

        self._task = None
        self._running = False

    async def start(self):
        """Start the background worker"""
        if not self.enabled or not self.circuit_breaker.enabled:
            logger.info("Background health probe worker disabled")
            return

        if self._running:
            logger.warning("Background probe worker already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._worker_loop())

        logger.info(
            "Background health probe worker started",
            interval_seconds=self.interval_seconds,
            max_concurrent=self.max_concurrent
        )

    async def stop(self):
        """Stop the background worker"""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Background health probe worker stopped")

    async def _worker_loop(self):
        """Main worker loop"""
        check_counter = 0

        while self._running:
            try:
                await self._probe_circuits()

                # Every 6 iterations (1 minute if interval=10s), check for stuck locks
                check_counter += 1
                if check_counter % 6 == 0:
                    stats = self.circuit_breaker.get_probe_lock_stats()

                    if stats["stuck_probe_locks"] > 0:
                        logger.error(
                            "ALERT: Stuck probe locks detected",
                            stuck_count=stats["stuck_probe_locks"],
                            stuck_circuits=stats["stuck_circuits"]
                        )

                await asyncio.sleep(self.interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in probe worker loop",
                    error=str(e)
                )
                await asyncio.sleep(self.interval_seconds)

    async def _probe_circuits(self):
        """Probe circuits ready for testing"""
        from datetime import datetime, timezone
        now_dt = datetime.now(timezone.utc)
        probe_timeout_seconds = int(os.getenv("PROBE_LOCK_TIMEOUT_SECONDS", "300"))
        stuck_circuits_cleared = 0

        # First, clean up stuck HALF_OPEN circuits
        for circuit in self.circuit_breaker.circuits.values():
            if circuit.state == CircuitStatus.HALF_OPEN:
                if circuit.test_probe_in_progress and circuit.test_probe_started_at:
                    elapsed = (now_dt - circuit.test_probe_started_at).total_seconds()

                    if elapsed > probe_timeout_seconds:
                        logger.warning(
                            "Background worker clearing stuck probe lock",
                            provider=circuit.provider,
                            model=circuit.model,
                            api_key_id=circuit.api_key_id,
                            elapsed_seconds=elapsed,
                            state=circuit.state.value
                        )

                        # Clear stuck probe lock and reopen circuit for fresh backoff
                        circuit.test_probe_in_progress = False
                        circuit.test_probe_started_at = None

                        # Reopen circuit to start fresh recovery cycle
                        await self.circuit_breaker._open_circuit(
                            circuit,
                            circuit.last_failure_type or FailureType.UNKNOWN_ERROR,
                            None
                        )

                        stuck_circuits_cleared += 1

        if stuck_circuits_cleared > 0:
            logger.info(
                "Background worker cleared stuck probe locks",
                count=stuck_circuits_cleared
            )

        # Find circuits ready to test
        circuits_to_probe = []

        for circuit in self.circuit_breaker.circuits.values():
            # Only probe OPEN circuits that:
            # 1. Have a next_test_time set
            # 2. next_test_time has been reached
            # 3. Require health probes
            # 4. Not currently being probed

            if circuit.state != CircuitStatus.OPEN:
                continue

            if circuit.next_test_time is None:
                continue

            if circuit.test_probe_in_progress:
                continue

            # Check if it's time to test
            from datetime import datetime, timezone
            now_dt = datetime.now(timezone.utc)

            if now_dt < circuit.next_test_time:
                continue

            # Check if this failure type requires probing
            if circuit.last_failure_type:
                if not self.health_probe.failure_classifier.needs_health_probe(circuit.last_failure_type):
                    continue

            circuits_to_probe.append(circuit)

        if not circuits_to_probe:
            return

        # Limit to max concurrent probes
        circuits_to_probe = circuits_to_probe[:self.max_concurrent]

        logger.info(
            "Probing circuits",
            count=len(circuits_to_probe)
        )

        # Mark as in progress
        for circuit in circuits_to_probe:
            circuit.test_probe_in_progress = True
            circuit.test_probe_started_at = now_dt
            circuit.record_state_change(CircuitStatus.HALF_OPEN, "background_probe")

        # Probe all circuits
        probe_results = await self.health_probe.batch_probe(circuits_to_probe, self.config)

        # Process results
        for circuit in circuits_to_probe:
            result = probe_results.get(circuit.combination_key)

            if result:
                success, failure_type, error_msg = result

                if success:
                    # Probe succeeded - record success (will close circuit)
                    await self.circuit_breaker.record_success(
                        circuit.provider,
                        circuit.model,
                        circuit.api_key_id
                    )
                else:
                    # Probe failed - record failure (will reopen circuit with increased backoff)
                    await self.circuit_breaker.record_failure(
                        circuit.provider,
                        circuit.model,
                        circuit.api_key_id,
                        error_msg
                    )
            else:
                # No result - clear probe lock
                circuit.test_probe_in_progress = False
