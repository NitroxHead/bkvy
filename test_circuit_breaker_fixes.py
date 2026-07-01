"""
Unit tests for circuit breaker bug fixes.

Bug #1: Flapping penalty can never clear automatically
Bug #2: Closed+flapping circuits invisible to probes
Bug #3: HALF_OPEN failure should always reopen circuit
Bug #4: backoff unbound for AUTH_ERROR_4XX
"""

import asyncio
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from bkvy.models.circuit_states import CircuitState, CircuitStatus, FailureType
from bkvy.core.circuit_breaker import CircuitBreakerManager


class FakeConfigManager:
    def __init__(self):
        self.providers = {}


class FakePersistence:
    """No-op persistence for tests"""
    async def save_state(self, circuit):
        pass

    async def load_all_states(self):
        return {}

    async def cleanup_old_states(self, valid):
        pass


def make_circuit(provider="test", model="m1", key="k1", state=CircuitStatus.CLOSED):
    c = CircuitState.create_new(provider, model, key)
    c.state = state
    return c


def make_manager():
    mgr = CircuitBreakerManager(FakeConfigManager(), FakePersistence())
    mgr.enabled = True
    return mgr


class TestBug4_BackoffUnbound(unittest.TestCase):
    """Bug #4: _open_circuit crashed with UnboundLocalError for AUTH_ERROR_4XX"""

    def test_open_circuit_auth_error_no_crash(self):
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.CLOSED)
        circuit.consecutive_failures = 3

        # This used to raise UnboundLocalError: local variable 'backoff' referenced before assignment
        asyncio.run(
            mgr._open_circuit(circuit, FailureType.AUTH_ERROR_4XX, None)
        )

        self.assertEqual(circuit.state, CircuitStatus.OPEN)
        self.assertIsNone(circuit.next_test_time)

    def test_open_circuit_rate_limit_still_works(self):
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.CLOSED)
        circuit.consecutive_failures = 3

        asyncio.run(
            mgr._open_circuit(circuit, FailureType.RATE_LIMIT_429, None)
        )

        self.assertEqual(circuit.state, CircuitStatus.OPEN)
        self.assertIsNotNone(circuit.next_test_time)

    def test_open_circuit_service_error_still_works(self):
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.CLOSED)
        circuit.consecutive_failures = 3

        asyncio.run(
            mgr._open_circuit(circuit, FailureType.SERVICE_ERROR_5XX, None)
        )

        self.assertEqual(circuit.state, CircuitStatus.OPEN)
        self.assertIsNotNone(circuit.next_test_time)


class TestBug1_FlappingNeverClears(unittest.TestCase):
    """Bug #1: Flapping penalty was permanent because _check_flapping_clear
    compared against last_success_time which was always ~now."""

    def test_flapping_clears_after_stable_period(self):
        """Flapping should clear when stable_since is > 10 minutes ago"""
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.CLOSED)
        circuit.is_flapping = True
        circuit.priority_penalty = 1000
        circuit.flapping_detected_at = datetime.now(timezone.utc) - timedelta(minutes=20)
        # Simulate: circuit was closed 15 minutes ago
        circuit.stable_since = datetime.now(timezone.utc) - timedelta(minutes=15)

        asyncio.run(
            mgr._check_flapping_clear(circuit)
        )

        self.assertFalse(circuit.is_flapping)
        self.assertEqual(circuit.priority_penalty, 0)
        self.assertIsNone(circuit.flapping_detected_at)

    def test_flapping_does_not_clear_too_early(self):
        """Flapping should NOT clear when stable_since is < 10 minutes ago"""
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.CLOSED)
        circuit.is_flapping = True
        circuit.priority_penalty = 1000
        # Simulate: circuit was closed 5 minutes ago
        circuit.stable_since = datetime.now(timezone.utc) - timedelta(minutes=5)

        asyncio.run(
            mgr._check_flapping_clear(circuit)
        )

        self.assertTrue(circuit.is_flapping)
        self.assertEqual(circuit.priority_penalty, 1000)

    def test_flapping_backfills_stable_since_if_missing(self):
        """Pre-existing circuits without stable_since get it set on first check"""
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.CLOSED)
        circuit.is_flapping = True
        circuit.priority_penalty = 1000
        circuit.stable_since = None  # pre-existing circuit

        asyncio.run(
            mgr._check_flapping_clear(circuit)
        )

        # Should have backfilled but NOT cleared yet
        self.assertIsNotNone(circuit.stable_since)
        self.assertTrue(circuit.is_flapping)

    def test_close_circuit_sets_stable_since(self):
        """_close_circuit should set stable_since for flapping timing"""
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.HALF_OPEN)
        circuit.consecutive_failures = 3
        circuit.stable_since = None

        before = datetime.now(timezone.utc)
        asyncio.run(
            mgr._close_circuit(circuit)
        )
        after = datetime.now(timezone.utc)

        self.assertEqual(circuit.state, CircuitStatus.CLOSED)
        self.assertIsNotNone(circuit.stable_since)
        self.assertGreaterEqual(circuit.stable_since, before)
        self.assertLessEqual(circuit.stable_since, after)

    def test_open_circuit_clears_stable_since(self):
        """_open_circuit should clear stable_since"""
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.CLOSED)
        circuit.stable_since = datetime.now(timezone.utc) - timedelta(hours=1)
        circuit.consecutive_failures = 3

        asyncio.run(
            mgr._open_circuit(circuit, FailureType.SERVICE_ERROR_5XX, None)
        )

        self.assertIsNone(circuit.stable_since)

    def test_record_success_checks_flapping_on_closed(self):
        """record_success on a CLOSED+flapping circuit should check flapping clear"""
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.CLOSED)
        circuit.is_flapping = True
        circuit.priority_penalty = 1000
        # Stable for 15 minutes
        circuit.stable_since = datetime.now(timezone.utc) - timedelta(minutes=15)

        key = circuit.combination_key
        mgr.circuits[key] = circuit

        asyncio.run(
            mgr.record_success("test", "m1", "k1")
        )

        self.assertFalse(circuit.is_flapping)
        self.assertEqual(circuit.priority_penalty, 0)

    def test_full_lifecycle_flapping_eventually_clears(self):
        """End-to-end: circuit flaps, closes, and after 10 min stable the penalty clears"""
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.CLOSED)
        key = circuit.combination_key
        mgr.circuits[key] = circuit

        async def lifecycle():
            # Simulate flapping: open 3 times in 5 minutes
            for _ in range(3):
                await mgr._open_circuit(circuit, FailureType.SERVICE_ERROR_5XX, None)
                circuit.state = CircuitStatus.HALF_OPEN
                await mgr._close_circuit(circuit)

            self.assertTrue(circuit.is_flapping)
            self.assertEqual(circuit.priority_penalty, 1000)

            # Simulate time passing: backdate stable_since by 11 minutes
            circuit.stable_since = datetime.now(timezone.utc) - timedelta(minutes=11)

            # A success should now clear flapping
            await mgr.record_success("test", "m1", "k1")

        asyncio.run(lifecycle())

        self.assertFalse(circuit.is_flapping)
        self.assertEqual(circuit.priority_penalty, 0)


class TestBug3_HalfOpenOrphan(unittest.TestCase):
    """Bug #3: HALF_OPEN circuit with non-circuit-breaking failure stayed stuck"""

    def test_half_open_failure_always_reopens(self):
        """Any failure during HALF_OPEN should reopen, even if should_circuit_break=False"""
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.HALF_OPEN)
        circuit.test_probe_in_progress = True
        circuit.consecutive_failures = 0  # Edge case: reset after previous close
        key = circuit.combination_key
        mgr.circuits[key] = circuit

        # CONTENT_ERROR has should_circuit_break=False
        asyncio.run(
            mgr.record_failure("test", "m1", "k1", "empty content response")
        )

        # Must be OPEN, not stuck in HALF_OPEN
        self.assertEqual(circuit.state, CircuitStatus.OPEN)
        self.assertFalse(circuit.test_probe_in_progress)

    def test_half_open_failure_unknown_error_reopens(self):
        """UNKNOWN_ERROR (should_circuit_break=False) in HALF_OPEN should also reopen"""
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.HALF_OPEN)
        circuit.test_probe_in_progress = True
        circuit.consecutive_failures = 0
        key = circuit.combination_key
        mgr.circuits[key] = circuit

        asyncio.run(
            mgr.record_failure("test", "m1", "k1", "some weird error nobody expected")
        )

        self.assertEqual(circuit.state, CircuitStatus.OPEN)
        self.assertFalse(circuit.test_probe_in_progress)

    def test_half_open_failure_normal_error_still_reopens(self):
        """Normal circuit-breaking errors in HALF_OPEN also reopen (no regression)"""
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.HALF_OPEN)
        circuit.test_probe_in_progress = True
        circuit.consecutive_failures = 3
        key = circuit.combination_key
        mgr.circuits[key] = circuit

        asyncio.run(
            mgr.record_failure("test", "m1", "k1", "HTTP 500: internal server error")
        )

        self.assertEqual(circuit.state, CircuitStatus.OPEN)


class TestBug2_ProbeWorkerFlappingCheck(unittest.TestCase):
    """Bug #2: Probe worker never checked CLOSED+flapping circuits"""

    def test_probe_worker_clears_flapping(self):
        """Probe worker should clear flapping on CLOSED circuits that have been stable"""
        from bkvy.core.health_probe import BackgroundProbeWorker, HealthProbe

        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.CLOSED)
        circuit.is_flapping = True
        circuit.priority_penalty = 1000
        circuit.stable_since = datetime.now(timezone.utc) - timedelta(minutes=15)
        key = circuit.combination_key
        mgr.circuits[key] = circuit

        config = FakeConfigManager()
        probe = HealthProbe()
        probe.enabled = False
        worker = BackgroundProbeWorker(mgr, config, probe)

        asyncio.run(worker._probe_circuits())

        self.assertFalse(circuit.is_flapping)
        self.assertEqual(circuit.priority_penalty, 0)


class TestCircuitStateSerialization(unittest.TestCase):
    """Verify stable_since survives serialization round-trip"""

    def test_stable_since_round_trip(self):
        circuit = make_circuit()
        circuit.stable_since = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        data = circuit.to_dict()
        restored = CircuitState.from_dict(data)

        self.assertEqual(restored.stable_since, circuit.stable_since)

    def test_stable_since_none_round_trip(self):
        circuit = make_circuit()
        circuit.stable_since = None

        data = circuit.to_dict()
        restored = CircuitState.from_dict(data)

        self.assertIsNone(restored.stable_since)

    def test_old_data_without_stable_since(self):
        """Pre-existing persisted data without stable_since should load fine"""
        circuit = make_circuit()
        data = circuit.to_dict()
        del data['stable_since']

        restored = CircuitState.from_dict(data)
        self.assertIsNone(restored.stable_since)


class TestResetCircuit(unittest.TestCase):
    """Verify reset_circuit properly clears everything including stable_since"""

    def test_reset_sets_stable_since(self):
        mgr = make_manager()
        circuit = make_circuit(state=CircuitStatus.OPEN)
        circuit.is_flapping = True
        circuit.priority_penalty = 1000
        circuit.stable_since = None
        key = circuit.combination_key
        mgr.circuits[key] = circuit

        before = datetime.now(timezone.utc)
        asyncio.run(
            mgr.reset_circuit("test", "m1", "k1")
        )
        after = datetime.now(timezone.utc)

        self.assertEqual(circuit.state, CircuitStatus.CLOSED)
        self.assertFalse(circuit.is_flapping)
        self.assertEqual(circuit.priority_penalty, 0)
        self.assertIsNotNone(circuit.stable_since)
        self.assertGreaterEqual(circuit.stable_since, before)
        self.assertLessEqual(circuit.stable_since, after)


if __name__ == "__main__":
    unittest.main()
