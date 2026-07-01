"""Request budgets and escalation reordering."""

import time

from bkvy.core.timeout_manager import GlobalTimeoutManager
from bkvy.models.data_classes import CompletionTimeAnalysis
from bkvy.models.circuit_states import CircuitStatus


def make_tm(monkeypatch, hard=120):
    monkeypatch.setenv("REQUEST_HARD_TIMEOUT_SECONDS", str(hard))
    return GlobalTimeoutManager()


class TestBudget:
    def test_effective_budget(self, monkeypatch):
        tm = make_tm(monkeypatch)
        assert tm.effective_budget(None) == 120
        assert tm.effective_budget(0) == 120       # non-positive = no override
        assert tm.effective_budget(-5) == 120
        assert tm.effective_budget(30) == 30       # caller tightens
        assert tm.effective_budget(500) == 120     # hard timeout still caps

    def test_per_attempt_bounded_by_remaining(self, monkeypatch):
        tm = make_tm(monkeypatch)
        now = time.time()
        assert tm.get_request_timeout(now, escalated=False) in (119, 120)
        assert tm.get_request_timeout(now - 100, escalated=False) in (19, 20)
        assert tm.get_request_timeout(now - 130, escalated=False) == 0
        assert tm.get_request_timeout(now, escalated=True) in (99, 100)

    def test_per_attempt_respects_caller_budget(self, monkeypatch):
        tm = make_tm(monkeypatch)
        now = time.time()
        assert tm.get_request_timeout(now, escalated=False, budget_seconds=30) in (29, 30)
        assert tm.should_abort(now - 31, budget_seconds=30)
        assert not tm.should_abort(now - 31, budget_seconds=None)


def _alt(provider, circuit_state, speed):
    return CompletionTimeAnalysis(
        rate_limit_wait_seconds=0, queue_wait_seconds=0, processing_time_seconds=speed,
        total_seconds=speed, cost_per_1k_tokens=0.0,
        combination_key=f"{provider}_k_m", provider=provider, model="m", api_key_id="k",
        circuit_state=circuit_state)


class TestEscalationReorder:
    def test_half_open_kept_at_tail_not_dropped(self, monkeypatch):
        tm = make_tm(monkeypatch)
        alts = [
            _alt("a", CircuitStatus.HALF_OPEN, 1),
            _alt("b", CircuitStatus.CLOSED, 5),
            _alt("c", CircuitStatus.CLOSED, 2),
        ]
        reordered = tm.reorder_for_escalation(alts, current_provider="b")
        # CLOSED first (diversity then speed), HALF_OPEN retained at the tail
        assert [a.provider for a in reordered] == ["c", "b", "a"]
        assert len(reordered) == 3
