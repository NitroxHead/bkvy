"""Queue accounting and per-combination concurrency caps."""

import asyncio

import pytest

from bkvy.core.queues import QueueManager


def run(coro):
    return asyncio.run(coro)


class TestInflightAccounting:
    def test_counts_reflect_load(self):
        async def scenario():
            q = QueueManager()
            assert await q.get_queue_wait_time("p", "m", "k", 2000) == 0
            q.note_inflight_start("p", "m", "k")
            q.note_inflight_start("p", "m", "k")
            wait = await q.get_queue_wait_time("p", "m", "k", 2000)
            assert wait == 4.0  # 2 in flight * 2s avg
            q.note_inflight_end("p", "m", "k")
            q.note_inflight_end("p", "m", "k")
            q.note_inflight_end("p", "m", "k")  # extra end never goes negative
            assert await q.get_queue_wait_time("p", "m", "k", 2000) == 0

        run(scenario())


class TestConcurrencyCap:
    def test_cap_enforced(self, monkeypatch):
        monkeypatch.setenv("MAX_CONCURRENT_PER_COMBINATION", "2")

        async def scenario():
            q = QueueManager()
            assert await q.try_acquire_slot("p", "m", "k", timeout=0.05)
            assert await q.try_acquire_slot("p", "m", "k", timeout=0.05)
            # Third concurrent slot must time out
            assert not await q.try_acquire_slot("p", "m", "k", timeout=0.05)
            q.release_slot("p", "m", "k")
            assert await q.try_acquire_slot("p", "m", "k", timeout=0.05)

        run(scenario())

    def test_zero_means_unlimited(self, monkeypatch):
        monkeypatch.setenv("MAX_CONCURRENT_PER_COMBINATION", "0")

        async def scenario():
            q = QueueManager()
            for _ in range(50):
                assert await q.try_acquire_slot("p", "m", "k", timeout=0.01)
            q.release_slot("p", "m", "k")  # no-op, must not raise

        run(scenario())

    def test_independent_per_combination(self, monkeypatch):
        monkeypatch.setenv("MAX_CONCURRENT_PER_COMBINATION", "1")

        async def scenario():
            q = QueueManager()
            assert await q.try_acquire_slot("p", "m", "k1", timeout=0.05)
            # Different key is a different combination: unaffected
            assert await q.try_acquire_slot("p", "m", "k2", timeout=0.05)
            assert not await q.try_acquire_slot("p", "m", "k1", timeout=0.05)

        run(scenario())
