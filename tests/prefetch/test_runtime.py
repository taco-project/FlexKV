"""Actual background threads and pipes; transfer bytes are tested separately."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace as NS

import pytest

from flexkv.prefetch.runtime import QueuedTransferHandle, TaskRuntime, on_runtime


class Engine:
    def __init__(self):
        self.transfer_handles = []
        self._prefetch = NS(tick=lambda: 0, sessions={}, reserved_bytes=0)
        self.owners = set()
        self.ticks = 0
        self._runtime = TaskRuntime(self)

    def _update_tasks(self, timeout):
        self.owners.add(threading.get_ident())
        self.ticks += 1

    def _reap_completed_tasks(self):
        pass

    @on_runtime
    def mutation(self):
        self.owners.add(threading.get_ident())
        return self.ticks


def test_single_owner_and_unpolled_completion_pump():
    engine = Engine()
    engine._runtime.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: engine.mutation(), range(200)))
        deadline = time.monotonic() + 2
        initial = engine.ticks
        while engine.ticks < initial + 2 and time.monotonic() < deadline:
            with engine._runtime.changed:
                engine._runtime.changed.wait(0.05)
        assert engine.ticks >= initial + 2
        assert engine.owners == {engine._runtime.thread.ident}
    finally:
        engine._runtime.stop()


def test_runtime_failure_wakes_callers_and_retains_failure():
    engine = Engine()
    engine._prefetch.tick = lambda: (_ for _ in ()).throw(RuntimeError("injected"))
    engine._runtime.start()
    engine._runtime.thread.join(timeout=2)
    assert not engine._runtime.thread.is_alive()
    with pytest.raises(RuntimeError, match="retained"):
        engine.mutation()


def test_outbox_hides_blocked_send_and_serializes_writes():
    entered, unblock = threading.Event(), threading.Event()
    threads, received = set(), []

    class Handle:
        _handle = None

        def submit_batch(self, graphs):
            threads.add(threading.get_ident())
            entered.set()
            assert unblock.wait(2)
            received.extend(graphs)

        def shutdown(self):
            pass

    handle = QueuedTransferHandle(Handle())
    try:
        handle.submit_batch([1])
        assert entered.wait(2)
        handle.submit_batch([2])
        assert received == []
        unblock.set()
        handle.queue.join()
        assert received == [1, 2] and len(threads) == 1
    finally:
        unblock.set()
        handle.shutdown()
