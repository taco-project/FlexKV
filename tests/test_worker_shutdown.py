"""Shutting a worker down must happen exactly once, however many times it is asked.

Two things conspired here.  ``TransferEngine`` registers one handle under
*several* ``TransferType`` keys -- mooncake-store and the peer worker each
answer a read and a write type, GDS answers DISK2D and D2DISK -- and
``_shutdown_worker_handles`` starts one thread *per list element*.  So the same
process got two concurrent threads running ``send(None)`` / ``join`` /
``terminate`` / ``close`` against it, and ``__del__`` could fire a third from
the GC thread later.  The fix is on both sides: dedupe the list by identity,
and make ``shutdown()`` itself idempotent and thread-safe.

Nothing here spawns a real process; the point is the call *pattern*, which a
stand-in observes exactly as well and without a 30s terminate timeout.
"""
import threading
import time


from flexkv.common.pool import PoolId
from flexkv.transfer.workers.handle import WorkerHandle


class FakeConn:
    def __init__(self):
        self.sent = []
        self.closed = 0

    def send(self, obj):
        self.sent.append(obj)

    def close(self):
        self.closed += 1


class FakeProcess:
    """A process that exits on the first join, and records every call.

    ``join_delay`` lets a test hold the first caller inside the critical
    section long enough for a second to arrive while it is still there.
    """

    def __init__(self, join_delay: float = 0.0):
        self.join_delay = join_delay
        self.calls = []
        self._alive = True
        self._lock = threading.Lock()

    def is_alive(self):
        with self._lock:
            return self._alive

    def join(self, timeout=None):
        self.calls.append(("join", timeout))
        if self.join_delay:
            time.sleep(self.join_delay)
        with self._lock:
            self._alive = False

    def terminate(self):
        self.calls.append(("terminate", None))
        with self._lock:
            self._alive = False

    def kill(self):
        self.calls.append(("kill", None))
        with self._lock:
            self._alive = False


def _handle(join_delay: float = 0.0):
    conn, proc = FakeConn(), FakeProcess(join_delay)
    return WorkerHandle(worker_id=7, transfer_conn=conn, process=proc,
                        ready_event=None), conn, proc


def test_repeated_shutdown_stops_the_process_once():
    h, conn, proc = _handle()
    h.shutdown()
    h.shutdown()
    h.shutdown()

    assert conn.sent == [None], "the stop sentinel must be sent once"
    assert conn.closed == 1, "closing a closed pipe is what raised OSError"
    assert [c[0] for c in proc.calls] == ["join"]


# ---------------------------------------------------------------------------
# The other half: the engine must not hand the same handle out twice.
# ---------------------------------------------------------------------------


def test_collect_worker_handles_dedupes_by_identity():
    """One worker serving two TransferTypes is one handle, not two.

    Deduped by identity rather than ``worker_id`` so the test does not encode
    an assumption about how ids are assigned -- and so two genuinely distinct
    workers that happened to share an id would still both be stopped.
    """
    from flexkv.transfer.transfer_engine import TransferEngine

    class H:
        def __init__(self, wid):
            self.worker_id = wid

    shared = H(0)      # e.g. mooncake: REMOTE2H and H2REMOTE
    gds = H(1)         # DISK2D and D2DISK
    solo = H(2)
    swa = H(3)

    engine = object.__new__(TransferEngine)
    engine._worker_map = {
        "REMOTE2H": shared,
        "H2REMOTE": shared,
        "DISK2D": {0: gds, 1: gds},   # per-device nesting, same handle twice
        "H2D": solo,
    }
    # A second pool in the same registry: its workers must be collected too,
    # and after FULL_KV's, since shutdown log order is asserted below.
    engine._workers[PoolId.SWA] = {"H2D": swa, "D2H": swa}

    handles = TransferEngine._collect_worker_handles(engine)

    assert handles == [shared, gds, solo, swa], "order must stay stable"
    assert len(handles) == len({id(h) for h in handles})


