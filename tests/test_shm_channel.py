"""
Tests for the shared memory IPC channel.

Tests:
1. ShmChannel: ring buffer correctness, sync request/response
2. ShmControlBlock: wake counter, server ready flag
3. Multi-process integration: server + client doing real operations
4. Message serialization round-trips
"""

import os
import sys
import time
import struct
import multiprocessing as mp
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flexkv.server.shm_channel import (
    ShmChannel, ShmControlBlock, ShmMsgType,
    pack_request, unpack_request, pack_response, unpack_response,
    SYNC_REQ_OFFSET, SYNC_RESP_OFFSET, TOTAL_SHM_SIZE,
    MSG_HEADER_SIZE,
)
from flexkv.common.request import KVResponseStatus, KVResponse


# ── Fixtures ────────────────────────────────────────────────────────────

SERVER_ID = "test_shm"


@pytest.fixture(autouse=True)
def cleanup_shm():
    """Clean up shared memory files before and after each test."""
    def _cleanup():
        for f in os.listdir("/dev/shm"):
            if f.startswith("flexkv_shm_"):
                try:
                    os.unlink(f"/dev/shm/{f}")
                except FileNotFoundError:
                    pass
    _cleanup()
    yield
    _cleanup()


# ── Unit Tests: Message Serialization ───────────────────────────────────

class TestMessageSerialization:
    def test_pack_unpack_put_request(self):
        """Test round-trip serialization of a PUT request with numpy arrays."""
        import mmap
        buf = mmap.mmap(-1, TOTAL_SHM_SIZE)

        token_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        slot_mapping = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        token_mask = np.array([True, True, False, True, True])

        pack_request(buf, 0, ShmMsgType.PUT_ASYNC, dp_client_id=0,
                     task_id=42, token_ids=token_ids,
                     slot_mapping=slot_mapping, token_mask=token_mask)

        msg = unpack_request(buf, 0)

        assert msg["msg_type"] == ShmMsgType.PUT_ASYNC
        assert msg["dp_client_id"] == 0
        assert msg["task_id"] == 42
        assert np.array_equal(msg["token_ids"], token_ids)
        assert np.array_equal(msg["slot_mapping"], slot_mapping)
        assert np.array_equal(msg["token_mask"], token_mask)
        buf.close()

    def test_pack_unpack_get_match_request(self):
        """Test GET_MATCH request with layer_granularity."""
        import mmap
        buf = mmap.mmap(-1, TOTAL_SHM_SIZE)

        token_ids = np.random.randint(0, 32000, size=512, dtype=np.int64)
        token_mask = np.ones(512, dtype=bool)

        pack_request(buf, 0, ShmMsgType.GET_MATCH, dp_client_id=1,
                     task_id=100, token_ids=token_ids,
                     token_mask=token_mask, layer_granularity=32)

        msg = unpack_request(buf, 0)
        assert msg["msg_type"] == ShmMsgType.GET_MATCH
        assert msg["task_id"] == 100
        assert msg["layer_granularity"] == 32
        assert np.array_equal(msg["token_ids"], token_ids)
        assert np.array_equal(msg["token_mask"], token_mask)
        buf.close()

    def test_pack_unpack_wait_request(self):
        """Test WAIT request with task_ids and timeout."""
        import mmap
        buf = mmap.mmap(-1, TOTAL_SHM_SIZE)

        task_ids = [100, 200, 300]
        pack_request(buf, 0, ShmMsgType.WAIT, dp_client_id=0,
                     task_ids=task_ids, wait_timeout=5.0, completely=True)

        msg = unpack_request(buf, 0)
        assert msg["msg_type"] == ShmMsgType.WAIT
        assert msg["task_ids"] == task_ids
        assert abs(msg["wait_timeout"] - 5.0) < 0.001
        assert msg["completely"] is True
        buf.close()

    def test_pack_unpack_launch_request(self):
        """Test LAUNCH_TASKS with slot_mappings."""
        import mmap
        buf = mmap.mmap(-1, TOTAL_SHM_SIZE)

        task_ids = [1, 2]
        sm1 = np.array([10, 20, 30], dtype=np.int64)
        sm2 = np.array([40, 50], dtype=np.int64)

        pack_request(buf, 0, ShmMsgType.LAUNCH_TASKS, dp_client_id=0,
                     task_ids=task_ids, slot_mappings=[sm1, sm2],
                     as_batch=True, batch_id=99)

        msg = unpack_request(buf, 0)
        assert msg["msg_type"] == ShmMsgType.LAUNCH_TASKS
        assert msg["task_ids"] == task_ids
        assert msg["as_batch"] is True
        assert msg["batch_id"] == 99
        assert len(msg["slot_mappings"]) == 2
        assert np.array_equal(msg["slot_mappings"][0], sm1)
        assert np.array_equal(msg["slot_mappings"][1], sm2)
        buf.close()

    def test_pack_unpack_namespace(self):
        """Test request with namespace."""
        import mmap
        buf = mmap.mmap(-1, TOTAL_SHM_SIZE)

        token_ids = np.array([1, 2, 3], dtype=np.int64)
        namespace = ["model_a", "tenant_1"]

        pack_request(buf, 0, ShmMsgType.PUT_ASYNC, dp_client_id=0,
                     task_id=1, token_ids=token_ids,
                     namespace=namespace)

        msg = unpack_request(buf, 0)
        assert msg["namespace"] == namespace
        buf.close()

    def test_pack_unpack_response_with_mask(self):
        """Test response with mask array."""
        import mmap
        buf = mmap.mmap(-1, TOTAL_SHM_SIZE)

        mask = np.array([1, 1, 0, 1, 0], dtype=np.uint8)
        pack_response(buf, 0, status_code=0, task_id=42, mask=mask)

        resp = unpack_response(buf, 0)
        assert resp["task_id"] == 42
        assert np.array_equal(resp["mask"], mask)
        buf.close()

    def test_pack_unpack_response_with_kv_responses(self):
        """Test response with KVResponse dict (as returned by wait/try_wait)."""
        import mmap
        buf = mmap.mmap(-1, TOTAL_SHM_SIZE)

        return_mask = np.array([1, 1, 0, 1], dtype=np.uint8)
        kv_responses = {
            100: KVResponse(status=KVResponseStatus.SUCCESS, task_id=100, return_mask=return_mask),
            200: KVResponse(status=KVResponseStatus.TIMEOUT, task_id=200, return_mask=None),
        }

        pack_response(buf, 0, kv_responses=kv_responses)

        resp = unpack_response(buf, 0)
        assert resp["kv_responses"] is not None
        assert len(resp["kv_responses"]) == 2
        assert resp["kv_responses"][100].status == KVResponseStatus.SUCCESS
        assert np.array_equal(resp["kv_responses"][100].return_mask, return_mask)
        assert resp["kv_responses"][200].status == KVResponseStatus.TIMEOUT
        assert resp["kv_responses"][200].return_mask is None
        buf.close()

    def test_pack_unpack_response_with_batched_mask(self):
        """Test response with list of masks (batched)."""
        import mmap
        buf = mmap.mmap(-1, TOTAL_SHM_SIZE)

        mask1 = np.array([1, 1, 0], dtype=np.uint8)
        mask2 = np.array([0, 1, 1, 1], dtype=np.uint8)
        kv_responses = {
            99: KVResponse(status=KVResponseStatus.SUCCESS, task_id=99, return_mask=[mask1, mask2]),
        }

        pack_response(buf, 0, kv_responses=kv_responses)
        resp = unpack_response(buf, 0)

        rm = resp["kv_responses"][99].return_mask
        assert isinstance(rm, list)
        assert len(rm) == 2
        assert np.array_equal(rm[0], mask1)
        assert np.array_equal(rm[1], mask2)
        buf.close()


# ── Unit Tests: ShmChannel ──────────────────────────────────────────────

class TestShmChannel:
    def test_async_ring_buffer(self):
        """Test async ring buffer: write N messages, read N messages."""
        ch = ShmChannel(SERVER_ID, client_id=0, create=True)
        ch_server = ShmChannel(SERVER_ID, client_id=0, create=False)

        token_ids = np.array([1, 2, 3], dtype=np.int64)
        N = 100

        for i in range(N):
            ch.async_send(ShmMsgType.PUT_ASYNC, dp_client_id=0,
                          task_id=i, token_ids=token_ids)

        for i in range(N):
            msg = ch_server.async_recv()
            assert msg is not None
            assert msg["task_id"] == i
            assert np.array_equal(msg["token_ids"], token_ids)

        # Ring should be empty now
        assert ch_server.async_recv() is None

        ch.close()
        ch.unlink()
        ch_server.close()

    def test_async_ring_wrap_around(self):
        """Test ring buffer wraps around correctly."""
        ch = ShmChannel(SERVER_ID, client_id=1, create=True)
        ch_server = ShmChannel(SERVER_ID, client_id=1, create=False)

        token_ids = np.array([1], dtype=np.int64)

        # Write and read repeatedly to exercise wrap-around
        for batch in range(10):
            for i in range(200):
                ch.async_send(ShmMsgType.GET_ASYNC, dp_client_id=0,
                              task_id=batch * 200 + i, token_ids=token_ids)

            for i in range(200):
                msg = ch_server.async_recv()
                assert msg is not None
                assert msg["task_id"] == batch * 200 + i

        ch.close()
        ch.unlink()
        ch_server.close()

    def test_sync_request_response_same_process(self):
        """Test sync request/response in same process (simulated)."""
        ch = ShmChannel(SERVER_ID, client_id=2, create=True)

        token_ids = np.array([10, 20, 30], dtype=np.int64)

        # Client side: write request
        ch.sync_send_request(ShmMsgType.GET_MATCH, dp_client_id=0,
                             task_id=42, token_ids=token_ids,
                             layer_granularity=32)

        # Server side: read request
        req = ch.check_sync_request()
        assert req is not None
        assert req["msg_type"] == ShmMsgType.GET_MATCH
        assert req["task_id"] == 42

        # Server side: write response
        mask = np.array([1, 1, 0], dtype=np.uint8)
        ch.send_sync_response(task_id=42, mask=mask)

        # Client side: read response (skip spin, flag is already set)
        resp = ch.sync_wait_response(spin_iters=1)
        assert resp["task_id"] == 42
        assert np.array_equal(resp["mask"], mask)

        ch.close()
        ch.unlink()


# ── Unit Tests: ShmControlBlock ─────────────────────────────────────────

class TestShmControlBlock:
    def test_server_ready_flag(self):
        """Test server ready flag notification."""
        ctrl = ShmControlBlock(SERVER_ID, create=True)
        ctrl2 = ShmControlBlock(SERVER_ID, create=False)

        # Not ready initially
        assert struct.unpack_from("<i", ctrl2.buf, 64)[0] == 0

        ctrl.set_server_ready()

        # Now ready
        assert ctrl2.wait_server_ready(timeout_s=1.0)

        ctrl.close()
        ctrl.unlink()
        ctrl2.close()

    def test_wake_counter(self):
        """Test wake counter increment."""
        ctrl = ShmControlBlock(SERVER_ID, create=True)

        assert ctrl.get_wake_counter() == 0
        ctrl.notify_server()
        assert ctrl.get_wake_counter() == 1
        ctrl.notify_server()
        assert ctrl.get_wake_counter() == 2

        ctrl.close()
        ctrl.unlink()


# ── Multi-Process Tests ─────────────────────────────────────────────────

def _sync_server_worker(server_id, n_clients, ready_event, stop_event):
    """Simple server that handles sync requests in a loop."""
    ctrl = ShmControlBlock(server_id, create=True)
    channels = {}
    for cid in range(n_clients):
        channels[cid] = ShmChannel(server_id, cid, create=True)

    ctrl.set_server_ready()
    ready_event.set()

    while not stop_event.is_set():
        for cid, ch in channels.items():
            req = ch.check_sync_request()
            if req is not None:
                msg_type = req["msg_type"]
                if msg_type == ShmMsgType.GET_MATCH:
                    n = len(req["token_ids"])
                    mask = np.ones(n, dtype=np.uint8)
                    ch.send_sync_response(task_id=req["task_id"], mask=mask)
                elif msg_type == ShmMsgType.PUT_MATCH:
                    n = len(req["token_ids"])
                    mask = np.ones(n, dtype=np.uint8)
                    ch.send_sync_response(task_id=req["task_id"], mask=mask)
                elif msg_type == ShmMsgType.WAIT:
                    kv_responses = {}
                    for tid in req["task_ids"]:
                        kv_responses[tid] = KVResponse(
                            status=KVResponseStatus.SUCCESS,
                            task_id=tid,
                            return_mask=np.array([1, 1], dtype=np.uint8),
                        )
                    ch.send_sync_response(kv_responses=kv_responses)
                elif msg_type == ShmMsgType.IS_READY:
                    ch.send_sync_response(is_ready=True)
                elif msg_type == ShmMsgType.SHUTDOWN:
                    ch.send_sync_response(status_code=0)
                    stop_event.set()

            # Also drain async ring
            msg = ch.async_recv()
            while msg is not None:
                msg = ch.async_recv()

    for ch in channels.values():
        ch.close()
        ch.unlink()
    ctrl.close()
    ctrl.unlink()


class TestMultiProcess:
    def test_sync_round_trip(self):
        """Test sync request/response across processes."""
        server_id = "test_mp_sync"
        ready = mp.Event()
        stop = mp.Event()

        p = mp.Process(target=_sync_server_worker, args=(server_id, 1, ready, stop))
        p.start()
        ready.wait(timeout=5)

        ctrl = ShmControlBlock(server_id, create=False)
        ch = ShmChannel(server_id, client_id=0, create=False)

        token_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        # Test get_match
        ch.sync_send_request(ShmMsgType.GET_MATCH, dp_client_id=0,
                             task_id=42, token_ids=token_ids,
                             layer_granularity=32)
        ctrl.notify_server()
        resp = ch.sync_wait_response()
        assert resp["task_id"] == 42
        assert len(resp["mask"]) == 5

        # Test wait
        ch.sync_send_request(ShmMsgType.WAIT, dp_client_id=0,
                             task_ids=[100, 200])
        ctrl.notify_server()
        resp = ch.sync_wait_response()
        assert resp["kv_responses"] is not None
        assert 100 in resp["kv_responses"]
        assert resp["kv_responses"][100].status == KVResponseStatus.SUCCESS

        # Shutdown
        ch.sync_send_request(ShmMsgType.SHUTDOWN, dp_client_id=0)
        ctrl.notify_server()
        ch.sync_wait_response()

        ch.close()
        ctrl.close()
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    def test_async_fire_and_forget(self):
        """Test async ring buffer across processes."""
        server_id = "test_mp_async"
        ready = mp.Event()
        stop = mp.Event()

        p = mp.Process(target=_sync_server_worker, args=(server_id, 1, ready, stop))
        p.start()
        ready.wait(timeout=5)

        ctrl = ShmControlBlock(server_id, create=False)
        ch = ShmChannel(server_id, client_id=0, create=False)

        token_ids = np.array([1, 2, 3], dtype=np.int64)
        slot_mapping = np.array([10, 20, 30], dtype=np.int64)

        # Send many async messages
        N = 500
        for i in range(N):
            ch.async_send(ShmMsgType.PUT_ASYNC, dp_client_id=0,
                          task_id=i, token_ids=token_ids,
                          slot_mapping=slot_mapping)
            ctrl.notify_server()

        time.sleep(0.5)  # let server drain

        # Shutdown
        ch.sync_send_request(ShmMsgType.SHUTDOWN, dp_client_id=0)
        ctrl.notify_server()
        ch.sync_wait_response()

        ch.close()
        ctrl.close()
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    def test_multi_client(self):
        """Test 2 clients communicating with server concurrently."""
        server_id = "test_mp_multi"
        ready = mp.Event()
        stop = mp.Event()

        p = mp.Process(target=_sync_server_worker, args=(server_id, 2, ready, stop))
        p.start()
        ready.wait(timeout=5)

        results = mp.Queue()

        def client_worker(cid, server_id, n_iters, result_queue):
            ctrl = ShmControlBlock(server_id, create=False)
            ch = ShmChannel(server_id, client_id=cid, create=False)

            token_ids = np.random.randint(0, 32000, size=64, dtype=np.int64)

            for i in range(n_iters):
                ch.sync_send_request(ShmMsgType.GET_MATCH, dp_client_id=cid,
                                     task_id=cid * 10000 + i,
                                     token_ids=token_ids,
                                     layer_granularity=32)
                ctrl.notify_server()
                resp = ch.sync_wait_response()
                assert resp["task_id"] == cid * 10000 + i

            result_queue.put(cid)
            ch.close()
            ctrl.close()

        workers = []
        for cid in range(2):
            w = mp.Process(target=client_worker, args=(cid, server_id, 200, results))
            workers.append(w)
            w.start()

        for w in workers:
            w.join(timeout=30)

        completed = []
        while not results.empty():
            completed.append(results.get())
        assert len(completed) == 2

        # Shutdown
        ctrl = ShmControlBlock(server_id, create=False)
        ch = ShmChannel(server_id, client_id=0, create=False)
        ch.sync_send_request(ShmMsgType.SHUTDOWN, dp_client_id=0)
        ctrl.notify_server()
        ch.sync_wait_response()
        ch.close()
        ctrl.close()

        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    def test_latency_benchmark(self):
        """Measure round-trip latency (informational, not a pass/fail test)."""
        server_id = "test_mp_bench"
        ready = mp.Event()
        stop = mp.Event()

        p = mp.Process(target=_sync_server_worker, args=(server_id, 1, ready, stop))
        p.start()
        ready.wait(timeout=5)

        ctrl = ShmControlBlock(server_id, create=False)
        ch = ShmChannel(server_id, client_id=0, create=False)

        token_ids = np.random.randint(0, 32000, size=512, dtype=np.int64)

        # Warmup
        for _ in range(500):
            ch.sync_send_request(ShmMsgType.GET_MATCH, dp_client_id=0,
                                 task_id=0, token_ids=token_ids,
                                 layer_granularity=32)
            ctrl.notify_server()
            ch.sync_wait_response()

        # Measure
        N = 5000
        latencies = []
        for i in range(N):
            t0 = time.perf_counter_ns()
            ch.sync_send_request(ShmMsgType.GET_MATCH, dp_client_id=0,
                                 task_id=i, token_ids=token_ids,
                                 layer_granularity=32)
            ctrl.notify_server()
            ch.sync_wait_response()
            t1 = time.perf_counter_ns()
            latencies.append(t1 - t0)

        latencies = np.array(latencies)
        print(f"\n  SHM get_match round-trip (512 tokens, {N} iters):")
        print(f"    mean={latencies.mean()/1000:.1f}us  "
              f"p50={np.median(latencies)/1000:.1f}us  "
              f"p99={np.percentile(latencies,99)/1000:.1f}us")

        # Shutdown
        ch.sync_send_request(ShmMsgType.SHUTDOWN, dp_client_id=0)
        ctrl.notify_server()
        ch.sync_wait_response()

        ch.close()
        ctrl.close()
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
