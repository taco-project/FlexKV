"""CPU regressions for blocked remote pipes, command ordering and shutdown.

Uses the production WorkerHandle and real pipes/processes. No CUDA/HSA calls
or RDMA I/O are issued; the local H2D-labelled command carries metadata only.
"""

import multiprocessing as mp
import os
import queue
import socket
import threading
import time
import types
import unittest
from unittest.mock import patch
import numpy as np
import pytest
import flexkv.transfer.workers.handle as worker
from flexkv.transfer.workers.remote import CPURemoteTransferWorker
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker
from flexkv.transfer.workers.runtime import TransferWorkerBase
from flexkv.common.transfer import TransferType
import flexkv.transfer.worker_sender as sender

pytestmark = pytest.mark.unit


def op(module, kind, n, ident):
    return types.SimpleNamespace(
        op_id=ident,
        graph_id=ident,
        transfer_type=kind,
        src_slot_id=0,
        dst_slot_id=1,
        valid_block_num=n,
        src_block_ids=np.arange(n, dtype=np.int64),
        dst_block_ids=np.arange(n, dtype=np.int64),
        src_block_node_ids=None,
        mooncake_store_block_hashes=np.arange(n, dtype=np.uint64),
        mooncake_store_swa_block_hashes=None,
    )


def small_pipe():
    a, b = mp.Pipe(duplex=True)
    with socket.socket(fileno=os.dup(a.fileno())) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
    return a, b


def independent_H2D(module):
    a, b = small_pipe()
    c, d = mp.Pipe(duplex=True)
    release = threading.Event()
    fast_done = threading.Event()
    seen = []
    errors = []
    failed = queue.Queue()
    remote = module.WorkerHandle(1, a, None, None, failed, True)
    local = module.WorkerHandle(2, c, None, None)
    assert not remote._remote_sender._started, "Must not start sender threads during worker initialization"

    def receiver():
        try:
            assert release.wait(3)
            while True:
                payload = b.recv()
                if payload is None:
                    seen.append(None)
                    return
                assert np.array_equal(payload.src_block_ids, np.arange(1117))
                seen.append(payload.transfer_op_id)
        except BaseException as exc:
            errors.append(repr(exc))

    def fast():
        try:
            assert d.poll(3)
            payload = d.recv()
            assert payload.transfer_op_id == 100
            fast_done.set()
        except BaseException as exc:
            errors.append(repr(exc))

    readers = [threading.Thread(target=receiver, daemon=True), threading.Thread(target=fast, daemon=True)]
    for t in readers:
        t.start()
    begin = time.monotonic()
    for i in range(8):
        remote.submit_transfer(op(module, TransferType.H2REMOTE, 1117, i))
    remote_submit_seconds = time.monotonic() - begin
    local.submit_transfer(op(module, TransferType.H2D, 4, 100))
    assert fast_done.wait(1), errors
    assert not release.is_set() and seen == [] and failed.empty(), "No remote success may be fabricated while blocked"
    remote._remote_sender.close()
    release.set()
    assert remote._remote_sender.join(3)
    for t in readers:
        t.join(3)
    assert not errors and seen == list(range(8)) + [None] and failed.empty(), (errors, seen)
    for x in [a, b, c, d]:
        x.close()
    return dict(
        remote_submit_seconds=remote_submit_seconds,
        unrelated_H2D_before_slow_receiver_release=True,
        all_eight_metadata_arrays_and_FIFO_verified=True,
        no_early_remote_completions=True,
    )


def control_FIFO(module):
    a, b = mp.Pipe(duplex=True)
    failed = queue.Queue()
    h = module.WorkerHandle(3, a, None, None, failed, True)
    seen = []

    def worker():
        while True:
            x = b.recv()
            if x is None:
                seen.append(None)
                return
            if isinstance(x, dict):
                assert seen == list(range(5))
                seen.append("control")
                b.send(dict(request_id=x["request_id"], result="barrier-ok"))
            else:
                seen.append(x.transfer_op_id)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    for i in range(5):
        h.submit_transfer(op(module, TransferType.H2REMOTE, 4, i))
    assert h.control("barrier", timeout=3) == "barrier-ok"
    for i in range(5, 10):
        h.submit_transfer(op(module, TransferType.H2REMOTE, 4, i))
    h._remote_sender.close()
    h._remote_sender.close()
    assert h._remote_sender.join(3)
    t.join(3)
    assert seen == list(range(5)) + ["control"] + list(range(5, 10)) + [None] and failed.empty(), seen
    a.close()
    b.close()
    return dict(FIFO=seen, sentinel_once=True)


def full_queue(module, sender):
    begun = threading.Event()
    release = threading.Event()
    seen = []

    class BlockedConnection:
        def send(self, payload):
            begun.set()
            assert release.wait(3)
            seen.append(None if payload is None else payload.transfer_op_id)

    q = queue.Queue()
    h = module.WorkerHandle(4, BlockedConnection(), None, None, q, False)
    h._remote_sender = sender.OrderedPipeSender(h.transfer_conn, h._fail_unsent, capacity=2)
    h.submit_transfer(op(module, TransferType.H2REMOTE, 4, 0))
    assert begun.wait(1)
    for i in [1, 2, 3]:
        h.submit_transfer(op(module, TransferType.H2REMOTE, 4, i))
    assert q.get(timeout=1) == (3, False, None) and q.empty()
    h._remote_sender.close()
    release.set()
    assert h._remote_sender.join(3)
    assert seen == [0, 1, 2, None] and q.empty(), seen
    return dict(queue_full_rejected_once=True, accepted_metadata_retained=True, shutdown_sentinel_reserved=True)


def broken_pipe(module):
    a, b = small_pipe()
    b.close()
    q = queue.Queue()
    h = module.WorkerHandle(5, a, None, None, q, True)
    for i in range(8):
        h.submit_transfer(op(module, TransferType.H2REMOTE, 1117, i))
    failed = [q.get(timeout=3) for _ in range(8)]
    assert sorted(x[0] for x in failed) == list(range(8)) and all(x[1:] == (False, None) for x in failed)
    assert q.empty() and h._remote_sender.join(3)
    try:
        h.control("barrier", timeout=0.1)
    except module.SenderClosed:
        pass
    else:
        raise AssertionError("Broken worker must not accept control")
    a.close()
    return dict(all_eight_unsent_ops_failed_exactly_once=True, no_success_ACK=True, no_resubmit=True)


def process_worker(child, parent_fd, ready, release, result):
    os.close(parent_fd)
    ready.set()
    release.wait(5)
    seen = []
    while True:
        payload = child.recv()
        if payload is None:
            result.put(seen)
            child.close()
            return
        seen.append(payload.transfer_op_id)


def real_shutdown(module, force):
    ctx = mp.get_context("fork")
    a, b = small_pipe()
    ready = ctx.Event()
    release = ctx.Event()
    result = ctx.Queue()
    failures = queue.Queue()
    p = ctx.Process(target=process_worker, args=(b, a.fileno(), ready, release, result))
    p.start()
    b.close()
    assert ready.wait(2)
    h = module.WorkerHandle(6, a, p, ready, failures, True)
    for i in range(4):
        h.submit_transfer(op(module, TransferType.H2REMOTE, 1117, i))
    if not force:
        release.set()
    begin = time.monotonic()
    h.shutdown()
    seconds = time.monotonic() - begin
    assert not p.is_alive() and h._remote_sender.join(1)
    if force:
        failed = [failures.get(timeout=3) for _ in range(4)]
        assert sorted(x[0] for x in failed) == list(range(4)) and all(x[1:] == (False, None) for x in failed)
        assert p.exitcode != 0 and failures.empty()
    else:
        assert p.exitcode == 0 and result.get(timeout=2) == list(range(4)) and failures.empty()
    result.close()
    result.join_thread()
    p.close()
    return dict(forced=force, seconds=seconds, no_child_or_sender_left_alive=True, failed_metadata_accounted=force)


class TestRemoteWorkerPipe(unittest.TestCase):
    def test_concurrent_shutdown_sends_only_one_sentinel(self):
        entered, release = threading.Event(), threading.Event()
        seen = []

        class Process:
            def join(self, timeout):
                entered.set()
                assert release.wait(3)

            def is_alive(self):
                return False

        class Connection:
            def send(self, payload):
                seen.append(payload)

            def close(self):
                seen.append("closed")

        handle = worker.WorkerHandle(7, Connection(), Process(), None, queue.Queue(), True)
        callers = [threading.Thread(target=handle.shutdown) for _ in range(2)]
        for caller in callers:
            caller.start()
        assert entered.wait(1)
        release.set()
        for caller in callers:
            caller.join(3)
            assert not caller.is_alive()
        assert seen == [None, "closed"]

    def test_blocked_remote_does_not_block_unrelated_metadata(self):
        independent_H2D(worker)

    def test_control_and_shutdown_preserve_data_fifo(self):
        control_FIFO(worker)

    def test_full_queue_reports_one_negative_completion(self):
        full_queue(worker, sender)

    def test_broken_pipe_accounts_for_all_unsent_operations(self):
        broken_pipe(worker)

    def test_graceful_shutdown_drains_and_joins(self):
        with patch.object(worker.GLOBAL_CONFIG_FROM_ENV, "worker_shutdown_timeout_s", 0.2):
            real_shutdown(worker, False)

    def test_forced_shutdown_joins_and_accounts_for_unsent_metadata(self):
        with patch.object(worker.GLOBAL_CONFIG_FROM_ENV, "worker_shutdown_timeout_s", 0.2):
            real_shutdown(worker, True)

    def test_sender_starts_lazily_and_rejects_after_close(self):
        class Connection:
            def send(self, payload):
                pass

        s = sender.OrderedPipeSender(Connection(), lambda payload, error: None)
        self.assertFalse(s._started)
        s.close()
        self.assertTrue(s.join(2))
        with self.assertRaises(sender.SenderClosed):
            s.submit(object())

    def test_only_store_workers_enable_isolation(self):
        self.assertTrue(CPURemoteTransferWorker._is_remote_worker)
        self.assertFalse(getattr(GPUCPUTransferWorker, "_is_remote_worker", False))
        self.assertFalse(getattr(TransferWorkerBase, "_is_remote_worker", False))
