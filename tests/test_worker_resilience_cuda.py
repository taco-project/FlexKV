"""Real CUDA/IPC regressions; fault injection touches only these tests' workers.

Run in the existing GPU container: pytest -q tests/test_worker_resilience_cuda.py
Two GPUs are required. No model weights or external KVServer are needed.
"""
import os
from multiprocessing.connection import wait as wait_connections
import signal
import time

import numpy as np
import pytest
import torch

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.pool import PoolId
from flexkv.common.storage import AccessHandleType, KVCacheLayout, KVCacheLayoutType, StorageHandle
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType, WorkerKey
from flexkv.transfer.transfer_engine import TransferEngine

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")]


def make_engine(swa):
    model = ModelConfig(num_layers=1, num_kv_heads=1, head_size=32,
                        kv_dim=2, dtype=torch.float16, tp_size=1)
    cache = CacheConfig(enable_cpu=True, enable_ssd=False,
                        num_cpu_blocks=16, tokens_per_block=4)
    gpu_layout = KVCacheLayout(type=KVCacheLayoutType.LAYERFIRST, num_layer=1,
        num_block=16, tokens_per_block=4, num_head=1, head_size=32, kv_dim=2, num_kv_heads=1)
    cpu_layout = KVCacheLayout(type=KVCacheLayoutType.BLOCKFIRST, num_layer=1,
        num_block=16, tokens_per_block=4, num_head=1, head_size=32, kv_dim=2, num_kv_heads=1)
    gpu = [torch.full(tuple(gpu_layout.kv_shape[1:]), 7., device=f"cuda:{i}", dtype=torch.float16)
           for i in range(2)]
    host = torch.full(tuple(cpu_layout.kv_shape), 5., dtype=torch.float16).share_memory_()
    handles = {WorkerKey(i, 0): [StorageHandle(AccessHandleType.TENSOR, [gpu[i]],
               gpu_layout, torch.float16, gpu_device_id=i)] for i in range(2)}
    host_handle = StorageHandle(AccessHandleType.TENSOR, host, cpu_layout, torch.float16)
    keepalive = [gpu, host]
    kwargs = {}
    if swa:
        swa_gpu = [torch.full_like(g, 11.) for g in gpu]
        swa_host = torch.full(tuple(cpu_layout.kv_shape), 9., dtype=torch.float16).share_memory_()
        kwargs.update(swa_gpu_handles={WorkerKey(i, 0): [StorageHandle(
            AccessHandleType.TENSOR, [swa_gpu[i]], gpu_layout, torch.float16, gpu_device_id=i)]
            for i in range(2)}, swa_cpu_handle=StorageHandle(
                AccessHandleType.TENSOR, swa_host, cpu_layout, torch.float16))
        keepalive.extend([swa_gpu, swa_host])
    te = TransferEngine(handles, model, cache, cpu_handle=host_handle, **kwargs)
    te.start()
    return te, keepalive


def graph(route, pool=PoolId.FULL_KV):
    g = TransferOpGraph.create_empty_graph()
    g.add_transfer_op(TransferOp(graph_id=g.graph_id, transfer_type=TransferType.H2D,
        src_block_ids=np.array([0], dtype=np.int64), dst_block_ids=np.array([1], dtype=np.int64),
        dp_client_id=route, pool_id=pool))
    return g


def terminal(te, ids, timeout=10.):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events.extend(te.get_completed_graphs_and_ops(timeout=.1))
        done = {event.graph_id for event in events if event.op_id == -1}
        if set(ids) <= done:
            return {event.graph_id: event.failed for event in events if event.op_id == -1}
    raise AssertionError(f"Missing terminal: ids={ids}, events={events}, tracked={te.op_id_to_op}")


@pytest.mark.parametrize("swa", [False, True])
def test_cuda_worker_exit_isolation_and_repeated_recovery(swa):
    te, memory = make_engine(swa)
    try:
        for cycle in range(2):
            baseline = graph(0)
            te.submit_transfer_graph(baseline)
            assert terminal(te, [baseline.graph_id]) == {baseline.graph_id: False}
            torch.cuda.synchronize(0)
            assert torch.all(memory[0][0][:, 1] == 5).item()
            bad, good = graph(999), graph(1)
            te.submit_transfer_graph([bad, good])
            assert terminal(te, [bad.graph_id, good.graph_id]) == {
                bad.graph_id: True, good.graph_id: False}
            old = te.h2d_workers[WorkerKey(0, 0)]
            old.transfer_conn.send(None)
            assert old.quiesced_event.wait(timeout=15)
            assert wait_connections([old.process.sentinel], timeout=15)
            with te._resilience_lock:  # serialize waitpid with the health monitor
                old.process.join(timeout=5)
                assert not old.process.is_alive()
            lost, healthy = graph(0), graph(1)
            te.submit_transfer_graph([lost, healthy])
            assert terminal(te, [lost.graph_id, healthy.graph_id]) == {
                lost.graph_id: True, healthy.graph_id: False}
            assert te.fence_gpu_route(0)["safe"]
            assert te.recover_gpu_route(0, cycle) == {"safe": True, "generation": cycle + 1}
            # A duplicate old completion must never alter a new graph.
            te.finished_ops_queue.put((-999999, True, None))
            restored = graph(0, PoolId.SWA if swa else PoolId.FULL_KV)
            te.submit_transfer_graph(restored)
            assert terminal(te, [restored.graph_id]) == {restored.graph_id: False}
            torch.cuda.synchronize(0)
            actual = memory[2][0] if swa else memory[0][0]
            assert torch.all(actual[:, 1] == (9 if swa else 5)).item()
        deadline = time.monotonic() + 2
        while te.op_id_to_op and time.monotonic() < deadline:
            time.sleep(.01)
        assert not te.op_id_to_op and not te._child_id_to_child and not te._op_workers
        assert te._scheduler_thread.is_alive()
    finally:
        te.shutdown()


def test_cuda_abrupt_worker_loss_quarantines_only_its_route():
    te, memory = make_engine(False)
    victim = te.h2d_workers[WorkerKey(0, 0)]
    try:
        # Stop before submitting, so a legal transfer stays accepted by the
        # pipe. SIGKILL deliberately bypasses the worker's drain acknowledgement.
        os.kill(victim.process.pid, signal.SIGSTOP)
        lost = graph(0)
        te.submit_transfer_graph(lost)
        deadline = time.monotonic() + 5
        while not te._op_workers and time.monotonic() < deadline:
            time.sleep(.01)
        assert te._op_workers
        os.kill(victim.process.pid, signal.SIGKILL)
        with te._resilience_lock:
            victim.process.join(timeout=5)
        assert not victim.quiesced_event.is_set()
        assert te.fence_gpu_route(0)["safe"] is False
        with pytest.raises(RuntimeError, match="quarantined"):
            te.recover_gpu_route(0, 0)
        healthy = graph(1)
        te.submit_transfer_graph(healthy)
        assert terminal(te, [healthy.graph_id]) == {healthy.graph_id: False}
        assert any(op.graph_id == lost.graph_id for op in te.op_id_to_op.values())
        assert te._child_id_to_child and te._op_workers
        assert te._scheduler_thread.is_alive()
    finally:
        if victim.process.is_alive():
            os.kill(victim.process.pid, signal.SIGCONT)
        te.shutdown()


def test_cuda_graceful_loss_of_accepted_op_drains_and_fails_once():
    te, memory = make_engine(False)
    victim = te.h2d_workers[WorkerKey(0, 0)]
    try:
        os.kill(victim.process.pid, signal.SIGSTOP)
        lost = graph(0)
        te.submit_transfer_graph(lost)
        deadline = time.monotonic() + 5
        while not te._op_workers and time.monotonic() < deadline:
            time.sleep(.01)
        assert te._op_workers
        os.kill(victim.process.pid, signal.SIGTERM)
        os.kill(victim.process.pid, signal.SIGCONT)
        assert terminal(te, [lost.graph_id]) == {lost.graph_id: True}
        assert victim.quiesced_event.is_set()
        assert not te.op_id_to_op and not te._child_id_to_child and not te._op_workers
        assert te.fence_gpu_route(0)["safe"]
        healthy = graph(1)
        te.submit_transfer_graph(healthy)
        assert terminal(te, [healthy.graph_id]) == {healthy.graph_id: False}
    finally:
        if victim.process.is_alive():
            os.kill(victim.process.pid, signal.SIGCONT)
        te.shutdown()
