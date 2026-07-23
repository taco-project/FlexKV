# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVTask lifecycle, transfer results, and SWA load locks.

The control-plane cases cover:

1. The ``KVTask`` dataclass state machine: status transitions, is_completed,
   early-response cleanup, ``swa_slot_mapping``, and shed_heavy_resources.

2. Per-block transfer-result aggregation and partial-load failure handling.

3. The SWA load-lock lifecycle on ``GlobalCacheEngine``: a GET pins the matched
   CPU SWA node while building the GET plan, and the SWA H2D completion callback
   releases it (``_swa_release_load_lock``), leaving the
   node cached (dec_swa_lock_ref, NOT dec_swa_lock_only). This documents that
   the SWA lock follows the SAME lifecycle as the full-KV node lock (both taken
   in get()/put(), both released via the op/transfer callbacks) — so a fresh
   GET after a completed GET re-locks cleanly with no residual pin.

The SWA engine cases require ``flexkv.c_ext``; none starts a TransferManager
subprocess or requires a GPU.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flexkv.common.request import KVResponseStatus
from flexkv.common.transfer import (
    CompletedOp,
    CompletionAwareCallback,
    TransferType,
)
from flexkv.kvtask import (
    KVTask,
    KVTaskEngine,
    KVTaskManager,
    TaskStatus,
    TaskType,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# 1. KVTask state machine (pure, no engine)                                    #
# --------------------------------------------------------------------------- #

def _make_task(**over):
    base = dict(
        task_id=1, task_type=TaskType.GET, task_end_op_id=0,
        task_end_op_finished=False, status=TaskStatus.UNREADY,
        token_ids=np.arange(4, dtype=np.int64),
        slot_mapping=np.arange(4, dtype=np.int64),
        token_mask=np.ones(4, dtype=np.int64),
        graph=None, return_mask=np.zeros(4, dtype=np.bool_),
        callback=None, op_callback_dict={},
    )
    base.update(over)
    return KVTask(**base)


class _FakeGraph:
    def __init__(self, graph_id=10, num_ops=2):
        self.graph_id = graph_id
        self.num_ops = num_ops


def _make_task_engine(task):
    engine = KVTaskEngine.__new__(KVTaskEngine)
    engine.tasks = {task.task_id: task}
    engine.graph_to_task = {task.graph.graph_id: task.task_id}
    # Completion is driven explicitly by each test.
    engine._update_tasks = lambda timeout=0: None
    return engine


def test_task_defaults_and_swa_field():
    t = _make_task()
    assert t.status == TaskStatus.UNREADY
    assert t.swa_slot_mapping is None          # new field defaults None
    assert t.request_returned is False
    assert not t.is_completed()


def test_task_swa_slot_mapping_settable():
    sm = np.arange(16, dtype=np.int64)
    t = _make_task(swa_slot_mapping=sm)
    assert t.swa_slot_mapping is sm


@pytest.mark.parametrize("status,done", [
    (TaskStatus.UNREADY, False),
    (TaskStatus.READY, False),
    (TaskStatus.RUNNING, False),
    (TaskStatus.COMPLETED, True),
    (TaskStatus.CANCELLED, True),
    (TaskStatus.FAILED, True),
])
def test_is_completed_matrix(status, done):
    assert _make_task(status=status).is_completed() is done


def test_shed_heavy_resources_keeps_status():
    t = _make_task(status=TaskStatus.COMPLETED,
                   token_ids=np.arange(4, dtype=np.int64))
    t.shed_heavy_resources()
    assert t.graph is None and t.token_ids is None
    assert t.slot_mapping is None and t.token_mask is None
    assert t.callback is None
    # status/return_mask survive so wait() can still report
    assert t.status == TaskStatus.COMPLETED
    assert t.return_mask is not None


def test_early_return_marks_request_returned_and_keeps_running_task():
    graph = _FakeGraph()
    return_mask = np.array([True, False], dtype=np.bool_)
    task = _make_task(
        status=TaskStatus.RUNNING,
        task_end_op_finished=True,
        graph=graph,
        return_mask=return_mask,
    )
    engine = _make_task_engine(task)

    response = engine._wait_impl(
        [task.task_id], timeout=0, completely=False
    )[task.task_id]

    assert response.status == KVResponseStatus.SUCCESS
    np.testing.assert_array_equal(response.return_mask, return_mask)
    assert task.request_returned is True
    assert engine.tasks[task.task_id] is task
    assert engine.graph_to_task[graph.graph_id] == task.task_id


def test_graph_completion_releases_task_after_early_return():
    graph = _FakeGraph()
    task = _make_task(
        status=TaskStatus.RUNNING,
        task_end_op_finished=True,
        graph=graph,
        request_returned=True,
    )
    engine = _make_task_engine(task)
    callback_calls = []

    def callback():
        assert task.task_id in engine.tasks
        callback_calls.append("completed")

    task.callback = callback
    engine._mark_completed(task.task_id)

    assert callback_calls == ["completed"]
    assert task.status == TaskStatus.COMPLETED
    assert task.task_end_op_finished is True
    assert task.graph is None
    assert task.task_id not in engine.tasks
    assert graph.graph_id not in engine.graph_to_task


def test_completed_task_remains_observable_until_first_response():
    graph = _FakeGraph()
    return_mask = np.array([True, True], dtype=np.bool_)
    task = _make_task(
        status=TaskStatus.RUNNING,
        graph=graph,
        return_mask=return_mask,
    )
    engine = _make_task_engine(task)

    engine._mark_completed(task.task_id)
    assert task.task_id in engine.tasks
    assert task.request_returned is False
    assert graph.graph_id not in engine.graph_to_task

    response = engine._wait_impl(
        [task.task_id], timeout=0, completely=True
    )[task.task_id]
    assert response.status == KVResponseStatus.SUCCESS
    np.testing.assert_array_equal(response.return_mask, return_mask)
    assert task.request_returned is True
    assert task.task_id not in engine.tasks


class _CompletedOpsHandle:
    def __init__(self, completed_ops):
        self.completed_ops = completed_ops

    def wait(self, _timeout):
        return self.completed_ops

    def shutdown(self):
        pass


def test_multi_handle_block_results_are_combined_with_and():
    """A block is reusable only when every transfer handle succeeded."""
    graph_id, op_id = 41, 73
    manager = object.__new__(KVTaskManager)
    manager.transfer_handles = [
        _CompletedOpsHandle([CompletedOp(
            graph_id=graph_id,
            op_id=op_id,
            num_blocks=3,
            block_results=(True, True, False),
        )]),
        _CompletedOpsHandle([CompletedOp(
            graph_id=graph_id,
            op_id=op_id,
            num_blocks=3,
            block_results=(True, False, True),
        )]),
    ]
    manager.required_completed_count = 2
    manager.uncompleted_ops = {}
    manager.uncompleted_op_results = {}
    manager.uncompleted_graphs = {}

    assert manager._get_completed_ops(timeout=0) == [CompletedOp(
        graph_id=graph_id,
        op_id=op_id,
        num_blocks=3,
        block_results=(True, False, False),
    )]
    assert manager.uncompleted_ops == {}
    assert manager.uncompleted_op_results == {}


@pytest.mark.parametrize("second_results", [None, (True, True)])
def test_multi_handle_invalid_block_results_fail_closed(second_results):
    graph_id, op_id = 43, 76
    manager = object.__new__(KVTaskManager)
    manager.transfer_handles = [
        _CompletedOpsHandle([CompletedOp(
            graph_id=graph_id,
            op_id=op_id,
            num_blocks=3,
            block_results=(True, True, True),
        )]),
        _CompletedOpsHandle([CompletedOp(
            graph_id=graph_id,
            op_id=op_id,
            num_blocks=3,
            block_results=second_results,
        )]),
    ]
    manager.required_completed_count = 2
    manager.uncompleted_ops = {}
    manager.uncompleted_op_results = {}
    manager.uncompleted_graphs = {}

    assert manager._get_completed_ops(timeout=0) == [CompletedOp(
        graph_id=graph_id,
        op_id=op_id,
        num_blocks=3,
        block_results=(False, False, False),
    )]


def test_partial_remote2h_does_not_return_early_success():
    graph_id, op_id, task_end_op_id = 42, 74, 75
    completion = CompletedOp(
        graph_id=graph_id,
        op_id=op_id,
        transfer_type=TransferType.REMOTE2H.value,
        num_blocks=3,
        block_results=(True, False, False),
    )
    seen = []
    task = _make_task(
        status=TaskStatus.RUNNING,
        task_end_op_id=task_end_op_id,
        graph=SimpleNamespace(graph_id=graph_id, num_ops=2, _op_map={}),
        op_callback_dict={
            op_id: CompletionAwareCallback(lambda result: seen.append(result))
        },
    )
    manager = object.__new__(KVTaskManager)
    manager.graph_to_task = {graph_id: task.task_id}
    manager.tasks = {task.task_id: task}
    manager._get_completed_ops = lambda _timeout: [completion]

    manager._update_tasks(timeout=0)

    assert task.transfer_failed is True
    assert task.task_end_op_finished is False
    assert seen == [completion]
    assert manager.check_completed(task.task_id) is False

    manager._get_completed_ops = lambda _timeout: [CompletedOp(
        graph_id=graph_id,
        op_id=task_end_op_id,
        transfer_type=TransferType.H2D.value,
    )]
    manager._update_tasks(timeout=0)

    assert task.task_end_op_finished is True
    assert manager.check_completed(task.task_id) is False

    manager._get_completed_ops = lambda _timeout: [
        CompletedOp.completed_graph(graph_id)
    ]
    manager._update_tasks(timeout=0)

    assert task.status == TaskStatus.FAILED


# --- M15/M16: SWA-aware get's core arithmetic (pure, mirrors kvtask logic) --- #

def _usable_and_swa_mask(full_hit, swa_hit, num_tokens, tpb):
    """Replicate the SWA-aware get arithmetic so the contract is pinned without
    standing up a KVTaskEngine (which needs a GPU TransferManager subprocess).
    usable=min(full,swa) clamps the full transfer; the trailing SWA window is
    the last block of the clamped hit."""
    usable = min(full_hit, swa_hit)
    full_mask = np.ones(num_tokens, dtype=np.bool_)
    truncated = full_mask.copy()
    truncated[usable * tpb:] = False
    swa_mask = np.zeros(num_tokens, dtype=np.bool_)
    if swa_hit > 0:
        swa_mask[(swa_hit - 1) * tpb: swa_hit * tpb] = True
    return usable, truncated, swa_mask


@pytest.mark.parametrize("full_hit,swa_hit,exp_usable", [
    (10, 6, 6),   # M15.1: SWA shorter than full -> clamp full to SWA
    (6, 6, 6),    # M15.4: SWA covers whole full hit -> no truncation loss
    (10, 0, 0),   # M15.3: no SWA hit -> usable 0 -> empty full graph
])
def test_get_swa_usable_is_min(full_hit, swa_hit, exp_usable):
    tpb, num_tokens = 16, 10 * 16
    usable, truncated, _ = _usable_and_swa_mask(full_hit, swa_hit, num_tokens, tpb)
    assert usable == exp_usable
    # The full transfer is shaped to exactly `usable` blocks (truncate-before-build).
    assert int(truncated.sum()) == exp_usable * tpb


@pytest.mark.parametrize("swa_hit", [1, 3, 6])
def test_get_swa_trailing_window_is_one_block(swa_hit):
    """M16.1: SWA is page-granular, so return_mask_swa
    marks exactly ONE block, ending at swa_hit."""
    tpb, num_tokens = 16, 6 * 16
    _, _, swa_mask = _usable_and_swa_mask(6, swa_hit, num_tokens, tpb)
    assert int(swa_mask.sum()) == tpb                       # exactly one block
    assert swa_mask[(swa_hit - 1) * tpb: swa_hit * tpb].all()  # the trailing one


def test_get_swa_no_hit_empty_window_no_underflow():
    """M16.2: swa_hit=0 -> return_mask_swa all-zero, and the (swa_hit-1) index is
    never evaluated (no negative-index wraparound marking the last block)."""
    tpb, num_tokens = 16, 6 * 16
    _, _, swa_mask = _usable_and_swa_mask(6, 0, num_tokens, tpb)
    assert not swa_mask.any()


# --------------------------------------------------------------------------- #
# 2. SWA load-lock lifecycle on the engine (needs c_ext)                       #
# --------------------------------------------------------------------------- #

c_ext = pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.debug import flexkv_logger

flexkv_logger.set_level("OFF")
TPB = 16


def _engine():
    mc = ModelConfig(num_layers=4, num_kv_heads=1, head_size=128,
                     use_mla=True, dtype=torch.bfloat16, tp_size=1, dp_size=1)
    cc = CacheConfig(tokens_per_block=TPB, enable_cpu=True, enable_ssd=False,
                     enable_remote=False, num_cpu_blocks=4096)
    cc.swa = SWAPoolConfig(
        enabled=True,
        num_slots=256,
        num_swa_layers=1,
        bytes_per_token_per_layer=64,
    )
    cc.enable_swa_transfer = True
    return GlobalCacheEngine(cc, mc)


def _tokens(base):
    rs = np.random.RandomState(base)
    return rs.randint(0, 30000, size=4 * TPB, dtype=np.int64)


def _put(eng, tok):
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    for c in op_cb.values():
        c()
    cb()


def test_get_pins_then_releases_swa_lock():
    """GET pins the matched CPU SWA node; the SWA H2D callback releases it
    (dec_swa_lock_ref -> node stays cached, lock back to 0)."""
    eng = _engine()
    tok = _tokens(21)
    _put(eng, tok)

    graph, _rm, cb, op_cb, end_id = eng.get(
        request_id=2, token_ids=tok, token_mask=np.ones_like(tok, dtype=np.int64),
        slot_mapping=np.arange(tok.shape[0], dtype=np.int64), dp_client_id=0,
        swa_aware=True)
    swa_h2d = [o for o in graph._op_map.values() if o.is_swa][0]

    # After building the GET graph, the matched CPU SWA node is pinned.
    sm = SequenceMeta(token_ids=tok, tokens_per_block=TPB); sm.gen_hashes()
    mr = eng.cpu_cache_engine.match(sm)
    assert mr.swa_hit_blocks == 4
    # The node carries the load pin (>=1) taken while building the GET plan.
    node = mr.last_swa_node
    assert node is not None
    assert node.swa_lock_ref >= 1

    # Complete the ops: the SWA H2D callback releases the pin.
    for c in op_cb.values():
        c()
    cb()
    node2 = eng.cpu_cache_engine.index.match_prefix(
        torch.from_numpy(sm.block_hashes[:4]).to(torch.int64), 4, False).last_swa_node
    assert node2.swa_lock_ref == 0, "SWA load lock not released on H2D completion"
    # slot still live (dec_swa_lock_ref keeps the cache, unlike dec_swa_lock_only)
    assert node2.has_swa()


def test_repeated_get_relocks_cleanly():
    """Two sequential GETs of the same prefix each pin+release with no residual
    (proves the release is paired, not leaking across requests)."""
    eng = _engine()
    tok = _tokens(22)
    _put(eng, tok)
    sm = SequenceMeta(token_ids=tok, tokens_per_block=TPB); sm.gen_hashes()
    bh = torch.from_numpy(sm.block_hashes[:4]).to(torch.int64)

    for req in (2, 3):
        _g, _rm, cb, op_cb, _e = eng.get(
            request_id=req, token_ids=tok,
            token_mask=np.ones_like(tok, dtype=np.int64),
            slot_mapping=np.arange(tok.shape[0], dtype=np.int64), dp_client_id=0,
            swa_aware=True)
        for c in op_cb.values():
            c()
        cb()
        node = eng.cpu_cache_engine.index.match_prefix(bh, 4, False).last_swa_node
        assert node.swa_lock_ref == 0, f"residual SWA lock after GET {req}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
