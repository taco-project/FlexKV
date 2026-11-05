"""Cancelling a never-launched task must roll back everything its plan holds.

``create_get_task`` / ``create_put_task`` run the cache engine's planner at
creation time, which locks matched radix nodes and allocates staging blocks for
the transfer to write into. Under insert-after those staging blocks are *not*
mounted on any radix tree at plan time: they stay detached until a completion
callback rematches and publishes the prefix that was actually written. The vLLM
adapter cancels unlaunched tasks routinely (a matched request that is not
scheduled this step, and every preemption), so dropping such a task without
aborting leaks its staging permanently -- the blocks are reachable from neither
the tree nor the mempool.

These tests pin the three halves of that lifecycle:

* **detached staging** -- a planned-but-unfinished transfer is invisible to
  ``match()`` and never shadows a later put of the same prefix;
* **completion publication** -- each tier becomes matchable when the ops that
  write *that tier* complete, not when the whole graph drains (PUT reports
  success at its D2H, so CPU must not wait on SSD);
* **cancellation recycling** -- ``abort()`` (wired into
  ``KVTaskManager._cancel_task``) returns every still-detached staging block to
  its mempool, releases every node the plan pinned, and is mutually exclusive
  with the completion callback.
"""
import numpy as np
import pytest

from flexkv import c_ext
from flexkv.cache.cache_engine import (
    CacheEngine,
    CacheEngineAccel,
    GlobalCacheEngine,
    TransferPlanHandle,
)
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.transfer import DeviceType, TransferType
from flexkv.kvtask import KVTaskManager, TaskStatus

pytestmark = pytest.mark.unit

TPB = 16

# The compiled extension is required for the Accel variants; the pure-Python
# variants run anywhere. (A stubbed c_ext module has no __file__.)
_HAS_REAL_C_EXT = getattr(c_ext, "__file__", None) is not None

ENGINE_CLASSES = [
    pytest.param(CacheEngine, id="CacheEngine"),
    pytest.param(CacheEngineAccel, id="CacheEngineAccel",
                 marks=pytest.mark.skipif(not _HAS_REAL_C_EXT,
                                          reason="requires compiled c_ext")),
]

INDEX_ACCEL_MODES = [
    pytest.param(False, id="python-index"),
    pytest.param(True, id="accel-index",
                 marks=pytest.mark.skipif(not _HAS_REAL_C_EXT,
                                          reason="requires compiled c_ext")),
]


def _tokens(num_blocks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 50000, num_blocks * TPB, dtype=np.int64)


def _seq(tokens: np.ndarray) -> SequenceMeta:
    return SequenceMeta(token_ids=tokens, tokens_per_block=TPB)


def _assert_no_locks(engine) -> None:
    """No plan may leave a radix node pinned once it is resolved."""
    index = engine.index
    root = getattr(index, "root_node", None)
    if root is None:  # the accel index does not expose the tree from Python
        return
    stack = [root]
    while stack:
        node = stack.pop()
        assert node.lock_cnt == 0, "no plan may leave a lock behind"
        stack.extend(node.children.values())


# --------------------------------------------------------------------------
# Tier-engine level: staging is detached until it is inserted
# --------------------------------------------------------------------------

@pytest.fixture(params=ENGINE_CLASSES)
def tier_engine(request):
    return request.param(device_type=DeviceType.CPU,
                         num_total_blocks=64,
                         tokens_per_block=TPB,
                         evict_ratio=0.05)


def test_staged_blocks_are_invisible_until_inserted(tier_engine):
    """``take()`` reserves blocks without touching the tree: detached staging."""
    tokens = _tokens(4, seed=1)
    free_before = tier_engine.mempool.num_free_blocks

    blocks = tier_engine.take(4)

    assert tier_engine.mempool.num_free_blocks == free_before - 4
    assert tier_engine.index.total_cached_blocks() == 0
    assert tier_engine.match(_seq(tokens)).num_matched_blocks == 0

    node = tier_engine.insert(_seq(tokens), blocks)

    assert node is not None
    assert tier_engine.index.total_cached_blocks() == 4
    assert tier_engine.match(_seq(tokens)).num_matched_blocks == 4


def test_recycling_staging_restores_the_pool_and_leaves_no_node(tier_engine):
    """The cancellation shape: staged, never inserted, handed straight back."""
    tokens = _tokens(4, seed=2)
    free_before = tier_engine.mempool.num_free_blocks

    blocks = tier_engine.take(4)
    tier_engine.recycle(blocks)

    assert tier_engine.mempool.num_free_blocks == free_before
    assert tier_engine.index.total_cached_blocks() == 0
    # the prefix was never claimed, so a fresh put of the same tokens lands
    node = tier_engine.insert(_seq(tokens), tier_engine.take(4))
    assert node is not None
    assert tier_engine.match(_seq(tokens)).num_matched_blocks == 4


def test_abandoned_staging_does_not_shadow_a_later_put(tier_engine):
    """No unready node exists, so a second writer of the same prefix wins.

    This is the regression the readiness API carried: an ``is_ready=False``
    node claimed the prefix, a concurrent put skipped those blocks, and the
    content ended up written by nobody. Insert-after cannot reach that state.
    """
    tokens = _tokens(4, seed=3)
    abandoned = tier_engine.take(4)  # a plan that will be cancelled

    second = tier_engine.take(4)
    node = tier_engine.insert(_seq(tokens), second)
    assert node is not None
    match = tier_engine.match(_seq(tokens))
    assert match.num_matched_blocks == 4
    np.testing.assert_array_equal(match.physical_blocks[:4], second)

    tier_engine.recycle(abandoned)
    # the published prefix is untouched by the cancellation
    assert tier_engine.match(_seq(tokens)).num_matched_blocks == 4


def test_locked_node_survives_eviction_pressure(tier_engine):
    """Plan-time pins are what keep a matched prefix alive mid-transfer."""
    tokens = _tokens(4, seed=4)
    node = tier_engine.insert(_seq(tokens), tier_engine.take(4))

    tier_engine.lock_node(node)
    with pytest.raises(RuntimeError):
        tier_engine.take(tier_engine.num_total_blocks, strict=True)
    assert tier_engine.match(_seq(tokens)).num_matched_blocks == 4

    tier_engine.unlock(node)
    drained = tier_engine.take(tier_engine.num_total_blocks, strict=True)
    assert drained.shape == (tier_engine.num_total_blocks,)
    assert tier_engine.index.total_cached_blocks() == 0


# --------------------------------------------------------------------------
# TransferPlanHandle: completion and abort are mutually exclusive
# --------------------------------------------------------------------------

def test_plan_handle_runs_each_path_at_most_once():
    calls = []
    handle = TransferPlanHandle(complete=lambda: calls.append("complete"),
                                abort=lambda: calls.append("abort"))
    handle()
    handle()
    handle.abort()
    assert calls == ["complete"]

    calls.clear()
    handle = TransferPlanHandle(complete=lambda: calls.append("complete"),
                                abort=lambda: calls.append("abort"))
    handle.abort()
    handle.abort()
    handle()
    assert calls == ["abort"]


# --------------------------------------------------------------------------
# GlobalCacheEngine plan abort, end to end on the control plane
# --------------------------------------------------------------------------

NUM_CPU = 128
NUM_SSD = 1024


@pytest.fixture(params=INDEX_ACCEL_MODES)
def global_engine(request, tmp_path, monkeypatch):
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "index_accel", request.param)
    model_config = ModelConfig(num_layers=2, num_kv_heads=2, head_size=8,
                               tp_size=1)
    cache_config = CacheConfig(tokens_per_block=TPB,
                               enable_cpu=True,
                               enable_ssd=True,
                               num_cpu_blocks=NUM_CPU,
                               num_ssd_blocks=NUM_SSD,
                               ssd_cache_dir=str(tmp_path / "ssd"))
    return GlobalCacheEngine(cache_config, model_config)


def _run_put(engine, tokens, request_id, complete=True):
    mask = np.ones_like(tokens, dtype=bool)
    slot_mapping = np.arange(tokens.size, dtype=np.int64)
    graph, mask_out, callback, op_callbacks, _end = engine.put(
        request_id, tokens, mask, slot_mapping, dp_client_id=0)
    if complete:
        for op_callback in op_callbacks.values():
            op_callback()
        callback()
    return mask_out, callback, op_callbacks, graph


def _run_get(engine, tokens, request_id):
    mask = np.ones_like(tokens, dtype=bool)
    slot_mapping = np.arange(tokens.size, dtype=np.int64)
    _graph, mask_out, callback, op_callbacks, _end = engine.get(
        request_id, tokens, mask, slot_mapping, dp_client_id=0)
    return mask_out, callback, op_callbacks


def _op_id_of(graph, transfer_type):
    """The full-KV op of this type in the graph, or None."""
    for op in graph._op_map.values():
        if op.transfer_type == transfer_type and not getattr(op, "is_swa", False):
            return op.op_id
    return None


def _seed_ssd_resident_data(engine, num_seqs=6, churn=10):
    """Fill the CPU tier past capacity so early sequences survive only on SSD,
    which is what makes a later GET allocate CPU staging blocks."""
    seqs = [_tokens(TPB // 2, seed=100 + i) for i in range(num_seqs)]
    request_id = 0
    for tokens in seqs:
        _run_put(engine, tokens, request_id)
        request_id += 1
    for i in range(churn):
        _run_put(engine, _tokens(TPB // 2, seed=500 + i), request_id)
        request_id += 1
    return seqs


def test_aborted_get_restores_cpu_pool(global_engine):
    engine = global_engine
    seqs = _seed_ssd_resident_data(engine)
    cpu = engine.cpu_cache_engine

    free_before = cpu.mempool.num_free_blocks
    cached_before = cpu.index.total_cached_blocks()
    aborted = 0
    for i, tokens in enumerate(seqs * 8):
        mask_out, callback, _ops = _run_get(engine, tokens, request_id=1000 + i)
        if mask_out.any():
            callback.abort()   # what _cancel_task does pre-launch
            aborted += 1
    assert aborted > 0, "scenario must produce at least one plan to abort"

    assert cpu.mempool.num_free_blocks == free_before
    # a GET publishes nothing, so an aborted one must leave the tree untouched
    assert cpu.index.total_cached_blocks() == cached_before
    _assert_no_locks(cpu)
    # the pool is usable: a fresh put both allocates and completes
    mask_out, _cb, _ops, _g = _run_put(engine, _tokens(TPB // 2, seed=9000),
                                       request_id=2000)
    assert mask_out.any()


def test_aborted_put_restores_both_tiers(global_engine):
    engine = global_engine
    cpu = engine.cpu_cache_engine
    ssd = engine.ssd_cache_engine
    cpu_free = cpu.mempool.num_free_blocks
    ssd_free = ssd.mempool.num_free_blocks

    tokens = _tokens(TPB // 2, seed=42)
    _mask, callback, _ops, _g = _run_put(engine, tokens, request_id=1,
                                         complete=False)
    assert cpu.mempool.num_free_blocks < cpu_free  # the plan holds blocks
    # Detached staging: nothing has been written, so the prefix is still a miss.
    assert cpu.match(_seq(tokens)).num_matched_blocks == 0
    assert cpu.index.total_cached_blocks() == 0

    callback.abort()

    assert cpu.mempool.num_free_blocks == cpu_free
    assert ssd.mempool.num_free_blocks == ssd_free
    assert cpu.index.total_cached_blocks() == 0
    assert ssd.index.total_cached_blocks() == 0
    _assert_no_locks(cpu)
    _assert_no_locks(ssd)
    # the prefix was rolled back, so the same put succeeds afterwards
    mask_out, _cb, _ops, _g = _run_put(engine, tokens, request_id=2)
    assert mask_out.any()
    assert cpu.match(_seq(tokens)).num_matched_blocks > 0


def test_completion_publishes_exactly_the_written_prefix(global_engine):
    """Completion publication: both tiers become matchable, once."""
    engine = global_engine
    cpu = engine.cpu_cache_engine
    ssd = engine.ssd_cache_engine
    tokens = _tokens(5, seed=43)

    _mask, callback, ops, _g = _run_put(engine, tokens, request_id=1,
                                        complete=True)

    assert cpu.match(_seq(tokens)).num_matched_blocks == 5
    assert ssd.match(_seq(tokens)).num_matched_blocks == 5
    assert cpu.index.total_cached_blocks() == 5
    assert ssd.index.total_cached_blocks() == 5
    assert cpu.mempool.num_used_blocks == 5
    _assert_no_locks(cpu)
    _assert_no_locks(ssd)

    # a duplicate completion must not double-publish or double-release
    for op_callback in ops.values():
        op_callback()
    callback()
    assert cpu.index.total_cached_blocks() == 5
    assert cpu.mempool.num_used_blocks == 5


def test_cpu_is_published_at_d2h_before_the_graph_drains(global_engine):
    """PUT reports success at its D2H, so the CPU copy must be readable there.

    Deferring CPU publication to graph completion would put CPU reuse behind
    SSD write latency.
    """
    engine = global_engine
    cpu = engine.cpu_cache_engine
    ssd = engine.ssd_cache_engine
    tokens = _tokens(4, seed=44)

    _mask, callback, ops, graph = _run_put(engine, tokens, request_id=1,
                                           complete=False)
    d2h_id = _op_id_of(graph, TransferType.D2H)
    h2disk_id = _op_id_of(graph, TransferType.H2DISK)
    assert d2h_id in ops and h2disk_id in ops

    ops[d2h_id]()
    assert cpu.match(_seq(tokens)).num_matched_blocks == 4, \
        "CPU must be readable at D2H completion"
    assert ssd.match(_seq(tokens)).num_matched_blocks == 0, \
        "SSD has not been written yet"

    ops[h2disk_id]()
    assert ssd.match(_seq(tokens)).num_matched_blocks == 4

    callback()
    assert cpu.match(_seq(tokens)).num_matched_blocks == 4
    _assert_no_locks(cpu)
    _assert_no_locks(ssd)


def test_cpu_copy_survives_a_failed_ssd_write(global_engine):
    """An H2DISK failure aborts the plan; the written CPU copy must survive."""
    engine = global_engine
    cpu = engine.cpu_cache_engine
    ssd = engine.ssd_cache_engine
    tokens = _tokens(4, seed=45)

    _mask, callback, ops, graph = _run_put(engine, tokens, request_id=1,
                                           complete=False)
    ops[_op_id_of(graph, TransferType.D2H)]()
    assert cpu.match(_seq(tokens)).num_matched_blocks == 4

    callback.abort()   # H2DISK failed -> _fail_task aborts the whole plan

    assert cpu.match(_seq(tokens)).num_matched_blocks == 4, \
        "a lower-tier failure must not discard a CPU copy that was written"
    assert ssd.match(_seq(tokens)).num_matched_blocks == 0
    assert ssd.mempool.num_used_blocks == 0, "SSD staging must be recycled"
    _assert_no_locks(cpu)
    _assert_no_locks(ssd)


def test_completed_then_cancelled_plan_does_not_double_release(global_engine):
    engine = global_engine
    cpu = engine.cpu_cache_engine
    tokens = _tokens(TPB // 2, seed=77)

    _mask, callback, _ops, _g = _run_put(engine, tokens, request_id=1,
                                         complete=True)
    free_after_complete = cpu.mempool.num_free_blocks

    callback.abort()   # late cancel racing completion: must be a no-op

    assert cpu.mempool.num_free_blocks == free_after_complete
    assert cpu.index.total_cached_blocks() > 0
    _assert_no_locks(cpu)


# --------------------------------------------------------------------------
# KVTaskManager._cancel_task wiring
# --------------------------------------------------------------------------

def _make_manager(engine) -> KVTaskManager:
    manager = KVTaskManager.__new__(KVTaskManager)  # skip transfer subprocess
    manager.cache_engine = engine
    manager.tasks = {}
    manager.graph_to_task = {}
    return manager


def test_cancel_task_aborts_unlaunched_plan(global_engine):
    engine = global_engine
    manager = _make_manager(engine)
    cpu = engine.cpu_cache_engine
    free_before = cpu.mempool.num_free_blocks

    tokens = _tokens(TPB // 2, seed=11)
    manager.create_put_task(task_id=1, token_ids=tokens,
                            slot_mapping=np.arange(tokens.size, dtype=np.int64),
                            dp_client_id=0,
                            token_mask=np.ones_like(tokens, dtype=bool))
    assert cpu.mempool.num_free_blocks < free_before

    manager._cancel_task(1)

    assert 1 not in manager.tasks
    assert cpu.mempool.num_free_blocks == free_before
    assert cpu.match(_seq(tokens)).num_matched_blocks == 0
    assert cpu.index.total_cached_blocks() == 0
    _assert_no_locks(cpu)


def test_cancel_task_leaves_running_tasks_alone(global_engine):
    """A RUNNING task's graph is in flight; its completion callbacks will still
    fire, so cancel must not abort (that would race the real completion)."""
    engine = global_engine
    manager = _make_manager(engine)
    cpu = engine.cpu_cache_engine

    tokens = _tokens(TPB // 2, seed=12)
    manager.create_put_task(task_id=1, token_ids=tokens,
                            slot_mapping=np.arange(tokens.size, dtype=np.int64),
                            dp_client_id=0,
                            token_mask=np.ones_like(tokens, dtype=bool))
    task = manager.tasks[1]
    task.status = TaskStatus.RUNNING
    held = cpu.mempool.num_free_blocks
    callback = task.callback
    op_callbacks = task.op_callback_dict

    manager._cancel_task(1)

    assert cpu.mempool.num_free_blocks == held  # nothing rolled back
    # the in-flight completion still lands normally afterwards
    for op_callback in op_callbacks.values():
        op_callback()
    callback()
    assert cpu.match(_seq(tokens)).num_matched_blocks > 0
    _assert_no_locks(cpu)


# --------------------------------------------------------------------------
# Raced plans: cancel one while an overlapping plan is pending
# --------------------------------------------------------------------------

def test_cancel_with_pending_extension_leaves_no_hole(global_engine):
    """The case the readiness API could not handle, now fully resolved.

    Put B (prefix + extension) arrives while put A (prefix) is pending. Under
    the old model A's ``is_ready=False`` node claimed the prefix, B skipped it,
    and cancelling A backed off (A's node had gained B's child) leaving a
    permanent unready hole. Insert-after mounts nothing at plan time, so B
    writes the whole span itself and A's cancellation is a pure recycle.
    """
    engine = global_engine
    cpu = engine.cpu_cache_engine
    prefix = _tokens(4, seed=61)
    extension = np.concatenate([prefix, _tokens(2, seed=62)])

    mask_a, callback_a, _ops_a, _ga = _run_put(engine, prefix, request_id=1,
                                               complete=False)
    assert mask_a.any()
    mask_b, callback_b, ops_b, _gb = _run_put(engine, extension, request_id=2,
                                              complete=False)
    assert mask_b.all(), "B must plan the whole span, prefix included"

    callback_a.abort()
    for op_callback in ops_b.values():
        op_callback()
    callback_b()

    # B owns the prefix outright: no hole, and A's blocks went back to the pool.
    assert cpu.match(_seq(prefix)).num_matched_blocks == 4
    assert cpu.match(_seq(extension)).num_matched_blocks == 6
    assert cpu.index.total_cached_blocks() == 6
    assert cpu.mempool.num_used_blocks == 6
    _assert_no_locks(cpu)


def test_cancel_get_with_racing_put_leaves_no_residue(global_engine):
    """A staging GET is cancelled while a put of the same prefix is pending.

    After the abort and the put's completion the tree holds exactly the put's
    data, nothing is pinned, and late abort/complete calls on the cancelled
    handle are inert.
    """
    engine = global_engine
    seqs = _seed_ssd_resident_data(engine)
    target = seqs[0]
    cpu = engine.cpu_cache_engine

    mask_get, callback_get, _get_ops = _run_get(engine, target, request_id=900)
    if not mask_get.any():
        pytest.skip("scenario needs an SSD-staging GET plan")
    _mask, callback_put, put_ops, _g = _run_put(engine, target, request_id=901,
                                                complete=False)

    callback_get.abort()
    for op_callback in put_ops.values():
        op_callback()
    callback_put()

    assert cpu.match(_seq(target)).num_matched_blocks > 0
    _assert_no_locks(cpu)
    _assert_no_locks(engine.ssd_cache_engine)
    callback_get.abort()   # late duplicate: inert
    callback_get()         # late complete on aborted handle: inert
