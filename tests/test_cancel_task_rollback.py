"""Cancelling a never-launched task must roll back everything its plan holds.

``create_get_task`` / ``create_put_task`` run the cache engine's planner at
creation time, which locks matched radix nodes, allocates CPU staging blocks
for SSD-resident fragments, and inserts ``is_ready=False`` index nodes that
only a completion callback can publish. The vLLM adapter cancels unlaunched
tasks routinely (a matched request that is not scheduled this step, and every
preemption), so dropping those resources on cancel leaks them permanently:
staging blocks vanish from the mempool, locked/unready nodes can never be
evicted, and unready nodes shadow future puts of the same prefix.

These tests pin the abort path: the plan handle's ``abort()`` (wired into
``KVTaskManager._cancel_task``) must restore the mempool, leave no unready
nodes behind, and be mutually exclusive with the completion callback.
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
from flexkv.common.transfer import DeviceType
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


# --------------------------------------------------------------------------
# Tier-engine level: rollback_unready_insert
# --------------------------------------------------------------------------

@pytest.fixture(params=ENGINE_CLASSES)
def tier_engine(request):
    return request.param(device_type=DeviceType.CPU,
                         num_total_blocks=64,
                         tokens_per_block=TPB,
                         evict_ratio=0.05)


def test_rollback_removes_unready_leaf_and_recycles(tier_engine):
    free_before = tier_engine.mempool.num_free_blocks
    blocks = tier_engine.take(4)
    node = tier_engine.insert(_seq(_tokens(4, seed=1)), blocks, is_ready=False)
    assert tier_engine.mempool.num_free_blocks == free_before - 4

    freed = tier_engine.rollback_unready_insert(node)

    assert freed == 4
    assert tier_engine.mempool.num_free_blocks == free_before
    assert tier_engine.index.total_unready_blocks() == 0
    # the prefix is gone: a fresh identical insert allocates fresh blocks
    blocks2 = tier_engine.take(4)
    node2 = tier_engine.insert(_seq(_tokens(4, seed=1)), blocks2, is_ready=True)
    assert node2 is not None


def test_rollback_is_noop_on_ready_node(tier_engine):
    blocks = tier_engine.take(4)
    node = tier_engine.insert(_seq(_tokens(4, seed=2)), blocks, is_ready=True)
    free_before = tier_engine.mempool.num_free_blocks

    freed = tier_engine.rollback_unready_insert(node)

    assert freed == 0
    assert tier_engine.mempool.num_free_blocks == free_before
    assert tier_engine.index.total_ready_blocks() == 4


def test_rollback_is_noop_while_node_is_locked(tier_engine):
    blocks = tier_engine.take(4)
    node = tier_engine.insert(_seq(_tokens(4, seed=3)), blocks, is_ready=False)
    tier_engine.lock_node(node)

    assert tier_engine.rollback_unready_insert(node) == 0
    assert tier_engine.index.total_unready_blocks() == 4

    tier_engine.unlock(node)
    assert tier_engine.rollback_unready_insert(node) == 4
    assert tier_engine.index.total_unready_blocks() == 0


def test_rollback_is_noop_when_node_gained_children(tier_engine):
    """The raced case: another request inserted below the unready node in the
    create-to-cancel window. The node is no longer a leaf, so rollback must
    leave it alone rather than detach blocks a descendant depends on."""
    toks = _tokens(4, seed=4)
    blocks = tier_engine.take(4)
    node = tier_engine.insert(_seq(toks), blocks, is_ready=False)

    longer = np.concatenate([toks, _tokens(2, seed=5)])
    blocks2 = tier_engine.take(2)
    child = tier_engine.insert(_seq(longer), blocks2, is_ready=True)
    assert child is not None

    free_before = tier_engine.mempool.num_free_blocks
    assert tier_engine.rollback_unready_insert(node) == 0
    assert tier_engine.mempool.num_free_blocks == free_before


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
    _graph, mask_out, callback, op_callbacks, _end = engine.put(
        request_id, tokens, mask, slot_mapping, dp_client_id=0)
    if complete:
        for op_callback in op_callbacks.values():
            op_callback()
        callback()
    return mask_out, callback, op_callbacks


def _run_get(engine, tokens, request_id):
    mask = np.ones_like(tokens, dtype=bool)
    slot_mapping = np.arange(tokens.size, dtype=np.int64)
    _graph, mask_out, callback, op_callbacks, _end = engine.get(
        request_id, tokens, mask, slot_mapping, dp_client_id=0)
    return mask_out, callback, op_callbacks


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
    unready_before = cpu.index.total_unready_blocks()
    aborted = 0
    for i, tokens in enumerate(seqs * 8):
        mask_out, callback, _ops = _run_get(engine, tokens, request_id=1000 + i)
        if mask_out.any():
            callback.abort()   # what _cancel_task now does pre-launch
            aborted += 1
    assert aborted > 0, "scenario must produce at least one plan to abort"

    assert cpu.mempool.num_free_blocks == free_before
    assert cpu.index.total_unready_blocks() == unready_before
    # the pool is usable: a fresh put both allocates and completes
    mask_out, _cb, _ops = _run_put(engine, _tokens(TPB // 2, seed=9000),
                                   request_id=2000)
    assert mask_out.any()


def test_aborted_put_restores_both_tiers(global_engine):
    engine = global_engine
    cpu = engine.cpu_cache_engine
    ssd = engine.ssd_cache_engine
    cpu_free = cpu.mempool.num_free_blocks
    ssd_free = ssd.mempool.num_free_blocks

    tokens = _tokens(TPB // 2, seed=42)
    _mask, callback, _ops = _run_put(engine, tokens, request_id=1, complete=False)
    assert cpu.mempool.num_free_blocks < cpu_free  # plan holds blocks

    callback.abort()

    assert cpu.mempool.num_free_blocks == cpu_free
    assert ssd.mempool.num_free_blocks == ssd_free
    assert cpu.index.total_unready_blocks() == 0
    assert ssd.index.total_unready_blocks() == 0
    # the prefix was rolled back, so the same put succeeds afterwards
    mask_out, _cb, _ops = _run_put(engine, tokens, request_id=2)
    assert mask_out.any()


def test_completed_then_cancelled_plan_does_not_double_release(global_engine):
    engine = global_engine
    cpu = engine.cpu_cache_engine
    tokens = _tokens(TPB // 2, seed=77)

    _mask, callback, _ops = _run_put(engine, tokens, request_id=1, complete=True)
    free_after_complete = cpu.mempool.num_free_blocks

    callback.abort()   # late cancel racing completion: must be a no-op

    assert cpu.mempool.num_free_blocks == free_after_complete
    assert cpu.index.total_ready_blocks() > 0


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
    assert cpu.index.total_unready_blocks() == 0


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

    manager._cancel_task(1)

    assert cpu.mempool.num_free_blocks == held  # nothing rolled back
    # the in-flight completion still lands normally afterwards
    for op_callback in task.op_callback_dict.values():
        op_callback()
    callback()
    assert cpu.index.total_ready_blocks() > 0


# --------------------------------------------------------------------------
# Raced plans: cancel one while an overlapping plan is pending
# --------------------------------------------------------------------------

def test_cancel_with_pending_extension_leaves_bounded_hole(global_engine):
    """The documented residual case, pinned so it is not overstated.

    Put B (prefix + extension) arrives while put A (prefix) is pending; B
    skips A's blocks because A's unready insert claims them, so A's content
    is never written by anyone. Cancelling A then backs off the rollback
    (A's node gained B's child and is no longer a leaf) and the prefix stays
    an unready hole -- semantically forced, since marking it ready would
    publish blocks whose transfer never ran. What the fix guarantees here is
    strictly-no-worse than before: the hole is the same size as upstream's,
    bounded by A's node, and unlike upstream it is no longer locked.
    """
    engine = global_engine
    cpu = engine.cpu_cache_engine
    prefix = _tokens(4, seed=61)
    extension = np.concatenate([prefix, _tokens(2, seed=62)])

    mask_a, callback_a, _ops_a = _run_put(engine, prefix, request_id=1,
                                          complete=False)
    assert mask_a.any()
    _mask_b, callback_b, ops_b = _run_put(engine, extension, request_id=2,
                                          complete=False)

    callback_a.abort()
    # B saw A's pending insert and skipped the prefix; complete it normally.
    for op_callback in ops_b.values():
        op_callback()
    callback_b()

    # A's node was extended by B, so the rollback must have backed off: the
    # hole is exactly A's insert (4 blocks per tier), bounded and no larger.
    # Upstream leaves the same 4-block hole PLUS a leaked lock on it; here
    # nothing may stay locked once both plans are resolved.
    assert cpu.index.total_unready_blocks() == 4

    if hasattr(cpu.index, "root_node"):  # python index exposes the tree
        stack = [cpu.index.root_node]
        while stack:
            node = stack.pop()
            assert node.lock_cnt == 0, "no plan may leave a lock behind"
            stack.extend(node.children.values())


def test_cancel_get_with_racing_put_leaves_no_residue(global_engine):
    """A staging GET is cancelled while a put of the same prefix is pending;
    the put saw the unready staging node and skipped, so after the abort and
    the put's completion nothing unready may remain and late abort/complete
    calls on the cancelled handle must be inert."""
    engine = global_engine
    seqs = _seed_ssd_resident_data(engine)
    target = seqs[0]

    mask_get, callback_get, _get_ops = _run_get(engine, target, request_id=900)
    if not mask_get.any():
        pytest.skip("scenario needs an SSD-staging GET plan")
    _mask, callback_put, put_ops = _run_put(engine, target, request_id=901,
                                            complete=False)

    callback_get.abort()
    for op_callback in put_ops.values():
        op_callback()
    callback_put()

    assert engine.cpu_cache_engine.index.total_unready_blocks() == 0
    assert engine.ssd_cache_engine.index.total_unready_blocks() == 0
    callback_get.abort()   # late duplicate: inert
    callback_get()         # late complete on aborted handle: inert
