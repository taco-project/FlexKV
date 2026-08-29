"""A failed transfer op must fail its graph, roll the plan back, and surface
FAILED -- not leak everything behind a TIMEOUT.

Before this change a worker that could not execute an op logged the error and
reported nothing (worker.py: "only put the op when transfer success"). The op
stayed RUNNING forever, its graph never completed, the task's wait() returned
a misleading TIMEOUT after 20s with no cleanup, and every resource the plan
held -- locked radix nodes, unready index nodes, CPU staging blocks -- leaked
permanently. One SSD read error was enough to poison the tree for good.

Now the worker reports (op_id, False); the engine loop stops dispatching the
graph, drains its in-flight ops (so late sibling completions still run their
callbacks against mounted nodes), then emits CompletedOp.failed_graph; the
task layer runs the plan's abort (TransferPlanHandle) and terminates the task
as FAILED.
"""
import numpy as np
import pytest

from flexkv import c_ext
from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.request import KVResponseStatus
from flexkv.common.transfer import (
    CompletedOp,
    TransferOp,
    TransferOpGraph,
    TransferOpStatus,
    TransferType,
)
from flexkv.kvtask import KVTaskManager, TaskStatus, convert_to_response_status
from flexkv.transfer.scheduler import TransferScheduler

try:
    from flexkv.transfer.transfer_engine import TransferEngine
    _TE_METHODS_AVAILABLE = all(
        hasattr(TransferEngine, m) for m in (
            '_handle_failed_op', '_discard_failed_op',
            '_finalize_or_discard', '_emit_drained_graph_failures',
            '_op_buffer_registered_here',
        )
    )
except Exception:
    TransferEngine = None
    _TE_METHODS_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _TE_METHODS_AVAILABLE,
                       reason="TransferEngine not fully loaded (c_ext or dependencies missing)"),
]

TPB = 16

_HAS_REAL_C_EXT = getattr(c_ext, "__file__", None) is not None

INDEX_ACCEL_MODES = [
    pytest.param(False, id="python-index"),
    pytest.param(True, id="accel-index",
                 marks=pytest.mark.skipif(not _HAS_REAL_C_EXT,
                                          reason="requires compiled c_ext")),
]


def _tokens(num_blocks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 50000, num_blocks * TPB, dtype=np.int64)


# --------------------------------------------------------------------------
# CompletedOp message shape
# --------------------------------------------------------------------------

def test_failed_graph_message_is_distinct_and_pickle_compatible():
    import pickle

    done = CompletedOp.completed_graph(7)
    failed = CompletedOp.failed_graph(7)

    assert done.is_graph_completed() and not done.is_graph_failed()
    assert failed.is_graph_failed() and not failed.is_graph_completed()

    roundtripped = pickle.loads(pickle.dumps(failed))
    assert roundtripped.is_graph_failed()
    # pre-change producers that never set `failed` keep their meaning
    legacy = CompletedOp(graph_id=1, op_id=-1)
    assert legacy.is_graph_completed() and not legacy.is_graph_failed()


# --------------------------------------------------------------------------
# Scheduler: fail_graph stops dispatch
# --------------------------------------------------------------------------

def _make_chain_graph(num_ops: int):
    graph = TransferOpGraph.create_empty_graph()
    ops = []
    for i in range(num_ops):
        op = TransferOp(graph_id=graph.graph_id,
                        transfer_type=TransferType.H2D if i % 2 else TransferType.D2H,
                        src_block_ids=np.arange(2, dtype=np.int64),
                        dst_block_ids=np.arange(2, dtype=np.int64))
        graph.add_transfer_op(op)
        ops.append(op)
    for i in range(1, num_ops):
        graph.add_dependency(ops[i].op_id, ops[i - 1].op_id)
    return graph, ops


def test_fail_graph_stops_dispatch_and_never_completes():
    scheduler = TransferScheduler()
    graph, ops = _make_chain_graph(3)
    scheduler.add_transfer_graph(graph)

    _done, dispatched = scheduler.schedule([])
    assert [o.op_id for o in dispatched] == [ops[0].op_id]

    scheduler.fail_graph(graph.graph_id)
    scheduler.fail_graph(graph.graph_id)  # idempotent

    # The in-flight op drains; its completion must neither dispatch the
    # successor nor report the graph completed.
    done, dispatched = scheduler.schedule([ops[0]])
    assert done == []
    assert dispatched == []
    assert ops[1].status == TransferOpStatus.PENDING


def test_fail_graph_leaves_other_graphs_alone():
    scheduler = TransferScheduler()
    doomed, doomed_ops = _make_chain_graph(2)
    healthy, healthy_ops = _make_chain_graph(2)
    scheduler.add_transfer_graph(doomed)
    scheduler.add_transfer_graph(healthy)
    scheduler.schedule([])

    scheduler.fail_graph(doomed.graph_id)

    done, dispatched = scheduler.schedule([healthy_ops[0]])
    assert [o.op_id for o in dispatched] == [healthy_ops[1].op_id]
    done, dispatched = scheduler.schedule([healthy_ops[1]])
    assert done == [healthy.graph_id]


# --------------------------------------------------------------------------
# Engine loop pieces, driven directly (the loop body is exercised through
# _handle_failed_op / _emit_drained_graph_failures without starting processes)
# --------------------------------------------------------------------------

class _EngineHarness:
    """The failure-path state of TransferEngine, minus workers and queues."""

    def __init__(self):
        from queue import Queue

        self.op_id_to_op = {}
        self.op_id_to_nvtx_range = {}
        self._child_id_to_child = {}
        self._child_to_parent_op_id = {}
        self._failed_graph_ids = set()
        self._failed_parent_op_ids = set()
        self._worker_map = {}
        self._swa_worker_map = {}
        self.pin_buffer = None
        self.completed_queue = Queue()
        self.scheduler = TransferScheduler()

    _handle_failed_op = staticmethod(getattr(TransferEngine, '_handle_failed_op', None))
    _discard_failed_op = staticmethod(getattr(TransferEngine, '_discard_failed_op', None))
    _finalize_or_discard = staticmethod(getattr(TransferEngine, '_finalize_or_discard', None))
    _emit_drained_graph_failures = staticmethod(getattr(TransferEngine, '_emit_drained_graph_failures', None))
    _op_buffer_registered_here = getattr(TransferEngine, '_op_buffer_registered_here', None)


def test_failed_op_fails_graph_and_emits_after_drain():
    engine = _EngineHarness()
    graph, ops = _make_chain_graph(3)
    engine.scheduler.add_transfer_graph(graph)
    engine.scheduler.schedule([])

    # two ops of the same graph are in flight; one fails
    for op in ops[:2]:
        op.pending_count = 1
        engine.op_id_to_op[op.op_id] = op

    engine._handle_failed_op(ops[0].op_id)

    assert graph.graph_id in engine._failed_graph_ids
    assert ops[0].op_id not in engine.op_id_to_op
    # the sibling has not drained yet: no failure message may be emitted
    engine._emit_drained_graph_failures()
    assert engine.completed_queue.empty()

    # sibling drains (as the loop's success path would on completion)
    del engine.op_id_to_op[ops[1].op_id]
    engine._emit_drained_graph_failures()
    message = engine.completed_queue.get_nowait()
    assert message.is_graph_failed()
    assert message.graph_id == graph.graph_id
    assert not engine._failed_graph_ids


def test_failed_replica_discards_parent_when_replicas_drain():
    engine = _EngineHarness()
    graph, ops = _make_chain_graph(1)
    parent = ops[0]
    parent.pending_count = 2  # two PP replicas
    engine.op_id_to_op[parent.op_id] = parent
    engine.scheduler.add_transfer_graph(graph)

    child_a = TransferOp(graph_id=graph.graph_id,
                         transfer_type=parent.transfer_type,
                         src_block_ids=np.arange(2, dtype=np.int64),
                         dst_block_ids=np.arange(2, dtype=np.int64))
    engine._child_to_parent_op_id[child_a.op_id] = parent.op_id
    engine._child_id_to_child[child_a.op_id] = child_a

    engine._handle_failed_op(child_a.op_id)

    # parent survives until its remaining replica drains, and is marked
    assert parent.op_id in engine.op_id_to_op
    assert parent.op_id in engine._failed_parent_op_ids
    assert graph.graph_id in engine._failed_graph_ids

    # remaining replica completes: the loop's routing helper must DISCARD the
    # parent, not finalize it (finalize would emit a completion message).
    parent.pending_count -= 1
    assert parent.pending_count == 0
    engine._finalize_or_discard(parent, finished_ops=[])
    assert parent.op_id not in engine.op_id_to_op
    assert parent.op_id not in engine._failed_parent_op_ids
    assert engine.completed_queue.empty(), \
        "a discarded op must not produce a completion message"

    engine._emit_drained_graph_failures()
    assert engine.completed_queue.get_nowait().is_graph_failed()


def test_failed_op_unknown_id_is_harmless():
    engine = _EngineHarness()
    engine._handle_failed_op(987654)
    assert not engine._failed_graph_ids
    engine._emit_drained_graph_failures()
    assert engine.completed_queue.empty()


# --------------------------------------------------------------------------
# Task layer: FAILED terminal state with full rollback
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


def _make_manager(engine) -> KVTaskManager:
    manager = KVTaskManager.__new__(KVTaskManager)  # no transfer subprocess
    manager.cache_engine = engine
    manager.tasks = {}
    manager.graph_to_task = {}
    return manager


def test_failed_graph_rolls_back_task_and_reports_failed(global_engine):
    engine = global_engine
    manager = _make_manager(engine)
    cpu = engine.cpu_cache_engine
    free_before = cpu.mempool.num_free_blocks

    tokens = _tokens(TPB // 2, seed=21)
    manager.create_put_task(task_id=1, token_ids=tokens,
                            slot_mapping=np.arange(tokens.size, dtype=np.int64),
                            dp_client_id=0,
                            token_mask=np.ones_like(tokens, dtype=bool))
    task = manager.tasks[1]
    task.status = TaskStatus.RUNNING  # graph launched
    assert cpu.mempool.num_free_blocks < free_before

    manager._fail_task(1)

    assert task.status == TaskStatus.FAILED
    assert task.is_completed()
    assert convert_to_response_status(task.status) == KVResponseStatus.FAILED
    # every plan resource returned: pool restored, no unready or locked residue
    assert cpu.mempool.num_free_blocks == free_before
    assert cpu.index.total_unready_blocks() == 0
    # the same prefix is usable again afterwards
    mask = np.ones_like(tokens, dtype=bool)
    slot_mapping = np.arange(tokens.size, dtype=np.int64)
    _g, mask_out, callback, ops, _e = engine.put(
        2, tokens, mask, slot_mapping, dp_client_id=0)
    assert mask_out.any()
    for op_callback in ops.values():
        op_callback()
    callback()


def test_failed_task_with_partial_completion_keeps_completed_tier(global_engine):
    """Drain-then-abort semantics: op callbacks that already ran (their
    transfers completed) published valid data; the abort keeps those nodes
    and removes only the never-transferred ones."""
    engine = global_engine
    manager = _make_manager(engine)
    cpu = engine.cpu_cache_engine
    ssd = engine.ssd_cache_engine

    tokens = _tokens(TPB // 2, seed=22)
    manager.create_put_task(task_id=1, token_ids=tokens,
                            slot_mapping=np.arange(tokens.size, dtype=np.int64),
                            dp_client_id=0,
                            token_mask=np.ones_like(tokens, dtype=bool))
    task = manager.tasks[1]
    task.status = TaskStatus.RUNNING

    # the D2H op completed (its callback ran); the H2DISK op then failed
    d2h_callbacks = [cb for op_id, cb in task.op_callback_dict.items()]
    assert d2h_callbacks
    d2h_callbacks[0]()  # CPU tier published

    manager._fail_task(1)

    assert cpu.index.total_ready_blocks() > 0, \
        "tier whose transfer completed must keep its data"
    assert cpu.index.total_unready_blocks() == 0
    assert ssd.index.total_unready_blocks() == 0, \
        "tier whose transfer never ran must be rolled back"


def test_update_tasks_routes_failed_message_to_fail_task(global_engine):
    engine = global_engine
    manager = _make_manager(engine)

    tokens = _tokens(TPB // 2, seed=23)
    manager.create_put_task(task_id=1, token_ids=tokens,
                            slot_mapping=np.arange(tokens.size, dtype=np.int64),
                            dp_client_id=0,
                            token_mask=np.ones_like(tokens, dtype=bool))
    task = manager.tasks[1]
    task.status = TaskStatus.RUNNING
    graph_id = task.graph.graph_id

    manager._get_completed_ops = \
        lambda timeout=None: [CompletedOp.failed_graph(graph_id)]
    manager._update_tasks()

    assert task.status == TaskStatus.FAILED


def test_fail_task_is_idempotent_and_mutex_with_completion(global_engine):
    engine = global_engine
    manager = _make_manager(engine)
    cpu = engine.cpu_cache_engine

    tokens = _tokens(TPB // 2, seed=24)
    manager.create_put_task(task_id=1, token_ids=tokens,
                            slot_mapping=np.arange(tokens.size, dtype=np.int64),
                            dp_client_id=0,
                            token_mask=np.ones_like(tokens, dtype=bool))
    task = manager.tasks[1]
    task.status = TaskStatus.RUNNING
    callback = task.callback

    manager._fail_task(1)
    free_after_fail = cpu.mempool.num_free_blocks
    manager._fail_task(1)          # duplicate failure message: no-op
    callback()                     # late completion racing the failure: no-op

    assert cpu.mempool.num_free_blocks == free_after_fail
    assert task.status == TaskStatus.FAILED


# --------------------------------------------------------------------------
# Multi-handle terminal aggregation (nnodes > 1 / trtllm-remote deployments)
# --------------------------------------------------------------------------

class _FakeHandle:
    def __init__(self):
        self.pending = []

    def wait(self, timeout=None):
        out, self.pending = self.pending, []
        return out

    def shutdown(self):
        # KVTaskManager.__del__ shuts handles down; without this the GC-time
        # AttributeError surfaces as a PytestUnraisableExceptionWarning.
        pass


def _make_multi_handle_manager(num_handles):
    manager = KVTaskManager.__new__(KVTaskManager)
    manager.transfer_handles = [_FakeHandle() for _ in range(num_handles)]
    manager.required_completed_count = num_handles
    manager.uncompleted_ops = {}
    manager.uncompleted_graphs = {}
    return manager


def test_multi_handle_failure_waits_for_every_handle():
    """The same graph is submitted to every handle; terminating on the first
    handle's failure would recycle plan blocks a sibling engine is still
    writing into. The graph may only terminate once ALL handles report a
    terminal state, and any failure among them fails the graph."""
    manager = _make_multi_handle_manager(2)
    handle_a, handle_b = manager.transfer_handles

    handle_a.pending = [CompletedOp.failed_graph(5)]
    assert manager._get_completed_ops() == []          # b still in flight
    assert manager.uncompleted_graphs == {5: (1, True)}

    handle_b.pending = [CompletedOp.completed_graph(5)]
    results = manager._get_completed_ops()
    assert len(results) == 1
    assert results[0].is_graph_failed()                # any failure fails it
    assert manager.uncompleted_graphs == {}


def test_multi_handle_failure_purges_stale_op_counters():
    """A failed handle never finalizes some of the graph's ops, so their
    N-way per-op counters could never complete; graph termination must purge
    them instead of leaking them forever."""
    manager = _make_multi_handle_manager(2)
    handle_a, handle_b = manager.transfer_handles

    # op 7 of graph 5 finalized on handle_a only; handle_a then fails.
    handle_a.pending = [CompletedOp(graph_id=5, op_id=7),
                        CompletedOp.failed_graph(5)]
    assert manager._get_completed_ops() == []
    assert manager.uncompleted_ops == {(5, 7): 1}

    handle_b.pending = [CompletedOp.failed_graph(5)]
    results = manager._get_completed_ops()
    assert len(results) == 1 and results[0].is_graph_failed()
    assert manager.uncompleted_ops == {}, "stale per-op counters must be purged"
    # an unrelated graph's op counter must survive the purge
    handle_a.pending = [CompletedOp(graph_id=6, op_id=9)]
    manager._get_completed_ops()
    assert manager.uncompleted_ops == {(6, 9): 1}


def test_single_handle_failure_terminates_immediately():
    """required_completed_count == 1 (the default deployment): a failure
    message terminates the graph in the same call, exactly as before."""
    manager = _make_multi_handle_manager(1)
    manager.transfer_handles[0].pending = [CompletedOp.failed_graph(3)]
    results = manager._get_completed_ops()
    assert len(results) == 1 and results[0].is_graph_failed()
    assert manager.uncompleted_graphs == {}
