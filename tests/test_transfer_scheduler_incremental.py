"""TransferScheduler dispatches from a dirty set instead of sweeping every graph.

The scheduler used to visit every in-flight graph on every call. Dependencies
never cross graphs, so a graph can only expose new ready ops after one of its
own ops completes -- visiting only the changed graphs is equivalent.

These tests pin that equivalence down: a reference implementation of the old
full-sweep behavior is driven in lockstep with the real scheduler over random
DAGs, and both must dispatch the same ops and complete the same graphs on every
tick. Separate tests cover where the new scheduler deliberately differs
(non-terminal virtual ops), where it must not regress (losing a dirty bit on
error), and the limit of that guarantee (ops an exception loses regardless).
"""
import random

import numpy as np
import pytest

from flexkv.common.transfer import (
    TransferOp,
    TransferOpGraph,
    TransferOpStatus,
    TransferType,
)
from flexkv.transfer.scheduler import TransferScheduler

pytestmark = pytest.mark.unit


class _FullSweepScheduler:
    """Reference: the pre-change scheduler, sweeping all graphs every call.

    Verbatim behavior of TransferScheduler as of the parent of this commit.
    `benchmarks/microbenchmark_transfer_scheduler.py` keeps its own copy for the
    --baseline table; keep the two in step if either is ever touched.

    Note this is only an oracle for *which graphs get visited* -- it calls the
    same TransferOpGraph methods the real scheduler does.
    """

    def __init__(self):
        self._transfer_graphs = {}

    def add_transfer_graph(self, graph):
        self._transfer_graphs[graph.graph_id] = graph

    def schedule(self, finished_ops):
        for op in finished_ops:
            if op.graph_id in self._transfer_graphs:
                self._transfer_graphs[op.graph_id].mark_completed(op.op_id)

        next_ops = []
        for graph in self._transfer_graphs.values():
            for op_id in graph.take_ready_ops():
                op = graph._op_map[op_id]
                if op.transfer_type == TransferType.VIRTUAL:
                    self._transfer_graphs[op.graph_id].mark_completed(op_id)
                next_ops.append(op)

        completed_graph_ids = [
            graph_id for graph_id, graph in self._transfer_graphs.items()
            if graph.all_transfer_ops_completed()
        ]
        for graph_id in completed_graph_ids:
            self._transfer_graphs.pop(graph_id)
        return completed_graph_ids, next_ops


def _make_op(graph, transfer_type=TransferType.H2D, n=2):
    return TransferOp(
        graph_id=graph.graph_id,
        transfer_type=transfer_type,
        src_block_ids=np.arange(n, dtype=np.int64),
        dst_block_ids=np.arange(n, dtype=np.int64),
    )


def _build_from_spec(spec):
    """spec = (list of TransferType, list of (successor_idx, predecessor_idx))"""
    types, edges = spec
    graph = TransferOpGraph.create_empty_graph()
    ops = []
    for t in types:
        n = 0 if t == TransferType.VIRTUAL else 2
        op = _make_op(graph, t, n)
        graph.add_transfer_op(op)
        ops.append(op)
    for succ, pred in edges:
        graph.add_dependency(ops[succ].op_id, ops[pred].op_id)
    return graph, ops


def _random_spec(rng, n_ops):
    """Random DAG. A virtual op is only ever placed LAST, with edges running
    from lower to higher index, so it is always a terminal sink -- which is
    what every virtual op in FlexKV actually is today. A non-terminal virtual
    op is deliberately NOT equivalent between the two schedulers; see
    test_non_terminal_virtual_op_dispatches_successor_same_tick.
    """
    types = [rng.choice([TransferType.H2D, TransferType.D2H]) for _ in range(n_ops)]
    if rng.random() < 0.25:
        types[-1] = TransferType.VIRTUAL
    edges = [(i, j) for i in range(n_ops) for j in range(i) if rng.random() < 0.35]
    return types, edges


# --------------------------------------------------------------------------
# Basic behavior
# --------------------------------------------------------------------------

def test_linear_chain_advances_one_op_per_tick():
    sched = TransferScheduler()
    graph = TransferOpGraph.create_empty_graph()
    ops = [_make_op(graph) for _ in range(4)]
    for op in ops:
        graph.add_transfer_op(op)
    for i in range(1, 4):
        graph.add_dependency(ops[i].op_id, ops[i - 1].op_id)
    sched.add_transfer_graph(graph)

    completed, dispatched = sched.schedule([])
    assert [o.op_id for o in dispatched] == [ops[0].op_id]
    assert completed == []

    for i in range(3):
        completed, dispatched = sched.schedule([ops[i]])
        assert [o.op_id for o in dispatched] == [ops[i + 1].op_id]

    completed, dispatched = sched.schedule([ops[3]])
    assert completed == [graph.graph_id]
    assert dispatched == []


def test_graph_with_no_ops_completes_immediately():
    sched = TransferScheduler()
    graph = TransferOpGraph.create_empty_graph()
    sched.add_transfer_graph(graph)
    completed, dispatched = sched.schedule([])
    assert completed == [graph.graph_id]
    assert dispatched == []


def test_terminal_virtual_op_completes_graph_in_same_tick():
    """A virtual task-end op is dispatched and self-completes, so the graph
    finishes without waiting for another external event."""
    sched = TransferScheduler()
    graph = TransferOpGraph.create_empty_graph()
    real = _make_op(graph)
    virt = _make_op(graph, TransferType.VIRTUAL, n=0)
    graph.add_transfer_op(real)
    graph.add_transfer_op(virt)
    graph.add_dependency(virt.op_id, real.op_id)
    sched.add_transfer_graph(graph)

    _, dispatched = sched.schedule([])
    assert [o.op_id for o in dispatched] == [real.op_id]

    completed, dispatched = sched.schedule([real])
    assert [o.op_id for o in dispatched] == [virt.op_id]
    assert completed == [graph.graph_id]


def test_idle_graphs_are_not_revisited():
    """A tick that advances one graph must not re-scan the others."""
    sched = TransferScheduler()
    graphs = []
    for _ in range(5):
        graph = TransferOpGraph.create_empty_graph()
        a, b = _make_op(graph), _make_op(graph)
        graph.add_transfer_op(a)
        graph.add_transfer_op(b)
        graph.add_dependency(b.op_id, a.op_id)
        sched.add_transfer_graph(graph)
        graphs.append((graph, a, b))
    sched.schedule([])

    target_graph, target_a, _ = graphs[0]
    calls = {g.graph_id: 0 for g, _, _ in graphs}
    try:
        for g, _, _ in graphs:
            def counting(_g=g, _orig=g.take_ready_ops):
                calls[_g.graph_id] += 1
                return _orig()

            g.take_ready_ops = counting
        sched.schedule([target_a])
    finally:
        for g, _, _ in graphs:
            # drop the instance attribute, restoring the class method
            g.__dict__.pop("take_ready_ops", None)

    assert calls[target_graph.graph_id] == 1
    for g, _, _ in graphs[1:]:
        assert calls[g.graph_id] == 0, "untouched graph should not be swept"


# --------------------------------------------------------------------------
# Where the new scheduler must differ, and where it must not regress
# --------------------------------------------------------------------------

def test_non_terminal_virtual_op_dispatches_successor_same_tick():
    """Deliberate divergence from the old full-sweep scheduler.

    With `real -> virtual -> real`, completing the virtual op unblocks its
    successor. The old code deferred that successor to the NEXT schedule()
    call, which the engine only makes on a new external event -- so a graph
    whose only remaining work sat behind a virtual op could stall. Every
    virtual op FlexKV builds today is a terminal sink, so this is latent
    rather than live, but the dirty-set drain fixes it for free.
    """
    sched = TransferScheduler()
    graph = TransferOpGraph.create_empty_graph()
    head = _make_op(graph)
    virt = _make_op(graph, TransferType.VIRTUAL, n=0)
    tail = _make_op(graph)
    for op in (head, virt, tail):
        graph.add_transfer_op(op)
    graph.add_dependency(virt.op_id, head.op_id)
    graph.add_dependency(tail.op_id, virt.op_id)
    sched.add_transfer_graph(graph)

    sched.schedule([])  # dispatches head
    _, dispatched = sched.schedule([head])

    assert [o.op_id for o in dispatched] == [virt.op_id, tail.op_id], \
        "successor of a virtual op must be dispatched in the same tick"


def _drain_with_exploding_graph(position, num_graphs=3):
    """Build `num_graphs` single-op graphs, make the one at `position` raise
    during the drain, then retry. Returns how many ops the retry dispatches."""
    sched = TransferScheduler()
    graphs = []
    for _ in range(num_graphs):
        graph = TransferOpGraph.create_empty_graph()
        graph.add_transfer_op(_make_op(graph))
        sched.add_transfer_graph(graph)
        graphs.append(graph)

    boom = graphs[position]

    def exploding():
        raise RuntimeError("boom")

    try:
        boom.take_ready_ops = exploding
        with pytest.raises(RuntimeError):
            sched.schedule([])
    finally:
        boom.__dict__.pop("take_ready_ops", None)
    _, dispatched = sched.schedule([])
    return len(dispatched)


def test_dirty_bit_survives_an_exception_mid_drain():
    """The engine's scheduler loop logs and continues on exception. Dropping a
    graph's dirty bit before it is processed would mean it is never revisited,
    so its request hangs. Dropping the bit only afterwards keeps the graph
    reachable: with the raise at the head of the drain, nothing has been
    dispatched yet and the retry recovers every graph."""
    assert _drain_with_exploding_graph(position=0) == 3


def test_exception_mid_drain_still_loses_already_collected_ops():
    """Pins the limit of the above, so the guarantee is not overstated.

    A raise partway through the drain discards the `next_ops` collected so far,
    and take_ready_ops() will not hand back an op it already flipped to RUNNING.
    Those ops are lost even though the dirty bit is intact. This is not a
    regression -- the full-sweep scheduler loses them identically -- but the
    dirty bit is not what saves you here.
    """
    assert _drain_with_exploding_graph(position=1) == 2


# --------------------------------------------------------------------------
# Equivalence with the old full-sweep scheduler
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(60))
def test_matches_full_sweep_scheduler_on_random_dags(seed):
    rng = random.Random(seed)
    specs = [_random_spec(rng, rng.randint(1, 8))
             for _ in range(rng.randint(1, 6))]
    # The engine adds graphs from the same loop that calls schedule(), so some
    # arrive while others are already draining. Hold a suffix back and inject it
    # mid-run, which makes add_transfer_graph's dirty-marking accountable to the
    # oracle rather than only to the liveness assert at the end.
    deferred_from = rng.randint(1, len(specs))

    ref_sched, new_sched = _FullSweepScheduler(), TransferScheduler()
    ref_graphs, new_graphs = [], []
    ref_idx, new_idx = {}, {}
    pending = []
    for pos, spec in enumerate(specs):
        rg, r_ops = _build_from_spec(spec)
        ng, n_ops = _build_from_spec(spec)
        ref_graphs.append((rg, r_ops))
        new_graphs.append((ng, n_ops))
        for k, op in enumerate(r_ops):
            ref_idx[op.op_id] = (pos, k)
        for k, op in enumerate(n_ops):
            new_idx[op.op_id] = (pos, k)
        if pos < deferred_from:
            ref_sched.add_transfer_graph(rg)
            new_sched.add_transfer_graph(ng)
        else:
            pending.append((rg, ng))

    ref_pos = {g.graph_id: i for i, (g, _) in enumerate(ref_graphs)}
    new_pos = {g.graph_id: i for i, (g, _) in enumerate(new_graphs)}

    ref_running, new_running = [], []
    ref_finished, new_finished = [], []

    for tick in range(200):
        # Inject one held-back graph per tick, into both schedulers alike.
        if pending and tick > 0:
            rg, ng = pending.pop(0)
            ref_sched.add_transfer_graph(rg)
            new_sched.add_transfer_graph(ng)

        ref_done, ref_next = ref_sched.schedule(ref_finished)
        new_done, new_next = new_sched.schedule(new_finished)

        assert sorted(ref_pos[g] for g in ref_done) == \
               sorted(new_pos[g] for g in new_done)
        assert sorted(ref_idx[o.op_id] for o in ref_next) == \
               sorted(new_idx[o.op_id] for o in new_next)

        # Queue dispatched work in a canonical order on both sides so the
        # random completion picks below refer to the same logical ops.
        ref_running += [o for o in sorted(ref_next, key=lambda o: ref_idx[o.op_id])
                        if o.transfer_type != TransferType.VIRTUAL]
        new_running += [o for o in sorted(new_next, key=lambda o: new_idx[o.op_id])
                        if o.transfer_type != TransferType.VIRTUAL]
        if not ref_running:
            if not pending:
                break
            ref_finished, new_finished = [], []
            continue  # nothing in flight, but graphs are still arriving

        order = list(range(len(ref_running)))
        rng.shuffle(order)
        picked = set(order[:rng.randint(1, len(ref_running))])
        ref_finished = [o for i, o in enumerate(ref_running) if i in picked]
        new_finished = [o for i, o in enumerate(new_running) if i in picked]
        ref_running = [o for i, o in enumerate(ref_running) if i not in picked]
        new_running = [o for i, o in enumerate(new_running) if i not in picked]

    # Liveness: every graph must drain, and every op must reach COMPLETED. A
    # scheduler that skipped a graph it should have revisited would park one
    # here forever -- in production a hung request rather than a wrong answer.
    assert not new_sched._transfer_graphs, \
        f"graphs left un-completed: {list(new_sched._transfer_graphs)}"
    assert not ref_sched._transfer_graphs, \
        f"reference left graphs un-completed: {list(ref_sched._transfer_graphs)}"
    assert not pending, "held-back graphs were never injected"
    for _graph, ops in new_graphs:
        for op in ops:
            assert op.status == TransferOpStatus.COMPLETED, \
                f"op {op.op_id} stuck in {op.status}"
