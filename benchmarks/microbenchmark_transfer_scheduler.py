"""Microbenchmark: TransferScheduler.schedule() cost vs in-flight concurrency.

The transfer scheduler runs on a single thread and is woken once per batch of
finished ops. What we want to know is whether the cost of one wake-up depends on
how much actually changed, or on how many requests happen to be in flight.

This drives a fixed amount of real work per tick -- exactly one op of one graph
completes -- while varying the number of other, idle in-flight graphs. A
scheduler whose cost tracks real work stays flat; one that sweeps every graph
grows linearly with concurrency. Note this is the extreme case, exactly one
dirty graph per tick; with k dirty the incremental cost is O(k), which is the
whole point.

CPU only: no GPU, no c_ext transfers, no storage. Run with:
    python benchmarks/microbenchmark_transfer_scheduler.py --baseline
"""
from argparse import ArgumentParser
import time

import numpy as np

from flexkv.common.transfer import (
    TransferOp,
    TransferOpGraph,
    TransferOpStatus,
    TransferType,
)
from flexkv.transfer.scheduler import TransferScheduler


class FullSweepScheduler:
    """The pre-change scheduler, kept here so `--baseline` can print a
    before/after table from one run instead of asking you to check out the
    previous revision.

    `tests/test_transfer_scheduler_incremental.py` keeps its own copy, where it
    serves as the equivalence oracle; keep the two in step.
    """

    def __init__(self) -> None:
        self._transfer_graphs = {}

    def add_transfer_graph(self, graph) -> None:
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


def build_chain_graph(num_ops: int):
    """A linear chain op0 -> op1 -> ... -> op[n-1], like a staged transfer."""
    graph = TransferOpGraph.create_empty_graph()
    ops = []
    for i in range(num_ops):
        op = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=TransferType.H2D if i % 2 else TransferType.D2H,
            src_block_ids=np.arange(4, dtype=np.int64),
            dst_block_ids=np.arange(4, dtype=np.int64),
        )
        graph.add_transfer_op(op)
        ops.append(op)
    for i in range(1, num_ops):
        graph.add_dependency(ops[i].op_id, ops[i - 1].op_id)
    return graph, ops


def measure(num_graphs: int, ops_per_graph: int, cls=TransferScheduler,
            repeat: int = 5) -> float:
    """Microseconds per schedule() call that advances exactly one op.

    Best of `repeat` runs: a single run only times `ops_per_graph - 1` calls,
    which at low concurrency is a window of tens of microseconds -- short
    enough that the sign of the delta flips between runs.
    """
    return min(_measure_once(num_graphs, ops_per_graph, cls)
               for _ in range(repeat))


def _measure_once(num_graphs: int, ops_per_graph: int, cls) -> float:
    scheduler = cls()
    target_ops = None
    for i in range(num_graphs):
        graph, ops = build_chain_graph(ops_per_graph)
        scheduler.add_transfer_graph(graph)
        if i == 0:
            target_ops = ops

    scheduler.schedule([])  # prime: dispatch the head op of every graph

    idx = 0
    ticks = 0
    start = time.perf_counter()
    while idx < ops_per_graph - 1:
        op = target_ops[idx]
        if op.status != TransferOpStatus.RUNNING:
            # Would silently shorten the run and still print a plausible
            # number, so fail loudly instead.
            raise RuntimeError(
                f"op {op.op_id} is {op.status}, expected RUNNING -- the chain "
                f"did not advance as expected")
        scheduler.schedule([op])
        idx += 1
        ticks += 1
    elapsed = time.perf_counter() - start
    return elapsed / max(ticks, 1) * 1e6


def main():
    parser = ArgumentParser()
    parser.add_argument("--ops-per-graph", type=int, default=16)
    parser.add_argument("--concurrency", type=int, nargs="+",
                        default=[1, 8, 32, 128, 512, 1024, 2048])
    parser.add_argument("--baseline", action="store_true",
                        help="also measure the pre-change full-sweep scheduler")
    parser.add_argument("--repeat", type=int, default=5,
                        help="runs per data point; the best is reported")
    args = parser.parse_args()

    print(f"one schedule() tick advancing exactly one op "
          f"({args.ops_per_graph} ops per graph)\n")
    if args.baseline:
        print(f"{'in-flight graphs':>17} | {'full sweep us':>14} | {'dirty set us':>13}")
        print("-" * 50)
        for num_graphs in args.concurrency:
            before = measure(num_graphs, args.ops_per_graph,
                             FullSweepScheduler, args.repeat)
            after = measure(num_graphs, args.ops_per_graph,
                            TransferScheduler, args.repeat)
            print(f"{num_graphs:>17} | {before:>14.2f} | {after:>13.2f}")
        print("\nThe left column grows with concurrency, the right one does not.")
    else:
        print(f"{'in-flight graphs':>17} | {'us / tick':>10} | "
              f"{'us per in-flight graph':>23}")
        print("-" * 58)
        for num_graphs in args.concurrency:
            us = measure(num_graphs, args.ops_per_graph,
                         repeat=args.repeat)
            print(f"{num_graphs:>17} | {us:>10.2f} | {us / num_graphs:>23.4f}")
        print("\nA flat middle column means per-tick cost tracks real work "
              "rather than total in-flight concurrency.")


if __name__ == "__main__":
    main()
