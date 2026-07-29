from dataclasses import dataclass
from typing import OrderedDict, List, Set, Tuple

from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType


class TransferScheduler:
    def __init__(self) -> None:
        # Store all transfer graphs
        self._transfer_graphs: OrderedDict[int, TransferOpGraph] = OrderedDict()
        # Graph ids whose op state changed since the last schedule() call, in
        # the order they became dirty. Dependencies never cross graphs, so a
        # graph can only expose new ready ops after one of its OWN ops
        # completes; visiting just these is equivalent to sweeping every
        # in-flight graph, at O(changed) instead of O(in-flight) per call.
        #
        # Standing requirement: any path that completes an op or otherwise
        # changes a graph's readiness must dirty that graph here. The old full
        # sweep tolerated such a path silently; this one will never revisit the
        # graph. TransferOpGraph.trigger_op() is one such path (no callers).
        self._dirty_graph_ids: OrderedDict[int, None] = OrderedDict()

    def add_transfer_graph(self, graph: TransferOpGraph) -> None:
        """Add a new transfer graph to the scheduler"""
        self._transfer_graphs[graph.graph_id] = graph
        self._dirty_graph_ids[graph.graph_id] = None

    def fail_graph(self, graph_id: int) -> None:
        """Drop a graph whose transfer failed: none of its remaining ops may be
        dispatched. Ops of this graph already running on workers are allowed to
        drain; schedule() already ignores finished ops whose graph is gone.
        Idempotent, and a no-op for graphs that already completed."""
        self._transfer_graphs.pop(graph_id, None)

    def schedule(self,
                finished_ops: List[TransferOp]
               ) -> Tuple[List[int], List[TransferOp]]:
        """
        Schedule transfer operations

        Args:
            finished_ops: Dictionary of completed transfer operations and their graph IDs

        Returns:
            Tuple[List[int], List[TransferOp]]:
                - List of completed transfer graph IDs
                - List of next executable transfer operations
        """
        # Mark completed operations. Dirty the graph before completing the op:
        # mark_completed() clears the op from its successors' predecessor sets
        # in a loop, so a raise partway through leaves the graph half-advanced
        # and needing a revisit. (Its leading assert fires before any mutation,
        # so that case needs no recovery either way -- this ordering is free
        # insurance for the partial-mutation one, not for the assert.)
        for op in finished_ops:
            if op.graph_id in self._transfer_graphs:
                self._dirty_graph_ids[op.graph_id] = None
                self._transfer_graphs[op.graph_id].mark_completed(op.op_id)

        # Drain the dirty set. Peek at the head and drop the dirty bit only once
        # the graph has been fully processed, so a raise below leaves the id
        # dirty and the graph gets revisited -- the caller logs and keeps
        # looping (TransferEngine._scheduler_loop), so a dropped dirty bit would
        # strand that graph for good. This recovers the dirty BIT, not the work:
        # ops already collected into next_ops are discarded along with the
        # exception, exactly as they were under the full sweep.
        next_ops = []
        completed_graph_ids = []
        while self._dirty_graph_ids:
            graph_id = next(iter(self._dirty_graph_ids))
            # Defensive: every id reaching here should resolve, since the only
            # writers pair the two dicts. Skipping beats a KeyError, which the
            # caller would swallow and then retry forever on the same id.
            graph = self._transfer_graphs.get(graph_id)
            revisit = False
            if graph is not None:
                for op_id in graph.take_ready_ops():
                    op = graph._op_map[op_id]
                    if op.transfer_type == TransferType.VIRTUAL:
                        # Self-completing, and that can unblock successors, so
                        # the graph needs another pass before this call returns.
                        graph.mark_completed(op_id)
                        revisit = True
                    next_ops.append(op)
                if graph.all_transfer_ops_completed():
                    completed_graph_ids.append(graph_id)
                    del self._transfer_graphs[graph_id]
                    # A finished graph needs no further pass. This is the common
                    # case -- a terminal virtual sink both sets revisit and
                    # completes the graph -- so skipping the re-queue keeps it
                    # off the hot path.
                    revisit = False
            del self._dirty_graph_ids[graph_id]
            if revisit:
                self._dirty_graph_ids[graph_id] = None  # re-queue at the tail

        return completed_graph_ids, next_ops
