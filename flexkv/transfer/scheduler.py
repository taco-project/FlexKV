from dataclasses import dataclass
from typing import OrderedDict, List, Set, Tuple

from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType


class TransferScheduler:
    def __init__(self) -> None:
        # Store all transfer graphs
        self._transfer_graphs: OrderedDict[int, TransferOpGraph] = OrderedDict()

    def add_transfer_graph(self, graph: TransferOpGraph) -> None:
        """Add a new transfer graph to the scheduler"""
        self._transfer_graphs[graph.graph_id] = graph

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
        # Mark completed operations
        for op in finished_ops:
            if op.graph_id in self._transfer_graphs:
                self._transfer_graphs[op.graph_id].mark_completed(op.op_id)

        # Get next batch of executable operations
        next_ops = []
        for graph in self._transfer_graphs.values():
            ready_op_ids = graph.take_ready_ops()
            for op_id in ready_op_ids:
                op = graph._op_map[op_id]
                if op.transfer_type == TransferType.VIRTUAL:
                    self._transfer_graphs[op.graph_id].mark_completed(op_id)
                next_ops.append(op)

        # Find completed transfer graphs
        completed_graph_ids = []
        for graph_id, graph in self._transfer_graphs.items():
            if graph.all_transfer_ops_completed():
                completed_graph_ids.append(graph_id)

        # Remove completed graphs from scheduler
        for graph_id in completed_graph_ids:
            self._transfer_graphs.pop(graph_id)

        return completed_graph_ids, next_ops
