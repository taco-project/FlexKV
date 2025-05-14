from typing import OrderedDict, List, Set, Tuple
from dataclasses import dataclass
from ..common.transfer import TransferOp, TransferOpGraph, TransferType

class TransferScheduler:
    def __init__(self):
        # Store all transfer graphs
        self._transfer_graphs: OrderedDict[int, TransferOpGraph] = OrderedDict()

    def add_transfer_graph(self, graph: TransferOpGraph):
        """Add a new transfer graph to the scheduler"""
        self._transfer_graphs[graph.transfer_graph_id] = graph

    def schedule(self,
                finished_ops: List[TransferOp]
               ) -> Tuple[List[TransferOpGraph], List[TransferOp]]:
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
            if op.transfer_graph_id in self._transfer_graphs:
                self._transfer_graphs[op.transfer_graph_id].mark_completed(op.transfer_op_id)

        # Get next batch of executable operations
        next_ops = []
        for graph in self._transfer_graphs.values():
            ready_op_ids = graph.take_ready_ops()
            for op_id in ready_op_ids:
                op = graph._op_map[op_id]
                if op.transfer_type == TransferType.VIRTUAL:
                    self._transfer_graphs[op.transfer_graph_id].mark_completed(op_id)
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
