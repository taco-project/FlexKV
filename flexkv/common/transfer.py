import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, List, Set, Dict, Callable, Tuple

import numpy as np


@dataclass(frozen=True)
class CompletedOp:
    graph_id: int
    op_id: int

    def is_graph_completed(self) -> bool:
        return self.op_id == -1

    def to_tuple(self) -> Tuple[int, int]:
        return (self.graph_id, self.op_id)

    @classmethod
    def from_tuple(cls, data: Tuple[int, int]) -> 'CompletedOp':
        return cls(graph_id=data[0], op_id=data[1])

    @classmethod
    def completed_graph(cls, graph_id: int) -> 'CompletedOp':
        return cls(graph_id=graph_id, op_id=-1)


class DeviceType(Enum):
    CPU = 0
    GPU = 1
    SSD = 2
    REMOTE = 3

class TransferType(Enum):
    H2D    = "H2D"
    D2H    = "D2H"
    DISK2H = "DISK2H"
    H2DISK = "H2DISK"
    DISK2D = "DISK2D"
    D2DISK = "D2DISK"
    REMOTE2H = "REMOTE2H"
    H2REMOTE = "H2REMOTE"
    # if we need to return a results when trasnfer op 1 and op 2 are completed
    # we can add a virtual transfer op 3 that depends on op 1 and op 2
    # so that the op 3 will not be executed actually, but can indicate the completion of
    # a group of transfer ops
    VIRTUAL = "Virtual"

class PartitionBlockType(Enum):
    ROUND_ROBIN = 0
    SEQUENTIAL = 1

class TransferOpStatus(Enum):
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2

@dataclass
class TransferOp:
    _next_op_id: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    op_id: int = field(init=False)
    graph_id: int
    transfer_type: TransferType
    src_block_ids: np.ndarray
    dst_block_ids: np.ndarray
    layer_id: int = 0
    layer_granularity: int = -1
    # this will change dynamically as transfer ops executed
    predecessors: Set[int] = field(default_factory=set)
    # this will keep the full info
    successors: Set[int] = field(default_factory=set)
    status: TransferOpStatus = TransferOpStatus.PENDING
    dp_id: int = 0
    # used for get block ids inner worker process
    src_slot_id: int = -1
    dst_slot_id: int = -1
    valid_block_num: int = 0

    def __post_init__(self) -> None:
        if self.transfer_type != TransferType.VIRTUAL and \
            self.src_block_ids.size != self.dst_block_ids.size:
            raise ValueError(f"src_block_ids and dst_block_ids must have the same number of physical blocks, but got "
                             f"src_block_ids.size={self.src_block_ids.size}, "
                             f"dst_block_ids.size={self.dst_block_ids.size}")
        with TransferOp._lock:
            self.op_id = TransferOp._next_op_id
            TransferOp._next_op_id += 1
        assert self.src_block_ids.dtype == np.int64
        assert self.dst_block_ids.dtype == np.int64
        self.valid_block_num = self.src_block_ids.size


class TransferOpGraph:
    _next_graph_id = 0
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.graph_id = self._get_graph_id()
        self._op_map: Dict[int, TransferOp] = {}
        self._ready_ops: Set[int] = set()
        self._trigger_ops: Set[int] = set()
        self._gpu_transfer_op_id: List[int] = []

    @classmethod
    def _get_graph_id(cls) -> int:
        with cls._lock:
            graph_id = cls._next_graph_id
            cls._next_graph_id += 1
            return graph_id

    def set_graph_id(self, graph_id: int) -> None:
        self.graph_id = graph_id

    @classmethod
    def create_empty_graph(cls) -> "TransferOpGraph":
        return cls()

    def add_virtual_op(self, op: TransferOp, need_trigger: bool = False) -> None:
        op.graph_id = self.graph_id
        op.transfer_type = TransferType.VIRTUAL
        self._op_map[op.op_id] = op
        if need_trigger:
            self._trigger_ops.add(op.op_id)
        else:
            self._ready_ops.add(op.op_id)

    def trigger_op(self, op_id: int) -> None:
        self._trigger_ops.remove(op_id)
        self._ready_ops.discard(op_id)
        self.mark_completed(op_id)

    def add_transfer_op(self, op: TransferOp) -> None:
        op.graph_id = self.graph_id
        self._op_map[op.op_id] = op
        if op.transfer_type == TransferType.H2D or \
            op.transfer_type == TransferType.D2H or \
            op.transfer_type == TransferType.D2DISK or \
            op.transfer_type == TransferType.DISK2D:
            self._gpu_transfer_op_id.append(op.op_id)
        self._ready_ops.add(op.op_id)

    def add_dependency(self, successor_op_id: int, predecessor_op_id: int) -> None:
        """successor_op_id depends on predecessor_op_id"""
        assert successor_op_id in self._op_map and predecessor_op_id in self._op_map
        self._op_map[successor_op_id].predecessors.add(predecessor_op_id)
        self._op_map[predecessor_op_id].successors.add(successor_op_id)
        self._ready_ops.discard(successor_op_id)

    def mark_completed(self, op_id: int) -> None:
        """mark an op as completed"""
        if op_id in self._op_map:
            assert self._op_map[op_id].status == TransferOpStatus.RUNNING
            self._op_map[op_id].status = TransferOpStatus.COMPLETED
            my_successors = self._op_map[op_id].successors
            for successor_id in my_successors:
                self._op_map[successor_id].predecessors.remove(op_id)

    def take_ready_ops(self) -> List[int]:
        """get a list of op ids that are ready to execute"""
        ready_ops = []
        to_remove = []
        to_add = []
        for op_id in self._ready_ops:
            op = self._op_map[op_id]
            if op.status == TransferOpStatus.COMPLETED:
                to_remove.append(op_id)
                for successor_id in op.successors:
                    if (self._op_map[successor_id].status == TransferOpStatus.PENDING and
                        len(self._op_map[successor_id].predecessors) == 0):
                        ready_ops.append(successor_id)
                        self._op_map[successor_id].status = TransferOpStatus.RUNNING
                        to_add.append(successor_id)
            elif op.status == TransferOpStatus.PENDING: # not supposed to happen now
                ready_ops.append(op_id)
                self._op_map[op_id].status = TransferOpStatus.RUNNING
                to_add.append(op_id)

        self._ready_ops.difference_update(to_remove)
        self._ready_ops.update(to_add)
        return ready_ops

    def all_transfer_ops_completed(self) -> bool:
        """check if all transfer ops are completed"""
        return all(op.status == TransferOpStatus.COMPLETED
                   for op in self._op_map.values())

    def set_gpu_blocks(self, gpu_blocks: np.ndarray) -> None:
        for op_id in self._gpu_transfer_op_id:
            transfer_type = self._op_map[op_id].transfer_type
            op = self._op_map[op_id]
            if transfer_type.name.endswith("2D"):
                if transfer_type == TransferType.DISK2D:
                    op.dst_block_ids = gpu_blocks[-op.dst_block_ids.size:]
                else:
                    op.dst_block_ids = gpu_blocks[:op.dst_block_ids.size]
            else:
                if transfer_type == TransferType.D2DISK:
                    op.src_block_ids = gpu_blocks[-op.src_block_ids.size:]
                else:
                    op.src_block_ids = gpu_blocks[:op.src_block_ids.size]
            assert op.src_block_ids.size == op.dst_block_ids.size, \
                f"src_block_ids.size={op.src_block_ids.size}, dst_block_ids.size={op.dst_block_ids.size}"

    @property
    def num_ops(self) -> int:
        return len(self._op_map)

    def bind_to_dp_group(self, dp_id: int) -> None:
        for op in self._op_map.values():
            op.dp_id = dp_id

    def visualize(self) -> str:
        """
        Visualize the transfer op graph in a readable format.
        Returns a string representation of the graph.
        """
        lines = []
        lines.append(f"╔{'═' * 70}╗")
        lines.append(f"║ TransferOpGraph (graph_id={self.graph_id}, num_ops={self.num_ops})".ljust(71) + "║")
        lines.append(f"╠{'═' * 70}╣")

        if not self._op_map:
            lines.append("║ (empty graph)".ljust(71) + "║")
            lines.append(f"╚{'═' * 70}╝")
            return "\n".join(lines)

        # Sort ops by op_id for consistent display
        sorted_ops = sorted(self._op_map.values(), key=lambda op: op.op_id)

        for i, op in enumerate(sorted_ops):
            # Op header
            status_symbol = {"PENDING": "○", "RUNNING": "◐", "COMPLETED": "●"}.get(op.status.name, "?")
            lines.append(f"║ [{status_symbol}] Op {op.op_id}: {op.transfer_type.value}".ljust(71) + "║")

            # Dependencies
            if op.predecessors:
                pred_str = ", ".join(str(p) for p in sorted(op.predecessors))
                lines.append(f"║     ├─ predecessors: [{pred_str}]".ljust(71) + "║")
            else:
                lines.append("║     ├─ predecessors: (none - ready)".ljust(71) + "║")

            if op.successors:
                succ_str = ", ".join(str(s) for s in sorted(op.successors))
                lines.append(f"║     ├─ successors:   [{succ_str}]".ljust(71) + "║")

            # Block info (truncate if too long)
            if op.transfer_type != TransferType.VIRTUAL:
                src_size = op.src_block_ids.size
                dst_size = op.dst_block_ids.size

                # Show first few and last few block ids
                def format_blocks(block_ids, max_show=4):
                    if block_ids.size == 0:
                        return "[]"
                    elif block_ids.size <= max_show * 2:
                        return str(block_ids.tolist())
                    else:
                        first = block_ids[:max_show].tolist()
                        last = block_ids[-max_show:].tolist()
                        return f"{first[:-1]}...{last[-1]}] (n={block_ids.size})"

                src_str = format_blocks(op.src_block_ids)
                dst_str = format_blocks(op.dst_block_ids)
                lines.append(f"║     ├─ src_blocks:   {src_str}".ljust(71) + "║")
                lines.append(f"║     ├─ dst_blocks:   {dst_str}".ljust(71) + "║")
                lines.append(f"║     └─ layer_id={op.layer_id}, dp_id={op.dp_id}".ljust(71) + "║")
            else:
                lines.append("║     └─ (VIRTUAL - no blocks)".ljust(71) + "║")

            # Separator between ops
            if i < len(sorted_ops) - 1:
                lines.append(f"║{'-' * 70}║")

        # Show ready ops
        lines.append(f"╠{'═' * 70}╣")
        ready_str = ", ".join(str(op_id) for op_id in sorted(self._ready_ops)) if self._ready_ops else "(none)"
        lines.append(f"║ Ready ops: [{ready_str}]".ljust(71) + "║")

        if self._trigger_ops:
            trigger_str = ", ".join(str(op_id) for op_id in sorted(self._trigger_ops))
            lines.append(f"║ Trigger ops: [{trigger_str}]".ljust(71) + "║")

        lines.append(f"╚{'═' * 70}╝")

        result = "\n".join(lines)
        print(result)
        return result

def merge_to_batch_graph(batch_id: int, transfer_graphs: List[TransferOpGraph], task_end_op_ids: List[int],
                         op_callback_dict: Dict[int, Callable]) -> Tuple[TransferOpGraph, int, Dict[int, Callable]]:
    """
    Merge multiple TransferOpGraphs into a single batch graph.

    Simplified assumption: each graph has at most two ops - DISK2H and H2D.
    - All DISK2H ops are merged into one (if any exist)
    - All H2D ops are merged into one
    - Dependency: DISK2H -> H2D
    - H2D becomes the task_end_op_id
    - For other transfer types (REMOTE, GDS, etc.), raise error

    Args:
        batch_id: ID for the new batch graph
        transfer_graphs: List of graphs to merge
        task_end_op_ids: List of end op IDs for each task (one per graph)
        op_callback_dict: Dict mapping old op_id -> callback

    Returns:
        (merged_graph, batch_end_op_id, new_op_callback_dict)
    """
    if not transfer_graphs:
        empty_graph = TransferOpGraph()
        empty_graph.set_graph_id(batch_id)
        return empty_graph, -1, {}

    merged_graph = TransferOpGraph()
    merged_graph.set_graph_id(batch_id)

    # Collect DISK2H and H2D ops separately
    disk2h_ops: List[TransferOp] = []
    h2d_ops: List[TransferOp] = []

    # New callback dict: merged_op_id -> list of callbacks
    disk2h_callbacks: List[Callable] = []
    h2d_callbacks: List[Callable] = []

    for graph in transfer_graphs:
        for op_id, op in graph._op_map.items():
            if op.transfer_type == TransferType.VIRTUAL:
                # Skip VIRTUAL ops
                continue
            elif op.transfer_type == TransferType.DISK2H:
                disk2h_ops.append(op)
                if op.op_id in op_callback_dict:
                    disk2h_callbacks.append(op_callback_dict[op.op_id])
            elif op.transfer_type == TransferType.H2D:
                h2d_ops.append(op)
                if op.op_id in op_callback_dict:
                    h2d_callbacks.append(op_callback_dict[op.op_id])
            else:
                # Unsupported transfer type for batch merge
                raise NotImplementedError(
                    f"Batch merge does not support transfer type: {op.transfer_type}. "
                    f"Only DISK2H and H2D are supported. "
                    f"Remote access and GDS are not yet supported for batch merge."
                )

    new_op_callback_dict: Dict[int, Callable] = {}
    merged_disk2h_op = None
    merged_h2d_op = None

    # Merge all DISK2H ops into one
    if disk2h_ops:
        src_blocks = np.concatenate([op.src_block_ids for op in disk2h_ops])
        dst_blocks = np.concatenate([op.dst_block_ids for op in disk2h_ops])

        merged_disk2h_op = TransferOp(
            graph_id=merged_graph.graph_id,
            transfer_type=TransferType.DISK2H,
            src_block_ids=src_blocks,
            dst_block_ids=dst_blocks,
            layer_id=disk2h_ops[0].layer_id,
            layer_granularity=disk2h_ops[0].layer_granularity,
            dp_id=disk2h_ops[0].dp_id,
        )
        merged_graph.add_transfer_op(merged_disk2h_op)

        # Attach callbacks - create a combined callback if multiple
        if disk2h_callbacks:
            if len(disk2h_callbacks) == 1:
                new_op_callback_dict[merged_disk2h_op.op_id] = disk2h_callbacks[0]
            else:
                # Create a combined callback that calls all original callbacks
                def make_combined_callback(callbacks):
                    def combined_callback(*args, **kwargs):
                        for cb in callbacks:
                            cb(*args, **kwargs)
                    return combined_callback
                new_op_callback_dict[merged_disk2h_op.op_id] = make_combined_callback(disk2h_callbacks)

    # Merge all H2D ops into one
    if h2d_ops:
        src_blocks = np.concatenate([op.src_block_ids for op in h2d_ops])
        dst_blocks = np.concatenate([op.dst_block_ids for op in h2d_ops])

        merged_h2d_op = TransferOp(
            graph_id=merged_graph.graph_id,
            transfer_type=TransferType.H2D,
            src_block_ids=src_blocks,
            dst_block_ids=dst_blocks,
            layer_id=h2d_ops[0].layer_id,
            layer_granularity=h2d_ops[0].layer_granularity,
            dp_id=h2d_ops[0].dp_id,
        )
        merged_graph.add_transfer_op(merged_h2d_op)

        # Attach callbacks - create a combined callback if multiple
        if h2d_callbacks:
            if len(h2d_callbacks) == 1:
                new_op_callback_dict[merged_h2d_op.op_id] = h2d_callbacks[0]
            else:
                def make_combined_callback(callbacks):
                    def combined_callback(*args, **kwargs):
                        for cb in callbacks:
                            cb(*args, **kwargs)
                    return combined_callback
                new_op_callback_dict[merged_h2d_op.op_id] = make_combined_callback(h2d_callbacks)

    # Add dependency: DISK2H -> H2D
    if merged_disk2h_op is not None and merged_h2d_op is not None:
        merged_graph.add_dependency(merged_h2d_op.op_id, merged_disk2h_op.op_id)

    # Determine the batch_end_op_id
    # Priority: H2D (if exists) -> DISK2H (if exists) -> -1
    if merged_h2d_op is not None:
        batch_end_op_id = merged_h2d_op.op_id
    elif merged_disk2h_op is not None:
        batch_end_op_id = merged_disk2h_op.op_id
    else:
        batch_end_op_id = -1

    return merged_graph, batch_end_op_id, new_op_callback_dict


def get_nvtx_default_color() -> int:
    return 0xD3D3D3

def get_nvtx_range_color(number: int) -> int:
    color = (number * 0x9e3779b1) % 0xffffff
    return color
