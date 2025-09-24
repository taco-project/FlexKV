import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, List, Set, Dict, Optional

import numpy as np


class DeviceType(Enum):
    CPU = 0
    GPU = 1
    SSD = 2
    REMOTE = 3
    PEERCPU = 4
    PEERSSD = 5

class TransferType(Enum):
    H2D    = "H2D"
    D2H    = "D2H"
    DISK2H = "DISK2H"
    H2DISK = "H2DISK"
    DISK2D = "DISK2D"
    D2DISK = "D2DISK"
    REMOTE2H = "REMOTE2H"
    H2REMOTE = "H2REMOTE"
    PEERH2H = "PEERH2H"
    H2PEERH = "H2PEERH"
    PEERSSD2H = "PEERSSD2H"
    H2PEERSSD = "H2PEERSSD"

    DIST2H = "DIST2H"
    # if we need to return a results when trasnfer op 1 and op 2 are completed
    # we can add a virtual transfer op 3 that depends on op 1 and op 2
    # so that the op 3 will not be executed actually, but can indicate the completion of
    # a group of transfer ops
    VIRTUAL = "Virtual"

class DistType(Enum):
    DISTH = "DISTH"
    DISTSSD = "DISTSSD"

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
    # src_block_node_ids: Optional[np.ndarray] = None
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
    remote_node_ids: Optional[np.ndarray] = None
    # used for distributed cpu and ssd
    src_block_node_ids: Optional[np.ndarray] = None
    dist_type: Optional[DistType] = None

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

def get_nvtx_default_color() -> int:
    return 0xD3D3D3

def get_nvtx_range_color(number: int) -> int:
    color = (number * 0x9e3779b1) % 0xffffff
    return color
