from enum import Enum, auto
from typing import List, Optional, Set, Dict, Tuple
from dataclasses import dataclass, field
import torch
import threading
import nvtx

class DeviceType(Enum):
    CPU = 0
    GPU = 1
    SSD = 2
    REMOTE = 3
    
class TransferType(Enum):
    H2D    = "Host to Device"
    D2H    = "Device to Host"
    DISK2H = "Disk to Host"
    H2DISK = "Host to Disk"
    DISK2D = "Disk to Device"
    D2DISK = "Device to Disk"
    REMOTE2H = "Remote to Host"
    H2REMOTE = "Host to Remote"
    # if we need to return a results when trasnfer op 1 and op 2 are completed
    # we can add a virtual transfer op 3 that depends on op 1 and op 2
    # so that the op 3 will not be executed actually, but can indicate the completion of
    # a group of transfer ops
    VIRTUAL = "Virtual"

@dataclass
class TransferDescriptor:
    device_type: DeviceType = DeviceType.CPU
    device_id: int = 0
    physical_block_ids: torch.Tensor = torch.tensor([], dtype=torch.int64)

    def __post_init__(self):
        assert self.physical_block_ids.ndim == 1
        assert self.physical_block_ids.dtype == torch.int64

class TransferIDAllocator:
    _op_id_counter = 0
    _graph_id_counter = 0
    _lock = threading.Lock()

    @staticmethod
    def reset():
        with TransferIDAllocator._lock:
            TransferIDAllocator._op_id_counter = 0
            TransferIDAllocator._graph_id_counter = 0

    @staticmethod
    def allocate_op_id() -> int:
        with TransferIDAllocator._lock:
            TransferIDAllocator._op_id_counter += 1
            return TransferIDAllocator._op_id_counter

    @staticmethod
    def allocate_graph_id() -> int:
        with TransferIDAllocator._lock:
            TransferIDAllocator._graph_id_counter += 1
            return TransferIDAllocator._graph_id_counter

class TransferOpStatus(Enum):
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2

@dataclass
class TransferOp:
    transfer_op_id: int
    transfer_graph_id: int
    transfer_type: TransferType
    layer_id: int
    layer_granularity: int
    src_descriptor: TransferDescriptor = None
    dst_descriptor: TransferDescriptor = None
    # this will change dynamically as transfer ops executed
    predecessors: Set[int] = field(default_factory=set)
    # this will keep the full info
    successors: Set[int] = field(default_factory=set)
    status: TransferOpStatus = TransferOpStatus.PENDING
    dp_id: Optional[int] = None


class TransferOpGraph:

    def __init__(self, transfer_graph_id: int):
        self.transfer_graph_id = transfer_graph_id
        self._op_map: dict[int, TransferOp] = {}
        self._ready_ops = set()
        self._trigger_ops = set()

    @classmethod
    def create_empty_graph(cls, transfer_graph_id: int):
        graph = cls(transfer_graph_id)
        graph._op_map = {}
        graph._ready_ops = set()
        graph._trigger_ops = set()
        return graph

    def add_virtual_op(self, op: TransferOp, need_trigger: bool = False):
        op.transfer_graph_id = self.transfer_graph_id
        op.transfer_type = TransferType.VIRTUAL
        self._op_map[op.transfer_op_id] = op
        if need_trigger:
            self._trigger_ops.add(op.transfer_op_id)
        else:
            self._ready_ops.add(op.transfer_op_id)

    def trigger_op(self, op_id: int):
        self._trigger_ops.remove(op_id)
        self._ready_ops.discard(op_id)
        self.mark_completed(op_id)

    def add_transfer_op(self, op: TransferOp):
        op.transfer_graph_id = self.transfer_graph_id
        self._op_map[op.transfer_op_id] = op
        self._ready_ops.add(op.transfer_op_id)

    def add_dependency(self, successor_op_id: int, predecessor_op_id: int):
        """successor_op_id depends on predecessor_op_id"""
        assert successor_op_id in self._op_map and predecessor_op_id in self._op_map
        self._op_map[successor_op_id].predecessors.add(predecessor_op_id)
        self._op_map[predecessor_op_id].successors.add(successor_op_id)
        self._ready_ops.discard(successor_op_id)

    def mark_completed(self, op_id: int):
        """mark an op as completed"""
        if op_id in self._op_map:
            assert self._op_map[op_id].status == TransferOpStatus.RUNNING
            self._op_map[op_id].status = TransferOpStatus.COMPLETED
            my_successors = self._op_map[op_id].successors
            if len(my_successors) == 0:
                return
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

    @property
    def num_ops(self) -> int:
        return len(self._op_map)

    def bind_to_dp_group(self, dp_id: int):
        for op in self._op_map.values():
            op.dp_id = dp_id

    def print_op_map(self):
        """Print transfer op graph in a visual format showing dependencies.

        Example output:
        Transfer Graph 5:
        ├── Op 1 (H2D) [Completed]
        │   └── No successors
        ├── Op 2 (D2H) [Pending]
        │   └── Followed by: 1
        └── Op 3 (DISK2H) [Pending]
            └── Followed by: 1, 2
        """
        print(f"Transfer Graph {self.transfer_graph_id}:")

        # get all op ids and sort them
        op_ids = sorted(self._op_map.keys())

        for i, op_id in enumerate(op_ids):
            op = self._op_map[op_id]
            is_last = (i == len(op_ids) - 1)

            # draw the tree structure branch
            prefix = "└── " if is_last else "├── "

            # get the op status
            status = "[Completed]" if op.status == TransferOpStatus.COMPLETED else "[Pending]"

            # print the op info
            print(f"{prefix}Op {op_id} ({op.transfer_type.name}) {status}")

            if op.transfer_type == TransferType.VIRTUAL:
                continue
            # print the dependency info
            dep_prefix = "    " if is_last else "│   "
            if not op.successors:
                print(f"{dep_prefix}└── No successors")
            else:
                deps_str = ", ".join(str(dep) for dep in sorted(op.successors))
                print(f"{dep_prefix}└── Followed by: {deps_str}")

            # print the transfer details
            src_info = f"From: {op.src_descriptor.device_type.name}:{op.src_descriptor.device_id}"
            dst_info = f"To: {op.dst_descriptor.device_type.name}:{op.dst_descriptor.device_id}"
            print(f"{dep_prefix}    └── {src_info} -> {dst_info}")

            print(f"{dep_prefix}    └── layers: {op.layer_id} - {op.layer_id + op.layer_granularity}")

            # if there are physical block ids, also print them
            if len(op.src_descriptor.physical_block_ids) > 0:
                blocks = op.src_descriptor.physical_block_ids.tolist()
                if len(blocks) > 3:
                    blocks_str = f"{blocks[:3]}... ({len(blocks)} blocks)"
                else:
                    blocks_str = str(blocks)
                print(f"{dep_prefix}    └── Src Blocks: {blocks_str}")
            if len(op.dst_descriptor.physical_block_ids) > 0:
                blocks = op.dst_descriptor.physical_block_ids.tolist()
                if len(blocks) > 3:
                    blocks_str = f"{blocks[:3]}... ({len(blocks)} blocks)"
                else:
                    blocks_str = str(blocks)
                print(f"{dep_prefix}    └── Dst Blocks: {blocks_str}")

def get_nvtx_default_color() -> int:
    return 0xD3D3D3

def get_nvtx_range_color(number: int) -> int:
    color = (number * 0x9e3779b1) % 0xffffff
    return color
