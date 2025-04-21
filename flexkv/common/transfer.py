from enum import Enum, auto
from typing import List, Optional, Set, Dict
from dataclasses import dataclass, field
import torch
import threading

from flexkv.common.block import BlockMeta

class DeviceType(Enum):
    CPU = 0
    GPU = 1
    SSD = 2

class TransferType(Enum):
    H2D    = 0  # Host to Device transfer
    D2H    = 1  # Device to Host transfer
    DISK2H = 2  # Disk to Host transfer
    H2DISK = 3  # Host to Disk transfer
    DISK2D = 4  # Disk to Device transfer
    D2DISK = 5  # Device to Disk transfer

@dataclass
class TransferDescriptor:
    device_type: DeviceType = DeviceType.CPU
    device_id: int = 0
    physical_block_ids: torch.Tensor = torch.tensor([], dtype=torch.int64)
    blockmeta_list: Optional[List[BlockMeta]] = None
    layers: Optional[List[int]] = None
    tp_rank: Optional[int] = None

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
    src_descriptor: TransferDescriptor
    dst_descriptor: TransferDescriptor
    dependencies: Set[int] = field(default_factory=set)
    status: TransferOpStatus = TransferOpStatus.PENDING

@dataclass
class TransferOpGraph:
    transfer_graph_id: int
    block_meta_to_free: Dict[DeviceType, List[BlockMeta]]
    return_mask: Optional[torch.Tensor]
    _op_map: Dict[int, TransferOp] = field(init=False)

    def __init__(self, transfer_graph_id: int):
        self.transfer_graph_id = transfer_graph_id
        self._op_map = {}

    def add_transfer_op(self, op: TransferOp):
        op.transfer_graph_id = self.transfer_graph_id
        self._op_map[op.transfer_op_id] = op

    def add_dependency(self, op_id: int, dependent_op_id: int):
        """op_id depends on dependent_op_id"""
        if op_id in self._op_map and dependent_op_id in self._op_map:
            self._op_map[op_id].dependencies.add(dependent_op_id)

    def is_ready_to_execute(self, op_id: int) -> bool:
        """check if an op is ready to execute
        (all its dependencies are completed)"""
        if op_id not in self._op_map:
            return False
        op = self._op_map[op_id]
        if len(op.dependencies) == 0:
            return True
        return all(self._op_map[dep_id].status == TransferOpStatus.COMPLETED
                   for dep_id in op.dependencies)

    def mark_completed(self, op_id: int):
        """mark an op as completed"""
        if op_id in self._op_map:
            assert self._op_map[op_id].status == TransferOpStatus.RUNNING
            self._op_map[op_id].status = TransferOpStatus.COMPLETED

    def get_ready_ops(self) -> List[int]:
        """get a list of op ids that are ready to execute"""
        ready_ops = [op.transfer_op_id for op in self._op_map.values()
                if op.status == TransferOpStatus.PENDING and self.is_ready_to_execute(op.transfer_op_id)]
        for op_id in ready_ops:
            assert self._op_map[op_id].status == TransferOpStatus.PENDING
            self._op_map[op_id].status = TransferOpStatus.RUNNING
        return ready_ops

    def all_transfer_ops_completed(self) -> bool:
        """check if all transfer ops are completed"""
        return all(op.status == TransferOpStatus.COMPLETED for op in self._op_map.values())

    def get_block_meta_to_free(self) -> Dict[DeviceType, List[BlockMeta]]:
        """get a dict of block metas to free"""
        return self.block_meta_to_free

    def print_op_map(self):
        """Print transfer op graph in a visual format showing dependencies.

        Example output:
        Transfer Graph 5:
        ├── Op 1 (H2D) [Completed]
        │   └── No dependencies
        ├── Op 2 (D2H) [Pending]
        │   └── Depends on: 1
        └── Op 3 (DISK2H) [Pending]
            └── Depends on: 1, 2
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

            # print the dependency info
            dep_prefix = "    " if is_last else "│   "
            if not op.dependencies:
                print(f"{dep_prefix}└── No dependencies")
            else:
                deps_str = ", ".join(str(dep) for dep in sorted(op.dependencies))
                print(f"{dep_prefix}└── Depends on: {deps_str}")

            # print the transfer details
            src_info = f"From: {op.src_descriptor.device_type.name}:{op.src_descriptor.device_id}"
            dst_info = f"To: {op.dst_descriptor.device_type.name}:{op.dst_descriptor.device_id}"
            print(f"{dep_prefix}    └── {src_info} -> {dst_info}")

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
