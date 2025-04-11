from enum import Enum, auto
from typing import List, Optional, Set, Dict
from dataclasses import dataclass, field
import torch


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

@dataclass
class TransferOp:
    transfer_op_id: int
    transfer_graph_id: int
    transfer_type: TransferType
    src_descriptor: TransferDescriptor
    dst_descriptor: TransferDescriptor
    dependencies: Set[int] = field(default_factory=set)
    is_completed: bool = False

@dataclass
class TransferOpGraph:
    transfer_graph_id: int
    transfer_ops: List[TransferOp]
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
        """check if an op is ready to execute (all its dependencies are completed)"""
        if op_id not in self._op_map:
            return False
        op = self._op_map[op_id]
        if len(op.dependencies) == 0:
            return True
        return all(self._op_map[dep_id].is_completed for dep_id in op.dependencies)
    
    def mark_completed(self, op_id: int):
        """mark an op as completed"""
        if op_id in self._op_map:
            self._op_map[op_id].is_completed = True
    
    def get_ready_ops(self) -> List[int]:
        """get a list of op ids that are ready to execute"""
        return [op.transfer_op_id for op in self.transfer_ops 
                if not op.is_completed and self.is_ready_to_execute(op.transfer_op_id)]
    
    def all_transfer_ops_completed(self) -> bool:
        """check if all transfer ops are completed"""
        return all(op.is_completed for op in self.transfer_ops)
    
    def get_block_meta_to_free(self) -> Dict[DeviceType, List[BlockMeta]]:
        """get a dict of block metas to free"""
        return self.block_meta_to_free