from dataclasses import dataclass

import numpy as np

from flexkv.common.transfer import TransferOp, TransferType, LayerwiseTransferOp


@dataclass
class WorkerTransferOp:
    transfer_op_id: int
    transfer_graph_id: int
    transfer_type: TransferType
    layer_id: int
    layer_granularity: int
    src_slot_id: int
    dst_slot_id: int
    valid_block_num: int
    src_block_ids: np.ndarray
    dst_block_ids: np.ndarray

    def __init__(self, transfer_op: TransferOp):
        self.transfer_op_id = transfer_op.op_id
        self.transfer_graph_id = transfer_op.graph_id
        self.transfer_type = transfer_op.transfer_type
        self.layer_id = transfer_op.layer_id
        self.layer_granularity = transfer_op.layer_granularity
        self.src_slot_id = transfer_op.src_slot_id
        self.dst_slot_id = transfer_op.dst_slot_id
        self.valid_block_num = transfer_op.valid_block_num
        if self.src_slot_id == -1:
            self.src_block_ids = transfer_op.src_block_ids
            self.dst_block_ids = transfer_op.dst_block_ids
        else:
            self.src_block_ids = np.empty(0)
            self.dst_block_ids = np.empty(0)


@dataclass
class WorkerLayerwiseTransferOp:
    transfer_op_id: int
    transfer_graph_id: int
    transfer_type: TransferType
    layer_id: int
    layer_granularity: int
    src_block_ids_h2d: np.ndarray
    dst_block_ids_h2d: np.ndarray
    src_block_ids_disk2h: np.ndarray
    dst_block_ids_disk2h: np.ndarray
    counter_id: int  # Counter set index for triple buffering eventfd notification

    def __init__(self, transfer_op: LayerwiseTransferOp):
        self.transfer_op_id = transfer_op.op_id
        self.transfer_graph_id = transfer_op.graph_id
        assert transfer_op.transfer_type == TransferType.LAYERWISE
        self.transfer_type = transfer_op.transfer_type
        self.layer_id = transfer_op.layer_id
        self.layer_granularity = transfer_op.layer_granularity
        self.src_block_ids_h2d = transfer_op.src_block_ids_h2d
        self.dst_block_ids_h2d = transfer_op.dst_block_ids_h2d
        self.src_block_ids_disk2h = transfer_op.src_block_ids_disk2h
        self.dst_block_ids_disk2h = transfer_op.dst_block_ids_disk2h
        self.counter_id = transfer_op.counter_id
