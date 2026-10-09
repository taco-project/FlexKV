"""CPU <-> CXL memory transfers.

CXL memory appears as a NUMA node with memory but no CPUs.
Transfers are simple memcpy operations between CPU and CXL tensors.
"""

import time
from multiprocessing.connection import Connection
from typing import Any, List, Optional, Union

import torch
from torch.multiprocessing import Queue as MPQueue

from flexkv.common.config import LayerGroupSpec
from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import KVCacheLayout
from flexkv.common.transfer import TransferType
from flexkv.storage.allocator import HugePageTensorHandle, materialize_worker_tensor
from flexkv.transfer.worker_op import WorkerTransferOp
from flexkv.transfer.workers.runtime import TransferWorkerBase


class CPUCXLTransferWorker(TransferWorkerBase):
    """Transfer worker for CPU <-> CXL memory transfers.

    CXL memory appears as a NUMA node with memory but no CPUs.
    Transfers are simple memcpy operations between CPU and CXL tensors.
    """

    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
        cxl_blocks: torch.Tensor,
        cpu_kv_layout: KVCacheLayout,
        cxl_kv_layout: KVCacheLayout,
        dtype: torch.dtype,
        layer_groups: Optional[List[LayerGroupSpec]] = None,
    ):
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        self._pin_op_buffer()

        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        self.cpu_blocks = cpu_blocks
        self.cxl_blocks = cxl_blocks
        self.dtype = dtype

        self.num_layers = cpu_kv_layout.num_layer
        self.kv_dim = cpu_kv_layout.kv_dim

        self.block_stride_in_bytes = cpu_kv_layout.get_block_stride() * dtype.itemsize
        self._bytes_per_block = self.block_stride_in_bytes

        flexkv_logger.info(
            f"CPUCXLTransferWorker initialized: "
            f"cpu_blocks={cpu_blocks.shape}, cxl_blocks={cxl_blocks.shape}, "
            f"block_stride={self.block_stride_in_bytes} bytes"
        )

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        **kwargs: Any,
    ) -> None:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2CXL:
            src_tensor = self.cpu_blocks
            dst_tensor = self.cxl_blocks
        elif transfer_type == TransferType.CXL2H:
            src_tensor = self.cxl_blocks
            dst_tensor = self.cpu_blocks
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for CPUCXLTransferWorker")

        block_size = self.block_stride_in_bytes // self.dtype.itemsize
        for i in range(len(src_block_ids)):
            src_idx = src_block_ids[i].item()
            dst_idx = dst_block_ids[i].item()
            src_start = src_idx * block_size
            src_end = src_start + block_size
            dst_start = dst_idx * block_size
            dst_end = dst_start + block_size
            dst_tensor[dst_start:dst_end] = src_tensor[src_start:src_end]

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)
        start_time = time.time()
        self._transfer_impl(
            src_block_ids,
            dst_block_ids,
            transfer_op.transfer_type,
        )
        end_time = time.time()
        transfer_size = self._bytes_per_block * transfer_op.valid_block_num
        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )
        return True
