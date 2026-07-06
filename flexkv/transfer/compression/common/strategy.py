from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from flexkv.transfer.worker import TransferWorkerBase
    from flexkv.transfer.worker_op import WorkerTransferOp


class CompressionStrategy(ABC):
    @abstractmethod
    def attach(self, worker: "TransferWorkerBase") -> None:
        ...

    @abstractmethod
    def run(
        self,
        worker: "TransferWorkerBase",
        op: "WorkerTransferOp",
        src_block_ids: "torch.Tensor",
        dst_block_ids: "torch.Tensor",
    ) -> None:
        ...

    def shutdown(self) -> None:
        pass

class NullCompressionStrategy(CompressionStrategy):
    def attach(self, worker: "TransferWorkerBase") -> None:
        pass

    def run(
        self,
        worker: "TransferWorkerBase",
        op: "WorkerTransferOp",
        src_block_ids: "torch.Tensor",
        dst_block_ids: "torch.Tensor",
    ) -> None:
        start_time = time.time()
        worker._transfer_impl(src_block_ids, dst_block_ids, op.transfer_type)
        end_time = time.time()
        transfer_size = (
            worker.chunk_size_in_bytes
            * worker.num_layers
            * op.valid_block_num
            * worker.kv_dim
        )
        worker._log_transfer_performance(op, transfer_size, start_time, end_time)
