from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from flexkv.transfer.worker import TransferWorkerBase
    from flexkv.transfer.worker_op import WorkerTransferOp


class CompressionStrategy(ABC):
    # Whether this strategy issues transfers through the worker's
    # ``TPTransferThreadGroup`` rather than its region batch. The GPU<->CPU
    # worker builds that object only when something will actually call it, so
    # a strategy that needs it has to say so before ``attach`` runs.
    needs_gpu_cpu_thread_group: bool = False

    def compressed_group_indices(self) -> "list[int]":
        """Which layer groups of a multi-group worker this strategy moves.

        Empty -- the default, and what every uniform-KV strategy returns --
        means the worker transfers all of its groups itself. A strategy that
        returns ordinals takes those groups and leaves the rest to the worker's
        ordinary uncompressed path; the two sets address disjoint regions of
        the same block, so they may run in either order.

        This is asked rather than assumed so the worker does not have to know
        which compression backend it was handed.
        """
        return []

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
