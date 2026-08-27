"""CPU <-> remote store transfers.

This edge has no native engine: every byte moves through a ``StorageBackend``
(PCFS or mooncake-store), so the class is little more than the host-side
geometry those backends read in ``attach``.
"""

from multiprocessing.connection import Connection
from typing import List, Union

import torch
from torch.multiprocessing import Queue as MPQueue

from flexkv.common.storage import KVCacheLayout
from flexkv.storage.allocator import HugePageTensorHandle, materialize_worker_tensor
from flexkv.transfer.backends import StorageBackend
from flexkv.transfer.worker_op import WorkerTransferOp
from flexkv.transfer.workers.runtime import TransferWorkerBase


class CPURemoteTransferWorker(TransferWorkerBase):
    """CPU<->Remote edge.  The remote tier's engine is a ``StorageBackend``.

    This worker owns exactly what is true of the edge regardless of who is on
    the far side: the CPU pool, its layout, and the block ids an op names.
    Everything else -- remote files, PCFS node ids, remote strides, or a
    key/value client and its block hashes -- belongs to the backend.

    It used to own the PCFS half directly, and that is precisely why
    mooncake-store could not reuse it: a key/value store has no remote layout
    to compute strides from, so it grew a second worker class that duplicated
    this one's CPU-side geometry. Both are backends now
    (``PcfsRemoteBackend``, ``MooncakeStoreBackend``), and adding a third
    remote tier adds no worker at all.
    """

    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 cpu_blocks: Union[List[torch.Tensor], torch.Tensor, HugePageTensorHandle],
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 backend: StorageBackend):
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        # Same flag that decides whether per-op block ids get pinned: a backend
        # that never hands block ids to CUDA gains nothing from pinning the
        # shared op buffer either, and mooncake-store never did.
        if backend.needs_pinned_block_ids:
            self._pin_op_buffer()

        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        self.cpu_blocks = cpu_blocks
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        self.cpu_kv_layout = cpu_kv_layout
        self.dtype = dtype

        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        self.kv_dim = cpu_kv_layout.kv_dim
        self.num_kv_heads = cpu_kv_layout.num_kv_heads
        self.has_multi_group = (
            getattr(cpu_kv_layout, "layer_groups", None) is not None
        )

        self._attach_backend(backend)

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        return self._run_backend(transfer_op)
