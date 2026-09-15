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
from flexkv.transfer.geometry import ChunkStrides, EdgeGeometry, HostSide
from flexkv.transfer.worker_op import WorkerTransferOp, WorkerTransferResult
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

        # Read before materializing: only the handle knows the mapping's
        # aligned length, which is >= the logical pool and is what an external
        # RDMA registration must cover. A plain tensor pool maps exactly what
        # it holds, so None means "logical size is the mapped size".
        self.cpu_blocks_mapped_size = (
            int(cpu_blocks.aligned)
            if isinstance(cpu_blocks, HugePageTensorHandle) else None
        )
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

        self._attach_backend(backend, self._build_geometry(cpu_kv_layout, dtype))

    def _build_geometry(
        self, cpu_kv_layout: KVCacheLayout, dtype: torch.dtype
    ) -> EdgeGeometry:
        """This edge, in the terms every remote engine reads it in.

        There is no ``ssd`` or ``gpu`` side: a remote tier is reached from the
        CPU pool and nothing else, so a backend asking for either gets a
        sentence saying so rather than an ``AttributeError``.
        """
        itemsize = dtype.itemsize
        if self.has_multi_group:
            # Groups differ in chunk size, so there is no uniform (layer, kv)
            # chunk. get_block_stride() is already bytes for this layout, and
            # is the whole block.
            block_stride = int(cpu_kv_layout.get_block_stride())
            strides = None
            bytes_per_block = block_stride
        else:
            block_stride = cpu_kv_layout.get_block_stride() * itemsize
            chunk_bytes = cpu_kv_layout.get_chunk_size() * itemsize
            strides = ChunkStrides(
                chunk_bytes=chunk_bytes,
                kv_stride=cpu_kv_layout.get_kv_stride() * itemsize,
                layer_stride=cpu_kv_layout.get_layer_stride() * itemsize,
                block_stride=block_stride,
            )
            # Not ``block_stride``: that is the *addressing* stride between
            # consecutive blocks of one chunk, which under LAYERFIRST is one
            # chunk rather than the whole block.
            bytes_per_block = chunk_bytes * self.num_layers * self.kv_dim
        return EdgeGeometry(
            num_layers=self.num_layers,
            kv_dim=self.kv_dim,
            num_kv_heads=self.num_kv_heads,
            dtype=dtype,
            has_multi_group=self.has_multi_group,
            bytes_per_block=bytes_per_block,
            cpu=HostSide(
                layout=cpu_kv_layout,
                blocks=self.cpu_blocks,
                layer_ptrs=self.cpu_layer_ptrs,
                block_stride=block_stride,
                mapped_size=self.cpu_blocks_mapped_size,
                strides=strides,
            ),
        )

    def launch_transfer(
        self, transfer_op: WorkerTransferOp
    ) -> Union[bool, WorkerTransferResult]:
        return self._run_backend(transfer_op)
