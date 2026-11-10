"""CPU <-> remote store transfers.

This edge has no native engine: every byte moves through a ``StorageBackend``
(PCFS or mooncake-store), so the class is little more than the host-side
geometry those backends read in ``attach``.
"""

import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from multiprocessing.connection import Connection
from typing import List, Union

import torch
from torch.multiprocessing import Queue as MPQueue

from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import KVCacheLayout
from flexkv.common.transfer import TransferType
from flexkv.storage.allocator import HugePageTensorHandle, materialize_worker_tensor
from flexkv.transfer import trace
from flexkv.transfer.backends import StorageBackend
from flexkv.transfer.completion import CompletionContract
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
        client = getattr(backend, "mooncake_client", None)
        if getattr(client, "_use_parallel_channels", False):
            workers = int(os.environ.get("FLEXKV_MOONCAKE_IO_WORKERS", "8"))
            capacity = int(os.environ.get("FLEXKV_MOONCAKE_IO_MAX_PENDING_OPS", str(workers * 2)))
            timeout = float(os.environ.get("FLEXKV_MOONCAKE_IO_TIMEOUT_S", "300"))
            if not 1 <= workers <= 8 or not workers <= capacity <= 256:
                raise ValueError("Mooncake requires 1 <= workers <= 8 and workers <= capacity <= 256")
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Mooncake IO timeout must be finite and positive")
            self._remote_capacity = capacity
            self._remote_timeout = timeout
            self._remote_pending = {}
            self._remote_executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="flexkv-remote-op")

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

    def _execute_remote(self, op):
        started_ns = time.perf_counter_ns()
        started = time.time()
        contract = getattr(op, "completion_contract", CompletionContract.WHOLE)
        if contract not in (CompletionContract.WHOLE, CompletionContract.WHOLE.value):
            raise ValueError("Remote host IO only implements CompletionContract.WHOLE")
        if op.transfer_type not in (TransferType.REMOTE2H, TransferType.H2REMOTE):
            raise ValueError("Remote pipeline only accepts host/store transfers")
        backend = self._backend
        client = backend.mooncake_client
        client._channel_stats.transferred_bytes = 0
        src, dst = self.get_transfer_block_ids(op, pinned=False)
        block_results, _ = backend.transfer_blocks(self, op, src, dst)
        moved = client._channel_stats.transferred_bytes
        completed_ns = time.perf_counter_ns()
        metrics = None
        try:
            self._log_transfer_performance(op, moved, started, time.time())
            metrics = trace.build_worker_metrics(
                op, getattr(op, "prof_submitted_ns", 0),
                getattr(op, "_received_ns", started_ns), started_ns, completed_ns,
                self.worker_id, self._trace_bytes_per_block(), self.kv_dim, False)
        except Exception:
            flexkv_logger.warning("Mooncake pipeline diagnostic failed", exc_info=True)
        return WorkerTransferResult(op.transfer_op_id, tuple(block_results)), metrics

    def _execute_admitted(self, op, gate, accepted):
        gate.wait()
        if accepted[0]:
            return self._execute_remote(op)
        return None

    def _remote_fatal(self, reason):
        self._remote_poisoned = True
        flexkv_logger.error(reason + "; retaining all remote ownership until process exit")
        os.abort()
        raise RuntimeError(reason)

    def _publish_remote(self, future):
        op, _ = self._remote_pending[future]
        try:
            result, metrics = future.result()
        except Exception:
            flexkv_logger.error("Mooncake operation failed after drain", exc_info=True)
            result = WorkerTransferResult(op.transfer_op_id, (False,) * op.valid_block_num)
            metrics = None
        except BaseException:
            self._remote_fatal("Mooncake operation exited without a DMA completion certificate")
        try:
            self.finished_ops_queue.put((result, True, metrics))
        except BaseException:
            self._remote_fatal("Mooncake completion queue failed")
        del self._remote_pending[future]

    def _poll_remote(self, *, block=False):
        if not self._remote_pending:
            return
        done, _ = wait(self._remote_pending, timeout=0.01 if block else 0,
                       return_when=FIRST_COMPLETED)
        for future in done:
            self._publish_remote(future)
        now = time.monotonic()
        if any(now - submitted > self._remote_timeout
               for _, submitted in self._remote_pending.values()):
            self._remote_fatal("Mooncake request completion deadline exceeded")

    def _drain_remote(self):
        while self._remote_pending:
            self._poll_remote(block=True)

    def run(self):
        if getattr(self, "_remote_executor", None) is None:
            return super().run()
        try:
            while True:
                self._poll_remote()
                if len(self._remote_pending) >= self._remote_capacity:
                    self._poll_remote(block=True)
                    continue
                if not self.transfer_conn.poll(timeout=0.01):
                    continue
                try:
                    op = self.transfer_conn.recv()
                except EOFError:
                    return
                if op is None:
                    return
                if isinstance(op, dict) and op.get("type") == "control":
                    self._drain_remote()
                    self._reply_control(op)
                    continue
                op._received_ns = time.perf_counter_ns()
                gate, accepted = threading.Event(), [False]
                future = None
                try:
                    future = self._remote_executor.submit(self._execute_admitted, op, gate, accepted)
                    self._remote_pending[future] = (op, time.monotonic())
                except Exception:
                    self._remote_pending.pop(future, None)
                    gate.set()
                    self.finished_ops_queue.put((op.transfer_op_id, False, None))
                    raise
                except BaseException:
                    gate.set()
                    self._remote_unacknowledged_op = op
                    self._remote_fatal("Mooncake operation submission interrupted")
                accepted[0] = True
                gate.set()
        finally:
            if not getattr(self, "_remote_poisoned", False):
                self._drain_remote()

    def shutdown(self):
        executor = getattr(self, "_remote_executor", None)
        if executor is not None:
            if getattr(self, "_remote_poisoned", False):
                self._remote_fatal("Refusing to unregister undrained remote buffers")
            self._drain_remote()
            executor.shutdown(wait=True)
            self._remote_executor = None
        super().shutdown()
