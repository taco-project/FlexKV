"""Process-side runtime every transfer worker shares.

``TransferWorkerBase`` owns the parts that are the same whatever edge a worker
serves: the child-process entry point, the receive/batch/report loop, host
pinning and its paired unregister, block-id fetch from the shared op buffer,
the perf record, and control-message dispatch.

Concrete workers live in sibling modules and this module must not import them:
``create_worker`` is a classmethod on the base, so the concrete class is always
the one the caller already holds.
"""
import contextlib
import logging
import math
import os
import copy
import signal

import torch.multiprocessing as mp
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.multiprocessing import Queue as MPQueue, Pipe as MPPipe
from multiprocessing.connection import Connection
from threading import Thread
from typing import List, Any, Dict, Union, Optional, Tuple

import numpy as np
import nvtx
import torch
import zmq
import json

from flexkv import c_ext

from flexkv.c_ext import transfer_kv_blocks, transfer_kv_blocks_ssd, TPTransferThreadGroup

# GDS imports are optional (only available when compiled with FLEXKV_ENABLE_GDS=1)
try:
    from flexkv.c_ext import transfer_kv_blocks_gds, TPGDSTransferThreadGroup
except ImportError:
    transfer_kv_blocks_gds = None
    TPGDSTransferThreadGroup = None

from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle, release_vmm_tensor
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferOp, TransferType, PartitionBlockType
from flexkv.common.transfer import get_nvtx_range_color, LayerwiseTransferOp
from flexkv.common.config import (
    CacheConfig, GLOBAL_CONFIG_FROM_ENV, MooncakeTransferEngineConfig, LayerGroupSpec,
)
from flexkv.storage.allocator import HugePageTensorHandle, materialize_worker_tensor
from flexkv.transfer.host_buffer import (
    allocate_host_buffer,
    cudaHostRegister,
    safe_cuda_host_unregister,
)


from flexkv.transfer.compression.common.strategy import (
    CompressionStrategy,
    NullCompressionStrategy,
)
from flexkv.transfer.worker_op import (
    WorkerLayerwiseTransferOp,
    WorkerTransferOp,
    WorkerTransferResult,
)
from flexkv.transfer import trace
from flexkv.mooncakeEngineWrapper import MoonCakeTransferEngineWrapper
from flexkv.external.mooncake_store_keys import PoolKind, build_key
from flexkv.external.mooncake_fault_inject import inject_mooncake_fault, is_mooncake_fault_inject_enabled
from flexkv.transfer.zmqHelper import NotifyMsg, NotifyStatus, SSDZMQServer, SSDZMQClient
from flexkv.cache.redis_meta import RedisMeta
from flexkv.transfer.utils import (
    group_blocks_by_node_and_segment,
    group_blocks_by_node,
    split_contiguous_blocks,
    RemoteSSD2HMetaInfo,
    NodeMetaInfo,
    RDMATaskInfo,
)
from flexkv.transfer.nixlutil import (
    NIXL_CPU_FILE_BACKENDS,
    NIXL_GPU_FILE_BACKENDS,
    NixlAgentSession,
    normalize_nixl_file_plugin_name,
    file_path_for_ssd_block,
    gpu_chunk_u8_view,
    kv_chunk_byte_offset_in_block,
    ssd_chunk_byte_offset_in_file,
)
try:
    from flexkv.c_ext import (
        transfer_kv_blocks_remote,
        shared_transfer_kv_blocks_remote_read,
    )
except ImportError:
    transfer_kv_blocks_remote = None
    shared_transfer_kv_blocks_remote_read = None


from flexkv.transfer.workers.handle import WorkerHandle


def ensure_cuda_device(device: Union[int, torch.device, None]) -> None:
    """Bind this process's CUDA context before IPC import / host register / Stream.

    Workers must call this *before* any CUDA API. Otherwise the default device
    (usually GPU 0) gets a context from every worker, which under DP exhausts
    GPU0 and makes ``torch.cuda.Stream()`` OOM while ``ready_event.wait()`` hangs.
    """
    if device is None:
        return
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return
        idx = 0 if device.index is None else int(device.index)
    else:
        idx = int(device)
        if idx < 0:
            return
    torch.cuda.set_device(idx)


def import_tensor_handles(
    handles: List["TensorSharedHandle"],
) -> List[torch.Tensor]:
    """Import CUDA IPC tensors after switching to their owning device."""
    if handles:
        ensure_cuda_device(handles[0].device)
    return [h.get_tensor() for h in handles]


class TransferWorkerBase(ABC):
    _worker_id_counter = 0
    _worker_id_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any):
        # Allocate first so ``_worker_process`` can always hold a reference and
        # call shutdown() even when ``__init__`` fails mid-way after some pins.
        obj = super().__new__(cls)
        obj._host_registered = []
        obj._shutdown_done = False
        obj._op_buffer_pinned = False
        return obj

    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,  # receive end of pipe
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor):
        self.worker_id = worker_id
        self.transfer_conn = transfer_conn  # receive end of pipe
        self.finished_ops_queue: MPQueue = finished_ops_queue

        self.op_buffer_tensor = op_buffer_tensor
        self._op_buffer_pinned = False
        # (tensor, label) pairs registered via _register_host_tensor / _pin_op_buffer.
        self._host_registered: List[Tuple[torch.Tensor, str]] = []
        self._shutdown_done = False

    def _register_host_tensor(self, tensor: torch.Tensor, label: str = "") -> None:
        """cudaHostRegister and track for paired unregister in shutdown()."""
        size_gb = tensor.numel() * tensor.element_size() / (1024 ** 3)
        flexkv_logger.info(
            f"[worker {self.worker_id}] cudaHostRegister {label or 'host'}: "
            f"ptr=0x{tensor.data_ptr():x} size={size_gb:.3f} GiB"
        )
        cudaHostRegister(tensor)
        self._host_registered.append((tensor, label or "host"))

    def _pin_op_buffer(self) -> None:
        """Pin the shared op buffer after the worker has bound its CUDA device.

        Must not run before ``ensure_cuda_device`` / ``import_tensor_handles``,
        or every worker creates a default CUDA context on GPU0.
        """
        if not self._op_buffer_pinned:
            self._register_host_tensor(self.op_buffer_tensor, "op_buffer")
            self._op_buffer_pinned = True

    def shutdown(self) -> None:
        """Unregister all host tensors pinned by this worker. Idempotent.

        Safe to call after a partially-failed ``__init__`` (only unregisters
        whatever was tracked in ``_host_registered``).
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        registered = getattr(self, "_host_registered", None) or []
        worker_id = getattr(self, "worker_id", "-1")
        msg = (
            f"[worker {worker_id}] shutdown: unregistering "
            f"{len(registered)} host region(s)"
        )
        flexkv_logger.info(msg)
        # Drain in-flight CUDA work before unpinning host memory that
        # DMA / kernels may still be touching.
        #
        # torch.cuda.synchronize() releases the GIL and blocks in the driver;
        # if the GPU is wedged (hung kernel, TDR, faulty NVLink) it can hang
        # forever. We run it in a daemon thread with a bounded join so a
        # wedged GPU cannot prevent cudaHostUnregister from firing — the
        # kernel behind the DMA is already dead, so proceeding with unpin
        # is the correct action; the sentinel thread dies with the process.
        self._drain_cuda_bounded(worker_id, timeout_s=30.0)
        # Unregister in reverse order of registration.
        while registered:
            tensor, label = registered.pop()
            safe_cuda_host_unregister(tensor, label=f"worker={worker_id} {label}")
        self._op_buffer_pinned = False
        self._host_registered = registered

    @staticmethod
    def _drain_cuda_bounded(worker_id: Any, timeout_s: float) -> None:
        """Best-effort torch.cuda.synchronize() with a wall-clock cap.

        Returns whether the sync actually completed. Failure / timeout is
        logged but not raised — unpin must proceed either way.
        """
        if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
            return
        done = threading.Event()
        err: List[BaseException] = []

        def _run() -> None:
            try:
                torch.cuda.synchronize()
            except BaseException as e:  # noqa: BLE001
                err.append(e)
            finally:
                done.set()

        t = threading.Thread(
            target=_run,
            name=f"flexkv-worker-{worker_id}-cuda-drain",
            daemon=True,
        )
        t.start()
        if not done.wait(timeout=timeout_s):
            flexkv_logger.warning(
                f"[worker {worker_id}] cuda synchronize did not finish in "
                f"{timeout_s:.0f}s (GPU likely wedged); proceeding with unpin"
            )
            return
        if err:
            flexkv_logger.warning(
                f"[worker {worker_id}] cuda synchronize before unpin failed: "
                f"{err[0]!r}"
            )

    @classmethod
    def _get_worker_id(cls) -> int:
        with cls._worker_id_lock:
            worker_id = cls._worker_id_counter
            cls._worker_id_counter += 1
            return worker_id

    def _get_layer_ptrs(self, layer_blocks: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(layer_blocks, torch.Tensor):
            layer_blocks = [layer_blocks]
        layer_ptrs = torch.zeros(
            len(layer_blocks),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        for lay_id in range(len(layer_blocks)):
            layer_ptrs[lay_id] = layer_blocks[lay_id][0].data_ptr()
        return layer_ptrs

    @staticmethod
    def _get_gpu_strides_from_tensor(
        tensor: torch.Tensor,
        tokens_per_block: int,
        dtype_size: int,
        kv_dim: int,
    ) -> tuple:
        """Compute (kv_stride, block_stride, layer_stride) in bytes from a GPU
        KV cache tensor's actual memory layout.

        Different attention backends use different dim orders for the 5D tensor:
          flash_attn:        [2, num_blocks, block_size, num_kv_heads, head_size]
          triton/flashinfer: [num_blocks, 2, block_size, num_kv_heads, head_size]

        Returns (gpu_kv_stride_bytes, gpu_block_stride_bytes, gpu_layer_stride_bytes).
        """
        if kv_dim == 1 or tensor.ndim != 5:
            return None  # caller should fall back to layout-based strides

        # Last 2 dims are always (num_kv_heads, head_size).
        # First 3 dims are a permutation of (num_blocks, kv_dim=2, block_size).
        dim_sizes = [tensor.shape[i] for i in range(3)]
        kv_dim_idx = None
        block_size_idx = None
        block_dim_idx = None

        # Identify kv_dim (size 2) and block_size (size tokens_per_block)
        for i in range(3):
            if dim_sizes[i] == 2 and kv_dim_idx is None:
                kv_dim_idx = i
        for i in range(3):
            if i != kv_dim_idx and dim_sizes[i] == tokens_per_block and block_size_idx is None:
                block_size_idx = i
        # Remaining dim is num_blocks
        for i in range(3):
            if i != kv_dim_idx and i != block_size_idx:
                block_dim_idx = i
                break

        if kv_dim_idx is None or block_dim_idx is None:
            return None  # ambiguous, fall back

        kv_stride = tensor.stride(kv_dim_idx) * dtype_size
        block_stride = tensor.stride(block_dim_idx) * dtype_size
        layer_stride = tensor.numel() * dtype_size
        return (kv_stride, block_stride, layer_stride)

    @classmethod
    def create_worker(cls,
                      mp_ctx: Any,
                      finished_ops_queue: MPQueue,
                      op_buffer_tensor: torch.Tensor,
                      *args: Any, **kwargs: Any) -> 'WorkerHandle':
        """Generic worker creation template method."""

        parent_conn, child_conn = mp_ctx.Pipe()  # create pipe
        ready_event = mp_ctx.Event()
        worker_id = cls._get_worker_id()

        process = mp_ctx.Process(
            target=cls._worker_process,
            args=(worker_id, child_conn, finished_ops_queue, op_buffer_tensor, ready_event, *args),
            kwargs=kwargs,
            daemon=True
        )
        process.start()

        return WorkerHandle(worker_id, parent_conn, process, ready_event)

    @classmethod
    def _worker_process(cls, worker_id: int, transfer_conn: Connection, finished_ops_queue: MPQueue,
                        op_buffer_tensor: torch.Tensor, ready_event: Any, *args: Any, **kwargs: Any) -> None:
        # Note: MPI initialization prevention is handled by create_safe_process
        # Environment variables are set before this function is called.
        #
        # Use ``__new__`` + ``__init__`` (not ``cls(...)``) so we keep a live
        # reference if ``__init__`` raises after partial cudaHostRegister; the
        # ``finally`` block can still unpin. ``run()`` only exits the loop —
        # this finally owns shutdown().
        worker: Optional["TransferWorkerBase"] = None

        def _on_sigterm(signum: int, frame: Any) -> None:
            # Raise SystemExit so the ``finally`` below still runs shutdown().
            flexkv_logger.warning(
                f"[worker {worker_id}] received signal {signum}; exiting for graceful cleanup"
            )
            raise SystemExit(0)

        try:
            # Ignore Ctrl+C (SIGINT): the foreground process group receives it
            # together with sglang/tee. Workers must only unpin when the parent
            # sends a shutdown sentinel / SIGTERM, otherwise they race and get
            # SIGKILL mid-unregister, leaking pinned CPU buffers.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, _on_sigterm)
        except Exception as e:
            flexkv_logger.warning(
                f"[worker {worker_id}] failed to install shutdown signal handlers: {e}"
            )


        try:
            worker = cls.__new__(cls)
            worker.__init__(
                worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor, *args, **kwargs
            )
            ready_event.set()
            worker.run()
        except Exception as e:
            # Init / run failure: log then re-raise so process exitcode != 0.
            # SIGTERM → SystemExit is BaseException and bypasses this handler,
            # still hitting ``finally`` for unpin.
            flexkv_logger.error(
                f"[worker {worker_id}] exited with error during init/run: {e}"
            )
            raise
        finally:
            if worker is not None:
                try:
                    worker.shutdown()
                except Exception as e:
                    flexkv_logger.error(f"[worker {worker_id}] final shutdown error: {e}")

    @abstractmethod
    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        **kwargs: Any
    ) -> None:
        pass

    def get_transfer_block_ids(self,
                               transfer_op: WorkerTransferOp,
                               pinned: bool = True) ->tuple[torch.Tensor, torch.Tensor]:
        """
        Get transfer block ids from op buffer tensor or directly from op
        Args:
            transfer_op: WorkerTransferOp
            pinned: whether to pin the block ids tensor
        Returns:
            tuple[torch.Tensor, torch.Tensor]: src_block_ids and dst_block_ids
        """
        src_slot_id = transfer_op.src_slot_id
        dst_slot_id = transfer_op.dst_slot_id
        valid_block_num = transfer_op.valid_block_num

        if src_slot_id == -1:
            src_block_ids = torch.from_numpy(transfer_op.src_block_ids).to(dtype=torch.int64)
            if pinned:
                src_block_ids = src_block_ids.pin_memory()
        else:
            src_block_ids = self.op_buffer_tensor[src_slot_id, :valid_block_num]

        if dst_slot_id == -1:
            dst_block_ids = torch.from_numpy(transfer_op.dst_block_ids).to(dtype=torch.int64)
            if pinned:
                dst_block_ids = dst_block_ids.pin_memory()
        else:
            dst_block_ids = self.op_buffer_tensor[dst_slot_id, :valid_block_num]

        return src_block_ids, dst_block_ids

    def _log_transfer_performance(self,
                                  transfer_op: WorkerTransferOp,
                                  transfer_size: int,
                                  start_time: float,
                                  end_time: float,
                                  uncompressed_size: Optional[int] = None) -> None:
        """Emit one terminal record per transfer op."""
        if not flexkv_logger.is_enabled_for(logging.INFO):
            return
        duration_s = max(end_time - start_time, 1e-9)
        is_layerwise = transfer_op.transfer_type == TransferType.LAYERWISE
        direction = "H2D" if is_layerwise else transfer_op.transfer_type.value
        blocks = (
            len(transfer_op.src_block_ids_h2d)
            if is_layerwise
            else transfer_op.valid_block_num
        )
        transfer_mode = "layerwise" if is_layerwise else "no-layerwise"
        bandwidth = transfer_size / duration_s / 1e9

        if (
            uncompressed_size is not None
            and transfer_size > 0
            and uncompressed_size != transfer_size
        ):
            flexkv_logger.info(
                "[FlexKV-IO] operation=transfer act=complete status=success "
                "direction=%s blocks=%d op_id=%d graph_id=%d mode=%s "
                "compressed_size=%.6gGB original_size=%.6gGB "
                "compression_ratio=%.2fx transfer_time=%.4fs "
                "bandwidth=%.2fGB/s",
                direction,
                blocks,
                transfer_op.transfer_op_id,
                transfer_op.transfer_graph_id,
                transfer_mode,
                transfer_size / (1024**3),
                uncompressed_size / (1024**3),
                uncompressed_size / transfer_size,
                duration_s,
                bandwidth,
            )
        else:
            flexkv_logger.info(
                "[FlexKV-IO] operation=transfer act=complete status=success "
                "direction=%s blocks=%d op_id=%d graph_id=%d mode=%s "
                "data_size=%.6gGB transfer_time=%.4fs bandwidth=%.2fGB/s",
                direction,
                blocks,
                transfer_op.transfer_op_id,
                transfer_op.transfer_graph_id,
                transfer_mode,
                transfer_size / (1024**3),
                duration_s,
                bandwidth,
            )

    @abstractmethod
    def launch_transfer(
        self, transfer_op: WorkerTransferOp
    ) -> Union[bool, WorkerTransferResult]:
        pass

    def _handle_control(self, command: str, payload: Any) -> Any:
        handler = getattr(self, f"_control_{command}", None)
        if handler is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support control {command}"
            )
        return handler(payload)

    def _reply_control(self, op: Dict[str, Any]) -> None:
        request_id = op["request_id"]
        try:
            reply = {
                "type": "control_ack",
                "request_id": request_id,
                "result": self._handle_control(
                    op["command"], op.get("payload")
                ),
            }
        except Exception as exc:
            flexkv_logger.exception(
                f"Worker control {op.get('command')} failed"
            )
            reply = {
                "type": "control_ack",
                "request_id": request_id,
                "error": str(exc),
            }
        self.transfer_conn.send(reply)

    def run(self) -> None:
        """Main loop for the worker process.

        Exit paths (``None`` sentinel, pipe EOF, or return) do not unregister
        themselves — ``_worker_process`` owns a single ``shutdown()`` in its
        ``finally`` block so cleanup is not duplicated.
        """
        while True:
            try:
                if not self.transfer_conn.poll(timeout=0.0001):
                    continue

                op = self.transfer_conn.recv()
                if op is None:
                    return
                if not isinstance(op, dict):
                    op._received_ns = time.perf_counter_ns()

                # Drain any already-queued ops into one batch, then process.
                # The for-loop MUST sit outside the drain while: a single-op
                # submit leaves poll() False immediately, and a while-else
                # continue would otherwise drop the first op forever (which
                # stalls D2H → H2REMOTE and leaves mooncake PutStart=0).
                batch_ops = [op]
                stop_after_batch = False
                while self.transfer_conn.poll(timeout=0):
                    try:
                        op = self.transfer_conn.recv()
                    except EOFError:
                        # A closed pipe is readable. Preserve the batch already
                        # received, then exit after reporting its completions.
                        stop_after_batch = True
                        break
                    if op is None:
                        stop_after_batch = True
                        break
                    if not isinstance(op, dict):
                        op._received_ns = time.perf_counter_ns()
                    batch_ops.append(op)
                for op in batch_ops:
                    if isinstance(op, dict) and op.get("type") == "control":
                        self._reply_control(op)
                        continue
                    transfer_status = False
                    transfer_start_ns = time.perf_counter_ns()
                    nvtx_pushed = False
                    try:
                        nvtx.push_range(f"launch {op.transfer_type.name} op_id: {op.transfer_op_id}, "
                                            f"graph_id: {op.transfer_graph_id}",
                                            color=get_nvtx_range_color(op.transfer_graph_id))
                        nvtx_pushed = True
                        transfer_status = self.launch_transfer(op)
                    except Exception as e:
                        is_layerwise = op.transfer_type == TransferType.LAYERWISE
                        direction = "H2D" if is_layerwise else op.transfer_type.value
                        blocks = (
                            len(op.src_block_ids_h2d)
                            if is_layerwise
                            else op.valid_block_num
                        )
                        flexkv_logger.error(
                            "[FlexKV-IO] operation=transfer act=complete "
                            "status=failed direction=%s blocks=%d op_id=%d "
                            "graph_id=%d mode=%s transfer_time=%.4fs "
                            "error=%r",
                            direction,
                            blocks,
                            op.transfer_op_id,
                            op.transfer_graph_id,
                            "layerwise" if is_layerwise else "no-layerwise",
                            (time.perf_counter_ns() - transfer_start_ns) / 1e9,
                            str(e),
                            exc_info=True,
                        )
                    finally:
                        if nvtx_pushed:
                            nvtx.pop_range()
                    launched_ns = time.perf_counter_ns()
                    is_h2d = (op.transfer_type == TransferType.H2D
                              or op.transfer_type == TransferType.LAYERWISE)
                    metrics = trace.build_worker_metrics(
                        op,
                        getattr(op, "prof_submitted_ns", 0),
                        getattr(op, "_received_ns", launched_ns),
                        transfer_start_ns,
                        launched_ns,
                        self.worker_id,
                        getattr(self, "_bytes_per_block", 0),
                        getattr(self, "kv_dim", 2),
                        is_h2d,
                    )
                    if isinstance(transfer_status, WorkerTransferResult):
                        # Partial-capable backends report completion even when
                        # zero blocks succeeded, so the graph can clean up and
                        # the caller can fall back instead of hanging forever.
                        # Carry metrics so FLEXKV_TRANSFER_TRACE still works.
                        self.finished_ops_queue.put(
                            (transfer_status, True, metrics))
                    elif transfer_status:
                        self.finished_ops_queue.put(
                            (op.transfer_op_id, True, metrics))
                    else:
                        # Report the failure instead of dropping it: a
                        # dropped op leaves its graph incomplete forever
                        # and leaks every resource its plan holds. A bare
                        # int still means success, so the queue format
                        # stays compatible.
                        self.finished_ops_queue.put(
                            (op.transfer_op_id, False, None))
                if stop_after_batch:
                    # _worker_process owns the single shutdown() call in its
                    # finally block. Calling subclass shutdown here can repeat
                    # external unregister work before super()'s idempotence
                    # guard is reached.
                    return
            except EOFError:
                flexkv_logger.warning(
                    f"[worker {self.worker_id}] transfer pipe EOF; exiting run loop"
                )
                return
            except Exception as e:
                flexkv_logger.error(f"Error in worker run loop: {e}")

