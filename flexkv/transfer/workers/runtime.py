"""Process-side runtime every transfer worker shares.

``TransferWorkerBase`` owns the parts that are the same whatever edge a worker
serves: the child-process entry point, the receive/batch/report loop, host
pinning and its paired unregister, the bounded CUDA drain on shutdown, block-id
fetch from the shared op buffer, backend attach/run, the perf record, and
control-message dispatch.

Concrete workers live in sibling modules (``gpu_cpu``, ``cpu_ssd``, ``remote``,
``gds``, ``peer``) and this module must not import them: ``create_worker`` is a
classmethod on the base, so the concrete class is always the one the caller
already holds.
"""

import gc
import logging
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Tuple, Union

import nvtx
import torch
from torch.multiprocessing import Queue as MPQueue

from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import (
    TensorSharedHandle,
    close_all_cuda_ipc_handles,
)
from flexkv.common.transfer import TransferType, get_nvtx_range_color
from flexkv.transfer import trace
from flexkv.transfer.backends import StorageBackend
from flexkv.transfer.host_buffer import (
    cudaHostRegister,
    safe_cuda_host_unregister,
)
from flexkv.transfer.template import gpu_strides_from_tensor
from flexkv.transfer.worker_op import WorkerTransferOp, WorkerTransferResult
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


# Idle wait in the worker's receive loop. Connection.poll() wakes as soon as
# the pipe is readable, so this does not add dispatch latency; it only bounds
# how long a worker sleeps between interruptibility checks. The previous value
# (0.1 ms) made every idle worker burn a full core.
_WORKER_IDLE_POLL_S = float(os.getenv("FLEXKV_WORKER_IDLE_POLL_S", "0.05"))


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
        # Not MPQueue[int]: a partial-capable worker reports a
        # WorkerTransferResult in the op-id slot instead of a bare int.
        self.finished_ops_queue: MPQueue = finished_ops_queue

        self.op_buffer_tensor = op_buffer_tensor
        self._op_buffer_pinned = False
        # (tensor, label) pairs registered via _register_host_tensor / _pin_op_buffer.
        self._host_registered: List[Tuple[torch.Tensor, str]] = []
        self._shutdown_done = False
        # Pluggable I/O engine for this worker's edge; None = the worker's own
        # native ``_transfer_impl``. See flexkv/transfer/backends.py.
        self._backend: Optional[StorageBackend] = None

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
        # Backends release first: mooncake unregisters the very buffer the
        # loop below is about to cudaHostUnregister, and NIXL deregisters
        # memory it handed to the agent.
        backend = getattr(self, "_backend", None)
        if backend is not None:
            try:
                backend.shutdown()
            except Exception as e:  # noqa: BLE001
                flexkv_logger.error(
                    f"[worker {getattr(self, 'worker_id', -1)}] backend "
                    f"{backend.name} shutdown failed: {e}"
                )
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

        # Drop every tensor that aliases an imported IPC mapping *before*
        # closing the mappings — the tensors carry raw device pointers, so
        # closing while they are still reachable leaves dangling pointers.
        self._release_imported_gpu_tensors()
        close_all_cuda_ipc_handles()

    def _release_imported_gpu_tensors(self) -> None:
        """Drop references to tensors backed by CUDA IPC mappings.

        Subclasses store these under a few different names; clear whichever
        exist. Called only from ``shutdown()``, after the CUDA drain.
        """
        for attr in ("gpu_blocks", "_multi_group_gpu_blocks_keepalive"):
            if getattr(self, attr, None) is not None:
                try:
                    setattr(self, attr, [])
                except Exception:  # noqa: BLE001
                    pass
        # GPUCPUTransferWorker keeps them per pool instead, in a registry keyed
        # by PoolId, so one worker can hold several without a name per pool.
        # getattr-guarded because the other worker classes have no ``_pools``.
        for pool in getattr(self, "_pools", {}).values():
            try:
                pool.keepalive = []
            except Exception:  # noqa: BLE001
                pass
        gc.collect()

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

        Returns (gpu_kv_stride_bytes, gpu_block_stride_bytes, gpu_layer_stride_bytes),
        or None when the dim order cannot be recovered and the caller should
        fall back to the declared layout.

        Kept as a method only so the single-group call sites below read the
        same as before; the formula itself lives in transfer.template so the
        multi-group compiler and these paths cannot drift apart.
        """
        return gpu_strides_from_tensor(tensor, tokens_per_block, dtype_size, kv_dim)

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

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        **kwargs: Any
    ) -> None:
        """The worker's *native* engine.

        Not abstract: a worker whose edge is served entirely by a pluggable
        ``StorageBackend`` (CPU<->Remote, whose engine is PCFS or
        mooncake-store) has no native engine to implement, and an abstract
        method would force it to write a stub that only raises.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no native transfer engine; it moves "
            f"bytes through a StorageBackend (see backends.py)"
        )

    def _attach_backend(self, backend: Optional["StorageBackend"]) -> None:
        """Bind a pluggable I/O engine, if this worker was given one.

        Call at the *end* of ``__init__``: the backend reads the edge geometry
        the worker just derived, and may open sessions that assume the CUDA
        device is already bound.
        """
        self._backend = backend
        if backend is not None:
            backend.attach(self)

    def _run_backend(
        self, transfer_op: WorkerTransferOp
    ) -> Union[bool, WorkerTransferResult]:
        """One timed backend transfer plus its perf record.

        The backend returns the byte count, which is what removes the
        per-engine ``transfer_size`` formula from every ``launch_transfer``.

        A backend that ``reports_block_results`` returns per-block outcomes
        alongside the byte count; those become a ``WorkerTransferResult`` so
        the engine can keep the blocks that landed instead of failing the
        whole op. Everything else stays on the plain bool path.
        """
        backend = self._backend
        assert backend is not None
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(
            transfer_op, pinned=backend.needs_pinned_block_ids)
        start_time = time.time()
        if backend.reports_block_results:
            block_results, transfer_size = backend.transfer_blocks(
                self, transfer_op, src_block_ids, dst_block_ids)
        else:
            block_results = None
            transfer_size = backend.transfer(
                self, transfer_op, src_block_ids, dst_block_ids)
        end_time = time.time()
        if block_results is None:
            # Bool path: a throw here propagates, exactly as it did when every
            # launch_transfer logged for itself. The run loop turns it into a
            # failed completion.
            self._log_transfer_performance(
                transfer_op, transfer_size, start_time, end_time)
            return True
        # Block-results path: the op has already completed one way or another,
        # so a logging fault must not be raised past here -- an op that never
        # reports hangs its graph and leaks every cache block the plan holds.
        # It does still fail the op closed, because a worker that cannot even
        # record what it did is not one whose success we should believe.
        try:
            self._log_transfer_performance(
                transfer_op, transfer_size, start_time, end_time)
        except Exception:
            flexkv_logger.error(
                f"[worker {self.worker_id}] transfer performance logging "
                "failed; reporting the operation unsuccessful for "
                f"op_id={transfer_op.transfer_op_id}",
                exc_info=True,
            )
            block_results = (False,) * len(block_results)
        return WorkerTransferResult(
            transfer_op_id=transfer_op.transfer_op_id,
            block_results=block_results,
        )

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
                # Blocking poll: it returns the instant the pipe becomes
                # readable, so a long timeout costs no dispatch latency, it
                # only stops this loop from spinning a full core while idle.
                # The timeout exists solely so the loop stays interruptible.
                if not self.transfer_conn.poll(timeout=_WORKER_IDLE_POLL_S):
                    continue

                op = self.transfer_conn.recv()
                if op is None:
                    return
                if not isinstance(op, dict):
                    op._received_ns = time.perf_counter_ns()

                batch_ops = [op]
                stop_after_batch = False
                drain_failed = False
                while self.transfer_conn.poll(timeout=0):
                    # A failure while draining must not discard the ops
                    # already received: fall through and process the batch,
                    # then let the outer handler deal with the pipe.
                    try:
                        op = self.transfer_conn.recv()
                    except EOFError:
                        # A closed pipe is readable, so poll() says yes and
                        # recv() raises. That is the parent going away, not a
                        # fault: preserve the batch already received and exit
                        # after reporting its completions.
                        stop_after_batch = True
                        break
                    except Exception as e:  # noqa: BLE001
                        flexkv_logger.error(
                            f"[worker {self.worker_id}] recv failed while "
                            f"draining batch ({len(batch_ops)} op(s) pending): {e}"
                        )
                        drain_failed = True
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
                    # Metrics are diagnostics: they must never decide whether
                    # a result is reported. Before this guard, a throw here
                    # (or anywhere below) escaped to the outer handler and
                    # dropped the results of *every remaining op in the
                    # batch*, hanging their graphs forever.
                    metrics = None
                    try:
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
                    except Exception as e:  # noqa: BLE001
                        flexkv_logger.error(
                            f"[worker {self.worker_id}] build_worker_metrics "
                            f"failed for op {op.transfer_op_id}: {e}",
                            exc_info=True,
                        )
                    # Report the outcome, success or failure: a dropped op
                    # leaves its graph incomplete forever and leaks every
                    # resource its plan holds. A bare int still means success,
                    # so the queue format stays compatible.
                    try:
                        if isinstance(transfer_status, WorkerTransferResult):
                            # Partial-capable backends report completion even
                            # when zero blocks succeeded, so the graph can
                            # clean up and the caller can fall back rather
                            # than hang. Carry metrics so the trace still
                            # sees the op.
                            self.finished_ops_queue.put(
                                (transfer_status, True, metrics))
                        elif transfer_status:
                            self.finished_ops_queue.put(
                                (op.transfer_op_id, True, metrics))
                        else:
                            self.finished_ops_queue.put(
                                (op.transfer_op_id, False, None))
                    except Exception as e:  # noqa: BLE001
                        # Queue put failing is unrecoverable for this op, but
                        # must not take the rest of the batch down with it.
                        flexkv_logger.error(
                            f"[worker {self.worker_id}] failed to report result "
                            f"for op {op.transfer_op_id}: {e}",
                            exc_info=True,
                        )
                if drain_failed:
                    flexkv_logger.error(
                        f"[worker {self.worker_id}] transfer pipe unusable "
                        f"after batch drain failure; exiting run loop"
                    )
                    return
                if stop_after_batch:
                    # No shutdown() here: ``_worker_process`` owns the single
                    # call in its finally block. Calling it from the loop as
                    # well repeats a subclass's external unregister work before
                    # super()'s idempotence guard is reached.
                    return
            except EOFError:
                flexkv_logger.warning(
                    f"[worker {self.worker_id}] transfer pipe EOF; exiting run loop"
                )
                return
            except Exception as e:
                # Reaching here means the failure happened outside the
                # per-op guards (i.e. in poll/recv of the first op), so no
                # accepted op is silently dropped.
                flexkv_logger.error(
                    f"Error in worker run loop: {e}", exc_info=True
                )
