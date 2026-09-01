"""Bidirectional bandwidth benchmarks using two concurrent worker processes.

Each direction runs in its own worker process with an independent CUDA stream
or io_uring context, so both directions execute concurrently.  The two
directions use disjoint block-id ranges ([0, n) and [n, 2n)) to avoid data
races on the shared KV pools, and the reported bandwidth is the aggregate of
both directions.

Supported links:
    gpu : H2D + D2H          (pinned CPU <-> GPU)
    ssd : H2DISK + DISK2H    (pinned CPU <-> SSD)
    gds : DISK2D + D2DISK    (GPU <-> SSD, requires FLEXKV_ENABLE_GDS=1)

Each link can run in two concurrency modes (--mode):
    process : two worker processes, independent CUDA contexts / io_uring (default)
    thread  : two worker threads in one process (gpu / ssd / gds links)

Examples:
    python benchmark_workers_bi.py --link gpu --num-blocks 64
    python benchmark_workers_bi.py --link gpu --mode thread --use-ce --num-blocks 256
    python benchmark_workers_bi.py --link ssd --num-blocks 64
    python benchmark_workers_bi.py --link ssd --ssd-io-mode direct --num-blocks 64
    python benchmark_workers_bi.py --link ssd --mode thread --ssd-io-mode direct --num-blocks 64
    python benchmark_workers_bi.py --link gds --num-blocks 64
    python benchmark_workers_bi.py --all --num-blocks 64
"""

import time
import threading
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch

from flexkv.common.transfer import TransferOp, TransferType
from flexkv.transfer.worker import (
    CPUSSDDiskTransferWorker,
    GDSTransferWorker,
    GPUCPUTransferWorker,
    WorkerHandle,
    tpGDSTransferWorker,
    tpGPUCPUTransferWorker,
)
from flexkv.storage.allocator import CPUAllocator, GPUAllocator, SSDAllocator
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV, ModelConfig
from utils import load_config

# GDS support is optional (only available when compiled with FLEXKV_ENABLE_GDS=1)
try:
    from flexkv.c_ext import transfer_kv_blocks_gds
except ImportError:
    transfer_kv_blocks_gds = None

# (forward direction, reverse direction) per link
LINK_TYPES = {
    "gpu": (TransferType.H2D, TransferType.D2H),
    "ssd": (TransferType.H2DISK, TransferType.DISK2H),
    "gds": (TransferType.DISK2D, TransferType.D2DISK),
}

WorkerPair = List[Tuple[WorkerHandle, mp.Queue]]


class _BenchmarkThreadProcessAdapter:
    """Expose the process API needed by WorkerHandle for a thread worker."""

    def __init__(self, thread: threading.Thread):
        self.thread = thread

    @property
    def exitcode(self) -> int | None:
        return None if self.thread.is_alive() else 0

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def terminate(self) -> None:
        pass

    kill = terminate


class _BenchmarkWorkerThreadPair:
    """Coordinate shutdown of two benchmark-owned worker threads."""

    def __init__(self):
        self.owner_worker = None
        self._stopped: set[int] = set()
        self._lock = threading.Lock()

    def set_owner_worker(self, worker) -> None:
        self.owner_worker = worker

    def notify_stopped(self, worker_id: int) -> None:
        owner = None
        with self._lock:
            self._stopped.add(worker_id)
            if len(self._stopped) == 2:
                owner = self.owner_worker
                self.owner_worker = None
        if owner is not None:
            owner.shutdown()

    def shutdown_owner_now(self) -> None:
        with self._lock:
            owner = self.owner_worker
            self.owner_worker = None
        if owner is not None:
            owner.shutdown()


class BenchmarkThreadWorkerHandle(WorkerHandle):
    """WorkerHandle backed by a benchmark-owned in-process worker thread."""

    def __init__(
        self,
        worker_id: int,
        transfer_conn,
        thread: threading.Thread,
        ready_event,
        pair: _BenchmarkWorkerThreadPair,
    ):
        super().__init__(
            worker_id,
            transfer_conn,
            _BenchmarkThreadProcessAdapter(thread),
            ready_event,
        )
        self.thread = thread
        self._pair = pair

    def shutdown(self) -> None:
        try:
            self.transfer_conn.send(None)
        except (BrokenPipeError, OSError, EOFError):
            pass

        timeout = float(GLOBAL_CONFIG_FROM_ENV.worker_shutdown_timeout_s)
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            print(
                f"benchmark worker {self.worker_id} did not stop within "
                f"{timeout:.0f}s"
            )
        try:
            self.transfer_conn.close()
        except Exception:
            pass
        self._pair.notify_stopped(self.worker_id)


class _InProcessTensorHandle:
    """Present an already-materialized CUDA tensor to worker.py."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def get_tensor(self) -> torch.Tensor:
        return self.tensor


class BenchmarkGPUCPUTransferWorker(GPUCPUTransferWorker):
    """GPU-CPU worker with benchmark-local shared host registration."""

    def __init__(self, *args, owns_host_registration: bool = True, **kwargs):
        self._owns_host_registration = owns_host_registration
        super().__init__(*args, **kwargs)

    def _register_host_tensor(self, tensor: torch.Tensor, label: str = "") -> None:
        if not self._owns_host_registration:
            print(
                f"[worker {self.worker_id}] sharing host registration for {label}"
            )
            return
        super()._register_host_tensor(tensor, label)


def _launch_benchmark_worker_thread(
    mp_ctx,
    worker_id: int,
    finished_ops_queue,
    op_buffer_tensor,
    target,
    args,
    name: str,
    pair: _BenchmarkWorkerThreadPair,
    handles: list[BenchmarkThreadWorkerHandle],
) -> BenchmarkThreadWorkerHandle:
    parent_conn, child_conn = mp_ctx.Pipe()
    ready_event = mp_ctx.Event()
    thread = threading.Thread(
        target=target,
        args=(
            worker_id,
            child_conn,
            finished_ops_queue,
            op_buffer_tensor,
            ready_event,
            pair,
            *args,
        ),
        name=name,
        daemon=True,
    )
    handle = BenchmarkThreadWorkerHandle(
        worker_id, parent_conn, thread, ready_event, pair
    )
    handles.append(handle)
    thread.start()

    deadline = time.monotonic() + 300.0
    while not ready_event.wait(timeout=0.05):
        if not thread.is_alive():
            pair.shutdown_owner_now()
            for existing_handle in handles:
                existing_handle.shutdown()
            raise RuntimeError(f"benchmark worker {worker_id} died during init")
        if time.monotonic() >= deadline:
            pair.shutdown_owner_now()
            for existing_handle in handles:
                existing_handle.shutdown()
            raise TimeoutError(
                f"benchmark worker {worker_id} did not become ready"
            )
    return handle


@dataclass
class BiBenchmarkConfig:
    link: str = "gpu"
    num_layers_to_transfer: int = -1
    num_blocks_per_direction: int = 16
    warmup_round: int = 1
    benchmark_round: int = 10
    gpu_layout_type: int = 0
    use_ce_transfer: bool = False
    transfer_num_cta: int = 4


def _gpu_layout(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    num_gpu_blocks: int,
    gpu_layout_type: int,
) -> KVCacheLayout:
    if gpu_layout_type in (0, 2):
        layout_type = KVCacheLayoutType.LAYERFIRST
    elif gpu_layout_type == 1:
        layout_type = KVCacheLayoutType.BLOCKFIRST
    else:
        raise ValueError(f"Invalid GPU layout type: {gpu_layout_type}")
    layout = KVCacheLayout(
        type=layout_type,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    return layout.div_head(model_config.tp_size) if model_config.num_kv_heads > 1 else layout


def _num_chunks(model_config: ModelConfig, gpu_layout_type: int) -> int:
    if gpu_layout_type == 0:
        return model_config.num_layers
    if gpu_layout_type == 1:
        return 1
    if gpu_layout_type == 2:
        return model_config.num_layers * 2
    raise ValueError(f"Invalid GPU layout type: {gpu_layout_type}")


def _allocate_gpu_handles(
    model_config: ModelConfig,
    gpu_layout: KVCacheLayout,
    num_chunks: int,
):
    handles = []
    for tp_id in range(model_config.tp_size):
        torch.cuda.set_device(tp_id)
        handles.append(
            GPUAllocator.allocate(
                layout=gpu_layout,
                dtype=model_config.dtype,
                num_chunks=num_chunks,
                device_id=tp_id,
            )
        )
    return handles


def create_cpu_gpu_worker_pair(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    num_blocks_total: int,
    gpu_layout_type: int = 0,
    use_ce_transfer: bool = False,
    transfer_num_cta: int = 4,
) -> WorkerPair:
    """Create two GPUCPU transfer workers sharing one CPU/GPU pool.

    Worker 0 is driven with H2D ops, worker 1 with D2H ops.  Each worker owns
    an independent CUDA stream, so the two directions overlap on PCIe.
    """
    mp.set_start_method("spawn", force=True)
    cpu_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_cpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    gpu_layout = _gpu_layout(model_config, cache_config, num_blocks_total, gpu_layout_type)
    num_chunks = _num_chunks(model_config, gpu_layout_type)

    cpu_handle = CPUAllocator.allocate(layout=cpu_layout, dtype=model_config.dtype, pin_memory=True)
    gpu_handles = _allocate_gpu_handles(model_config, gpu_layout, num_chunks)

    max_block_num = max(1024, cache_config.num_cpu_blocks)
    pairs: WorkerPair = []
    for _ in range(2):
        finished_ops_queue = mp.Queue()
        op_buffer_tensor = torch.empty((4, max_block_num), dtype=torch.int64).share_memory_()
        if model_config.tp_size == 1:
            handle = GPUCPUTransferWorker.create_worker(
                mp_ctx=mp.get_context("spawn"),
                finished_ops_queue=finished_ops_queue,
                op_buffer_tensor=op_buffer_tensor,
                gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                cpu_blocks=cpu_handle.get_tensor(),
                gpu_kv_layout=gpu_handles[0].kv_layout,
                cpu_kv_layout=cpu_handle.kv_layout,
                dtype=model_config.dtype,
                gpu_device_id=0,
                use_ce_transfer_h2d=use_ce_transfer,
                use_ce_transfer_d2h=use_ce_transfer,
                transfer_num_cta_h2d=transfer_num_cta,
                transfer_num_cta_d2h=transfer_num_cta,
            )
        else:
            handle = tpGPUCPUTransferWorker.create_worker(
                mp_ctx=mp.get_context("spawn"),
                finished_ops_queue=finished_ops_queue,
                op_buffer_tensor=op_buffer_tensor,
                gpu_blocks=[h.get_tensor_handle_list() for h in gpu_handles],
                cpu_blocks=cpu_handle.get_tensor(),
                gpu_kv_layouts=[h.kv_layout for h in gpu_handles],
                cpu_kv_layout=cpu_handle.kv_layout,
                dtype=model_config.dtype,
                tp_group_size=model_config.tp_size,
                use_ce_transfer_h2d=use_ce_transfer,
                use_ce_transfer_d2h=use_ce_transfer,
                transfer_num_cta_h2d=transfer_num_cta,
                transfer_num_cta_d2h=transfer_num_cta,
            )
        pairs.append((handle, finished_ops_queue))
    return pairs


def create_cpu_gpu_thread_pair(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    num_blocks_total: int,
    gpu_layout_type: int = 0,
    use_ce_transfer: bool = False,
    transfer_num_cta: int = 4,
) -> WorkerPair:
    """Create H2D/D2H workers as two threads in the current process.

    The benchmark creates both worker threads locally. The first worker owns
    the CUDA host registration for the shared CPU pool and op buffer, while the
    second reuses those mappings. Each worker thread gets its own CUDA stream.

    Both workers share a single finished-ops queue; callers must sync 2x the
    per-direction op count (handled by bench_bidirectional via queue identity).
    Only tp_size=1 is supported (same restriction as the production engine).
    """
    mp.set_start_method("spawn", force=True)
    assert model_config.tp_size == 1, (
        "thread mode currently supports tp_size=1 only"
    )
    cpu_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_cpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    gpu_layout = _gpu_layout(model_config, cache_config, num_blocks_total, gpu_layout_type)
    num_chunks = _num_chunks(model_config, gpu_layout_type)

    cpu_handle = CPUAllocator.allocate(layout=cpu_layout, dtype=model_config.dtype, pin_memory=True)
    gpu_handles = _allocate_gpu_handles(model_config, gpu_layout, num_chunks)

    finished_ops_queue = mp.Queue()
    max_block_num = max(1024, cache_config.num_cpu_blocks)
    op_buffer_tensor = torch.empty((4, max_block_num), dtype=torch.int64).share_memory_()

    mp_ctx = mp.get_context("spawn")
    pair = _BenchmarkWorkerThreadPair()
    handles = []
    gpu_blocks = [
        _InProcessTensorHandle(tensor)
        for tensor in gpu_handles[0].get_tensor_list()
    ]
    for direction in range(2):
        worker_id = BenchmarkGPUCPUTransferWorker._get_worker_id()
        owns_host_registration = direction == 0
        handle = _launch_benchmark_worker_thread(
            mp_ctx,
            worker_id,
            finished_ops_queue,
            op_buffer_tensor,
            _run_cpu_gpu_benchmark_worker,
            (
                owns_host_registration,
                gpu_blocks,
                cpu_handle.get_tensor(),
                gpu_handles[0].kv_layout,
                cpu_handle.kv_layout,
                model_config.dtype,
                use_ce_transfer,
                transfer_num_cta,
            ),
            f"benchmark-gpu-worker-{worker_id}",
            pair,
            handles,
        )
    return [(handles[0], finished_ops_queue), (handles[1], finished_ops_queue)]


def _run_cpu_gpu_benchmark_worker(
    worker_id: int,
    transfer_conn,
    finished_ops_queue,
    op_buffer_tensor,
    ready_event,
    pair: _BenchmarkWorkerThreadPair,
    owns_host_registration: bool,
    gpu_blocks,
    cpu_blocks,
    gpu_kv_layout,
    cpu_kv_layout,
    dtype: torch.dtype,
    use_ce_transfer: bool,
    transfer_num_cta: int,
) -> None:
    worker = None
    try:
        worker = BenchmarkGPUCPUTransferWorker.__new__(
            BenchmarkGPUCPUTransferWorker
        )
        if owns_host_registration:
            pair.set_owner_worker(worker)
        worker.__init__(
            worker_id,
            transfer_conn,
            finished_ops_queue,
            op_buffer_tensor,
            gpu_blocks,
            cpu_blocks,
            gpu_kv_layout,
            cpu_kv_layout,
            dtype,
            0,
            use_ce_transfer_h2d=use_ce_transfer,
            use_ce_transfer_d2h=use_ce_transfer,
            transfer_num_cta_h2d=transfer_num_cta,
            transfer_num_cta_d2h=transfer_num_cta,
            owns_host_registration=owns_host_registration,
        )
        ready_event.set()
        worker.run()
    finally:
        if worker is not None and not owns_host_registration:
            worker.shutdown()


def allocate_process_shared_cpu_tensor(
    layout: KVCacheLayout, dtype: torch.dtype
) -> torch.Tensor:
    """Allocate a CPU pool whose 4K offset survives spawn remapping (O_DIRECT)."""
    element_size = torch.empty((), dtype=dtype).element_size()
    if 4096 % element_size != 0:
        raise ValueError(f"CPU dtype size {element_size} does not divide 4K")

    offset_elements = 4096 // element_size
    total_elements = layout.get_total_elements()
    padded_tensor = torch.empty(
        total_elements + offset_elements, dtype=dtype
    ).share_memory_()
    cpu_tensor = padded_tensor.narrow(0, offset_elements, total_elements)
    byte_offset = cpu_tensor.storage_offset() * element_size
    assert byte_offset % 4096 == 0
    return cpu_tensor


def create_cpu_ssd_worker_pair(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    ssd_force_direct: bool = False,
) -> WorkerPair:
    """Create two CPU<->SSD transfer workers sharing one CPU pool and SSD files."""
    mp.set_start_method("spawn", force=True)
    cpu_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_cpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    ssd_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.ssd_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_ssd_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    if ssd_force_direct:
        cpu_tensor = allocate_process_shared_cpu_tensor(
            cpu_layout, model_config.dtype
        )
        cpu_layout_handle = cpu_layout
        print(
            "[BENCH-BI] SSD process workers use a fixed 4K storage offset "
            "to preserve O_DIRECT after tensor remapping"
        )
    else:
        cpu_handle = CPUAllocator.allocate(
            layout=cpu_layout, dtype=model_config.dtype, pin_memory=True
        )
        cpu_tensor = cpu_handle.get_tensor()
        cpu_layout_handle = cpu_handle.kv_layout
    ssd_handle = SSDAllocator.allocate(
        layout=ssd_layout,
        dtype=model_config.dtype,
        num_chunks=model_config.num_layers,
        cache_dir=cache_config.ssd_cache_dir,
        max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb,
    )

    max_block_num = max(1024, cache_config.num_cpu_blocks)
    pairs: WorkerPair = []
    for _ in range(2):
        finished_ops_queue = mp.Queue()
        op_buffer_tensor = torch.empty((4, max_block_num), dtype=torch.int64).share_memory_()
        handle = CPUSSDDiskTransferWorker.create_worker(
            mp_ctx=mp.get_context("spawn"),
            finished_ops_queue=finished_ops_queue,
            op_buffer_tensor=op_buffer_tensor,
                cpu_blocks=cpu_tensor,
                ssd_files=ssd_handle.get_file_list(),
                cpu_kv_layout=cpu_layout_handle,
            ssd_kv_layout=ssd_handle.kv_layout,
            dtype=model_config.dtype,
            num_blocks_per_file=ssd_handle.num_blocks_per_file,
            cache_config=cache_config,
        )
        pairs.append((handle, finished_ops_queue))
    return pairs


def create_cpu_ssd_thread_pair(
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> WorkerPair:
    """Create H2DISK/DISK2H workers as two threads in the current process.

    Each thread worker owns an independent SSDIOCTX (io_uring context) over
    the same SSD files, so the two directions issue concurrent reads/writes
    without sharing ioctx state.  Both workers share one finished-ops queue;
    bench_bidirectional syncs 2x the per-direction op count via queue identity.
    """
    mp.set_start_method("spawn", force=True)
    cpu_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_cpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    ssd_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.ssd_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_ssd_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    cpu_handle = CPUAllocator.allocate(layout=cpu_layout, dtype=model_config.dtype, pin_memory=True)
    ssd_handle = SSDAllocator.allocate(
        layout=ssd_layout,
        dtype=model_config.dtype,
        num_chunks=model_config.num_layers,
        cache_dir=cache_config.ssd_cache_dir,
        max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb,
    )

    finished_ops_queue = mp.Queue()
    max_block_num = max(1024, cache_config.num_cpu_blocks)
    mp_ctx = mp.get_context("spawn")
    pair = _BenchmarkWorkerThreadPair()
    handles = []
    for direction in range(2):
        worker_id = CPUSSDDiskTransferWorker._get_worker_id()
        op_buffer_tensor = torch.empty(
            (4, max_block_num), dtype=torch.int64
        ).share_memory_()
        handle = _launch_benchmark_worker_thread(
            mp_ctx,
            worker_id,
            finished_ops_queue,
            op_buffer_tensor,
            _run_cpu_ssd_benchmark_worker,
            (
                direction == 0,
                cpu_handle.get_tensor(),
                ssd_handle.get_file_list(),
                cpu_handle.kv_layout,
                ssd_handle.kv_layout,
                model_config.dtype,
                ssd_handle.num_blocks_per_file,
                cache_config,
            ),
            f"benchmark-ssd-worker-{worker_id}",
            pair,
            handles,
        )
    return [(handles[0], finished_ops_queue), (handles[1], finished_ops_queue)]


def _run_cpu_ssd_benchmark_worker(
    worker_id: int,
    transfer_conn,
    finished_ops_queue,
    op_buffer_tensor,
    ready_event,
    pair: _BenchmarkWorkerThreadPair,
    is_owner: bool,
    cpu_blocks,
    ssd_files,
    cpu_kv_layout,
    ssd_kv_layout,
    dtype: torch.dtype,
    num_blocks_per_file: int,
    cache_config,
) -> None:
    worker = None
    try:
        worker = CPUSSDDiskTransferWorker.__new__(CPUSSDDiskTransferWorker)
        if is_owner:
            pair.set_owner_worker(worker)
        worker.__init__(
            worker_id,
            transfer_conn,
            finished_ops_queue,
            op_buffer_tensor,
            cpu_blocks,
            ssd_files,
            cpu_kv_layout,
            ssd_kv_layout,
            dtype,
            num_blocks_per_file,
            cache_config,
        )
        ready_event.set()
        worker.run()
    finally:
        if worker is not None and not is_owner:
            worker.shutdown()


def create_gpu_ssd_worker_pair(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    num_blocks_total: int,
    gpu_layout_type: int = 0,
) -> WorkerPair:
    """Create two GDS transfer workers sharing one GPU pool and SSD files."""
    mp.set_start_method("spawn", force=True)
    gpu_layout = _gpu_layout(model_config, cache_config, num_blocks_total, gpu_layout_type)
    num_chunks = _num_chunks(model_config, gpu_layout_type)
    ssd_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.ssd_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_ssd_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    gpu_handles = _allocate_gpu_handles(model_config, gpu_layout, num_chunks)
    ssd_handle = SSDAllocator.allocate(
        layout=ssd_layout,
        dtype=model_config.dtype,
        num_chunks=model_config.num_layers,
        cache_dir=cache_config.ssd_cache_dir,
        max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb,
    )

    max_block_num = max(1024, cache_config.num_ssd_blocks)
    pairs: WorkerPair = []
    for _ in range(2):
        finished_ops_queue = mp.Queue()
        op_buffer_tensor = torch.empty((4, max_block_num), dtype=torch.int64).share_memory_()
        if model_config.tp_size == 1:
            handle = GDSTransferWorker.create_worker(
                mp_ctx=mp.get_context("spawn"),
                finished_ops_queue=finished_ops_queue,
                op_buffer_tensor=op_buffer_tensor,
                gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                ssd_files=ssd_handle.get_file_list(),
                num_blocks_per_file=ssd_handle.num_blocks_per_file,
                gpu_kv_layout=gpu_handles[0].kv_layout,
                ssd_kv_layout=ssd_handle.kv_layout,
                dtype=model_config.dtype,
                gpu_device_id=0,
            )
        else:
            handle = tpGDSTransferWorker.create_worker(
                mp_ctx=mp.get_context("spawn"),
                finished_ops_queue=finished_ops_queue,
                op_buffer_tensor=op_buffer_tensor,
                gpu_blocks=[h.get_tensor_handle_list() for h in gpu_handles],
                ssd_files=ssd_handle.get_file_list(),
                num_blocks_per_file=ssd_handle.num_blocks_per_file,
                gpu_kv_layouts=[h.kv_layout for h in gpu_handles],
                ssd_kv_layout=ssd_handle.kv_layout,
                dtype=model_config.dtype,
                tp_group_size=model_config.tp_size,
            )
        pairs.append((handle, finished_ops_queue))
    return pairs


def _run_gds_benchmark_worker(
    worker_id: int,
    transfer_conn,
    finished_ops_queue: mp.Queue,
    op_buffer_tensor: torch.Tensor,
    ready_event,
    pair: _BenchmarkWorkerThreadPair,
    is_owner: bool,
    gpu_blocks,
    ssd_files,
    num_blocks_per_file: int,
    gpu_kv_layout,
    ssd_kv_layout,
    dtype: torch.dtype,
) -> None:
    """Run one benchmark-owned GDS worker in the current process."""
    worker = None
    try:
        worker = GDSTransferWorker.__new__(GDSTransferWorker)
        if is_owner:
            pair.set_owner_worker(worker)
        worker.__init__(
            worker_id,
            transfer_conn,
            finished_ops_queue,
            op_buffer_tensor,
            gpu_blocks=gpu_blocks,
            ssd_files=ssd_files,
            num_blocks_per_file=num_blocks_per_file,
            gpu_kv_layout=gpu_kv_layout,
            ssd_kv_layout=ssd_kv_layout,
            dtype=dtype,
            gpu_device_id=0,
        )
        ready_event.set()
        worker.run()
    finally:
        if worker is not None and not is_owner:
            worker.shutdown()


def create_gpu_ssd_thread_pair(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    num_blocks_total: int,
    gpu_layout_type: int = 0,
) -> WorkerPair:
    """Create DISK2D/D2DISK workers as two threads in the current process."""
    mp.set_start_method("spawn", force=True)
    assert model_config.tp_size == 1, (
        "GDS thread mode currently supports tp_size=1 only"
    )

    gpu_layout = _gpu_layout(model_config, cache_config, num_blocks_total, gpu_layout_type)
    num_chunks = _num_chunks(model_config, gpu_layout_type)
    ssd_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.ssd_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_ssd_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    gpu_handles = _allocate_gpu_handles(model_config, gpu_layout, num_chunks)
    ssd_handle = SSDAllocator.allocate(
        layout=ssd_layout,
        dtype=model_config.dtype,
        num_chunks=model_config.num_layers,
        cache_dir=cache_config.ssd_cache_dir,
        max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb,
    )

    finished_ops_queue = mp.Queue()
    max_block_num = max(1024, cache_config.num_ssd_blocks)
    mp_ctx = mp.get_context("spawn")
    pair = _BenchmarkWorkerThreadPair()
    handles = []
    gpu_blocks = [
        _InProcessTensorHandle(tensor)
        for tensor in gpu_handles[0].get_tensor_list()
    ]
    for direction in range(2):
        worker_id = GDSTransferWorker._get_worker_id()
        op_buffer_tensor = torch.empty(
            (4, max_block_num), dtype=torch.int64
        ).share_memory_()
        _launch_benchmark_worker_thread(
            mp_ctx,
            worker_id,
            finished_ops_queue,
            op_buffer_tensor,
            _run_gds_benchmark_worker,
            (
                direction == 0,
                gpu_blocks,
                ssd_handle.get_file_list(),
                ssd_handle.num_blocks_per_file,
                gpu_handles[0].kv_layout,
                ssd_handle.kv_layout,
                model_config.dtype,
            ),
            f"benchmark-gds-worker-{worker_id}",
            pair,
            handles,
        )

    disk2d_handle, d2disk_handle = handles
    return [(disk2d_handle, finished_ops_queue), (d2disk_handle, finished_ops_queue)]


def launch_transfer(worker_handle: WorkerHandle, transfer_op: TransferOp) -> None:
    worker_handle.submit_transfer(transfer_op)


def sync_all(finished_ops_queue: mp.Queue, num_ops: int) -> None:
    for _ in range(num_ops):
        finished_ops_queue.get()


def _make_op(transfer_type: TransferType, block_ids: np.ndarray) -> TransferOp:
    return TransferOp(
        transfer_type=transfer_type,
        src_block_ids=block_ids,
        dst_block_ids=block_ids,
        graph_id=0,
        dp_client_id=0,
        successors=[],
        predecessors=[],
    )


def bench_bidirectional(args) -> list:
    link = args.link
    forward_type, reverse_type = LINK_TYPES[link]

    model_config, cache_config = load_config(args.config)
    if link == "gpu":
        cache_config.enable_ssd = False
    else:
        assert cache_config.enable_ssd, f"SSD cache must be enabled for {link} benchmark"

    if args.sweep_blocks:
        block_counts = [int(x.strip()) for x in args.sweep_blocks.split(",")]
        print(f"Sweep mode: {len(block_counts)} block counts (per direction): {block_counts}")
    else:
        block_counts = [args.num_blocks]

    max_blocks_per_dir = max(block_counts)
    # Two directions use disjoint block ranges [0, n) and [n, 2n).
    num_blocks_total = 2 * max_blocks_per_dir
    cache_config.num_ssd_blocks = max(cache_config.num_ssd_blocks, num_blocks_total)
    assert cache_config.num_cpu_blocks >= num_blocks_total, (
        f"CPU pool too small for bidirectional benchmark: "
        f"{cache_config.num_cpu_blocks} blocks < {num_blocks_total} required"
    )

    mode = getattr(args, "mode", "process")
    if link == "gpu":
        if mode == "thread":
            worker_pair = create_cpu_gpu_thread_pair(
                model_config,
                cache_config,
                num_blocks_total,
                args.gpu_layout_type,
                args.use_ce,
                args.cta,
            )
        else:
            worker_pair = create_cpu_gpu_worker_pair(
                model_config,
                cache_config,
                num_blocks_total,
                args.gpu_layout_type,
                args.use_ce,
                args.cta,
            )
    elif link == "ssd":
        ssd_force_direct = getattr(args, "ssd_io_mode", "auto") == "direct"
        if mode == "thread":
            worker_pair = create_cpu_ssd_thread_pair(
                model_config, cache_config
            )
        else:
            worker_pair = create_cpu_ssd_worker_pair(
                model_config, cache_config, ssd_force_direct
            )
    elif link == "gds":
        if transfer_kv_blocks_gds is None:
            print("[BENCH-BI] GDS not compiled, skipping gds link")
            return []
        if mode == "thread":
            worker_pair = create_gpu_ssd_thread_pair(
                model_config, cache_config, num_blocks_total, args.gpu_layout_type
            )
        else:
            worker_pair = create_gpu_ssd_worker_pair(
                model_config, cache_config, num_blocks_total, args.gpu_layout_type
            )
    else:
        raise ValueError(f"Unsupported link: {link}")

    (fwd_handle, fwd_queue), (rev_handle, rev_queue) = worker_pair
    # Thread mode shares one finished-ops queue between both workers.
    shared_queue = fwd_queue is rev_queue

    num_layers = (
        args.num_layers if args.num_layers != -1 else model_config.num_layers
    )
    results = []

    for num_blocks_per_dir in block_counts:
        ids_fwd = np.arange(0, num_blocks_per_dir, dtype=np.int64)
        ids_rev = np.arange(num_blocks_per_dir, 2 * num_blocks_per_dir, dtype=np.int64)
        op_fwd = _make_op(forward_type, ids_fwd)
        op_rev = _make_op(reverse_type, ids_rev)

        # Prefill SSD-backed directions so reads hit real data instead of
        # sparse holes (which would over-report read bandwidth).
        if link == "ssd":
            # DISK2H (reverse) reads SSD: prefill reverse blocks via H2DISK.
            launch_transfer(rev_handle, _make_op(TransferType.H2DISK, ids_rev))
            sync_all(rev_queue, 1)
        elif link == "gds":
            # DISK2D (forward) reads SSD: prefill forward blocks via D2DISK.
            launch_transfer(fwd_handle, _make_op(TransferType.D2DISK, ids_fwd))
            sync_all(fwd_queue, 1)

        for _ in range(args.warmup_round):
            launch_transfer(fwd_handle, op_fwd)
            launch_transfer(rev_handle, op_rev)
        if shared_queue:
            sync_all(fwd_queue, 2 * args.warmup_round)
        else:
            sync_all(fwd_queue, args.warmup_round)
            sync_all(rev_queue, args.warmup_round)

        desc = (
            f"Blocks={num_blocks_per_dir}/dir"
            if len(block_counts) > 1
            else "Benchmarking (bi)"
        )
        pbar = tqdm(total=args.benchmark_round, desc=desc)
        start_time = time.time()
        for _ in range(args.benchmark_round):
            launch_transfer(fwd_handle, op_fwd)
            launch_transfer(rev_handle, op_rev)
            pbar.update(1)
        pbar.close()
        if shared_queue:
            sync_all(fwd_queue, 2 * args.benchmark_round)
        else:
            sync_all(fwd_queue, args.benchmark_round)
            sync_all(rev_queue, args.benchmark_round)
        end_time = time.time()

        total_data_size_GB = (
            2  # both directions
            * num_blocks_per_dir
            * cache_config.tokens_per_block
            * model_config.token_size_in_bytes
            * num_layers
            / (model_config.num_layers * 1024 * 1024 * 1024)
        )
        avg_time = (end_time - start_time) / args.benchmark_round
        bw = total_data_size_GB / avg_time
        results.append(
            {
                "num_blocks_per_direction": num_blocks_per_dir,
                "total_gb": total_data_size_GB,
                "avg_time_s": avg_time,
                "bw_gbps": bw,
            }
        )
        if len(block_counts) == 1:
            print(f"Link: {link} ({forward_type.name} + {reverse_type.name}) | mode={mode}")
            if link == "ssd":
                print(f"SSD I/O mode: {getattr(args, 'ssd_io_mode', 'auto')}")
            print(f"Blocks per direction: {num_blocks_per_dir}")
            print(f"Total data size (both directions): {total_data_size_GB:.2f} GB")
            print(f"Avg Time taken: {avg_time:.6f} seconds")
            print(f"Avg Aggregate Bandwidth: {bw:.2f} GB/s")
        else:
            print(
                f"  -> {total_data_size_GB:.2f} GB | "
                f"{avg_time * 1000:.3f} ms | {bw:.2f} GB/s"
            )

    fwd_handle.shutdown()
    rev_handle.shutdown()

    if len(block_counts) > 1:
        print("\n" + "=" * 70)
        print(f"  Bidirectional Sweep Summary: {link} | "
              f"CE={'on' if args.use_ce else 'off'} | CTA={args.cta}")
        print("=" * 70)
        hdr = "{:>14s}  {:>10s}  {:>12s}  {:>12s}".format(
            "Blocks/dir", "Total GB", "Avg ms", "BW GB/s"
        )
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for r in results:
            print(
                "  {:>14d}  {:>10.2f}  {:>12.3f}  {:>12.2f}".format(
                    r["num_blocks_per_direction"],
                    r["total_gb"],
                    r["avg_time_s"] * 1000,
                    r["bw_gbps"],
                )
            )

    return results


def parse_args():
    parser = ArgumentParser(
        description="Bidirectional KV transfer bandwidth benchmark (two worker processes)."
    )
    parser.add_argument("--link", type=str, default="gpu", choices=list(LINK_TYPES.keys()),
                        help="transfer link pair to benchmark")
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--num-blocks", type=int, default=16,
                        help="blocks per direction (pool must hold 2x this)")
    parser.add_argument("--config", type=str, default="./example_config.yml")
    parser.add_argument("--warmup-round", type=int, default=1)
    parser.add_argument("--benchmark-round", type=int, default=10)
    parser.add_argument("--gpu-layout-type", type=int, default=0, choices=[0, 1, 2],
                        help="GPU KV cache layout type")
    parser.add_argument("--use-ce", action="store_true",
                        help="Use CE (cudaMemcpyAsync) transfer path instead of CUDA kernel (gpu link)")
    parser.add_argument("--cta", type=int, default=4,
                        help="transfer_num_cta for kernel path (gpu link, default: 4)")
    parser.add_argument("--sweep-blocks", type=str, default=None,
                        help="Comma-separated block counts per direction (e.g. '64,128,256')")
    parser.add_argument("--mode", type=str, default="process", choices=["process", "thread"],
                        help="concurrency mode: two processes (default) or two threads (gpu/ssd links)")
    parser.add_argument("--ssd-io-mode", type=str, default="auto", choices=["auto", "direct"],
                        help="ssd CPU pool: auto is the default spawn mapping; "
                             "direct uses a 4K storage offset so process workers "
                             "keep O_DIRECT. Thread workers are already aligned, "
                             "so direct is accepted and equivalent")
    parser.add_argument("--all", action="store_true",
                        help="run process+thread for gpu, ssd auto, ssd direct, and gds")
    return parser.parse_args()


def _all_cases():
    cases = []
    for mode in ("process", "thread"):
        cases.append(("gpu", mode, "auto"))
        cases.append(("ssd", mode, "auto"))
        cases.append(("ssd", mode, "direct"))
        cases.append(("gds", mode, "auto"))
    return cases


def _case_label(link: str, mode: str, ssd_io_mode: str) -> str:
    if link == "ssd":
        return f"{link}/{mode}/{ssd_io_mode}"
    return f"{link}/{mode}"


if __name__ == "__main__":
    args = parse_args()
    if getattr(args, "all", False):
        summary = []
        for link, mode, ssd_io_mode in _all_cases():
            label = _case_label(link, mode, ssd_io_mode)
            if link == "gds" and transfer_kv_blocks_gds is None:
                print(f"\n{'=' * 70}\n[--all] SKIPPED: {label} (GDS not compiled)\n{'=' * 70}")
                summary.append((label, link, mode, ssd_io_mode, "SKIPPED"))
                continue
            args.link = link
            args.mode = mode
            args.ssd_io_mode = ssd_io_mode
            print(f"\n{'=' * 70}\n[--all] running {label}\n{'=' * 70}")
            try:
                results = bench_bidirectional(args)
                last = results[-1] if results else None
                summary.append((label, link, mode, ssd_io_mode, last))
            except Exception as exc:
                import traceback
                print(f"[--all] {label} FAILED: {exc}")
                traceback.print_exc()
                summary.append((label, link, mode, ssd_io_mode, None))
        print(f"\n{'=' * 70}\n[--all] Bidirectional Summary\n{'=' * 70}")
        hdr = "{:>22s}  {:>8s}  {:>8s}  {:>10s}  {:>10s}  {:>12s}  {:>12s}".format(
            "Case", "Link", "Mode", "SSD I/O", "Blocks/dir", "Avg ms", "BW GB/s"
        )
        print(hdr)
        print("-" * len(hdr))
        for label, link, mode, ssd_io_mode, r in summary:
            io_mode = ssd_io_mode if link == "ssd" else "-"
            if r == "SKIPPED":
                print("{:>22s}  {:>8s}  {:>8s}  {:>10s}  {:>10s}  {:>12s}  {:>12s}".format(
                    label, link, mode, io_mode, "-", "-", "SKIPPED"))
            elif r is not None:
                print("{:>22s}  {:>8s}  {:>8s}  {:>10s}  {:>10d}  {:>12.3f}  {:>12.2f}".format(
                    label, link, mode, io_mode,
                    r["num_blocks_per_direction"],
                    r["avg_time_s"] * 1000, r["bw_gbps"]))
            else:
                print("{:>22s}  {:>8s}  {:>8s}  {:>10s}  {:>10s}  {:>12s}  {:>12s}".format(
                    label, link, mode, io_mode, "-", "-", "FAILED"))
    else:
        bench_bidirectional(args)
