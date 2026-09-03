"""Storage backends: *how* bytes move on an edge the worker already describes.

A transfer worker owns an **edge**: which two pools, which block ids index
them, and what the per-(layer, kv) byte strides are on each side.  A
**backend** owns the engine that actually moves the bytes across that edge:
io_uring, cuFile/GDS, PCFS, NIXL, a mooncake-store key/value client.

Before this module the two were fused, so a new engine meant a new
``TransferWorkerBase`` subclass -- a new process class, a new
``launch_transfer``, a new copy of the stride derivation, a new
``transfer_size`` formula and a new branch in ``TransferEngine._init_workers``.
Two of those existed:

  NixlTransferWorker           GPU<->SSD (GDS_MT) and CPU<->SSD (POSIX/3FS)
  MooncakeStoreTransferWorker  CPU<->Remote

Both described edges that already had a worker.  ``NixlTransferWorker``
recomputed, from the same layouts, the SSD strides ``GDSTransferWorker`` had
already computed one screen above it -- and got a *different* answer on the
GPU side, because it read the declared layout while ``GDSTransferWorker``
reads the real tensor.  ``MooncakeStoreTransferWorker`` recomputed the
CPU-block geometry ``CPURemoteTransferWorker`` had.  That is the duplication
this refactor is aimed at: the edge is the reusable part, the engine is the
plug-in part, and they were the wrong way round.

So a backend is a strategy object the worker holds, exactly like
``CompressionStrategy``:

  attach(worker)   -- open sessions, register memory, derive whatever extra
                      geometry only this engine needs.  Runs inside the worker
                      process, after CUDA device binding and after the worker
                      has computed its edge geometry.
  transfer(...)    -- move the bytes for one op, return how many.  Returning
                      the count is what lets ``launch_transfer`` stop carrying
                      one ``transfer_size`` formula per engine.
  shutdown()       -- release what ``attach`` took.  Driven by the base
                      worker's ``shutdown``, before host memory is unpinned.

``backend=None`` means "the worker's own native engine" (``_transfer_impl``),
which is still the common case: io_uring for CPU<->SSD, the CUDA kernels/CE
for CPU<->GPU, cuFile for GPU<->SSD.  Nothing about the native paths changed;
they simply are no longer the only path a worker class can take.

Everything a backend holds before ``attach`` must be picklable -- backends are
constructed in the engine process and handed to ``create_worker``, which sends
them across a ``spawn`` boundary.  Sessions, clients, CUDA streams and raw
pointers are therefore created in ``attach``, never in ``__init__``.
"""
from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferType
from flexkv.external.mooncake_fault_inject import (
    inject_mooncake_fault,
    is_mooncake_fault_inject_enabled,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flexkv.transfer.worker_op import WorkerTransferOp
    from flexkv.transfer.workers import TransferWorkerBase


def op_layer_range(op: "WorkerTransferOp", num_layers: int) -> Tuple[int, int]:
    """``(layer_id, layer_granularity)`` for an op, defaulting to all layers.

    Layer-sliced ops are a read-path feature that no producer currently emits:
    ``WorkerTransferOp`` (and the ``TransferOp`` it is built from) carry no
    ``layer_id``/``layer_granularity``, and the one function that would have
    set them -- ``cache.transfer_pattern.convert_read_graph_to_layer_wise_graph``
    -- has no live caller.  Reading the attributes directly therefore raises
    ``AttributeError`` on every production op, so read them defensively: a
    backend that supports slicing keeps working the day the fields come back,
    and moves whole blocks until then.

    ``-1`` is the "unset" sentinel the old worker used, kept so a revived
    producer can leave either field out.
    """
    layer_id = getattr(op, "layer_id", -1)
    layer_granularity = getattr(op, "layer_granularity", -1)
    if layer_id is None or layer_id < 0:
        layer_id = 0
    if layer_granularity is None or layer_granularity < 0:
        layer_granularity = num_layers - layer_id
    return layer_id, layer_granularity


class StorageBackend(ABC):
    """One I/O engine for one edge.  Lives in the worker process."""

    #: Shown in worker logs; keep it engine-shaped ("nixl:GDS_MT").
    name: str = "backend"

    #: Whether ``launch_transfer`` should hand this backend pinned block-id
    #: tensors.  Pinning costs a host allocation per op and only pays off when
    #: the ids are consumed by CUDA; a pure-host engine sets this False.
    needs_pinned_block_ids: bool = True

    def attach(self, worker: "TransferWorkerBase") -> None:
        """Open sessions / register memory.  Called once, at worker init.

        Runs after the worker has bound its CUDA device and imported its IPC
        tensors, so it is safe to touch CUDA here.  A backend that needs
        geometry the worker did not compute derives it here and stores it on
        itself -- not on the worker, whose attributes describe the edge and
        must mean the same thing for every backend of that edge.
        """

    #: Whether this backend reports *per-block* outcomes rather than an
    #: all-or-nothing one.  A key/value tier can succeed on some keys and miss
    #: others, and a whole-op failure there throws away the blocks that did
    #: arrive; a file or kernel engine has no such notion, and leaving this
    #: False keeps its ops on the plain bool completion path.
    reports_block_results: bool = False

    @abstractmethod
    def transfer(
        self,
        worker: "TransferWorkerBase",
        op: "WorkerTransferOp",
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
    ) -> int:
        """Move the bytes for one op.  Returns bytes moved.

        Raises on failure; the worker's run loop turns that into a failed op
        rather than a silent short transfer.

        A backend that sets ``reports_block_results`` instead implements
        ``transfer_blocks`` and leaves this to the base implementation.
        """

    def transfer_blocks(
        self,
        worker: "TransferWorkerBase",
        op: "WorkerTransferOp",
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
    ) -> Tuple[Tuple[bool, ...], int]:
        """``(per-block outcomes, bytes moved)`` for one op.

        Only called when ``reports_block_results`` is set.  Unlike
        ``transfer``, a partial or total failure is a *return value*, not an
        exception: the op still completes, carrying which blocks landed, so
        the cache can keep the ones that did and the graph can release the
        rest instead of hanging on an op that never reports.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets reports_block_results but does not "
            "implement transfer_blocks")

    def shutdown(self) -> None:
        """Release whatever ``attach`` took.  Must tolerate a failed attach."""


# ---------------------------------------------------------------------------
# NIXL FILE backends
# ---------------------------------------------------------------------------


class NixlFileBackend(StorageBackend):
    """NIXL FILE plugins: GDS_MT (GPU<->file) or POSIX / HF3FS (CPU<->file).

    Attaches to ``GDSTransferWorker`` for the GPU plugins and to
    ``CPUSSDDiskTransferWorker`` for the CPU ones.  In both cases the worker
    has already decided the edge; this only replaces the engine (cuFile and
    io_uring respectively).

    NIXL addresses one (layer, kv) chunk at a time, so it needs per-chunk
    strides on both sides.  Those are read off the worker rather than
    recomputed: the worker derived them from the same ``KVCacheLayout``
    objects, and a second derivation is exactly the drift this refactor
    removes.  Only the file-routing constants -- how a global SSD block id maps
    to a path plus an in-file block index -- are the backend's own, because the
    native engines push that mapping into C++ and never materialize it in
    Python.

    One consequence of reading the worker's numbers: on the GPU side these are
    now the strides *measured from the real KV tensor* when the framework
    allocated one tensor per layer, where the old NIXL worker used the declared
    layout.  Those disagree whenever the attention backend permutes the 5D KV
    dims (flash_attn vs triton/flashinfer), and the measured pair is the one
    that matches the memory ``gpu_chunk_u8_view`` actually slices.
    """

    def __init__(
        self,
        nixl_backend: str,
        ssd_files: Dict[int, List[str]],
        nixl_extra_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # nixlutil imports the NIXL SDK, which most deployments do not have
        # installed -- keep it out of module import.
        from flexkv.transfer.nixlutil import (
            NIXL_CPU_FILE_BACKENDS,
            NIXL_GPU_FILE_BACKENDS,
            normalize_nixl_file_plugin_name,
        )

        be = normalize_nixl_file_plugin_name(str(nixl_backend).upper())
        if be not in NIXL_GPU_FILE_BACKENDS and be not in NIXL_CPU_FILE_BACKENDS:
            raise ValueError(
                f"nixl_backend must be one of "
                f"{sorted(NIXL_GPU_FILE_BACKENDS | NIXL_CPU_FILE_BACKENDS)}, "
                f"got {nixl_backend}"
            )
        self.nixl_backend = be
        self.name = f"nixl:{be}"
        self.is_gpu_plugin = be in NIXL_GPU_FILE_BACKENDS
        self.ssd_files = ssd_files
        self.num_devices = len(ssd_files)
        self.num_files_per_device = len(ssd_files[0])
        self.round_robin = 1
        self._extra_config = nixl_extra_config or {}
        self._session: Any = None

    def attach(self, worker: "TransferWorkerBase") -> None:
        from flexkv.transfer.nixlutil import NixlAgentSession

        if getattr(worker, "has_multi_group", False):
            raise ValueError(
                f"NIXL {self.nixl_backend} does not support heterogeneous "
                "(multi-group) KV layouts; the per-chunk addressing below "
                "assumes one uniform (layer, kv) chunk size"
            )
        self.num_layers = worker.num_layers
        self.kv_dim = worker.kv_dim

        # SSD side: identical geometry for every FILE plugin, and identical to
        # what the native engine of this edge uses -- read it off the worker.
        self.ssd_layer_stride_in_bytes = worker.ssd_layer_stride_in_bytes
        self.ssd_kv_stride_in_bytes = worker.ssd_kv_stride_in_bytes
        self.ssd_block_stride_in_bytes = worker.ssd_block_stride_in_bytes

        self._session = NixlAgentSession(self.nixl_backend, self._extra_config)
        if not self._session.prepare_all_ssd_files(self.ssd_files):
            raise RuntimeError("NIXL: prepare_all_ssd_files failed")

        if self.is_gpu_plugin:
            self._attach_gpu(worker)
        else:
            self._attach_cpu(worker)

        worker._bytes_per_block = (
            self.chunk_size_in_bytes * self.num_layers * self.kv_dim
        )
        flexkv_logger.info(
            f"[worker {worker.worker_id}] backend {self.name} attached: "
            f"chunk={self.chunk_size_in_bytes}B layers={self.num_layers} "
            f"kv_dim={self.kv_dim}"
        )

    def _attach_gpu(self, worker: "TransferWorkerBase") -> None:
        """GDS_MT: bind to the single device's tensors and strides.

        ``enable_nixl`` is validated to ``effective_tp_size_per_node == 1``
        (``KVTaskManager``), so the worker's per-device lists hold exactly one
        entry and index 0 is the whole TP group.
        """
        if worker.num_gpus != 1:
            raise RuntimeError(
                f"NIXL {self.nixl_backend} requires a single-GPU worker, got "
                f"num_gpus={worker.num_gpus}"
            )
        self.gpu_blocks = worker.gpu_blocks[0]
        n = len(self.gpu_blocks)
        # How the framework laid its KV tensors out: one fused tensor, one per
        # layer, or one per (layer, kv). The native engine derives the same
        # fork inside TPGDSTransferThreadGroup from num_tensors_per_gpu.
        if n == 1:
            self.gpu_block_type_ = 1
        elif n == self.num_layers:
            self.gpu_block_type_ = 0
        elif n == self.num_layers * 2:
            self.gpu_block_type_ = 2
        else:
            raise ValueError(f"Invalid GPU block count for NIXL: {n}")

        self.gpu_kv_stride_in_bytes = worker.gpu_kv_strides_in_bytes[0]
        self.gpu_block_stride_in_bytes = worker.gpu_block_strides_in_bytes[0]
        self.gpu_layer_stride_in_bytes = worker.gpu_layer_strides_in_bytes[0]
        self.chunk_size_in_bytes = worker.gpu_chunk_sizes_in_bytes[0]

        # NIXL VRAM xfers run on a stream of our own so they do not serialize
        # behind whatever the default stream is doing.
        self.transfer_stream = torch.cuda.Stream()
        if not self._session.prepare_vram_gpu(self.gpu_blocks):
            raise RuntimeError("NIXL: prepare_vram_gpu failed")

    def _attach_cpu(self, worker: "TransferWorkerBase") -> None:
        """POSIX / HF3FS: bind to the worker's CPU pool."""
        self.cpu_blocks = worker.cpu_blocks
        self.mem_layer_stride_in_bytes = worker.cpu_layer_stride_in_bytes
        self.mem_kv_stride_in_bytes = worker.cpu_kv_stride_in_bytes
        self.mem_block_stride_in_bytes = worker.block_stride_in_bytes
        self.chunk_size_in_bytes = worker.chunk_size_in_bytes
        gib = self.cpu_blocks.numel() * self.cpu_blocks.element_size() / (1024 ** 3)
        flexkv_logger.info(
            f"[worker {worker.worker_id}] {self.name}: pinning CPU pool "
            f"{gib:.2f} GiB"
        )
        worker._register_host_tensor(self.cpu_blocks, "nixl_cpu_pool")
        if not self._session.prepare_dram_cpu(self.cpu_blocks):
            raise RuntimeError("NIXL: prepare_dram_cpu failed")

    def _route(
        self,
        transfer_type: TransferType,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """(ssd ids, memory ids, READ/WRITE) for this plugin's transfer types."""
        if self.is_gpu_plugin:
            read, write = TransferType.DISK2D, TransferType.D2DISK
        else:
            read, write = TransferType.DISK2H, TransferType.H2DISK
        if transfer_type == read:
            return src_block_ids, dst_block_ids, "READ"
        if transfer_type == write:
            return dst_block_ids, src_block_ids, "WRITE"
        raise ValueError(
            f"NIXL {self.nixl_backend} expects {read} or {write}, "
            f"got {transfer_type}"
        )

    def transfer(
        self,
        worker: "TransferWorkerBase",
        op: "WorkerTransferOp",
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
    ) -> int:
        from flexkv.transfer.nixlutil import (
            file_path_for_ssd_block,
            gpu_chunk_u8_view,
            kv_chunk_byte_offset_in_block,
            ssd_chunk_byte_offset_in_file,
        )

        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        layer_id, layer_granularity = op_layer_range(op, self.num_layers)
        layer_end = layer_id + layer_granularity

        ssd_block_ids, mem_block_ids, direction = self._route(
            op.transfer_type, src_block_ids, dst_block_ids)

        n = ssd_block_ids.numel()
        if n == 0:
            return 0

        file_paths: List[str] = []
        region_offsets: List[int] = []
        region_lens: List[int] = []

        def ssd_route(ssd_b: int) -> Tuple[str, int]:
            return file_path_for_ssd_block(
                self.ssd_files, ssd_b, self.num_devices,
                self.num_files_per_device, self.round_robin,
            )

        def ssd_offset(lid: int, kv: int, block_in_file: int) -> int:
            return ssd_chunk_byte_offset_in_file(
                lid, kv, block_in_file,
                self.ssd_layer_stride_in_bytes,
                self.ssd_kv_stride_in_bytes,
                self.ssd_block_stride_in_bytes,
                self.kv_dim,
            )

        if self.is_gpu_plugin:
            gpu_tensors: List[torch.Tensor] = []
            for i in range(n):
                path, block_in_file = ssd_route(int(ssd_block_ids[i].item()))
                mem_b = int(mem_block_ids[i].item())
                for lid in range(layer_id, layer_end):
                    for kv in range(self.kv_dim):
                        gpu_tensors.append(gpu_chunk_u8_view(
                            self.gpu_blocks, self.gpu_block_type_,
                            self.num_layers, mem_b, lid, kv,
                            self.gpu_kv_stride_in_bytes,
                            self.gpu_block_stride_in_bytes,
                            self.gpu_layer_stride_in_bytes,
                            self.chunk_size_in_bytes, self.kv_dim,
                        ))
                        file_paths.append(path)
                        region_offsets.append(ssd_offset(lid, kv, block_in_file))
                        region_lens.append(self.chunk_size_in_bytes)

            with torch.cuda.stream(self.transfer_stream):
                if not self._session.xfer_vram_file(
                        direction, gpu_tensors, file_paths, region_lens,
                        region_offsets):
                    raise RuntimeError("NIXL GDS_MT transfer failed")
                torch.cuda.synchronize()
        else:
            base = self.cpu_blocks.data_ptr()
            dram_ptr_len: List[Tuple[int, int]] = []
            for i in range(n):
                path, block_in_file = ssd_route(int(ssd_block_ids[i].item()))
                mem_b = int(mem_block_ids[i].item())
                for lid in range(layer_id, layer_end):
                    for kv in range(self.kv_dim):
                        dram_ptr_len.append((
                            base + kv_chunk_byte_offset_in_block(
                                lid, kv, mem_b,
                                self.mem_layer_stride_in_bytes,
                                self.mem_kv_stride_in_bytes,
                                self.mem_block_stride_in_bytes, self.kv_dim,
                            ),
                            self.chunk_size_in_bytes,
                        ))
                        file_paths.append(path)
                        region_offsets.append(ssd_offset(lid, kv, block_in_file))
                        region_lens.append(self.chunk_size_in_bytes)

            if not self._session.xfer_dram_file(
                    direction, dram_ptr_len, file_paths, region_lens,
                    region_offsets):
                raise RuntimeError(
                    f"NIXL {self.nixl_backend} CPU<->file transfer failed")

        return (self.chunk_size_in_bytes * layer_granularity
                * op.valid_block_num * self.kv_dim)


# ---------------------------------------------------------------------------
# CPU<->Remote backends
# ---------------------------------------------------------------------------


class PcfsRemoteBackend(StorageBackend):
    """PCFS: the CPU<->Remote engine behind ``transfer_kv_blocks_remote``.

    Owns everything remote-file-shaped -- the ``Pcfs`` handle, the per-file
    node ids, the remote-side strides -- because none of it means anything to
    a remote tier that is not a filesystem.  ``CPURemoteTransferWorker`` used
    to hold all of it directly, which is precisely why a key/value remote tier
    (mooncake-store) could not reuse that worker and grew a worker of its own.

    It also owns the CPU-side strides, even though the worker owns the CPU
    pool.  That is deliberate: this engine's unit is a *(layer, kv) chunk*,
    and for a heterogeneous (multi-group) layout it flattens the whole block
    into one pseudo-layer of one pseudo-KV -- numbers that are true for PCFS
    and false for the edge.  A backend may reinterpret the edge's bytes; it
    may not redefine them for everyone else.
    """

    name = "pcfs"

    def __init__(
        self,
        remote_files: List[str],
        remote_kv_layout: KVCacheLayout,
        remote_config_custom: Dict[str, Any],
        enable_pcfs_sharing: bool = False,
    ) -> None:
        if not remote_config_custom:
            raise RuntimeError("remote_config_custom is not provided")
        self.remote_files = remote_files
        self.num_remote_files = len(remote_files)
        self.remote_kv_layout = remote_kv_layout
        self.remote_config_custom = remote_config_custom
        self.enable_pcfs_sharing = enable_pcfs_sharing
        self.round_robin = 1
        self.pcfs: Any = None

    def attach(self, worker: "TransferWorkerBase") -> None:
        from flexkv import c_ext
        # Deliberately the ``worker`` façade, not ``workers``: this is a c_ext
        # entry point re-exported there, resolved per call so a test can
        # monkeypatch the module attribute. Importing it at module scope would
        # also freeze the ``None`` a non-CFS build starts with.
        from flexkv.transfer.worker import transfer_kv_blocks_remote

        if transfer_kv_blocks_remote is None:
            raise RuntimeError(
                "transfer_kv_blocks_remote not available, please build with "
                "FLEXKV_ENABLE_CFS=1")

        cpu_layout = worker.cpu_kv_layout
        remote_layout = self.remote_kv_layout
        itemsize = worker.dtype.itemsize

        if cpu_layout.type != remote_layout.type:
            raise ValueError(
                f"CPU layout {cpu_layout.type} and remote layout "
                f"{remote_layout.type} must match")
        if worker.has_multi_group and cpu_layout.type != KVCacheLayoutType.BLOCKFIRST:
            raise ValueError(
                "Multi-group CPU/remote transfer requires BLOCKFIRST layouts")

        # The PCFS kernel walks ``base + layer*layer_stride + block*chunk`` and
        # transfers ``chunk`` bytes -- one value serves as both the block
        # multiplier and the transfer length, so the addressing it can express
        # is exactly LAYERFIRST.  Under BLOCKFIRST a block's layers are
        # contiguous, so flatten the whole block into a single MLA-shaped
        # "layer": ``block*whole_block`` is then the correct offset, and one
        # large I/O replaces num_layer*kv_dim small ones.  Multi-group is only
        # a special case of this -- its block was already a byte-flat blob.
        if cpu_layout.type == KVCacheLayoutType.BLOCKFIRST:
            self.block_size = cpu_layout.get_block_stride()
            self.num_layers = 1
            self.kv_dim = 1
        else:
            self.block_size = cpu_layout.get_chunk_size()
            self.num_layers = cpu_layout.num_layer
            self.kv_dim = cpu_layout.kv_dim

        num_cpu_blocks = cpu_layout.num_block
        num_remote_blocks = remote_layout.num_block
        if num_remote_blocks % self.num_remote_files != 0:
            raise ValueError(
                f"num_remote_blocks {num_remote_blocks} is not divisible by "
                f"num_remote_files {self.num_remote_files}")
        self.num_remote_blocks_per_file = (
            num_remote_blocks // self.num_remote_files)
        if self.num_remote_blocks_per_file % self.round_robin != 0:
            raise ValueError(
                f"num_remote_blocks_per_file {self.num_remote_blocks_per_file} "
                f"is not divisible by round_robin {self.round_robin}")

        # Strides are expressed in the flattened terms chosen above, so one
        # formula covers both layouts: under LAYERFIRST these are the real
        # per-layer/per-kv strides, and under BLOCKFIRST ``num_layers`` and
        # ``kv_dim`` are 1, so they never enter the addressing and only size
        # the file below.
        self.cpu_layer_stride_in_bytes = (
            num_cpu_blocks * self.block_size * itemsize * self.kv_dim)
        self.cpu_kv_stride_in_bytes = num_cpu_blocks * self.block_size * itemsize
        remote_layer_stride = (
            num_remote_blocks * self.block_size * itemsize * self.kv_dim)
        remote_kv_stride = num_remote_blocks * self.block_size * itemsize
        self.remote_layer_stride_in_bytes_per_file = (
            remote_layer_stride // self.num_remote_files)
        self.remote_kv_stride_in_bytes_per_file = (
            remote_kv_stride // self.num_remote_files)
        self.remote_block_stride_in_bytes = self.block_size * itemsize
        self.chunk_size_in_bytes = self.block_size * itemsize
        worker._bytes_per_block = (
            self.chunk_size_in_bytes * self.num_layers * self.kv_dim)

        cfg = self.remote_config_custom
        fsid = cfg.get("pcfs_fsid")
        port = cfg.get("pcfs_port")
        ip = cfg.get("pcfs_ip")
        parent_nodeid = cfg.get("pcfs_parent_nodeid")
        if None in (fsid, port, ip, parent_nodeid):
            raise RuntimeError("Some required PCFS config fields are missing")
        # 144115188075855883 only use int not c_types.u_int64
        self.pcfs = c_ext.Pcfs(fsid, port, ip, False, parent_nodeid)
        if not self.pcfs.init():
            raise RuntimeError(f"PCFS init failed: fsid={fsid}, ip={ip}")

        self.file_nodeid_list = []
        for remote_file_single in self.remote_files:
            nodeid = self.pcfs.lookup_or_create_file(
                remote_file_single,
                self.remote_layer_stride_in_bytes_per_file * self.num_layers,
                False)
            if nodeid == 0:
                raise RuntimeError(
                    f"lookup or create file failed for file: {remote_file_single}")
            self.file_nodeid_list.append(nodeid)

        c_ext.set_pcfs_instance(self.pcfs)

    def transfer(
        self,
        worker: "TransferWorkerBase",
        op: "WorkerTransferOp",
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
    ) -> int:
        import numpy as np

        from flexkv.common.transfer import PartitionBlockType
        from flexkv.transfer.worker import (
            shared_transfer_kv_blocks_remote_read,
            transfer_kv_blocks_remote,
        )

        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        transfer_type = op.transfer_type
        # Partial hits split a request: some blocks come from CPU and the rest
        # from remote, so both directions reach this backend.
        if transfer_type == TransferType.H2REMOTE:
            remote_block_id_list, cpu_block_id_list = dst_block_ids, src_block_ids
        elif transfer_type == TransferType.REMOTE2H:
            remote_block_id_list, cpu_block_id_list = src_block_ids, dst_block_ids
        else:
            raise ValueError(
                f"Invalid transfer type: {transfer_type} for PcfsRemoteBackend")

        layer_id_list = torch.arange(0, self.num_layers, dtype=torch.int32)
        cpu_base_ptr = worker.cpu_layer_ptrs[0].item()
        src_block_node_ids = op.src_block_node_ids

        if self.enable_pcfs_sharing and transfer_type == TransferType.REMOTE2H:
            if src_block_node_ids is not None and not isinstance(
                    src_block_node_ids, np.ndarray):
                raise TypeError(
                    "src_block_node_ids must be a numpy.ndarray if provided")
            assert len(src_block_node_ids) == len(remote_block_id_list)

            # Partition by owning node so each remote file is read by its own
            # thread fan-out rather than by one global sweep.
            file_nodeids_list = list(set(src_block_node_ids))
            cfs_blocks_partition: List[List[int]] = [[] for _ in file_nodeids_list]
            cpu_blocks_partition: List[List[int]] = [[] for _ in file_nodeids_list]
            file2fid = {nid: fid for fid, nid in enumerate(file_nodeids_list)}
            # Every FlexKV instance holds the same number of files, so turning a
            # global block id into an in-file one divides by that count; this
            # must match the C++ side in pcfs.cpp exactly.
            total_file_num = len(self.file_nodeid_list)
            for i in range(len(remote_block_id_list)):
                fid = file2fid[src_block_node_ids[i]]
                block_id_in_file = int(
                    ((remote_block_id_list[i] / self.round_robin) / total_file_num)
                    * self.round_robin
                    + (remote_block_id_list[i] % self.round_robin)
                )
                cfs_blocks_partition[fid].append(block_id_in_file)
                cpu_blocks_partition[fid].append(cpu_block_id_list[i].item())

            shared_transfer_kv_blocks_remote_read(
                file_nodeids_list,
                cfs_blocks_partition,
                cpu_blocks_partition,
                layer_id_list,
                cpu_base_ptr,
                self.cpu_layer_stride_in_bytes,
                self.cpu_kv_stride_in_bytes,
                self.remote_layer_stride_in_bytes_per_file,
                self.remote_block_stride_in_bytes,
                self.remote_kv_stride_in_bytes_per_file,
                self.chunk_size_in_bytes,
                self.num_layers,
                self.kv_dim,
                num_threads_per_file=32,
            )
        else:
            transfer_kv_blocks_remote(
                self.file_nodeid_list,
                layer_id_list,
                cpu_base_ptr,
                remote_block_id_list,
                cpu_block_id_list,
                self.cpu_layer_stride_in_bytes,
                self.cpu_kv_stride_in_bytes,
                self.remote_layer_stride_in_bytes_per_file,
                self.remote_block_stride_in_bytes,
                self.remote_kv_stride_in_bytes_per_file,
                self.chunk_size_in_bytes,
                self.num_layers,
                (transfer_type == TransferType.REMOTE2H),
                PartitionBlockType.SEQUENTIAL.value,
                self.round_robin,
                self.num_remote_blocks_per_file,
                False,
                32,
                self.kv_dim,
            )

        return (self.chunk_size_in_bytes * self.num_layers
                * op.valid_block_num * self.kv_dim)


def _split_mooncake_registration_regions(
    base_ptr: int,
    logical_size: int,
    mapped_size: int,
    block_size: int,
    max_mr_size: int,
    size_alignment: int,
    pointer_alignment: int,
    mr_split_policy: str = "strict",
) -> List[Tuple[int, int]]:
    """Split a mapped KV pool according to an explicit MR policy.

    ``strict`` keeps every region aligned to the KV block, external mapping,
    and HugePage. ``block_boundary`` keeps every KV block inside one MR while
    allowing derived MR pointers/sizes to be externally unaligned.
    """
    values = {
        "base_ptr": base_ptr,
        "logical_size": logical_size,
        "mapped_size": mapped_size,
        "block_size": block_size,
        "max_mr_size": max_mr_size,
        "size_alignment": size_alignment,
        "pointer_alignment": pointer_alignment,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if mapped_size < logical_size:
        raise ValueError(
            "HugePage mapped length is smaller than the logical CPU pool: "
            f"mapped={mapped_size}, logical={logical_size}"
        )
    if mr_split_policy not in {"strict", "block_boundary"}:
        raise ValueError(
            "mr_split_policy must be one of {'strict', 'block_boundary'}, got "
            f"{mr_split_policy!r}"
        )
    if logical_size % block_size != 0:
        raise ValueError(
            "Logical CPU pool must contain whole KV blocks: "
            f"logical_size={logical_size}, block_size={block_size}"
        )
    if base_ptr % pointer_alignment != 0:
        raise ValueError(
            "Mooncake MR base pointer is not HugePage aligned: "
            f"ptr=0x{base_ptr:x}, alignment={pointer_alignment}"
        )
    if mapped_size % size_alignment != 0:
        raise ValueError(
            "Mooncake mapped size is not externally aligned: "
            f"mapped_size={mapped_size}, alignment={size_alignment}"
        )
    if mapped_size <= max_mr_size:
        return [(base_ptr, mapped_size)]

    regions: List[Tuple[int, int]] = []
    if mr_split_policy == "block_boundary":
        mapped_padding = mapped_size - logical_size
        regular_region_size = (max_mr_size // block_size) * block_size
        final_logical_capacity = (
            (max_mr_size - mapped_padding) // block_size
        ) * block_size
        if regular_region_size <= 0 or final_logical_capacity <= 0:
            raise ValueError(
                "Mooncake max MR size cannot hold one KV block plus mapping tail: "
                f"max_mr_size={max_mr_size}, block_size={block_size}, "
                f"mapped_padding={mapped_padding}"
            )
        offset = 0
        while logical_size - offset > final_logical_capacity:
            required = logical_size - offset - final_logical_capacity
            size = min(
                regular_region_size,
                ((required + block_size - 1) // block_size) * block_size,
            )
            regions.append((base_ptr + offset, size))
            offset += size
        regions.append((base_ptr + offset, mapped_size - offset))
    else:
        region_unit = math.lcm(block_size, size_alignment, pointer_alignment)
        aligned_region_size = (max_mr_size // region_unit) * region_unit
        if aligned_region_size <= 0:
            raise ValueError(
                "Mooncake max MR size cannot hold one aligned KV region: "
                f"max_mr_size={max_mr_size}, region_unit={region_unit}"
            )
        offset = 0
        while offset < mapped_size:
            remaining = mapped_size - offset
            size = (
                remaining
                if remaining <= max_mr_size
                else aligned_region_size
            )
            regions.append((base_ptr + offset, size))
            offset += size

    for index, (ptr, size) in enumerate(regions):
        is_last = index == len(regions) - 1
        if (
            mr_split_policy == "strict"
            and not is_last
            and size % block_size != 0
        ):
            raise ValueError(
                "Non-final Mooncake MR is not KV-block aligned: "
                f"index={index}, size={size}, block_size={block_size}"
            )
        if mr_split_policy == "strict" and ptr % pointer_alignment != 0:
            raise ValueError(
                "Mooncake MR pointer is not HugePage aligned: "
                f"ptr=0x{ptr:x}, alignment={pointer_alignment}"
            )
        if mr_split_policy == "strict" and size % size_alignment != 0:
            raise ValueError(
                "Mooncake MR size is not externally aligned: "
                f"size={size}, alignment={size_alignment}"
            )
        if size > max_mr_size:
            raise ValueError(
                "Mooncake MR exceeds configured maximum: "
                f"size={size}, max_mr_size={max_mr_size}"
            )

    # Explicit last index, not ``regions[-1]``: release builds cythonize this
    # module with ``wraparound=False``, under which a negative index is NOT
    # folded to ``len + i`` -- it reads off the front of the list and segfaults.
    # The pure-Python path is unaffected, so this only ever failed in a
    # compiled build.
    last_ptr, last_size = regions[len(regions) - 1]
    if last_ptr + last_size != base_ptr + mapped_size:
        raise ValueError("Mooncake MR split does not cover mapped extent")
    if base_ptr + logical_size > last_ptr + last_size:
        raise ValueError("Mooncake MR split does not cover logical KV pool")
    return regions


def _register_mooncake_regions(
    client: Any, regions: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Register regions transactionally and roll back a partial failure."""
    registered: List[Tuple[int, int]] = []
    try:
        for ptr, size in regions:
            client.register_buffer(ptr, size)
            registered.append((ptr, size))
    except Exception:
        for ptr, _ in reversed(registered):
            try:
                client.unregister_buffer(ptr)
            except Exception as rollback_error:  # noqa: BLE001
                flexkv_logger.error(
                    "Mooncake MR rollback failed for "
                    f"ptr=0x{ptr:x}: {rollback_error}"
                )
        raise
    return registered


def _unregister_mooncake_regions(
    client: Any, regions: List[Tuple[int, int]]
) -> None:
    """Best-effort reverse-order cleanup for registered Mooncake MRs."""
    for ptr, size in reversed(regions):
        try:
            client.unregister_buffer(ptr)
        except Exception as error:  # noqa: BLE001
            flexkv_logger.error(
                "Mooncake MR unregister failed for "
                f"ptr=0x{ptr:x} size={size}: {error}"
            )


class MooncakeStoreBackend(StorageBackend):
    """mooncake-store: CPU<->Remote as a key/value tier rather than a file.

    Whole-block, opaque I/O keyed by block hash, which is why it needs none of
    ``PcfsRemoteBackend``'s remote-side strides -- the store has no layout of
    its own to address into.  What it does need, and a file backend does not,
    is the op's block hashes; those already ride on the op.

    ``pool_kind`` picks which hash list to key off: the main KV pool keys per
    block, the SWA pool keys per tail-hash snapshot (one key per CPU slot).
    """

    name = "mooncake-store"
    # A key/value tier hits or misses per key, so an op is not all-or-nothing.
    reports_block_results = True
    # Keys and pointers are computed on the host and nothing here feeds CUDA,
    # so do not pay a pinned host allocation per op for the block ids.
    needs_pinned_block_ids = False

    def __init__(
        self,
        cache_config: Any,
        pool_kind: Any,
        override_global_segment_size: Optional[int] = None,
    ) -> None:
        self.cache_config = cache_config
        self.pool_kind = pool_kind
        self.override_global_segment_size = override_global_segment_size
        self.mooncake_client: Any = None
        self._cpu_buffer: Any = None
        self._registered_regions: List[Tuple[int, int]] = []
        # PP isolation + layer-range key suffix; see ``build_key``.
        self.pp_rank = int(getattr(cache_config, 'mooncake_store_pp_rank', 0) or 0)
        self.pp_size = int(getattr(cache_config, 'mooncake_store_pp_size', 1) or 1)
        self.node_layer_start = int(
            getattr(cache_config, 'mooncake_store_node_layer_start', 0) or 0)
        self.node_layer_end = int(
            getattr(cache_config, 'mooncake_store_node_layer_end', 0) or 0)
        self.total_layers = int(
            getattr(cache_config, 'mooncake_store_total_layers', 0) or 0)

    def attach(self, worker: "TransferWorkerBase") -> None:
        from flexkv.external.mooncake_store_utils import (
            MooncakeStoreClient,
            MooncakeStoreConfig,
        )

        cpu_layout = worker.cpu_kv_layout
        assert cpu_layout.type == KVCacheLayoutType.BLOCKFIRST
        # Opaque whole-block I/O. Multi-group BLOCKFIRST stores bytes_per_block
        # directly in kv_shape[1] (via get_block_stride()), so it must not be
        # multiplied by itemsize; single-group layouts count elements.
        if cpu_layout.layer_groups is not None:
            self.block_size_bytes = int(cpu_layout.get_block_stride())
        else:
            self.block_size_bytes = int(
                cpu_layout.get_elements_per_block() * worker.dtype.itemsize)

        cpu_blocks = worker.cpu_blocks
        self._cpu_buffer = (
            cpu_blocks[0] if isinstance(cpu_blocks, (list, tuple)) else cpu_blocks)
        # No worker._register_host_tensor here: Mooncake owns the RDMA
        # registration and this backend only ever uses host pointers. A second
        # CUDA/HIP host registration of the same shared pool is redundant, and
        # for a large cache it exhausts the host mapping budget.
        flexkv_logger.info(
            "[MooncakeStoreBackend] skip CUDA host registration for the CPU "
            "KV pool; Mooncake owns the external MR")

        store_config = MooncakeStoreConfig.from_file(
            self.cache_config,
            override_global_segment_size=self.override_global_segment_size,
        )
        self.mooncake_client = MooncakeStoreClient(store_config)
        self._registered_regions = _register_mooncake_regions(
            self.mooncake_client, self._registration_regions(worker))
        worker._bytes_per_block = self.block_size_bytes

    def _registration_regions(
        self, worker: "TransferWorkerBase"
    ) -> List[Tuple[int, int]]:
        """The MRs to register for this worker's CPU pool.

        One region unless the pool is HugePage-backed and larger than the
        transport's MR limit, in which case it is split under
        ``mooncake_mr_split_policy`` -- a block straddling two MRs cannot be
        transferred by older Mooncake releases, so every policy keeps blocks
        whole and they differ only in whether the derived MR pointers and
        sizes stay externally aligned.
        """
        base_ptr = self._cpu_buffer.data_ptr()
        logical_size = (
            self._cpu_buffer.numel() * self._cpu_buffer.element_size())
        # ``mapped_size`` is the HugePage mapping's aligned length, which is
        # >= the logical pool. Only a HugePage handle knows it; a plain tensor
        # pool maps exactly what it holds.
        mapped_size = getattr(worker, "cpu_blocks_mapped_size", None)
        if mapped_size is None:
            return [(base_ptr, logical_size)]

        hugepage_size = int(self.cache_config.hugepage_size_bytes)
        size_alignment = int(os.getenv(
            "FLEXKV_HUGEPAGE_MAPPING_ALIGNMENT_BYTES", str(hugepage_size)))
        regions = _split_mooncake_registration_regions(
            base_ptr=base_ptr,
            logical_size=logical_size,
            mapped_size=int(mapped_size),
            block_size=self.block_size_bytes,
            max_mr_size=int(self.cache_config.mooncake_max_mr_size_bytes),
            size_alignment=size_alignment,
            pointer_alignment=hugepage_size,
            mr_split_policy=self.cache_config.mooncake_mr_split_policy,
        )
        flexkv_logger.info(
            "[MooncakeStoreBackend] registering external MRs: "
            f"logical_size={logical_size} mapped_size={mapped_size} "
            f"mr_split_policy={self.cache_config.mooncake_mr_split_policy} "
            f"regions={regions}")
        return regions

    def shutdown(self) -> None:
        client = self.mooncake_client
        if client is None:
            return
        _unregister_mooncake_regions(client, self._registered_regions)
        self._registered_regions = []

    def _keys_and_ptrs(
        self,
        op: "WorkerTransferOp",
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
    ) -> Tuple[List[int], List[int], List[str]]:
        from flexkv.external.mooncake_store_keys import PoolKind, build_key

        cpu_block_ids = (
            dst_block_ids if op.transfer_type == TransferType.REMOTE2H
            else src_block_ids
        )
        if self.pool_kind == PoolKind.SWA:
            # Each request contributes one (CPU slot id, tail_hash) pair; after
            # batch merging both lists hold N entries in the same order.
            hashes = op.mooncake_store_swa_block_hashes
            if hashes is None:
                raise ValueError(
                    "SWA mooncake transfer requires "
                    "mooncake_store_swa_block_hashes")
            if len(hashes) != len(cpu_block_ids):
                raise ValueError(
                    "SWA mooncake transfer requires len(swa_block_hashes) == "
                    f"len(cpu_block_ids): got {len(hashes)} vs "
                    f"{len(cpu_block_ids)}")
            raw = [str(h) for h in hashes]
        else:
            hashes = op.mooncake_store_block_hashes
            assert hashes is not None
            raw = list(hashes)

        base_ptr = self._cpu_buffer.data_ptr()
        size = self.block_size_bytes
        ptrs = [base_ptr + int(b) * size for b in cpu_block_ids]
        keys = [
            build_key(
                h, self.pool_kind,
                pp_rank=self.pp_rank, pp_size=self.pp_size,
                node_layer_start=self.node_layer_start,
                node_layer_end=self.node_layer_end,
                total_layers=self.total_layers,
            )
            for h in raw[:len(ptrs)]
        ]
        return ptrs, [size] * len(ptrs), keys

    def transfer(
        self,
        worker: "TransferWorkerBase",
        op: "WorkerTransferOp",
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
    ) -> int:
        raise NotImplementedError(
            "MooncakeStoreBackend reports per-block results; the worker must "
            "call transfer_blocks")

    def transfer_blocks(
        self,
        worker: "TransferWorkerBase",
        op: "WorkerTransferOp",
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
    ) -> Tuple[Tuple[bool, ...], int]:
        """Per-block outcomes for one key/value op.

        A store hit is per key, so an op is not all-or-nothing: raising on the
        first miss threw away every block that *did* arrive and forced the
        whole GET to re-prefill. The engine now AND-merges these outcomes into
        the op's ``block_results``, so a partial REMOTE2H keeps the blocks that
        landed and only the missing ones fall back; a partial H2REMOTE leaves
        only the keys that stored marked as stored.

        Failure is still reported, just not as an exception: an op that never
        completes hangs its graph and leaks every cache block the plan holds,
        which is strictly worse than completing with all-False.
        """
        expected = int(op.valid_block_num)
        sizes: List[int] = []
        try:
            ptrs, sizes, keys = self._keys_and_ptrs(
                op, src_block_ids, dst_block_ids)
            if len(keys) != expected:
                raise ValueError(
                    "Mooncake key count does not match transfer block count: "
                    f"keys={len(keys)}, blocks={expected}")
            if op.transfer_type == TransferType.H2REMOTE:
                results = self.mooncake_client.batch_put(keys, ptrs, sizes)
                verb = "put"
            elif op.transfer_type == TransferType.REMOTE2H:
                results = self.mooncake_client.batch_get(keys, ptrs, sizes)
                verb = "get"
                if is_mooncake_fault_inject_enabled():
                    results = inject_mooncake_fault(results, op.transfer_type)
                    flexkv_logger.info(
                        "Mooncake-store batch get results after fault "
                        f"injection: {results}")
            else:
                raise ValueError(
                    "MooncakeStoreBackend only supports H2REMOTE/REMOTE2H, "
                    f"got {op.transfer_type}")
            if len(results) != expected:
                raise ValueError(
                    "Mooncake result count does not match transfer block "
                    f"count: results={len(results)}, blocks={expected}")
            block_results = tuple(bool(ok) for ok in results)
            if not all(block_results):
                failed = [i for i, ok in enumerate(block_results) if not ok]
                flexkv_logger.warning(
                    f"Mooncake-store batch {verb} partially failed for "
                    f"{len(failed)}/{len(block_results)} block(s) "
                    f"(op {op.transfer_op_id}, graph {op.transfer_graph_id}); "
                    f"first failing key(s): {[keys[i] for i in failed[:4]]}")
        except Exception:
            flexkv_logger.error(
                "Mooncake transfer failed; reporting all blocks unsuccessful "
                f"for op_id={op.transfer_op_id}",
                exc_info=True,
            )
            block_results = (False,) * expected
        # Bytes actually moved, not the batch's nominal size: a miss transfers
        # nothing, and counting it inflates the bandwidth the trace reports.
        moved = sum(
            size for size, ok in zip(sizes, block_results) if ok)
        return block_results, moved
