"""GPU <-> SSD transfers over GPUDirect Storage.

``TPGDSTransferThreadGroup`` only exists in an ``FLEXKV_ENABLE_GDS=1`` build, so
the import is optional and the symbol may be ``None`` here.
"""

import time
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional

import torch
from torch.multiprocessing import Queue as MPQueue

# GDS is an optional build (FLEXKV_ENABLE_GDS=1); import must not be fatal.
try:
    from flexkv.c_ext import TPGDSTransferThreadGroup
except ImportError:
    TPGDSTransferThreadGroup = None

from flexkv.common.config import LayerGroupSpec
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout
from flexkv.common.transfer import TransferType
from flexkv.transfer.backends import StorageBackend
from flexkv.transfer.geometry import (
    ChunkStrides,
    DeviceSide,
    DiskSide,
    EdgeGeometry,
)
from flexkv.transfer.template import compile_gpu_regions, compile_host_regions
from flexkv.transfer.worker_op import WorkerTransferOp
from flexkv.transfer.workers.runtime import (
    TransferWorkerBase,
    ensure_cuda_device,
    import_tensor_handles,
)


class GDSTransferWorker(TransferWorkerBase):
    """GPU<->SSD transfers via GDS for a TP group of any size, including size 1.

    Same merge as ``GPUCPUTransferWorker``, for the same reason. There used to
    be two of these: this one (then ``tpGDSTransferWorker``) and a singular
    ``GDSTransferWorker`` that took one device's ``gpu_blocks``/``gpu_kv_layout``
    and drove ``transfer_kv_blocks_gds`` on a Python-side ``torch.cuda.Stream``.
    Both ended in the same templated ``flexkv::transfer_kv_blocks_gds``; they
    differed only in who spawned the threads. ``TPGDSTransferThreadGroup``
    already spawns one thread, one stream and one ``GDSManager`` per GPU, and
    ``num_gpus_ == 1`` *is* the non-TP case -- so the fan-out was being decided
    one level too high.

    Splitting on it cost two ``__init__``, two ``_init_*_multi_group_gds``, two
    ``_transfer_impl``, two byte-count formulas in ``launch_transfer``, and an
    ``effective_tp_size_per_node == 1`` branch at all four construction sites.
    The singular path also carried its own ``gpu_block_type_`` derivation of
    what the thread group already derives from ``num_tensors_per_gpu``.

    The one behavioural knob TP adds is ``ssd_tp_stride``: with more than one
    KV head each rank owns a slice of the block, so rank ``i`` reads at offset
    ``i * ssd_tp_stride``. At ``tp_group_size == 1`` that offset is 0 and the
    stride is the whole block, which is exactly what the singular worker
    passed.

    ``backend`` swaps out cuFile for another engine on the same edge -- NIXL's
    GDS_MT plugin is the one in tree. That used to be ``NixlTransferWorker``,
    a third GPU<->SSD worker whose only real difference was which library
    issued the read; it re-derived the SSD strides computed here and derived
    the GPU ones *differently*, from the declared layout rather than the real
    tensor. With the engine plugged in below, both views come from this one
    ``__init__``.
    """

    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        gpu_blocks: List[List[TensorSharedHandle]],
        ssd_files: Dict[int, List[str]],
        num_blocks_per_file: int,
        gpu_kv_layouts: List[KVCacheLayout],
        ssd_kv_layout: KVCacheLayout,
        dtype: torch.dtype,
        tp_group_size: int,
        layer_groups: Optional[List[LayerGroupSpec]] = None,
        gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]] = None,
        gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]] = None,
        backend: Optional[StorageBackend] = None,
    ) -> None:
        """
        Initialize the GDS Transfer Worker

        Args:
            worker_id: Worker ID
            transfer_queue: Queue for incoming transfer operations
            finished_ops_queue: Queue for completed operations
            gpu_blocks: List of GPU memory block handles for each GPU in TP group
            ssd_files: Dict of SSD file paths
            num_blocks_per_file: Number of blocks per file
            gpu_kv_layouts: Layout of GPU KV cache
            ssd_kv_layout: Layout of SSD KV cache
            dtype: Data type
            tp_group_size: Effective tp-group size on this node
                (``effective_tp_size_per_node`` =
                ``tp_size_per_node × cp_size_per_node``).
            layer_groups: Optional per-group KV layouts for heterogeneous models
                (including DSA/NSA indexer-as-group).
            backend: Optional ``StorageBackend`` replacing cuFile as the engine
                (e.g. ``NixlFileBackend("GDS_MT", ...)``). ``None`` = native.
        """
        # Initialize base class first
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        assert len(gpu_blocks) == tp_group_size
        if gpu_blocks and gpu_blocks[0]:
            ensure_cuda_device(gpu_blocks[0][0].device)
        self._pin_op_buffer()
        # Handle tensor import for multi-process case — set_device per GPU first.
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            imported_gpu_blocks.append(import_tensor_handles(handles_in_one_gpu))
        self.gpu_blocks = imported_gpu_blocks
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(file_list) for file_list in ssd_files.values())

        self.dtype = dtype
        self.kv_dim = gpu_kv_layouts[0].kv_dim
        self.num_kv_heads = gpu_kv_layouts[0].num_kv_heads
        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.has_multi_group = layer_groups is not None

        # Layout information
        self.num_layers = gpu_kv_layouts[0].num_layer

        if self.has_multi_group:
            self._init_multi_group_gds(
                gpu_kv_layouts, ssd_kv_layout, layer_groups,
                gpu_blocks_per_group, gpu_layouts_per_group,
                ssd_files)
        else:
            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)
            self.ssd_chunk_size_in_bytes = ssd_kv_layout_per_file.get_chunk_size() * self.dtype.itemsize
            self.chunk_size_in_bytes = self.ssd_chunk_size_in_bytes
            self.ssd_block_stride_in_bytes = ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize
            if self.num_kv_heads > 1:
                ssd_kv_layout_per_file = ssd_kv_layout_per_file.div_head(self.tp_group_size)

            # GPU layout calculations — compute strides from actual tensor to handle
            # different attention backend layouts (flash_attn vs triton/flashinfer).
            dtype_sz = self.dtype.itemsize
            tpb = gpu_kv_layouts[0].tokens_per_block
            self.gpu_chunk_sizes_in_bytes = []
            self.gpu_kv_strides_in_bytes = []
            self.gpu_block_strides_in_bytes = []
            self.gpu_layer_strides_in_bytes = []
            for i, gpu_kv_layout in enumerate(gpu_kv_layouts):
                gpu_strides = self._get_gpu_strides_from_tensor(
                    self.gpu_blocks[i][0], tpb, dtype_sz, self.kv_dim,
                ) if len(self.gpu_blocks[i]) > 1 else None
                if gpu_strides is not None:
                    kv_s, blk_s, layer_s = gpu_strides
                else:
                    kv_s = gpu_kv_layout.get_kv_stride() * dtype_sz
                    blk_s = gpu_kv_layout.get_block_stride() * dtype_sz
                    layer_s = gpu_kv_layout.get_layer_stride() * dtype_sz
                self.gpu_chunk_sizes_in_bytes.append(gpu_kv_layout.get_chunk_size() * dtype_sz)
                self.gpu_kv_strides_in_bytes.append(kv_s)
                self.gpu_block_strides_in_bytes.append(blk_s)
                self.gpu_layer_strides_in_bytes.append(layer_s)

            # SSD layout calculations
            self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            self.ssd_tp_stride_in_bytes = (self.ssd_block_stride_in_bytes // self.tp_group_size
                                           if self.num_kv_heads > 1 else self.ssd_block_stride_in_bytes)

            # Resolve pointers in Python
            gpu_block_ptrs_flat = [
                self.gpu_blocks[i][j].data_ptr()
                for i in range(self.num_gpus)
                for j in range(len(self.gpu_blocks[i]))
            ]
            gpu_device_ids = [self.gpu_blocks[i][0].device.index for i in range(self.num_gpus)]
            num_tensors_per_gpu = len(self.gpu_blocks[0])

            # cuFile is this edge's native engine; a backend replaces it
            # wholesale, so do not spawn its threads/streams/GDSManagers (nor
            # require a FLEXKV_ENABLE_GDS build) when one is present.
            if backend is None:
                self.gds_transfer_thread_group = TPGDSTransferThreadGroup(
                    self.num_gpus,
                    gpu_block_ptrs_flat,
                    num_tensors_per_gpu,
                    ssd_files,
                    self.num_layers,
                    self.gpu_kv_strides_in_bytes,
                    self.gpu_block_strides_in_bytes,
                    self.gpu_layer_strides_in_bytes,
                    self.gpu_chunk_sizes_in_bytes,
                    gpu_device_ids,
                )

        self._attach_backend(backend, self._build_geometry())

    def _build_geometry(self) -> EdgeGeometry:
        """This edge, in the terms every GPU<->SSD engine reads it in.

        There is no ``cpu`` side: GDS moves bytes between device memory and a
        file without a host bounce, so an engine that asks for a CPU pool gets
        a sentence saying this edge has none.

        Note which chunk size the SSD side carries. This worker's own
        ``chunk_size_in_bytes`` is the *SSD* chunk, while
        ``CPUSSDDiskTransferWorker``'s is the *CPU* chunk -- one attribute name
        for two different sides of two different transfers. Here each side
        states its own, so a backend cannot read one meaning it as the other.
        """
        if self.has_multi_group:
            ssd_strides = None
            gpu_strides = None
            ssd_block_stride = self.ssd_block_stride_in_bytes
            bytes_per_block = ssd_block_stride
        else:
            ssd_block_stride = self.ssd_block_stride_in_bytes
            ssd_strides = ChunkStrides(
                chunk_bytes=self.ssd_chunk_size_in_bytes,
                kv_stride=self.ssd_kv_stride_in_bytes,
                layer_stride=self.ssd_layer_stride_in_bytes,
                block_stride=ssd_block_stride,
            )
            gpu_strides = tuple(
                ChunkStrides(
                    chunk_bytes=self.gpu_chunk_sizes_in_bytes[i],
                    kv_stride=self.gpu_kv_strides_in_bytes[i],
                    layer_stride=self.gpu_layer_strides_in_bytes[i],
                    block_stride=self.gpu_block_strides_in_bytes[i],
                )
                for i in range(self.num_gpus)
            )
            bytes_per_block = (
                self.chunk_size_in_bytes * self.num_layers * self.kv_dim)
        return EdgeGeometry(
            num_layers=self.num_layers,
            kv_dim=self.kv_dim,
            num_kv_heads=self.num_kv_heads,
            dtype=self.dtype,
            has_multi_group=self.has_multi_group,
            bytes_per_block=bytes_per_block,
            ssd=DiskSide(
                block_stride=ssd_block_stride,
                strides=ssd_strides,
            ),
            gpu=DeviceSide(blocks=self.gpu_blocks, strides=gpu_strides),
        )

    def _init_multi_group_gds(
        self,
        gpu_kv_layouts: List[KVCacheLayout],
        ssd_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
        gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]],
        gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]],
        ssd_files: Dict[int, List[str]],
    ) -> None:
        """Initialize per-group TPGDSTransferThreadGroup instances.

        SSD buffer is byte-flat (uint8) in multi-group mode; per-group strides
        use g.dtype.itemsize so groups with different element sizes (e.g.
        bf16 main + uint8 indexer) interleave correctly within a block.
        """
        # TP width only enters through the per-rank slice stride below; the
        # SSD block geometry itself is the same at every tp_group_size.
        ssd_regions = compile_host_regions(
            layer_groups, ssd_kv_layout, self.kv_dim, self.dtype)
        tpb = ssd_kv_layout.tokens_per_block
        self.ssd_block_stride_in_bytes = ssd_kv_layout.get_block_stride()

        self.group_gds_params: list = []
        # Keep imported CUDA-IPC tensors alive: only data_ptr()s are recorded
        # below, so dropping the tensors would free the IPC mapping and dangle
        # the stored pointers.
        self._multi_group_gpu_blocks_keepalive: list = []

        gpu_device_ids = [self.gpu_blocks[i][0].device.index for i in range(self.num_gpus)]

        # strict=: compile_host_regions() emits exactly one region per group, so
        # a length mismatch is a broken invariant, not an input to truncate past.
        for gi, (g, ssd) in enumerate(zip(layer_groups, ssd_regions, strict=True)):
            # TP stride for SSD: partition the block across TP ranks. With one
            # KV head the ranks share the same heads, so there is nothing to
            # partition and every rank addresses the whole block.
            #
            # Pool-wide ``self.num_kv_heads``, not ``g.num_kv_heads``, and the
            # same value is handed to the C++ transfer below -- the stride and
            # the mode selector have to be derived from one number or the two
            # ends of the addressing disagree. See the note on the RegionSpec
            # in ``_init_tp_multi_group`` for why the pool-wide one is right.
            ssd_tp_stride = self.ssd_block_stride_in_bytes // self.tp_group_size if self.num_kv_heads > 1 \
                else self.ssd_block_stride_in_bytes

            # Per-group GPU strides and pointers
            if gpu_blocks_per_group is not None and gpu_layouts_per_group is not None:
                gpu_ptrs_flat = []
                num_tensors = None
                tensors_per_device: list = []

                for gpu_idx in range(self.num_gpus):
                    grp_handles = gpu_blocks_per_group[gi][gpu_idx]
                    grp_tensors = [h.get_tensor() for h in grp_handles]
                    self._multi_group_gpu_blocks_keepalive.append(grp_tensors)
                    tensors_per_device.append(
                        grp_tensors[0] if len(grp_tensors) > 1 else None)
                    for t in grp_tensors:
                        gpu_ptrs_flat.append(t.data_ptr())
                    if num_tensors is None:
                        num_tensors = len(grp_tensors)

                gpu_regions = compile_gpu_regions(
                    [g], [gpu_layouts_per_group[gi]], tpb, self.kv_dim, self.dtype,
                    tensors_per_group_device=[tensors_per_device],
                )
            else:
                gpu_regions = compile_gpu_regions(
                    [g], [list(gpu_kv_layouts)], tpb, self.kv_dim, self.dtype,
                    tensors_per_group_device=[[
                        (self.gpu_blocks[i][0]
                         if len(self.gpu_blocks[i]) > 1 else None)
                        for i in range(len(gpu_kv_layouts))
                    ]],
                )
                gpu_ptrs_flat = [
                    self.gpu_blocks[i][j].data_ptr()
                    for i in range(self.num_gpus)
                    for j in range(len(self.gpu_blocks[i]))
                ]
                num_tensors = len(self.gpu_blocks[0])

            gpu_kv_strides = [r.kv_stride for r in gpu_regions]
            gpu_block_strides = [r.block_stride for r in gpu_regions]
            gpu_layer_strides = [r.layer_stride for r in gpu_regions]
            gpu_chunk_sizes = [ssd.chunk_bytes] * self.num_gpus

            tp_gds_group = TPGDSTransferThreadGroup(
                self.num_gpus,
                gpu_ptrs_flat,
                num_tensors,
                ssd_files,
                g.num_layers,
                gpu_kv_strides,
                gpu_block_strides,
                gpu_layer_strides,
                gpu_chunk_sizes,
                gpu_device_ids,
            )

            self.group_gds_params.append({
                'num_layers': g.num_layers,
                'tp_gds_group': tp_gds_group,
                'ssd_layer_stride': ssd.layer_stride,
                'ssd_kv_stride': ssd.kv_stride,
                'ssd_tp_stride': ssd_tp_stride,
                'ssd_copy_offset': ssd.base_offset,
            })

        flexkv_logger.info(
            f"GDSTransferWorker multi-group initialized: {len(layer_groups)} groups, "
            f"ssd_block_stride={self.ssd_block_stride_in_bytes} bytes"
        )

    def _transfer_impl(self,
                       src_block_ids: torch.Tensor,
                       dst_block_ids: torch.Tensor,
                       transfer_type: TransferType,
                       **kwargs: Any,
                       ) -> None:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        # GDS uses DISK2D/D2DISK transfer types
        if transfer_type == TransferType.D2DISK:
            gpu_block_ids = src_block_ids
            ssd_block_ids = dst_block_ids
            is_read = False
        elif transfer_type == TransferType.DISK2D:
            gpu_block_ids = dst_block_ids
            ssd_block_ids = src_block_ids
            is_read = True
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for GDSTransferWorker. "
                             f"Expected DISK2D or D2DISK.")

        gpu_block_id_list = gpu_block_ids
        ssd_block_id_list = ssd_block_ids

        assert len(gpu_block_id_list) == len(ssd_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        if self.has_multi_group:
            for gp in self.group_gds_params:
                gp['tp_gds_group'].tp_group_transfer(
                    gpu_block_id_list,
                    ssd_block_id_list,
                    gp['ssd_layer_stride'],
                    gp['ssd_kv_stride'],
                    self.ssd_block_stride_in_bytes,
                    gp['ssd_tp_stride'],
                    self.num_blocks_per_file,
                    is_read,
                    0,  # layer_id always 0 for per-group
                    gp['num_layers'],
                    self.kv_dim,
                    self.num_kv_heads,
                )
        else:
            self.gds_transfer_thread_group.tp_group_transfer(
                gpu_block_id_list,
                ssd_block_id_list,
                self.ssd_layer_stride_in_bytes,
                self.ssd_kv_stride_in_bytes,
                self.ssd_block_stride_in_bytes,
                self.ssd_tp_stride_in_bytes,
                self.num_blocks_per_file,
                is_read,
                0,
                self.num_layers,
                self.kv_dim,
                self.num_kv_heads,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """Launch a TP GDS transfer operation"""
        if self._backend is not None:
            return self._run_backend(transfer_op)

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        start_time = time.time()
        self._transfer_impl(
            src_block_ids,
            dst_block_ids,
            transfer_op.transfer_type,
        )
        end_time = time.time()

        if self.has_multi_group:
            transfer_size = 0
            for gp in self.group_gds_params:
                transfer_size += gp['ssd_kv_stride'] * gp['num_layers'] * transfer_op.valid_block_num * self.kv_dim
        else:
            transfer_size = self.ssd_chunk_size_in_bytes * self.num_layers * transfer_op.valid_block_num * self.kv_dim

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )

        return True
