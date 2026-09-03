"""GPU <-> CPU transfers: every KV pool of one TP group on one worker.

The pools address the *same* physical resources -- the same pinned host buffer
and the same GPU tensors -- so they are pools inside one worker rather than one
worker each; ``op.pool_id`` selects which. Layerwise (per-layer eventfd)
delivery is a completion contract on this worker, not a separate class.
"""

import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.multiprocessing import Queue as MPQueue

from flexkv.c_ext import TPTransferThreadGroup
from flexkv.common.config import (
    GLOBAL_CONFIG_FROM_ENV,
    LayerGroupSpec,
    build_layer_member_map,
)
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle, release_vmm_tensor
from flexkv.common.pool import PoolId
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferType
from flexkv.storage.allocator import HugePageTensorHandle, materialize_worker_tensor
from flexkv.transfer.completion import CompletionContract
from flexkv.transfer.compression.common.strategy import (
    CompressionStrategy,
    NullCompressionStrategy,
)
from flexkv.transfer.layer_eventfd import receive_layer_eventfds
from flexkv.transfer.region_batch import (
    RegionSpec,
    build_region_batch,
    make_requests,
    rank_share_mode,
    region_batch_available,
)
from flexkv.transfer.template import compile_gpu_regions, compile_host_regions
from flexkv.transfer.worker_op import WorkerLayerwiseTransferOp, WorkerTransferOp
from flexkv.transfer.workers.runtime import (
    TransferWorkerBase,
    ensure_cuda_device,
    import_tensor_handles,
)

# Stand-in for the block ids of a cached layerwise request, which are rebound
# on every submit. Never read: a plan is only handed to cpp after every
# request's two tensors have been overwritten with the op's own.
_EMPTY_BLOCK_IDS = torch.empty(0, dtype=torch.int64)


def _validate_multi_group_chunk_layout(
    group_chunk_size: int,
    layout_chunk_size: int,
    group_index: int,
    group_tpb: int,
    layout_tpb: int,
    head_size: int,
    compress_ratio: int,
) -> None:
    """Reject a transfer descriptor that disagrees with GPU storage.

    Only meaningful at tp_group_size == 1. Under TP the device layout holds
    this rank's *shard* of the heads, so its chunk is smaller than the
    whole-group chunk by construction and the two are supposed to differ.
    """
    if group_chunk_size != layout_chunk_size:
        raise ValueError(
            "Multi-group chunk/layout mismatch for group "
            f"{group_index}: group_chunk={group_chunk_size} B, "
            f"layout_chunk={layout_chunk_size} B, "
            f"group_tpb={group_tpb}, layout_tpb={layout_tpb}, "
            f"head_size={head_size}, compress_ratio={compress_ratio}"
        )


@dataclass
class _Pool:
    """One KV pool a CPU<->GPU worker moves, as regions plus a fallback.

    Main KV and SWA are two instances of this, not two workers. They are the
    same shape of thing -- a host buffer, per-rank device tensors, a stride
    table, N layer groups -- differing only in which slot-id space their block
    ids are drawn from, and a block id is an argument, not a class.

    ``region_indices`` are this pool's entries in the worker's single
    ``RegionBatchGroup``: one submit covering full KV *and* SWA is one fan-out
    instead of two workers' worth of processes, queues and joins.

    ``thread_groups`` is the pre-region fallback, kept for builds whose
    extension has no ``RegionBatchGroup``. Each entry is the per-group dict the
    old ``tp_group_transfer_groups`` loop consumed; a uniform (single-group)
    pool has exactly one.
    """
    pool_id: PoolId
    name: str
    regions: List[RegionSpec] = field(default_factory=list)
    region_indices: List[int] = field(default_factory=list)
    thread_groups: List[dict] = field(default_factory=list)
    # ``layer_members[L]`` lists the ``(region_ordinal, local_layer_id)`` of
    # this pool that make up *original model* layer L. Region ordinal is
    # pool-local (an index into ``regions``/``region_indices``), because the
    # global region index is only assigned later, when every pool's regions
    # are concatenated into one RegionBatchGroup.
    #
    # An original layer this pool has no state for is an empty list, which a
    # per-layer completion check must read as "already satisfied" rather than
    # "never satisfied": a model with SWA on half its layers is the normal
    # case, not an error.
    layer_members: List[List[Tuple[int, int]]] = field(default_factory=list)
    # Bytes one whole block of this pool occupies across all its regions; used
    # for the transfer-rate log line.
    bytes_per_block: int = 0
    # Kept alive for the worker's lifetime: the C++ side stores only raw
    # data_ptr()s, so dropping these would release the CUDA IPC mapping and
    # leave those pointers dangling.
    keepalive: list = field(default_factory=list)


class GPUCPUTransferWorker(TransferWorkerBase):
    """CPU<->GPU transfers for a TP group of any size, including size 1.

    There used to be two of these: this one (then ``tpGPUCPUTransferWorker``)
    and a separate ``GPUCPUTransferWorker`` that took singular
    ``gpu_blocks``/``gpu_kv_layout`` and drove ``transfer_kv_blocks`` on a
    Python-side ``torch.cuda.Stream``. They were the same worker with the
    fan-out decided one level too high: how many device threads a transfer
    needs is a property of how many devices there are, which
    ``TPTransferThreadGroup`` already knows -- it spawns one thread and one
    stream per GPU, and ``num_gpus_ == 1`` *is* the non-TP case.

    Splitting on it in Python bought nothing and cost: two ``_transfer_impl``,
    two ``_control_suspend_gpu``, two nvcomp strategies bound to two different
    attribute shapes, and an ``effective_tp_size_per_node == 1`` branch at
    every construction site. It also meant only one of the two paths ever got
    a fix -- launch-then-wait completion landed on the non-TP side only.

    So: one worker, plural shape, thread count pushed down to cpp.
    """

    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 gpu_blocks: List[List[TensorSharedHandle]],
                 cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
                 gpu_kv_layouts: List[KVCacheLayout],
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 tp_group_size: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 transfer_num_cta_h2d: int = 4,
                 transfer_num_cta_d2h: int = 4,
                 compressor: Optional[CompressionStrategy] = None,
                 layer_groups: Optional[List[LayerGroupSpec]] = None,
                 gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]] = None,
                 gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]] = None,
                 # ---- SWA, as a second pool rather than a second worker ----
                 # All-or-nothing: supply the whole group or none of it. When
                 # present, this worker serves both is_swa=False and is_swa=True
                 # ops, and both pools' regions go out in one fan-out.
                 swa_gpu_blocks: Optional[List[List[TensorSharedHandle]]] = None,
                 swa_cpu_blocks: Optional[Union[torch.Tensor, HugePageTensorHandle]] = None,
                 swa_gpu_kv_layouts: Optional[List[KVCacheLayout]] = None,
                 swa_cpu_kv_layout: Optional[KVCacheLayout] = None,
                 swa_dtype: Optional[torch.dtype] = None,
                 swa_layer_groups: Optional[List[LayerGroupSpec]] = None,
                 swa_gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]] = None,
                 swa_gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]] = None,
                 # ---- per-layer completion, as a contract rather than a class -
                 # ``completion=PER_LAYER`` makes this worker also serve
                 # TransferType.LAYERWISE: same pools, same regions, same
                 # fan-out, plus a per-layer eventfd post. WHOLE (the default)
                 # never opens the UDS and never records a marker.
                 completion: Union[str, CompletionContract, None] = None,
                 layerwise_eventfd_socket: Optional[str] = None):

        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        assert len(gpu_blocks) == tp_group_size
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        # Bind primary GPU + pin op buffer before any CUDA IPC import.
        if gpu_blocks and gpu_blocks[0]:
            ensure_cuda_device(gpu_blocks[0][0].device)
        self._pin_op_buffer()
        # Handle tensor import for multi-process case — set_device per GPU first.
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            imported_gpu_blocks.append(import_tensor_handles(handles_in_one_gpu))
        self._gpu_block_counts = [len(handles) for handles in gpu_blocks]
        self.gpu_blocks = imported_gpu_blocks
        self.dtype = dtype # note this should be quantized data type
        self.kv_dim = gpu_kv_layouts[0].kv_dim
        self.num_kv_heads = gpu_kv_layouts[0].num_kv_heads

        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.layer_groups = layer_groups
        self.cpu_tensor = cpu_blocks

        flexkv_logger.info(f"Pinning CPU Memory: {cpu_blocks.numel() * cpu_blocks.element_size() / (1024 ** 3):.2f} GB")
        self._register_host_tensor(cpu_blocks, "tp_cpu_kv_pool")

        self.num_layers = gpu_kv_layouts[0].num_layer
        # The *original model* layer count for this PP stage -- the index space
        # the consumer's per-layer eventfds are numbered in. With layer groups
        # a group's own num_layer is a subset of the stage (the indexer covers
        # every layer, SWA covers some), so only the CPU layout carries the
        # full count; using a group's would make layer_indices go out of range.
        self._num_original_layers = (
            cpu_kv_layout.num_layer if layer_groups is not None
            else self.num_layers
        )

        self.transfer_num_cta_h2d = transfer_num_cta_h2d
        self.transfer_num_cta_d2h = transfer_num_cta_d2h
        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h

        # Read KV shared across ranks D2H mode from global config
        self.kv_shared_across_ranks_mode = GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode
        flexkv_logger.debug(f"[GPUCPUTransferWorker] kv_shared_across_ranks_mode={self.kv_shared_across_ranks_mode}")

        # Launch-then-wait completion. With sync=False cpp issues onto each
        # rank's stream and returns, and we drain every rank once at the end.
        # Two things this buys that per-call sync=True cannot:
        #   1. multi-group used to synchronize once *per group*, so group N+1
        #      could not be launched while group N drained.
        #   2. the binding holds the GIL across cudaStreamSynchronize, which
        #      stalls every other thread in this worker process (notably the
        #      mp.Queue feeder) for the whole transfer -- wait_all_streams
        #      releases it.
        # Off (FLEXKV_GPU_CPU_EVENT_SYNC=0) puts us back on the per-call
        # in-cpp synchronize, for bisecting.
        self._use_async_launch = bool(GLOBAL_CONFIG_FROM_ENV.gpu_cpu_event_sync)

        self._device_ids = [
            self.gpu_blocks[i][0].device.index for i in range(self.num_gpus)
        ]
        self.cpu_is_blockfirst = (
            cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST
        )
        # Region-batch is the default path; C++ handles the rank-sharing modes
        # (see RegionBatchGroup::build_args) so single-KV-head no longer opts
        # out. The per-group thread groups are built either way as the fallback
        # for a build whose extension predates RegionBatchGroup.
        self.region_batch = None
        self._rank_share_mode = None

        # Pools this worker serves, keyed by the id an op names. Ordered by
        # PoolId, because that order fixes the region indices below and those
        # numbers cross into cpp -- a dict whose iteration order drifted would
        # silently renumber every region.
        self._pools: Dict[PoolId, _Pool] = {}

        if layer_groups is not None and gpu_blocks_per_group is not None and gpu_layouts_per_group is not None:
            main_pool = self._init_tp_multi_group(
                gpu_blocks_per_group, gpu_layouts_per_group,
                cpu_kv_layout, layer_groups, pool_id=PoolId.FULL_KV,
            )
            self.tp_group_transfer_groups = main_pool.thread_groups
        else:
            main_pool = self._init_uniform(
                self.gpu_blocks, cpu_blocks, gpu_kv_layouts, cpu_kv_layout,
                self.dtype, pool_id=PoolId.FULL_KV, expose_on_self=True,
            )
            self.tp_group_transfer_groups = None
        self._pools[PoolId.FULL_KV] = main_pool

        # ---- SWA: a second pool on the same worker --------------------------
        # It used to be a second *worker* (TransferEngine._swa_worker_map), with
        # its own process, queue, dispatch branch and PP fan-out -- all of it a
        # copy of the main-KV path whose only real difference is that the block
        # ids index a different pool. The design doc's region model says what
        # varies is the region list, so that is what varies here.
        if swa_cpu_blocks is not None:
            if swa_cpu_kv_layout is None:
                raise ValueError("SWA pool needs swa_cpu_kv_layout")
            swa_multi_group = (swa_layer_groups is not None
                               and swa_gpu_blocks_per_group is not None
                               and swa_gpu_layouts_per_group is not None)
            if swa_multi_group and swa_gpu_blocks is not None:
                raise ValueError(
                    "pass either uniform swa_gpu_blocks or "
                    "swa_layer_groups/swa_gpu_blocks_per_group, not both")
            if not swa_multi_group and (swa_gpu_blocks is None
                                        or swa_gpu_kv_layouts is None):
                raise ValueError(
                    "uniform SWA pool needs swa_gpu_blocks and swa_gpu_kv_layouts")
            swa_cpu_tensor = materialize_worker_tensor(swa_cpu_blocks)
            self._register_host_tensor(swa_cpu_tensor, "tp_swa_cpu_kv_pool")
            self.swa_cpu_tensor = swa_cpu_tensor
            swa_dt = swa_dtype or self.dtype
            if swa_multi_group:
                swa_pool = self._init_tp_multi_group(
                    swa_gpu_blocks_per_group, swa_gpu_layouts_per_group,
                    swa_cpu_kv_layout, swa_layer_groups, pool_id=PoolId.SWA,
                    cpu_tensor=swa_cpu_tensor, dtype=swa_dt,
                )
            else:
                imported_swa = [import_tensor_handles(h) for h in swa_gpu_blocks]
                swa_pool = self._init_uniform(
                    imported_swa, swa_cpu_tensor, swa_gpu_kv_layouts,
                    swa_cpu_kv_layout, swa_dt, pool_id=PoolId.SWA,
                )
                # Keep the imported SWA IPC tensors alive: the thread group and
                # the region descs hold only their raw data_ptr()s.
                swa_pool.keepalive.append(imported_swa)
            self._pools[PoolId.SWA] = swa_pool

        self._build_region_batch(cpu_kv_layout)
        self._init_completion(completion, layerwise_eventfd_socket,
                              tp_group_size)

        self._compressor = compressor or NullCompressionStrategy()
        self._compressor.attach(self)

    def _ordered_pools(self) -> List["_Pool"]:
        """This worker's pools in PoolId order.

        The order is the region numbering: ``_build_region_batch`` assigns
        each pool a contiguous slice of the single RegionBatchGroup's index
        space in exactly this sequence, and those indices are what cpp
        addresses. Sorted rather than insertion-ordered so adding a pool at a
        different point in ``__init__`` cannot renumber the existing ones.
        """
        return [self._pools[pid] for pid in sorted(self._pools)]

    def _init_uniform(
        self,
        gpu_blocks: List[List[torch.Tensor]],
        cpu_tensor: torch.Tensor,
        gpu_kv_layouts: List[KVCacheLayout],
        cpu_kv_layout: KVCacheLayout,
        dtype: torch.dtype,
        *,
        pool_id: PoolId,
        expose_on_self: bool = False,
    ) -> "_Pool":
        """Geometry for a pool with no layer groups: one region, one stride table.

        ``expose_on_self`` publishes the derived numbers as the worker
        attributes the ANS compression strategy reads (``chunk_size_in_bytes``,
        ``cpu_*_stride_in_bytes``, ``tp_transfer_thread_group``, ...). Only the
        main pool does that: the compressor is sized for the main KV pool
        alone, and any other pool overwriting those would silently redirect it.
        """
        name = pool_id.name.lower()
        num_gpus = len(gpu_blocks)
        dtype_sz = dtype.itemsize
        tpb = gpu_kv_layouts[0].tokens_per_block
        kv_dim = gpu_kv_layouts[0].kv_dim
        num_kv_heads = gpu_kv_layouts[0].num_kv_heads
        num_layers = gpu_kv_layouts[0].num_layer

        # Compute GPU strides from the actual tensor to handle different
        # attention backend layouts (flash_attn: [2,N,B,H,D], triton:
        # [N,2,B,H,D]). Each GPU may differ, so compute per-GPU.
        gpu_chunk_sizes, gpu_kv_strides = [], []
        gpu_block_strides, gpu_layer_strides = [], []
        for i, gpu_kv_layout in enumerate(gpu_kv_layouts):
            gpu_strides = self._get_gpu_strides_from_tensor(
                gpu_blocks[i][0], tpb, dtype_sz, kv_dim,
            ) if len(gpu_blocks[i]) > 1 else None
            if gpu_strides is not None:
                kv_s, blk_s, layer_s = gpu_strides
            else:
                kv_s = gpu_kv_layout.get_kv_stride() * dtype_sz
                blk_s = gpu_kv_layout.get_block_stride() * dtype_sz
                layer_s = gpu_kv_layout.get_layer_stride() * dtype_sz
            gpu_chunk_sizes.append(gpu_kv_layout.get_chunk_size() * dtype_sz)
            gpu_kv_strides.append(kv_s)
            gpu_block_strides.append(blk_s)
            gpu_layer_strides.append(layer_s)

        cpu_block_stride = cpu_kv_layout.get_block_stride() * dtype_sz
        cpu_chunk_size = cpu_kv_layout.get_chunk_size() * dtype_sz
        # tp has effect on the layout of the cpu tensor: the tp dim is always
        # right after the block dim, on both BLOCKFIRST and LAYERFIRST.
        eff_cpu_layout = cpu_kv_layout
        if (cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST
                and num_kv_heads > 1):
            eff_cpu_layout = cpu_kv_layout.div_head(self.tp_group_size)
        cpu_layer_stride = eff_cpu_layout.get_layer_stride() * dtype_sz
        cpu_kv_stride = eff_cpu_layout.get_kv_stride() * dtype_sz
        cpu_tp_stride = cpu_block_stride // self.tp_group_size

        # Resolve pointers in Python (where storage is valid) and pass them to
        # C++: calling .data_ptr() there on a tensor that crossed the pybind11
        # boundary from a spawn'd subprocess raises "Tensor that doesn't have
        # storage" (shared memory / CUDA IPC).
        gpu_block_ptrs_flat = [
            gpu_blocks[i][j].data_ptr()
            for i in range(num_gpus)
            for j in range(len(gpu_blocks[i]))
        ]
        num_tensors_per_gpu = len(gpu_blocks[0])
        device_ids = [gpu_blocks[i][0].device.index for i in range(num_gpus)]

        thread_group = TPTransferThreadGroup(
            num_gpus,
            gpu_block_ptrs_flat,
            num_tensors_per_gpu,
            cpu_tensor.data_ptr(),
            num_layers,
            gpu_kv_strides,
            gpu_block_strides,
            gpu_layer_strides,
            gpu_chunk_sizes,
            device_ids,
            GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold,
            GLOBAL_CONFIG_FROM_ENV.ce_path_opt,
            GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d,
            (cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST),
            num_kv_heads,
            ce_gather_threads=GLOBAL_CONFIG_FROM_ENV.ce_gather_threads,
            ce_gather_nt=GLOBAL_CONFIG_FROM_ENV.ce_gather_nt,
        )

        if expose_on_self:
            self.gpu_chunk_sizes_in_bytes = gpu_chunk_sizes
            self.gpu_kv_strides_in_bytes = gpu_kv_strides
            self.gpu_block_strides_in_bytes = gpu_block_strides
            self.gpu_layer_strides_in_bytes = gpu_layer_strides
            self.cpu_block_stride_in_bytes = cpu_block_stride
            self.cpu_chunk_size_in_bytes = cpu_chunk_size
            self.chunk_size_in_bytes = cpu_chunk_size
            self.cpu_layer_stride_in_bytes = cpu_layer_stride
            self.cpu_kv_stride_in_bytes = cpu_kv_stride
            self.cpu_tp_stride_in_bytes = cpu_tp_stride
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = cpu_chunk_size * num_layers * kv_dim
            self.tp_transfer_thread_group = thread_group

        pool = _Pool(pool_id=pool_id, name=pool_id.name.lower())
        pool.bytes_per_block = cpu_chunk_size * num_layers * kv_dim
        pool.regions.append(RegionSpec(
            name=name,
            cpu_ptr=cpu_tensor.data_ptr(),
            cpu_kv_stride=cpu_kv_stride,
            cpu_layer_stride=cpu_layer_stride,
            cpu_block_stride=cpu_block_stride,
            cpu_tp_stride=cpu_tp_stride,
            gpu_block_ptrs_flat=gpu_block_ptrs_flat,
            num_tensors_per_gpu=num_tensors_per_gpu,
            gpu_kv_strides=gpu_kv_strides,
            gpu_block_strides=gpu_block_strides,
            gpu_layer_strides=gpu_layer_strides,
            gpu_chunk_sizes=gpu_chunk_sizes,
            num_layers=num_layers,
            kv_dim=kv_dim,
            num_kv_heads=num_kv_heads,
        ))
        pool.thread_groups.append({
            'tp_thread_group': thread_group,
            'cpu_kv_stride': cpu_kv_stride,
            'cpu_layer_stride': cpu_layer_stride,
            'cpu_block_stride': cpu_block_stride,
            'cpu_tp_stride': cpu_tp_stride,
            'cpu_offset_bytes': 0,
            'num_layers': num_layers,
            'chunk_size': cpu_chunk_size,
        })
        # One region covering every layer, so original layer L is exactly
        # region 0's local layer L.
        pool.layer_members = [[(0, layer)] for layer in range(num_layers)]
        return pool

    def _build_region_batch(self, cpu_kv_layout: KVCacheLayout) -> None:
        """One ``RegionBatchGroup`` over every pool's regions.

        Every pool in one group, not one group each: a request list can then
        name full-KV regions, SWA regions or both, and whatever it names goes
        out in a single fan-out. That is the "submit SWA / full / states /
        indexer as one batch" the design asks for -- the batch is the request
        list, so mixing pools costs nothing.

        Best-effort: an older extension has no ``RegionBatchGroup`` and every
        pool still has its thread groups, so ``_transfer_impl`` always has a
        path.
        """
        pools = self._ordered_pools()
        specs: List[RegionSpec] = []
        for pool in pools:
            pool.region_indices = list(
                range(len(specs), len(specs) + len(pool.regions)))
            specs.extend(pool.regions)

        if not (GLOBAL_CONFIG_FROM_ENV.region_batch and region_batch_available()):
            return
        # Parsed once: the string -> enum mapping (and its degrade-to-sharded
        # warning) belongs at init, not on every transfer.
        self._rank_share_mode = rank_share_mode(self.kv_shared_across_ranks_mode)
        try:
            self.region_batch = build_region_batch(
                specs, self._device_ids,
                ce_segment_threshold=GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold,
                ce_path_opt=GLOBAL_CONFIG_FROM_ENV.ce_path_opt,
                ce_enable_memcpy2d=GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d,
                is_blockfirst=(cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST),
                num_kv_heads=self.num_kv_heads,
                ce_gather_threads=GLOBAL_CONFIG_FROM_ENV.ce_gather_threads,
                ce_gather_nt=GLOBAL_CONFIG_FROM_ENV.ce_gather_nt,
            )
        except Exception as e:  # noqa: BLE001 - fall back, do not fail init
            flexkv_logger.warning(
                f"region batch unavailable, using per-group transfers: {e}")
            self.region_batch = None
        flexkv_logger.info(
            f"GPUCPUTransferWorker pools={[p.name for p in pools]} "
            f"regions={len(specs)} "
            f"region_batch={'on' if self.region_batch is not None else 'off'}")

    def _init_completion(
        self,
        completion: Union[str, "CompletionContract", None],
        layerwise_eventfd_socket: Optional[str],
        tp_group_size: int,
    ) -> None:
        """Decide what "done" means for this worker's CPU->GPU transfers.

        This is all that used to be ``LayerwiseTransferWorker``'s reason to
        exist. The launch shape (one batch per layer) is ``submit_layerwise``
        on the region batch, the stride table is the pools' regions -- neither
        needed a worker class. What is left is a contract: whether the consumer
        is told once at the end (WHOLE) or once per original layer (PER_LAYER),
        and PER_LAYER needs its eventfds, which is the handshake below.
        """
        if completion is None:
            completion = GLOBAL_CONFIG_FROM_ENV.layerwise_completion_contract
        if isinstance(completion, str):
            completion = CompletionContract.from_str(completion)
        self._completion = completion
        self._layer_milestones = self._build_layer_milestones()
        # {has_swa: (requests, pool_of, empty_layers)}; see _layerwise_plan.
        self._layerwise_plans: Dict[
            bool, Tuple[List["c_ext.RegionRequest"], List[PoolId], List[int]]
        ] = {}
        # Layers this model has no state for at all. Reported here only; the
        # list that gets posted is per transfer (see _layerwise_transfer_impl),
        # because an op that carries no SWA blocks leaves more layers uncovered
        # than this one.
        structurally_empty = sum(1 for m in self._layer_milestones if not m)
        self._layerwise_completion_timeout_s = float(
            GLOBAL_CONFIG_FROM_ENV.layerwise_completion_timeout_s)

        if not self._completion.needs_eventfd:
            return
        if self.region_batch is None:
            raise RuntimeError(
                "[worker %s] completion=per_layer needs the region batch "
                "path; this build's extension has no RegionBatchGroup"
                % self.worker_id)
        if layerwise_eventfd_socket is None:
            raise RuntimeError(
                "[worker %s] completion=per_layer without an eventfd socket "
                "path; nothing could ever signal a layer" % self.worker_id)
        fds = receive_layer_eventfds(
            layerwise_eventfd_socket, tp_group_size, self._num_original_layers,
            log_prefix=f"[worker {self.worker_id}]",
        )
        if fds.numel() == 0:
            raise RuntimeError(
                "[worker %s] completion=per_layer but the consumer handed "
                "over no eventfds; nothing could ever signal a layer. Use "
                "FLEXKV_LAYERWISE_COMPLETION_CONTRACT=whole if the consumer "
                "does not want per-layer notification." % self.worker_id)
        self.region_batch.set_layer_eventfds(
            fds, tp_group_size, self._num_original_layers,
            GLOBAL_CONFIG_FROM_ENV.layerwise_notify_mode,
        )
        flexkv_logger.info(
            f"[worker {self.worker_id}] per-layer completion armed: "
            f"layers={self._num_original_layers} "
            f"empty={structurally_empty} "
            f"notify={GLOBAL_CONFIG_FROM_ENV.layerwise_notify_mode}")

    def _build_layer_milestones(self) -> List[List[Tuple[PoolId, int, int]]]:
        """Per original layer, the ``(pool_id, global_region_index, local_layer)``.

        Every pool contributes: an original layer of a model with SWA is not
        complete until both its full-KV region and its SWA region have landed,
        and the consumer waits on *one* fd for it. Pool-local region ordinals
        are lifted to the region batch's global indices here, which is the one
        place that knows both numberings.

        ``pool_id`` rides along because the pools' block ids are drawn from
        different slot-id spaces, so the request built for a member has to know
        which id tensor to read -- that is the only thing the pool has ever
        changed about a transfer.
        """
        milestones: List[List[Tuple[PoolId, int, int]]] = [
            [] for _ in range(self._num_original_layers)
        ]
        for pool in self._ordered_pools():
            for layer, members in enumerate(pool.layer_members):
                if layer >= self._num_original_layers:
                    # A pool declaring more layers than the stage has is a
                    # geometry bug, not something to silently truncate.
                    raise ValueError(
                        f"pool {pool.name!r} declares layer {layer} but this "
                        f"PP stage has {self._num_original_layers} layers")
                for region_ordinal, local_layer in members:
                    milestones[layer].append(
                        (pool.pool_id, pool.region_indices[region_ordinal],
                         local_layer))
        return milestones

    def _init_tp_multi_group(
        self,
        gpu_blocks_per_group: List[List[List[TensorSharedHandle]]],
        gpu_layouts_per_group: List[List[KVCacheLayout]],
        cpu_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
        *,
        pool_id: PoolId = PoolId.FULL_KV,
        cpu_tensor: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "_Pool":
        """Compile one pool's layer groups into regions + per-group thread groups.

        CPU buffer is byte-flat (uint8) in multi-group mode: each block has
        size kv_shape[1] = bytes_per_block (see KVCacheLayout._compute_kv_shape).
        Per-group strides use g.dtype.itemsize so groups with different element
        sizes (e.g. bf16 main + uint8 indexer) interleave correctly within a
        block.

        ``cpu_tensor``/``dtype`` default to the main pool's; any other pool
        passes its own so the same compiler serves them all instead of a
        parallel copy per pool.
        """
        name = pool_id.name.lower()
        cpu_tensor = self.cpu_tensor if cpu_tensor is None else cpu_tensor
        dtype = self.dtype if dtype is None else dtype
        # Same host geometry as the non-TP worker: the CPU block layout does
        # not depend on how many GPUs write into it, only tp stride does.
        host_regions = compile_host_regions(
            layer_groups, cpu_kv_layout, self.kv_dim, dtype)

        tpb = cpu_kv_layout.tokens_per_block
        cpu_layout_type = cpu_kv_layout.type

        # For BLOCKFIRST multi-group, get_block_stride() returns bytes_per_block
        # directly (already accounts for tp_size and per-group dtype sizes).
        total_block_bytes = (
            cpu_kv_layout.get_block_stride()
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST else None
        )

        pool = _Pool(pool_id=pool_id, name=pool_id.name.lower())
        # pool.keepalive holds the imported CUDA-IPC tensors for the worker's
        # lifetime: TPTransferThreadGroup below stores only their raw
        # data_ptr()s, so if the tensors were dropped PyTorch would release the
        # IPC mapping and the pointers would dangle.
        for gi, (g, host) in enumerate(zip(layer_groups, host_regions)):
            # Per-group dtype: indexer uses uint8 even when main KV is bf16/fp16.
            dtype_size_g = (g.dtype or dtype).itemsize

            # gpu_blocks_per_group[gi] = list of per-GPU handle lists for this group
            # gpu_blocks_per_group[gi][gpu_idx] = handles for this group on GPU gpu_idx
            group_gpu_blocks_per_gpu = gpu_blocks_per_group[gi]

            # Import tensors from handles (bind CUDA device per GPU first)
            imported_group_blocks = []
            for handles_in_one_gpu in group_gpu_blocks_per_gpu:
                imported_group_blocks.append(import_tensor_handles(handles_in_one_gpu))
            pool.keepalive.append(imported_group_blocks)

            # Build flat pointer list for this group
            gpu_block_ptrs_flat = [
                imported_group_blocks[i][j].data_ptr()
                for i in range(self.num_gpus)
                for j in range(len(imported_group_blocks[i]))
            ]
            gpu_device_ids = [imported_group_blocks[i][0].device.index for i in range(self.num_gpus)]
            num_tensors_per_gpu = len(imported_group_blocks[0])

            # Per-group GPU strides: compute from the actual tensor to handle
            # different attention backend layouts (flash_attn vs
            # triton/flashinfer).  A GPU that exposes a single tensor carries no
            # recoverable per-layer dim, so its stride is left to the layout.
            group_gpu_layouts = gpu_layouts_per_group[gi]  # one layout per GPU
            gpu_regions = compile_gpu_regions(
                [g], [group_gpu_layouts], tpb, self.kv_dim, dtype,
                tensors_per_group_device=[[
                    (imported_group_blocks[i][0]
                     if len(imported_group_blocks[i]) > 1 else None)
                    for i in range(len(group_gpu_layouts))
                ]],
            )
            gpu_kv_strides = [r.kv_stride for r in gpu_regions]
            gpu_block_strides = [r.block_stride for r in gpu_regions]
            gpu_layer_strides = [r.layer_stride for r in gpu_regions]
            # Not gpu_regions[i].chunk_bytes: under TP the device layout holds
            # this rank's *shard* of the heads, so its chunk is smaller than the
            # whole-group chunk the CPU side uses.
            gpu_chunk_sizes = [
                layout.get_chunk_size() * dtype_size_g
                for layout in group_gpu_layouts
            ]

            # Fail closed before submitting a native transfer if the
            # declarative LayerGroupSpec disagrees with the actual tensor
            # layout. Catches page-packed GLM DSA indexer buffers (tpb=1, one
            # 8448-byte row) being described as tpb=64. Only at tp=1: under TP
            # the two are *meant* to differ by the head shard, see above.
            if self.tp_group_size == 1:
                group_tpb = tpb // g.compress_ratio
                group_chunk = (group_tpb * g.num_kv_heads * g.head_size
                               * dtype_size_g)
                for layout, layout_chunk in zip(group_gpu_layouts,
                                                gpu_chunk_sizes):
                    _validate_multi_group_chunk_layout(
                        group_chunk,
                        layout_chunk,
                        gi,
                        group_tpb,
                        layout.tokens_per_block,
                        g.head_size,
                        g.compress_ratio,
                    )

            # CPU strides for this group (all in bytes)
            cpu_block_stride = host.block_stride
            cpu_layer_stride = host.layer_stride
            cpu_kv_stride = host.kv_stride
            cpu_tp_stride = cpu_block_stride // self.tp_group_size

            # CPU tensor offset for this group (cpu_tensor is uint8 in multi-group)
            cpu_blocks_ptr = cpu_tensor.view(-1)[host.base_offset:].data_ptr()

            tp_thread_group = TPTransferThreadGroup(
                self.num_gpus,
                gpu_block_ptrs_flat,
                num_tensors_per_gpu,
                cpu_blocks_ptr,
                g.num_layers,
                gpu_kv_strides,
                gpu_block_strides,
                gpu_layer_strides,
                gpu_chunk_sizes,
                gpu_device_ids,
                GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold,
                GLOBAL_CONFIG_FROM_ENV.ce_path_opt,
                GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d,
                (cpu_layout_type == KVCacheLayoutType.BLOCKFIRST),
                # Deliberately the pool-wide head count, NOT g.num_kv_heads.
                # See the note on the RegionSpec below: this argument selects an
                # addressing mode for the whole block, and the block is laid out
                # per rank whatever a single group's head count happens to be.
                self.num_kv_heads,
            )

            pool.thread_groups.append({
                'tp_thread_group': tp_thread_group,
                'cpu_kv_stride': cpu_kv_stride,
                'cpu_layer_stride': cpu_layer_stride,
                'cpu_block_stride': cpu_block_stride,
                'cpu_tp_stride': cpu_tp_stride,
                'cpu_offset_bytes': host.base_offset,
                'num_layers': g.num_layers,
                'chunk_size': host.chunk_bytes,
            })

            # Same numbers, expressed as a region rather than as a group with
            # its own thread group. Collected here (not in a second loop) so
            # the two descriptions cannot drift.
            pool.regions.append(RegionSpec(
                name=f"{name}.group{gi}",
                cpu_ptr=cpu_blocks_ptr,
                cpu_kv_stride=cpu_kv_stride,
                cpu_layer_stride=cpu_layer_stride,
                cpu_block_stride=cpu_block_stride,
                cpu_tp_stride=cpu_tp_stride,
                gpu_block_ptrs_flat=gpu_block_ptrs_flat,
                num_tensors_per_gpu=num_tensors_per_gpu,
                gpu_kv_strides=gpu_kv_strides,
                gpu_block_strides=gpu_block_strides,
                gpu_layer_strides=gpu_layer_strides,
                gpu_chunk_sizes=gpu_chunk_sizes,
                num_layers=g.num_layers,
                kv_dim=self.kv_dim,
                # Pool-wide, not g.num_kv_heads -- this is an *addressing mode*
                # selector, not a shape. >1 means "each rank owns a private
                # stretch at rank*cpu_tp_stride"; ==1 means "every rank holds
                # identical bytes, so the rank-share modes get to divide the
                # write up". Which one is right is decided by how the CPU block
                # was allocated, and KVCacheLayout._compute_kv_shape multiplies
                # the summed group spans by tp_size for *every* group -- so a
                # 1-head indexer or state sidecar still has one private slot per
                # rank. Passing g.num_kv_heads here would send those groups down
                # the rank-shared branch and make ranks overwrite each other:
                # measured at TP=2 and TP=4, it corrupts the sidecar under all
                # three share modes (sharded/all_write/rank0_only) while the
                # pool-wide value round-trips byte-exact.
                num_kv_heads=self.num_kv_heads,
            ))

        # Bytes per block across every group of this pool: with layer groups a
        # "block" is the whole interleaved record, so its size is the sum of
        # the groups' spans rather than any single group's chunk.
        pool.bytes_per_block = sum(host.span_bytes for host in host_regions)

        # Which of this pool's regions carry each original model layer. Group
        # ordinal == region ordinal here: the loop above appended exactly one
        # region per group, in group order.
        member_map = build_layer_member_map(
            list(layer_groups), self._num_original_layers)
        pool.layer_members = [list(m) for m in member_map.members]

        flexkv_logger.info(
            f"multi-group pool {name!r} initialized: {len(layer_groups)} groups, "
            f"total_block_bytes={total_block_bytes}"
        )
        return pool


    def _control_suspend_gpu(self, payload: Any) -> int:
        if self.tp_group_transfer_groups is not None:
            raise NotImplementedError(
                "GPU hot remap does not support multi-group KV layouts"
            )
        if not self.gpu_blocks:
            return 0
        zero_ptrs = [0] * sum(self._gpu_block_counts)
        self.tp_transfer_thread_group.update_gpu_block_ptrs(zero_ptrs)
        old_blocks = self.gpu_blocks
        self.gpu_blocks = []
        released = sum(
            release_vmm_tensor(tensor)
            for blocks_in_one_gpu in old_blocks
            for tensor in blocks_in_one_gpu
        )
        expected = sum(self._gpu_block_counts)
        if released != expected:
            raise RuntimeError(
                f"Expected {expected} VMM mappings, released {released}"
            )
        return released

    def _control_resume_gpu(
        self, gpu_blocks: List[List[TensorSharedHandle]]
    ) -> int:
        if self.gpu_blocks:
            raise RuntimeError("GPU blocks are already registered")
        counts = [len(handles) for handles in gpu_blocks]
        if counts != self._gpu_block_counts:
            raise ValueError(
                f"Expected GPU block counts {self._gpu_block_counts}, got {counts}"
            )
        imported_gpu_blocks = [
            import_tensor_handles(handles) for handles in gpu_blocks
        ]
        gpu_block_ptrs_flat = [
            tensor.data_ptr()
            for blocks_in_one_gpu in imported_gpu_blocks
            for tensor in blocks_in_one_gpu
        ]
        self.tp_transfer_thread_group.update_gpu_block_ptrs(
            gpu_block_ptrs_flat
        )
        self.gpu_blocks = imported_gpu_blocks
        return len(gpu_block_ptrs_flat)

    def _use_region_batch(self) -> bool:
        """Whether this transfer can go through the batched path.

        Built means usable.  This used to also require ``num_kv_heads > 1``,
        because the rank-sharing modes (sharded / all_write / rank0_only /
        layer_parallel / rank_rotate) rewrite the per-rank offsets, chunk size
        or layer range and that logic existed only in
        ``TPTransferThreadGroup``.  It now lives in ``RegionBatchGroup`` too --
        carried per request as ``rank_share_mode`` -- so the single-KV-head
        case no longer has to fall back to the per-group loop.
        """
        return getattr(self, "region_batch", None) is not None

    def _pool_for(self, op: Optional[WorkerTransferOp]) -> "_Pool":
        """Which pool an op addresses.

        The only thing the pool changes about a transfer is which slot-id space
        the block ids index -- same direction, same layout family, same worker.
        So it is a lookup here instead of a whole second worker upstream.
        """
        pool_id = PoolId.FULL_KV if op is None else getattr(
            op, "pool_id", PoolId.FULL_KV)
        pool = self._pools.get(pool_id)
        if pool is None:
            raise RuntimeError(
                f"[worker {self.worker_id}] got a {pool_id.name} op but this "
                f"worker has no {pool_id.name} pool registered")
        return pool

    def _transfer_impl(self,
                       src_block_ids: torch.Tensor,
                       dst_block_ids: torch.Tensor,
                       transfer_type: TransferType,
                       pool: Optional["_Pool"] = None,
                       **kwargs: Any,
                       )->None:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)
        pool = self._pools[PoolId.FULL_KV] if pool is None else pool

        if transfer_type == TransferType.H2D:
            gpu_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
            use_ce_transfer = self.use_ce_transfer_h2d
            transfer_num_cta = self.transfer_num_cta_h2d
        elif transfer_type == TransferType.D2H:
            gpu_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
            use_ce_transfer = self.use_ce_transfer_d2h
            transfer_num_cta = self.transfer_num_cta_d2h
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for GPUCPUTransferWorker")


        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        # sync=False launches and returns; we drain below. The observable
        # contract of _transfer_impl is unchanged -- it still returns only once
        # the data has landed.
        launch_sync = not self._use_async_launch

        if self._use_region_batch():
            # Every region in one fan-out: each rank walks the request list on
            # its own stream, so region N+1's launch overlaps region N's copy
            # and there is a single join for the whole batch.
            #
            # region_indices restricts the batch to this pool's regions, so the
            # block ids are read against the pool they belong to. share_mode is
            # what used to be TPTransferThreadGroup's job: with it carried per
            # request, the single-KV-head case no longer has to fall back.
            self.region_batch.submit(
                make_requests(
                    self.region_batch.num_regions,
                    gpu_block_id_list,
                    cpu_block_id_list,
                    transfer_type == TransferType.H2D,
                    transfer_num_cta=transfer_num_cta,
                    use_ce_transfer=use_ce_transfer,
                    share_mode=self._rank_share_mode,
                    region_indices=pool.region_indices,
                ),
                launch_sync,
            )
            if self._use_async_launch:
                self.region_batch.wait_all_streams()
        elif (pool.pool_id is not PoolId.FULL_KV
              or self.tp_group_transfer_groups is not None):
            # Multi-group transfer: one call per group. With launch_sync=False
            # every group is in flight at once instead of one at a time.
            # A non-default pool always comes through here even when it has a
            # single group -- its geometry lives on the pool, not on self.
            for gp in pool.thread_groups:
                g_gpu = gpu_block_id_list
                g_cpu = cpu_block_id_list

                gp['tp_thread_group'].tp_group_transfer(
                    g_gpu,
                    g_cpu,
                    gp['cpu_kv_stride'],
                    gp['cpu_layer_stride'],
                    gp['cpu_block_stride'],
                    gp['cpu_tp_stride'],
                    transfer_num_cta,
                    transfer_type == TransferType.H2D,
                    use_ce_transfer,
                    0,                 # start_layer_id (always 0 within group)
                    gp['num_layers'],  # all layers in this group
                    self.kv_dim,
                    self.num_kv_heads,
                    self.kv_shared_across_ranks_mode,
                    0,                 # designated_rank
                    launch_sync,
                )
            if self._use_async_launch:
                for gp in pool.thread_groups:
                    gp['tp_thread_group'].wait_all_streams()
        else:
            self.tp_transfer_thread_group.tp_group_transfer(
                gpu_block_id_list,
                cpu_block_id_list,
                self.cpu_kv_stride_in_bytes,
                self.cpu_layer_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                self.cpu_tp_stride_in_bytes,
                transfer_num_cta,
                transfer_type == TransferType.H2D,
                use_ce_transfer,
                0,                  # start_layer_id (whole-model)
                self.num_layers,    # layer_granularity = all layers
                self.kv_dim,
                self.num_kv_heads,
                self.kv_shared_across_ranks_mode,
                0,                  # designated_rank
                launch_sync,
            )
            if self._use_async_launch:
                self.tp_transfer_thread_group.wait_all_streams()


    def _layerwise_plan(
        self, has_swa: bool,
    ) -> Tuple[List["c_ext.RegionRequest"], List[PoolId], List[int]]:
        """The request list for a layerwise transfer, built once and reused.

        Everything about these requests except the two block-id tensors is
        fixed by the worker's geometry: the region index, the local layer, the
        direction, the CTA count, the share mode. Only which blocks move
        changes from op to op. So the objects are built the first time a shape
        is seen and kept -- at DSv4 scale that is 128 ``RegionRequest``
        constructions plus 11 pybind setter calls each, per transfer, for a
        result that is byte-identical every time.

        Reuse is safe because ``submit_layerwise`` copies: pybind11's
        ``type_caster_base`` declares the non-movable ``cast_op_type``, so the
        vector cast copies each element rather than moving from it, and the
        objects come back intact. It is also *only* safe while transfers are
        serialized on this worker's thread -- a second op overwriting the id
        tensors while the previous one's DMA still reads them would corrupt it.
        That is the case today: ``_layerwise_transfer_impl`` drains before it
        returns.

        Keyed on ``has_swa`` because an op with no SWA blocks skips the SWA
        members entirely, which changes both the request list and the set of
        layers left empty.
        """
        cached = self._layerwise_plans.get(has_swa)
        if cached is not None:
            return cached

        requests: List["c_ext.RegionRequest"] = []
        pool_of: List[PoolId] = []
        # Layers this transfer will not launch anything for. The consumer waits
        # on every layer's fd regardless, so these have to be posted up front or
        # it hangs. It is per shape, not per worker: an op with no SWA blocks
        # leaves every SWA-only layer uncovered too.
        empty_layers: List[int] = []
        for layer, members in enumerate(self._layer_milestones):
            live = [m for m in members
                    if has_swa or m[0] is PoolId.FULL_KV]
            if not live:
                empty_layers.append(layer)
                continue
            for pool_id, region_index, local_layer in live:
                req = make_requests(
                    self.region_batch.num_regions,
                    # Placeholders; every submit rebinds both tensors.
                    _EMPTY_BLOCK_IDS,
                    _EMPTY_BLOCK_IDS,
                    True,  # layerwise is CPU->GPU only
                    transfer_num_cta=self.transfer_num_cta_h2d,
                    use_ce_transfer=self.use_ce_transfer_h2d,
                    share_mode=self._rank_share_mode,
                    region_indices=[region_index],
                    layer_id=local_layer,
                    # PER_LAYER pins this to 1: a coarser batch would post
                    # layer L's fd only after L+1 had also landed, so the
                    # consumer would read a layer it was told was ready.
                    layer_granularity=self._completion.layer_granularity(1),
                )[0]
                req.milestone_layer = layer
                requests.append(req)
                pool_of.append(pool_id)

        plan = (requests, pool_of, empty_layers)
        self._layerwise_plans[has_swa] = plan
        return plan

    def _layerwise_transfer_impl(
        self,
        cpu_block_ids: torch.Tensor,
        gpu_block_ids: torch.Tensor,
        swa_cpu_block_ids: Optional[torch.Tensor],
        swa_gpu_block_ids: Optional[torch.Tensor],
        counter_id: int,
    ) -> None:
        """One H2D transfer, launched and notified one original layer at a time.

        Same regions and the same fan-out as a whole-block H2D; the only
        difference is that each request is tagged with the model layer it
        completes, so C++ can record a marker after that layer's last launch
        and post the consumer's eventfd when every rank's marker fires.

        The batch is one request per (region, layer) member across *both*
        pools, in ascending layer order -- so a layer whose state spans main KV
        and SWA posts once, after both have landed, which is what the consumer
        means by "layer L is readable".
        """
        if self.region_batch is None:
            raise RuntimeError(
                f"[worker {self.worker_id}] layerwise transfer needs the "
                "region batch path; this build's extension has no "
                "RegionBatchGroup")
        has_swa = (swa_cpu_block_ids is not None
                   and swa_cpu_block_ids.numel() > 0)
        if has_swa and PoolId.SWA not in self._pools:
            raise RuntimeError(
                f"[worker {self.worker_id}] layerwise op carries SWA block ids "
                "but this worker has no SWA pool registered")

        requests, pool_of, empty_layers = self._layerwise_plan(has_swa)
        # The layerwise op carries one id pair per pool it touches. Only two
        # exist today, so this is a dict rather than a widening of the op: the
        # cpp entry point takes the two tensors positionally.
        ids_by_pool = {
            PoolId.FULL_KV: (gpu_block_ids, cpu_block_ids),
            PoolId.SWA: (swa_gpu_block_ids, swa_cpu_block_ids),
        }
        for req, pool_id in zip(requests, pool_of):
            req.gpu_block_id_tensor, req.cpu_block_id_tensor = \
                ids_by_pool[pool_id]

        self.region_batch.submit_layerwise(
            requests, empty_layers, counter_id)
        # Launched, not landed. Returning here would let the engine recycle the
        # source blocks while DMA is still reading them -- and we cannot drain
        # inline before the posts, because the consumer is blocked on the very
        # eventfds this transfer still has to write.
        ok, err = self.region_batch.wait_layer_completion(
            self._layerwise_completion_timeout_s)
        if not ok:
            raise RuntimeError(
                f"[worker {self.worker_id}] layerwise transfer did not "
                f"complete: {err}")

    def _launch_layerwise(self, transfer_op: WorkerLayerwiseTransferOp) -> bool:
        src = torch.from_numpy(
            transfer_op.src_block_ids_h2d).to(dtype=torch.int64).pin_memory()
        dst = torch.from_numpy(
            transfer_op.dst_block_ids_h2d).to(dtype=torch.int64).pin_memory()
        if transfer_op.swa_src_block_ids_h2d.size > 0:
            swa_src = torch.from_numpy(
                transfer_op.swa_src_block_ids_h2d).to(
                    dtype=torch.int64).pin_memory()
            swa_dst = torch.from_numpy(
                transfer_op.swa_dst_block_ids_h2d).to(
                    dtype=torch.int64).pin_memory()
        else:
            swa_src = swa_dst = None

        start_time = time.time()
        self._layerwise_transfer_impl(
            src, dst, swa_src, swa_dst, transfer_op.counter_id)
        end_time = time.time()

        transfer_size = self._pools[PoolId.FULL_KV].bytes_per_block * len(src)
        swa_pool = self._pools.get(PoolId.SWA)
        if swa_src is not None and swa_pool is not None:
            transfer_size += swa_pool.bytes_per_block * len(swa_src)
        self._log_transfer_performance(
            transfer_op, transfer_size, start_time, end_time)
        return True

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        if isinstance(transfer_op, WorkerLayerwiseTransferOp):
            return self._launch_layerwise(transfer_op)
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)
        pool = self._pool_for(transfer_op)
        if (pool.pool_id is not PoolId.FULL_KV
                or self.tp_group_transfer_groups is not None):
            # Multi-group or a non-default pool — compression is sized for the
            # main uniform pool only, so it does not apply here.
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
                pool=pool,
            )
            end_time = time.time()

            transfer_size = 0
            for gp in pool.thread_groups:
                transfer_size += gp['chunk_size'] * gp['num_layers'] * transfer_op.valid_block_num * self.kv_dim

            self._log_transfer_performance(
                transfer_op,
                transfer_size,
                start_time,
                end_time,
            )
        else:
            # Uniform path — supports (optional) nvcomp compression.
            self._compressor.run(
                self, src_block_ids=src_block_ids,
                dst_block_ids=dst_block_ids, op=transfer_op)
        return True
