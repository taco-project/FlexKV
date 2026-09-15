"""Build a C++ ``RegionBatchGroup`` from compiled template geometry.

``template.py`` already turns a model's ``LayerGroupSpec`` list into
``HostRegion``/``GpuRegion`` numbers.  This is the other half: it turns those
numbers into the descriptors the C++ region-batch entry point wants, so a
worker can say "move these four regions" in one call instead of looping over
groups and paying a fan-out per group.

Why this and not another loop in the worker: the loop is where the model's
shape leaks into the transfer layer.  ``_transfer_impl`` iterating over
``tp_group_transfer_groups`` means the worker has to know that this model has
an indexer, that that one has SWA -- and every new model shape is a new branch
there.  A region list is the thing that varies, so the region list is what
should be passed down.

The C++ side is available only when the extension exposes ``RegionBatchGroup``;
older builds do not.  ``region_batch_available()`` is the check, and callers
keep their existing per-group path as the fallback.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

try:
    from flexkv import c_ext
except ImportError:  # no CUDA runtime here; geometry is still buildable
    c_ext = None


def region_batch_available() -> bool:
    """True when the built extension exposes the region-batch entry point."""
    return c_ext is not None and hasattr(c_ext, "RegionBatchGroup")


@dataclass
class RegionSpec:
    """One region's geometry, in the shape ``build_region_batch`` consumes.

    Mirrors the C++ ``RegionDesc`` field for field.  Kept as a plain
    dataclass rather than constructing ``c_ext.RegionDesc`` directly so the
    geometry can be built, asserted on and unit-tested on a machine with no
    CUDA extension at all.
    """
    name: str
    # host side (bytes)
    cpu_ptr: int
    cpu_kv_stride: int
    cpu_layer_stride: int
    cpu_block_stride: int
    cpu_tp_stride: int
    # device side, one entry per rank
    gpu_block_ptrs_flat: List[int]
    num_tensors_per_gpu: int
    gpu_kv_strides: List[int]
    gpu_block_strides: List[int]
    gpu_layer_strides: List[int]
    gpu_chunk_sizes: List[int]
    # shape
    num_layers: int
    kv_dim: int = 1
    num_kv_heads: int = 1

    def validate(self, num_ranks: int) -> None:
        """Raise on the mismatches C++ would otherwise report from a worker
        thread, where the message is harder to attribute to a region."""
        for field_name in ("gpu_kv_strides", "gpu_block_strides",
                           "gpu_layer_strides", "gpu_chunk_sizes"):
            values = getattr(self, field_name)
            if len(values) != num_ranks:
                raise ValueError(
                    f"region {self.name!r}: {field_name} has {len(values)} "
                    f"entries, expected {num_ranks} (one per rank)")
        expected_ptrs = num_ranks * self.num_tensors_per_gpu
        if len(self.gpu_block_ptrs_flat) != expected_ptrs:
            raise ValueError(
                f"region {self.name!r}: gpu_block_ptrs_flat has "
                f"{len(self.gpu_block_ptrs_flat)} entries, expected "
                f"{expected_ptrs}")
        if self.num_layers <= 0:
            raise ValueError(f"region {self.name!r}: num_layers must be positive")

    def to_desc(self) -> "c_ext.RegionDesc":
        d = c_ext.RegionDesc()
        d.name = self.name
        d.cpu_ptr = self.cpu_ptr
        d.cpu_kv_stride_in_bytes = self.cpu_kv_stride
        d.cpu_layer_stride_in_bytes = self.cpu_layer_stride
        d.cpu_block_stride_in_bytes = self.cpu_block_stride
        d.cpu_tp_stride_in_bytes = self.cpu_tp_stride
        d.gpu_block_ptrs_flat = list(self.gpu_block_ptrs_flat)
        d.num_tensors_per_gpu = self.num_tensors_per_gpu
        d.gpu_kv_strides_in_bytes = list(self.gpu_kv_strides)
        d.gpu_block_strides_in_bytes = list(self.gpu_block_strides)
        d.gpu_layer_strides_in_bytes = list(self.gpu_layer_strides)
        d.gpu_chunk_sizes_in_bytes = list(self.gpu_chunk_sizes)
        d.num_layers = self.num_layers
        d.kv_dim = self.kv_dim
        d.num_kv_heads = self.num_kv_heads
        return d


def build_region_batch(
    regions: Sequence[RegionSpec],
    gpu_device_ids: Sequence[int],
    *,
    ce_segment_threshold: int = 8,
    ce_path_opt: bool = True,
    ce_force_path: int = -1,
    ce_enable_memcpy2d: bool = False,
    is_blockfirst: bool = False,
    num_kv_heads: int = 1,
    ce_gather_threads: int = 4,
    ce_gather_nt: bool = True,
) -> "c_ext.RegionBatchGroup":
    """Construct the C++ group, validating every region first."""
    if not region_batch_available():
        raise RuntimeError(
            "this build of the flexkv extension has no RegionBatchGroup; "
            "rebuild, or use the per-group TPTransferThreadGroup path")
    if not regions:
        raise ValueError("build_region_batch: no regions")
    num_ranks = len(gpu_device_ids)
    for r in regions:
        r.validate(num_ranks)
    return c_ext.RegionBatchGroup(
        gpu_device_ids=[int(d) for d in gpu_device_ids],
        regions=[r.to_desc() for r in regions],
        ce_segment_threshold=ce_segment_threshold,
        ce_path_opt=ce_path_opt,
        ce_force_path=ce_force_path,
        ce_enable_memcpy2d=ce_enable_memcpy2d,
        is_blockfirst=is_blockfirst,
        num_kv_heads=num_kv_heads,
        ce_gather_threads=ce_gather_threads,
        ce_gather_nt=ce_gather_nt,
    )


def rank_share_mode(name: Optional[str]) -> "c_ext.RankShareMode":
    """Map the ``kv_shared_across_ranks_mode`` config string to the enum.

    Parsing happens in C++ so the accepted spellings and the
    degrade-to-``sharded`` fallback are stated once; this is just the call.
    """
    if not region_batch_available():
        raise RuntimeError(
            "this build of the flexkv extension has no RankShareMode")
    return c_ext.parse_rank_share_mode(name or "sharded")


def make_requests(
    num_regions: int,
    gpu_block_ids: torch.Tensor,
    cpu_block_ids: torch.Tensor,
    is_host_to_device: bool,
    *,
    transfer_num_cta: int = 4,
    use_ce_transfer: bool = False,
    backend: Optional["c_ext.TransferBackendKind"] = None,
    layer_id: int = 0,
    layer_granularity: int = 0,
    share_mode: Optional["c_ext.RankShareMode"] = None,
    designated_rank: int = 0,
    region_indices: Optional[Sequence[int]] = None,
) -> List["c_ext.RegionRequest"]:
    """One request per region, all moving the same blocks.

    The common case: a whole-block transfer touches every region of the block,
    with the same block ids and direction.  ``layer_granularity=0`` means "all
    layers of the region", which is what a non-layerwise transfer wants and
    which keeps a caller from having to know each region's layer count.

    ``region_indices`` restricts the batch to a subset of the registered
    regions -- an SWA-only D2H names just the SWA regions, and the block ids
    given are then that pool's slot ids.  Default is every region, in order.

    ``share_mode`` applies only to rank-shared regions (``num_kv_heads == 1``)
    on the way out; C++ ignores it everywhere else.
    """
    if not region_batch_available():
        raise RuntimeError(
            "this build of the flexkv extension has no RegionBatchGroup; "
            "make_requests has nothing to describe")
    kind = backend if backend is not None else c_ext.TransferBackendKind.AUTO
    mode = share_mode if share_mode is not None else c_ext.RankShareMode.SHARDED
    indices = range(num_regions) if region_indices is None else region_indices
    requests = []
    for gi in indices:
        req = c_ext.RegionRequest()
        req.region_index = gi
        req.gpu_block_id_tensor = gpu_block_ids
        req.cpu_block_id_tensor = cpu_block_ids
        req.layer_id = layer_id
        req.layer_granularity = layer_granularity
        req.is_host_to_device = is_host_to_device
        req.transfer_num_cta = transfer_num_cta
        req.backend = kind
        req.use_ce_transfer = use_ce_transfer
        req.rank_share_mode = mode
        req.designated_rank = designated_rank
        requests.append(req)
    return requests
