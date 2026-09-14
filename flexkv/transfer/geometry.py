"""The typed edge geometry a ``StorageBackend`` reads in ``attach``.

``template.py`` compiles a *model*'s KV layout into per-group regions.  This
module is one level up: it names what a *worker* publishes about the edge it
owns, so an engine plugged into that edge reads one documented surface instead
of a bag of worker attributes.

The bag was a real hazard, not a stylistic one:

* ``chunk_size_in_bytes`` means the **CPU** chunk on ``CPUSSDDiskTransferWorker``
  and the **SSD** chunk on ``GDSTransferWorker``.  One name, two sides of the
  transfer.  ``NixlFileBackend`` reads it in both workers and happens to be
  correct only because its GPU branch reads ``gpu_chunk_sizes_in_bytes``
  instead -- nothing enforced that pairing.
* the same CPU quantity is published as ``block_stride_in_bytes`` by
  ``CPUSSDDiskTransferWorker`` and as ``cpu_block_stride_in_bytes`` by
  ``GPUCPUTransferWorker``.  A backend can only be written against one of them.
* a worker under a heterogeneous (multi-group) layout has *no* uniform chunk
  stride at all -- ``KVCacheLayout.get_chunk_size()`` raises there -- so those
  attributes are simply absent, and a backend that reads them gets an
  ``AttributeError`` from deep inside ``attach`` rather than a statement of
  what it does not support.

So the optionality here is the point.  ``strides is None`` says "this side has
no single (layer, kv) chunk", which is exactly the case every chunk-addressing
engine must refuse; ``require_strides`` turns that into one sentence naming the
engine.  A side that is absent entirely (``geometry.gpu`` on a CPU<->SSD edge)
says the edge does not have that side at all.

Nothing here computes geometry.  Workers still derive their own numbers -- from
``KVCacheLayout``, from ``template.compile_host_regions``, or from the real KV
tensors -- and then hand the result over.  This is the *shape* of that hand-off,
so the reading side has one place to look and one place to fail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeVar

import torch

from flexkv.common.storage import KVCacheLayout

_S = TypeVar("_S")


@dataclass(frozen=True)
class ChunkStrides:
    """Byte geometry of one uniform (layer, kv) chunk on one side of an edge.

    Only meaningful when every layer of every group has the same chunk size,
    i.e. not under a heterogeneous layout.  ``block_stride`` is the distance
    between consecutive blocks *of one chunk*, which is the whole block under
    BLOCKFIRST and one chunk under LAYERFIRST.
    """
    chunk_bytes: int
    kv_stride: int
    layer_stride: int
    block_stride: int


@dataclass(frozen=True)
class HostSide:
    """The CPU pool of an edge that has one.

    ``mapped_size`` is the HugePage mapping's aligned length, which is ``>=``
    the logical pool and is what an external RDMA registration must cover.
    ``None`` means the pool maps exactly what it holds, which is the case for
    a plain tensor.
    """
    layout: KVCacheLayout
    blocks: torch.Tensor
    layer_ptrs: torch.Tensor
    block_stride: int
    mapped_size: Optional[int] = None
    strides: Optional[ChunkStrides] = None

    def __post_init__(self) -> None:
        _check_block_stride(self.block_stride, self.strides, "CPU")

    def require_strides(self, who: str) -> ChunkStrides:
        return _require_strides(self.strides, who, "CPU")


@dataclass(frozen=True)
class DiskSide:
    """The local-file side of an edge, per file.

    ``block_stride`` is per file: an SSD block id is global across the file
    set, and both the native engines and the backends turn it into (file,
    block-in-file) before addressing.
    """
    block_stride: int
    strides: Optional[ChunkStrides] = None

    def __post_init__(self) -> None:
        _check_block_stride(self.block_stride, self.strides, "SSD")

    def require_strides(self, who: str) -> ChunkStrides:
        return _require_strides(self.strides, who, "SSD")


@dataclass(frozen=True)
class DeviceSide:
    """The GPU side of an edge, one entry per device in the TP group.

    ``blocks[d]`` is device ``d``'s KV tensors as the framework allocated
    them: one fused tensor, one per layer, or one per (layer, kv).  The
    strides are *measured from those tensors* wherever they could be
    recovered, because attention backends disagree on the 5D dim order and
    the declared layout records the logical shape rather than the allocated
    one.
    """
    blocks: List[List[torch.Tensor]]
    strides: Optional[Tuple[ChunkStrides, ...]] = None

    @property
    def num_gpus(self) -> int:
        return len(self.blocks)

    def require_strides(self, who: str) -> Tuple[ChunkStrides, ...]:
        if self.strides is None:
            raise ValueError(
                f"{who} needs uniform per-(layer, kv) GPU chunk strides, but "
                "this edge has a heterogeneous (multi-group) KV layout whose "
                "groups differ in chunk size"
            )
        return self.strides


@dataclass(frozen=True)
class EdgeGeometry:
    """What a worker publishes about its edge, for the engine plugged into it.

    ``bytes_per_block`` is the whole logical block across all layers -- the
    number transfer tracing divides by to report bandwidth.  It is what the
    *worker's native* engine moves; an engine that moves a different amount
    (PCFS flattening a BLOCKFIRST block, mooncake storing one opaque value)
    declares its own via ``StorageBackend.bytes_per_block`` rather than
    writing back onto the worker.  Backends used to do exactly that, which
    made a worker attribute mean different things depending on which engine
    had attached to it.
    """
    num_layers: int
    kv_dim: int
    num_kv_heads: int
    dtype: torch.dtype
    has_multi_group: bool
    bytes_per_block: int = 0
    cpu: Optional[HostSide] = None
    ssd: Optional[DiskSide] = None
    gpu: Optional[DeviceSide] = None

    def require_cpu(self, who: str) -> HostSide:
        return _require_side(self.cpu, who, "CPU pool")

    def require_ssd(self, who: str) -> DiskSide:
        return _require_side(self.ssd, who, "local SSD files")

    def require_gpu(self, who: str) -> DeviceSide:
        return _require_side(self.gpu, who, "GPU KV tensors")


def _check_block_stride(
    block_stride: int, strides: Optional[ChunkStrides], which: str
) -> None:
    """The two derivations of one number must agree.

    ``block_stride`` reaches a side both on its own and inside ``strides``,
    because a heterogeneous layout has the former and not the latter.  Where
    both exist they are the same quantity computed twice, so say so here
    rather than let a future edit make one of them drift silently.
    """
    if strides is None:
        return
    if strides.block_stride != block_stride:
        raise ValueError(
            f"{which} block stride disagrees with itself: {block_stride} on "
            f"the side, {strides.block_stride} in its chunk strides"
        )


def _require_strides(
    strides: Optional[ChunkStrides], who: str, which: str
) -> ChunkStrides:
    if strides is None:
        raise ValueError(
            f"{who} needs uniform per-(layer, kv) {which} chunk strides, but "
            "this edge has a heterogeneous (multi-group) KV layout whose "
            "groups differ in chunk size"
        )
    return strides


def _require_side(side: Optional[_S], who: str, which: str) -> _S:
    if side is None:
        raise ValueError(f"{who} needs the {which} of this edge, which has none")
    return side
