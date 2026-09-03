"""``EdgeGeometry``: what it *refuses*, which is the whole reason it exists.

Every worker builds one of these and every backend reads it, so the happy path
is covered wherever those meet (``test_mooncake_backend.py``,
``test_nixl_backend_e2e.py``). What is only covered here is the refusals --
the cases that used to surface as an ``AttributeError`` from deep inside
``attach``, or not at all:

  * a side the edge does not have (``geometry.gpu`` on a CPU<->remote edge)
  * a side with no uniform (layer, kv) chunk, i.e. a heterogeneous layout
  * a ``block_stride`` that disagrees with the one inside its own strides

The third is the one no error message could have named before: the same
quantity reaches a side twice, because a heterogeneous layout has the plain
number and not the strides, and nothing tied the two together.
"""
from __future__ import annotations

import pytest
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.transfer.geometry import (
    ChunkStrides,
    DeviceSide,
    DiskSide,
    EdgeGeometry,
    HostSide,
)


def _layout(num_block: int = 8) -> KVCacheLayout:
    return KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=4, num_block=num_block, tokens_per_block=16,
        num_head=8, head_size=128, kv_dim=2, num_kv_heads=8,
    )


def _strides(block_stride: int) -> ChunkStrides:
    return ChunkStrides(chunk_bytes=64, kv_stride=64, layer_stride=128,
                        block_stride=block_stride)


def _host(block_stride: int = 512, strides=None) -> HostSide:
    return HostSide(
        layout=_layout(),
        blocks=torch.zeros(4, dtype=torch.uint8),
        layer_ptrs=torch.zeros(1, dtype=torch.int64),
        block_stride=block_stride,
        strides=strides,
    )


def test_an_absent_side_names_itself_rather_than_raising_attributeerror():
    """A remote edge has no GPU and no SSD; asking must say so.

    The old contract was a bag of worker attributes, so a backend written
    against the wrong worker got ``AttributeError: 'CPURemoteTransferWorker'
    object has no attribute 'ssd_block_stride_in_bytes'`` from inside
    ``attach`` -- a stack trace naming a stride rather than the mismatch.
    """
    geometry = EdgeGeometry(
        num_layers=4, kv_dim=2, num_kv_heads=8, dtype=torch.bfloat16,
        has_multi_group=False, bytes_per_block=512, cpu=_host(),
    )
    assert geometry.require_cpu("some-engine") is geometry.cpu
    with pytest.raises(ValueError, match="local SSD files"):
        geometry.require_ssd("some-engine")
    with pytest.raises(ValueError, match="GPU KV tensors"):
        geometry.require_gpu("some-engine")


@pytest.mark.parametrize(
    "side, who",
    [
        (_host(), "CPU"),
        (DiskSide(block_stride=512), "SSD"),
        (DeviceSide(blocks=[[torch.zeros(1)]]), "GPU"),
    ],
    ids=["cpu", "ssd", "gpu"],
)
def test_a_heterogeneous_side_refuses_chunk_addressing_by_name(side, who):
    """``strides is None`` means "no single (layer, kv) chunk on this side".

    Every chunk-addressing engine (NIXL, PCFS) must refuse a heterogeneous
    layout, and the refusal has to name the engine: the operator's question is
    "why did enabling NIXL break my DSv4 run", not "what is a chunk stride".
    """
    with pytest.raises(ValueError, match=f"NIXL GDS_MT needs.*{who}"):
        side.require_strides("NIXL GDS_MT")


def test_block_stride_must_agree_with_the_one_inside_its_own_strides():
    """One quantity, derived twice, checked once.

    A side carries ``block_stride`` on its own *and* inside ``strides``,
    because a heterogeneous layout has the former and not the latter. Nothing
    stops a future worker edit from computing them differently, and the two
    feed different halves of the same backend -- so the disagreement is caught
    where it is created rather than as wrong bytes on disk.
    """
    with pytest.raises(ValueError, match="CPU block stride disagrees"):
        _host(block_stride=512, strides=_strides(256))
    with pytest.raises(ValueError, match="SSD block stride disagrees"):
        DiskSide(block_stride=512, strides=_strides(256))
    # Agreeing is not an error, obviously; assert it so the check cannot be
    # "fixed" by making it unconditional.
    _host(block_stride=512, strides=_strides(512))
    DiskSide(block_stride=512, strides=_strides(512))
