"""Multi-group + SWA per-layer H2D, through the ordinary region batch.

This is the shape production actually runs: a model whose layers are spread
over several groups (main KV, a smaller-head group, an indexer) plus an SWA
sidecar whose blocks live in a *different* slot-id space.  One original model
layer can therefore have members in two groups and in SWA at once, and the
consumer waits on a single fd for it.

There is no layerwise transfer group any more.  Both pools' regions go into one
``RegionBatchGroup``, and "per layer" is expressed by tagging each request with
the ``milestone_layer`` it closes -- so this file exercises the same
``submit_layerwise`` the worker calls, over the same region descriptors the
worker builds.  See ``_build_milestones`` for the one piece of index arithmetic
that used to live in C++.

Covers:
  - T6: main + indexer members per layer + SWA H2D byte correctness
  - T4: empty member layer (no main/indexer) but SWA still transfers

Run:
    pytest tests/test_layerwise_multi_group_swa.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from flexkv.common.config import LayerGroupSpec, build_layer_member_map
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.transfer.region_batch import (
    RegionSpec,
    build_region_batch,
    make_requests,
)

# -------------------------- shared geometry --------------------------
DEVICE_ID = 0
NUM_GPU_BLOCKS = 8
NUM_CPU_BLOCKS = 8
TOKENS_PER_BLOCK = 16
MAIN_HEAD_SIZE = 32
INDEXER_HEAD_SIZE = 16
SWA_BYTES_PER_TOKEN = 64
CPU_SRC = 2
GPU_DST = 5
SWA_CPU_SRC = 3
SWA_GPU_DST = 6


def _device() -> torch.device:
    return torch.device(f"cuda:{DEVICE_ID}")


def _make_group_gpu_tensors(
    g: LayerGroupSpec,
    num_blocks: int,
    tpb: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """VLLM-style: one GPU tensor per local layer."""
    tpb_g = tpb // g.compress_ratio
    return [
        torch.zeros(
            num_blocks,
            tpb_g,
            g.num_kv_heads,
            g.head_size,
            dtype=g.dtype,
            device=device,
        )
        for _ in range(g.num_layers)
    ]


def _make_gpu_layout(g: LayerGroupSpec, num_blocks: int, tpb: int) -> KVCacheLayout:
    tpb_g = tpb // g.compress_ratio
    return KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=g.num_layers,
        num_block=num_blocks,
        tokens_per_block=tpb_g,
        num_head=g.num_kv_heads,
        head_size=g.head_size,
        kv_dim=(1 if g.num_kv_heads == 1 else 2),
        num_kv_heads=g.num_kv_heads,
    )


def _make_multi_group_cpu_layout(
    layer_groups: Sequence[LayerGroupSpec],
    num_original_layers: int,
    num_cpu_blocks: int,
    tpb: int,
) -> KVCacheLayout:
    return KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=num_original_layers,
        num_block=num_cpu_blocks,
        tokens_per_block=tpb,
        num_head=1,
        head_size=MAIN_HEAD_SIZE,
        kv_dim=1,
        num_kv_heads=1,
        layer_groups=list(layer_groups),
        tp_size=1,
    )


def _compute_multi_group_strides(
    layer_groups: Sequence[LayerGroupSpec],
    cpu_kv_layout: KVCacheLayout,
    gpu_layouts_per_group: Sequence[KVCacheLayout],
    tp_size: int = 1,
) -> Dict[str, object]:
    """Mirror ``GPUCPUTransferWorker._init_tp_multi_group``'s stride table.

    Deliberately still computed here rather than imported from the worker: if
    the worker's arithmetic drifts, these tests should notice by producing
    wrong bytes, not by drifting along with it.
    """
    kv_dim = cpu_kv_layout.kv_dim
    tpb = cpu_kv_layout.tokens_per_block
    num_original_layers = cpu_kv_layout.num_layer

    layer_member_map = build_layer_member_map(layer_groups, num_original_layers)
    layer_members = [list(m) for m in layer_member_map.members]

    cpu_block_stride = cpu_kv_layout.get_block_stride()
    cpu_tp_stride = cpu_block_stride // tp_size

    group_num_layers: List[int] = []
    group_cpu_offset_bytes: List[int] = []
    group_cpu_layer_strides: List[int] = []
    group_cpu_kv_strides: List[int] = []
    group_chunk_sizes: List[int] = []
    group_cpu_block_strides: List[int] = []
    group_cpu_tp_strides: List[int] = []
    group_gpu_kv_strides: List[int] = []
    group_gpu_block_strides: List[int] = []
    group_gpu_layer_strides: List[int] = []
    group_gpu_chunk_sizes: List[int] = []

    offset_bytes = 0
    for gi, g in enumerate(layer_groups):
        dtype_size_g = g.dtype.itemsize
        tpb_g = tpb // g.compress_ratio
        chunk_elements = tpb_g * g.num_kv_heads * g.head_size
        chunk_size_bytes = chunk_elements * dtype_size_g
        layer_stride_bytes = kv_dim * chunk_size_bytes
        kv_stride_bytes = chunk_size_bytes

        group_num_layers.append(g.num_layers)
        group_cpu_offset_bytes.append(offset_bytes)
        group_cpu_layer_strides.append(layer_stride_bytes)
        group_cpu_kv_strides.append(kv_stride_bytes)
        group_chunk_sizes.append(chunk_size_bytes)
        group_cpu_block_strides.append(cpu_block_stride)
        group_cpu_tp_strides.append(cpu_tp_stride)

        layout = gpu_layouts_per_group[gi]
        group_gpu_kv_strides.append(layout.get_kv_stride() * dtype_size_g)
        group_gpu_block_strides.append(layout.get_block_stride() * dtype_size_g)
        group_gpu_layer_strides.append(layout.get_layer_stride() * dtype_size_g)
        group_gpu_chunk_sizes.append(layout.get_chunk_size() * dtype_size_g)

        offset_bytes += g.num_layers * layer_stride_bytes

    return dict(
        layer_members=layer_members,
        layer_member_map=layer_member_map,
        group_num_layers=group_num_layers,
        group_cpu_offset_bytes=group_cpu_offset_bytes,
        group_cpu_layer_strides=group_cpu_layer_strides,
        group_cpu_kv_strides=group_cpu_kv_strides,
        group_chunk_sizes=group_chunk_sizes,
        group_cpu_block_strides=group_cpu_block_strides,
        group_cpu_tp_strides=group_cpu_tp_strides,
        group_gpu_kv_strides=group_gpu_kv_strides,
        group_gpu_block_strides=group_gpu_block_strides,
        group_gpu_layer_strides=group_gpu_layer_strides,
        group_gpu_chunk_sizes=group_gpu_chunk_sizes,
        cpu_block_stride=cpu_block_stride,
    )


def _compute_swa_strides(
    swa_cpu_layout: KVCacheLayout,
    swa_gpu_layout: KVCacheLayout,
    dtype: torch.dtype = torch.uint8,
) -> Dict[str, object]:
    dtype_size = dtype.itemsize
    return dict(
        swa_cpu_chunk_size_in_bytes=swa_cpu_layout.get_chunk_size() * dtype_size,
        swa_cpu_block_stride_in_bytes=swa_cpu_layout.get_block_stride() * dtype_size,
        swa_cpu_kv_stride_in_bytes=swa_cpu_layout.get_kv_stride() * dtype_size,
        swa_cpu_layer_stride_in_bytes=swa_cpu_layout.get_layer_stride() * dtype_size,
        swa_cpu_tp_stride_in_bytes=(
            swa_cpu_layout.get_block_stride() * dtype_size
        ),
        swa_gpu_kv_stride_in_bytes=swa_gpu_layout.get_kv_stride() * dtype_size,
        swa_gpu_block_stride_in_bytes=swa_gpu_layout.get_block_stride() * dtype_size,
        swa_gpu_layer_stride_in_bytes=swa_gpu_layout.get_layer_stride() * dtype_size,
        swa_gpu_chunk_size_in_bytes=swa_gpu_layout.get_chunk_size() * dtype_size,
    )


def _seed_main_cpu_layer(
    cpu_blocks: torch.Tensor,
    cpu_block_id: int,
    orig_layer: int,
    group_idx: int,
    local_layer_id: int,
    strides: Dict[str, object],
    layer_groups: Sequence[LayerGroupSpec],
    tpb: int,
) -> torch.Tensor:
    """Write a deterministic pattern into the multi-group CPU block; return expected GPU slice."""
    g = layer_groups[group_idx]
    tpb_g = tpb // g.compress_ratio
    block_stride = strides["cpu_block_stride"]  # type: ignore[index]
    base = (
        cpu_block_id * block_stride
        + strides["group_cpu_offset_bytes"][group_idx]  # type: ignore[index]
        + local_layer_id * strides["group_cpu_layer_strides"][group_idx]  # type: ignore[index]
    )

    expected = torch.zeros(
        tpb_g, g.num_kv_heads, g.head_size, dtype=g.dtype,
    )
    flat_cpu = cpu_blocks.view(-1)
    for tok in range(tpb_g):
        for h in range(g.num_kv_heads):
            for b in range(g.head_size):
                val = (
                    (orig_layer * 100 + group_idx * 10 + local_layer_id * 3 + tok) ^ b
                ) & 0xFF
                expected[tok, h, b] = val
                byte_off = base + tok * g.num_kv_heads * g.head_size + h * g.head_size + b
                flat_cpu[byte_off] = val
    return expected


def _seed_swa_cpu_layer(
    swa_cpu: torch.Tensor,
    cpu_block_id: int,
    orig_layer: int,
    salt: int = 0xB2,
) -> torch.Tensor:
    """Seed one SWA layer inside LAYERFIRST CPU pool; return expected GPU bytes."""
    tpb = swa_cpu.shape[3]
    head_size = swa_cpu.shape[5]
    expected = torch.zeros(tpb, 1, head_size, dtype=torch.uint8)
    for tok in range(tpb):
        for b in range(head_size):
            val = ((orig_layer * 31 + tok + salt) ^ b) & 0xFF
            expected[tok, 0, b] = val
            swa_cpu[orig_layer, 0, cpu_block_id, tok, 0, b] = val
    return expected


def _main_region_specs(
    layer_groups: Sequence[LayerGroupSpec],
    strides: Dict[str, object],
    cpu_blocks: torch.Tensor,
    gpu_tensors_per_group: Sequence[List[torch.Tensor]],
) -> List[RegionSpec]:
    """One region per layer group, in group order -- as ``_init_tp_multi_group``."""
    specs: List[RegionSpec] = []
    cpu_base = cpu_blocks.view(-1).data_ptr()
    for gi, g in enumerate(layer_groups):
        specs.append(RegionSpec(
            name=f"full_kv.group{gi}",
            cpu_ptr=cpu_base + strides["group_cpu_offset_bytes"][gi],  # type: ignore[index]
            cpu_kv_stride=strides["group_cpu_kv_strides"][gi],  # type: ignore[index]
            cpu_layer_stride=strides["group_cpu_layer_strides"][gi],  # type: ignore[index]
            cpu_block_stride=strides["group_cpu_block_strides"][gi],  # type: ignore[index]
            cpu_tp_stride=strides["group_cpu_tp_strides"][gi],  # type: ignore[index]
            gpu_block_ptrs_flat=[t.data_ptr() for t in gpu_tensors_per_group[gi]],
            num_tensors_per_gpu=g.num_layers,
            gpu_kv_strides=[strides["group_gpu_kv_strides"][gi]],  # type: ignore[index]
            gpu_block_strides=[strides["group_gpu_block_strides"][gi]],  # type: ignore[index]
            gpu_layer_strides=[strides["group_gpu_layer_strides"][gi]],  # type: ignore[index]
            gpu_chunk_sizes=[strides["group_gpu_chunk_sizes"][gi]],  # type: ignore[index]
            num_layers=g.num_layers,
            kv_dim=1,
            num_kv_heads=1,
        ))
    return specs


def _swa_region_spec(
    num_original_layers: int,
    swa_strides: Dict[str, object],
    swa_cpu: torch.Tensor,
    swa_gpu_tensors: Sequence[torch.Tensor],
) -> RegionSpec:
    """SWA is one more region, not a sidecar with its own transfer entry point."""
    return RegionSpec(
        name="swa.group0",
        cpu_ptr=swa_cpu.view(-1).data_ptr(),
        cpu_kv_stride=swa_strides["swa_cpu_kv_stride_in_bytes"],  # type: ignore[index]
        cpu_layer_stride=swa_strides["swa_cpu_layer_stride_in_bytes"],  # type: ignore[index]
        cpu_block_stride=swa_strides["swa_cpu_block_stride_in_bytes"],  # type: ignore[index]
        cpu_tp_stride=swa_strides["swa_cpu_tp_stride_in_bytes"],  # type: ignore[index]
        gpu_block_ptrs_flat=[t.data_ptr() for t in swa_gpu_tensors],
        num_tensors_per_gpu=num_original_layers,
        gpu_kv_strides=[swa_strides["swa_gpu_kv_stride_in_bytes"]],  # type: ignore[index]
        gpu_block_strides=[swa_strides["swa_gpu_block_stride_in_bytes"]],  # type: ignore[index]
        gpu_layer_strides=[swa_strides["swa_gpu_layer_stride_in_bytes"]],  # type: ignore[index]
        gpu_chunk_sizes=[swa_strides["swa_gpu_chunk_size_in_bytes"]],  # type: ignore[index]
        num_layers=num_original_layers,
        kv_dim=1,
        num_kv_heads=1,
    )


def _build_milestones(
    layer_members: Sequence[Sequence[Tuple[int, int]]],
    num_original_layers: int,
    *,
    has_swa: bool,
) -> List[List[Tuple[bool, int, int]]]:
    """``(is_swa, global_region_index, local_layer)`` per original layer.

    The test-side twin of ``GPUCPUTransferWorker._build_layer_milestones``.
    Main-KV regions are indices ``0..num_groups-1`` (group ordinal == region
    ordinal), and SWA -- when present -- is the region right after them, which
    is the order ``_build_region_batch`` lays the two pools out in.

    ``is_swa`` rides along because the two pools draw block ids from different
    slot-id spaces, so a request has to know which id tensor to read.
    """
    num_groups = max((gi for members in layer_members for gi, _ in members),
                     default=-1) + 1
    milestones: List[List[Tuple[bool, int, int]]] = []
    for layer in range(num_original_layers):
        members: List[Tuple[bool, int, int]] = [
            (False, gi, local) for gi, local in layer_members[layer]
        ]
        if has_swa:
            # SWA's own layer numbering is the model's: one SWA layer per
            # original layer, so local == layer.
            members.append((True, num_groups, layer))
        milestones.append(members)
    return milestones


@dataclass
class MultiGroupFixture:
    group: "object"  # c_ext.RegionBatchGroup
    layer_groups: List[LayerGroupSpec]
    gpu_tensors_per_group: List[List[torch.Tensor]]
    cpu_blocks: torch.Tensor
    strides: Dict[str, object]
    num_original_layers: int
    tokens_per_block: int = TOKENS_PER_BLOCK
    milestones: List[List[Tuple[bool, int, int]]] = field(default_factory=list)
    swa_gpu_tensors: Optional[List[torch.Tensor]] = None
    swa_cpu: Optional[torch.Tensor] = None
    swa_strides: Optional[Dict[str, object]] = None


def build_fixture(
    layer_groups: List[LayerGroupSpec],
    num_original_layers: int,
    *,
    has_swa: bool = True,
    tokens_per_block: int = TOKENS_PER_BLOCK,
    num_cpu_blocks: int = NUM_CPU_BLOCKS,
    num_gpu_blocks: int = NUM_GPU_BLOCKS,
    swa_bytes_per_token: int = SWA_BYTES_PER_TOKEN,
) -> MultiGroupFixture:
    """Both pools' regions in one batch, exactly as the worker builds them.

    The geometry defaults are this module's own tests'; DSv4-shaped callers
    pass their own rather than rebuilding the region specs by hand.
    """
    device = _device()
    tpb = tokens_per_block
    cpu_layout = _make_multi_group_cpu_layout(
        layer_groups, num_original_layers, num_cpu_blocks, tpb,
    )
    gpu_layouts = [
        _make_gpu_layout(g, num_gpu_blocks, tpb) for g in layer_groups
    ]
    strides = _compute_multi_group_strides(layer_groups, cpu_layout, gpu_layouts)

    gpu_tensors_per_group = [
        _make_group_gpu_tensors(g, num_gpu_blocks, tpb, device)
        for g in layer_groups
    ]

    block_stride = cpu_layout.get_block_stride()
    cpu_blocks = torch.zeros(
        num_cpu_blocks, block_stride, dtype=torch.uint8, pin_memory=True,
    )

    specs = _main_region_specs(
        layer_groups, strides, cpu_blocks, gpu_tensors_per_group)

    swa_gpu_tensors: Optional[List[torch.Tensor]] = None
    swa_cpu: Optional[torch.Tensor] = None
    swa_strides: Optional[Dict[str, object]] = None
    if has_swa:
        swa_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=num_original_layers, num_block=num_cpu_blocks,
            tokens_per_block=tpb, num_head=1, head_size=swa_bytes_per_token,
            kv_dim=1, num_kv_heads=1,
        )
        swa_gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=num_original_layers, num_block=num_gpu_blocks,
            tokens_per_block=tpb, num_head=1, head_size=swa_bytes_per_token,
            kv_dim=1, num_kv_heads=1,
        )
        swa_strides = _compute_swa_strides(swa_layout, swa_gpu_layout)
        swa_cpu = torch.zeros(
            swa_layout.kv_shape, dtype=torch.uint8, pin_memory=True,
        )
        swa_gpu_tensors = [
            torch.zeros(
                num_gpu_blocks,
                tpb,
                1,
                swa_bytes_per_token,
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(num_original_layers)
        ]
        specs.append(_swa_region_spec(
            num_original_layers, swa_strides, swa_cpu, swa_gpu_tensors))

    group = build_region_batch(
        specs, [DEVICE_ID],
        is_blockfirst=True,
        num_kv_heads=1,
    )

    return MultiGroupFixture(
        group=group,
        layer_groups=layer_groups,
        gpu_tensors_per_group=gpu_tensors_per_group,
        cpu_blocks=cpu_blocks,
        strides=strides,
        num_original_layers=num_original_layers,
        tokens_per_block=tpb,
        milestones=_build_milestones(
            strides["layer_members"],  # type: ignore[arg-type]
            num_original_layers, has_swa=has_swa),
        swa_gpu_tensors=swa_gpu_tensors,
        swa_cpu=swa_cpu,
        swa_strides=swa_strides,
    )


# Old name, kept because three test modules call it.


def h2d_requests(
    fx: MultiGroupFixture,
    *,
    with_swa: bool,
    cpu_block: int = CPU_SRC,
    gpu_block: int = GPU_DST,
    swa_cpu_block: int = SWA_CPU_SRC,
    swa_gpu_block: int = SWA_GPU_DST,
) -> Tuple[List["object"], List[int]]:
    """The per-layer request list and the layers nothing will be launched for.

    Same construction as ``GPUCPUTransferWorker._layerwise_transfer_impl``: one
    request per (region, layer) member, tagged with the original layer it
    closes, so a layer spanning main KV and SWA posts once, after both landed.
    """
    gpu_dst = torch.tensor([gpu_block], dtype=torch.int64)
    cpu_src = torch.tensor([cpu_block], dtype=torch.int64)
    swa_gpu_dst = torch.tensor([swa_gpu_block], dtype=torch.int64)
    swa_cpu_src = torch.tensor([swa_cpu_block], dtype=torch.int64)

    requests: List["object"] = []
    empty_layers: List[int] = []
    for layer, members in enumerate(fx.milestones):
        live = [m for m in members if with_swa or not m[0]]
        if not live:
            empty_layers.append(layer)
            continue
        for is_swa, region_index, local_layer in live:
            req = make_requests(
                fx.group.num_regions,
                swa_gpu_dst if is_swa else gpu_dst,
                swa_cpu_src if is_swa else cpu_src,
                True,
                transfer_num_cta=4,
                use_ce_transfer=True,
                region_indices=[region_index],
                layer_id=local_layer,
                layer_granularity=1,
            )[0]
            req.milestone_layer = layer
            requests.append(req)
    return requests, empty_layers


def _run_h2d(
    fx: MultiGroupFixture,
    *,
    with_swa: bool = False,
    counter_id: int = 0,
    **blocks: int,
) -> None:
    requests, empty_layers = h2d_requests(fx, with_swa=with_swa, **blocks)
    fx.group.submit_layerwise(requests, empty_layers, counter_id)
    ok, err = fx.group.wait_layer_completion(120.0)
    assert ok, f"layerwise H2D did not complete: {err}"
    torch.cuda.synchronize()


def _assert_main_gpu_matches(
    fx: MultiGroupFixture,
    orig_layer: int,
    group_idx: int,
    local_layer_id: int,
    expected: torch.Tensor,
) -> None:
    actual = fx.gpu_tensors_per_group[group_idx][local_layer_id][GPU_DST].cpu()
    assert torch.equal(actual, expected), (
        f"main group={group_idx} orig={orig_layer} local={local_layer_id} mismatch"
    )


def _assert_swa_gpu_matches(
    fx: MultiGroupFixture,
    orig_layer: int,
    expected: torch.Tensor,
) -> None:
    assert fx.swa_gpu_tensors is not None
    actual = fx.swa_gpu_tensors[orig_layer][SWA_GPU_DST].cpu()
    assert torch.equal(actual, expected), f"SWA orig={orig_layer} mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestLayerwiseMultiGroupSwa:
    def test_multi_group_main_indexer_and_swa_h2d(self) -> None:
        """T6: two members (main + indexer) per layer plus SWA on every layer."""
        num_layers = 4
        main = LayerGroupSpec(
            num_layers=num_layers,
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=list(range(num_layers)),
            dtype=torch.uint8,
        )
        indexer = LayerGroupSpec(
            num_layers=3,
            num_kv_heads=1,
            head_size=INDEXER_HEAD_SIZE,
            layer_indices=[1, 2, 3],
            dtype=torch.uint8,
        )
        fx = build_fixture([main, indexer], num_layers)

        expected_main: Dict[Tuple[int, int, int], torch.Tensor] = {}
        member_map = fx.strides["layer_member_map"]
        for orig in range(num_layers):
            for gi, local_id in member_map.members_of(orig):  # type: ignore[union-attr]
                expected_main[(orig, gi, local_id)] = _seed_main_cpu_layer(
                    fx.cpu_blocks,
                    CPU_SRC,
                    orig,
                    gi,
                    local_id,
                    fx.strides,
                    fx.layer_groups,
                    TOKENS_PER_BLOCK,
                )

        expected_swa = {
            orig: _seed_swa_cpu_layer(fx.swa_cpu, SWA_CPU_SRC, orig)  # type: ignore[arg-type]
            for orig in range(num_layers)
        }

        _run_h2d(fx, with_swa=True)

        for (orig, gi, local_id), exp in expected_main.items():
            _assert_main_gpu_matches(fx, orig, gi, local_id, exp)
        for orig, exp in expected_swa.items():
            _assert_swa_gpu_matches(fx, orig, exp)

    def test_empty_member_layer_swa_only_h2d(self) -> None:
        """T4: layer 0 has no main/indexer member; SWA still copies that layer."""
        num_layers = 4
        main = LayerGroupSpec(
            num_layers=3,
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=[1, 2, 3],
            dtype=torch.uint8,
        )
        fx = build_fixture([main], num_layers)

        member_map = fx.strides["layer_member_map"]
        assert member_map.members_of(0) == (), "layer 0 must have empty members"

        expected_main: Dict[Tuple[int, int, int], torch.Tensor] = {}
        for orig in range(1, num_layers):
            for gi, local_id in member_map.members_of(orig):
                expected_main[(orig, gi, local_id)] = _seed_main_cpu_layer(
                    fx.cpu_blocks,
                    CPU_SRC,
                    orig,
                    gi,
                    local_id,
                    fx.strides,
                    fx.layer_groups,
                    TOKENS_PER_BLOCK,
                )

        expected_swa = {
            orig: _seed_swa_cpu_layer(fx.swa_cpu, SWA_CPU_SRC, orig)  # type: ignore[arg-type]
            for orig in range(num_layers)
        }

        _run_h2d(fx, with_swa=True)

        for (orig, gi, local_id), exp in expected_main.items():
            _assert_main_gpu_matches(fx, orig, gi, local_id, exp)
        for orig, exp in expected_swa.items():
            _assert_swa_gpu_matches(fx, orig, exp)
