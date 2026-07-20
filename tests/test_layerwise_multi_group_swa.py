"""Unit tests for multi-group LayerwiseTransferGroup + SWA sidecar fusion.

Covers:
  - T6: main + indexer members per layer + SWA H2D byte correctness
  - T4: empty member layer (no main/indexer) but SWA still transfers

Run:
    pytest tests/test_layerwise_multi_group_swa.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from flexkv.c_ext import LayerwiseTransferGroup
from flexkv.common.config import LayerGroupSpec, build_layer_member_map
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

# -------------------------- shared geometry --------------------------
DEVICE_ID = 0
NUM_GPU_BLOCKS = 8
NUM_CPU_BLOCKS = 8
TOKENS_PER_BLOCK = 16
MAIN_HEAD_SIZE = 32
INDEXER_HEAD_SIZE = 16
SWA_BYTES_PER_TOKEN = 64
IOURING_ENTRIES = 512
IOURING_FLAGS = 0

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
        is_mla=(g.num_kv_heads == 1),
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
        is_mla=True,
        layer_groups=list(layer_groups),
        tp_size=1,
    )


def _make_swa_layout(num_layers: int, num_blocks: int, tpb: int) -> KVCacheLayout:
    return KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers,
        num_block=num_blocks,
        tokens_per_block=tpb,
        num_head=1,
        head_size=SWA_BYTES_PER_TOKEN,
        is_mla=True,
    )


def _compute_multi_group_strides(
    layer_groups: Sequence[LayerGroupSpec],
    cpu_kv_layout: KVCacheLayout,
    gpu_layouts_per_group: Sequence[KVCacheLayout],
    tp_size: int = 1,
) -> Dict[str, object]:
    """Mirror flexkv.transfer.layerwise.LayerwiseWorker._init_multi_group strides."""
    kv_dim = 1 if cpu_kv_layout.is_mla else 2
    tpb = cpu_kv_layout.tokens_per_block
    num_original_layers = cpu_kv_layout.num_layer

    layer_member_map = build_layer_member_map(layer_groups, num_original_layers)
    layer_members = [list(m) for m in layer_member_map.members]

    cpu_block_stride = cpu_kv_layout.get_block_stride()
    cpu_tp_stride = cpu_block_stride // tp_size

    group_num_layers: List[int] = []
    group_cpu_offset_bytes: List[int] = []
    group_ssd_offset_bytes: List[int] = []
    group_cpu_layer_strides: List[int] = []
    group_cpu_kv_strides: List[int] = []
    group_ssd_layer_strides: List[int] = []
    group_ssd_kv_strides: List[int] = []
    group_chunk_sizes: List[int] = []
    group_h2d_cpu_kv_strides: List[int] = []
    group_h2d_cpu_layer_strides: List[int] = []
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
        group_ssd_offset_bytes.append(offset_bytes)
        group_cpu_layer_strides.append(layer_stride_bytes)
        group_cpu_kv_strides.append(kv_stride_bytes)
        group_ssd_layer_strides.append(layer_stride_bytes)
        group_ssd_kv_strides.append(kv_stride_bytes)
        group_chunk_sizes.append(chunk_size_bytes)
        group_h2d_cpu_kv_strides.append(kv_stride_bytes)
        group_h2d_cpu_layer_strides.append(layer_stride_bytes)
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
        group_ssd_offset_bytes=group_ssd_offset_bytes,
        group_cpu_layer_strides=group_cpu_layer_strides,
        group_cpu_kv_strides=group_cpu_kv_strides,
        group_ssd_layer_strides=group_ssd_layer_strides,
        group_ssd_kv_strides=group_ssd_kv_strides,
        group_chunk_sizes=group_chunk_sizes,
        group_h2d_cpu_kv_strides=group_h2d_cpu_kv_strides,
        group_h2d_cpu_layer_strides=group_h2d_cpu_layer_strides,
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
        swa_h2d_cpu_kv_stride_in_bytes=swa_cpu_layout.get_kv_stride() * dtype_size,
        swa_h2d_cpu_layer_stride_in_bytes=swa_cpu_layout.get_layer_stride() * dtype_size,
        swa_cpu_tp_stride_in_bytes=(
            swa_cpu_layout.get_block_stride() * dtype_size
        ),
        swa_gpu_kv_strides_tensor=torch.tensor(
            [swa_gpu_layout.get_kv_stride() * dtype_size], dtype=torch.int64),
        swa_gpu_block_strides_tensor=torch.tensor(
            [swa_gpu_layout.get_block_stride() * dtype_size], dtype=torch.int64),
        swa_gpu_layer_strides_tensor=torch.tensor(
            [swa_gpu_layout.get_layer_stride() * dtype_size], dtype=torch.int64),
        swa_gpu_chunk_sizes_tensor=torch.tensor(
            [swa_gpu_layout.get_chunk_size() * dtype_size], dtype=torch.int64),
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
    chunk_bytes = strides["group_chunk_sizes"][group_idx]  # type: ignore[index]
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


@dataclass
class MultiGroupFixture:
    group: LayerwiseTransferGroup
    layer_groups: List[LayerGroupSpec]
    gpu_tensors_per_group: List[List[torch.Tensor]]
    cpu_blocks: torch.Tensor
    strides: Dict[str, object]
    num_original_layers: int
    swa_gpu_tensors: Optional[List[torch.Tensor]] = None
    swa_cpu: Optional[torch.Tensor] = None
    swa_strides: Optional[Dict[str, object]] = None


def _build_fixture(
    layer_groups: List[LayerGroupSpec],
    num_original_layers: int,
) -> MultiGroupFixture:
    device = _device()
    cpu_layout = _make_multi_group_cpu_layout(
        layer_groups, num_original_layers, NUM_CPU_BLOCKS, TOKENS_PER_BLOCK,
    )
    gpu_layouts = [
        _make_gpu_layout(g, NUM_GPU_BLOCKS, TOKENS_PER_BLOCK) for g in layer_groups
    ]
    strides = _compute_multi_group_strides(layer_groups, cpu_layout, gpu_layouts)

    gpu_tensors_per_group = [
        _make_group_gpu_tensors(g, NUM_GPU_BLOCKS, TOKENS_PER_BLOCK, device)
        for g in layer_groups
    ]
    gpu_blocks_per_group = [[tensors] for tensors in gpu_tensors_per_group]

    block_stride = cpu_layout.get_block_stride()
    cpu_blocks = torch.zeros(
        NUM_CPU_BLOCKS, block_stride, dtype=torch.uint8, pin_memory=True,
    )

    empty_eventfds = torch.empty(0, dtype=torch.int32)
    ssd_files: Dict[int, List[str]] = {}

    swa_gpu_tensors: Optional[List[torch.Tensor]] = None
    swa_cpu: Optional[torch.Tensor] = None
    swa_strides: Optional[Dict[str, object]] = None

    common_ctor = dict(
        num_gpus=1,
        gpu_blocks_per_group=gpu_blocks_per_group,
        cpu_blocks=cpu_blocks,
        ssd_files=ssd_files,
        num_original_layers=num_original_layers,
        layer_members=strides["layer_members"],
        group_num_layers=strides["group_num_layers"],
        group_cpu_offset_bytes=strides["group_cpu_offset_bytes"],
        group_ssd_offset_bytes=strides["group_ssd_offset_bytes"],
        group_cpu_layer_strides=strides["group_cpu_layer_strides"],
        group_cpu_kv_strides=strides["group_cpu_kv_strides"],
        group_ssd_layer_strides=strides["group_ssd_layer_strides"],
        group_ssd_kv_strides=strides["group_ssd_kv_strides"],
        group_chunk_sizes=strides["group_chunk_sizes"],
        group_h2d_cpu_kv_strides=strides["group_h2d_cpu_kv_strides"],
        group_h2d_cpu_layer_strides=strides["group_h2d_cpu_layer_strides"],
        group_cpu_block_strides=strides["group_cpu_block_strides"],
        group_cpu_tp_strides=strides["group_cpu_tp_strides"],
        group_gpu_kv_strides=strides["group_gpu_kv_strides"],
        group_gpu_block_strides=strides["group_gpu_block_strides"],
        group_gpu_layer_strides=strides["group_gpu_layer_strides"],
        group_gpu_chunk_sizes=strides["group_gpu_chunk_sizes"],
        iouring_entries=IOURING_ENTRIES,
        iouring_flags=IOURING_FLAGS,
        layer_eventfds_tensor=empty_eventfds,
        tp_size=1,
        is_blockfirst=True,
        is_mla=True,
    )

    swa_layout = _make_swa_layout(
        num_original_layers, NUM_CPU_BLOCKS, TOKENS_PER_BLOCK,
    )
    swa_gpu_layout = _make_swa_layout(
        num_original_layers, NUM_GPU_BLOCKS, TOKENS_PER_BLOCK,
    )
    swa_strides = _compute_swa_strides(swa_layout, swa_gpu_layout)
    swa_cpu = torch.zeros(
        swa_layout.kv_shape, dtype=torch.uint8, pin_memory=True,
    )
    swa_gpu_tensors = [
        torch.zeros(
            NUM_GPU_BLOCKS,
            TOKENS_PER_BLOCK,
            1,
            SWA_BYTES_PER_TOKEN,
            dtype=torch.uint8,
            device=device,
        )
        for _ in range(num_original_layers)
    ]
    group = LayerwiseTransferGroup(
        **common_ctor,
        has_swa=True,
        swa_gpu_blocks=[swa_gpu_tensors],
        swa_cpu_blocks=swa_cpu,
        swa_ssd_files=ssd_files,
        swa_gpu_kv_strides_tensor=swa_strides["swa_gpu_kv_strides_tensor"],
        swa_gpu_block_strides_tensor=swa_strides["swa_gpu_block_strides_tensor"],
        swa_gpu_layer_strides_tensor=swa_strides["swa_gpu_layer_strides_tensor"],
        swa_gpu_chunk_sizes_tensor=swa_strides["swa_gpu_chunk_sizes_tensor"],
    )

    return MultiGroupFixture(
        group=group,
        layer_groups=layer_groups,
        gpu_tensors_per_group=gpu_tensors_per_group,
        cpu_blocks=cpu_blocks,
        strides=strides,
        num_original_layers=num_original_layers,
        swa_gpu_tensors=swa_gpu_tensors,
        swa_cpu=swa_cpu,
        swa_strides=swa_strides,
    )


def _run_h2d(
    fx: MultiGroupFixture,
    *,
    with_swa: bool = False,
) -> None:
    empty = torch.empty(0, dtype=torch.int64)
    gpu_dst = torch.tensor([GPU_DST], dtype=torch.int64)
    cpu_src = torch.tensor([CPU_SRC], dtype=torch.int64)

    swa_kwargs: Dict[str, object] = {}
    if with_swa:
        assert fx.swa_strides is not None
        swa_kwargs = dict(
            swa_h2d_src=torch.tensor([SWA_CPU_SRC], dtype=torch.int64),
            swa_h2d_dst=torch.tensor([SWA_GPU_DST], dtype=torch.int64),
            swa_disk2h_src=empty,
            swa_disk2h_dst=empty,
            swa_cpu_kv_stride_in_bytes=fx.swa_strides["swa_cpu_kv_stride_in_bytes"],
            swa_cpu_layer_stride_in_bytes=fx.swa_strides[
                "swa_cpu_layer_stride_in_bytes"
            ],
            swa_cpu_block_stride_in_bytes=fx.swa_strides[
                "swa_cpu_block_stride_in_bytes"
            ],
            swa_cpu_chunk_size_in_bytes=fx.swa_strides["swa_cpu_chunk_size_in_bytes"],
            swa_h2d_cpu_kv_stride_in_bytes=fx.swa_strides[
                "swa_h2d_cpu_kv_stride_in_bytes"
            ],
            swa_h2d_cpu_layer_stride_in_bytes=fx.swa_strides[
                "swa_h2d_cpu_layer_stride_in_bytes"
            ],
            swa_cpu_tp_stride_in_bytes=fx.swa_strides["swa_cpu_tp_stride_in_bytes"],
            swa_ssd_layer_stride_in_bytes=0,
            swa_ssd_kv_stride_in_bytes=0,
            swa_num_blocks_per_file=0,
        )

    fx.group.layerwise_transfer_multi_group(
        empty,
        empty,
        num_blocks_per_file=0,
        round_robin=1,
        num_threads_per_device=4,
        gpu_block_id_tensor=gpu_dst,
        cpu_block_id_tensor=cpu_src,
        transfer_cta_num=4,
        use_ce_transfer=True,
        is_mla=True,
        counter_id=0,
        **swa_kwargs,
    )
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
        fx = _build_fixture([main, indexer], num_layers)

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
        fx = _build_fixture([main], num_layers)

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
