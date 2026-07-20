"""DSv4-style layerwise multi-group + SWA roundtrip correctness test.

Simulates the production data path:

  1. PUT phase (D2H): seed GPU -> write main KV (c4 / c128 / indexer groups) + SWA to CPU
  2. Zero GPU pools (prove the next step reads from CPU, not stale GPU)
  3. GET phase (layerwise): ``layerwise_transfer_multi_group`` H2D for main KV + SWA
  4. Byte-exact compare restored GPU blocks against the original seed

Geometry mirrors DeepSeek V4:
  - ``c4`` group: compress_ratio=4 on CSA layers
  - ``c128`` group: compress_ratio=128 on HCA layers
  - ``c4_indexer`` group: indexer K on CSA layers (uint8)
  - SWA sidecar: all original layers, independent LAYERFIRST pool

Run:
    pytest tests/test_layerwise_dsv4_multi_group_swa_roundtrip.py -v
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from flexkv.c_ext import LayerwiseTransferGroup, transfer_kv_blocks
from flexkv.common.config import LayerGroupSpec, build_layer_member_map
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

from test_layerwise_multi_group_swa import (
    MultiGroupFixture,
    _compute_multi_group_strides,
    _device,
    _make_gpu_layout,
    _make_group_gpu_tensors,
    _make_multi_group_cpu_layout,
)

IOURING_ENTRIES = 512
IOURING_FLAGS = 0


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


def _compute_swa_strides(
    swa_cpu_layout: KVCacheLayout,
    swa_gpu_layout: KVCacheLayout,
) -> Dict[str, object]:
    dtype_size = torch.uint8.itemsize
    return dict(
        swa_cpu_chunk_size_in_bytes=swa_cpu_layout.get_chunk_size() * dtype_size,
        swa_cpu_block_stride_in_bytes=swa_cpu_layout.get_block_stride() * dtype_size,
        swa_cpu_kv_stride_in_bytes=swa_cpu_layout.get_kv_stride() * dtype_size,
        swa_cpu_layer_stride_in_bytes=swa_cpu_layout.get_layer_stride() * dtype_size,
        swa_h2d_cpu_kv_stride_in_bytes=swa_cpu_layout.get_kv_stride() * dtype_size,
        swa_h2d_cpu_layer_stride_in_bytes=swa_cpu_layout.get_layer_stride() * dtype_size,
        swa_cpu_tp_stride_in_bytes=swa_cpu_layout.get_block_stride() * dtype_size,
        swa_gpu_kv_strides_tensor=torch.tensor(
            [swa_gpu_layout.get_kv_stride() * dtype_size], dtype=torch.int64),
        swa_gpu_block_strides_tensor=torch.tensor(
            [swa_gpu_layout.get_block_stride() * dtype_size], dtype=torch.int64),
        swa_gpu_layer_strides_tensor=torch.tensor(
            [swa_gpu_layout.get_layer_stride() * dtype_size], dtype=torch.int64),
        swa_gpu_chunk_sizes_tensor=torch.tensor(
            [swa_gpu_layout.get_chunk_size() * dtype_size], dtype=torch.int64),
    )

# -------------------------- DSv4-like geometry --------------------------
DEVICE_ID = 0
NUM_GPU_BLOCKS = 8
NUM_CPU_BLOCKS = 8
TOKENS_PER_BLOCK = 128  # divisible by 4 and 128

C4_LAYER_IDS = [0, 1, 2, 3]
C128_LAYER_IDS = [4, 5, 6, 7]
NUM_ORIGINAL_LAYERS = 8

C4_HEAD_SIZE = 64
C128_HEAD_SIZE = 32
INDEXER_HEAD_SIZE = 40
SWA_BYTES_PER_TOKEN = 128

GPU_SRC = 1
CPU_DST = 2
GPU_BACK = 5
SWA_CPU_DST = 3
SWA_GPU_BACK = 6

# C++ transfer kernels operate on raw bytes; use uint8 for the D2H leg.
MAIN_DTYPE = torch.uint8


def _dsv4_layer_groups() -> List[LayerGroupSpec]:
    return [
        LayerGroupSpec(
            num_layers=len(C4_LAYER_IDS),
            num_kv_heads=1,
            head_size=C4_HEAD_SIZE,
            layer_indices=C4_LAYER_IDS,
            dtype=MAIN_DTYPE,
            compress_ratio=4,
        ),
        LayerGroupSpec(
            num_layers=len(C128_LAYER_IDS),
            num_kv_heads=1,
            head_size=C128_HEAD_SIZE,
            layer_indices=C128_LAYER_IDS,
            dtype=MAIN_DTYPE,
            compress_ratio=128,
        ),
        LayerGroupSpec(
            num_layers=len(C4_LAYER_IDS),
            num_kv_heads=1,
            head_size=INDEXER_HEAD_SIZE,
            layer_indices=C4_LAYER_IDS,
            dtype=torch.uint8,
            compress_ratio=4,
        ),
    ]


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare tensors byte-for-byte (fp8-safe)."""
    if a.shape != b.shape:
        return False
    if a.dtype == b.dtype:
        return bool(torch.equal(a.cpu(), b.cpu()))
    return bool(torch.equal(a.cpu().view(torch.uint8), b.cpu().view(torch.uint8)))


def _seed_gpu_layer(
    tensor: torch.Tensor,
    block_id: int,
    orig_layer: int,
    group_idx: int,
    local_layer_id: int,
) -> torch.Tensor:
    """Fill one GPU layer/block with a deterministic pattern; return golden copy."""
    tpb_g = tensor.shape[1]
    num_heads = tensor.shape[2]
    head_size = tensor.shape[3]
    expected = torch.zeros(tpb_g, num_heads, head_size, dtype=tensor.dtype, device="cpu")
    plane = tensor[block_id]
    for tok in range(tpb_g):
        for h in range(num_heads):
            for b in range(head_size):
                val = ((orig_layer * 97 + group_idx * 13 + local_layer_id * 7 + tok) ^ b) & 0xFF
                expected[tok, h, b] = val
                plane[tok, h, b] = val
    return expected


def _seed_swa_gpu_layer(
    tensor: torch.Tensor,
    block_id: int,
    orig_layer: int,
) -> torch.Tensor:
    tpb = tensor.shape[1]
    head_size = tensor.shape[3]
    expected = torch.zeros(tpb, 1, head_size, dtype=torch.uint8, device="cpu")
    plane = tensor[block_id]
    for tok in range(tpb):
        for b in range(head_size):
            val = ((orig_layer * 31 + tok + 0xD4) ^ b) & 0xFF
            expected[tok, 0, b] = val
            plane[tok, 0, b] = val
    return expected


def _group_gpu_ptrs(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64).pin_memory()


def _build_dsv4_fixture(layer_groups: List[LayerGroupSpec]) -> MultiGroupFixture:
    device = _device()
    cpu_layout = _make_multi_group_cpu_layout(
        layer_groups, NUM_ORIGINAL_LAYERS, NUM_CPU_BLOCKS, TOKENS_PER_BLOCK,
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

    swa_layout = _make_swa_layout(
        NUM_ORIGINAL_LAYERS, NUM_CPU_BLOCKS, TOKENS_PER_BLOCK,
    )
    swa_gpu_layout = _make_swa_layout(
        NUM_ORIGINAL_LAYERS, NUM_GPU_BLOCKS, TOKENS_PER_BLOCK,
    )
    swa_strides = _compute_swa_strides(swa_layout, swa_gpu_layout)
    swa_cpu = torch.zeros(swa_layout.kv_shape, dtype=torch.uint8, pin_memory=True)
    swa_gpu_tensors = [
        torch.zeros(
            NUM_GPU_BLOCKS,
            TOKENS_PER_BLOCK,
            1,
            SWA_BYTES_PER_TOKEN,
            dtype=torch.uint8,
            device=device,
        )
        for _ in range(NUM_ORIGINAL_LAYERS)
    ]

    group = LayerwiseTransferGroup(
        num_gpus=1,
        gpu_blocks_per_group=gpu_blocks_per_group,
        cpu_blocks=cpu_blocks,
        ssd_files=ssd_files,
        num_original_layers=NUM_ORIGINAL_LAYERS,
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
        has_swa=True,
        swa_gpu_blocks=[swa_gpu_tensors],
        swa_cpu_blocks=swa_cpu,
        swa_ssd_files=ssd_files,
        swa_gpu_kv_strides_tensor=swa_strides["swa_gpu_kv_strides_tensor"],
        swa_gpu_block_strides_tensor=swa_strides["swa_gpu_block_strides_tensor"],
        swa_gpu_layer_strides_tensor=swa_strides["swa_gpu_layer_strides_tensor"],
        swa_gpu_chunk_sizes_tensor=swa_strides["swa_gpu_chunk_sizes_tensor"],
        is_blockfirst=True,
        is_mla=True,
    )

    return MultiGroupFixture(
        group=group,
        layer_groups=layer_groups,
        gpu_tensors_per_group=gpu_tensors_per_group,
        cpu_blocks=cpu_blocks,
        strides=strides,
        num_original_layers=NUM_ORIGINAL_LAYERS,
        swa_gpu_tensors=swa_gpu_tensors,
        swa_cpu=swa_cpu,
        swa_strides=swa_strides,
    )


def _d2h_main_group(
    fx: MultiGroupFixture,
    group_idx: int,
    gpu_src: int,
    cpu_dst: int,
) -> None:
    """GPU->CPU for one multi-group member (mirrors GPUCPUTransferWorker D2H)."""
    g = fx.layer_groups[group_idx]
    dtype_size = g.dtype.itemsize
    gpu_layout = _make_gpu_layout(g, NUM_GPU_BLOCKS, TOKENS_PER_BLOCK)
    tensors = fx.gpu_tensors_per_group[group_idx]

    cpu_flat = fx.cpu_blocks.contiguous().view(-1)
    cpu_off = fx.strides["group_cpu_offset_bytes"][group_idx]  # type: ignore[index]
    cpu_for_group = cpu_flat[cpu_off:]

    transfer_kv_blocks(
        torch.tensor([gpu_src], dtype=torch.int64),
        _group_gpu_ptrs(tensors),
        gpu_layout.get_kv_stride() * dtype_size,
        gpu_layout.get_block_stride() * dtype_size,
        gpu_layout.get_layer_stride() * dtype_size,
        torch.tensor([cpu_dst], dtype=torch.int64),
        cpu_for_group,
        fx.strides["group_cpu_kv_strides"][group_idx],  # type: ignore[index]
        fx.strides["group_cpu_layer_strides"][group_idx],  # type: ignore[index]
        fx.strides["group_cpu_block_strides"][group_idx],  # type: ignore[index]
        fx.strides["group_chunk_sizes"][group_idx],  # type: ignore[index]
        0,
        g.num_layers,
        4,
        False,  # D2H
        True,
        True,
        0,
    )


def _d2h_swa(
    fx: MultiGroupFixture,
    gpu_src: int,
    cpu_dst: int,
) -> None:
    assert fx.swa_strides is not None and fx.swa_gpu_tensors is not None
    swa_layout = _make_swa_layout(
        NUM_ORIGINAL_LAYERS, NUM_GPU_BLOCKS, TOKENS_PER_BLOCK,
    )

    transfer_kv_blocks(
        torch.tensor([gpu_src], dtype=torch.int64),
        _group_gpu_ptrs(fx.swa_gpu_tensors),
        swa_layout.get_kv_stride(),
        swa_layout.get_block_stride(),
        swa_layout.get_layer_stride(),
        torch.tensor([cpu_dst], dtype=torch.int64),
        fx.swa_cpu.contiguous(),  # type: ignore[union-attr]
        fx.swa_strides["swa_cpu_kv_stride_in_bytes"],  # type: ignore[index]
        fx.swa_strides["swa_cpu_layer_stride_in_bytes"],  # type: ignore[index]
        fx.swa_strides["swa_cpu_block_stride_in_bytes"],  # type: ignore[index]
        fx.swa_strides["swa_cpu_chunk_size_in_bytes"],  # type: ignore[index]
        0,
        NUM_ORIGINAL_LAYERS,
        4,
        False,
        True,
        True,
        0,
    )


def _layerwise_h2d_main_and_swa(fx: MultiGroupFixture, cpu_src: int, gpu_dst: int,
                                swa_cpu_src: int, swa_gpu_dst: int) -> None:
    assert fx.swa_strides is not None
    empty = torch.empty(0, dtype=torch.int64)
    fx.group.layerwise_transfer_multi_group(
        empty,
        empty,
        num_blocks_per_file=0,
        round_robin=1,
        num_threads_per_device=4,
        gpu_block_id_tensor=torch.tensor([gpu_dst], dtype=torch.int64),
        cpu_block_id_tensor=torch.tensor([cpu_src], dtype=torch.int64),
        transfer_cta_num=4,
        use_ce_transfer=True,
        is_mla=True,
        counter_id=0,
        swa_h2d_src=torch.tensor([swa_cpu_src], dtype=torch.int64),
        swa_h2d_dst=torch.tensor([swa_gpu_dst], dtype=torch.int64),
        swa_disk2h_src=empty,
        swa_disk2h_dst=empty,
        swa_cpu_kv_stride_in_bytes=fx.swa_strides["swa_cpu_kv_stride_in_bytes"],
        swa_cpu_layer_stride_in_bytes=fx.swa_strides["swa_cpu_layer_stride_in_bytes"],
        swa_cpu_block_stride_in_bytes=fx.swa_strides["swa_cpu_block_stride_in_bytes"],
        swa_cpu_chunk_size_in_bytes=fx.swa_strides["swa_cpu_chunk_size_in_bytes"],
        swa_h2d_cpu_kv_stride_in_bytes=fx.swa_strides["swa_h2d_cpu_kv_stride_in_bytes"],
        swa_h2d_cpu_layer_stride_in_bytes=fx.swa_strides["swa_h2d_cpu_layer_stride_in_bytes"],
        swa_cpu_tp_stride_in_bytes=fx.swa_strides["swa_cpu_tp_stride_in_bytes"],
        swa_ssd_layer_stride_in_bytes=0,
        swa_ssd_kv_stride_in_bytes=0,
        swa_num_blocks_per_file=0,
    )
    torch.cuda.synchronize()


def _zero_all_gpu(fx: MultiGroupFixture) -> None:
    for group_tensors in fx.gpu_tensors_per_group:
        for t in group_tensors:
            t.zero_()
    assert fx.swa_gpu_tensors is not None
    for t in fx.swa_gpu_tensors:
        t.zero_()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestLayerwiseDsv4MultiGroupSwaRoundtrip:
    def test_dsv4_write_read_roundtrip_main_indexer_swa(self) -> None:
        """GPU seed -> D2H (3 groups + SWA) -> layerwise H2D -> byte-exact restore."""
        torch.cuda.set_device(DEVICE_ID)
        layer_groups = _dsv4_layer_groups()
        fx = _build_dsv4_fixture(layer_groups)
        member_map = fx.strides["layer_member_map"]

        # --- Phase 0: seed GPU at GPU_SRC ---
        expected_main: Dict[Tuple[int, int, int], torch.Tensor] = {}
        for orig in range(NUM_ORIGINAL_LAYERS):
            for gi, local_id in member_map.members_of(orig):  # type: ignore[union-attr]
                golden = _seed_gpu_layer(
                    fx.gpu_tensors_per_group[gi][local_id],
                    GPU_SRC,
                    orig,
                    gi,
                    local_id,
                )
                expected_main[(orig, gi, local_id)] = golden

        expected_swa: Dict[int, torch.Tensor] = {}
        assert fx.swa_gpu_tensors is not None
        for orig in range(NUM_ORIGINAL_LAYERS):
            expected_swa[orig] = _seed_swa_gpu_layer(
                fx.swa_gpu_tensors[orig], GPU_SRC, orig,
            )
        torch.cuda.synchronize()

        # --- Phase 1: PUT (D2H) — write GPU -> CPU for all groups + SWA ---
        for gi in range(len(layer_groups)):
            _d2h_main_group(fx, gi, GPU_SRC, CPU_DST)
        _d2h_swa(fx, GPU_SRC, SWA_CPU_DST)
        torch.cuda.synchronize()

        # --- Phase 2: zero GPU to ensure H2D really reads CPU ---
        _zero_all_gpu(fx)
        for gi, local_id in [(0, 0)]:
            assert fx.gpu_tensors_per_group[gi][local_id][GPU_SRC].sum().item() == 0
        assert fx.swa_gpu_tensors[0][GPU_SRC].sum().item() == 0

        # --- Phase 3: GET (layerwise H2D) — restore into different GPU blocks ---
        _layerwise_h2d_main_and_swa(
            fx, CPU_DST, GPU_BACK, SWA_CPU_DST, SWA_GPU_BACK,
        )

        # --- Phase 4: byte-exact verification ---
        failures: List[str] = []
        for (orig, gi, local_id), golden in expected_main.items():
            actual = fx.gpu_tensors_per_group[gi][local_id][GPU_BACK].cpu()
            if not _bytes_equal(actual, golden):
                failures.append(
                    f"main orig={orig} group={gi} local={local_id} "
                    f"dtype={golden.dtype}"
                )

        for orig, golden in expected_swa.items():
            actual = fx.swa_gpu_tensors[orig][SWA_GPU_BACK].cpu()
            if not _bytes_equal(actual, golden):
                failures.append(f"SWA orig={orig}")

        assert not failures, "Roundtrip byte mismatches:\n  " + "\n  ".join(failures)

    def test_dsv4_layer_members_match_production_shape(self) -> None:
        """Sanity: c4 layers carry c4+indexer; c128 layers carry c128 only."""
        layer_groups = _dsv4_layer_groups()
        member_map = build_layer_member_map(layer_groups, NUM_ORIGINAL_LAYERS)

        for orig in C4_LAYER_IDS:
            members = member_map.members_of(orig)
            groups = {gi for gi, _ in members}
            assert groups == {0, 2}, f"layer {orig}: expected c4+indexer, got {members}"

        for orig in C128_LAYER_IDS:
            members = member_map.members_of(orig)
            assert len(members) == 1 and members[0][0] == 1, (
                f"layer {orig}: expected only c128 member, got {members}"
            )
