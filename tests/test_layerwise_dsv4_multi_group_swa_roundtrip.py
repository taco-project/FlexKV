"""DSv4-style multi-group + SWA roundtrip: what goes down comes back up.

Simulates the production data path end to end, through the one region batch:

  1. PUT phase (D2H): seed GPU -> write main KV (c4 / c128 / indexer groups)
     plus SWA to CPU, as one batched submit over every region
  2. Zero GPU pools (prove the next step reads from CPU, not stale GPU)
  3. GET phase: per-layer H2D via ``submit_layerwise`` for main KV + SWA
  4. Byte-exact compare restored GPU blocks against the original seed

Geometry mirrors DeepSeek V4:
  - ``c4`` group: compress_ratio=4 on CSA layers
  - ``c128`` group: compress_ratio=128 on HCA layers
  - ``c4_indexer`` group: indexer K on CSA layers (uint8)
  - SWA sidecar: all original layers, its own region and its own slot ids

The D2H leg deliberately goes out as a single ``submit`` naming all four
regions. That is the "submit SWA / full / states / indexer in one batch" the
transfer refactor is for: the batch is just the request list, so a model with
four regions costs one fan-out rather than four.

Run:
    pytest tests/test_layerwise_dsv4_multi_group_swa_roundtrip.py -v
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest
import torch

from flexkv.common.config import LayerGroupSpec, build_layer_member_map
from flexkv.transfer.region_batch import make_requests

from test_layerwise_multi_group_swa import (
    MultiGroupFixture,
    build_fixture,
    h2d_requests,
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


def _build_dsv4_fixture() -> MultiGroupFixture:
    return build_fixture(
        _dsv4_layer_groups(),
        NUM_ORIGINAL_LAYERS,
        has_swa=True,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        num_gpu_blocks=NUM_GPU_BLOCKS,
        swa_bytes_per_token=SWA_BYTES_PER_TOKEN,
    )


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


def _d2h_everything(fx: MultiGroupFixture) -> None:
    """One submit, every region: the main groups and SWA go out together.

    Two request lists rather than one because the two pools number their blocks
    independently -- SWA block ``SWA_CPU_DST`` is not main-KV block
    ``CPU_DST``. Same submit, so still one fan-out.
    """
    num_main_regions = len(fx.layer_groups)
    requests = make_requests(
        num_main_regions,
        torch.tensor([GPU_SRC], dtype=torch.int64),
        torch.tensor([CPU_DST], dtype=torch.int64),
        False,  # D2H
        transfer_num_cta=4,
        use_ce_transfer=True,
        region_indices=list(range(num_main_regions)),
    )
    requests += make_requests(
        fx.group.num_regions,
        torch.tensor([GPU_SRC], dtype=torch.int64),
        torch.tensor([SWA_CPU_DST], dtype=torch.int64),
        False,
        transfer_num_cta=4,
        use_ce_transfer=True,
        region_indices=[num_main_regions],
    )
    fx.group.submit(requests, True)


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
        """GPU seed -> D2H (3 groups + SWA) -> per-layer H2D -> byte-exact restore."""
        torch.cuda.set_device(DEVICE_ID)
        fx = _build_dsv4_fixture()
        member_map = fx.strides["layer_member_map"]

        # --- Phase 0: seed GPU at GPU_SRC ---
        expected_main: Dict[Tuple[int, int, int], torch.Tensor] = {}
        for orig in range(NUM_ORIGINAL_LAYERS):
            for gi, local_id in member_map.members_of(orig):  # type: ignore[union-attr]
                expected_main[(orig, gi, local_id)] = _seed_gpu_layer(
                    fx.gpu_tensors_per_group[gi][local_id],
                    GPU_SRC,
                    orig,
                    gi,
                    local_id,
                )

        expected_swa: Dict[int, torch.Tensor] = {}
        assert fx.swa_gpu_tensors is not None
        for orig in range(NUM_ORIGINAL_LAYERS):
            expected_swa[orig] = _seed_swa_gpu_layer(
                fx.swa_gpu_tensors[orig], GPU_SRC, orig,
            )
        torch.cuda.synchronize()

        # --- Phase 1: PUT (D2H) — every region in one batch ---
        _d2h_everything(fx)

        # --- Phase 2: zero GPU to ensure H2D really reads CPU ---
        _zero_all_gpu(fx)
        assert fx.gpu_tensors_per_group[0][0][GPU_SRC].sum().item() == 0
        assert fx.swa_gpu_tensors[0][GPU_SRC].sum().item() == 0

        # --- Phase 3: GET (per-layer H2D) — restore into different GPU blocks ---
        requests, empty_layers = h2d_requests(
            fx, with_swa=True,
            cpu_block=CPU_DST, gpu_block=GPU_BACK,
            swa_cpu_block=SWA_CPU_DST, swa_gpu_block=SWA_GPU_BACK,
        )
        assert empty_layers == [], (
            "every DSv4 layer has a member; nothing may be posted early")
        fx.group.submit_layerwise(requests, empty_layers, 0)
        ok, err = fx.group.wait_layer_completion(120.0)
        assert ok, f"layerwise H2D did not complete: {err}"
        torch.cuda.synchronize()

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
