"""Mock roundtrip test for SGLang FP4 pack/unpack logic.

Validates the data layout math used by transfer_kv_blocks_fp4_kernel
WITHOUT requiring CUDA or real model weights. Uses pure Python/numpy
to simulate the GPU kernel's pack/unpack behavior.

SGLang FP4 GPU layout (4 separate buffers per layer):
  k_buffer[layer]:       (num_blocks * tokens_per_block, num_heads, head_dim // 2)  uint8
  k_scale_buffer[layer]: (num_blocks * tokens_per_block, num_heads * head_dim // 16) uint8
  v_buffer, v_scale_buffer: same shapes

FlexKV CPU packed layout (per head):
  cpu[layer, kv, block, token, head, 0:packed_head_size]
  where packed_head_size = head_dim // 2 + head_dim // 16
  each head slot = [data: head_dim//2 bytes][scale: head_dim//16 bytes]
"""
from __future__ import annotations
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Model parameters (GLM-5.2 style)
# ---------------------------------------------------------------------------
HEAD_DIM = 128        # logical head dimension
NUM_HEADS = 4         # num_kv_heads (MQA/GQA heads, simplified)
TOKENS_PER_BLOCK = 8  # sglang page_size
NUM_BLOCKS = 3        # test with 3 blocks
NUM_LAYERS = 2        # test with 2 layers

DATA_PER_HEAD = HEAD_DIM // 2    # 64 bytes of fp4 data
SCALE_PER_HEAD = HEAD_DIM // 16  # 8 bytes of fp8 scale
PACKED_HEAD_SIZE = DATA_PER_HEAD + SCALE_PER_HEAD  # 72 bytes


def _make_gpu_buffers():
    """Create mock SGLang FP4 GPU buffers (4 buffers × num_layers)."""
    total_tokens = NUM_BLOCKS * TOKENS_PER_BLOCK
    # k/v data: (total_tokens, num_heads, data_per_head) -- 3D
    # k/v scale: (total_tokens, num_heads * scale_per_head) -- 2D
    data_bufs = {}  # (kv_idx, layer_idx) -> np array
    scale_bufs = {}
    for kv_idx in range(2):
        for layer_idx in range(NUM_LAYERS):
            data_bufs[(kv_idx, layer_idx)] = np.random.randint(
                0, 256, (total_tokens, NUM_HEADS, DATA_PER_HEAD), dtype=np.uint8
            )
            scale_bufs[(kv_idx, layer_idx)] = np.random.randint(
                0, 256, (total_tokens, NUM_HEADS * SCALE_PER_HEAD), dtype=np.uint8
            )
    return data_bufs, scale_bufs


def _make_cpu_buffer():
    """Create mock FlexKV CPU packed buffer."""
    # cpu layout: (num_layers, 2(kv), num_blocks, tokens_per_block, num_heads, packed_head_size)
    return np.zeros(
        (NUM_LAYERS, 2, NUM_BLOCKS, TOKENS_PER_BLOCK, NUM_HEADS, PACKED_HEAD_SIZE),
        dtype=np.uint8,
    )


def _simulate_d2h_pack(
    data_bufs, scale_bufs, cpu_buf,
    gpu_block_ids, cpu_block_ids,
    start_layer_id, num_layers,
):
    """Simulate the FP4 kernel's D2H (GPU→CPU) pack operation.

    Mirrors transfer_kv_blocks_fp4_kernel with is_host_to_device=False.
    """
    for layer_off in range(num_layers):
        layer_idx = start_layer_id + layer_off
        for block_pos in range(len(gpu_block_ids)):
            gpu_block_idx = gpu_block_ids[block_pos]
            cpu_block_idx = cpu_block_ids[block_pos]
            for kv_idx in range(2):
                for tok in range(TOKENS_PER_BLOCK):
                    global_token = gpu_block_idx * TOKENS_PER_BLOCK + tok
                    for head in range(NUM_HEADS):
                        # GPU data: data_bufs[(kv, layer)][token, head, :data_per_head]
                        gpu_data = data_bufs[(kv_idx, layer_idx)][global_token, head, :]

                        # GPU scale: scale_bufs[(kv, layer)][token, head*scale:(head+1)*scale]
                        scale_start = head * SCALE_PER_HEAD
                        scale_end = scale_start + SCALE_PER_HEAD
                        gpu_scale = scale_bufs[(kv_idx, layer_idx)][global_token, scale_start:scale_end]

                        # CPU packed: [data | scale] per head
                        cpu_buf[layer_idx, kv_idx, cpu_block_idx, tok, head, :DATA_PER_HEAD] = gpu_data
                        cpu_buf[layer_idx, kv_idx, cpu_block_idx, tok, head, DATA_PER_HEAD:] = gpu_scale


def _simulate_h2d_unpack(
    cpu_buf, data_bufs_out, scale_bufs_out,
    gpu_block_ids, cpu_block_ids,
    start_layer_id, num_layers,
):
    """Simulate the FP4 kernel's H2D (CPU→GPU) unpack operation.

    Mirrors transfer_kv_blocks_fp4_kernel with is_host_to_device=True.
    """
    for layer_off in range(num_layers):
        layer_idx = start_layer_id + layer_off
        for block_pos in range(len(gpu_block_ids)):
            gpu_block_idx = gpu_block_ids[block_pos]
            cpu_block_idx = cpu_block_ids[block_pos]
            for kv_idx in range(2):
                for tok in range(TOKENS_PER_BLOCK):
                    global_token = gpu_block_idx * TOKENS_PER_BLOCK + tok
                    for head in range(NUM_HEADS):
                        # Read from CPU packed
                        packed = cpu_buf[layer_idx, kv_idx, cpu_block_idx, tok, head, :]

                        # Write data portion to GPU data buffer
                        data_bufs_out[(kv_idx, layer_idx)][global_token, head, :] = packed[:DATA_PER_HEAD]

                        # Write scale portion to GPU scale buffer
                        scale_start = head * SCALE_PER_HEAD
                        scale_end = scale_start + SCALE_PER_HEAD
                        scale_bufs_out[(kv_idx, layer_idx)][global_token, scale_start:scale_end] = packed[DATA_PER_HEAD:]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_d2h_h2d_roundtrip_exact():
    """D2H pack then H2D unpack should perfectly reconstruct original GPU data."""
    np.random.seed(42)

    # Create original GPU buffers with random data
    orig_data, orig_scale = _make_gpu_buffers()
    cpu_buf = _make_cpu_buffer()

    gpu_block_ids = list(range(NUM_BLOCKS))
    cpu_block_ids = list(range(NUM_BLOCKS))

    # D2H: GPU → CPU (pack)
    _simulate_d2h_pack(orig_data, orig_scale, cpu_buf, gpu_block_ids, cpu_block_ids, 0, NUM_LAYERS)

    # Create fresh GPU buffers for reconstruction
    recon_data = {}
    recon_scale = {}
    total_tokens = NUM_BLOCKS * TOKENS_PER_BLOCK
    for kv_idx in range(2):
        for layer_idx in range(NUM_LAYERS):
            recon_data[(kv_idx, layer_idx)] = np.zeros(
                (total_tokens, NUM_HEADS, DATA_PER_HEAD), dtype=np.uint8
            )
            recon_scale[(kv_idx, layer_idx)] = np.zeros(
                (total_tokens, NUM_HEADS * SCALE_PER_HEAD), dtype=np.uint8
            )

    # H2D: CPU → GPU (unpack)
    _simulate_h2d_unpack(cpu_buf, recon_data, recon_scale, gpu_block_ids, cpu_block_ids, 0, NUM_LAYERS)

    # Verify byte-exact reconstruction
    for kv_idx in range(2):
        for layer_idx in range(NUM_LAYERS):
            np.testing.assert_array_equal(
                orig_data[(kv_idx, layer_idx)],
                recon_data[(kv_idx, layer_idx)],
                err_msg=f"Data mismatch at kv={kv_idx}, layer={layer_idx}",
            )
            np.testing.assert_array_equal(
                orig_scale[(kv_idx, layer_idx)],
                recon_scale[(kv_idx, layer_idx)],
                err_msg=f"Scale mismatch at kv={kv_idx}, layer={layer_idx}",
            )


def test_d2h_h2d_with_remapped_block_ids():
    """Block ID remapping: GPU blocks [2,0,1] → CPU blocks [5,3,4]."""
    np.random.seed(123)

    # Need more blocks for remapping
    total_tokens_gpu = 6 * TOKENS_PER_BLOCK  # 6 GPU blocks
    total_tokens_cpu = 8 * TOKENS_PER_BLOCK  # 8 CPU blocks

    orig_data = {}
    orig_scale = {}
    for kv_idx in range(2):
        for layer_idx in range(NUM_LAYERS):
            orig_data[(kv_idx, layer_idx)] = np.random.randint(
                0, 256, (total_tokens_gpu, NUM_HEADS, DATA_PER_HEAD), dtype=np.uint8
            )
            orig_scale[(kv_idx, layer_idx)] = np.random.randint(
                0, 256, (total_tokens_gpu, NUM_HEADS * SCALE_PER_HEAD), dtype=np.uint8
            )

    cpu_buf = np.zeros(
        (NUM_LAYERS, 2, 8, TOKENS_PER_BLOCK, NUM_HEADS, PACKED_HEAD_SIZE),
        dtype=np.uint8,
    )

    gpu_block_ids = [2, 0, 1]
    cpu_block_ids = [5, 3, 4]

    # D2H pack
    _simulate_d2h_pack(orig_data, orig_scale, cpu_buf, gpu_block_ids, cpu_block_ids, 0, NUM_LAYERS)

    # H2D unpack back to different GPU slots
    recon_data = {}
    recon_scale = {}
    for kv_idx in range(2):
        for layer_idx in range(NUM_LAYERS):
            recon_data[(kv_idx, layer_idx)] = np.zeros(
                (total_tokens_gpu, NUM_HEADS, DATA_PER_HEAD), dtype=np.uint8
            )
            recon_scale[(kv_idx, layer_idx)] = np.zeros(
                (total_tokens_gpu, NUM_HEADS * SCALE_PER_HEAD), dtype=np.uint8
            )

    _simulate_h2d_unpack(cpu_buf, recon_data, recon_scale, gpu_block_ids, cpu_block_ids, 0, NUM_LAYERS)

    # Verify only the transferred blocks match
    for kv_idx in range(2):
        for layer_idx in range(NUM_LAYERS):
            for i, gpu_bid in enumerate(gpu_block_ids):
                tok_start = gpu_bid * TOKENS_PER_BLOCK
                tok_end = tok_start + TOKENS_PER_BLOCK
                np.testing.assert_array_equal(
                    orig_data[(kv_idx, layer_idx)][tok_start:tok_end],
                    recon_data[(kv_idx, layer_idx)][tok_start:tok_end],
                )
                np.testing.assert_array_equal(
                    orig_scale[(kv_idx, layer_idx)][tok_start:tok_end],
                    recon_scale[(kv_idx, layer_idx)][tok_start:tok_end],
                )


def test_packed_head_size_formula():
    """Verify the packed head size formula matches nvfp4_kv_cache_full_dim."""
    for head_dim in [64, 128, 192, 256]:
        data = head_dim // 2
        scale = head_dim // 16
        packed = data + scale
        # Same formula as nvfp4_kv_cache_full_dim
        assert packed == head_dim // 2 + head_dim // 16
        # Verify it's smaller than original
        assert packed < head_dim


def test_layer_subset_transfer():
    """Transfer only a subset of layers (layer_granularity > 1 scenario)."""
    np.random.seed(99)
    orig_data, orig_scale = _make_gpu_buffers()
    cpu_buf = _make_cpu_buffer()

    gpu_block_ids = [0, 1]
    cpu_block_ids = [0, 1]

    # Only transfer layer 1
    _simulate_d2h_pack(orig_data, orig_scale, cpu_buf, gpu_block_ids, cpu_block_ids,
                       start_layer_id=1, num_layers=1)

    # Layer 0 in CPU should be all zeros
    assert np.all(cpu_buf[0] == 0)
    # Layer 1 should have data
    assert not np.all(cpu_buf[1] == 0)

    # Roundtrip layer 1 only
    recon_data = {}
    recon_scale = {}
    total_tokens = NUM_BLOCKS * TOKENS_PER_BLOCK
    for kv_idx in range(2):
        for layer_idx in range(NUM_LAYERS):
            recon_data[(kv_idx, layer_idx)] = np.zeros(
                (total_tokens, NUM_HEADS, DATA_PER_HEAD), dtype=np.uint8
            )
            recon_scale[(kv_idx, layer_idx)] = np.zeros(
                (total_tokens, NUM_HEADS * SCALE_PER_HEAD), dtype=np.uint8
            )

    _simulate_h2d_unpack(cpu_buf, recon_data, recon_scale, gpu_block_ids, cpu_block_ids,
                         start_layer_id=1, num_layers=1)

    # Layer 1, transferred blocks only, should match
    for kv_idx in range(2):
        for gpu_bid in gpu_block_ids:
            tok_s = gpu_bid * TOKENS_PER_BLOCK
            tok_e = tok_s + TOKENS_PER_BLOCK
            np.testing.assert_array_equal(
                orig_data[(kv_idx, 1)][tok_s:tok_e],
                recon_data[(kv_idx, 1)][tok_s:tok_e],
            )
            np.testing.assert_array_equal(
                orig_scale[(kv_idx, 1)][tok_s:tok_e],
                recon_scale[(kv_idx, 1)][tok_s:tok_e],
            )


def test_cpu_layout_matches_kernel_pointer_math():
    """Verify the CPU pointer arithmetic in the kernel matches our simulation.

    The kernel computes:
      cpu_packed = cpu_ptr + layer_idx * cpu_layer_stride + kv_idx * cpu_kv_stride
                   + cpu_block_idx * cpu_block_stride
                   + (tok * num_heads + head) * packed_head_size

    With LAYERFIRST layout and strides in int64_t units:
      cpu_layer_stride = 2 * num_blocks * tokens_per_block * num_heads * packed_head_size / 8
      cpu_kv_stride = num_blocks * tokens_per_block * num_heads * packed_head_size / 8
      cpu_block_stride = tokens_per_block * num_heads * packed_head_size / 8
    """
    elem_per_block = TOKENS_PER_BLOCK * NUM_HEADS * PACKED_HEAD_SIZE

    # Strides in bytes
    cpu_block_stride_bytes = elem_per_block
    cpu_kv_stride_bytes = NUM_BLOCKS * elem_per_block
    cpu_layer_stride_bytes = 2 * NUM_BLOCKS * elem_per_block

    # Verify these strides would give correct offsets in a flat buffer
    total_size = NUM_LAYERS * 2 * NUM_BLOCKS * TOKENS_PER_BLOCK * NUM_HEADS * PACKED_HEAD_SIZE
    flat_buf = np.arange(total_size, dtype=np.uint8)
    shaped_buf = flat_buf.reshape(
        (NUM_LAYERS, 2, NUM_BLOCKS, TOKENS_PER_BLOCK, NUM_HEADS, PACKED_HEAD_SIZE)
    )

    # Check that stride-based addressing matches shaped indexing
    for layer_idx in range(NUM_LAYERS):
        for kv_idx in range(2):
            for block_idx in range(NUM_BLOCKS):
                for tok in range(TOKENS_PER_BLOCK):
                    for head in range(NUM_HEADS):
                        # Stride-based offset (what the kernel does)
                        byte_offset = (
                            layer_idx * cpu_layer_stride_bytes
                            + kv_idx * cpu_kv_stride_bytes
                            + block_idx * cpu_block_stride_bytes
                            + (tok * NUM_HEADS + head) * PACKED_HEAD_SIZE
                        )
                        kernel_view = flat_buf[byte_offset:byte_offset + PACKED_HEAD_SIZE]

                        # Direct indexing (what our simulation does)
                        direct_view = shaped_buf[layer_idx, kv_idx, block_idx, tok, head, :]

                        np.testing.assert_array_equal(kernel_view, direct_view)


# ---------------------------------------------------------------------------
# MLA FP4 tests (GLM5.2 / DeepSeek-V3 style)
# MLA: kv_buffer (not k+v separate), kv_scale_buffer (not k_scale+v_scale)
# Only num_layers data+scale buffers (not 2×num_layers)
# ---------------------------------------------------------------------------

MLA_KV_DIM = 576       # kv_lora_rank(512) + qk_rope_head_dim(64)
MLA_NUM_HEADS = 1       # MLA always 1 head
MLA_DATA_PER_HEAD = MLA_KV_DIM // 2    # 288
MLA_SCALE_PER_HEAD = MLA_KV_DIM // 16  # 36
MLA_PACKED = MLA_DATA_PER_HEAD + MLA_SCALE_PER_HEAD  # 324


def _make_mla_gpu_buffers():
    """Create mock MLATokenToKVPoolFP4 buffers (num_layers only, no K/V split)."""
    total_tokens = NUM_BLOCKS * TOKENS_PER_BLOCK
    data_bufs = {}
    scale_bufs = {}
    for layer_idx in range(NUM_LAYERS):
        # kv_buffer: (total_tokens, 1, kv_dim//2)
        data_bufs[(0, layer_idx)] = np.random.randint(
            0, 256, (total_tokens, MLA_NUM_HEADS, MLA_DATA_PER_HEAD), dtype=np.uint8
        )
        # kv_scale_buffer: (total_tokens, kv_dim//16)
        scale_bufs[(0, layer_idx)] = np.random.randint(
            0, 256, (total_tokens, MLA_NUM_HEADS * MLA_SCALE_PER_HEAD), dtype=np.uint8
        )
    return data_bufs, scale_bufs


def _simulate_mla_d2h_pack(data_bufs, scale_bufs, cpu_buf,
                            gpu_block_ids, cpu_block_ids,
                            start_layer_id, num_layers):
    """MLA FP4 D2H: no kv_idx loop (only kv_idx=0)."""
    for layer_off in range(num_layers):
        layer_idx = start_layer_id + layer_off
        for block_pos in range(len(gpu_block_ids)):
            gpu_block_idx = gpu_block_ids[block_pos]
            cpu_block_idx = cpu_block_ids[block_pos]
            kv_idx = 0  # MLA: single KV
            for tok in range(TOKENS_PER_BLOCK):
                global_token = gpu_block_idx * TOKENS_PER_BLOCK + tok
                for head in range(MLA_NUM_HEADS):
                    gpu_data = data_bufs[(0, layer_idx)][global_token, head, :]
                    scale_start = head * MLA_SCALE_PER_HEAD
                    scale_end = scale_start + MLA_SCALE_PER_HEAD
                    gpu_scale = scale_bufs[(0, layer_idx)][global_token, scale_start:scale_end]

                    cpu_buf[layer_idx, kv_idx, cpu_block_idx, tok, head, :MLA_DATA_PER_HEAD] = gpu_data
                    cpu_buf[layer_idx, kv_idx, cpu_block_idx, tok, head, MLA_DATA_PER_HEAD:] = gpu_scale


def _simulate_mla_h2d_unpack(cpu_buf, data_bufs_out, scale_bufs_out,
                              gpu_block_ids, cpu_block_ids,
                              start_layer_id, num_layers):
    """MLA FP4 H2D: no kv_idx loop."""
    for layer_off in range(num_layers):
        layer_idx = start_layer_id + layer_off
        for block_pos in range(len(gpu_block_ids)):
            gpu_block_idx = gpu_block_ids[block_pos]
            cpu_block_idx = cpu_block_ids[block_pos]
            kv_idx = 0
            for tok in range(TOKENS_PER_BLOCK):
                global_token = gpu_block_idx * TOKENS_PER_BLOCK + tok
                for head in range(MLA_NUM_HEADS):
                    packed = cpu_buf[layer_idx, kv_idx, cpu_block_idx, tok, head, :]
                    data_bufs_out[(0, layer_idx)][global_token, head, :] = packed[:MLA_DATA_PER_HEAD]
                    scale_start = head * MLA_SCALE_PER_HEAD
                    scale_end = scale_start + MLA_SCALE_PER_HEAD
                    scale_bufs_out[(0, layer_idx)][global_token, scale_start:scale_end] = packed[MLA_DATA_PER_HEAD:]


def test_mla_fp4_roundtrip():
    """MLA FP4 D2H pack then H2D unpack should perfectly reconstruct."""
    np.random.seed(7777)
    orig_data, orig_scale = _make_mla_gpu_buffers()

    # MLA CPU buffer: kv_idx dimension has size 1 (but we use 2 for consistency,
    # only index 0 is used)
    cpu_buf = np.zeros(
        (NUM_LAYERS, 2, NUM_BLOCKS, TOKENS_PER_BLOCK, MLA_NUM_HEADS, MLA_PACKED),
        dtype=np.uint8,
    )

    gpu_block_ids = list(range(NUM_BLOCKS))
    cpu_block_ids = list(range(NUM_BLOCKS))

    _simulate_mla_d2h_pack(orig_data, orig_scale, cpu_buf, gpu_block_ids, cpu_block_ids, 0, NUM_LAYERS)

    recon_data = {}
    recon_scale = {}
    total_tokens = NUM_BLOCKS * TOKENS_PER_BLOCK
    for layer_idx in range(NUM_LAYERS):
        recon_data[(0, layer_idx)] = np.zeros(
            (total_tokens, MLA_NUM_HEADS, MLA_DATA_PER_HEAD), dtype=np.uint8
        )
        recon_scale[(0, layer_idx)] = np.zeros(
            (total_tokens, MLA_NUM_HEADS * MLA_SCALE_PER_HEAD), dtype=np.uint8
        )

    _simulate_mla_h2d_unpack(cpu_buf, recon_data, recon_scale, gpu_block_ids, cpu_block_ids, 0, NUM_LAYERS)

    for layer_idx in range(NUM_LAYERS):
        np.testing.assert_array_equal(
            orig_data[(0, layer_idx)],
            recon_data[(0, layer_idx)],
            err_msg=f"MLA data mismatch at layer={layer_idx}",
        )
        np.testing.assert_array_equal(
            orig_scale[(0, layer_idx)],
            recon_scale[(0, layer_idx)],
            err_msg=f"MLA scale mismatch at layer={layer_idx}",
        )


def test_mla_packed_head_size():
    """GLM5.2 MLA FP4: kv_dim=576 → packed=324."""
    assert MLA_KV_DIM // 2 + MLA_KV_DIM // 16 == 324


if __name__ == "__main__":
    test_d2h_h2d_roundtrip_exact()
    print("PASS: D2H/H2D roundtrip exact (MHA)")

    test_d2h_h2d_with_remapped_block_ids()
    print("PASS: D2H/H2D with remapped block IDs (MHA)")

    test_packed_head_size_formula()
    print("PASS: packed head size formula")

    test_layer_subset_transfer()
    print("PASS: layer subset transfer")

    test_cpu_layout_matches_kernel_pointer_math()
    print("PASS: CPU layout matches kernel pointer math")

    test_mla_fp4_roundtrip()
    print("PASS: MLA FP4 roundtrip")

    test_mla_packed_head_size()
    print("PASS: MLA packed head size")

    print("\nAll mock roundtrip tests passed!")
