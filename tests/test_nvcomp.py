"""
Test nvcomp ANS compression integrated with FlexKV GPU-CPU transfer.

Aligned with real FlexKV configuration:
  - GPU: LAYERFIRST (num_layers, 2, num_blocks, tpb, nh, hs)
  - CPU: configurable via FLEXKV_CPU_LAYOUT env var
    - BLOCKFIRST (default): [num_blocks, num_layers, 2, tpb, nh, hs]
    - LAYERFIRST:           [num_layers, 2, num_blocks, tpb, nh, hs]
  - Baseline: kernel mode (transfer_num_cta=4, use_ce_transfer=False)

Usage:
  FLEXKV_ENABLE_NVCOMP=1 FLEXKV_NVCOMP_LOG_LEVEL=1 python test_nvcomp.py
  FLEXKV_NVCOMP_BATCH_SIZE=4096 ...
  FLEXKV_ENABLE_NVCOMP=1 FLEXKV_NVCOMP_LOG_LEVEL=1 nsys profile -o ../.misc/profile/flexkv_nvcomp_double_buffer_ldg --force-overwrite true --trace=cuda,nvtx python test_nvcomp.py
  
  FLEXKV_NVCOMP_GREEN_SMS=112 FLEXKV_NVCOMP_BATCH_SIZE=4096 FLEXKV_ENABLE_NVCOMP=1 FLEXKV_NVCOMP_LOG_LEVEL=1 python test_nvcomp.py
  FLEXKV_NVCOMP_GREEN_SMS=112 FLEXKV_ENABLE_NVCOMP=1 FLEXKV_NVCOMP_LOG_LEVEL=1 nsys profile -o ../.misc/profile/flexkv_nvcomp_double_buffer_ldg_gc=112 --force-overwrite true --trace=cuda,nvtx python test_nvcomp.py


  FLEXKV_ENABLE_NVCOMP=1 FLEXKV_NVCOMP_LOG_LEVEL=1 ncu --kernel-name "regex:ans_d2h_scatter" --launch-skip 30 --launch-count 3 --set full --force-overwrite -o ../.misc/profile/ans_d2h_scatter_kernel python test_nvcomp.py
  ncu --kernel-name "regex:transfer_kv_blocks" \
    --launch-skip 3 --launch-count 3 \
    --set full --force-overwrite \
    -o ../.misc/profile/transfer_kv_blocks_kernel \
    python test_nvcomp.py

"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flexkv.c_ext import transfer_kv_blocks

try:
    from flexkv.c_ext import ANSTransferContext, ans_compress_and_d2h, ans_h2d_and_decompress
    NVCOMP_AVAILABLE = True
except ImportError:
    NVCOMP_AVAILABLE = False
    print("WARNING: nvcomp not available. Build with FLEXKV_ENABLE_NVCOMP=1")

BATCH_SIZE = int(os.environ.get("FLEXKV_NVCOMP_BATCH_SIZE", "0"))
TRANSFER_NUM_CTA = int(os.environ.get("FLEXKV_TRANSFER_NUM_CTA", "4"))
CPU_LAYOUT = os.environ.get("FLEXKV_CPU_LAYOUT", "BLOCKFIRST").upper()
assert CPU_LAYOUT in ("BLOCKFIRST", "LAYERFIRST"), \
    f"FLEXKV_CPU_LAYOUT must be BLOCKFIRST or LAYERFIRST, got {CPU_LAYOUT}"


def make_kv_cache(num_layers, num_blocks, tokens_per_block, num_heads, head_size,
                  dtype=torch.bfloat16, device="cuda:0"):
    """Create GPU (LAYERFIRST) and CPU KV cache tensors.

    GPU: list of per-layer tensors, each [2, num_blocks, tpb, nh, hs]
         Allocated as a single contiguous tensor (like vllm's KV cache pool)
         to avoid HBM partition imbalance from scattered allocations.
    CPU shape depends on CPU_LAYOUT:
      BLOCKFIRST: [num_blocks, num_layers, 2, tpb, nh, hs]
      LAYERFIRST: [num_layers, 2, num_blocks, tpb, nh, hs]
    """
    shape_per_layer = (2, num_blocks, tokens_per_block, num_heads, head_size)
    gpu_cache = torch.randn(
        (num_layers,) + shape_per_layer, dtype=dtype, device=device)
    gpu_blocks = [gpu_cache[i] for i in range(num_layers)]
    if CPU_LAYOUT == "BLOCKFIRST":
        cpu_shape = (num_blocks, num_layers, 2, tokens_per_block, num_heads, head_size)
    else:
        cpu_shape = (num_layers, 2, num_blocks, tokens_per_block, num_heads, head_size)
    cpu_blocks = torch.zeros(cpu_shape, dtype=dtype, device="cpu").pin_memory()
    return gpu_blocks, cpu_blocks


def get_gpu_tensor_ptrs(gpu_blocks):
    ptrs = torch.tensor(
        [block.data_ptr() for block in gpu_blocks],
        dtype=torch.int64,
    ).pin_memory()
    return ptrs


def compute_strides(num_layers, num_blocks, tokens_per_block, num_heads, head_size, dtype):
    """Compute byte strides for GPU (LAYERFIRST) and CPU (configurable)."""
    elem = dtype.itemsize if hasattr(dtype, 'itemsize') else torch.tensor([], dtype=dtype).element_size()
    chunk = tokens_per_block * num_heads * head_size  # elements per chunk
    if CPU_LAYOUT == "LAYERFIRST":
        print(f"[DEBUG] CPU layout is LAYERFIRST, chunk size is {chunk}")
        # TODO
    chunk_size = chunk * elem

    # GPU LAYERFIRST: per-layer tensor [2, num_blocks, tpb, nh, hs]
    gpu_kv_stride = num_blocks * chunk * elem
    gpu_block_stride = chunk_size
    gpu_layer_stride = 2 * num_blocks * chunk * elem

    if CPU_LAYOUT == "BLOCKFIRST":
        # [num_blocks, num_layers, 2, tpb, nh, hs]
        cpu_kv_stride = chunk_size
        cpu_layer_stride = 2 * chunk_size
        cpu_block_stride = num_layers * 2 * chunk_size
    else:
        # LAYERFIRST: [num_layers, 2, num_blocks, tpb, nh, hs]
        cpu_kv_stride = num_blocks * chunk_size
        cpu_layer_stride = 2 * num_blocks * chunk_size
        cpu_block_stride = chunk_size

    return (chunk_size,
            gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride)


def test_roundtrip(num_layers=4, num_blocks=2, tokens_per_block=16,
                   num_heads=8, head_size=128, dtype=torch.bfloat16):
    """Test: GPU -> compress -> D2H -> H2D -> decompress -> GPU, then compare."""
    if not NVCOMP_AVAILABLE:
        print("SKIP: nvcomp not available")
        return True

    device = "cuda:0"
    transfer_stream = torch.cuda.Stream()
    print(f"\n{'='*60}")
    print(f"Roundtrip test: layers={num_layers}, blocks={num_blocks}, "
          f"tpb={tokens_per_block}, heads={num_heads}, head_size={head_size}, dtype={dtype}")
    print(f"{'='*60}")

    gpu_blocks, cpu_blocks = make_kv_cache(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size, dtype, device)
    gpu_ptrs = get_gpu_tensor_ptrs(gpu_blocks)
    (chunk_size,
     gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
     cpu_kv_stride, cpu_layer_stride, cpu_block_stride) = compute_strides(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size, dtype)

    gpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    cpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()

    kv_dim = 2
    num_chunks = num_layers * kv_dim * num_blocks
    log_level = int(os.environ.get("FLEXKV_NVCOMP_LOG_LEVEL", "1"))

    original_data = [block.clone() for block in gpu_blocks]

    ctx = ANSTransferContext(BATCH_SIZE, chunk_size, 0, log_level)
    print(f"[PYTHON]  ANS context: max_chunks={ctx.max_num_chunks}, "
          f"chunk_size={ctx.max_chunk_size}, max_comp_chunk={ctx.max_comp_chunk_bytes}")
    num_batches = (num_chunks + ctx.max_num_chunks - 1) // ctx.max_num_chunks
    print(f"[PYTHON]  Total chunks={num_chunks}, will use {num_batches} batch(es)")

    # --- D2H ---
    print("\n--- D2H (compress + pack + transfer + scatter) ---")
    comp_sizes_out = torch.zeros(num_chunks, dtype=torch.int64).pin_memory()

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.cuda.stream(transfer_stream):
        ans_compress_and_d2h(
            ctx,
            gpu_block_ids, gpu_ptrs,
            gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
            cpu_block_ids, cpu_blocks,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            chunk_size,
            0, num_layers,
            False, 0,
            comp_sizes_out,
        )
    transfer_stream.synchronize()
    t1 = time.time()

    total_uncomp = num_chunks * chunk_size
    total_comp = comp_sizes_out.sum().item()
    print(f"  D2H time: {(t1-t0)*1000:.2f} ms")
    print(f"  Uncompressed: {total_uncomp / 1024 / 1024:.2f} MB")
    print(f"  Compressed:   {total_comp / 1024 / 1024:.2f} MB")
    print(f"  Ratio:        {total_uncomp / total_comp:.2f}x")

    # --- H2D ---
    print("\n--- H2D (gather + transfer + unpack + decompress) ---")
    comp_sizes_meta = comp_sizes_out.reshape(num_layers * kv_dim, num_blocks)
    t2 = time.time()
    with torch.cuda.stream(transfer_stream):
        ans_h2d_and_decompress(
            ctx,
            gpu_block_ids, gpu_ptrs,
            gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
            cpu_block_ids, cpu_blocks,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            chunk_size,
            0, num_layers,
            False, 0,
            comp_sizes_meta,
        )
    transfer_stream.synchronize()
    t3 = time.time()
    print(f"  H2D time: {(t3-t2)*1000:.2f} ms")

    # --- Verify ---
    print("\n--- Verification ---")
    all_match = True
    for layer_idx in range(num_layers):
        if not torch.equal(gpu_blocks[layer_idx], original_data[layer_idx]):
            all_match = False
            diff = (gpu_blocks[layer_idx].float() - original_data[layer_idx].float()).abs()
            print(f"  Layer {layer_idx}: MISMATCH! max_diff={diff.max().item():.6f}")
            break

    if all_match:
        print(f"  All {num_layers} layers: PASSED (exact match)")
    else:
        print("  FAILED")

    ctx.destroy()
    return all_match


def test_baseline_comparison(num_layers=4, num_blocks=4, tokens_per_block=16,
                             num_heads=8, head_size=128, dtype=torch.bfloat16,
                             num_warmup=3, num_iters=10):
    """Compare baseline (kernel mode) vs nvcomp."""
    device = "cuda:0"
    transfer_stream = torch.cuda.Stream()
    print(f"\n{'='*60}")
    print(f"Baseline comparison: layers={num_layers}, blocks={num_blocks}")
    print(f"  CPU layout={CPU_LAYOUT}, baseline: kernel mode (transfer_num_cta={TRANSFER_NUM_CTA})")
    print(f"{'='*60}")

    gpu_blocks, cpu_blocks = make_kv_cache(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size, dtype, device)
    gpu_ptrs = get_gpu_tensor_ptrs(gpu_blocks)
    (chunk_size,
     gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
     cpu_kv_stride, cpu_layer_stride, cpu_block_stride) = compute_strides(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size, dtype)

    gpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    cpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    kv_dim = 2
    num_chunks = num_layers * kv_dim * num_blocks
    total_bytes = num_chunks * chunk_size

    print(f"\n  Total data per direction: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"  Total chunks={num_chunks}")

    # --- Baseline: kernel mode (use_ce_transfer=False, transfer_num_cta=TRANSFER_NUM_CTA) ---
    print(f"\n  Baseline (kernel mode, num_cta={TRANSFER_NUM_CTA}, no compression):")

    def run_baseline(is_h2d):
        with torch.cuda.stream(transfer_stream):
            transfer_kv_blocks(
                gpu_block_ids, gpu_ptrs,
                gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
                cpu_block_ids, cpu_blocks,
                cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
                chunk_size, 0, num_layers,
                TRANSFER_NUM_CTA, is_h2d, False, False, 0)

    for _ in range(num_warmup):
        run_baseline(is_h2d=False)
    transfer_stream.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        run_baseline(is_h2d=False)
    transfer_stream.synchronize()
    t1 = time.time()
    baseline_d2h_ms = (t1 - t0) / num_iters * 1000

    for _ in range(num_warmup):
        run_baseline(is_h2d=True)
    transfer_stream.synchronize()
    t2 = time.time()
    for _ in range(num_iters):
        run_baseline(is_h2d=True)
    transfer_stream.synchronize()
    t3 = time.time()
    baseline_h2d_ms = (t3 - t2) / num_iters * 1000

    print(f"    D2H: {baseline_d2h_ms:.2f} ms  "
          f"({total_bytes / baseline_d2h_ms * 1000 / 1e9:.2f} GB/s)")
    print(f"    H2D: {baseline_h2d_ms:.2f} ms  "
          f"({total_bytes / baseline_h2d_ms * 1000 / 1e9:.2f} GB/s)")

    # --- nvcomp ---
    ctx = ANSTransferContext(BATCH_SIZE, chunk_size, 0, 0)
    comp_sizes = torch.zeros(num_chunks, dtype=torch.int64).pin_memory()

    num_batches = (num_chunks + ctx.max_num_chunks - 1) // ctx.max_num_chunks
    print(f"  Total chunks={num_chunks}, nvcomp batches={num_batches}")


    def run_nvcomp_d2h():
        with torch.cuda.stream(transfer_stream):
            ans_compress_and_d2h(
                ctx, gpu_block_ids, gpu_ptrs,
                gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
                cpu_block_ids, cpu_blocks,
                cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
                chunk_size, 0, num_layers, False, 0, comp_sizes)

    comp_sizes_meta = comp_sizes.reshape(num_layers * kv_dim, num_blocks)

    def run_nvcomp_h2d():
        with torch.cuda.stream(transfer_stream):
            ans_h2d_and_decompress(
                ctx, gpu_block_ids, gpu_ptrs,
                gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
                cpu_block_ids, cpu_blocks,
                cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
                chunk_size, 0, num_layers, False, 0, comp_sizes_meta)

    for _ in range(num_warmup):
        run_nvcomp_d2h()
    transfer_stream.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        run_nvcomp_d2h()
    transfer_stream.synchronize()
    t1 = time.time()
    nvcomp_d2h_ms = (t1 - t0) / num_iters * 1000

    for _ in range(num_warmup):
        run_nvcomp_h2d()
    transfer_stream.synchronize()
    t2 = time.time()
    for _ in range(num_iters):
        run_nvcomp_h2d()
    transfer_stream.synchronize()
    t3 = time.time()
    nvcomp_h2d_ms = (t3 - t2) / num_iters * 1000

    total_comp = comp_sizes.sum().item()
    print(f"    Compression ratio: {total_bytes / total_comp:.2f}x")
    print(f"    D2H: {nvcomp_d2h_ms:.2f} ms  (speedup {baseline_d2h_ms / nvcomp_d2h_ms:.2f}x)")
    print(f"    H2D: {nvcomp_h2d_ms:.2f} ms  (speedup {baseline_h2d_ms / nvcomp_h2d_ms:.2f}x)")

    ctx.destroy()


if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("No CUDA device available, exiting.")
        sys.exit(1)

    # Qwen2.5-7B-Instruct: 28 layers, 4 KV heads (GQA), head_size=128
    # 32K total tokens / 16 tokens_per_block = 2048 blocks
    NUM_LAYERS = 28
    NUM_KV_HEADS = 4
    HEAD_SIZE = 128
    TOKENS_PER_BLOCK = 64
    TOTAL_TOKENS = 32768 # total blocks = total tokens / 16 tokens_per_block
    NUM_BLOCKS = TOTAL_TOKENS // TOKENS_PER_BLOCK  # 2048
    # nvcomp total chunks = 28 * 2 * 2048 = 114,688 | [num_blocks, num_layers, 2, tpb, nh, hs]
    # nvcomp total batch = 114688 // 4096 = 28

    passed = test_roundtrip(num_layers=NUM_LAYERS, num_blocks=NUM_BLOCKS,
                            tokens_per_block=TOKENS_PER_BLOCK,
                            num_heads=NUM_KV_HEADS, head_size=HEAD_SIZE,)
    if not passed:
        print("\nRoundtrip test FAILED!")
        sys.exit(1)

    test_baseline_comparison(num_layers=NUM_LAYERS, num_blocks=NUM_BLOCKS,
                             tokens_per_block=TOKENS_PER_BLOCK,
                             num_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)

    print("\n\nAll tests passed!")
