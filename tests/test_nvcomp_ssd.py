"""
Test nvcomp ANS compression integrated with FlexKV CPU-SSD transfer.

Measures CPU<->SSD bandwidth for baseline (no compression) vs nvcomp ANS.
The nvcomp path requires GPU-compressed data in CPU buffer, so the flow is:
  GPU -> compress -> D2H -> CPU (compressed) -> SSD -> CPU -> H2D -> decompress -> GPU

Usage:
  FLEXKV_ENABLE_NVCOMP=1 python tests/test_nvcomp_ssd.py
  FLEXKV_ENABLE_NVCOMP=1 FLEXKV_NVCOMP_SSD_THREADS=32 python tests/test_nvcomp_ssd.py
"""

import os
import sys
import time
import tempfile
import shutil
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flexkv.c_ext import transfer_kv_blocks, transfer_kv_blocks_ssd, SSDIOCTX

try:
    from flexkv.c_ext import (
        ANSTransferContext, ans_compress_and_d2h, ans_h2d_and_decompress,
        ans_transfer_kv_blocks_ssd,
    )
    NVCOMP_AVAILABLE = True
except ImportError:
    NVCOMP_AVAILABLE = False
    print("WARNING: nvcomp not available. Build with FLEXKV_ENABLE_NVCOMP=1")

BATCH_SIZE = int(os.environ.get("FLEXKV_NVCOMP_BATCH_SIZE", "4096"))
NUM_THREADS = int(os.environ.get("FLEXKV_NVCOMP_SSD_THREADS", "16"))
TRANSFER_NUM_CTA = int(os.environ.get("FLEXKV_TRANSFER_NUM_CTA", "4"))


def make_kv_caches(num_layers, num_blocks, tokens_per_block, num_heads,
                   head_size, dtype=torch.bfloat16, device="cuda:0"):
    """Create GPU (LAYERFIRST) and CPU (BLOCKFIRST) KV caches."""
    # GPU: [num_layers, 2, num_blocks, tpb, nh, hs]
    shape_per_layer = (2, num_blocks, tokens_per_block, num_heads, head_size)
    gpu_cache = torch.randn(
        (num_layers,) + shape_per_layer, dtype=dtype, device=device)
    gpu_blocks = [gpu_cache[i] for i in range(num_layers)]

    # CPU BLOCKFIRST: [num_blocks, num_layers, 2, tpb, nh, hs]
    cpu_shape = (num_blocks, num_layers, 2, tokens_per_block, num_heads, head_size)
    cpu_cache = torch.zeros(cpu_shape, dtype=dtype, device="cpu").pin_memory()

    return gpu_blocks, cpu_cache


def compute_strides(num_layers, num_blocks, tokens_per_block, num_heads,
                    head_size, dtype):
    """Compute byte strides for GPU (LAYERFIRST) and CPU (BLOCKFIRST)."""
    elem = dtype.itemsize if hasattr(dtype, 'itemsize') else \
        torch.tensor([], dtype=dtype).element_size()
    chunk = tokens_per_block * num_heads * head_size
    chunk_bytes = chunk * elem

    # GPU LAYERFIRST: per-layer [2, num_blocks, tpb, nh, hs]
    gpu_kv_stride = num_blocks * chunk_bytes
    gpu_block_stride = chunk_bytes
    gpu_layer_stride = 2 * num_blocks * chunk_bytes

    # CPU BLOCKFIRST: [num_blocks, num_layers, 2, tpb, nh, hs]
    cpu_kv_stride = chunk_bytes
    cpu_layer_stride = 2 * chunk_bytes
    cpu_block_stride = num_layers * 2 * chunk_bytes

    # SSD BLOCKFIRST (same layout as CPU per file)
    ssd_kv_stride = cpu_kv_stride
    ssd_layer_stride = cpu_layer_stride

    return (chunk_bytes,
            gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            ssd_kv_stride, ssd_layer_stride)


def get_gpu_tensor_ptrs(gpu_blocks):
    return torch.tensor(
        [b.data_ptr() for b in gpu_blocks], dtype=torch.int64).pin_memory()


ANS_COMP_HEADER_SIZE = 16
ANS_DIRECT_IO_ALIGN = 512


def compute_compression_ratio(cpu_cache, num_blocks, num_layers, chunk_bytes,
                              cpu_kv_stride, cpu_layer_stride, cpu_block_stride):
    """Scan ANS headers in CPU buffer to compute actual compression ratio.

    Each compressed chunk starts with an int64 compressed_size at offset 0.
    Actual I/O size = align_up(HEADER + compressed_size, 512), capped at chunk_bytes.
    """
    import ctypes
    base_ptr = cpu_cache.data_ptr()
    total_comp = 0
    total_raw = 0
    for bid in range(num_blocks):
        for lid in range(num_layers):
            for kv in range(2):  # K, V
                off = bid * cpu_block_stride + lid * cpu_layer_stride + kv * cpu_kv_stride
                comp_size = ctypes.c_int64.from_address(base_ptr + off).value
                actual = ANS_COMP_HEADER_SIZE + comp_size
                actual = (actual + ANS_DIRECT_IO_ALIGN - 1) & ~(ANS_DIRECT_IO_ALIGN - 1)
                actual = min(actual, chunk_bytes)
                total_comp += actual
                total_raw += chunk_bytes
    ratio = total_raw / total_comp if total_comp > 0 else 0
    return total_raw, total_comp, ratio


def setup_ssd(num_blocks, block_size_bytes, num_devices=1,
              num_files_per_device=1, tmpdir=None):
    """Create temp SSD files and SSDIOCTX."""
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="flexkv_ssd_test_")

    total_files = num_devices * num_files_per_device
    num_blocks_per_file = (num_blocks + total_files - 1) // total_files
    file_size = num_blocks_per_file * block_size_bytes

    ssd_files = {}
    for dev in range(num_devices):
        ssd_files[dev] = []
        dev_dir = os.path.join(tmpdir, f"dev{dev}")
        os.makedirs(dev_dir, exist_ok=True)
        for f_idx in range(num_files_per_device):
            fpath = os.path.join(dev_dir, f"ssd_cache_{dev}_{f_idx}.bin")
            with open(fpath, "wb+", buffering=0) as f:
                os.truncate(f.fileno(), file_size)
                os.fsync(f.fileno())
            ssd_files[dev].append(fpath)

    ioctx = SSDIOCTX(ssd_files, num_devices, 512, 0)
    return ioctx, tmpdir, num_blocks_per_file


def compress_gpu_to_cpu(ctx, gpu_blocks, gpu_ptrs, cpu_cache,
                        gpu_block_ids, cpu_block_ids,
                        gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
                        cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
                        chunk_bytes, num_layers, stream):
    """GPU -> ANS compress -> D2H to CPU."""
    with torch.cuda.stream(stream):
        ans_compress_and_d2h(
            ctx, gpu_block_ids, gpu_ptrs,
            gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
            cpu_block_ids, cpu_cache,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            chunk_bytes, 0, num_layers, False, 0)
    stream.synchronize()


def decompress_cpu_to_gpu(ctx, gpu_blocks, gpu_ptrs, cpu_cache,
                          gpu_block_ids, cpu_block_ids,
                          gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
                          cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
                          chunk_bytes, num_layers, stream):
    """CPU -> H2D -> ANS decompress to GPU."""
    with torch.cuda.stream(stream):
        ans_h2d_and_decompress(
            ctx, gpu_block_ids, gpu_ptrs,
            gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
            cpu_block_ids, cpu_cache,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            chunk_bytes, 0, num_layers, False, 0)
    stream.synchronize()


def test_ssd_roundtrip(num_layers=28, num_blocks=64, tokens_per_block=16,
                       num_heads=4, head_size=128, dtype=torch.bfloat16):
    """Full roundtrip: GPU -> compress -> CPU -> SSD -> CPU -> decompress -> GPU."""
    if not NVCOMP_AVAILABLE:
        print("SKIP: nvcomp not available")
        return True

    print(f"\n{'='*60}")
    print(f"SSD Roundtrip: layers={num_layers}, blocks={num_blocks}, "
          f"tpb={tokens_per_block}, heads={num_heads}, hs={head_size}")
    print(f"{'='*60}")

    gpu_blocks, cpu_cache = make_kv_caches(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size, dtype)
    gpu_ptrs = get_gpu_tensor_ptrs(gpu_blocks)
    original_gpu = [b.clone() for b in gpu_blocks]

    (chunk_bytes,
     gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
     cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
     ssd_kv_stride, ssd_layer_stride) = compute_strides(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size, dtype)

    ioctx, tmpdir, num_blocks_per_file = setup_ssd(num_blocks, cpu_block_stride)

    gpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    cpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    ssd_block_ids = torch.arange(num_blocks, dtype=torch.int64)
    layer_id_list = torch.arange(num_layers, dtype=torch.int32)
    stream = torch.cuda.Stream()
    ctx = ANSTransferContext(BATCH_SIZE, chunk_bytes, 0, 1)

    ssd_kwargs = dict(
        ioctx=ioctx, cpu_layer_id_list=layer_id_list,
        cpu_tensor_ptr=cpu_cache.data_ptr(),
        ssd_block_ids=ssd_block_ids, cpu_block_ids=cpu_block_ids,
        cpu_layer_stride_in_bytes=cpu_layer_stride,
        cpu_kv_stride_in_bytes=cpu_kv_stride,
        ssd_layer_stride_in_bytes=ssd_layer_stride,
        ssd_kv_stride_in_bytes=ssd_kv_stride,
        chunk_size_in_bytes=chunk_bytes,
        block_stride_in_bytes=cpu_block_stride,
        num_blocks_per_file=num_blocks_per_file,
        round_robin=1, num_threads_per_device=NUM_THREADS, is_mla=False,
    )

    try:
        # Step 1: GPU -> compress -> CPU
        print("  [1] GPU -> ANS compress -> CPU")
        compress_gpu_to_cpu(
            ctx, gpu_blocks, gpu_ptrs, cpu_cache,
            gpu_block_ids, cpu_block_ids,
            gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            chunk_bytes, num_layers, stream)

        # Step 2: CPU (compressed) -> SSD
        print("  [2] CPU (compressed) -> SSD")
        ans_transfer_kv_blocks_ssd(is_read=False, **ssd_kwargs)

        # Step 3: Clear CPU
        cpu_cache.zero_()

        # Step 4: SSD -> CPU (compressed)
        print("  [3] SSD -> CPU (compressed)")
        ans_transfer_kv_blocks_ssd(is_read=True, **ssd_kwargs)

        # Step 5: CPU -> decompress -> GPU
        print("  [4] CPU -> H2D -> ANS decompress -> GPU")
        decompress_cpu_to_gpu(
            ctx, gpu_blocks, gpu_ptrs, cpu_cache,
            gpu_block_ids, cpu_block_ids,
            gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            chunk_bytes, num_layers, stream)

        # Verify
        all_match = True
        for i in range(num_layers):
            if not torch.equal(gpu_blocks[i], original_gpu[i]):
                diff = (gpu_blocks[i].float() - original_gpu[i].float()).abs()
                print(f"  Layer {i}: MISMATCH! max_diff={diff.max().item():.6f}")
                all_match = False
                break

        if all_match:
            print(f"  PASSED: {num_blocks} blocks x {num_layers} layers exact match")
        else:
            print("  FAILED")

        ctx.destroy()
        return all_match
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_ssd_perf(num_layers=28, num_blocks=2048, tokens_per_block=16,
                  num_heads=4, head_size=128, dtype=torch.bfloat16,
                  num_warmup=3, num_iters=5):
    """Benchmark baseline vs nvcomp for CPU<->SSD."""
    print(f"\n{'='*60}")
    print(f"SSD Performance: layers={num_layers}, blocks={num_blocks}, "
          f"tpb={tokens_per_block}, heads={num_heads}, hs={head_size}")
    print(f"  ssd_threads={NUM_THREADS}")
    print(f"{'='*60}")

    gpu_blocks, cpu_cache = make_kv_caches(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size, dtype)
    gpu_ptrs = get_gpu_tensor_ptrs(gpu_blocks)

    (chunk_bytes,
     gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
     cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
     ssd_kv_stride, ssd_layer_stride) = compute_strides(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size, dtype)

    ioctx, tmpdir, num_blocks_per_file = setup_ssd(num_blocks, cpu_block_stride)

    gpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    cpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    ssd_block_ids = torch.arange(num_blocks, dtype=torch.int64)
    layer_id_list = torch.arange(num_layers, dtype=torch.int32)
    stream = torch.cuda.Stream()

    kv_dim = 2
    total_bytes = num_blocks * num_layers * kv_dim * chunk_bytes
    print(f"  Total data (uncompressed): {total_bytes / 1024 / 1024:.2f} MB")

    ssd_kwargs = dict(
        ioctx=ioctx, cpu_layer_id_list=layer_id_list,
        cpu_tensor_ptr=cpu_cache.data_ptr(),
        ssd_block_ids=ssd_block_ids, cpu_block_ids=cpu_block_ids,
        cpu_layer_stride_in_bytes=cpu_layer_stride,
        cpu_kv_stride_in_bytes=cpu_kv_stride,
        ssd_layer_stride_in_bytes=ssd_layer_stride,
        ssd_kv_stride_in_bytes=ssd_kv_stride,
        chunk_size_in_bytes=chunk_bytes,
        block_stride_in_bytes=cpu_block_stride,
        num_blocks_per_file=num_blocks_per_file,
        round_robin=1, num_threads_per_device=NUM_THREADS, is_mla=False,
    )

    def bench_ssd(fn, is_read, label, **extra_kwargs):
        kwargs = {**ssd_kwargs, **extra_kwargs}
        for _ in range(num_warmup):
            fn(is_read=is_read, **kwargs)
        t0 = time.time()
        for _ in range(num_iters):
            fn(is_read=is_read, **kwargs)
        t1 = time.time()
        avg_ms = (t1 - t0) / num_iters * 1000
        bw = total_bytes / avg_ms * 1000 / 1e9
        print(f"    {label}: {avg_ms:.2f} ms  ({bw:.2f} GB/s)")
        return avg_ms

    try:
        # === Baseline: raw data CPU <-> SSD ===
        # First populate CPU with raw GPU data via baseline D2H
        print("\n  Preparing: baseline D2H (GPU -> CPU, no compression)...")
        with torch.cuda.stream(stream):
            transfer_kv_blocks(
                gpu_block_ids, gpu_ptrs,
                gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
                cpu_block_ids, cpu_cache,
                cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
                chunk_bytes, 0, num_layers,
                TRANSFER_NUM_CTA, False, False, False, 0)
        stream.synchronize()

        print(f"\n  Baseline (no compression, raw I/O):")
        baseline_write = bench_ssd(transfer_kv_blocks_ssd, False, "H2DISK (write)")
        baseline_read = bench_ssd(transfer_kv_blocks_ssd, True, "DISK2H (read) ")

        # === NVComp: compressed data CPU <-> SSD ===
        if NVCOMP_AVAILABLE:
            num_chunks = num_layers * kv_dim * num_blocks
            ctx = ANSTransferContext(BATCH_SIZE, chunk_bytes, 0, 0)
            num_batches = (num_chunks + ctx.max_num_chunks - 1) // ctx.max_num_chunks

            # Populate CPU with compressed data
            print(f"\n  Preparing: nvcomp D2H (GPU -> compress -> CPU)...")
            compress_gpu_to_cpu(
                ctx, gpu_blocks, gpu_ptrs, cpu_cache,
                gpu_block_ids, cpu_block_ids,
                gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
                cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
                chunk_bytes, num_layers, stream)

            total_raw, total_comp, ratio = compute_compression_ratio(
                cpu_cache, num_blocks, num_layers, chunk_bytes,
                cpu_kv_stride, cpu_layer_stride, cpu_block_stride)
            print(f"  Compression: {total_raw / 1024 / 1024:.2f} MB -> "
                  f"{total_comp / 1024 / 1024:.2f} MB (ratio {ratio:.2f}x)")

            # --- Full I/O mode (compressed_io=False, default) ---
            print(f"\n  NVComp ANS full I/O (compressed_io=False, {num_batches} batches):")
            full_write = bench_ssd(ans_transfer_kv_blocks_ssd, False, "H2DISK (write)",
                                   compressed_io=False)
            full_read = bench_ssd(ans_transfer_kv_blocks_ssd, True, "DISK2H (read) ",
                                  compressed_io=False)

            # --- Compressed I/O mode (compressed_io=True) ---
            print(f"\n  NVComp ANS compressed I/O (compressed_io=True, {num_batches} batches):")
            comp_write = bench_ssd(ans_transfer_kv_blocks_ssd, False, "H2DISK (write)",
                                   compressed_io=True)
            comp_read = bench_ssd(ans_transfer_kv_blocks_ssd, True, "DISK2H (read) ",
                                  compressed_io=True)

            print(f"\n  Summary (vs baseline):")
            print(f"    Full I/O:       H2DISK {baseline_write / full_write:.2f}x, "
                  f"DISK2H {baseline_read / full_read:.2f}x")
            print(f"    Compressed I/O: H2DISK {baseline_write / comp_write:.2f}x, "
                  f"DISK2H {baseline_read / comp_read:.2f}x")

            ctx.destroy()
        else:
            print("\n  NVComp not available, skipping.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA device, exiting.")
        sys.exit(1)

    # Qwen2.5-7B: 28 layers, 4 KV heads (GQA), head_size=128, tpb=16
    NUM_LAYERS = 28
    NUM_KV_HEADS = 4
    HEAD_SIZE = 128
    TOKENS_PER_BLOCK = 16
    TOTAL_TOKENS = 32768
    NUM_BLOCKS = TOTAL_TOKENS // TOKENS_PER_BLOCK  # 2048

    print(f"Config: Qwen2.5-7B, {NUM_LAYERS}L, {NUM_KV_HEADS}H, "
          f"hs={HEAD_SIZE}, tpb={TOKENS_PER_BLOCK}, blocks={NUM_BLOCKS}")

    # Roundtrip correctness
    passed = test_ssd_roundtrip(
        num_layers=NUM_LAYERS, num_blocks=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)
    if not passed:
        print("\nRoundtrip FAILED!")
        sys.exit(1)

    # Performance benchmark
    test_ssd_perf(
        num_layers=NUM_LAYERS, num_blocks=NUM_BLOCKS,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)

    print("\n\nAll tests passed!")
