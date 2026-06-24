import pytest
import torch
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.transfer.compression.common.strategy import NullCompressionStrategy
from nvcomp_common_utils import (
    make_kv_cache, make_gpu_cache, compute_strides,
    DEFUALT_MHA_CONFIG, resolve_case,
)

try:
    from flexkv.c_ext import (
        ANSTransferContext,
        transfer_kv_blocks_ans_comp,
        transfer_kv_blocks_ans_decomp,
        TPTransferThreadGroup,
    )
    NVCOMP_AVAILABLE = True
except ImportError:
    NVCOMP_AVAILABLE = False

from flexkv.transfer.compression.ans import ans_utils

device = "cuda:0"


@pytest.mark.parametrize("cpu_layout_name", [pytest.param("BLOCKFIRST", id="bfirst"), pytest.param("LAYERFIRST", id="lfirst")])
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float8_e4m3fn, id="fp8"),])
@pytest.mark.parametrize("tp_size", [pytest.param(1, id="tp_1")])
@pytest.mark.parametrize("is_mla", [pytest.param(False, id="mha"), pytest.param(True, id="mla")])
@pytest.mark.parametrize("chunk_size_per_device", [
    pytest.param(8 * 1024, id="cpd_8KB"),
    pytest.param(16 * 1024, id="cpd_16KB"),
    pytest.param(32 * 1024, id="cpd_32KB"),
    pytest.param(18 * 1024, id="cpd_18KB"), # MLA
])
def test_roundtrip(cpu_layout_name, dtype, tp_size, is_mla, chunk_size_per_device,
                   num_layers=4, num_blocks=2):
    if not NVCOMP_AVAILABLE:
        pytest.skip("nvcomp not available")
    if is_mla != (chunk_size_per_device == 18 * 1024):
        pytest.skip("18KB is MLA-only; 8/16/32KB are MHA-only")

    case = resolve_case(chunk_size_per_device, tp_size, is_mla, dtype)

    transfer_stream = torch.cuda.Stream()

    num_heads, head_size = case["num_head"], case["head_dim"]
    tokens_per_block, kv_dim = case["tokens_per_block"], case["kv_dim"]
    print(f"\n{'='*60}")
    print(f"    {"MLA" if is_mla else "MHA"} roundtrip test: layout={cpu_layout_name},chunk_size_per_device={case['chunk_bytes']}, layers={num_layers}, blocks={num_blocks}, kv_dim={kv_dim}, "
          f"tokens_per_block={tokens_per_block}, num_heads={num_heads}, head_size={head_size}, "
          f"dtype={dtype}, tp_size={tp_size}")
    

    gpu_blocks, cpu_blocks = make_kv_cache(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size,
        dtype, device, cpu_layout_name, is_mla)
    # int64 pointer table, pinned (C++ reads it as an array of GPU pointers).
    gpu_ptrs = torch.tensor(
        [b.data_ptr() for b in gpu_blocks], dtype=torch.int64).pin_memory()
    (chunk_size,
     gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
     cpu_kv_stride, cpu_layer_stride, cpu_block_stride) = compute_strides(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size,
        dtype, cpu_layout_name, is_mla)
    assert chunk_size == case["chunk_bytes"]

    block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    original = [b.clone() for b in gpu_blocks]

    # External per-(block, layer, kv) compressed-size table.
    size_table = torch.zeros(
        (num_blocks, num_layers, kv_dim), dtype=torch.uint32).pin_memory()
    st_ptr, st_block_stride, st_layer_stride = (
        size_table.data_ptr(), size_table.stride(0), size_table.stride(1))

    data_type = ans_utils.data_type(dtype)
    batch_size, _ = ans_utils.get_nvcomp_batch_size(
        chunk_size, data_type=data_type)
    ctx = ANSTransferContext(batch_size, chunk_size, data_type)

    # --- D2H: compress GPU -> CPU ---
    with torch.cuda.stream(transfer_stream):
        d2h_bytes = transfer_kv_blocks_ans_comp(
            ctx, block_ids, gpu_ptrs,
            gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
            block_ids, cpu_blocks,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            chunk_size, 0, num_layers, is_mla, 0,
            st_ptr, st_block_stride, st_layer_stride)
    transfer_stream.synchronize()
    
    # Every (block, layer, kv) entry must be > 0 -> kernel actually wrote it.
    table_i64 = size_table.to(torch.int64)
    assert (table_i64 > 0).all(), "size_table has zero entries after D2H"
    expected_bytes = int(table_i64.sum().item())
    assert d2h_bytes == expected_bytes

    uncompressed_bytes = size_table.numel() * chunk_size
    print(f"    compression ratio: {uncompressed_bytes / d2h_bytes:.3f}x "
          f"({uncompressed_bytes} -> {d2h_bytes} B)")
    print(f"{'='*60}")

    # Zero GPU so a no-op H2D can't masquerade as a passing roundtrip.
    for b in gpu_blocks:
        b.zero_()

    # --- H2D: decompress CPU -> GPU ---
    with torch.cuda.stream(transfer_stream):
        h2d_bytes = transfer_kv_blocks_ans_decomp(
            ctx, block_ids, gpu_ptrs,
            gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
            block_ids, cpu_blocks,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            chunk_size, 0, num_layers, is_mla, 0,
            st_ptr, st_block_stride, st_layer_stride)
    transfer_stream.synchronize()
    assert h2d_bytes == expected_bytes

    ctx.destroy()
    for li in range(num_layers):
        assert torch.equal(gpu_blocks[li], original[li]), \
            f"roundtrip mismatch at layer {li}"


@pytest.mark.parametrize("cpu_layout_name", [pytest.param("BLOCKFIRST", id="bfirst"), pytest.param("LAYERFIRST", id="lfirst")])
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float8_e4m3fn, id="fp8"),])
@pytest.mark.parametrize("tp_size", [pytest.param(2, id="tp_2"), pytest.param(4, id="tp_4"), pytest.param(8, id="tp_8")])
@pytest.mark.parametrize("is_mla", [pytest.param(False, id="mha"), pytest.param(True, id="mla")])
@pytest.mark.parametrize("chunk_size_per_device", [
    pytest.param(8 * 1024, id="cpd_8KB"),
    pytest.param(16 * 1024, id="cpd_16KB"),
    pytest.param(32 * 1024, id="cpd_32KB"),
    pytest.param(18 * 1024, id="cpd_18KB"), # MLA
])
def test_roundtrip_tp(cpu_layout_name, dtype, tp_size, is_mla, chunk_size_per_device,
                           num_layers=4, num_blocks=2):
    if not NVCOMP_AVAILABLE:
        pytest.skip("nvcomp not available")
    if is_mla != (chunk_size_per_device == 18 * 1024):
        pytest.skip("18KB is MLA-only; 8/16/32KB are MHA-only")
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"tp_size={tp_size} needs {tp_size} GPUs, "
                    f"have {torch.cuda.device_count()}")

    case = resolve_case(chunk_size_per_device, tp_size, is_mla, dtype)
    num_head, head_dim = case["num_head"], case["head_dim"]
    tokens_per_block, kv_dim = case["tokens_per_block"], case["kv_dim"]
    elem = dtype.itemsize
    layout_type = KVCacheLayoutType[cpu_layout_name.upper()]
    data_type = ans_utils.data_type(dtype)

    print(f"\n{'='*60}")
    print(f"    {"MLA" if is_mla else "MHA"} TP roundtrip: layout={cpu_layout_name}, chunk_size_per_device={chunk_size_per_device}, "
          f"tp_size={tp_size}, layers={num_layers}, blocks={num_blocks}, kv_dim={kv_dim}, "
          f"tokens_per_block={tokens_per_block}, num_heads={num_head}, head_size={head_dim}, dtype={dtype}")

    # Per-rank GPU caches (LAYERFIRST). MHA shards heads across ranks; MLA
    # replicates the KV so every rank starts with *identical* data -- required
    # because D2H compresses each block from its owner rank (cpu_block_id % tp)
    # into one canonical copy and H2D fans that copy back to all ranks.
    heads_per_gpu = num_head if is_mla else num_head // tp_size
    shape_per_layer = (kv_dim, num_blocks, tokens_per_block, heads_per_gpu, head_dim)
    all_gpu = []
    if is_mla:
        # Stage through host so replication is bit-exact on every device
        # (device-to-device copies can silently fail across NVLink islands).
        base = make_gpu_cache((num_layers,) + shape_per_layer, dtype, "cuda:0").cpu()
        for gi in range(tp_size):
            c = base.to(f"cuda:{gi}")
            all_gpu.append([c[li] for li in range(num_layers)])
    else:
        for gi in range(tp_size):
            c = make_gpu_cache((num_layers,) + shape_per_layer, dtype, f"cuda:{gi}")
            all_gpu.append([c[li] for li in range(num_layers)])
    originals = [[b.clone() for b in all_gpu[gi]] for gi in range(tp_size)]

    # Full-head CPU cache (shared across ranks), shaped by cpu_layout.
    cpu_layout = KVCacheLayout(
        type=layout_type, num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tokens_per_block, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    cpu = torch.zeros(tuple(cpu_layout.kv_shape), dtype=dtype).pin_memory()

    # Per-rank GPU strides (LAYERFIRST; MHA divides heads across ranks).
    gpu_per_rank = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST, num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tokens_per_block, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    if not is_mla:
        gpu_per_rank = gpu_per_rank.div_head(tp_size)
    per_rank_chunk = gpu_per_rank.get_chunk_size() * elem
    assert per_rank_chunk == case["chunk_bytes"]

    # CPU strides: BLOCKFIRST MHA needs the in-chunk head subview (div_head);
    # LAYERFIRST and MLA keep chunks contiguous so no division.
    if not is_mla and layout_type == KVCacheLayoutType.BLOCKFIRST:
        cpu_stride_layout = cpu_layout.div_head(tp_size)
    else:
        cpu_stride_layout = cpu_layout
    cpu_kv_stride = cpu_stride_layout.get_kv_stride() * elem
    cpu_layer_stride = cpu_stride_layout.get_layer_stride() * elem
    cpu_block_stride = cpu_layout.get_block_stride() * elem
    cpu_tp_stride = cpu_block_stride // tp_size

    # External size table. MHA: per-rank [tp, blocks, layers, kv]. MLA: canonical
    # (KV replicated) [blocks, layers, 1] with rank_stride=0.
    if is_mla:
        size_table = torch.zeros(
            (num_blocks, num_layers, 1), dtype=torch.uint32).pin_memory()
        st_args = (size_table.data_ptr(), 0,
                   size_table.stride(0), size_table.stride(1))
    else:
        size_table = torch.zeros(
            (tp_size, num_blocks, num_layers, kv_dim), dtype=torch.uint32).pin_memory()
        st_args = (size_table.data_ptr(), size_table.stride(0),
                   size_table.stride(1), size_table.stride(2))

    gpu_block_ptrs_flat = [all_gpu[i][l].data_ptr()
                           for i in range(tp_size) for l in range(num_layers)]
    gpu_device_ids = [all_gpu[i][0].device.index for i in range(tp_size)]
    batch_size, _ = ans_utils.get_nvcomp_batch_size(
        per_rank_chunk, data_type=data_type)
    tg = TPTransferThreadGroup(
        tp_size, gpu_block_ptrs_flat, num_layers, cpu.data_ptr(), num_layers,
        [gpu_per_rank.get_kv_stride() * elem] * tp_size,
        [gpu_per_rank.get_block_stride() * elem] * tp_size,
        [gpu_per_rank.get_layer_stride() * elem] * tp_size,
        [per_rank_chunk] * tp_size,
        gpu_device_ids,
        True,
        batch_size,
        data_type,
    )

    block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    transfer_num_cta = 4  # thread blocks the transfer kernel launches
    layer_id = 0          # start layer; with layer_granularity=num_layers -> all layers
    try:
        # D2H: compress GPU -> CPU.
        tg.tp_group_transfer_ans(
            block_ids, block_ids,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride,
            transfer_num_cta, False, False, layer_id, num_layers, is_mla, *st_args)
        for g in range(tp_size):
            torch.cuda.synchronize(g)

        assert (size_table.to(torch.int64) > 0).all(), \
            "size_table has zero entries after D2H"

        compressed_bytes = int(size_table.to(torch.int64).sum().item())
        uncompressed_bytes = size_table.numel() * per_rank_chunk
        print(f"    compression ratio: {uncompressed_bytes / compressed_bytes:.3f}x "
              f"({uncompressed_bytes} -> {compressed_bytes} B)")
        print(f"{'='*60}")
        
        # Zero every rank so a no-op H2D can't masquerade as a passing roundtrip.
        for gi in range(tp_size):
            for b in all_gpu[gi]:
                b.zero_()

        # H2D: decompress CPU -> GPU.
        tg.tp_group_transfer_ans(
            block_ids, block_ids,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride,
            transfer_num_cta, True, False, layer_id, num_layers, is_mla, *st_args)
        for g in range(tp_size):
            torch.cuda.synchronize(g)

        for gi in range(tp_size):
            for li in range(num_layers):
                assert torch.equal(all_gpu[gi][li], originals[gi][li]), \
                    f"TP roundtrip mismatch: rank={gi} layer={li}"
    finally:
        del tg


def test_nvcomp_engine_rejects_too_small_chunks(monkeypatch):
    """Engine disables nvcomp when per-device chunks are below ANS minimum."""
    if not NVCOMP_AVAILABLE:
        pytest.skip("nvcomp not available")

    from flexkv.common.config import CacheConfig, ModelConfig
    from flexkv.common.storage import AccessHandleType, StorageHandle
    from flexkv.transfer.transfer_engine import TransferEngine

    monkeypatch.setenv("FLEXKV_ENABLE_NVCOMP", "1")

    dtype = torch.float8_e4m3fn
    num_layers, num_blocks = 1, 4
    num_heads, head_size = DEFUALT_MHA_CONFIG["num_head"], DEFUALT_MHA_CONFIG["head_dim"]
    # 1 * 1 * 128 * 1B = 128B per-device chunk with tp_size=8.
    tokens_per_block, tp_size = 1, 8
    heads_per_rank = num_heads // tp_size
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST, num_layer=num_layers,
        num_block=1, tokens_per_block=tokens_per_block,
        num_head=heads_per_rank, head_size=head_size, is_mla=False)
    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST, num_layer=num_layers,
        num_block=num_blocks, tokens_per_block=tokens_per_block,
        num_head=heads_per_rank, head_size=head_size, is_mla=False)
    gpu_handle = StorageHandle(
        AccessHandleType.TENSOR, [torch.empty((1,), dtype=dtype)],
        gpu_layout, dtype, gpu_device_id=0)
    cpu_handle = StorageHandle(
        AccessHandleType.TENSOR,
        torch.empty((cpu_layout.get_total_elements(),), dtype=dtype),
        cpu_layout, dtype)

    engine = TransferEngine(
        {0: [gpu_handle]},
        ModelConfig(num_layers=num_layers, num_kv_heads=num_heads,
                    head_size=head_size, dtype=dtype, tp_size=tp_size,
                    dp_size=1),
        CacheConfig(tokens_per_block=tokens_per_block, enable_cpu=True,
                    num_cpu_blocks=num_blocks),
        cpu_handle=cpu_handle)

    assert isinstance(engine._compressors["gpu_cpu"], NullCompressionStrategy)
    assert isinstance(engine._compressors["gpu_cpu_tp"], NullCompressionStrategy)
    assert isinstance(engine._compressors["cpu_ssd"], NullCompressionStrategy)

