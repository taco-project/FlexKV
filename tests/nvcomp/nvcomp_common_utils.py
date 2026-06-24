import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

DEFUALT_MHA_CONFIG = dict(num_head=8, head_dim=128)
DEFUALT_MLA_SHAPE = dict(num_head=1, head_dim=576)


def resolve_case(chunk_size_per_device, tp, is_mla, dtype):
    elem = dtype.itemsize
    shape = DEFUALT_MLA_SHAPE if is_mla else DEFUALT_MHA_CONFIG
    num_head, head_dim = shape["num_head"], shape["head_dim"]
    kv_dim = 1 if is_mla else 2
    heads_per_device = num_head if is_mla else num_head // tp
    tpb = max(
        1,
        1 << (
            (chunk_size_per_device // (heads_per_device * head_dim * elem))
            .bit_length() - 1
        ),
    )
    return dict(
        num_head=num_head,
        head_dim=head_dim,
        kv_dim=kv_dim,
        tokens_per_block=tpb,
        chunk_bytes=tpb * heads_per_device * head_dim * elem,
    )


def make_gpu_cache(shape, dtype, device):
    """randn cache on `device`, FP8-aware.

    torch.randn has no FP8 path -> generate bf16 then cast (matches real model
    quantization: bf16_tensor.to(float8_e4m3fn)).
    """
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.randn(shape, dtype=torch.bfloat16, device=device).to(dtype)
    return torch.randn(shape, dtype=dtype, device=device)


def make_kv_cache(num_layers, num_blocks, tokens_per_block, num_heads, head_size,
                  dtype=torch.bfloat16, device="cuda:0",
                  cpu_layout_name="BLOCKFIRST", is_mla=False):
    """Create GPU (LAYERFIRST per-layer list) and CPU (cpu_layout) KV caches.

    GPU: list of per-layer tensors, each [kv_dim, num_blocks, tpb, nh, hs],
         backed by one contiguous tensor (like vLLM's KV cache pool).
    CPU: a single tensor shaped by KVCacheLayout (BLOCKFIRST or LAYERFIRST).
    """
    kv_dim = 1 if is_mla else 2
    shape = (num_layers, kv_dim, num_blocks, tokens_per_block, num_heads, head_size)
    gpu_cache = make_gpu_cache(shape, dtype, device)
    gpu_blocks = [gpu_cache[i] for i in range(num_layers)]

    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType[cpu_layout_name.upper()],
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tokens_per_block, num_head=num_heads,
        head_size=head_size, is_mla=is_mla)
    cpu_blocks = torch.zeros(tuple(cpu_layout.kv_shape), dtype=dtype).pin_memory()
    return gpu_blocks, cpu_blocks

def compute_strides(num_layers, num_blocks, tokens_per_block, num_heads, head_size,
                    dtype, cpu_layout_name="BLOCKFIRST", is_mla=False):
    """Byte strides via KVCacheLayout (GPU=LAYERFIRST, CPU=cpu_layout).

    Returns (chunk_size,
             gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
             cpu_kv_stride, cpu_layer_stride, cpu_block_stride).
    """
    elem = dtype.itemsize
    gpu = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST, num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tokens_per_block, num_head=num_heads,
        head_size=head_size, is_mla=is_mla)
    cpu = KVCacheLayout(
        type=KVCacheLayoutType[cpu_layout_name.upper()], num_layer=num_layers,
        num_block=num_blocks, tokens_per_block=tokens_per_block, num_head=num_heads,
        head_size=head_size, is_mla=is_mla)
    return (gpu.get_chunk_size() * elem,
            gpu.get_kv_stride() * elem,
            gpu.get_block_stride() * elem,
            gpu.get_layer_stride() * elem,
            cpu.get_kv_stride() * elem,
            cpu.get_layer_stride() * elem,
            cpu.get_block_stride() * elem)

