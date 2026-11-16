"""vLLM KV cache groups with mixed page sizes (DeepSeek-V4.1) as FlexKV layer groups."""
import pytest
import torch

pytest.importorskip("vllm.v1.kv_cache_layout")

from vllm.v1.kv_cache_interface import (  # noqa: E402
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from flexkv.common.config import LayerGroupSpec  # noqa: E402
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType  # noqa: E402
from flexkv.integration.config import FlexKVConfig  # noqa: E402
from flexkv.transfer.template import compile_gpu_regions  # noqa: E402

pytestmark = pytest.mark.unit

P = "language_model.model.layers"
NUM_BLOCKS = 16
BLOCK = 116928  # one block of the MLA group: every page below, back to back


def _mla(ratio, head, content):
    return MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=head, dtype=torch.uint8,
                            tokens_per_state=ratio, cache_dtype_str="fp8_ds_mla", alignment=576,
                            model_version="deepseek_v4", state_content_bytes=content)


def _dsv41_config(scheduler: bool) -> KVCacheConfig:
    """The H20 TP8 layout: SWA (bounded replay), compressed KV + indexer K,
    compressor rings; only the MLA group is prefix-cacheable."""
    swa = SlidingWindowMLASpec(block_size=32, num_kv_heads=1, head_size=512, dtype=torch.uint8,
                               sliding_window=128, cache_dtype_str="fp8_ds_mla",
                               state_content_bytes=584, alignment=576,
                               model_version="deepseek_v4", bounded_replay=True)
    swa_names = [f"{P}.{i}.attn.swa_cache" for i in range(40)]
    specs = {}
    for i in (2, 8, 14, 20):
        ratio = 1 if i == 20 else 2
        specs[f"{P}.{i}.attn"] = _mla(ratio, 512, 584)
        specs[f"{P}.{i}.attn.indexer.k_cache"] = _mla(ratio, 132, None)
    mla_spec = (next(iter(specs.values())) if scheduler
                else UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs))
    ring = CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0,
                              dtype=torch.float32)
    ring_names = [f"{P}.{i}.attn.compressor.state_cache" for i in (2, 8, 14)]
    size = NUM_BLOCKS * BLOCK
    tensors = [
        KVCacheTensor(size, swa_names, 19008, BLOCK),
        KVCacheTensor(size, [f"{P}.{i}.attn" for i in (2, 8, 14)], 19008, BLOCK, 0),
        KVCacheTensor(size, [f"{P}.{i}.attn.indexer.k_cache" for i in (2, 8, 14)], 4608,
                      BLOCK, 57024),
        KVCacheTensor(size, [f"{P}.20.attn"], 37440, BLOCK, 70848),
        KVCacheTensor(size, [f"{P}.20.attn.indexer.k_cache"], 8640, BLOCK, 108288),
        KVCacheTensor(size, ring_names, 32768, BLOCK),
    ]
    groups = [KVCacheGroupSpec(swa_names, swa), KVCacheGroupSpec(list(specs), mla_spec),
              KVCacheGroupSpec(ring_names, ring)]
    config = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=tensors, kv_cache_groups=groups)
    config.kv_cache_layout = "BLHNC"
    return config


@pytest.mark.parametrize("scheduler", [True, False])
def test_dsv41_mla_group_as_layer_groups(scheduler):
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 8
    flexkv_config._configure_kv_layout_from_vllm_cache_config(_dsv41_config(scheduler))

    group = flexkv_config.vllm_kv_group
    assert group.group_id == 1 and group.num_blocks == NUM_BLOCKS
    assert group.layer_names == tuple(f"{P}.{i}.attn" for i in (2, 8, 14, 20))
    assert flexkv_config.cache_config.tokens_per_block == 64
    model_config = flexkv_config.model_config
    assert (model_config.num_layers, model_config.num_kv_heads, model_config.kv_dim) == (4, 1, 1)
    assert [(g.head_size, g.layer_indices, g.compress_ratio)
            for g in model_config.layer_groups] == [
        (19008, [0, 1, 2], 64), (4608, [0, 1, 2], 64), (37440, [3], 64), (8640, [3], 64)]
    assert group.group_layer_names[1] == tuple(
        f"{P}.{i}.attn.indexer.k_cache" for i in (2, 8, 14))
    # Every TP rank holds the same MLA bytes: one copy per host block.
    assert model_config.host_rank_copies == 1


def test_block_stride_from_interleaved_view():
    raw = torch.zeros(NUM_BLOCKS * BLOCK, dtype=torch.uint8)
    view = torch.as_strided(raw, (NUM_BLOCKS, 1, 32, 584), (BLOCK, 0, 584, 1), 57024)
    group = LayerGroupSpec(num_layers=1, num_kv_heads=1, head_size=4608, layer_indices=[0],
                           dtype=torch.uint8, compress_ratio=64)
    layout = KVCacheLayout(type=KVCacheLayoutType.LAYERFIRST, num_layer=1, num_block=NUM_BLOCKS,
                           tokens_per_block=1, num_head=1, head_size=4608, kv_dim=1,
                           num_kv_heads=1)
    (region,) = compile_gpu_regions([group], [[layout]], 64, 1, torch.uint8,
                                    tensors_per_group_device=[[view]])
    assert (region.chunk_bytes, region.block_stride) == (4608, BLOCK)
