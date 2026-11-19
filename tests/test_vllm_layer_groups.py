"""vLLM KV cache groups with mixed page sizes (DeepSeek-V4.1, GQA) as FlexKV layer groups."""
import pytest
import torch

pytest.importorskip("vllm.v1.kv_cache_layout")

from vllm.v1.kv_cache_interface import (  # noqa: E402
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout  # noqa: E402

from flexkv.common.config import LayerGroupSpec, block_size_in_bytes_for_cache  # noqa: E402
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType  # noqa: E402
from flexkv.integration.config import FlexKVConfig  # noqa: E402
from flexkv.integration.vllm.vllm_v1_adapter import FlexKVWorkerConnector  # noqa: E402
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
    compressor rings; only the MLA group is prefix-cacheable. Every TP rank
    holds the same caches, yet vLLM's kv_tp_replicas is 1: its DeepSeek-V4.1
    specs set no max_tp_shards."""
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


def _vllm_cache_views(config, layer_names, kernel_block_size=None):
    """The caches of ``layer_names`` as vLLM's worker hands them over: a
    logical ``[B, H, N, C]`` view per layer into one allocation."""
    specs = {}
    for group in config.kv_cache_groups:
        specs.update(group.kv_cache_spec.kv_cache_specs
                     if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                     else dict.fromkeys(group.layer_names, group.kv_cache_spec))
    raw = torch.zeros(config.kv_cache_tensors[0].size, dtype=torch.int8)
    layout = VllmKVCacheLayout[config.kv_cache_layout]
    views = {}
    for tensor in config.kv_cache_tensors:
        if tensor.layers[0] in layer_names:
            views.update(zip(tensor.layers, create_kv_cache_views(
                raw, specs[tensor.layers[0]], config.num_blocks, layout, tensor,
                kernel_block_size)))
    return views


class _Registration:
    """Stands in for KVTPClient: keeps what the connector registers."""

    def register_to_server(self, **kwargs):
        self.kwargs = kwargs


def _register(flexkv_config, caches):
    connector = object.__new__(FlexKVWorkerConnector)
    connector.flexkv_config = flexkv_config
    connector.enable_layerwise_transfer = False
    connector.layerwise_counter_pool = None
    connector.tp_client = _Registration()
    connector._register_layer_groups(caches, flexkv_config.vllm_kv_group, resume=False)
    return connector.tp_client.kwargs


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


def test_dsv41_layouts_are_one_shared_head_per_page():
    config = _dsv41_config(scheduler=False)
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 8
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)
    group = flexkv_config.vllm_kv_group

    registration = _register(flexkv_config, _vllm_cache_views(config, group.cache_layer_names))

    assert [(layout.num_head, layout.head_size, layout.num_kv_heads, layout.kv_dim)
            for layout in registration["gpu_layouts"]] == [
        (1, 19008, 1, 1), (1, 4608, 1, 1), (1, 37440, 1, 1), (1, 8640, 1, 1)]


G = "model.layers"
WIDE_SPEC = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=128,
                              dtype=torch.bfloat16)
NARROW_SPEC = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64,
                                dtype=torch.bfloat16)
WIDE, NARROW = WIDE_SPEC.page_size_bytes, NARROW_SPEC.page_size_bytes
GQA_BLOCK = 2 * WIDE + 2 * NARROW


def _gqa_config(scheduler: bool, layout: str = "BLHNC") -> KVCacheConfig:
    """A GQA model's one KV cache group: four full-attention layers whose
    pages alternate between two sizes, at one TP rank's heads. TP splits the
    heads, so the ranks' pages differ."""
    specs = {f"{G}.{i}.self_attn.attn": NARROW_SPEC if i % 2 else WIDE_SPEC
             for i in range(4)}
    spec = (next(iter(specs.values())) if scheduler
            else UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=specs))
    wide_layers = [f"{G}.{i}.self_attn.attn" for i in (0, 2)]
    narrow_layers = [f"{G}.{i}.self_attn.attn" for i in (1, 3)]
    size = NUM_BLOCKS * GQA_BLOCK
    if layout == "BLHNC":
        tensors = [KVCacheTensor(size, wide_layers, WIDE, GQA_BLOCK),
                   KVCacheTensor(size, narrow_layers, NARROW, GQA_BLOCK, 2 * WIDE)]
    else:
        tensors = [KVCacheTensor(size, wide_layers, NUM_BLOCKS * WIDE, WIDE),
                   KVCacheTensor(size, narrow_layers, NUM_BLOCKS * NARROW, NARROW,
                                 2 * NUM_BLOCKS * WIDE)]
    config = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=tensors,
                           kv_cache_groups=[KVCacheGroupSpec(list(specs), spec)])
    config.kv_cache_layout = layout
    return config


@pytest.mark.parametrize("layout", ["BLHNC", "LBHNC"])
@pytest.mark.parametrize("scheduler", [True, False])
def test_gqa_group_with_mixed_pages_as_layer_groups(scheduler, layout):
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4
    flexkv_config._configure_kv_layout_from_vllm_cache_config(_gqa_config(scheduler, layout))

    group = flexkv_config.vllm_kv_group
    assert group.group_id == 0 and group.num_blocks == NUM_BLOCKS
    assert group.layer_names == tuple(f"{G}.{i}.self_attn.attn" for i in range(4))
    model_config = flexkv_config.model_config
    assert [(g.head_size, g.layer_indices, g.compress_ratio)
            for g in model_config.layer_groups] == [(WIDE, [0, 2], 16), (NARROW, [1, 3], 16)]
    assert model_config.num_kv_heads == 2 * 4
    assert model_config.host_rank_copies == 4
    assert block_size_in_bytes_for_cache(
        model_config, flexkv_config.cache_config) == 4 * GQA_BLOCK


@pytest.mark.parametrize("layout", ["BLHNC", "LBHNC"])
def test_gqa_layouts_give_each_rank_heads(layout):
    config = _gqa_config(scheduler=False, layout=layout)
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)

    caches = _vllm_cache_views(config, config.kv_cache_groups[0].layer_names)

    registration = _register(flexkv_config, caches)

    assert [(layout.num_head, layout.head_size, layout.num_kv_heads, layout.kv_dim)
            for layout in registration["gpu_layouts"]] == [
        (2, WIDE // 2, 2 * 4, 1), (1, NARROW, 1 * 4, 1)]
    main = registration["kv_layout"]
    assert (main.num_layer, main.num_head, main.num_kv_heads) == (4, 2, 2 * 4)


@pytest.mark.parametrize("scheduler", [True, False])
def test_group_mixing_mla_and_split_caches_keeps_a_copy_per_rank(scheduler):
    config = _gqa_config(scheduler)
    if not scheduler:
        group = config.kv_cache_groups[0]
        specs = dict(group.kv_cache_spec.kv_cache_specs)
        specs[f"{G}.1.self_attn.attn"] = MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
        config.kv_cache_groups[0] = KVCacheGroupSpec(
            group.layer_names, UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=specs))
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)

    assert not flexkv_config.vllm_kv_group.kv_shared_across_ranks
    assert flexkv_config.model_config.host_rank_copies == 4


def test_kernel_block_split_view_moves_whole_blocks():
    config = _gqa_config(scheduler=False, layout="LBHNC")
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)
    caches = _vllm_cache_views(config, config.kv_cache_groups[0].layer_names,
                               kernel_block_size=8)

    registration = _register(flexkv_config, caches)
    view = registration["handles_per_group"][0][0]
    (region,) = compile_gpu_regions(
        flexkv_config.model_config.layer_groups[:1], [registration["gpu_layouts"][:1]],
        16, 1, torch.uint8, tensors_per_group_device=[[view]])

    assert view.shape[0] == 2 * NUM_BLOCKS
    assert (region.chunk_bytes, region.block_stride) == (WIDE, WIDE)


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


def test_group_led_by_mla_with_split_caches_is_refused():
    config = _gqa_config(scheduler=False)
    group = config.kv_cache_groups[0]
    specs = dict(group.kv_cache_spec.kv_cache_specs)
    specs[f"{G}.0.self_attn.attn"] = MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    config.kv_cache_groups[0] = KVCacheGroupSpec(
        group.layer_names, UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=specs))
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4

    with pytest.raises(ValueError, match="first layer is MLA"):
        flexkv_config._configure_kv_layout_from_vllm_cache_config(config)


def test_caches_vllm_replicates_keep_one_copy():
    config = _gqa_config(scheduler=False)
    config.kv_tp_replicas = 4
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)

    model_config = flexkv_config.model_config
    assert flexkv_config.vllm_kv_group.kv_shared_across_ranks
    assert (model_config.num_kv_heads, model_config.host_rank_copies) == (1, 1)
