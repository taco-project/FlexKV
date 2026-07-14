from __future__ import annotations

import torch

from flexkv.common.config import (
    CacheConfig,
    LayerGroupSpec,
    ModelConfig,
    RankInfo,
    UserConfig,
    convert_to_block_num,
    recompute_cache_block_counts,
    update_default_config_from_user_config,
)
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType


def test_recompute_matches_heterogeneous_layout_block_size() -> None:
    """Deferred sizing must use the same bytes per block as KVCacheLayout."""
    model_config = ModelConfig(
        num_layers=62,
        num_kv_heads=8,
        head_size=128,
        use_mla=True,
        tp_size=8,
        dtype=torch.bfloat16,
    )
    cache_config = CacheConfig(tokens_per_block=256)
    user_config = UserConfig(cpu_cache_gb=100, ssd_cache_gb=0)

    rank_info = RankInfo(model_config=model_config)
    update_default_config_from_user_config(rank_info, cache_config, user_config)
    uniform_blocks = cache_config.num_cpu_blocks

    model_config.layer_groups = [
        LayerGroupSpec(
            num_layers=21,
            num_kv_heads=1,
            head_size=585,
            layer_indices=list(range(21)),
            compress_ratio=4,
            dtype=torch.uint8,
        ),
        LayerGroupSpec(
            num_layers=20,
            num_kv_heads=1,
            head_size=9,
            layer_indices=list(range(21, 41)),
            compress_ratio=128,
            dtype=torch.uint8,
        ),
        LayerGroupSpec(
            num_layers=21,
            num_kv_heads=1,
            head_size=44,
            layer_indices=list(range(21)),
            compress_ratio=4,
            dtype=torch.uint8,
        ),
    ]

    layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=model_config.num_layers,
        num_block=1,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
        layer_groups=model_config.layer_groups,
        tp_size=model_config.tp_size,
    )
    expected_blocks = convert_to_block_num(
        user_config.cpu_cache_gb, layout.kv_shape[1])

    assert recompute_cache_block_counts(model_config, cache_config) is True
    assert cache_config.num_cpu_blocks == expected_blocks
    assert cache_config.num_cpu_blocks != uniform_blocks
