from __future__ import annotations

import torch

from flexkv.common.config import (
    CacheConfig,
    LayerGroupSpec,
    ModelConfig,
    RankInfo,
    UserConfig,
    recompute_cache_block_counts,
    update_default_config_from_user_config,
)


def test_recompute_shrinks_cpu_blocks_for_heterogeneous_layer_groups() -> None:
    """Uniform init over-estimates block count; layer_groups correct it."""
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
    assert uniform_blocks > 9000

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

    assert recompute_cache_block_counts(model_config, cache_config) is True
    assert cache_config.num_cpu_blocks < uniform_blocks
    assert 6500 < cache_config.num_cpu_blocks < 7000
