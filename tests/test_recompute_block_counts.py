from __future__ import annotations

import pytest
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
from flexkv.common.storage import (
    KVCacheLayout, KVCacheLayoutType, align_host_block_bytes)

pytestmark = pytest.mark.unit


def test_recompute_matches_heterogeneous_layout_block_size() -> None:
    """Deferred sizing must use the same bytes per block as KVCacheLayout."""
    model_config = ModelConfig(
        num_layers=62,
        num_kv_heads=8,
        head_size=128,
        kv_dim=1,
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
        kv_dim=model_config.kv_dim,
        layer_groups=model_config.layer_groups,
        tp_size=model_config.tp_size,
    )
    expected_blocks = convert_to_block_num(
        user_config.cpu_cache_gb, layout.kv_shape[1])

    assert recompute_cache_block_counts(model_config, cache_config) is True
    assert cache_config.num_cpu_blocks == expected_blocks
    assert cache_config.num_cpu_blocks != uniform_blocks


def test_compact_indexer_reclaims_capacity_at_fixed_bytes() -> None:
    num_layers = 8
    active_indexer_layers = [0, 3, 7]
    model_config = ModelConfig(
        num_layers=num_layers,
        num_kv_heads=1,
        head_size=576,
        kv_dim=1,
        tp_size=4,
        dtype=torch.bfloat16,
    )
    cache_config = CacheConfig(tokens_per_block=64)
    user_config = UserConfig(
        # Synthetic fixed byte budget.
        cpu_cache_gb=1,
        ssd_cache_gb=0,
    )
    rank_info = RankInfo(model_config=model_config)
    update_default_config_from_user_config(rank_info, cache_config, user_config)
    model_config.layer_groups = [
        LayerGroupSpec(
            num_layers=num_layers,
            num_kv_heads=1,
            head_size=576,
            layer_indices=list(range(num_layers)),
            compress_ratio=1,
            dtype=torch.bfloat16,
        ),
        LayerGroupSpec(
            num_layers=len(active_indexer_layers),
            num_kv_heads=1,
            head_size=8448,
            layer_indices=active_indexer_layers,
            compress_ratio=64,
            dtype=torch.uint8,
        ),
    ]

    assert recompute_cache_block_counts(model_config, cache_config) is True
    # Rank-shared KV (num_kv_heads=1): one copy per block whatever tp_size is.
    compact_bytes = align_host_block_bytes(8 * 64 * 576 * 2 + 3 * 8448)
    aliased_bytes = align_host_block_bytes(8 * 64 * 576 * 2 + 8 * 8448)
    assert cache_config.num_cpu_blocks == (1 << 30) // compact_bytes
    assert cache_config.num_cpu_blocks > (1 << 30) // aliased_bytes


@pytest.mark.parametrize("num_kv_heads, copies", [(1, 1), (8, 4)])
def test_multi_group_block_holds_one_copy_of_rank_shared_kv(
        num_kv_heads: int, copies: int) -> None:
    """MLA-style KV is identical on every TP rank and is written once.

    Sizing the block for tp_size copies left (tp_size - 1)/tp_size of every
    CPU/SSD block unused and moved that much extra on every CPU<->SSD op.
    Head-sharded KV still needs one slice per rank.
    """
    group = LayerGroupSpec(
        num_layers=4,
        num_kv_heads=1 if num_kv_heads == 1 else num_kv_heads // 4,
        head_size=512,
        layer_indices=list(range(4)),
        compress_ratio=1,
        dtype=torch.bfloat16,
    )
    model_config = ModelConfig(
        num_layers=4,
        num_kv_heads=num_kv_heads,
        head_size=512,
        kv_dim=1,
        tp_size=4,
        dtype=torch.bfloat16,
        layer_groups=[group],
    )
    cache_config = CacheConfig(tokens_per_block=64)
    one_copy = 4 * 64 * group.num_kv_heads * 512 * 2

    assert model_config.host_rank_copies == copies
    layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=4,
        num_block=1,
        tokens_per_block=64,
        num_head=group.num_kv_heads,
        head_size=512,
        kv_dim=1,
        num_kv_heads=num_kv_heads,
        layer_groups=[group],
        tp_size=model_config.host_rank_copies,
    )
    assert layout.get_block_stride() == copies * one_copy
    assert model_config.token_size_in_bytes * 64 == copies * one_copy


def test_multi_group_block_is_padded_for_direct_io(monkeypatch) -> None:
    """With FLEXKV_HOST_BLOCK_ALIGN=4096 a DSv4.1 FULL block (417,920 B) is
    padded so O_DIRECT can engage."""
    import flexkv.common.storage as storage
    assert align_host_block_bytes(417920) == 417920
    monkeypatch.setattr(storage, "HOST_BLOCK_ALIGN", 4096)
    assert align_host_block_bytes(417920) % 4096 == 0
    assert 0 <= align_host_block_bytes(417920) - 417920 < 4096
    assert align_host_block_bytes(8192) == 8192
