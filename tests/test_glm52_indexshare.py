"""GLM-5.2 IndexShare detection tests for FlexKV.

GLM-5.2 (arxiv 2603.12201) extends DeepSeek-V3.2 DSA with IndexShare: a
subset of layers ("Full") run the lightning indexer to compute top-k, and
the rest ("Shared") reuse the previous Full layer's top-k. FlexKV parses
the same ``index_topk_pattern`` / ``index_topk_freq`` fields sglang uses to
populate ``IndexerCacheConfig.full_layer_indices``.

These tests do not require GPUs, running-model weights, or torch.distributed
— they exercise the pure-Python auto-detection path.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from flexkv.common.config import IndexerCacheConfig
from flexkv.integration.config import (
    FlexKVConfig,
    _derive_full_layer_indices,
)


def _make_hf_cfg(**kwargs) -> SimpleNamespace:
    """Build a stub HF config for the DSA branch of the detector."""
    defaults = dict(
        qk_rope_head_dim=64,
        index_head_dim=128,
        num_hidden_layers=64,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _derive_full_layer_indices — pure function
# ---------------------------------------------------------------------------


def test_derive_returns_none_when_no_indexshare_fields() -> None:
    hf = _make_hf_cfg()
    assert _derive_full_layer_indices(hf, hf.num_hidden_layers) is None


def test_derive_returns_none_when_freq_is_one() -> None:
    hf = _make_hf_cfg(index_topk_freq=1)
    assert _derive_full_layer_indices(hf, hf.num_hidden_layers) is None


def test_derive_from_freq_matches_sglang_semantics() -> None:
    # sglang: skips_topk = max(layer_id - 1, 0) % freq != 0
    # → Full when max(layer_id - 1, 0) % freq == 0
    # With freq=4 and 12 layers, Full = {0, 1, 5, 9}.
    hf = _make_hf_cfg(index_topk_freq=4, num_hidden_layers=12)
    full = _derive_full_layer_indices(hf, hf.num_hidden_layers)
    assert full == [0, 1, 5, 9]


def test_derive_from_freq_with_offset() -> None:
    # With offset=1, Full when max(layer_id, 0) % freq == 0.
    hf = _make_hf_cfg(
        index_topk_freq=4,
        index_skip_topk_offset=1,
        num_hidden_layers=12,
    )
    full = _derive_full_layer_indices(hf, hf.num_hidden_layers)
    assert full == [0, 4, 8]


def test_derive_from_pattern() -> None:
    # 'F' = Full, 'S' = Shared. Any non-'S' char is Full.
    hf = _make_hf_cfg(index_topk_pattern="FSSSFSSSFSSS", num_hidden_layers=12)
    full = _derive_full_layer_indices(hf, hf.num_hidden_layers)
    assert full == [0, 4, 8]


def test_derive_pattern_extends_beyond_pattern_length() -> None:
    # Layers past the pattern default to Full (skips_topk returns False).
    hf = _make_hf_cfg(index_topk_pattern="FSSS", num_hidden_layers=6)
    full = _derive_full_layer_indices(hf, hf.num_hidden_layers)
    assert full == [0, 4, 5]


def test_derive_pattern_all_full_returns_none() -> None:
    # A pattern with no 'S' means everything is Full — same as no IndexShare.
    hf = _make_hf_cfg(index_topk_pattern="FFFF", num_hidden_layers=4)
    assert _derive_full_layer_indices(hf, hf.num_hidden_layers) is None


def test_derive_prefers_indexer_types_list() -> None:
    # GLM-5.2 exposes indexer_types directly; it wins over freq/offset.
    hf = _make_hf_cfg(
        indexer_types=["full", "shared", "shared", "shared", "full", "shared"],
        # Deliberately mismatched freq to prove the list takes priority.
        index_topk_freq=3,
        num_hidden_layers=6,
    )
    assert _derive_full_layer_indices(hf, hf.num_hidden_layers) == [0, 4]


def test_derive_indexer_types_extends_beyond_list_length() -> None:
    # Trailing MTP/nextn layers past the list default to Full.
    hf = _make_hf_cfg(
        indexer_types=["full", "shared", "shared"],
        num_hidden_layers=5,
    )
    assert _derive_full_layer_indices(hf, hf.num_hidden_layers) == [0, 3, 4]


def test_derive_glm52_freq_and_offset_matches_paper_ratio() -> None:
    # Real GLM-5.2 (verified against the zai-org/GLM-5.2-FP8 config.json):
    # 78 layers, freq=4, offset=3 -> 21 Full layers (~27%, matches the
    # ~25% Full ratio in arxiv 2603.12201).
    hf = _make_hf_cfg(
        index_topk_freq=4,
        index_skip_topk_offset=3,
        num_hidden_layers=78,
    )
    full = _derive_full_layer_indices(hf, hf.num_hidden_layers)
    assert full is not None
    assert len(full) == 21
    assert full[:6] == [0, 1, 2, 6, 10, 14]
    assert full[-3:] == [66, 70, 74]


# ---------------------------------------------------------------------------
# FlexKVConfig._detect_indexer_config_from_hf
# ---------------------------------------------------------------------------


def test_detect_indexer_populates_full_layer_indices() -> None:
    cfg = FlexKVConfig()
    cfg.cache_config.tokens_per_block = 64
    hf = _make_hf_cfg(
        index_topk_freq=4,
        num_hidden_layers=12,
    )
    cfg._detect_indexer_config_from_hf(hf, source="unit")

    indexer = cfg.cache_config.indexer
    assert isinstance(indexer, IndexerCacheConfig)
    assert indexer.head_size > 0
    assert indexer.num_kv_heads == 1
    assert indexer.dtype == torch.uint8
    assert indexer.full_layer_indices == [0, 1, 5, 9]


def test_detect_indexer_defaults_to_none_full_layers() -> None:
    """Vanilla DSA (no IndexShare fields) leaves full_layer_indices None."""
    cfg = FlexKVConfig()
    cfg.cache_config.tokens_per_block = 64
    cfg._detect_indexer_config_from_hf(_make_hf_cfg(), source="unit")

    indexer = cfg.cache_config.indexer
    assert indexer is not None
    assert indexer.full_layer_indices is None


def test_detect_indexer_skips_when_not_dsa() -> None:
    """Non-DSA models (no qk_rope_head_dim) leave cache_config.indexer None."""
    cfg = FlexKVConfig()
    cfg.cache_config.tokens_per_block = 64
    hf = SimpleNamespace(num_hidden_layers=32)  # no qk_rope_head_dim
    cfg._detect_indexer_config_from_hf(hf, source="unit")
    assert cfg.cache_config.indexer is None


# ---------------------------------------------------------------------------
# IndexerCacheConfig.is_full_layer
# ---------------------------------------------------------------------------


def test_is_full_layer_defaults_to_true_when_unset() -> None:
    ic = IndexerCacheConfig(head_size=64)
    assert ic.is_full_layer(0)
    assert ic.is_full_layer(37)


def test_is_full_layer_respects_full_layer_indices() -> None:
    ic = IndexerCacheConfig(head_size=64, full_layer_indices=[0, 4, 8])
    assert ic.is_full_layer(0)
    assert not ic.is_full_layer(1)
    assert ic.is_full_layer(4)
    assert not ic.is_full_layer(7)
    assert ic.is_full_layer(8)
