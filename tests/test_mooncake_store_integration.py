# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for FlexKV + mooncake-store integration:
  * MooncakeStoreClient: batch_put / batch_get / batch_exists round-trip
  * MooncakeStoreCacheEngine.match():
      - single-pool (KV-only) longest-prefix semantics
      - multi-pool (KV + indexer) joint-existence intersection
  * End-to-end: put-with-pattern -> match -> get -> verify data

Requires
--------
A running mooncake-store cluster reachable from the test host, plus a JSON
config file describing the local client. Set one of:

    export FLEXKV_MOONCAKE_STORE_CONFIG_PATH=/path/to/mooncake_store.json

The test module is skipped automatically when the SDK is missing or the
config env-var is not set, so it is safe to run in CI without a cluster.

Running
-------
    pytest tests/test_mooncake_store_integration.py -m mooncake -v

Run only mooncake tests (skip when not configured):
    pytest tests/test_mooncake_store_integration.py -v
"""
from __future__ import annotations

import os
import uuid
import pytest
import torch
import numpy as np

# Skip the entire module if the mooncake-store SDK is not installed.
pytest.importorskip("mooncake.store", reason="mooncake-store SDK not installed")

from flexkv.common.config import CacheConfig, IndexerCacheConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.block import SequenceMeta
from flexkv.external.mooncake_store_keys import PoolKind, build_key
from flexkv.common.debug import flexkv_logger

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _mooncake_configured() -> bool:
    return bool(os.environ.get("FLEXKV_MOONCAKE_STORE_CONFIG_PATH"))


def _config_path() -> str:
    return os.environ.get("FLEXKV_MOONCAKE_STORE_CONFIG_PATH", "")


def _make_cache_config(*, with_indexer: bool = False) -> CacheConfig:
    """Build a minimal CacheConfig that targets mooncake-store."""
    indexer = IndexerCacheConfig() if with_indexer else None
    return CacheConfig(
        tokens_per_block=16,
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=True,
        use_mooncake_store_backend=True,
        mooncake_store_config_path=_config_path(),
        num_cpu_blocks=64,
        num_remote_blocks=128,
        indexer=indexer,
    )


def _make_blockfirst_layout(num_blocks=16, num_layers=1, tokens_per_block=16):
    return KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=num_layers,
        num_block=num_blocks,
        tokens_per_block=tokens_per_block,
        num_head=2,
        head_size=64,
        is_mla=True,
    )


def _unique_key(prefix: str) -> str:
    """Avoid collisions between test runs that share a live cluster."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mooncake_client():
    """Module-scoped MooncakeStoreClient bound to the configured cluster."""
    if not _mooncake_configured():
        pytest.skip(
            "mooncake-store not configured: "
            "set FLEXKV_MOONCAKE_STORE_CONFIG_PATH to a JSON config file"
        )

    from flexkv.external.mooncake_store_utils import (
        MooncakeStoreConfig,
        MooncakeStoreClient,
    )

    cache_config = _make_cache_config()
    cfg = MooncakeStoreConfig.from_file(cache_config)
    # Read/write client (NOT query_only) so we can call batch_put/get.
    client = MooncakeStoreClient(cfg, query_only=False)
    flexkv_logger.info("client setup done")
    yield client
    # No explicit teardown: the SDK's underlying store has its own GC.


@pytest.fixture
def buffer_and_keys():
    """Allocate a small CPU tensor and register it as a Mooncake MR.

    Returns ``(buffer, layout, ptrs, sizes)`` for the caller to populate
    with KV-style data and then publish via ``batch_put`` etc.
    """
    layout = _make_blockfirst_layout(num_blocks=4)
    dtype = torch.bfloat16
    elements_per_block = layout.get_elements_per_block()
    block_size_bytes = elements_per_block * dtype.itemsize
    total_bytes = layout.num_block * block_size_bytes

    buffer = torch.zeros(total_bytes // dtype.itemsize, dtype=dtype)
    ptrs = [
        int(buffer.data_ptr() + i * block_size_bytes)
        for i in range(layout.num_block)
    ]
    sizes = [block_size_bytes] * layout.num_block
    return buffer, layout, ptrs, sizes


# ---------------------------------------------------------------------------
# Layer 1: low-level client tests (mirrors test_simm_client_query/transfer)
# ---------------------------------------------------------------------------

@pytest.mark.mooncake
def test_mooncake_client_query(mooncake_client, buffer_and_keys):
    """batch_put then batch_exists returns the expected longest-prefix length."""
    buffer, layout, ptrs, sizes = buffer_and_keys
    mooncake_client.register_buffer(buffer)
    print("register buffer done")
    # Use 3 of the 4 buffer slots as test keys.
    keys = [_unique_key(f"query_{i}") for i in range(3)]
    put_ok = mooncake_client.batch_put(
        key_strs=keys,
        buffer_ptrs=ptrs[: len(keys)],
        buffer_sizes=sizes[: len(keys)],
    )
    print("register buffer done")

    assert all(put_ok), f"batch_put failed: {put_ok}"

    # All three should exist -> prefix length == 3.
    n = mooncake_client.batch_exists(keys)
    assert n == len(keys), f"expected batch_exists={len(keys)}, got {n}"

    # Inserting a non-existent key in the middle must shrink the prefix.
    mixed = [keys[0], _unique_key("missing"), keys[2]]
    n2 = mooncake_client.batch_exists(mixed)
    assert n2 == 1, f"expected prefix=1 (only first exists), got {n2}"

    mooncake_client.unregister_buffer(buffer)


@pytest.mark.mooncake
def test_mooncake_client_transfer(mooncake_client, buffer_and_keys):
    """batch_put then batch_get must round-trip the original payload."""
    buffer, layout, ptrs, sizes = buffer_and_keys
    mooncake_client.register_buffer(buffer)

    elements_per_block = layout.get_elements_per_block()
    keys = [_unique_key(f"xfer_{i}") for i in range(layout.num_block)]

    # Seed each block with a distinguishable scalar.
    flat = buffer.view(-1)
    for i in range(layout.num_block):
        flat[i * elements_per_block : (i + 1) * elements_per_block] = float(i + 100)

    put_ok = mooncake_client.batch_put(
        key_strs=keys, buffer_ptrs=ptrs, buffer_sizes=sizes
    )
    assert all(put_ok), f"batch_put failed: {put_ok}"

    # Wipe and read back from the cluster.
    buffer.zero_()
    get_ok = mooncake_client.batch_get(
        key_strs=keys, buffer_ptrs=ptrs, buffer_sizes=sizes
    )
    assert all(get_ok), f"batch_get failed: {get_ok}"

    for i in range(layout.num_block):
        actual = flat[i * elements_per_block].item()
        expected = float(i + 100)
        assert actual == expected, f"block {i}: expected {expected}, got {actual}"

    mooncake_client.unregister_buffer(buffer)

# ---------------------------------------------------------------------------
# Layer 2: MooncakeStoreCacheEngine.match() — single-pool (KV-only)
# ---------------------------------------------------------------------------

@pytest.mark.mooncake
def test_match_kv_only_full_hit(mooncake_client, buffer_and_keys):
    """When all KV keys exist, match() returns num_blocks."""
    from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine

    buffer, layout, ptrs, sizes = buffer_and_keys
    mooncake_client.register_buffer(buffer)

    tokens_per_block = layout.tokens_per_block
    num_blocks = layout.num_block
    token_ids = np.random.randint(
        0, 1_000_000, size=num_blocks * tokens_per_block, dtype=np.int64
    )
    seq = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)
    assert seq.num_blocks == num_blocks

    # Publish KV blocks under the suffix the worker would actually write.
    kv_keys = [build_key(seq.block_hashes[i], PoolKind.KV) for i in range(num_blocks)]
    put_ok = mooncake_client.batch_put(
        key_strs=kv_keys, buffer_ptrs=ptrs, buffer_sizes=sizes
    )
    assert all(put_ok)

    cache_config = _make_cache_config(with_indexer=False)
    engine = MooncakeStoreCacheEngine(cache_config)

    result = engine.match(seq)
    assert result.matched_pos == MooncakeStoreCacheEngine.MATCHED_POS
    assert result.num_matched_blocks == num_blocks, (
        f"expected match={num_blocks}, got {result.num_matched_blocks}"
    )
    assert result.num_ready_matched_blocks == result.num_matched_blocks

    mooncake_client.unregister_buffer(buffer)

@pytest.mark.mooncake
def test_match_kv_only_partial_prefix(mooncake_client, buffer_and_keys):
    """If only the first K KV keys exist, match() returns exactly K."""
    from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine

    buffer, layout, ptrs, sizes = buffer_and_keys
    mooncake_client.register_buffer(buffer)

    tokens_per_block = layout.tokens_per_block
    num_blocks = layout.num_block  # 4
    token_ids = np.random.randint(
        0, 1_000_000, size=num_blocks * tokens_per_block, dtype=np.int64
    )
    seq = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)

    # Only publish the first 2 KV keys.
    publish = 2
    kv_keys_full = [build_key(seq.block_hashes[i], PoolKind.KV) for i in range(num_blocks)]
    put_ok = mooncake_client.batch_put(
        key_strs=kv_keys_full[:publish],
        buffer_ptrs=ptrs[:publish],
        buffer_sizes=sizes[:publish],
    )
    assert all(put_ok)

    cache_config = _make_cache_config(with_indexer=False)
    engine = MooncakeStoreCacheEngine(cache_config)

    result = engine.match(seq)
    assert result.num_matched_blocks == publish, (
        f"expected partial prefix={publish}, got {result.num_matched_blocks}"
    )

    mooncake_client.unregister_buffer(buffer)

# ---------------------------------------------------------------------------
# Layer 3: MooncakeStoreCacheEngine.match() — multi-pool (KV + indexer)
# ---------------------------------------------------------------------------

@pytest.mark.mooncake
def test_match_kv_indexer_joint_full_hit(mooncake_client, buffer_and_keys):
    """When KV and indexer are BOTH published, match() returns num_blocks."""
    from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine

    buffer, layout, ptrs, sizes = buffer_and_keys
    mooncake_client.register_buffer(buffer)

    tokens_per_block = layout.tokens_per_block
    num_blocks = layout.num_block
    token_ids = np.random.randint(
        0, 1_000_000, size=num_blocks * tokens_per_block, dtype=np.int64
    )
    seq = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)

    kv_keys = [build_key(seq.block_hashes[i], PoolKind.KV) for i in range(num_blocks)]
    idx_keys = [build_key(seq.block_hashes[i], PoolKind.INDEXER) for i in range(num_blocks)]

    # We re-use the same buffer slots for both pools; the test only cares
    # about key-existence, not block content.
    assert all(mooncake_client.batch_put(kv_keys, ptrs, sizes))
    assert all(mooncake_client.batch_put(idx_keys, ptrs, sizes))

    cache_config = _make_cache_config(with_indexer=True)
    engine = MooncakeStoreCacheEngine(cache_config)
    assert len(engine.hit_pool_specs) == 2  # KV + indexer

    result = engine.match(seq)
    assert result.num_matched_blocks == num_blocks, (
        f"joint hit should return {num_blocks}, got {result.num_matched_blocks}"
    )

    mooncake_client.unregister_buffer(buffer)

@pytest.mark.mooncake
def test_match_kv_indexer_joint_indexer_missing(mooncake_client, buffer_and_keys):
    """KV-hit but indexer-miss must NOT count as a hit (prefix truncates)."""
    from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine

    buffer, layout, ptrs, sizes = buffer_and_keys
    mooncake_client.register_buffer(buffer)

    tokens_per_block = layout.tokens_per_block
    num_blocks = layout.num_block  # 4
    token_ids = np.random.randint(
        0, 1_000_000, size=num_blocks * tokens_per_block, dtype=np.int64
    )
    seq = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)

    kv_keys = [build_key(seq.block_hashes[i], PoolKind.KV) for i in range(num_blocks)]
    idx_keys = [build_key(seq.block_hashes[i], PoolKind.INDEXER) for i in range(num_blocks)]

    # Publish ALL KV keys, but only the FIRST 2 indexer keys.
    assert all(mooncake_client.batch_put(kv_keys, ptrs, sizes))
    publish_idx = 2
    assert all(
        mooncake_client.batch_put(
            idx_keys[:publish_idx], ptrs[:publish_idx], sizes[:publish_idx]
        )
    )

    cache_config = _make_cache_config(with_indexer=True)
    engine = MooncakeStoreCacheEngine(cache_config)

    result = engine.match(seq)
    # Joint-AND prefix must stop at the first indexer miss.
    assert result.num_matched_blocks == publish_idx, (
        f"joint match must truncate at first indexer miss; "
        f"expected {publish_idx}, got {result.num_matched_blocks}"
    )

    mooncake_client.unregister_buffer(buffer)

@pytest.mark.mooncake
def test_match_kv_indexer_joint_kv_missing(mooncake_client, buffer_and_keys):
    """Symmetric: KV-miss but indexer-hit also truncates."""
    from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine

    buffer, layout, ptrs, sizes = buffer_and_keys
    mooncake_client.register_buffer(buffer)

    tokens_per_block = layout.tokens_per_block
    num_blocks = layout.num_block
    token_ids = np.random.randint(
        0, 1_000_000, size=num_blocks * tokens_per_block, dtype=np.int64
    )
    seq = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)

    kv_keys = [build_key(seq.block_hashes[i], PoolKind.KV) for i in range(num_blocks)]
    idx_keys = [build_key(seq.block_hashes[i], PoolKind.INDEXER) for i in range(num_blocks)]

    # Publish only the first KV key, but ALL indexer keys.
    publish_kv = 1
    assert all(
        mooncake_client.batch_put(
            kv_keys[:publish_kv], ptrs[:publish_kv], sizes[:publish_kv]
        )
    )
    assert all(mooncake_client.batch_put(idx_keys, ptrs, sizes))

    cache_config = _make_cache_config(with_indexer=True)
    engine = MooncakeStoreCacheEngine(cache_config)

    result = engine.match(seq)
    assert result.num_matched_blocks == publish_kv, (
        f"joint match must truncate at first KV miss; "
        f"expected {publish_kv}, got {result.num_matched_blocks}"
    )

    mooncake_client.unregister_buffer(buffer)

# ---------------------------------------------------------------------------
# Layer 4: end-to-end (put pattern -> match -> get -> verify content)
# ---------------------------------------------------------------------------

@pytest.mark.mooncake
def test_mooncake_e2e_kv_only(mooncake_client, buffer_and_keys):
    """End-to-end with KV-only pool: write recognisable pattern, match,
    fetch back, verify byte-for-byte equality."""
    from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine

    buffer, layout, ptrs, sizes = buffer_and_keys
    mooncake_client.register_buffer(buffer)

    tokens_per_block = layout.tokens_per_block
    num_blocks = layout.num_block
    elements_per_block = layout.get_elements_per_block()

    token_ids = np.random.randint(
        0, 1_000_000, size=num_blocks * tokens_per_block, dtype=np.int64
    )
    seq = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)

    flat = buffer.view(-1)
    for i in range(num_blocks):
        flat[i * elements_per_block : (i + 1) * elements_per_block] = float(200 + i)

    kv_keys = [build_key(seq.block_hashes[i], PoolKind.KV) for i in range(num_blocks)]
    assert all(mooncake_client.batch_put(kv_keys, ptrs, sizes))

    cache_config = _make_cache_config(with_indexer=False)
    engine = MooncakeStoreCacheEngine(cache_config)

    result = engine.match(seq)
    assert result.num_matched_blocks == num_blocks, "all blocks must be matchable"

    # Wipe and pull data back through batch_get.
    buffer.zero_()
    assert all(mooncake_client.batch_get(kv_keys, ptrs, sizes))
    for i in range(num_blocks):
        assert flat[i * elements_per_block].item() == float(200 + i), (
            f"block {i} content mismatch after round-trip"
        )

    mooncake_client.unregister_buffer(buffer)

# ---------------------------------------------------------------------------
# Layer 5: PoolSpec/CacheConfig wiring sanity (no cluster needed)
# ---------------------------------------------------------------------------

def test_enable_pool_specs_kv_only_when_indexer_none():
    """Without indexer, only the KV pool is active."""
    cfg = CacheConfig(
        tokens_per_block=16,
        enable_remote=True,
        use_mooncake_store_backend=True,
        mooncake_store_config_path="/tmp/dummy.json",  # not loaded here
        indexer=None,
    )
    specs = cfg.enable_pool_specs()
    kinds = [s.kind for s in specs]
    assert kinds == [PoolKind.KV]


def test_enable_pool_specs_includes_indexer_when_configured():
    """Configuring an IndexerCacheConfig must add the INDEXER pool."""
    cfg = CacheConfig(
        tokens_per_block=16,
        enable_remote=True,
        use_mooncake_store_backend=True,
        mooncake_store_config_path="/tmp/dummy.json",
        indexer=IndexerCacheConfig(),
    )
    specs = cfg.enable_pool_specs()
    kinds = [s.kind for s in specs]
    assert kinds == [PoolKind.KV, PoolKind.INDEXER]
    assert all(s.required_for_hit for s in specs), (
        "all current pools must participate in joint-hit"
    )


def test_build_key_format_matches_worker_contract():
    """Centralised key builder must produce '<hash>_<suffix>' literally."""
    assert build_key(123, PoolKind.KV) == "123_FlexKV"
    assert build_key("abc", PoolKind.INDEXER) == "abc_FlexKV_indexer"
