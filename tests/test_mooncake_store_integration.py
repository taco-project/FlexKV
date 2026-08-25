# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for FlexKV + mooncake-store integration:
  * MooncakeStoreClient: batch_put / batch_get / batch_exists round-trip
  * MooncakeStoreCacheEngine.match():
      - single-pool (KV-only) longest-prefix semantics
      - joint-hit with the SWA pool (KV prefix + largest SWA snapshot)
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

from flexkv.common.config import CacheConfig, SWAPoolConfig
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


def _make_cache_config(swa: bool = False) -> CacheConfig:
    """Build a minimal CacheConfig that targets mooncake-store.

    When ``swa=True`` the SWA remote pool is enabled, arming
    ``MooncakeStoreCacheEngine`` joint-hit matching.
    """
    cfg = CacheConfig(
        tokens_per_block=16,
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=True,
        use_mooncake_store_backend=True,
        mooncake_store_config_path=_config_path(),
        num_cpu_blocks=64,
        num_remote_blocks=128,
    )
    if swa:
        cfg.swa = SWAPoolConfig(enabled=True)
    return cfg


def _make_blockfirst_layout(num_blocks=16, num_layers=1, tokens_per_block=16):
    return KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=num_layers,
        num_block=num_blocks,
        tokens_per_block=tokens_per_block,
        num_head=2,
        head_size=64,
        kv_dim=1,
        num_kv_heads=1,
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

    cache_config = _make_cache_config()
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

    cache_config = _make_cache_config()
    engine = MooncakeStoreCacheEngine(cache_config)

    result = engine.match(seq)
    assert result.num_matched_blocks == publish, (
        f"expected partial prefix={publish}, got {result.num_matched_blocks}"
    )

    mooncake_client.unregister_buffer(buffer)

# ---------------------------------------------------------------------------
# Layer 2b: MooncakeStoreCacheEngine.match() — KV + SWA joint hit
# ---------------------------------------------------------------------------
#
# Semantics (_joint_match_length): joint_matched is the LARGEST L in
# [1, kv_matched] such that SWA(hash[L-1]) exists — a right-to-left scan,
# NOT an intersection prefix. SWA snapshots accumulate historically, so a
# hit at an inner position is valid as long as the full-KV prefix up to it
# is present. With SWA enabled but no snapshot, joint hit is 0 even when KV
# is fully present.

@pytest.mark.mooncake
def test_match_swa_joint_full_hit(mooncake_client, buffer_and_keys):
    """KV + SWA both present at every block -> joint hit == num_blocks."""
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
    swa_keys = [build_key(seq.block_hashes[i], PoolKind.SWA) for i in range(num_blocks)]
    assert all(mooncake_client.batch_put(kv_keys, ptrs, sizes))
    assert all(mooncake_client.batch_put(swa_keys, ptrs, sizes))

    engine = MooncakeStoreCacheEngine(_make_cache_config(swa=True))
    assert engine.swa_enabled

    result = engine.match(seq)
    assert result.kv_matched_blocks == num_blocks
    assert result.num_matched_blocks == num_blocks
    assert result.swa_hit_blocks == num_blocks

    mooncake_client.unregister_buffer(buffer)


@pytest.mark.mooncake
def test_match_swa_joint_swa_at_inner_position(mooncake_client, buffer_and_keys):
    """SWA only at an inner block -> joint hit is that position (largest L).

    kv_matched == 4 (all KV present); SWA published only at block index 2,
    so the largest L in [1..4] with SWA(hash[L-1]) present is L == 3.
    """
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
    assert all(mooncake_client.batch_put(kv_keys, ptrs, sizes))

    swa_pos = 2  # 0-indexed; corresponds to L == 3
    swa_keys = [build_key(seq.block_hashes[swa_pos], PoolKind.SWA)]
    assert all(mooncake_client.batch_put(
        swa_keys, [ptrs[swa_pos]], [sizes[swa_pos]]))

    engine = MooncakeStoreCacheEngine(_make_cache_config(swa=True))

    result = engine.match(seq)
    assert result.kv_matched_blocks == num_blocks
    assert result.num_matched_blocks == swa_pos + 1
    assert result.swa_hit_blocks == swa_pos + 1

    mooncake_client.unregister_buffer(buffer)


@pytest.mark.mooncake
def test_match_swa_joint_swa_missing(mooncake_client, buffer_and_keys):
    """KV present everywhere but no SWA snapshot -> joint hit == 0."""
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
    assert all(mooncake_client.batch_put(kv_keys, ptrs, sizes))
    # Publish no SWA keys.

    engine = MooncakeStoreCacheEngine(_make_cache_config(swa=True))

    result = engine.match(seq)
    assert result.kv_matched_blocks == num_blocks
    assert result.num_matched_blocks == 0
    assert result.swa_hit_blocks == 0

    mooncake_client.unregister_buffer(buffer)


@pytest.mark.mooncake
def test_match_swa_joint_kv_prefix_caps_joint(mooncake_client, buffer_and_keys):
    """KV prefix shorter than a later SWA snapshot -> joint capped by KV prefix.

    KV published for the first 2 blocks (kv_matched == 2). SWA published at
    block index 1 (inside the prefix) and index 3 (beyond it). The joint
    scan only covers [1..2], so the unreachable SWA at index 3 is ignored
    and joint hit == 2 (SWA at index 1 -> L == 2).
    """
    from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine

    buffer, layout, ptrs, sizes = buffer_and_keys
    mooncake_client.register_buffer(buffer)

    tokens_per_block = layout.tokens_per_block
    num_blocks = layout.num_block  # 4
    token_ids = np.random.randint(
        0, 1_000_000, size=num_blocks * tokens_per_block, dtype=np.int64
    )
    seq = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)

    publish_kv = 2
    kv_keys = [build_key(seq.block_hashes[i], PoolKind.KV) for i in range(num_blocks)]
    assert all(mooncake_client.batch_put(
        kv_keys[:publish_kv], ptrs[:publish_kv], sizes[:publish_kv]))

    swa_keys = [
        build_key(seq.block_hashes[1], PoolKind.SWA),
        build_key(seq.block_hashes[3], PoolKind.SWA),
    ]
    assert all(mooncake_client.batch_put(
        swa_keys, [ptrs[1], ptrs[3]], [sizes[1], sizes[3]]))

    engine = MooncakeStoreCacheEngine(_make_cache_config(swa=True))

    result = engine.match(seq)
    assert result.kv_matched_blocks == publish_kv
    assert result.num_matched_blocks == 2  # SWA at index 1 -> L == 2
    assert result.swa_hit_blocks == 2

    mooncake_client.unregister_buffer(buffer)

# ---------------------------------------------------------------------------
# Layer 3: end-to-end (put pattern -> match -> get -> verify content)
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

    cache_config = _make_cache_config()
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


def test_build_key_format_matches_worker_contract():
    """Centralised key builder must produce '<hash>_<suffix>' literally."""
    assert build_key(123, PoolKind.KV) == "123_FlexKV"
    assert build_key("abc", PoolKind.SWA) == "abc_FlexKV_swa"
