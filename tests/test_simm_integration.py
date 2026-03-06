# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for FlexKV + SIMM integration: query (batch_exists), transfer (batch_set_v1 / batch_get_v1),
and SimmCacheEngine match.

Requires: running SIMM manager; set FLEXKV_SIMM_CONFIG_PATH (path to JSON with manager_address)
  or FLEXKV_SIMM_MANAGER_ADDRESS (e.g. "ip:port"). Skip if SIMM is not configured or simm not installed.

To run the tests:

export FLEXKV_SIMM_MANAGER_ADDRESS="127.0.0.1:9400"

export FLEXKV_SIMM_CONFIG_PATH="/path/to/simm_config.json"

pytest tests/test_simm_integration.py -m simm -v

Run only SIMM tests:
  pytest tests/test_simm_integration.py -m simm -v

Run all tests (SIMM tests skip when not configured):
  pytest tests/test_simm_integration.py -v
"""
import os
import pytest
import torch
import numpy as np

# Skip entire module if simm is not installed
pytest.importorskip("simm.kv", reason="simm not installed")

from flexkv.common.config import CacheConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.block import SequenceMeta


def _simm_configured() -> bool:
    return bool(
        os.environ.get("FLEXKV_SIMM_CONFIG_PATH")
        or os.environ.get("FLEXKV_SIMM_MANAGER_ADDRESS")
    )


def _get_manager_address() -> str:
    addr = os.environ.get("FLEXKV_SIMM_MANAGER_ADDRESS", "")
    if addr:
        return addr
    try:
        import json
        path = os.environ.get("FLEXKV_SIMM_CONFIG_PATH")
        if path and os.path.isfile(path):
            with open(path) as f:
                cfg = json.load(f)
            return cfg.get("manager_address", "")
    except Exception:
        pass
    return ""


@pytest.fixture(scope="module")
def simm_client():
    """Create SiMMClient; skip if SIMM not configured."""
    if not _simm_configured():
        pytest.skip("SIMM not configured: set FLEXKV_SIMM_CONFIG_PATH or FLEXKV_SIMM_MANAGER_ADDRESS")
    from flexkv.external.simm_utils import SiMMClient
    manager_address = _get_manager_address()
    if not manager_address:
        pytest.skip("SIMM manager_address not found in config or env")
    client = SiMMClient(manager_address=manager_address or None, extra_config=None)
    yield client


def _make_blockfirst_layout(num_blocks=4, num_layers=1, tokens_per_block=16):
    return KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=num_layers,
        num_block=num_blocks,
        tokens_per_block=tokens_per_block,
        num_head=2,
        head_size=64,
        is_mla=False,
    )


@pytest.mark.simm
def test_simm_client_query(simm_client):
    """Test SIMM query: batch_set_v1 then batch_exists returns correct count."""
    layout = _make_blockfirst_layout(num_blocks=3)
    dtype = torch.float16
    elements_per_block = layout.get_elements_per_block()
    block_size_bytes = elements_per_block * dtype.itemsize
    total_bytes = layout.num_block * block_size_bytes

    buffer = torch.zeros(total_bytes // dtype.itemsize, dtype=dtype)
    simm_client.register_mr(buffer.data_ptr(), total_bytes)

    keys = ["test_query_key_1", "test_query_key_2", "test_query_key_3"]
    ptrs = [
        int(buffer.data_ptr() + i * block_size_bytes)
        for i in range(len(keys))
    ]
    sizes = [block_size_bytes] * len(keys)

    # Put all keys
    set_ok = simm_client.batch_set_v1(cpu_ptrs=ptrs, block_sizes_list=sizes, keys_strs=keys)
    assert all(set_ok), f"batch_set_v1 failed: {set_ok}"

    # Query: all should exist
    n = simm_client.batch_exists(keys_strs=keys)
    assert n == len(keys), f"expected batch_exists to return {len(keys)}, got {n}"

    # Query with one non-existent key
    n2 = simm_client.batch_exists(keys_strs=[keys[0], "nonexistent_key_xyz", keys[2]])
    assert n2 == 1, f"expected 1 (first key exists), got {n2}"


@pytest.mark.simm
def test_simm_client_transfer(simm_client):
    """Test SIMM transfer: batch_set_v1 then batch_get_v1, verify data round-trip."""
    layout = _make_blockfirst_layout(num_blocks=2)
    dtype = torch.float16
    elements_per_block = layout.get_elements_per_block()
    block_size_bytes = elements_per_block * dtype.itemsize
    total_bytes = layout.num_block * block_size_bytes

    buffer = torch.zeros(total_bytes // dtype.itemsize, dtype=dtype)
    simm_client.register_mr(buffer.data_ptr(), total_bytes)

    keys = ["test_transfer_key_1", "test_transfer_key_2"]
    ptrs = [
        int(buffer.data_ptr() + i * block_size_bytes)
        for i in range(len(keys))
    ]
    sizes = [block_size_bytes] * len(keys)

    # Fill with recognizable pattern (per-block)
    for i in range(layout.num_block):
        base = i * elements_per_block
        buffer.view(-1)[base : base + elements_per_block] = float(i + 100)

    set_ok = simm_client.batch_set_v1(cpu_ptrs=ptrs, block_sizes_list=sizes, keys_strs=keys)
    assert all(set_ok), f"batch_set_v1 failed: {set_ok}"

    # Zero buffer and get back from SIMM
    buffer.zero_()
    get_ok = simm_client.batch_get_v1(cpu_ptrs=ptrs, block_sizes=sizes, keys=keys)
    assert all(get_ok), f"batch_get_v1 failed: {get_ok}"

    for i in range(layout.num_block):
        base = i * elements_per_block
        expected = float(i + 100)
        actual = buffer.view(-1)[base].item()
        assert actual == expected, f"block {i}: expected {expected}, got {actual}"


@pytest.mark.simm
def test_simm_cache_engine_match(simm_client):
    """Test SimmCacheEngine.match: put blocks for a sequence's block_hashes, then match returns num_blocks."""
    from flexkv.external.simm_utils import SimmCacheEngine

    tokens_per_block = 16
    num_blocks = 3
    token_ids = np.random.randint(0, 1000, size=num_blocks * tokens_per_block, dtype=np.int64)
    sequence_meta = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)
    assert sequence_meta.num_blocks == num_blocks

    layout = _make_blockfirst_layout(num_blocks=num_blocks, tokens_per_block=tokens_per_block)
    dtype = torch.float16
    elements_per_block = layout.get_elements_per_block()
    block_size_bytes = elements_per_block * dtype.itemsize
    total_bytes = layout.num_block * block_size_bytes

    buffer = torch.zeros(total_bytes // dtype.itemsize, dtype=dtype)
    simm_client.register_mr(buffer.data_ptr(), total_bytes)

    keys = [str(sequence_meta.block_hashes[i]) for i in range(sequence_meta.num_blocks)]
    ptrs = [
        int(buffer.data_ptr() + i * block_size_bytes)
        for i in range(len(keys))
    ]
    sizes = [block_size_bytes] * len(keys)
    set_ok = simm_client.batch_set_v1(cpu_ptrs=ptrs, block_sizes_list=sizes, keys_strs=keys)
    assert all(set_ok), f"batch_set_v1 failed: {set_ok}"

    cache_config = CacheConfig(
        tokens_per_block=tokens_per_block,
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=True,
        use_simm_backend=True,
        simm_manager_address=_get_manager_address(),
        num_cpu_blocks=64,
        num_remote_blocks=128,
    )
    engine = SimmCacheEngine(cache_config)
    match_result = engine.match(sequence_meta)
    assert match_result.num_matched_blocks == num_blocks, (
        f"expected match {num_blocks} blocks, got {match_result.num_matched_blocks}"
    )


@pytest.mark.simm
def test_simm_integration_e2e(simm_client):
    """E2E: put blocks with known data -> match -> get -> verify content."""
    from flexkv.external.simm_utils import SimmCacheEngine

    tokens_per_block = 16
    num_blocks = 2
    token_ids = np.random.randint(0, 1000, size=num_blocks * tokens_per_block, dtype=np.int64)
    sequence_meta = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)

    layout = _make_blockfirst_layout(num_blocks=num_blocks, tokens_per_block=tokens_per_block)
    dtype = torch.float16
    elements_per_block = layout.get_elements_per_block()
    block_size_bytes = elements_per_block * dtype.itemsize
    total_bytes = layout.num_block * block_size_bytes

    buffer = torch.zeros(total_bytes // dtype.itemsize, dtype=dtype)
    simm_client.register_mr(buffer.data_ptr(), total_bytes)

    # Fill with pattern
    for i in range(num_blocks):
        base = i * elements_per_block
        buffer.view(-1)[base : base + elements_per_block] = float(200 + i)

    keys = [str(sequence_meta.block_hashes[i]) for i in range(sequence_meta.num_blocks)]
    ptrs = [int(buffer.data_ptr() + i * block_size_bytes) for i in range(len(keys))]
    sizes = [block_size_bytes] * len(keys)

    set_ok = simm_client.batch_set_v1(cpu_ptrs=ptrs, block_sizes_list=sizes, keys_strs=keys)
    assert all(set_ok), "batch_set_v1 failed"

    cache_config = CacheConfig(
        tokens_per_block=tokens_per_block,
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=True,
        use_simm_backend=True,
        simm_manager_address=_get_manager_address(),
        num_cpu_blocks=64,
        num_remote_blocks=128,
    )
    engine = SimmCacheEngine(cache_config)
    match_result = engine.match(sequence_meta)
    assert match_result.num_matched_blocks == num_blocks, "match should return all blocks"

    buffer.zero_()
    get_ok = simm_client.batch_get_v1(cpu_ptrs=ptrs, block_sizes=sizes, keys=keys)
    assert all(get_ok), "batch_get_v1 failed"
    for i in range(num_blocks):
        base = i * elements_per_block
        assert buffer.view(-1)[base].item() == float(200 + i), f"block {i} content mismatch"
