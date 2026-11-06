"""Tests for the KV-cache reset path (KVTaskEngine.reset_cache / KVManager.reset).

This is the feature that lets vLLM's reset_prefix_cache(reset_connector=True)
propagate down to FlexKV so that KV computed against stale weights (e.g. after a
verl RL weight update) is fully invalidated.

Layout:
  - Level A (this file): in-process, no GPU needed for the pure-reset assertions.
    * test_reset_empty_is_noop            — idempotent / cheap short-circuit
    * test_reset_after_put_clears_match   — put -> match hits -> reset -> match misses
                                            (GPU: registers TP client, mirrors test_kvmanager.py)
  - Level A server-client (test_reset_cache_server_client): reset over ZMQ IPC.

Run:
    pytest -s tests/test_reset_cache.py
GPU-dependent cases self-skip when no CUDA device is present.
"""
import time

import pytest
import torch
import multiprocessing as mp

from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.request import KVResponseStatus
from flexkv.kvmanager import KVManager
from flexkv.common.debug import flexkv_logger

from common_utils import (
    generate_request_pair,
    block_ids_2_slot_mapping,
    skip_if_insufficient_gpus,
    create_gpu_kv_layout,
)

# Reuse the TP-client boot helpers from the main kvmanager test.
from test_kvmanager import run_tp_client, shutdown_tp_client


# ---------------------------------------------------------------------------
# Small configs (kept independent of the shared fixtures so this file can be
# run in isolation).
# ---------------------------------------------------------------------------
def _model_config():
    return ModelConfig(
        num_layers=4,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
        use_mla=False,
        tp_size=1,
        dp_size=1,
    )


def _cache_config(**overrides):
    cfg = dict(
        tokens_per_block=16,
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=False,
        num_cpu_blocks=1024,
    )
    cfg.update(overrides)
    return CacheConfig(**cfg)


def _cpu_engine(kvmanager):
    """Convenience accessor for the CPU tier engine (for white-box assertions)."""
    return kvmanager.kv_task_engine.cache_engine.cpu_cache_engine


# ---------------------------------------------------------------------------
# Level A — pure reset behavior
# ---------------------------------------------------------------------------
def test_reset_empty_is_noop():
    """reset() on a fresh manager must be a cheap no-op and not raise.

    Covers the §2.5 idempotency requirement: because verl triggers reset up to
    3x per weight update, reset on an already-empty cache must be safe/cheap.
    """
    kvm = KVManager(model_config=_model_config(), cache_config=_cache_config(), dp_client_id=0)
    kvm.start()
    try:
        # No in-flight tasks, empty tree -> should return immediately.
        kvm.reset()
        kvm.reset()  # second call must also be fine (idempotent)
    finally:
        kvm.shutdown()


def test_reset_after_put_clears_match():
    """Full behavioral test: put KV, confirm it matches, reset, confirm it no
    longer matches (prefix tree dropped) and the mempool is fully freed.

    Requires 1 GPU because put/get transfers go through a registered TP client
    (same pattern as tests/test_kvmanager.py).
    """
    skip_if_insufficient_gpus(1)

    model_config = _model_config()
    cache_config = _cache_config(num_cpu_blocks=1024)
    tokens_per_block = cache_config.tokens_per_block
    num_gpu_blocks = 128
    block_per_request = 16
    gpu_layout_type = 0

    kvm = KVManager(model_config=model_config, cache_config=cache_config, dp_client_id=0)
    kvm.start()

    # Boot a single TP client that registers GPU blocks with FlexKV.
    mp_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe()
    tp_proc = mp_ctx.Process(
        target=run_tp_client,
        args=(0, 0, kvm.gpu_register_port, model_config, cache_config,
              num_gpu_blocks, child_conn, gpu_layout_type),
        daemon=True,
    )
    tp_proc.start()
    # Drain the pipe (the helper sends shared GPU block handles back).
    try:
        parent_conn.recv()
    except Exception:
        pass

    try:
        while not kvm.is_ready():
            time.sleep(0.5)
            flexkv_logger.info("waiting for flexkv to be ready")

        token_ids, block_ids, _ = generate_request_pair(
            0, block_per_request, num_gpu_blocks, tokens_per_block, dp_size=1
        )
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)

        # 1. PUT a prefix and wait until it lands in the CPU tier.
        put_id = kvm.put_async(token_ids=token_ids, slot_mapping=slot_mapping, token_mask=None)
        res = kvm.wait([put_id], completely=True)
        assert res[put_id].status == KVResponseStatus.SUCCESS

        # 2. Before reset: get_match should find the cached prefix.
        _, matched_before = kvm.get_match(token_ids=token_ids, token_mask=None)
        assert matched_before.sum().item() > 0, "expected a cache hit before reset"

        # 3. RESET (the feature under test).
        kvm.reset()

        # 4. After reset: same tokens must NOT match (prefix tree dropped).
        _, matched_after = kvm.get_match(token_ids=token_ids, token_mask=None)
        assert matched_after.sum().item() == 0, (
            f"expected 0 cache hits after reset, got {matched_after.sum().item()}"
        )

        # 5. White-box: CPU mempool fully freed (all blocks returned).
        mempool = _cpu_engine(kvm).mempool
        # num_free_blocks should be back to total capacity right after a reset.
        assert mempool.num_free_blocks == mempool.num_total_blocks, (
            f"mempool not fully freed after reset: "
            f"free={mempool.num_free_blocks} total={mempool.num_total_blocks}"
        )
    finally:
        shutdown_tp_client([tp_proc])
        kvm.shutdown()


# ---------------------------------------------------------------------------
# Level A (server-client) — reset over ZMQ IPC
# ---------------------------------------------------------------------------
def test_reset_cache_server_client(monkeypatch):
    """reset() must work (not no-op-error) in server-client mode via ResetRequest.

    We force server-client mode with FLEXKV_SERVER_CLIENT_MODE=1 and assert the
    round-trip completes without raising. Requires 1 GPU for the TP client that
    the server needs to become ready.
    """
    skip_if_insufficient_gpus(1)
    monkeypatch.setenv("FLEXKV_SERVER_CLIENT_MODE", "1")
    # NOTE: GLOBAL_CONFIG_FROM_ENV is read at import time; if it was already
    # imported, also flip the flag directly.
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "server_client_mode", True, raising=False)

    model_config = _model_config()
    cache_config = _cache_config()
    num_gpu_blocks = 128
    gpu_layout_type = 0

    kvm = KVManager(model_config=model_config, cache_config=cache_config, dp_client_id=0)
    assert kvm.server_client_mode, "expected server-client mode to be active"
    kvm.start()

    mp_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe()
    tp_proc = mp_ctx.Process(
        target=run_tp_client,
        args=(0, 0, kvm.gpu_register_port, model_config, cache_config,
              num_gpu_blocks, child_conn, gpu_layout_type),
        daemon=True,
    )
    tp_proc.start()
    try:
        parent_conn.recv()
    except Exception:
        pass

    try:
        while not kvm.is_ready():
            time.sleep(0.5)
        # The key assertion: this used to be a hard no-op error
        # ("clear_cache is not supported in server client mode").
        # Now it must complete the ZMQ round-trip without raising.
        kvm.reset()
    finally:
        shutdown_tp_client([tp_proc])
        kvm.shutdown()
