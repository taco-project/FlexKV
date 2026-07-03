# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the launch-time SWA GPU-slot late-bind path.

The connector calls ``KVManager.launch(task, slot_mapping, swa_slot_mapping)``.
In-process that reaches ``KVTaskEngine.set_slot_mappings`` ->
``_set_slot_mapping_impl`` -> ``graph.set_gpu_blocks`` (full-KV) +
``graph.set_swa_gpu_blocks`` (SWA). Standing up a full ``KVTaskEngine`` spawns a
TransferManager subprocess (needs GPU), so this test drives the exact same
graph-level binding logic directly on a graph produced by the real
``GlobalCacheEngine.get()`` — the piece that would otherwise only be covered by
the GPU e2e test.

Verifies:
  * full-KV GPU ops rebind from the full slot_mapping; SWA GPU op rebinds from
    the SWA slot_mapping; the two never cross-contaminate.
  * swa_slot_mapping_to_slot_ids folds token indices to SWA-pool slot ids.
  * an empty/None swa_slot_mapping leaves the SWA op's placeholder untouched
    (degrade path: SWA GPU pool not registered by the connector).

CPU-only, requires flexkv.c_ext (production CacheEngineAccel).
"""
import numpy as np
import pytest
import torch

pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.transfer import TransferType
from flexkv.common.debug import flexkv_logger

flexkv_logger.set_level("OFF")

TPB = 16


def _model_config():
    return ModelConfig(num_layers=4, num_kv_heads=1, head_size=128,
                       use_mla=True, dtype=torch.bfloat16, tp_size=1, dp_size=1)


def _cache_config():
    cc = CacheConfig(tokens_per_block=TPB, enable_cpu=True, enable_ssd=False,
                     enable_remote=False, num_cpu_blocks=4096)
    cc.swa = SWAPoolConfig(enabled=True, num_slots=256, window_size=TPB,
                           num_swa_layers=1, bytes_per_token_per_layer=64)
    cc.enable_swa_transfer = True
    return cc


def _tokens(n_blocks, base):
    rs = np.random.RandomState(base)
    return rs.randint(0, 30000, size=n_blocks * TPB, dtype=np.int64)


def _seed_swa_hit(eng, tok):
    """PUT tok so the tail node carries an SWA slot; complete the ops."""
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    for c in op_cb.values():
        c()
    cb()


def test_swa_slot_mapping_to_slot_ids_folds_by_window():
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    # 3 windows worth of token-index slot_mapping starting at GPU slot 5,6,7.
    sm = np.concatenate([
        np.arange(5 * TPB, 6 * TPB),
        np.arange(6 * TPB, 7 * TPB),
        np.arange(7 * TPB, 8 * TPB),
    ]).astype(np.int64)
    ids = eng.swa_slot_mapping_to_slot_ids(sm)
    assert ids.tolist() == [5, 6, 7]


def test_launch_bind_get_rebinds_swa_gpu_only():
    """GET graph: set_gpu_blocks binds full-KV H2D dst; set_swa_gpu_blocks binds
    the SWA H2D dst; neither touches the other's ops."""
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tok = _tokens(4, base=11)
    _seed_swa_hit(eng, tok)

    # GET with a fake slot_mapping (UNREADY-style): build graph, then late-bind.
    fake_sm = np.zeros_like(tok)
    graph, _rm, cb, op_cb, end_id = eng.get(
        request_id=2, token_ids=tok, token_mask=np.ones_like(tok, dtype=np.int64),
        slot_mapping=fake_sm, dp_client_id=0)
    full_h2d = [o for o in graph._op_map.values()
                if not o.is_swa and o.transfer_type == TransferType.H2D]
    swa_h2d = [o for o in graph._op_map.values()
               if o.is_swa and o.transfer_type == TransferType.H2D]
    assert len(swa_h2d) == 1, "expected one SWA H2D"
    # Unified model (PR#191): the SWA H2D is tracked in BOTH lists. The safety
    # property is verified below — single-arg set_gpu_blocks(full_gpu) leaves the
    # SWA op untouched, and set_swa_gpu_blocks binds it independently.
    assert swa_h2d[0].op_id in graph._swa_gpu_transfer_op_id
    assert swa_h2d[0].op_id in graph._gpu_transfer_op_id

    # Bind full-KV GPU blocks (as _set_slot_mapping_impl does).
    full_gpu = np.arange(100, 100 + len(tok) // TPB, dtype=np.int64)
    graph.set_gpu_blocks(full_gpu)
    # SWA GPU slot_mapping -> slot ids -> set_swa_gpu_blocks.
    swa_sm = np.arange(9 * TPB, 10 * TPB, dtype=np.int64)  # -> slot 9
    graph.set_swa_gpu_blocks(eng.swa_slot_mapping_to_slot_ids(swa_sm))

    if full_h2d:
        assert full_h2d[0].dst_block_ids.tolist() == full_gpu[:full_h2d[0].dst_block_ids.size].tolist()
    assert swa_h2d[0].dst_block_ids.tolist() == [9]  # SWA rebound, independent
    for c in op_cb.values():
        c()
    cb()


def test_launch_bind_put_rebinds_swa_gpu_src():
    """PUT graph: SWA D2H has GPU on the src side; set_swa_gpu_blocks binds src."""
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tok = _tokens(4, base=12)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    graph, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    swa_d2h = [o for o in graph._op_map.values()
               if o.is_swa and o.transfer_type == TransferType.D2H]
    assert len(swa_d2h) == 1
    assert swa_d2h[0].src_block_ids.tolist() == [0]  # placeholder
    graph.set_swa_gpu_blocks(eng.swa_slot_mapping_to_slot_ids(
        np.arange(4 * TPB, 5 * TPB, dtype=np.int64)))  # -> slot 4
    assert swa_d2h[0].src_block_ids.tolist() == [4]
    for c in op_cb.values():
        c()
    cb()


def test_no_swa_slot_mapping_leaves_placeholder():
    """Degrade path: connector did not register an SWA GPU pool, so it supplies
    no swa_slot_mapping. set_gpu_blocks (full-KV) must NOT touch the SWA op; its
    GPU placeholder stays as built (the SWA transfer simply won't be launched by
    a connector that has no SWA GPU pool)."""
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tok = _tokens(4, base=13)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    graph, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    swa_d2h = [o for o in graph._op_map.values() if o.is_swa][0]
    before = swa_d2h.src_block_ids.tolist()
    # Only full-KV late-bind runs (no swa_slot_mapping).
    graph.set_gpu_blocks(np.arange(50, 50 + len(tok) // TPB, dtype=np.int64))
    assert swa_d2h.src_block_ids.tolist() == before  # untouched by full-KV bind
    for c in op_cb.values():
        c()
    cb()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
