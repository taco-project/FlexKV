# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-plane SWA: graph build (GlobalCacheEngine.put/get) + launch-time
late-bind of the SWA GPU slot.

Enters from the top (GlobalCacheEngine), NOT the data plane, using the REAL
(de-stubbed) SWA slot sources — the control plane's job is to turn a request
into a Full+SWA transfer graph + masks against the node-mounted radix tree:

    put()       -> full-KV D2H graph + SWA peer D2H (alloc slot + set_swa)
    swa_align() -> full_hit, swa_hit, usable = min(full, swa)
    get()       -> full-KV H2D graph + SWA peer H2D (matched slot), joined by
                   the VIRTUAL barrier; the matched CPU SWA node is pinned for
                   load and released by the H2D completion callback.

The launch-time bind path mirrors what the connector triggers via
``KVManager.launch(task, slot_mapping, swa_slot_mapping)`` ->
``KVTaskEngine.set_slot_mappings`` -> ``graph.set_gpu_blocks`` (full-KV) /
``graph.set_swa_gpu_blocks`` (SWA). Standing up a full KVTaskEngine spawns a
TransferManager subprocess (needs GPU); this drives the exact graph-level bind
logic directly on a graph the real GlobalCacheEngine produced.

CPU-only (the GPU SWA slot is a placeholder bound late); byte movement is the
data plane's job (test_swa_control_plane_e2e.py / the KVManager GPU e2e).
Requires flexkv.c_ext (production CacheEngineAccel / CRadixTreeIndex).
"""
import numpy as np
import pytest
import torch

pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.transfer import TransferType
from flexkv.common.debug import flexkv_logger

flexkv_logger.set_level("OFF")

pytestmark = pytest.mark.smoke

TPB = 16


def _model_config():
    return ModelConfig(
        num_layers=4, num_kv_heads=1, head_size=128,
        use_mla=True, dtype=torch.bfloat16, tp_size=1, dp_size=1,
    )


def _cache_config(enable_swa_transfer: bool = True):
    cc = CacheConfig(
        tokens_per_block=TPB,
        enable_cpu=True, enable_ssd=False, enable_remote=False,
        num_cpu_blocks=4096,
    )
    cc.swa = SWAPoolConfig(
        enabled=True, num_slots=256, window_size=TPB,
        num_swa_layers=1, bytes_per_token_per_layer=64,
    )
    cc.enable_swa_transfer = enable_swa_transfer
    return cc


def _swa_ops(graph):
    return [op for op in graph._op_map.values() if getattr(op, "is_swa", False)]


def _full_ops(graph):
    return [op for op in graph._op_map.values()
            if not getattr(op, "is_swa", False)
            and op.transfer_type != TransferType.VIRTUAL]


def _tokens(n_blocks, base):
    rs = np.random.RandomState(base)
    return rs.randint(0, 30000, size=n_blocks * TPB, dtype=np.int64)


def _complete(op_cb, cb):
    for c in op_cb.values():
        c()
    cb()


def _seed_swa_hit(eng, tok):
    """PUT tok so the tail node carries an SWA slot; complete the ops."""
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    _complete(op_cb, cb)


# =========================================================================== #
# 1. control-plane graph build (put / swa_align / get)                        #
# =========================================================================== #

def test_put_builds_full_plus_swa_store_chain():
    eng = GlobalCacheEngine(_cache_config(True), _model_config())
    tok = _tokens(4, base=1)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)

    graph, return_mask, cb, op_cb, end_id = eng.put(
        request_id=1, token_ids=tok, token_mask=mask,
        slot_mapping=slot_mapping, dp_client_id=0)

    full = _full_ops(graph)
    swa = _swa_ops(graph)
    assert any(o.transfer_type == TransferType.D2H for o in full), "full-KV D2H missing"
    assert len(swa) == 1 and swa[0].transfer_type == TransferType.D2H and swa[0].is_swa
    # the SWA D2H's CPU dst is a real allocated pool slot; GPU src is the
    # size-1 placeholder (bound late via set_swa_gpu_blocks).
    assert swa[0].op_id in graph._swa_gpu_transfer_op_id
    assert eng.cpu_cache_engine.swa_pool.num_used == 1  # one slot allocated
    _complete(op_cb, cb)
    # after completion the SWA slot is mounted on the stored tail node.
    sm = SequenceMeta(token_ids=tok, tokens_per_block=TPB); sm.gen_hashes()
    hit, slot, key = eng.cpu_cache_engine.match_swa(sm, upper_bound_blocks=4)
    assert hit == 4 and slot >= 0


def test_swa_align_clamps_full_to_swa_hit():
    eng = GlobalCacheEngine(_cache_config(True), _model_config())
    tok = _tokens(4, base=2)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)

    _g, _rm, cb, op_cb, _e = eng.put(1, tok, mask, slot_mapping, dp_client_id=0)
    _complete(op_cb, cb)

    # swa_align: full_hit=4, swa_hit=4 (SWA on the stored tail), usable=min=4.
    full_hit, swa_hit = eng.swa_align(tok, np.ones_like(tok, dtype=np.bool_))
    assert full_hit == 4, f"full_hit={full_hit}"
    assert swa_hit == 4, f"swa_hit={swa_hit}"
    assert min(full_hit, swa_hit) == 4


def test_swa_align_hit_equals_fetched_cpu_tier(monkeypatch):
    """#9 tier-consistency guard: swa_align's returned SWA hit shapes the transfer
    graph, but the GET data plane (_swa_get_slots) fetches the CPU tier ONLY.
    match_swa_prefix reports best_hit_blocks = cross-tier MAX. If a non-CPU tier
    ever reports a LONGER hit than CPU, returning best would shape a graph the
    CPU-only fetch cannot fill (src/dst mismatch / stale SWA). swa_align must
    clamp its return to the CPU (fetched) tier. We fake a longer SSD hit and
    assert the returned hit stays at the CPU tier's value."""
    eng = GlobalCacheEngine(_cache_config(True), _model_config())
    tok = _tokens(4, base=7)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(1, tok, mask, slot_mapping, dp_client_id=0)
    _complete(op_cb, cb)

    # CPU-only truth: best == cpu == 4.
    full_hit, swa_hit = eng.swa_align(tok, np.ones_like(tok, dtype=np.bool_))
    assert swa_hit == 4

    # Simulate a future SSD/REMOTE tier reporting a LONGER SWA hit than CPU.
    real_match = eng.match_swa_prefix

    def _fake_match(sequence_meta, cpu_full_hit_blocks, ssd_full_hit_blocks=0,
                    remote_full_hit_blocks=0, lock_for_load=False):
        r = real_match(sequence_meta, cpu_full_hit_blocks=cpu_full_hit_blocks,
                       ssd_full_hit_blocks=ssd_full_hit_blocks,
                       remote_full_hit_blocks=remote_full_hit_blocks,
                       lock_for_load=lock_for_load)
        r.ssd_hit_blocks = r.cpu_hit_blocks + 2  # SSD claims a longer hit
        r.ssd_slot = 999
        return r

    monkeypatch.setattr(eng, "match_swa_prefix", _fake_match)
    full_hit2, swa_hit2 = eng.swa_align(tok, np.ones_like(tok, dtype=np.bool_))
    # Must clamp to the fetched CPU tier (4), NOT the cross-tier max (6).
    assert swa_hit2 == 4, (
        f"swa_align returned {swa_hit2} (cross-tier best) but GET fetches CPU-only "
        f"(cpu_hit=4); graph would be shaped for blocks the fetch cannot fill")


def test_get_builds_full_plus_swa_load_chain():
    eng = GlobalCacheEngine(_cache_config(True), _model_config())
    tok = _tokens(4, base=3)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(1, tok, mask, slot_mapping, dp_client_id=0)
    _complete(op_cb, cb)

    # GET the same prefix: full-KV H2D + SWA peer H2D, joined by VIRTUAL barrier.
    graph, return_mask, gcb, gop_cb, end_id = eng.get(
        request_id=2, token_ids=tok, token_mask=np.ones_like(tok, dtype=np.int64),
        slot_mapping=slot_mapping, dp_client_id=0)
    swa = _swa_ops(graph)
    assert len(swa) == 1 and swa[0].transfer_type == TransferType.H2D and swa[0].is_swa
    assert swa[0].op_id in graph._swa_gpu_transfer_op_id
    barrier = graph._op_map[end_id]
    assert barrier.transfer_type == TransferType.VIRTUAL
    assert swa[0].op_id in barrier.predecessors, "SWA H2D not joined into barrier"
    # the matched CPU SWA node was pinned for load; releasing via the H2D callback
    # must drop the pin (no leak).
    sm = SequenceMeta(token_ids=tok, tokens_per_block=TPB); sm.gen_hashes()
    _complete(gop_cb, gcb)
    # after release, the node's SWA is unlocked (a fresh match can lock again).
    hit, slot, key, node = eng.cpu_cache_engine.match_swa_locked(sm, upper_bound_blocks=4)
    assert node is not None and node.swa_lock_ref == 1
    node.dec_swa_lock_ref()


def test_gate_off_no_swa_ops_in_control_plane_graph():
    """gate OFF: control plane emits NO SWA ops and allocates no slot."""
    eng = GlobalCacheEngine(_cache_config(False), _model_config())
    tok = _tokens(4, base=4)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)
    graph, _rm, cb, op_cb, _e = eng.put(1, tok, mask, slot_mapping, dp_client_id=0)
    assert len(_swa_ops(graph)) == 0, "SWA ops emitted with enable_swa_transfer=False"
    assert eng.cpu_cache_engine.swa_pool.num_used == 0  # no slot allocated
    _complete(op_cb, cb)


# =========================================================================== #
# 2. launch-time late-bind of the SWA GPU slot                                #
# =========================================================================== #

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
    _complete(op_cb, cb)


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
    _complete(op_cb, cb)


def test_no_swa_slot_mapping_leaves_placeholder():
    """Degrade path: connector did not register an SWA GPU pool, so it supplies
    no swa_slot_mapping. Single-arg set_gpu_blocks (full-KV) must NOT touch the
    SWA op; its GPU placeholder stays as built (the SWA transfer simply won't be
    launched by a connector that has no SWA GPU pool)."""
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
    _complete(op_cb, cb)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
