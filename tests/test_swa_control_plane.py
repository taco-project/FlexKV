# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-plane SWA: graph build (GlobalCacheEngine.put/get) + launch-time
late-bind of the SWA GPU slot.

Enters from the top (GlobalCacheEngine), NOT the data plane, using the REAL
(de-stubbed) SWA slot sources — the control plane's job is to turn a request
into a Full+SWA transfer graph + masks against the node-mounted radix tree:

    put()       -> full-KV D2H graph + SWA peer D2H (reserve slot, publish the
                   node mount from the SWA op completion callback)
    get(swa_aware=True) -> full-KV H2D clamped to usable=min(full,swa) + SWA peer
                   H2D (matched slot), joined by the VIRTUAL barrier; the matched
                   CPU SWA node is pinned for
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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import (
    CPUONLY_CACHE_STRATEGY,
    CacheEngineAccel,
    GlobalCacheEngine,
)
from flexkv.cache.hie_cache_engine import HierarchyLRCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.transfer import DeviceType, TransferType
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
        enabled=True,
        num_slots=256,
        num_swa_layers=1,
        bytes_per_token_per_layer=64,
    )
    cc.enable_swa_transfer = enable_swa_transfer
    return cc


def _cache_config_ssd(enable_swa_transfer: bool = True):
    """CPU + SSD tiers, both with an SWA host pool. The SSD cache engine is an
    in-memory radix+mempool+swa_pool (no real SSD files at construction — files
    are only touched by the data-plane worker at transfer time), so multi-tier
    SWA orchestration is built inside _put_impl_* / _get_impl_* and is testable
    at smoke level without disk I/O."""
    cc = CacheConfig(
        tokens_per_block=TPB,
        enable_cpu=True, enable_ssd=True, enable_remote=False,
        num_cpu_blocks=4096, num_ssd_blocks=4096,
        ssd_cache_dir="./ssd_cache_swa_test",
    )
    cc.swa = SWAPoolConfig(
        enabled=True,
        num_slots=256,
        num_ssd_slots=256,
        num_swa_layers=1,
        bytes_per_token_per_layer=64,
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


def test_cpu_cache_engine_initializes_swa_pool_from_constructor():
    swa_config = SWAPoolConfig(
        enabled=True,
        num_slots=7,
        num_swa_layers=1,
        bytes_per_token_per_layer=64,
    )
    eng = CacheEngineAccel(
        DeviceType.CPU,
        num_total_blocks=32,
        tokens_per_block=TPB,
        evict_ratio=0.1,
        swa_config=swa_config,
    )

    assert eng.swa_enabled
    assert eng.swa_pool.num_slots == 7


def test_hierarchy_engine_does_not_advertise_swa_support():
    eng = HierarchyLRCacheEngine.__new__(HierarchyLRCacheEngine)
    eng.swa_pool = object()

    assert not eng.swa_enabled
    with pytest.raises(NotImplementedError):
        eng.init_swa(_cache_config().swa)


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


def _seed_long_ssd_short_cpu_hit(eng, tok):
    """Seed a 4-block SSD hit and a 2-block CPU hit, both with SWA."""
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(
        1, tok, mask.copy(), slot_mapping, dp_client_id=0)
    _complete(op_cb, cb)

    # Keep the SSD copy, but rebuild CPU with only the shorter prefix. This
    # makes the CPU SWA node a leaf: evicting its SWA without protecting the
    # node also recycles the Full-KV blocks that a concurrent GET may reference.
    eng.cpu_cache_engine.reset()
    prefix = tok[:2 * TPB]
    prefix_mask = np.ones_like(prefix, dtype=np.int64)
    prefix_mapping = np.arange(prefix.shape[0], dtype=np.int64)
    _g2, _rm2, cb2, op_cb2, _e2 = eng.put(
        2,
        prefix,
        prefix_mask,
        prefix_mapping,
        dp_client_id=0,
        temp_cache_strategy=CPUONLY_CACHE_STRATEGY,
    )
    _complete(op_cb2, cb2)

    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB)
    cpu_match = eng.cpu_cache_engine.match(seq)
    ssd_match = eng.ssd_cache_engine.match(seq)
    assert cpu_match.num_ready_matched_blocks == 2
    assert cpu_match.swa_hit_blocks == 2
    assert ssd_match.num_ready_matched_blocks == 4
    assert ssd_match.swa_hit_blocks == 4
    return cpu_match, ssd_match


# =========================================================================== #
# 1. control-plane graph build (put / get)                                    #
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
    sm = SequenceMeta(token_ids=tok, tokens_per_block=TPB); sm.gen_hashes()
    pending = eng.cpu_cache_engine.match(sm)
    assert pending.num_ready_matched_blocks == 0
    assert pending.last_node.swa_host_slot == -1
    assert pending.swa_hit_blocks == 0

    full_d2h = next(o for o in full if o.transfer_type == TransferType.D2H)
    swa_d2h = swa[0]
    barrier = graph._op_map[end_id]
    assert barrier.transfer_type == TransferType.VIRTUAL
    assert barrier.predecessors == {full_d2h.op_id, swa_d2h.op_id}

    # Full-KV may become readable first, but the reserved SWA slot must remain a
    # miss until its independent sibling D2H completes.
    op_cb[full_d2h.op_id]()
    full_only = eng.cpu_cache_engine.match(sm)
    assert full_only.num_ready_matched_blocks == 4
    assert full_only.swa_hit_blocks == 0
    assert full_only.last_node.swa_host_slot == -1

    op_cb[swa_d2h.op_id]()
    cb()
    # SWA completion publishes the reserved slot independently.
    ready = eng.cpu_cache_engine.match(sm)
    assert ready.swa_hit_blocks == 4
    assert ready.last_swa_node is not None
    assert ready.last_swa_node.swa_host_slot >= 0


def test_put_swa_first_stays_hidden_until_full_kv_is_ready():
    eng = GlobalCacheEngine(_cache_config(True), _model_config())
    tok = _tokens(4, base=2)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)
    graph, _return_mask, cb, op_cb, _end_id = eng.put(
        request_id=1,
        token_ids=tok,
        token_mask=mask,
        slot_mapping=slot_mapping,
        dp_client_id=0,
    )
    full_d2h = next(o for o in _full_ops(graph)
                    if o.transfer_type == TransferType.D2H)
    swa_d2h = next(o for o in _swa_ops(graph)
                   if o.transfer_type == TransferType.D2H)
    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()

    # The SWA bytes may land first.  Mounting the completed slot is safe because
    # match_prefix still requires the owning Full-KV node to be ready.
    op_cb[swa_d2h.op_id]()
    swa_only = eng.cpu_cache_engine.match(seq)
    assert swa_only.num_ready_matched_blocks == 0
    assert swa_only.swa_hit_blocks == 0
    assert swa_only.last_node.swa_host_slot >= 0

    op_cb[full_d2h.op_id]()
    cb()
    ready = eng.cpu_cache_engine.match(seq)
    assert ready.num_ready_matched_blocks == 4
    assert ready.swa_hit_blocks == 4


def test_put_full_swa_pool_keeps_match_node_alive_until_insert():
    """Allocating a replacement SWA slot must not evict the match_result node."""
    cc = _cache_config(True)
    cc.swa.num_slots = 1
    eng = GlobalCacheEngine(cc, _model_config())
    prefix = _tokens(2, base=26)
    extended = np.concatenate([prefix, _tokens(2, base=27)])

    mask2 = np.ones_like(prefix, dtype=np.int64)
    sm2 = np.arange(prefix.shape[0], dtype=np.int64)
    _g1, _rm1, cb1, op_cb1, _e1 = eng.put(
        1, prefix, mask2, sm2, dp_client_id=0)
    _complete(op_cb1, cb1)
    assert eng.cpu_cache_engine.swa_pool.num_used == 1

    mask4 = np.ones_like(extended, dtype=np.int64)
    sm4 = np.arange(extended.shape[0], dtype=np.int64)
    graph, return_mask, cb2, op_cb2, _e2 = eng.put(
        2, extended, mask4, sm4, dp_client_id=0)
    assert graph.num_ops > 0
    assert return_mask[2 * TPB:4 * TPB].all()
    _complete(op_cb2, cb2)

    seq = SequenceMeta(token_ids=extended, tokens_per_block=TPB)
    ready = eng.cpu_cache_engine.match(seq)
    assert ready.num_ready_matched_blocks == 4
    assert ready.swa_hit_blocks == 4
    assert eng.cpu_cache_engine.swa_pool.num_used == 1


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
        slot_mapping=slot_mapping, dp_client_id=0, swa_aware=True)
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
    ready = eng.cpu_cache_engine.match(sm)
    assert ready.swa_hit_blocks == 4
    node = ready.last_swa_node
    assert node is not None
    eng.cpu_cache_engine._pin_swa_node(node)
    assert node is not None and node.swa_lock_ref == 1
    assert node.get_lock_cnt() >= node.swa_lock_ref
    eng._swa_release_load_lock(node, engine=eng.cpu_cache_engine)
    assert node.swa_lock_ref == 0


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


def test_put_fails_when_cpu_swa_slot_unavailable():
    """SWA-enabled PUT must not degrade into a ready full-only CPU leaf."""
    cc = _cache_config(True)
    cc.swa.num_slots = 0
    eng = GlobalCacheEngine(cc, _model_config())
    tok = _tokens(4, base=5)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)

    graph, return_mask, cb, op_cb, end_id = eng.put(
        request_id=1, token_ids=tok, token_mask=mask,
        slot_mapping=slot_mapping, dp_client_id=0)

    assert end_id == -1
    assert not return_mask.any()
    assert len(graph._op_map) == 0
    assert len(op_cb) == 0
    assert eng.cpu_cache_engine.mempool.num_free_blocks == \
        eng.cpu_cache_engine.mempool.num_total_blocks
    assert eng.cpu_cache_engine.index.total_cached_blocks() == 0
    cb()


def test_put_fails_when_ssd_swa_slot_unavailable():
    """If SSD gets a full node, its SWA slot is mandatory too."""
    cc = _cache_config_ssd(True)
    cc.swa.num_ssd_slots = 0
    eng = GlobalCacheEngine(cc, _model_config())
    tok = _tokens(4, base=6)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)

    graph, return_mask, cb, op_cb, end_id = eng.put(
        request_id=1, token_ids=tok, token_mask=mask,
        slot_mapping=slot_mapping, dp_client_id=0)

    assert end_id == -1
    assert not return_mask.any()
    assert len(graph._op_map) == 0
    assert len(op_cb) == 0
    assert eng.cpu_cache_engine.swa_pool.num_used == 0
    assert eng.cpu_cache_engine.mempool.num_free_blocks == \
        eng.cpu_cache_engine.mempool.num_total_blocks
    assert eng.ssd_cache_engine.mempool.num_free_blocks == \
        eng.ssd_cache_engine.mempool.num_total_blocks
    assert eng.cpu_cache_engine.index.total_cached_blocks() == 0
    assert eng.ssd_cache_engine.index.total_cached_blocks() == 0
    cb()


# =========================================================================== #
# 2. launch-time late-bind of the SWA GPU slot                                #
# =========================================================================== #

def test_swa_slot_mapping_to_slot_ids_folds_by_page():
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
        slot_mapping=fake_sm, dp_client_id=0, swa_aware=True)
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


# =========================================================================== #
# 3. multi-tier SWA orchestration (CPU + SSD): write-through + get staging     #
#    These exercise SWA graph append inside _put_impl_* / _get_impl_*.         #
# =========================================================================== #

def test_put_writethrough_ssd_builds_swa_h2disk():
    """PUT with an SSD tier: SWA store must write through to SSD, mirroring the
    full-KV H2DISK. The SWA graph should carry a SWA D2H (GPU->CPU) AND a SWA
    H2DISK (CPU->SSD) that depends on the D2H (fire-and-forget). The SSD SWA
    slot remains reserved and invisible until its own write-through completes."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    tok = _tokens(4, base=21)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)

    graph, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    swa = _swa_ops(graph)
    kinds = sorted(o.transfer_type.name for o in swa)
    assert "D2H" in kinds, f"SWA D2H missing: {kinds}"
    assert "H2DISK" in kinds, f"SWA write-through H2DISK missing (CPU-only bug): {kinds}"
    swa_d2h = [o for o in swa if o.transfer_type == TransferType.D2H][0]
    swa_h2disk = [o for o in swa if o.transfer_type == TransferType.H2DISK][0]
    assert swa_d2h.op_id in swa_h2disk.predecessors, "H2DISK must depend on SWA D2H"
    full_d2h = next(o for o in _full_ops(graph)
                    if o.transfer_type == TransferType.D2H)
    full_h2disk = next(o for o in _full_ops(graph)
                       if o.transfer_type == TransferType.H2DISK)
    barrier = graph._op_map[_e]
    assert barrier.transfer_type == TransferType.VIRTUAL
    assert barrier.predecessors == {full_d2h.op_id, swa_d2h.op_id}
    assert swa_h2disk.op_id not in barrier.predecessors

    # SSD SWA is allocated but is not mounted while write-through is in flight.
    assert eng.ssd_cache_engine.swa_pool.num_used == 1
    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
    pending = eng.ssd_cache_engine.match(seq)
    assert pending.num_ready_matched_blocks == 0
    assert pending.last_node.swa_host_slot == -1
    assert pending.swa_hit_blocks == 0

    op_cb[full_d2h.op_id]()
    op_cb[full_h2disk.op_id]()
    full_only = eng.ssd_cache_engine.match(seq)
    assert full_only.num_ready_matched_blocks == 4
    assert full_only.swa_hit_blocks == 0

    op_cb[swa_d2h.op_id]()
    op_cb[swa_h2disk.op_id]()
    for op_id, callback in op_cb.items():
        if op_id not in {
            full_d2h.op_id,
            full_h2disk.op_id,
            swa_d2h.op_id,
            swa_h2disk.op_id,
        }:
            callback()
    cb()
    ready = eng.ssd_cache_engine.match(seq)
    assert ready.swa_hit_blocks == 4
    assert ready.last_swa_node is not None


def test_get_ssd_staging_when_only_ssd_has_swa():
    """GET where the SWA window lives ONLY on SSD (CPU SWA evicted): the load
    must stage SSD->CPU (SWA DISK2H) then CPU->GPU (SWA H2D), mirroring full-KV
    fragment2. CPU-only code returns no SSD slot -> no DISK2H -> FAIL."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    tok = _tokens(4, base=22)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    # store to both tiers, complete, then drop the CPU SWA slot so only SSD holds it
    _pg, _rm, pcb, pop, _pe = eng.put(1, tok, mask, sm, dp_client_id=0)
    _complete(pop, pcb)
    # evict the CPU SWA (SWA-only eviction) so the CPU tier no longer matches it
    eng.cpu_cache_engine._evict_swa_slots(eng.cpu_cache_engine.swa_pool.num_used)

    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
    cpu_hit = eng.cpu_cache_engine.match(seq).swa_hit_blocks
    assert cpu_hit == 0, "precondition: CPU SWA must be gone"
    ssd_hit = eng.ssd_cache_engine.match(seq).swa_hit_blocks
    assert ssd_hit > 0, "precondition: SSD SWA must still hold the window"

    graph, _rm2, gcb, gop, _ge = eng.get(
        request_id=2, token_ids=tok, token_mask=mask, slot_mapping=sm,
        dp_client_id=0, swa_aware=True)
    swa = _swa_ops(graph)
    kinds = sorted(o.transfer_type.name for o in swa)
    assert "H2D" in kinds and "DISK2H" in kinds, (
        f"SSD staging chain missing (CPU-only bug): {kinds}")
    swa_h2d = [o for o in swa if o.transfer_type == TransferType.H2D][0]
    swa_disk2h = [o for o in swa if o.transfer_type == TransferType.DISK2H][0]
    assert swa_disk2h.op_id in swa_h2d.predecessors, "H2D must depend on SSD DISK2H"
    _complete(gop, gcb)


def test_get_ssd_staging_failure_does_not_report_fullkv_hit(monkeypatch):
    """An SWA-aware GET must fail closed when no SWA restore can be staged."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    tok = _tokens(4, base=42)
    cpu_match, ssd_match = _seed_long_ssd_short_cpu_hit(eng, tok)
    cpu_blocks = cpu_match.physical_blocks[:2].copy()
    ssd_swa_node = ssd_match.last_swa_node

    monkeypatch.setattr(
        eng.cpu_cache_engine,
        "_alloc_swa_slot",
        lambda protected_node=None: -1,
    )
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)
    graph, return_mask, cb, op_cb, end_id = eng.get(
        request_id=3,
        token_ids=tok,
        token_mask=mask,
        slot_mapping=slot_mapping,
        dp_client_id=0,
        swa_aware=True,
    )

    assert end_id == -1
    assert not return_mask.any()
    assert graph.num_ops == 0
    assert op_cb == {}
    assert ssd_swa_node.swa_lock_ref == 0, "failed source pin leaked"
    cpu_after = eng.cpu_cache_engine.match(
        SequenceMeta(token_ids=tok, tokens_per_block=TPB))
    assert cpu_after.num_ready_matched_blocks == 2
    assert cpu_after.swa_hit_blocks == 2
    assert np.array_equal(cpu_after.physical_blocks[:2], cpu_blocks)
    cb()


def test_get_ssd_staging_protects_referenced_cpu_fullkv(monkeypatch):
    """SWA staging pressure must not recycle CPU Full-KV used by this GET."""
    cc = _cache_config_ssd()
    cc.swa.num_slots = 1
    eng = GlobalCacheEngine(cc, _model_config())
    tok = _tokens(4, base=44)
    cpu_match, _ssd_match = _seed_long_ssd_short_cpu_hit(eng, tok)
    cpu_blocks = cpu_match.physical_blocks[:2].copy()

    seen_protected_nodes = []
    original_alloc = eng.cpu_cache_engine._alloc_swa_slot

    def recording_alloc(protected_node=None):
        seen_protected_nodes.append(protected_node)
        return original_alloc(protected_node=protected_node)

    monkeypatch.setattr(
        eng.cpu_cache_engine, "_alloc_swa_slot", recording_alloc)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)
    graph, return_mask, cb, op_cb, _end_id = eng.get(
        request_id=3,
        token_ids=tok,
        token_mask=mask,
        slot_mapping=slot_mapping,
        dp_client_id=0,
        swa_aware=True,
    )

    assert return_mask.all()
    assert len(seen_protected_nodes) == 1
    assert seen_protected_nodes[0] is not None
    assert seen_protected_nodes[0].size() == 2
    swa_kinds = {op.transfer_type for op in _swa_ops(graph)}
    assert swa_kinds == {TransferType.DISK2H, TransferType.H2D}
    full_h2d = next(
        op for op in _full_ops(graph) if op.transfer_type == TransferType.H2D)
    assert np.array_equal(full_h2d.src_block_ids[:2], cpu_blocks)

    # The staging allocation evicted the CPU leaf's SWA, but the protected leaf
    # and its Full-KV blocks must still exist until the GET finishes.
    cpu_during_get = eng.cpu_cache_engine.match(
        SequenceMeta(token_ids=tok, tokens_per_block=TPB))
    assert cpu_during_get.num_ready_matched_blocks == 2
    assert np.array_equal(cpu_during_get.physical_blocks[:2], cpu_blocks)
    _complete(op_cb, cb)


def test_swa_aware_get_uses_exact_source_for_final_usable_end():
    """CPU may have a shorter SWA hit than SSD. The SWA-aware clamp must choose
    the final usable end and source together; a shorter CPU hit must not shadow
    the longer SSD source selected for the Full-KV window."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    tok = _tokens(4, base=24)
    tok_div = np.concatenate([tok[:2 * TPB], _tokens(2, base=25)])
    mask4 = np.ones_like(tok, dtype=np.int64)
    sm4 = np.arange(tok.shape[0], dtype=np.int64)

    # Full 4-block store reaches CPU+SSD, then remove only CPU SWA.
    _pg, _rm, pcb, pop, _pe = eng.put(1, tok, mask4, sm4, dp_client_id=0)
    _complete(pop, pcb)
    eng.cpu_cache_engine._evict_swa_slots(eng.cpu_cache_engine.swa_pool.num_used)

    # Split the CPU radix at the shared 2-block prefix with a divergent sequence.
    eng.cache_config.enable_ssd = False
    _pg2, _rm2, pcb2, pop2, _pe2 = eng.put(2, tok_div, mask4, sm4, dp_client_id=0)
    _complete(pop2, pcb2)
    eng.cache_config.enable_ssd = True
    seq_prefix = SequenceMeta(token_ids=tok[:2 * TPB], tokens_per_block=TPB)
    prefix_node = eng.cpu_cache_engine.match(seq_prefix).last_node
    prefix_slot = eng.cpu_cache_engine._alloc_swa_slot()
    eng.cpu_cache_engine.index.set_swa(prefix_node, int(prefix_slot))

    graph, return_mask, gcb, gop, _ge = eng.get(
        request_id=3,
        token_ids=tok,
        token_mask=mask4.copy(),
        slot_mapping=sm4,
        dp_client_id=0,
        swa_aware=True,
    )

    assert return_mask[:4 * TPB].all()
    swa = _swa_ops(graph)
    kinds = sorted(o.transfer_type.name for o in swa)
    assert "H2D" in kinds and "DISK2H" in kinds, (
        f"expected SSD exact source for 4-block usable end, got {kinds}")
    _complete(gop, gcb)


def test_swa_source_selection_skips_hits_past_request_end():
    eng = GlobalCacheEngine(_cache_config_ssd(True), _model_config())
    too_deep = SimpleNamespace(
        swa_hit_blocks=4,
        last_swa_node=SimpleNamespace(swa_host_slot=3),
    )
    usable = SimpleNamespace(
        swa_hit_blocks=2,
        last_swa_node=SimpleNamespace(swa_host_slot=7),
    )

    usable_end, source = eng._select_swa_read_source(
        block_mask_start=0,
        block_mask_end=2,
        tier_match_results={
            DeviceType.CPU: too_deep,
            DeviceType.SSD: usable,
        },
    )

    assert usable_end == 2
    assert source.found
    assert source.device_type == DeviceType.SSD
    assert source.host_slot == 7


def test_get_prefers_cpu_when_both_tiers_have_swa():
    """Tier priority CPU>SSD: when CPU still holds the SWA window, GET sources
    from CPU (plain H2D, no staging) even though SSD also has it."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    tok = _tokens(4, base=23)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _pg, _rm, pcb, pop, _pe = eng.put(1, tok, mask, sm, dp_client_id=0)
    _complete(pop, pcb)

    graph, _rm2, gcb, gop, _ge = eng.get(
        request_id=2, token_ids=tok, token_mask=mask, slot_mapping=sm,
        dp_client_id=0, swa_aware=True)
    swa = _swa_ops(graph)
    kinds = sorted(o.transfer_type.name for o in swa)
    assert "H2D" in kinds, kinds
    assert "DISK2H" not in kinds, f"CPU-resident SWA must not stage from SSD: {kinds}"
    _complete(gop, gcb)


def test_multitier_match_promotes_swa_in_each_tier():
    """Multi-tier heat parity: one full-KV match promotes the matched SWA copy in
    EVERY tier that holds it (CPU AND SSD), so a reused prefix survives SWA
    eviction over a never-reused one independently per tier — mirroring how
    full-KV match_prefix(update_cache_info=True) bumps each tier's heat.

    Store A then B (distinct prefixes) to both tiers with SWA -> per-tier SWA-LRU
    order tail->head = A, B. A real multi-tier match of A must promote A to MRU in
    BOTH tiers, so a single SWA eviction per tier drops B and keeps A."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    tok_a = _tokens(4, base=40)
    tok_b = _tokens(4, base=41)
    for i, tok in enumerate((tok_a, tok_b)):
        m = np.ones_like(tok, dtype=np.int64)
        sm = np.arange(tok.shape[0], dtype=np.int64)
        _pg, _rm, pcb, pop, _pe = eng.put(i + 1, tok, m, sm, dp_client_id=0)
        _complete(pop, pcb)

    # A real match of A on each tier (match() -> match_prefix(update_cache_info=True)).
    seq_a = SequenceMeta(token_ids=tok_a, tokens_per_block=TPB); seq_a.gen_hashes()
    eng.cpu_cache_engine.match(seq_a)
    eng.ssd_cache_engine.match(seq_a)

    # Evict one SWA slot per tier: B (never reused) must go, A (reused) survives.
    eng.cpu_cache_engine._evict_swa_slots(1)
    eng.ssd_cache_engine._evict_swa_slots(1)

    seq_b = SequenceMeta(token_ids=tok_b, tokens_per_block=TPB); seq_b.gen_hashes()
    for name, engine in (("cpu", eng.cpu_cache_engine), ("ssd", eng.ssd_cache_engine)):
        a_hit = engine.match(seq_a).swa_hit_blocks
        b_hit = engine.match(seq_b).swa_hit_blocks
        assert a_hit == 4, f"{name}: reused SWA A was evicted (tier not promoted)"
        assert b_hit == 0, f"{name}: never-reused SWA B should have been evicted"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
