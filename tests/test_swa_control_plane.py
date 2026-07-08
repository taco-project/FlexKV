# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-plane SWA: graph build (GlobalCacheEngine.put/get) + launch-time
late-bind of the SWA GPU slot.

Enters from the top (GlobalCacheEngine), NOT the data plane, using the REAL
(de-stubbed) SWA slot sources — the control plane's job is to turn a request
into a Full+SWA transfer graph + masks against the node-mounted radix tree:

    put()       -> full-KV D2H graph + SWA peer D2H (alloc slot + set_swa)
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
import numpy as np
import pytest
import torch

pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import CacheEngineAccel, GlobalCacheEngine
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


def _cache_config_small_swa(num_slots: int):
    cc = _cache_config()
    cc.swa = SWAPoolConfig(
        enabled=True,
        num_slots=num_slots,
        num_swa_layers=1,
        bytes_per_token_per_layer=64,
    )
    return cc


def _cache_config_ssd(enable_swa_transfer: bool = True):
    """CPU + SSD tiers, both with an SWA host pool. The SSD cache engine is an
    in-memory radix+mempool+swa_pool (no real SSD files at construction — files
    are only touched by the data-plane worker at transfer time), so multi-tier
    SWA orchestration (_swa_put_slots / _swa_get_slots) is testable at smoke
    level without disk I/O."""
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


def _cache_config_small_swa_ssd(num_slots: int, num_ssd_slots: int):
    cc = _cache_config_ssd()
    cc.swa = SWAPoolConfig(
        enabled=True,
        num_slots=num_slots,
        num_ssd_slots=num_ssd_slots,
        num_swa_layers=1,
        bytes_per_token_per_layer=64,
    )
    return cc


def _cache_config_remote(enable_swa_transfer: bool = True):
    """CPU + REMOTE tiers using the local REMOTE cache engine.

    This exercises the production REMOTE control-plane graph and SWA slot
    lifecycle without requiring a PCFS server; byte movement remains covered by
    explicitly gated E2E environments.
    """
    cc = CacheConfig(
        tokens_per_block=TPB,
        enable_cpu=True,
        enable_remote=True,
        enable_3rd_remote=False,
        enable_kv_sharing=False,
        num_cpu_blocks=4096,
        num_remote_blocks=4096,
    )
    cc.enable_remote = True
    cc.enable_kv_sharing = False
    cc.swa = SWAPoolConfig(
        enabled=True,
        num_slots=256,
        num_remote_slots=256,
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
    _complete(op_cb, cb)
    # after completion the SWA slot is mounted on the stored tail node.
    sm = SequenceMeta(token_ids=tok, tokens_per_block=TPB); sm.gen_hashes()
    hit, slot = eng.cpu_cache_engine.match_swa(sm, upper_bound_blocks=4)
    assert hit == 4 and slot >= 0


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
    hit, slot, node = eng.cpu_cache_engine.match_swa_locked(sm, upper_bound_blocks=4)
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


# =========================================================================== #
# 3. multi-tier SWA orchestration (CPU + SSD): write-through + get staging     #
#    These exercise _swa_put_slots / _swa_get_slots across tiers (the layer    #
#    that was CPU-only). Written TDD-first: they FAIL against CPU-only code.    #
# =========================================================================== #

def test_put_writethrough_ssd_builds_swa_h2disk():
    """PUT with an SSD tier: SWA store must write through to SSD, mirroring the
    full-KV H2DISK. The SWA graph should carry a SWA D2H (GPU->CPU) AND a SWA
    H2DISK (CPU->SSD) that depends on the D2H (fire-and-forget), and an SSD SWA
    slot must be allocated + mounted on the SSD store node."""
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
    # an SSD SWA slot was allocated (mounted on the SSD store node)
    assert eng.ssd_cache_engine.swa_pool.num_used == 1
    _complete(op_cb, cb)


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
    eng.cpu_cache_engine._evict_swa(eng.cpu_cache_engine.swa_pool.num_used)

    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
    cpu_hit, _s = eng.cpu_cache_engine.match_swa(seq, upper_bound_blocks=4)
    assert cpu_hit == 0, "precondition: CPU SWA must be gone"
    ssd_hit, _s2 = eng.ssd_cache_engine.match_swa(seq, upper_bound_blocks=4)
    assert ssd_hit > 0, "precondition: SSD SWA must still hold the window"

    graph, _rm2, gcb, gop, _ge = eng.get(
        request_id=2, token_ids=tok, token_mask=mask, slot_mapping=sm, dp_client_id=0)
    swa = _swa_ops(graph)
    kinds = sorted(o.transfer_type.name for o in swa)
    assert "H2D" in kinds and "DISK2H" in kinds, (
        f"SSD staging chain missing (CPU-only bug): {kinds}")
    swa_h2d = [o for o in swa if o.transfer_type == TransferType.H2D][0]
    swa_disk2h = [o for o in swa if o.transfer_type == TransferType.DISK2H][0]
    assert swa_disk2h.op_id in swa_h2d.predecessors, "H2D must depend on SSD DISK2H"
    _complete(gop, gcb)


def test_put_writethrough_remote_builds_swa_h2remote():
    """REMOTE write-through: SWA store allocates a REMOTE SWA slot and emits
    H2REMOTE dependent on the SWA D2H, mirroring full-KV remote write-through."""
    eng = GlobalCacheEngine(_cache_config_remote(), _model_config())
    tok = _tokens(4, base=24)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)

    graph, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    swa = _swa_ops(graph)
    kinds = sorted(o.transfer_type.name for o in swa)
    assert "D2H" in kinds, kinds
    assert "H2REMOTE" in kinds, f"SWA remote write-through missing: {kinds}"
    swa_d2h = [o for o in swa if o.transfer_type == TransferType.D2H][0]
    swa_h2remote = [o for o in swa if o.transfer_type == TransferType.H2REMOTE][0]
    assert swa_d2h.op_id in swa_h2remote.predecessors
    assert eng.remote_cache_engine.swa_pool.num_used == 1
    _complete(op_cb, cb)


def test_get_remote_staging_when_only_remote_has_swa():
    """REMOTE-sourced GET stages REMOTE->CPU then CPU->GPU and releases both
    source pin and transient CPU staging slot on the H2D completion callback."""
    eng = GlobalCacheEngine(_cache_config_remote(), _model_config())
    tok = _tokens(4, base=25)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _pg, _rm, pcb, pop, _pe = eng.put(1, tok, mask, sm, dp_client_id=0)
    _complete(pop, pcb)
    eng.cpu_cache_engine._evict_swa(eng.cpu_cache_engine.swa_pool.num_used)

    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
    cpu_hit, _ = eng.cpu_cache_engine.match_swa(seq, upper_bound_blocks=4)
    remote_hit, _ = eng.remote_cache_engine.match_swa(seq, upper_bound_blocks=4)
    assert cpu_hit == 0
    assert remote_hit > 0
    before_cpu_used = eng.cpu_cache_engine.swa_pool.num_used

    graph, _rm2, gcb, gop, _ge = eng.get(
        request_id=2, token_ids=tok, token_mask=mask, slot_mapping=sm, dp_client_id=0)
    swa = _swa_ops(graph)
    kinds = sorted(o.transfer_type.name for o in swa)
    assert "H2D" in kinds and "REMOTE2H" in kinds, kinds
    swa_h2d = [o for o in swa if o.transfer_type == TransferType.H2D][0]
    swa_remote2h = [o for o in swa if o.transfer_type == TransferType.REMOTE2H][0]
    assert swa_remote2h.op_id in swa_h2d.predecessors
    assert eng.cpu_cache_engine.swa_pool.num_used == before_cpu_used + 1

    remote_mr = eng.remote_cache_engine.match(seq)
    remote_node = getattr(remote_mr, "last_swa_node", None)
    assert remote_node is not None
    assert remote_node.swa_lock_ref == 1

    _complete(gop, gcb)
    assert remote_node.swa_lock_ref == 0
    assert eng.cpu_cache_engine.swa_pool.num_used == before_cpu_used


def test_remote_source_pin_released_when_cpu_staging_slot_unavailable():
    """If a lower-tier SWA hit is pinned but no CPU staging slot is available,
    _swa_get_slots must drop the source pin and emit no SWA ops."""
    eng = GlobalCacheEngine(_cache_config_remote(), _model_config())
    tok = _tokens(4, base=26)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _pg, _rm, pcb, pop, _pe = eng.put(1, tok, mask, sm, dp_client_id=0)
    _complete(pop, pcb)
    eng.cpu_cache_engine._evict_swa(eng.cpu_cache_engine.swa_pool.num_used)
    held_slots = []
    while True:
        slot = eng.cpu_cache_engine.swa_alloc_slot()
        if slot < 0:
            break
        held_slots.append(slot)
    assert eng.cpu_cache_engine.swa_pool.num_free == 0

    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
    remote_hit, _ = eng.remote_cache_engine.match_swa(seq, upper_bound_blocks=4)
    assert remote_hit > 0

    graph, _rm2, gcb, gop, _ge = eng.get(
        request_id=2, token_ids=tok, token_mask=mask, slot_mapping=sm, dp_client_id=0)
    assert _swa_ops(graph) == []
    remote_hit2, _slot, remote_node = eng.remote_cache_engine.match_swa_locked(
        seq, upper_bound_blocks=4)
    assert remote_hit2 > 0 and remote_node is not None
    assert remote_node.swa_lock_ref == 1
    remote_node.dec_swa_lock_ref()
    _complete(gop, gcb)
    for slot in held_slots:
        eng.cpu_cache_engine.swa_pool.free(slot)


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
        request_id=2, token_ids=tok, token_mask=mask, slot_mapping=sm, dp_client_id=0)
    swa = _swa_ops(graph)
    kinds = sorted(o.transfer_type.name for o in swa)
    assert "H2D" in kinds, kinds
    assert "DISK2H" not in kinds, f"CPU-resident SWA must not stage from SSD: {kinds}"
    _complete(gop, gcb)


def test_put_skips_swa_when_all_cpu_swa_nodes_are_locked():
    """When every CPU SWA slot is locked and no eviction is possible, PUT should
    leave full-KV store intact but emit no SWA store ops."""
    eng = GlobalCacheEngine(_cache_config_small_swa(num_slots=3), _model_config())
    locked_nodes = []
    for req in range(1, eng.cpu_cache_engine.swa_pool.num_slots + 1):
        tok = _tokens(4, base=1000 + req)
        _pg, _rm, cb, op_cb, _e = eng.put(
            req, tok, np.ones_like(tok, dtype=np.int64),
            np.arange(tok.shape[0], dtype=np.int64), dp_client_id=0)
        _complete(op_cb, cb)
        seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB)
        hit, _slot, node = eng.cpu_cache_engine.match_swa_locked(seq, upper_bound_blocks=4)
        assert hit == 4 and node is not None
        locked_nodes.append(node)
    assert eng.cpu_cache_engine.swa_pool.num_free == 0

    tok = _tokens(4, base=2000)
    graph, _rm, cb, op_cb, _e = eng.put(
        9999, tok, np.ones_like(tok, dtype=np.int64),
        np.arange(tok.shape[0], dtype=np.int64), dp_client_id=0)
    assert any(op.transfer_type == TransferType.D2H for op in _full_ops(graph))
    assert _swa_ops(graph) == []
    _complete(op_cb, cb)
    for node in locked_nodes:
        node.dec_swa_lock_ref()


def test_put_keeps_cpu_swa_when_lower_tier_pool_full_and_locked():
    """A full locked lower-tier SWA pool should skip write-through only; the CPU
    SWA copy still gets allocated and mounted."""
    eng = GlobalCacheEngine(
        _cache_config_small_swa_ssd(num_slots=8, num_ssd_slots=3),
        _model_config(),
    )
    locked_nodes = []
    for req in range(1, eng.ssd_cache_engine.swa_pool.num_slots + 1):
        tok = _tokens(4, base=3000 + req)
        _pg, _rm, cb, op_cb, _e = eng.put(
            req, tok, np.ones_like(tok, dtype=np.int64),
            np.arange(tok.shape[0], dtype=np.int64), dp_client_id=0)
        _complete(op_cb, cb)
        seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB)
        hit, _slot, node = eng.ssd_cache_engine.match_swa_locked(seq, upper_bound_blocks=4)
        assert hit == 4 and node is not None
        locked_nodes.append(node)
    assert eng.ssd_cache_engine.swa_pool.num_free == 0

    tok = _tokens(4, base=4000)
    graph, _rm, cb, op_cb, _e = eng.put(
        9999, tok, np.ones_like(tok, dtype=np.int64),
        np.arange(tok.shape[0], dtype=np.int64), dp_client_id=0)
    kinds = sorted(op.transfer_type.name for op in _swa_ops(graph))
    assert "D2H" in kinds
    assert "H2DISK" not in kinds
    _complete(op_cb, cb)
    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB)
    cpu_hit, _slot = eng.cpu_cache_engine.match_swa(seq, upper_bound_blocks=4)
    assert cpu_hit == 4
    for node in locked_nodes:
        node.dec_swa_lock_ref()


def test_swa_load_release_callback_is_idempotent_for_staging_slot():
    """Callback re-entry must not double-free transient staging slots."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    slot = eng.cpu_cache_engine.swa_alloc_slot()
    assert slot >= 0
    free_before = eng.cpu_cache_engine.swa_pool.num_free
    cb = eng._make_swa_release_load_callback(node=None, staging_slot=slot)

    cb()
    assert eng.cpu_cache_engine.swa_pool.num_free == free_before + 1
    cb()
    assert eng.cpu_cache_engine.swa_pool.num_free == free_before + 1


def test_swa_load_release_callback_frees_staging_even_if_source_unlock_raises(monkeypatch):
    """Source unlock exceptions must not leak the transient CPU staging slot."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    slot = eng.cpu_cache_engine.swa_alloc_slot()
    assert slot >= 0
    free_before = eng.cpu_cache_engine.swa_pool.num_free

    class BadNode:
        swa_lock_ref = 1

        def dec_swa_lock_ref(self):
            raise RuntimeError("boom")

    eng._swa_release_load_lock(BadNode(), staging_slot=slot)
    assert eng.cpu_cache_engine.swa_pool.num_free == free_before + 1


def test_remote_worker_successful_launch_reports_completion(monkeypatch):
    """REMOTE worker launch_transfer must return True after a successful copy.

    The base worker only posts completed op ids when launch_transfer returns
    True. A missing return here makes REMOTE2H/H2REMOTE look hung even if the C++
    transfer completed.
    """
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

    worker_mod = pytest.importorskip("flexkv.transfer.worker")
    layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=1,
        num_block=4,
        tokens_per_block=TPB,
        num_head=1,
        head_size=64,
        is_mla=True,
    )
    worker = worker_mod.CPURemoteTransferWorker.__new__(worker_mod.CPURemoteTransferWorker)
    worker.cpu_kv_layout = layout
    worker.remote_kv_layout = layout
    worker.dtype = torch.uint8
    worker.chunk_size_in_bytes = TPB * 64
    worker.num_layers = 1
    worker.kv_dim = 1

    seen = {}

    def fake_get_transfer_block_ids(_op):
        return (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
        )

    def fake_transfer_impl(src_block_ids, dst_block_ids, transfer_type, **kwargs):
        seen["src"] = src_block_ids
        seen["dst"] = dst_block_ids
        seen["type"] = transfer_type
        seen["kwargs"] = kwargs

    monkeypatch.setattr(worker, "get_transfer_block_ids", fake_get_transfer_block_ids)
    monkeypatch.setattr(worker, "_transfer_impl", fake_transfer_impl)
    monkeypatch.setattr(worker, "_log_transfer_performance", lambda *args, **kwargs: None)

    class DummyOp:
        transfer_op_id = 123
        transfer_graph_id = 456
        transfer_type = TransferType.REMOTE2H
        valid_block_num = 1
        src_block_node_ids = np.array([7], dtype=np.int64)

    assert worker.launch_transfer(DummyOp()) is True
    assert seen["type"] == TransferType.REMOTE2H
    assert seen["kwargs"]["src_block_node_ids"].tolist() == [7]


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
    eng.cpu_cache_engine._evict_swa(1)
    eng.ssd_cache_engine._evict_swa(1)

    seq_b = SequenceMeta(token_ids=tok_b, tokens_per_block=TPB); seq_b.gen_hashes()
    for name, engine in (("cpu", eng.cpu_cache_engine), ("ssd", eng.ssd_cache_engine)):
        a_hit, _sa = engine.match_swa(seq_a, upper_bound_blocks=4)
        b_hit, _sb = engine.match_swa(seq_b, upper_bound_blocks=4)
        assert a_hit == 4, f"{name}: reused SWA A was evicted (tier not promoted)"
        assert b_hit == 0, f"{name}: never-reused SWA B should have been evicted"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
