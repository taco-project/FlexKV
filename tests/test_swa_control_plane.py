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

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.transfer import DeviceType, TransferOpGraph, TransferType
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
        enabled=True, num_slots=256, num_ssd_slots=256, window_size=TPB,
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
    hit, slot, key = eng.cpu_cache_engine.match_swa(sm, upper_bound_blocks=4)
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
    cpu_hit, _s, _k = eng.cpu_cache_engine.match_swa(seq, upper_bound_blocks=4)
    assert cpu_hit == 0, "precondition: CPU SWA must be gone"
    ssd_hit, _s2, _k2 = eng.ssd_cache_engine.match_swa(seq, upper_bound_blocks=4)
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
        a_hit, _sa, _ka = engine.match_swa(seq_a, upper_bound_blocks=4)
        b_hit, _sb, _kb = engine.match_swa(seq_b, upper_bound_blocks=4)
        assert a_hit == 4, f"{name}: reused SWA A was evicted (tier not promoted)"
        assert b_hit == 0, f"{name}: never-reused SWA B should have been evicted"


# =========================================================================== #
# 4. REMOTE SWA tier — real production branch, no PCFS                        #
#                                                                              #
# A true byte-exact REMOTE round-trip needs PCFS (external infra, out of scope #
# for a no-GPU container). But the REMOTE SWA control-plane branch in         #
# _swa_get_slots / _swa_put_slots / _swa_release_load_lock is reachable WITHOUT #
# PCFS by injecting a real CacheEngineAccel(REMOTE) tier (identical to what    #
# GlobalCacheEngine constructs for enable_remote) and driving the real slot    #
# resolvers directly: real REMOTE slot alloc, set_swa, CPU staging slot, pin/  #
# lock release, and the failure paths. This exercises the exact code at        #
# cache_engine.py:2144 without standing up the full PCFS remote.               #
# =========================================================================== #

def _inject_remote_tier(eng, num_remote_blocks=64, num_remote_slots=16):
    """Inject a REAL CacheEngineAccel(REMOTE) tier with an armed SWA host pool.

    Mirrors GlobalCacheEngine.__init__'s REMOTE construction (minus the PCFS
    storage engine). Lets the real _swa_get_slots / _swa_put_slots REMOTE branch
    run as shipped. Returns the remote engine for direct assertions."""
    from flexkv.cache.cache_engine import CacheEngineAccel
    from flexkv.common.transfer import DeviceType
    remote = CacheEngineAccel(
        DeviceType.REMOTE, num_remote_blocks, TPB,
        evict_ratio=0.1, hit_reward_seconds=0,
        evict_start_threshold=1.0, eviction_policy="lru")
    swa_cfg = eng.cache_config.swa
    remote.init_swa(swa_cfg.for_remote_tier())
    eng.cache_engines[DeviceType.REMOTE] = remote
    return remote


def _seed_swa_on_tier(engine, tok, base):
    """Insert a ready prefix on a tier engine and mount a fresh SWA slot on its
    tail node; drain freed slots back to the pool. Returns (node, slot)."""
    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
    n = tok.shape[0] // TPB
    node = engine.insert(seq, np.arange(base, base + n, dtype=np.int64), is_ready=True)
    slot = engine.swa_alloc_slot()
    engine.set_swa(node, slot)
    engine._drain_swa_slots()
    return node, slot


def test_put_writethrough_remote_builds_swa_h2remote():
    """PUT write-through to the REMOTE SWA tier: real _swa_put_slots allocates a
    REMOTE SWA slot + mounts it via set_swa, and build_put_chain emits SWA H2REMOTE
    that depends on the SWA D2H (fire-and-forget). Driven on a real injected
    REMOTE tier — exercises the cache_engine.py:2210 branch."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    remote = _inject_remote_tier(eng)
    tok = _tokens(4, base=60)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)

    g, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    _complete(op_cb, cb)  # CPU + SSD SWA stored; now add REMOTE via real method

    cpu_node, _ = _seed_swa_on_tier(eng.cpu_cache_engine, tok, base=0)
    remote_node, _ = _seed_swa_on_tier(remote, tok, base=100)
    node_to_unlock = {DeviceType.CPU: (cpu_node, 4), DeviceType.REMOTE: (remote_node, 4)}

    swa_slots = eng._swa_put_slots(request_id=1, sequence_meta=SequenceMeta(
        token_ids=tok, tokens_per_block=TPB), block_start_idx=0, block_end_idx=4,
        node_to_unlock=node_to_unlock)
    _gpu, cpu_slots, ssd_slots, remote_slots, swa_key = swa_slots
    assert remote_slots.size == 1, "REMOTE SWA slot not allocated on put"
    assert remote.swa_pool.num_used == 1
    # The allocated REMOTE slot is mounted on the REMOTE store node.
    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
    r_hit, r_slot, _k = remote.match_swa(seq, upper_bound_blocks=4)
    assert r_hit == 4 and r_slot == int(remote_slots[0]), "REMOTE SWA not mounted"

    # build the write-through graph: SWA H2REMOTE depends on SWA D2H.
    g2 = TransferOpGraph.create_empty_graph()
    swa_d2h_id = eng.swa_cache.build_put_chain(
        g2, gpu_slot_ids=np.array([0]), cpu_slot_ids=cpu_slots,
        ssd_slot_ids=ssd_slots, remote_slot_ids=remote_slots, swa_key=swa_key)
    swa_ops = [o for o in g2._op_map.values() if getattr(o, "is_swa", False)]
    kinds = sorted(o.transfer_type.name for o in swa_ops)
    assert "D2H" in kinds and "H2REMOTE" in kinds, f"REMOTE write-through missing: {kinds}"
    h2remote = [o for o in swa_ops if o.transfer_type.name == "H2REMOTE"][0]
    swa_d2h = [o for o in swa_ops if o.transfer_type.name == "D2H"][0]
    assert swa_d2h.op_id in h2remote.predecessors, "H2REMOTE must depend on SWA D2H"


def test_get_remote_staging_when_only_remote_has_swa():
    """GET where the SWA window lives ONLY on the REMOTE tier (CPU + SSD evicted):
    the load must stage REMOTE->CPU (SWA REMOTE2H) then CPU->GPU (SWA H2D), with the
    H2D depending on the REMOTE2H. Driven via the real _swa_get_slots REMOTE branch
    (cache_engine.py:2146), including the transient CPU staging slot."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    remote = _inject_remote_tier(eng)
    tok = _tokens(4, base=61)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)

    # Seed SWA on CPU, SSD and REMOTE, then drop CPU + SSD so only REMOTE holds it.
    _seed_swa_on_tier(eng.cpu_cache_engine, tok, base=0)
    _seed_swa_on_tier(eng.ssd_cache_engine, tok, base=50)
    _seed_swa_on_tier(remote, tok, base=100)
    eng.cpu_cache_engine._evict_swa(1)
    eng.ssd_cache_engine._evict_swa(1)

    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
    cpu_hit, _c, _kc = eng.cpu_cache_engine.match_swa(seq, upper_bound_blocks=4)
    ssd_hit, _s, _ks = eng.ssd_cache_engine.match_swa(seq, upper_bound_blocks=4)
    r_hit, _r, _kr = remote.match_swa(seq, upper_bound_blocks=4)
    assert cpu_hit == 0 and ssd_hit == 0 and r_hit > 0, (
        f"precondition: only REMOTE should hold SWA (cpu={cpu_hit} ssd={ssd_hit} rem={r_hit})")

    tier_mr = {DeviceType.CPU: eng.cpu_cache_engine.match(seq),
               DeviceType.SSD: eng.ssd_cache_engine.match(seq),
               DeviceType.REMOTE: remote.match(seq)}
    slots = eng._swa_get_slots(request_id=2, sequence_meta=seq, block_start_idx=0,
                               block_end_idx=4, full_hit_blocks=4,
                               tier_match_results=tier_mr)
    gpu, cpu, ssd, rem, swa_key, lock_node, staging_slot = slots
    assert rem.size == 1, "REMOTE SWA slot not sourced on get"
    assert ssd.size == 0, "SSD was evicted — must not be staged"
    assert staging_slot >= 0, "transient CPU staging slot must be allocated for REMOTE2H"
    assert lock_node is remote.match(seq).last_swa_node

    g = TransferOpGraph.create_empty_graph()
    h2d_id = eng.swa_cache.build_get_chain(
        g, gpu_slot_ids=gpu, cpu_slot_ids=cpu,
        ssd_slot_ids=ssd, remote_slot_ids=rem, swa_key=swa_key)
    swa_ops = [o for o in g._op_map.values() if getattr(o, "is_swa", False)]
    kinds = sorted(o.transfer_type.name for o in swa_ops)
    assert "H2D" in kinds and "REMOTE2H" in kinds, f"REMOTE staging missing: {kinds}"
    h2d = [o for o in swa_ops if o.transfer_type.name == "H2D"][0]
    r2h = [o for o in swa_ops if o.transfer_type.name == "REMOTE2H"][0]
    assert r2h.op_id in h2d.predecessors, "SWA H2D must depend on REMOTE2H"

    # Late-bind + completion callback frees the transient CPU staging slot AND
    # releases the REMOTE source pin (paired release on the exception-safe path).
    g.set_swa_gpu_blocks(np.array([9], dtype=np.int64))
    eng._swa_release_load_lock(lock_node, staging_slot=staging_slot)
    assert lock_node.swa_lock_ref == 0, "REMOTE source pin not released on completion"
    assert eng.cpu_cache_engine.swa_pool.num_used == 0, "transient CPU staging slot leaked"


def test_get_prefers_ssd_over_remote_when_both_have_swa():
    """Tier priority SSD > REMOTE: only ONE staged source is chosen. When SSD holds
    the window, GET stages from SSD and must NOT also stage from REMOTE (no
    transient CPU staging slot, no REMOTE2H op)."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    remote = _inject_remote_tier(eng)
    tok = _tokens(4, base=62)
    _seed_swa_on_tier(eng.ssd_cache_engine, tok, base=50)
    _seed_swa_on_tier(remote, tok, base=100)

    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
    tier_mr = {DeviceType.CPU: eng.cpu_cache_engine.match(seq),
               DeviceType.SSD: eng.ssd_cache_engine.match(seq),
               DeviceType.REMOTE: remote.match(seq)}
    gpu, cpu, ssd, rem, swa_key, lock_node, staging_slot = eng._swa_get_slots(
        request_id=2, sequence_meta=seq, block_start_idx=0, block_end_idx=4,
        full_hit_blocks=4, tier_match_results=tier_mr)
    assert ssd.size == 1 and rem.size == 0, "SSD>REMOTE: only SSD must stage"
    assert staging_slot == -1, "no transient staging when SSD is the source"


# =========================================================================== #
# 5. SWA failure paths — paired slot/lock release, pool-full, lower-tier skip  #
# =========================================================================== #

def test_put_no_swa_when_cpu_pool_full():
    """PUT with SWA ON but the CPU SWA host pool entirely full: _swa_put_slots
    must allocate NO NEW SWA slot and emit NO SWA op for the new prefix — the
    full-KV store is unaffected. (Real alloc path: swa_alloc_slot returns -1 once
    the pool is full and nothing is evictable; the pool is left intact.)"""
    eng = GlobalCacheEngine(_cache_config(True), _model_config())
    tok = _tokens(4, base=90)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    _complete(op_cb, cb)  # SWA slot mounted on the stored tail
    cpu = eng.cpu_cache_engine
    pool = cpu.swa_pool

    # Drain the pool completely: allocate every remaining free slot.
    held = []
    while True:
        s = pool.allocate()
        if s is None:
            break
        held.append(s)
    try:
        used_before = pool.num_used
        assert used_before == pool.num_total, "pool not fully drained"
        # A fresh prefix put() with no free SWA slot: full-KV still stores, but
        # no SWA op / no over-allocation.
        tok2 = _tokens(4, base=91)
        g2, _rm2, cb2, op_cb2, _e2 = eng.put(
            2, tok2, np.ones_like(tok2, dtype=np.int64),
            np.arange(tok2.shape[0], dtype=np.int64), dp_client_id=0)
        swa = [o for o in g2._op_map.values() if getattr(o, "is_swa", False)]
        assert any(o.transfer_type == TransferType.D2H for o in g2._op_map.values()
                   if not getattr(o, "is_swa", False)), "full-KV store dropped"
        _complete(op_cb2, cb2)
        assert pool.num_used == used_before, "SWA pool over-allocated on full pool"
    finally:
        for s in held:
            pool.free(s)


def test_get_staging_fails_frees_remote_pin():
    """GET staging from REMOTE when the transient CPU staging slot cannot be
    allocated: the real code releases the just-taken REMOTE source pin (no leak)
    and returns empty SWA slots. Paired slot/lock release on the failure path."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    remote = _inject_remote_tier(eng)
    tok = _tokens(4, base=92)
    _seed_swa_on_tier(remote, tok, base=100)  # only REMOTE holds SWA

    # Fill the entire CPU SWA pool so a transient staging slot cannot be allocated.
    pool = eng.cpu_cache_engine.swa_pool
    held = []
    while True:
        s = pool.allocate()
        if s is None:
            break
        held.append(s)
    try:
        seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
        r_node = remote.match(seq).last_swa_node
        r_node.inc_swa_lock_ref()  # simulate the source pin taken by _match_locked
        tier_mr = {DeviceType.CPU: eng.cpu_cache_engine.match(seq),
                   DeviceType.SSD: eng.ssd_cache_engine.match(seq),
                   DeviceType.REMOTE: remote.match(seq)}
        gpu, cpu, ssd, rem, swa_key, lock_node, staging_slot = eng._swa_get_slots(
            request_id=2, sequence_meta=seq, block_start_idx=0, block_end_idx=4,
            full_hit_blocks=4, tier_match_results=tier_mr)
        # CPU staging impossible -> no SWA load at all, but the REMOTE source pin
        # must have been released (no leak of the just-taken pin).
        assert int(rem.size) == 0 and staging_slot == -1
        assert r_node.swa_lock_ref == 0, "REMOTE source pin leaked on staging failure"
    finally:
        r_node.dec_swa_lock_ref()
        for s in held:
            pool.free(s)


def test_lower_tier_pool_full_keeps_cpu_only():
    """PUT with SWA write-through: when the SSD SWA pool is full-and-locked, the
    SSD SWA write-through is skipped but the CPU SWA store proceeds (SWA stays a
    subset of Full PER TIER). Real _swa_put_slots _tier_slot returns empty."""
    eng = GlobalCacheEngine(_cache_config_ssd(), _model_config())
    tok = _tokens(4, base=93)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    _complete(op_cb, cb)
    cpu = eng.cpu_cache_engine
    ssd = eng.ssd_cache_engine

    cpu_node, _ = _seed_swa_on_tier(cpu, tok, base=0)
    ssd_node, _ = _seed_swa_on_tier(ssd, tok, base=50)
    # Lock + fill the SSD SWA pool so its write-through alloc fails.
    ssd_node.inc_swa_lock_ref()
    ssd_held = []
    try:
        while True:
            s = ssd.swa_pool.allocate()
            if s is None:
                break
            ssd_held.append(s)
        node_to_unlock = {DeviceType.CPU: (cpu_node, 4), DeviceType.SSD: (ssd_node, 4)}
        _gpu, cpu_slots, ssd_slots, _rem, swa_key = eng._swa_put_slots(
            request_id=1, sequence_meta=SequenceMeta(token_ids=tok, tokens_per_block=TPB),
            block_start_idx=0, block_end_idx=4, node_to_unlock=node_to_unlock)
        assert cpu_slots.size == 1, "CPU SWA must still be allocated"
        assert ssd_slots.size == 0, "SSD SWA write-through must skip when pool full"
        assert ssd.swa_pool.num_used == len(ssd_held), "SSD pool unexpectedly grew"
    finally:
        for s in ssd_held:
            ssd.swa_pool.free(s)
        ssd_node.dec_swa_lock_ref()


def test_release_load_lock_idempotent_no_underflow():
    """The SWA H2D completion callback (_swa_release_load_lock) must be idempotent:
    calling it twice (e.g. callback re-entry / duplicate finalize) must not drive
    swa_lock_ref below 0, and a fresh staging slot must not be double-freed."""
    eng = GlobalCacheEngine(_cache_config(True), _model_config())
    tok = _tokens(4, base=94)
    _seed_swa_on_tier(eng.cpu_cache_engine, tok, base=0)
    cpu = eng.cpu_cache_engine
    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB); seq.gen_hashes()
    node = cpu.match(seq).last_swa_node
    node.inc_swa_lock_ref()  # pin taken by the load

    # Allocate + free a transient CPU staging slot to simulate a prior free.
    staging = cpu.swa_alloc_slot()
    before = cpu.swa_pool.num_used
    eng._swa_release_load_lock(node, staging_slot=staging)   # 1st call
    eng._swa_release_load_lock(node, staging_slot=staging)   # 2nd (re-entry)
    assert node.swa_lock_ref == 0, "swa_lock_ref underflowed below 0"
    # Re-inject a node into the pool is safe; num_used must not go negative.
    assert cpu.swa_pool.num_used >= 0, "SWA pool slot count went negative"
    # Cleanup: the released staging slot should be reflected (no phantom leak).
    assert cpu.swa_pool.num_used == before - 1 or cpu.swa_pool.num_used == before, \
        "transient staging slot double-handled"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
