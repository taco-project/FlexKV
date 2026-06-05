"""SWA wiring on CacheEngineAccel (the default, index_accel=1 path).

Verifies the node-attached SWA support that was moved onto CacheEngineAccel:
  - init_swa() creates a node-attached SWAProductionManager;
  - put a snapshot on a real CRadixNode returned by insert(), read it back;
  - take()/evict() drains the evicted node's SWA slot back to the pool (no leak).

Requires a real flexkv.c_ext (CRadixTreeIndex). Skips otherwise.
"""
import numpy as np
import pytest

c_ext = pytest.importorskip("flexkv.c_ext")
if "swa_host_slot" not in dir(c_ext.CRadixNode):
    pytest.skip("CRadixNode SWA bindings not present (rebuild c_ext)",
                allow_module_level=True)

from flexkv.cache.cache_engine import CacheEngineAccel
from flexkv.common.transfer import DeviceType
from flexkv.common.block import SequenceMeta
from flexkv.common.config import SWAPoolConfig

TPB = 4


def _engine(num_blocks=64):
    return CacheEngineAccel(
        DeviceType.CPU, num_total_blocks=num_blocks, tokens_per_block=TPB,
        evict_ratio=0.0, hit_reward_seconds=0, evict_start_threshold=1.0,
        eviction_policy="lru",
    )


def _swa_config(num_slots):
    return SWAPoolConfig(
        enabled=True, num_slots=num_slots, window_size=4, num_swa_layers=2,
        bytes_per_token_per_layer=8, evict_ratio=0.25, pin_memory=False,
    )


def _seq(token_ids):
    s = SequenceMeta(token_ids=np.asarray(token_ids, dtype=np.int64), tokens_per_block=TPB)
    s.gen_hashes()
    return s


def _insert(engine, token_ids):
    """Allocate blocks and insert; return the leaf CRadixNode."""
    seq = _seq(token_ids)
    mr = engine.match(seq)
    num_new = seq.num_blocks - mr.num_matched_blocks
    if num_new <= 0:
        return mr.last_node
    phys = engine.take(num_new, strict=True)
    # take() returns an np.ndarray and insert() expects one (it calls
    # torch.from_numpy() internally), so pass it through unchanged.
    return engine.insert(seq, phys,
                         num_insert_blocks=seq.num_blocks, is_ready=True, match_result=mr)


def _data(cfg, value):
    return np.full(cfg.slot_size_bytes, value, dtype=np.uint8)


def test_init_swa_and_roundtrip():
    eng = _engine()
    eng.init_swa(_swa_config(8))
    assert eng.swa_manager is not None

    node = _insert(eng, np.arange(0, TPB * 3, dtype=np.int64))
    assert node is not None

    assert eng.swa_manager.put(node, _data(eng.swa_manager.config, 77)) is True
    assert eng.swa_manager.has(node) is True
    out = eng.swa_manager.get(node)
    assert out is not None and int(out[0]) == 77


def test_take_drains_swa_slot_no_leak():
    # Small block pool so take() must evict; SWA pool large enough to hold all.
    eng = _engine(num_blocks=6)
    eng.init_swa(_swa_config(16))

    nodes = []
    for i in range(3):
        toks = np.arange(1000 * (i + 1), 1000 * (i + 1) + TPB * 2, dtype=np.int64)
        n = _insert(eng, toks)
        assert eng.swa_manager.put(n, _data(eng.swa_manager.config, i)) is True
        nodes.append(n)

    used_before = eng.swa_manager.pool.num_used
    assert used_before == 3

    # Force eviction by requesting more blocks than free -> some nodes deleted,
    # their SWA slots recorded and drained by take().
    eng.take(6, strict=False)
    # At least one node's slot must have been reclaimed via the cascade.
    assert eng.swa_manager.pool.num_used < used_before


def test_swa_only_eviction_keeps_full_kv():
    # SWA pool smaller than node count -> SWA-only eviction (tombstone) kicks in,
    # but full KV (the tree nodes) stays intact.
    eng = _engine(num_blocks=64)
    eng.init_swa(_swa_config(2))

    nodes, seqs = [], []
    for i in range(3):
        toks = np.arange(2000 * (i + 1), 2000 * (i + 1) + TPB * 2, dtype=np.int64)
        n = _insert(eng, toks)
        assert eng.swa_manager.put(n, _data(eng.swa_manager.config, i)) is True
        nodes.append(n); seqs.append(toks)

    # First node should have been tombstoned (pool held only 2 slots).
    assert nodes[0].swa_tombstone is True
    assert nodes[0].swa_host_slot == -1
    # But its full KV is still matchable in the tree.
    mr = eng.match(_seq(seqs[0]))
    assert mr.num_matched_blocks == 2
    assert eng.swa_manager.has(nodes[0]) is False
    assert eng.swa_manager.has(nodes[2]) is True
