# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWA single-pass match: SWA rides the full-KV match, no redundant second pass.

Node-mount makes ``radix.match_prefix`` return the full prefix AND the deepest
in-range SWA node in one forward pass. A SWA-aware GET therefore resolves
``usable = min(full_hit, swa_hit)`` and builds the (clamped) graph from a SINGLE
per-tier match.

These tests pin that contract at the GlobalCacheEngine layer (no GPU /
KVTaskEngine needed):

  * a SWA-aware get triggers exactly ONE round of per-tier full-KV matching;
  * the full-KV transfer is clamped to usable = min(full, swa);
  * a plain (non-SWA) get on an SWA-enabled cache is never clamped and matches once.

Requires flexkv.c_ext (production CacheEngineAccel / CRadixTreeIndex).
"""
import numpy as np
import pytest
import torch

pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.debug import flexkv_logger

flexkv_logger.set_level("OFF")

pytestmark = pytest.mark.smoke

TPB = 16


def _model_config():
    return ModelConfig(
        num_layers=4, num_kv_heads=1, head_size=128,
        use_mla=True, dtype=torch.bfloat16, tp_size=1, dp_size=1,
    )


def _cache_config():
    cc = CacheConfig(
        tokens_per_block=TPB,
        enable_cpu=True, enable_ssd=False, enable_remote=False,
        num_cpu_blocks=4096,
    )
    cc.swa = SWAPoolConfig(
        enabled=True, num_slots=256, window_size=TPB,
        num_swa_layers=1, bytes_per_token_per_layer=64,
    )
    cc.enable_swa_transfer = True
    return cc


def _tokens(n_blocks, base):
    rs = np.random.RandomState(base)
    return rs.randint(0, 30000, size=n_blocks * TPB, dtype=np.int64)


def _complete(op_cb, cb):
    for c in op_cb.values():
        c()
    cb()


def _put(eng, tok, req=1):
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(req, tok, mask, sm, dp_client_id=0)
    _complete(op_cb, cb)


class _MatchCounter:
    """Wrap the engine's per-tier match entry points and count invocations.

    CPU-only config routes through ``match_local_accel``; we also wrap
    ``match_all_accel`` so the same counter works if a tier config changes.
    Each call = one round of per-tier radix matching.
    """

    def __init__(self, eng):
        self.eng = eng
        self.n = 0
        self._orig_local = eng.match_local_accel
        self._orig_all = eng.match_all_accel

    def __enter__(self):
        def local(*a, **k):
            self.n += 1
            return self._orig_local(*a, **k)

        def allm(*a, **k):
            self.n += 1
            return self._orig_all(*a, **k)

        self.eng.match_local_accel = local
        self.eng.match_all_accel = allm
        return self

    def __exit__(self, *exc):
        self.eng.match_local_accel = self._orig_local
        self.eng.match_all_accel = self._orig_all
        return False


# =========================================================================== #
# single-pass match                                                           #
# =========================================================================== #

def test_swa_aware_get_matches_once():
    """A SWA-aware GET triggers exactly ONE round of per-tier full-KV matching.

    get(swa_aware=True) matches once, reads the SWA hit off that same match, and
    builds the clamped graph — no separate match pass.
    """
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tok = _tokens(4, base=31)
    _put(eng, tok)

    with _MatchCounter(eng) as mc:
        eng.get(request_id=2, token_ids=tok,
                token_mask=np.ones_like(tok, dtype=np.int64),
                slot_mapping=np.arange(tok.shape[0], dtype=np.int64),
                dp_client_id=0, swa_aware=True)
    assert mc.n == 1, (
        f"SWA-aware get matched {mc.n} rounds; must be exactly 1")


def test_swa_aware_get_clamps_full_to_usable():
    """The SWA-aware path clamps the full-KV transfer to usable = min(full_hit,
    swa_hit).

    With SWA on the full stored tail, full_hit == swa_hit == 4, so return_mask
    covers all 4 blocks. This pins that the clamp does not over- or under-clamp.
    """
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tok = _tokens(4, base=32)
    _put(eng, tok)

    graph, return_mask, cb, op_cb, end_id = eng.get(
        request_id=2, token_ids=tok,
        token_mask=np.ones_like(tok, dtype=np.int64),
        slot_mapping=np.arange(tok.shape[0], dtype=np.int64),
        dp_client_id=0, swa_aware=True)
    # usable = min(4, 4) = 4 -> all 4 blocks resident, SWA H2D present.
    assert int(return_mask.sum()) == 4 * TPB
    swa = [o for o in graph._op_map.values() if getattr(o, "is_swa", False)]
    assert len(swa) == 1, "SWA H2D must still attach to the same graph"
    _complete(op_cb, cb)


# =========================================================================== #
# Guardrail — the plain (non-SWA) path must be untouched                      #
# =========================================================================== #

def test_swa_aware_get_clamps_when_full_beats_swa():
    """REAL production path (eng.get(swa_aware=True)) where full_hit > swa_hit.

    Construct full=4 / swa=2 by storing two nested prefixes (SWA at block 2 and
    block 4), locking the deep tail's Full KV, promoting the shallow node, then
    evicting one SWA so the deep leaf KEEPS its Full KV but loses its SWA
    (full-locked-leaf drop). The real get() must clamp usable = min(4, 2) = 2,
    exercising _clamp_end_to_swa — not a re-implemented formula.

    The original test_swa_level2 single-pass contract is preserved: still ONE
    match round, and the SWA H2D slot still attaches (now for 2 blocks)."""
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tpb = TPB

    # PUT 4-block prefix with SWA at block 2 (shallow) and block 4 (deep tail).
    tok4 = _tokens(4, base=70)
    mask4 = np.ones_like(tok4, dtype=np.int64)
    sm4 = np.arange(tok4.shape[0], dtype=np.int64)
    eng.put(1, tok4, mask4, sm4, dp_client_id=0)
    # PUT a 2-block prefix (shares [1..2*TPB]) with its own SWA slot.
    tok2 = _tokens(2, base=71)
    mask2 = np.ones_like(tok2, dtype=np.int64)
    sm2 = np.arange(tok2.shape[0], dtype=np.int64)
    eng.put(2, tok2, mask2, sm2, dp_client_id=0)

    cpu = eng.cpu_cache_engine
    # Lock the deep tail's Full KV so its SWA-only eviction keeps the Full KV.
    seq4 = SequenceMeta(token_ids=tok4, tokens_per_block=tpb); seq4.gen_hashes()
    deep_mr = cpu.match(seq4)
    deep_node = deep_mr.last_node
    cpu.lock_node(deep_node)
    # Promote the shallow node to SWA-LRU MRU, then evict one SWA slot: the LRU
    # victim is the deep leaf (never promoted) -> it loses SWA but keeps Full.
    seq2 = SequenceMeta(token_ids=tok2, tokens_per_block=tpb); seq2.gen_hashes()
    cpu.match_swa(seq2, upper_bound_blocks=2)
    cpu._evict_swa(1)

    # Precondition: full hit still 4, SWA hit dropped to 2.
    f_hit = cpu.match(seq4).num_ready_matched_blocks
    s_hit, _s, _k = cpu.match_swa(seq4, upper_bound_blocks=4)
    assert f_hit == 4, f"precondition full hit={f_hit}"
    assert s_hit == 2, f"precondition swa hit={s_hit}"

    with _MatchCounter(eng) as mc:
        graph, return_mask, cb, op_cb, end_id = eng.get(
            request_id=3, token_ids=tok4,
            token_mask=np.ones_like(tok4, dtype=np.int64),
            slot_mapping=sm4, dp_client_id=0, swa_aware=True)
    assert mc.n == 1, f"SWA-aware get still matches once; got {mc.n} rounds"
    # usable = min(4, 2) = 2 -> only 2 blocks resident in the returned mask.
    assert int(return_mask.sum()) == 2 * tpb, (
        f"clamp to min(full,swa) failed: got {int(return_mask.sum())} tokens")
    swa = [o for o in graph._op_map.values() if getattr(o, "is_swa", False)]
    assert len(swa) == 1, "SWA H2D must still attach (now for the in-window block)"
    _complete(op_cb, cb)
    cpu.unlock(deep_node)


def test_swa_aware_get_empty_when_no_swa_hit():
    """REAL production path where swa_hit == 0 but full_hit > 0.

    Store 4 blocks with SWA, lock the tail's Full KV, then evict its SWA so the
    window survives ONLY as Full KV. The real get(swa_aware=True) must clamp
    usable = min(4, 0) = 0 -> empty Full-KV window (no stale KV fed to the
    SWA-layer attention) and NO SWA op. Driven through eng.get, not a formula."""
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tpb = TPB

    tok = _tokens(4, base=72)
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    eng.put(1, tok, mask, sm, dp_client_id=0)

    cpu = eng.cpu_cache_engine
    seq = SequenceMeta(token_ids=tok, tokens_per_block=tpb); seq.gen_hashes()
    node = cpu.match(seq).last_node
    cpu.lock_node(node)            # full-locked leaf -> SWA drop keeps Full
    cpu._evict_swa(1)             # drop the only SWA slot, keep Full KV

    f_hit = cpu.match(seq).num_ready_matched_blocks
    s_hit, _s, _k = cpu.match_swa(seq, upper_bound_blocks=4)
    assert f_hit == 4 and s_hit == 0, f"precondition full={f_hit} swa={s_hit}"

    graph, return_mask, cb, op_cb, end_id = eng.get(
        request_id=2, token_ids=tok, token_mask=np.ones_like(tok, dtype=np.int64),
        slot_mapping=sm, dp_client_id=0, swa_aware=True)
    # usable = min(4, 0) = 0 -> no resident blocks reported, no SWA op.
    assert int(return_mask.sum()) == 0, (
        f"swa_hit=0 must clamp full window to empty; got {int(return_mask.sum())}")
    swa = [o for o in graph._op_map.values() if getattr(o, "is_swa", False)]
    assert len(swa) == 0, "no SWA op when swa_hit == 0"
    _complete(op_cb, cb)
    cpu.unlock(node)


def test_plain_get_on_swa_cache_matches_once_and_unclamped():
    """A plain get() (swa_aware defaults False) on an SWA-enabled cache matches
    exactly once and is NOT clamped by any SWA window — the shared _get_impl_*
    hot path is unaffected for non-SWA callers."""
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tok = _tokens(4, base=33)
    _put(eng, tok)

    with _MatchCounter(eng) as mc:
        graph, return_mask, cb, op_cb, end_id = eng.get(
            request_id=2, token_ids=tok,
            token_mask=np.ones_like(tok, dtype=np.int64),
            slot_mapping=np.arange(tok.shape[0], dtype=np.int64),
            dp_client_id=0)
    assert mc.n == 1, f"plain get matched {mc.n} rounds; must be 1"
    # full 4-block hit, unclamped by SWA.
    assert int(return_mask.sum()) == 4 * TPB
    _complete(op_cb, cb)


# =========================================================================== #
# kvtask dispatch — get_match threads swa_aware into the single match         #
# =========================================================================== #

def test_get_match_threads_swa_aware():
    """KVTaskEngine.get_match(swa_aware=True) drives exactly ONE _get_match_impl
    with swa_aware=True. Exercised via the real unbound method on a fake self —
    no GPU subprocess, mirroring the pure-logic tests in test_kvtask_lifecycle.py."""
    import types
    from flexkv.kvtask import KVTaskEngine

    calls = {"swa_aware": [], "task": None}
    tok = np.arange(4 * TPB, dtype=np.int64)

    def _fake_get_match_impl(token_ids, slot_mapping, **kw):
        calls["swa_aware"].append(kw.get("swa_aware"))
        rm = np.zeros(token_ids.shape[0], dtype=np.bool_)
        rm[:4 * TPB] = True
        return 7, rm

    fake = types.SimpleNamespace(
        cache_engine=types.SimpleNamespace(tokens_per_block=TPB),
        _update_tasks=lambda timeout=0: None,
        _get_match_impl=_fake_get_match_impl,
        tracer=types.SimpleNamespace(trace_request=lambda **k: None),
    )
    full_mask = np.ones(4 * TPB, dtype=np.bool_)

    tid, mask = KVTaskEngine.get_match(
        fake, tok, token_mask=full_mask, dp_client_id=0, swa_aware=True)

    assert calls["swa_aware"] == [True], \
        "must drive exactly one _get_match_impl with swa_aware=True"
    assert tid == 7
    assert int(mask.sum()) == 4 * TPB

