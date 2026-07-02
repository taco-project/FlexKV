# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-plane SWA smoke test — enters from the top (GlobalCacheEngine), NOT
the data plane, using the REAL (de-stubbed) SWA slot sources.

The sibling data-plane smoke test (tests/test_swa_dispatch.py) hand-builds
TransferOp(is_swa=True) and submits them straight to the TransferEngine. This
test drives the real control-plane API and checks that a request is turned into
a Full+SWA plan against the node-mounted radix tree, now that _swa_get_slots /
_swa_put_slots are wired (no stubs):

    GlobalCacheEngine.put()   -> full-KV D2H graph + SWA peer D2H (alloc+set_swa)
    GlobalCacheEngine.swa_align() -> full_hit, swa_hit, usable = min(full, swa)
    GlobalCacheEngine.get()   -> full-KV H2D graph + SWA peer H2D (matched slot)

Runs CPU-only (no GPU): the control plane's job is to produce the transfer graph
+ masks; the GPU SWA slot is a placeholder bound late (set_swa_gpu_blocks). Byte
movement is the data plane's job (test_swa_dispatch.py / the KVManager e2e test).

Requires flexkv.c_ext (uses the production CRadixTreeIndex via CacheEngineAccel).
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

TPB = 16


def _model_config():
    return ModelConfig(
        num_layers=4, num_kv_heads=1, head_size=128,
        use_mla=True, dtype=torch.bfloat16, tp_size=1, dp_size=1,
    )


def _cache_config(enable_swa_transfer: bool):
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


# --------------------------------------------------------------------------- #
# 1. control-plane PUT builds a full-KV + SWA peer store chain (real alloc)    #
# --------------------------------------------------------------------------- #

def test_put_builds_full_plus_swa_store_chain():
    eng = GlobalCacheEngine(_cache_config(enable_swa_transfer=True), _model_config())
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


# --------------------------------------------------------------------------- #
# 2. control-plane swa_align: usable = min(full_hit, swa_hit)                  #
# --------------------------------------------------------------------------- #

def test_swa_align_clamps_full_to_swa_hit():
    eng = GlobalCacheEngine(_cache_config(enable_swa_transfer=True), _model_config())
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


# --------------------------------------------------------------------------- #
# 3. control-plane GET builds full-KV + SWA peer load chain (matched slot)     #
# --------------------------------------------------------------------------- #

def test_get_builds_full_plus_swa_load_chain():
    eng = GlobalCacheEngine(_cache_config(enable_swa_transfer=True), _model_config())
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


# --------------------------------------------------------------------------- #
# 4. late-bind: set_swa_gpu_blocks rebinds the GPU side of the SWA op          #
# --------------------------------------------------------------------------- #

def test_set_swa_gpu_blocks_rebinds_gpu_side():
    eng = GlobalCacheEngine(_cache_config(enable_swa_transfer=True), _model_config())
    tok = _tokens(4, base=7)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)
    graph, _rm, cb, op_cb, _e = eng.put(1, tok, mask, slot_mapping, dp_client_id=0)
    swa = _swa_ops(graph)[0]
    # placeholder GPU src before late-bind
    assert swa.src_block_ids.tolist() == [0]
    # rebind GPU SWA slots (as launch would): swa_slot_mapping -> slot ids
    swa_gpu_slot_ids = eng.swa_slot_mapping_to_slot_ids(np.arange(TPB, dtype=np.int64) + 5 * TPB)
    graph.set_swa_gpu_blocks(swa_gpu_slot_ids)
    assert swa.src_block_ids.tolist() == [5]   # D2H: GPU is src, rebound
    _complete(op_cb, cb)


# --------------------------------------------------------------------------- #
# 5. gate OFF: control plane emits NO SWA ops                                  #
# --------------------------------------------------------------------------- #

def test_gate_off_no_swa_ops_in_control_plane_graph():
    eng = GlobalCacheEngine(_cache_config(enable_swa_transfer=False), _model_config())
    tok = _tokens(4, base=4)
    mask = np.ones_like(tok, dtype=np.int64)
    slot_mapping = np.arange(tok.shape[0], dtype=np.int64)
    graph, _rm, cb, op_cb, _e = eng.put(1, tok, mask, slot_mapping, dp_client_id=0)
    assert len(_swa_ops(graph)) == 0, "SWA ops emitted with enable_swa_transfer=False"
    assert eng.cpu_cache_engine.swa_pool.num_used == 0  # no slot allocated
    _complete(op_cb, cb)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
