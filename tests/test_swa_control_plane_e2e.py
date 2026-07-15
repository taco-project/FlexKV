# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end SWA data-plane test entered from the ENGINE control-plane API
(GlobalCacheEngine.put/get), with real byte movement — NO stubs.

This is the production-path proof for the SWA data plane: the transfer graph is
produced by ``GlobalCacheEngine.put()`` / ``get()`` (which append SWA peer ops
inside ``_put_impl_*`` / ``_get_impl_*`` — real SWA-pool alloc + node-mounted
match, plus the size-1 GPU placeholder), the GPU sides are bound LATE via
``set_gpu_blocks`` (full-KV) and ``set_swa_gpu_blocks`` (SWA) exactly as
``KVTaskEngine.launch`` does, and the resulting graph is submitted to a real
``TransferEngine`` with GPU pools. We then assert a byte-exact main-KV + SWA
GPU->CPU->GPU roundtrip.

Flow (mirrors KVManager get_match(swa_aware=True) + launch, minus the tp_client subprocess):
  PUT : engine.put() -> graph {full D2H, SWA D2H} -> bind GPU slots -> submit
        -> full+SWA bytes land in the shared CPU pool + SWA host pool
  GET : engine.get(swa_aware=True) -> graph {full H2D, SWA H2D}
        -> bind GPU slots -> submit
        -> bytes restored to fresh GPU blocks -> byte-exact compare

Run INSIDE the container on a free GPU:
    CUDA_VISIBLE_DEVICES=0 python3 tests/test_swa_control_plane_e2e.py
"""
import gc
import sys
import time

import numpy as np
import pytest
import torch

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import DeviceType, WorkerKey
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine

NUM_LAYERS = 4
NUM_BLOCKS_GPU = 64
NUM_BLOCKS_CPU = 64
TOKENS_PER_BLOCK = 16
BYTES_PER_TOKEN_PER_LAYER = 64
DEVICE_ID = 0
NUM_SWA_SLOTS = 64

# GPU SWA slot the connector would pick for the trailing window (arbitrary free
# slot in the SWA GPU pool). The engine builds the SWA op with a placeholder GPU
# slot; late-bind (set_swa_gpu_blocks) rebinds it to this.
SWA_GPU_SLOT = 9


def make_gpu_pool(num_blocks):
    return torch.zeros((NUM_LAYERS, num_blocks, TOKENS_PER_BLOCK,
                        BYTES_PER_TOKEN_PER_LAYER), dtype=torch.uint8,
                       device=f"cuda:{DEVICE_ID}")


def seed_block(pool, block, salt):
    for layer in range(NUM_LAYERS):
        plane = torch.zeros((TOKENS_PER_BLOCK, BYTES_PER_TOKEN_PER_LAYER),
                            dtype=torch.uint8)
        for tok in range(TOKENS_PER_BLOCK):
            for b in range(BYTES_PER_TOKEN_PER_LAYER):
                plane[tok, b] = ((layer * 31 + tok + salt) ^ b) & 0xFF
        pool[layer, block].copy_(plane.to(pool.device))


def block_bytes(pool, block):
    chunks = [pool[layer, block].contiguous().view(-1).cpu().clone()
              for layer in range(NUM_LAYERS)]
    return torch.cat(chunks).numpy().tobytes()


def make_layout(num_blocks):
    return KVCacheLayout(type=KVCacheLayoutType.LAYERFIRST, num_layer=NUM_LAYERS,
                         num_block=num_blocks, tokens_per_block=TOKENS_PER_BLOCK,
                         num_head=1, head_size=BYTES_PER_TOKEN_PER_LAYER, is_mla=True)


def wait_for_op(te, op_ids, timeout_s=30.0):
    pending = set(op_ids)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        for c in te.get_completed_graphs_and_ops(timeout=1.0):
            pending.discard(c.op_id)
    if pending:
        raise TimeoutError(f"Ops {pending} did not complete in {timeout_s}s")


def _run_graph(te, engine, graph, op_cb, cb, full_gpu_blocks, swa_gpu_slot,
               reported_op_ids):
    """Late-bind GPU slots (full-KV + SWA), submit to the TransferEngine, wait,
    then run the engine's op/transfer callbacks (set_ready/unlock/lock-release),
    exactly as KVTaskEngine.launch + _update_tasks would."""
    graph.set_gpu_blocks(np.asarray(full_gpu_blocks, dtype=np.int64))
    if graph._swa_gpu_transfer_op_id:
        graph.set_swa_gpu_blocks(np.asarray([swa_gpu_slot], dtype=np.int64))
    te.submit_transfer_graph(graph)
    wait_for_op(te, reported_op_ids)
    torch.cuda.synchronize()
    for op_id, c in op_cb.items():
        c()
    cb()


def _run(transfer_engines) -> int:
    if not torch.cuda.is_available():
        print("[verify] CUDA not available", flush=True)
        return 2
    torch.cuda.set_device(DEVICE_ID)
    print(f"[verify] device cuda:{DEVICE_ID}", flush=True)

    # ---- config: engine mempool and TransferEngine CPU pool share num_cpu_blocks
    model_config = ModelConfig(num_layers=NUM_LAYERS, num_kv_heads=1,
                               head_size=BYTES_PER_TOKEN_PER_LAYER, use_mla=True,
                               dtype=torch.uint8, tp_size=1, pp_size=1, dp_size=1,
                               cp_size=1)
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK, enable_cpu=True, enable_ssd=False,
        enable_remote=False, num_cpu_blocks=NUM_BLOCKS_CPU,
        swa=SWAPoolConfig(enabled=True, num_slots=NUM_SWA_SLOTS,
                          num_swa_layers=NUM_LAYERS,
                          bytes_per_token_per_layer=BYTES_PER_TOKEN_PER_LAYER,
                          pin_memory=True))
    cache_config.enable_swa_transfer = True

    # ---- GPU pools: main-KV + SWA (dedicated), seed one input block of each ----
    mk_pool = make_gpu_pool(NUM_BLOCKS_GPU)
    sw_pool = make_gpu_pool(NUM_SWA_SLOTS)
    # One-block request (== one page == one SWA window on DSv4). Put reads GPU
    # block 0 (full) and SWA GPU slot SWA_GPU_SLOT; get restores into block 3 /
    # SWA slot SWA_GPU_SLOT (a different block for full to prove CPU sourcing).
    PUT_FULL_GPU, GET_FULL_GPU = 0, 3
    seed_block(mk_pool, PUT_FULL_GPU, salt=0xA1)
    seed_block(sw_pool, SWA_GPU_SLOT, salt=0xB2)
    expected_mk = block_bytes(mk_pool, PUT_FULL_GPU)
    expected_sw = block_bytes(sw_pool, SWA_GPU_SLOT)

    mk_handles = [TensorSharedHandle(mk_pool[l].contiguous(), DEVICE_ID)
                  for l in range(NUM_LAYERS)]
    sw_handles = [TensorSharedHandle(sw_pool[l].contiguous(), DEVICE_ID)
                  for l in range(NUM_LAYERS)]

    # ---- StorageEngine: CPU pool (shared block space with engine mempool) +
    #      main-KV GPU pool + SWA GPU pool ----
    se = StorageEngine(model_config, cache_config, num_layers_per_pp_stage=NUM_LAYERS)
    assert se.has_storage_handle(DeviceType.CPU, device_id=0, is_swa=True), \
        "SWA CPU host pool missing"
    gpu_layout = make_layout(NUM_BLOCKS_GPU)
    swa_gpu_layout = make_layout(NUM_SWA_SLOTS)
    se.register_gpu_blocks(mk_handles, gpu_layout, device_id=DEVICE_ID, dtype=torch.uint8)
    se.register_swa_gpu_blocks(sw_handles, swa_gpu_layout, device_id=DEVICE_ID, dtype=torch.uint8)

    worker_key = WorkerKey(dp_client_id=0, pp_rank=0)
    te = TransferEngine(
        gpu_handles={worker_key: [se.get_storage_handle(DeviceType.GPU, device_id=DEVICE_ID)]},
        model_config=model_config, cache_config=cache_config,
        cpu_handle=se.get_storage_handle(DeviceType.CPU),
        swa_gpu_handles={worker_key: [se.get_swa_storage_handle(DEVICE_ID)]},
        swa_cpu_handle=se.get_storage_handle(DeviceType.CPU, device_id=0, is_swa=True),
    )
    transfer_engines.append(te)
    te.start()
    print("[verify] TransferEngine started (main-KV + SWA workers)", flush=True)

    # ---- the control-plane engine (radix + SWA host pool), sharing num_cpu_blocks
    engine = GlobalCacheEngine(cache_config, model_config)

    tok = np.arange(1, TOKENS_PER_BLOCK + 1, dtype=np.int64)  # 1 block
    put_slot_mapping = np.arange(PUT_FULL_GPU * TOKENS_PER_BLOCK,
                                 (PUT_FULL_GPU + 1) * TOKENS_PER_BLOCK, dtype=np.int64)
    mask = np.ones_like(tok, dtype=np.int64)

    # ===== PUT via GlobalCacheEngine.put() (SWA append inside _put_impl_*) ======
    put_graph, _rm, put_cb, put_op_cb, put_end = engine.put(
        request_id=1, token_ids=tok, token_mask=mask,
        slot_mapping=put_slot_mapping, dp_client_id=0)
    swa_put_ops = [o for o in put_graph._op_map.values() if getattr(o, "is_swa", False)]
    assert len(swa_put_ops) == 1, f"expected 1 SWA D2H, got {len(swa_put_ops)}"
    reported = {put_end}
    reported |= {o.op_id for o in swa_put_ops}
    print(f"[verify] PUT graph from engine: {len(put_graph._op_map)} ops, "
          f"1 SWA D2H (cpu_slot={swa_put_ops[0].dst_block_ids.tolist()})", flush=True)
    # PUT full GPU src = block 0 (already in slot_mapping); SWA GPU src = SWA_GPU_SLOT.
    _run_graph(te, engine, put_graph, put_op_cb, put_cb,
               full_gpu_blocks=[PUT_FULL_GPU], swa_gpu_slot=SWA_GPU_SLOT,
               reported_op_ids=reported)
    print("[verify] PUT done (full + SWA D2H committed to CPU pools)", flush=True)

    # zero the GPU pools so GET must source from CPU
    mk_pool.zero_(); sw_pool.zero_(); torch.cuda.synchronize()

    # ===== GET via GlobalCacheEngine.get(swa_aware=True) ========================
    get_slot_mapping = np.arange(GET_FULL_GPU * TOKENS_PER_BLOCK,
                                 (GET_FULL_GPU + 1) * TOKENS_PER_BLOCK, dtype=np.int64)
    get_graph, _rm2, get_cb, get_op_cb, get_end = engine.get(
        request_id=2, token_ids=tok, token_mask=np.ones_like(tok, dtype=np.int64),
        slot_mapping=get_slot_mapping, dp_client_id=0, swa_aware=True)
    swa_get_ops = [o for o in get_graph._op_map.values() if getattr(o, "is_swa", False)]
    assert len(swa_get_ops) == 1 and swa_get_ops[0].transfer_type.name == "H2D", \
        f"expected 1 SWA H2D, got {[o.transfer_type.name for o in swa_get_ops]}"
    reported2 = {swa_get_ops[0].op_id}
    # the full-KV H2D reported op(s): everything appended to finished except swa
    # is inside the graph; wait on the barrier's predecessors (full H2D + SWA H2D).
    barrier = get_graph._op_map[get_end]
    reported2 |= set(barrier.predecessors)
    print(f"[verify] GET graph from engine: {len(get_graph._op_map)} ops, "
          f"1 SWA H2D (cpu_slot={swa_get_ops[0].src_block_ids.tolist()})", flush=True)
    _run_graph(te, engine, get_graph, get_op_cb, get_cb,
               full_gpu_blocks=[GET_FULL_GPU], swa_gpu_slot=SWA_GPU_SLOT,
               reported_op_ids=reported2)
    print("[verify] GET done (full + SWA H2D restored to GPU)", flush=True)

    # ===== byte-exact compare ==================================================
    actual_mk = block_bytes(mk_pool, GET_FULL_GPU)
    actual_sw = block_bytes(sw_pool, SWA_GPU_SLOT)
    failed = False
    if actual_mk != expected_mk:
        print("[verify] FAIL main-KV byte mismatch", flush=True); failed = True
    else:
        print(f"[verify] OK    main-KV: {len(expected_mk)} bytes match "
              f"GPU[{PUT_FULL_GPU}]->CPU->GPU[{GET_FULL_GPU}]", flush=True)
    if actual_sw != expected_sw:
        print("[verify] FAIL SWA byte mismatch", flush=True); failed = True
    else:
        print(f"[verify] OK    SWA    : {len(expected_sw)} bytes match "
              f"GPU[{SWA_GPU_SLOT}]->CPU(host pool)->GPU[{SWA_GPU_SLOT}]", flush=True)

    # SWA lock released by the H2D callback (no leak).
    sm = SequenceMeta(token_ids=tok, tokens_per_block=TOKENS_PER_BLOCK); sm.gen_hashes()
    mr = engine.cpu_cache_engine.match(sm)
    node = mr.last_swa_node if mr.swa_hit_blocks == 1 else None
    if node is not None:
        engine.cpu_cache_engine._pin_swa_node(node)
        lock_ok = (node.swa_lock_ref == 1)  # our fresh probe lock; prior load lock released
        engine._swa_release_load_lock(node, engine=engine.cpu_cache_engine)
        print(f"[verify] {'OK   ' if lock_ok else 'FAIL '} SWA load lock released "
              f"(lock_ref after fresh probe == 1: {lock_ok})", flush=True)
        failed = failed or not lock_ok

    if failed:
        return 4
    print("[verify] PASS: engine control-plane graph moved main-KV + SWA byte-exact, "
          "no stub, lock released", flush=True)
    return 0


def main() -> int:
    """Run the E2E flow and always tear down spawned CUDA workers."""
    transfer_engines = []
    try:
        return _run(transfer_engines)
    finally:
        for transfer_engine in reversed(transfer_engines):
            transfer_engine.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@pytest.mark.e2e
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SWA byte-exact e2e needs a GPU")
def test_swa_control_plane_e2e_byte_exact():
    """pytest entry: run the byte-exact GPU->CPU->GPU roundtrip (main() returns 0
    on success). Collected by `pytest -m e2e`; also runnable standalone."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
