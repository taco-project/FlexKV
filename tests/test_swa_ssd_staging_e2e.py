# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end SWA SSD-staging byte-exact test (multi-tier), entered from the
ENGINE control-plane API with real byte movement — NO stubs.

Proves the SSD SWA tier round-trip that the CPU-only e2e (test_swa_control_plane_e2e.py)
does not cover:

  PUT : engine.put() -> graph {full D2H, SWA D2H, SWA H2DISK write-through}
        -> the SWA window lands in BOTH the CPU SWA host pool AND the SSD SWA
           files (write-through).
  EVICT: drop the CPU SWA slot (SWA-only eviction) so the window survives ONLY
         on SSD — forcing the GET to stage from SSD.
  GET : engine.get(swa_aware=True) -> graph {full H2D, SWA DISK2H
        (SSD->CPU staging) -> SWA H2D (CPU->GPU)} -> bytes restored to a fresh
        GPU SWA slot -> byte-exact.

This exercises SWA graph append inside _put_impl_* (write-through) and
_get_impl_* (SSD->CPU transient staging slot + H2D), plus the transient staging
slot free on H2D completion.

Run INSIDE the container on a free GPU:
    CUDA_VISIBLE_DEVICES=0 python3 tests/test_swa_ssd_staging_e2e.py
"""
import gc
import os
import shutil
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
NUM_BLOCKS_SSD = 128
TOKENS_PER_BLOCK = 16
BYTES_PER_TOKEN_PER_LAYER = 64
DEVICE_ID = 0
NUM_SWA_SLOTS = 32
NUM_SWA_SSD_SLOTS = 32
SWA_GPU_SLOT = 9
SSD_CACHE_DIR = "./_swa_ssd_e2e_cache"


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


def _run_graph(te, graph, op_cb, cb, full_gpu_blocks, swa_gpu_slot, reported_op_ids):
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
    if os.path.isdir(SSD_CACHE_DIR):
        shutil.rmtree(SSD_CACHE_DIR)
    os.makedirs(SSD_CACHE_DIR, exist_ok=True)
    print(f"[verify] device cuda:{DEVICE_ID}, ssd_dir={SSD_CACHE_DIR}", flush=True)

    model_config = ModelConfig(num_layers=NUM_LAYERS, num_kv_heads=1,
                               head_size=BYTES_PER_TOKEN_PER_LAYER, use_mla=True,
                               dtype=torch.uint8, tp_size=1, pp_size=1, dp_size=1,
                               cp_size=1)
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK, enable_cpu=True, enable_ssd=True,
        enable_remote=False, num_cpu_blocks=NUM_BLOCKS_CPU,
        num_ssd_blocks=NUM_BLOCKS_SSD, ssd_cache_dir=SSD_CACHE_DIR,
        swa=SWAPoolConfig(enabled=True, num_slots=NUM_SWA_SLOTS,
                          num_ssd_slots=NUM_SWA_SSD_SLOTS,
                          num_swa_layers=NUM_LAYERS,
                          bytes_per_token_per_layer=BYTES_PER_TOKEN_PER_LAYER,
                          pin_memory=True))
    cache_config.enable_swa_transfer = True

    mk_pool = make_gpu_pool(NUM_BLOCKS_GPU)
    sw_pool = make_gpu_pool(NUM_SWA_SLOTS)
    PUT_FULL_GPU, GET_FULL_GPU = 0, 3
    seed_block(mk_pool, PUT_FULL_GPU, salt=0xA1)
    seed_block(sw_pool, SWA_GPU_SLOT, salt=0xB2)
    expected_sw = block_bytes(sw_pool, SWA_GPU_SLOT)

    mk_handles = [TensorSharedHandle(mk_pool[l].contiguous(), DEVICE_ID)
                  for l in range(NUM_LAYERS)]
    sw_handles = [TensorSharedHandle(sw_pool[l].contiguous(), DEVICE_ID)
                  for l in range(NUM_LAYERS)]

    se = StorageEngine(model_config, cache_config, num_layers_per_pp_stage=NUM_LAYERS)
    assert se.has_storage_handle(DeviceType.CPU, device_id=0, is_swa=True), "SWA CPU pool missing"
    assert se.has_storage_handle(DeviceType.SSD, device_id=0, is_swa=True), "SWA SSD pool missing"
    gpu_layout = make_layout(NUM_BLOCKS_GPU)
    swa_gpu_layout = make_layout(NUM_SWA_SLOTS)
    se.register_gpu_blocks(mk_handles, gpu_layout, device_id=DEVICE_ID, dtype=torch.uint8)
    se.register_swa_gpu_blocks(sw_handles, swa_gpu_layout, device_id=DEVICE_ID, dtype=torch.uint8)

    worker_key = WorkerKey(dp_client_id=0, pp_rank=0)
    te = TransferEngine(
        gpu_handles={worker_key: [se.get_storage_handle(DeviceType.GPU, device_id=DEVICE_ID)]},
        model_config=model_config, cache_config=cache_config,
        cpu_handle=se.get_storage_handle(DeviceType.CPU),
        ssd_handle=se.get_storage_handle(DeviceType.SSD),
        swa_gpu_handles={worker_key: [se.get_swa_storage_handle(DEVICE_ID)]},
        swa_cpu_handle=se.get_storage_handle(DeviceType.CPU, device_id=0, is_swa=True),
        swa_ssd_handle=se.get_storage_handle(DeviceType.SSD, device_id=0, is_swa=True),
    )
    transfer_engines.append(te)
    te.start()
    print("[verify] TransferEngine started (main-KV + SWA CPU/SSD workers)", flush=True)

    engine = GlobalCacheEngine(cache_config, model_config)
    assert engine.ssd_cache_engine is not None and engine.ssd_cache_engine.swa_enabled, \
        "engine SSD SWA tier not enabled"

    tok = np.arange(1, TOKENS_PER_BLOCK + 1, dtype=np.int64)
    put_slot_mapping = np.arange(PUT_FULL_GPU * TOKENS_PER_BLOCK,
                                 (PUT_FULL_GPU + 1) * TOKENS_PER_BLOCK, dtype=np.int64)
    mask = np.ones_like(tok, dtype=np.int64)

    # ===== PUT: full D2H + SWA D2H + SWA H2DISK write-through ====================
    put_graph, _rm, put_cb, put_op_cb, put_end = engine.put(
        request_id=1, token_ids=tok, token_mask=mask,
        slot_mapping=put_slot_mapping, dp_client_id=0)
    swa_put_ops = [o for o in put_graph._op_map.values() if getattr(o, "is_swa", False)]
    kinds = sorted(o.transfer_type.name for o in swa_put_ops)
    assert "D2H" in kinds and "H2DISK" in kinds, f"expected SWA D2H + H2DISK, got {kinds}"
    assert engine.ssd_cache_engine.swa_pool.num_used == 1, "SSD SWA slot not allocated on put"
    # This helper invokes every per-op callback manually, so wait for every op
    # carrying one.  The production KVTask loop invokes each callback from its
    # own completion event; task_end intentionally does not wait for SSD
    # write-through.
    reported = {put_end} | set(put_op_cb)
    print(f"[verify] PUT graph: SWA ops={kinds} (write-through to SSD)", flush=True)
    _run_graph(te, put_graph, put_op_cb, put_cb,
               full_gpu_blocks=[PUT_FULL_GPU], swa_gpu_slot=SWA_GPU_SLOT,
               reported_op_ids=reported)
    print("[verify] PUT done (SWA in CPU pool + SSD files)", flush=True)

    # ===== EVICT the CPU SWA so the window survives only on SSD =================
    n_cpu = engine.cpu_cache_engine.swa_pool.num_used
    engine.cpu_cache_engine._evict_swa_slots(n_cpu)
    seq = SequenceMeta(token_ids=tok, tokens_per_block=TOKENS_PER_BLOCK); seq.gen_hashes()
    cpu_hit = engine.cpu_cache_engine.match(seq).swa_hit_blocks
    ssd_hit = engine.ssd_cache_engine.match(seq).swa_hit_blocks
    assert cpu_hit == 0 and ssd_hit > 0, f"precondition failed: cpu_hit={cpu_hit} ssd_hit={ssd_hit}"
    print(f"[verify] evicted CPU SWA (cpu_hit=0, ssd_hit={ssd_hit}); GET must stage from SSD", flush=True)

    # zero the GPU SWA pool so GET must restore from SSD->CPU->GPU
    sw_pool.zero_(); mk_pool.zero_(); torch.cuda.synchronize()

    # ===== GET: SWA DISK2H (SSD->CPU staging) -> SWA H2D (CPU->GPU) =============
    get_slot_mapping = np.arange(GET_FULL_GPU * TOKENS_PER_BLOCK,
                                 (GET_FULL_GPU + 1) * TOKENS_PER_BLOCK, dtype=np.int64)
    get_graph, _rm2, get_cb, get_op_cb, get_end = engine.get(
        request_id=2, token_ids=tok, token_mask=np.ones_like(tok, dtype=np.int64),
        slot_mapping=get_slot_mapping, dp_client_id=0, swa_aware=True)
    swa_get_ops = [o for o in get_graph._op_map.values() if getattr(o, "is_swa", False)]
    gkinds = sorted(o.transfer_type.name for o in swa_get_ops)
    assert "DISK2H" in gkinds and "H2D" in gkinds, f"expected SWA DISK2H+H2D staging, got {gkinds}"
    swa_h2d = [o for o in swa_get_ops if o.transfer_type.name == "H2D"][0]
    swa_disk2h = [o for o in swa_get_ops if o.transfer_type.name == "DISK2H"][0]
    assert swa_disk2h.op_id in swa_h2d.predecessors, "SWA H2D must depend on SSD DISK2H"
    barrier = get_graph._op_map[get_end]
    reported2 = {swa_h2d.op_id} | set(barrier.predecessors)
    print(f"[verify] GET graph: SWA ops={gkinds} (SSD staging chain)", flush=True)
    _run_graph(te, get_graph, get_op_cb, get_cb,
               full_gpu_blocks=[GET_FULL_GPU], swa_gpu_slot=SWA_GPU_SLOT,
               reported_op_ids=reported2)
    print("[verify] GET done (SWA restored via SSD->CPU->GPU)", flush=True)

    # ===== byte-exact compare ==================================================
    actual_sw = block_bytes(sw_pool, SWA_GPU_SLOT)
    failed = actual_sw != expected_sw
    if failed:
        print("[verify] FAIL SWA byte mismatch after SSD staging", flush=True)
    else:
        print(f"[verify] OK    SWA: {len(expected_sw)} bytes match "
              f"GPU->CPU/SSD->(evict CPU)->DISK2H->H2D->GPU", flush=True)

    # transient CPU staging slot must be freed on H2D completion (no leak).
    staging_used = engine.cpu_cache_engine.swa_pool.num_used
    if staging_used != 0:
        print(f"[verify] FAIL transient CPU staging slot leaked (num_used={staging_used})", flush=True)
        failed = True
    else:
        print("[verify] OK    transient CPU staging slot freed (no leak)", flush=True)

    if failed:
        return 4
    print("[verify] PASS: SSD SWA staging byte-exact, transient slot freed", flush=True)
    return 0


def main() -> int:
    """Run the E2E flow and always release workers and temporary SSD data."""
    transfer_engines = []
    try:
        return _run(transfer_engines)
    finally:
        for transfer_engine in reversed(transfer_engines):
            transfer_engine.shutdown()
        shutil.rmtree(SSD_CACHE_DIR, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@pytest.mark.e2e
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SWA SSD staging e2e needs a GPU")
def test_swa_ssd_staging_e2e_byte_exact():
    """pytest entry: SSD SWA write-through + staged GET byte-exact roundtrip."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
