# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated REMOTE SWA staging E2E with real byte movement.

This test uses the production control plane and data plane:

  PUT   : GlobalCacheEngine.put() emits full D2H plus SWA D2H/H2REMOTE.
  EVICT : CPU SWA is removed so the SWA window only survives in REMOTE.
  GET   : GlobalCacheEngine.get() emits SWA REMOTE2H -> H2D.

It is skipped by default because it requires a CFS-enabled build and a reachable
PCFS service. Run it explicitly with:

    FLEXKV_RUN_REMOTE_E2E=1 \
    FLEXKV_PCFS_FSID=<fsid> \
    FLEXKV_PCFS_PORT=<port> \
    FLEXKV_PCFS_IP=<ip> \
    FLEXKV_PCFS_PARENT_NODEID=<nodeid> \
    FLEXKV_REMOTE_CACHE_PATH=<remote-file-prefix> \
    CUDA_VISIBLE_DEVICES=0 \
    python -m pytest -m e2e tests/test_swa_remote_staging_e2e.py
"""
import os
import sys
import time

import numpy as np
import pytest
import torch

pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import DeviceType, TransferType, WorkerKey
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine
from flexkv.transfer import worker as transfer_worker

NUM_LAYERS = 4
NUM_BLOCKS_GPU = 64
NUM_BLOCKS_CPU = 64
NUM_BLOCKS_REMOTE = 128
TOKENS_PER_BLOCK = 16
BYTES_PER_TOKEN_PER_LAYER = 64
DEVICE_ID = 0
NUM_SWA_SLOTS = 32
NUM_SWA_REMOTE_SLOTS = 32
SWA_GPU_SLOT = 9

_REMOTE_E2E_ENV = "FLEXKV_RUN_REMOTE_E2E"
_PCFS_ENV_KEYS = (
    "FLEXKV_PCFS_FSID",
    "FLEXKV_PCFS_PORT",
    "FLEXKV_PCFS_IP",
    "FLEXKV_PCFS_PARENT_NODEID",
)


def _remote_e2e_enabled() -> bool:
    return os.getenv(_REMOTE_E2E_ENV) == "1"


def _missing_remote_env() -> list[str]:
    missing = [key for key in _PCFS_ENV_KEYS if not os.getenv(key)]
    if not os.getenv("FLEXKV_REMOTE_CACHE_PATH"):
        missing.append("FLEXKV_REMOTE_CACHE_PATH")
    return missing


def _remote_config() -> dict:
    missing = [key for key in _PCFS_ENV_KEYS if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"missing PCFS env for REMOTE E2E: {', '.join(missing)}")
    return {
        "pcfs_fsid": os.environ["FLEXKV_PCFS_FSID"],
        "pcfs_port": int(os.environ["FLEXKV_PCFS_PORT"]),
        "pcfs_ip": os.environ["FLEXKV_PCFS_IP"],
        "pcfs_parent_nodeid": int(os.environ["FLEXKV_PCFS_PARENT_NODEID"]),
    }


def make_gpu_pool(num_blocks):
    return torch.zeros(
        (NUM_LAYERS, num_blocks, TOKENS_PER_BLOCK, BYTES_PER_TOKEN_PER_LAYER),
        dtype=torch.uint8,
        device=f"cuda:{DEVICE_ID}",
    )


def seed_block(pool, block, salt):
    for layer in range(NUM_LAYERS):
        plane = torch.zeros(
            (TOKENS_PER_BLOCK, BYTES_PER_TOKEN_PER_LAYER),
            dtype=torch.uint8,
        )
        for tok in range(TOKENS_PER_BLOCK):
            for b in range(BYTES_PER_TOKEN_PER_LAYER):
                plane[tok, b] = ((layer * 31 + tok + salt) ^ b) & 0xFF
        pool[layer, block].copy_(plane.to(pool.device))


def block_bytes(pool, block):
    chunks = [
        pool[layer, block].contiguous().view(-1).cpu().clone()
        for layer in range(NUM_LAYERS)
    ]
    return torch.cat(chunks).numpy().tobytes()


def make_layout(num_blocks):
    return KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=NUM_LAYERS,
        num_block=num_blocks,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_head=1,
        head_size=BYTES_PER_TOKEN_PER_LAYER,
        is_mla=True,
    )


def wait_for_op(te, op_ids, timeout_s=60.0):
    pending = set(op_ids)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        for completed in te.get_completed_graphs_and_ops(timeout=1.0):
            pending.discard(completed.op_id)
        if pending:
            time.sleep(0.01)
    if pending:
        raise TimeoutError(f"Ops {pending} did not complete in {timeout_s}s")


def _run_graph(te, graph, op_cb, cb, full_gpu_blocks, swa_gpu_slot, reported_op_ids):
    graph.set_gpu_blocks(np.asarray(full_gpu_blocks, dtype=np.int64))
    if graph._swa_gpu_transfer_op_id:
        graph.set_swa_gpu_blocks(np.asarray([swa_gpu_slot], dtype=np.int64))
    te.submit_transfer_graph(graph)
    wait_for_op(te, reported_op_ids)
    torch.cuda.synchronize()
    for callback in op_cb.values():
        callback()
    cb()


def _cache_config(remote_cache_path: str, remote_config: dict) -> CacheConfig:
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=True,
        enable_3rd_remote=False,
        enable_kv_sharing=False,
        num_cpu_blocks=NUM_BLOCKS_CPU,
        num_remote_blocks=NUM_BLOCKS_REMOTE,
        remote_cache_path=remote_cache_path,
        remote_config_custom=remote_config,
        swa=SWAPoolConfig(
            enabled=True,
            num_slots=NUM_SWA_SLOTS,
            num_remote_slots=NUM_SWA_REMOTE_SLOTS,
            num_swa_layers=NUM_LAYERS,
            bytes_per_token_per_layer=BYTES_PER_TOKEN_PER_LAYER,
            pin_memory=True,
        ),
    )
    cache_config.enable_remote = True
    cache_config.enable_kv_sharing = False
    cache_config.enable_swa_transfer = True
    return cache_config


def main() -> int:
    if not _remote_e2e_enabled():
        print(f"[verify] set {_REMOTE_E2E_ENV}=1 to run REMOTE E2E", flush=True)
        return 2
    missing = _missing_remote_env()
    if missing:
        print(f"[verify] missing REMOTE E2E env: {', '.join(missing)}", flush=True)
        return 2
    if transfer_worker.transfer_kv_blocks_remote is None:
        print("[verify] transfer_kv_blocks_remote unavailable; build with FLEXKV_ENABLE_CFS=1", flush=True)
        return 2
    if not torch.cuda.is_available():
        print("[verify] CUDA not available", flush=True)
        return 2

    remote_config = _remote_config()
    remote_cache_path = os.environ["FLEXKV_REMOTE_CACHE_PATH"]
    torch.cuda.set_device(DEVICE_ID)
    print(f"[verify] device cuda:{DEVICE_ID}, remote_path={remote_cache_path}", flush=True)

    model_config = ModelConfig(
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_size=BYTES_PER_TOKEN_PER_LAYER,
        use_mla=True,
        dtype=torch.uint8,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        cp_size=1,
    )
    cache_config = _cache_config(remote_cache_path, remote_config)

    mk_pool = make_gpu_pool(NUM_BLOCKS_GPU)
    sw_pool = make_gpu_pool(NUM_SWA_SLOTS)
    put_full_gpu, get_full_gpu = 0, 3
    seed_block(mk_pool, put_full_gpu, salt=0xC1)
    seed_block(sw_pool, SWA_GPU_SLOT, salt=0xD2)
    expected_sw = block_bytes(sw_pool, SWA_GPU_SLOT)

    mk_handles = [
        TensorSharedHandle(mk_pool[layer].contiguous(), DEVICE_ID)
        for layer in range(NUM_LAYERS)
    ]
    sw_handles = [
        TensorSharedHandle(sw_pool[layer].contiguous(), DEVICE_ID)
        for layer in range(NUM_LAYERS)
    ]

    se = StorageEngine(model_config, cache_config, num_layers_per_pp_stage=NUM_LAYERS)
    assert se.has_storage_handle(DeviceType.REMOTE, device_id=0, is_swa=True), (
        "SWA REMOTE pool missing"
    )
    se.register_gpu_blocks(
        mk_handles,
        make_layout(NUM_BLOCKS_GPU),
        device_id=DEVICE_ID,
        dtype=torch.uint8,
    )
    se.register_swa_gpu_blocks(
        sw_handles,
        make_layout(NUM_SWA_SLOTS),
        device_id=DEVICE_ID,
        dtype=torch.uint8,
    )

    worker_key = WorkerKey(dp_client_id=0, pp_rank=0)
    te = TransferEngine(
        gpu_handles={worker_key: [se.get_storage_handle(DeviceType.GPU, device_id=DEVICE_ID)]},
        model_config=model_config,
        cache_config=cache_config,
        cpu_handle=se.get_storage_handle(DeviceType.CPU),
        remote_handle=se.get_storage_handle(DeviceType.REMOTE),
        swa_gpu_handles={worker_key: [se.get_swa_storage_handle(DEVICE_ID)]},
        swa_cpu_handle=se.get_storage_handle(DeviceType.CPU, device_id=0, is_swa=True),
        swa_remote_handle=se.get_storage_handle(DeviceType.REMOTE, device_id=0, is_swa=True),
    )
    te.start()
    print("[verify] TransferEngine started (main-KV + SWA CPU/REMOTE workers)", flush=True)

    engine = GlobalCacheEngine(cache_config, model_config)
    assert engine.remote_cache_engine is not None and engine.remote_cache_engine.swa_enabled, (
        "engine REMOTE SWA tier not enabled"
    )

    tok = np.arange(1, TOKENS_PER_BLOCK + 1, dtype=np.int64)
    put_slot_mapping = np.arange(
        put_full_gpu * TOKENS_PER_BLOCK,
        (put_full_gpu + 1) * TOKENS_PER_BLOCK,
        dtype=np.int64,
    )
    mask = np.ones_like(tok, dtype=np.int64)

    put_graph, _rm, put_cb, put_op_cb, put_end = engine.put(
        request_id=1,
        token_ids=tok,
        token_mask=mask,
        slot_mapping=put_slot_mapping,
        dp_client_id=0,
    )
    swa_put_ops = [
        op for op in put_graph._op_map.values() if getattr(op, "is_swa", False)
    ]
    kinds = sorted(op.transfer_type.name for op in swa_put_ops)
    assert "D2H" in kinds and "H2REMOTE" in kinds, (
        f"expected SWA D2H + H2REMOTE, got {kinds}"
    )
    reported = {put_end} | {op.op_id for op in swa_put_ops}
    print(f"[verify] PUT graph: SWA ops={kinds} (write-through to REMOTE)", flush=True)
    _run_graph(
        te,
        put_graph,
        put_op_cb,
        put_cb,
        full_gpu_blocks=[put_full_gpu],
        swa_gpu_slot=SWA_GPU_SLOT,
        reported_op_ids=reported,
    )
    print("[verify] PUT done (SWA in CPU pool + REMOTE files)", flush=True)

    engine.cpu_cache_engine._evict_swa(engine.cpu_cache_engine.swa_pool.num_used)
    seq = SequenceMeta(token_ids=tok, tokens_per_block=TOKENS_PER_BLOCK)
    seq.gen_hashes()
    cpu_hit, _cpu_slot = engine.cpu_cache_engine.match_swa(seq, upper_bound_blocks=1)
    remote_hit, _remote_slot = engine.remote_cache_engine.match_swa(seq, upper_bound_blocks=1)
    assert cpu_hit == 0 and remote_hit > 0, (
        f"precondition failed: cpu_hit={cpu_hit} remote_hit={remote_hit}"
    )
    sw_pool.zero_()
    mk_pool.zero_()
    torch.cuda.synchronize()

    get_slot_mapping = np.arange(
        get_full_gpu * TOKENS_PER_BLOCK,
        (get_full_gpu + 1) * TOKENS_PER_BLOCK,
        dtype=np.int64,
    )
    get_graph, _rm2, get_cb, get_op_cb, get_end = engine.get(
        request_id=2,
        token_ids=tok,
        token_mask=np.ones_like(tok, dtype=np.int64),
        slot_mapping=get_slot_mapping,
        dp_client_id=0,
    )
    failed = False
    swa_get_ops = [
        op for op in get_graph._op_map.values() if getattr(op, "is_swa", False)
    ]
    get_kinds = sorted(op.transfer_type.name for op in swa_get_ops)
    assert "REMOTE2H" in get_kinds and "H2D" in get_kinds, (
        f"expected SWA REMOTE2H + H2D, got {get_kinds}"
    )
    remote_mr = engine.remote_cache_engine.match(seq)
    remote_node = getattr(remote_mr, "last_swa_node", None)
    assert remote_node is not None and remote_node.swa_lock_ref == 1
    swa_h2d = [op for op in swa_get_ops if op.transfer_type == TransferType.H2D][0]
    swa_remote2h = [
        op for op in swa_get_ops if op.transfer_type == TransferType.REMOTE2H
    ][0]
    assert swa_remote2h.op_id in swa_h2d.predecessors
    barrier = get_graph._op_map[get_end]
    reported2 = {swa_h2d.op_id} | set(barrier.predecessors)
    print(f"[verify] GET graph: SWA ops={get_kinds} (REMOTE staging chain)", flush=True)
    _run_graph(
        te,
        get_graph,
        get_op_cb,
        get_cb,
        full_gpu_blocks=[get_full_gpu],
        swa_gpu_slot=SWA_GPU_SLOT,
        reported_op_ids=reported2,
    )
    print("[verify] GET done (SWA restored via REMOTE->CPU->GPU)", flush=True)
    if remote_node.swa_lock_ref != 0:
        print(
            f"[verify] FAIL REMOTE source SWA lock leaked (lock_ref={remote_node.swa_lock_ref})",
            flush=True,
        )
        failed = True
    else:
        print("[verify] OK    REMOTE source SWA lock released", flush=True)

    actual_sw = block_bytes(sw_pool, SWA_GPU_SLOT)
    byte_failed = actual_sw != expected_sw
    failed = failed or byte_failed
    if byte_failed:
        print("[verify] FAIL SWA byte mismatch after REMOTE staging", flush=True)
    else:
        print(
            f"[verify] OK    SWA: {len(expected_sw)} bytes match "
            "GPU->CPU/REMOTE->(evict CPU)->REMOTE2H->H2D->GPU",
            flush=True,
        )

    staging_used = engine.cpu_cache_engine.swa_pool.num_used
    if staging_used != 0:
        print(
            f"[verify] FAIL transient CPU staging slot leaked (num_used={staging_used})",
            flush=True,
        )
        failed = True
    else:
        print("[verify] OK    transient CPU staging slot freed (no leak)", flush=True)

    te.shutdown()
    if failed:
        return 4
    print("[verify] PASS: REMOTE SWA staging byte-exact, transient slot freed", flush=True)
    return 0


pytestmark = pytest.mark.e2e


@pytest.mark.skipif(not _remote_e2e_enabled(), reason="set FLEXKV_RUN_REMOTE_E2E=1")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SWA REMOTE staging e2e needs a GPU")
@pytest.mark.skipif(
    transfer_worker.transfer_kv_blocks_remote is None,
    reason="requires CFS-enabled build with transfer_kv_blocks_remote",
)
def test_swa_remote_staging_e2e_byte_exact():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
