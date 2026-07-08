"""
End-to-end smoke + correctness test for COMBINED main-KV + SWA transfer paths.

Verifies that a single TransferEngine instance can drive main-KV and SWA
transfers in parallel within one transfer graph, and that the data survives a
GPU -> CPU -> GPU roundtrip byte-for-byte.

Flow:
  1. Seed main-KV GPU pool block_mk_src with magic_mk;
     seed SWA     GPU pool block_sw_src with magic_sw.
  2. PUT graph: ONE main-KV D2H op + ONE SWA D2H op (no deps, run in parallel)
     copy each into its CPU pool slot (block_mk_dst / block_sw_dst).
  3. Zero out both GPU pools to prove the next step actually reads from CPU.
  4. GET graph: ONE main-KV H2D op + ONE SWA H2D op, copying back to a
     DIFFERENT GPU block (block_mk_back / block_sw_back) so we don't reuse the
     original seed location.
  5. Assert byte-exact: magic_mk == GPU main-KV block_mk_back,
                       magic_sw == GPU SWA     block_sw_back.

Run INSIDE the container:
    python3 verify_swa_end_to_end.py
"""

import sys
import time

import numpy as np
import pytest
import torch

from flexkv.common.config import (
    CacheConfig,
    ModelConfig,
    SWAPoolConfig,
)
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import (
    DeviceType,
    TransferOp,
    TransferOpGraph,
    TransferType,
    WorkerKey,
)
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine


# -------------------------- shared geometry --------------------------
NUM_LAYERS = 4
NUM_BLOCKS_GPU = 16
NUM_BLOCKS_CPU = 16
TOKENS_PER_BLOCK = 32
BYTES_PER_TOKEN_PER_LAYER = 64
DEVICE_ID = 0

MK_SRC = 5         # main-KV GPU block we seed
MK_DST_CPU = 2     # main-KV CPU slot
MK_BACK = 11       # main-KV GPU block we restore into (different from MK_SRC)

SW_SRC = 7         # SWA GPU block we seed
SW_DST_CPU = 3     # SWA CPU slot
SW_BACK = 13       # SWA GPU block we restore into (different from SW_SRC)


def make_pool() -> torch.Tensor:
    return torch.zeros(
        (NUM_LAYERS, NUM_BLOCKS_GPU, TOKENS_PER_BLOCK, BYTES_PER_TOKEN_PER_LAYER),
        dtype=torch.uint8,
        device=f"cuda:{DEVICE_ID}",
    )


def seed_block(pool: torch.Tensor, block: int, salt: int) -> None:
    """Write a deterministic per-layer/per-token pattern into ``pool[:, block]``."""
    for layer in range(NUM_LAYERS):
        plane = torch.zeros(
            (TOKENS_PER_BLOCK, BYTES_PER_TOKEN_PER_LAYER), dtype=torch.uint8,
        )
        for tok in range(TOKENS_PER_BLOCK):
            for b in range(BYTES_PER_TOKEN_PER_LAYER):
                plane[tok, b] = ((layer * 31 + tok + salt) ^ b) & 0xFF
        pool[layer, block].copy_(plane.to(pool.device))


def block_bytes(pool: torch.Tensor, block: int) -> bytes:
    """Flatten ``pool[:, block]`` across all layers into a single bytes blob."""
    chunks = [pool[layer, block].contiguous().view(-1).cpu().clone()
              for layer in range(NUM_LAYERS)]
    return torch.cat(chunks).numpy().tobytes()


def make_layerfirst_layout() -> KVCacheLayout:
    return KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=NUM_LAYERS,
        num_block=NUM_BLOCKS_GPU,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_head=1,
        head_size=BYTES_PER_TOKEN_PER_LAYER,
        is_mla=True,
    )


def wait_for_op(te: TransferEngine, op_ids: set, timeout_s: float = 30.0) -> set:
    """Block until all given op_ids appear on completed_queue (or timeout)."""
    pending = set(op_ids)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        completed = te.get_completed_graphs_and_ops(timeout=1.0)
        for c in completed:
            pending.discard(c.op_id)
            print(f"[verify] completed event: graph_id={c.graph_id} op_id={c.op_id}",
                  flush=True)
    if pending:
        raise TimeoutError(f"Ops {pending} did not complete in {timeout_s}s")
    return set(op_ids)


def main() -> int:
    if not torch.cuda.is_available():
        print("[verify] CUDA not available", flush=True)
        return 2
    torch.cuda.set_device(DEVICE_ID)
    print(f"[verify] device cuda:{DEVICE_ID}", flush=True)

    # ----- 1. allocate two GPU pools, seed magic on chosen src blocks -----
    mk_pool = make_pool()
    sw_pool = make_pool()
    seed_block(mk_pool, MK_SRC, salt=0xA1)
    seed_block(sw_pool, SW_SRC, salt=0xB2)
    expected_mk = block_bytes(mk_pool, MK_SRC)
    expected_sw = block_bytes(sw_pool, SW_SRC)
    print(f"[verify] seeded MAIN_KV block {MK_SRC} ({len(expected_mk)} bytes), "
          f"SWA block {SW_SRC} ({len(expected_sw)} bytes)", flush=True)

    mk_handles = [TensorSharedHandle(mk_pool[l].contiguous(), DEVICE_ID)
                  for l in range(NUM_LAYERS)]
    sw_handles = [TensorSharedHandle(sw_pool[l].contiguous(), DEVICE_ID)
                  for l in range(NUM_LAYERS)]

    # ----- 2. configs -----
    model_config = ModelConfig(
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_size=BYTES_PER_TOKEN_PER_LAYER,
        use_mla=True,
        dtype=torch.uint8,
        tp_size=1, pp_size=1, dp_size=1, cp_size=1,
    )
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=False,
        num_cpu_blocks=NUM_BLOCKS_CPU,
        swa=SWAPoolConfig(
            enabled=True,
            num_slots=NUM_BLOCKS_CPU,
            num_swa_layers=NUM_LAYERS,
            bytes_per_token_per_layer=BYTES_PER_TOKEN_PER_LAYER,
            pin_memory=True,
        ),
    )

    # ----- 3. StorageEngine + register both pools -----
    se = StorageEngine(model_config, cache_config, num_layers_per_pp_stage=NUM_LAYERS)
    assert se.has_storage_handle(DeviceType.CPU, device_id=0, is_swa=True), \
        "SWA CPU pool missing after __init__"

    layout = make_layerfirst_layout()
    se.register_gpu_blocks(mk_handles, layout, device_id=DEVICE_ID, dtype=torch.uint8)
    se.register_swa_gpu_blocks(sw_handles, layout, device_id=DEVICE_ID,
                               dtype=torch.uint8)
    print("[verify] registered main-KV + SWA GPU pools", flush=True)

    # ----- 4. TransferEngine -----
    worker_key = WorkerKey(dp_client_id=0, pp_rank=0)
    te = TransferEngine(
        gpu_handles={worker_key: [se.get_storage_handle(DeviceType.GPU,
                                                       device_id=DEVICE_ID)]},
        model_config=model_config,
        cache_config=cache_config,
        cpu_handle=se.get_storage_handle(DeviceType.CPU),
        swa_gpu_handles={worker_key: [se.get_swa_storage_handle(DEVICE_ID)]},
        swa_cpu_handle=se.get_storage_handle(DeviceType.CPU, device_id=0,
                                             is_swa=True),
    )
    te.start()
    print("[verify] TransferEngine started", flush=True)

    # ----- 5. PUT graph: main-KV D2H || SWA D2H (parallel) -----
    put_graph = TransferOpGraph()
    op_mk_d2h = TransferOp(
        graph_id=put_graph.graph_id,
        transfer_type=TransferType.D2H,
        src_block_ids=np.array([MK_SRC], dtype=np.int64),
        dst_block_ids=np.array([MK_DST_CPU], dtype=np.int64),
        dp_client_id=0,
        is_swa=False,
    )
    op_sw_d2h = TransferOp(
        graph_id=put_graph.graph_id,
        transfer_type=TransferType.D2H,
        src_block_ids=np.array([SW_SRC], dtype=np.int64),
        dst_block_ids=np.array([SW_DST_CPU], dtype=np.int64),
        dp_client_id=0,
        is_swa=True,
    )
    put_graph.add_transfer_op(op_mk_d2h)
    put_graph.add_transfer_op(op_sw_d2h)
    print(f"[verify] PUT graph: main-KV D2H op_id={op_mk_d2h.op_id} || "
          f"SWA D2H op_id={op_sw_d2h.op_id}", flush=True)
    te.submit_transfer_graph(put_graph)
    wait_for_op(te, {op_mk_d2h.op_id, op_sw_d2h.op_id}, timeout_s=30.0)
    print("[verify] PUT phase done", flush=True)

    # ----- 6. zero out GPU pools to make sure GET actually reads from CPU -----
    mk_pool.zero_()
    sw_pool.zero_()
    torch.cuda.synchronize()
    # paranoia: confirm seed location is now zero
    assert int(mk_pool[:, MK_SRC].sum().item()) == 0
    assert int(sw_pool[:, SW_SRC].sum().item()) == 0
    print("[verify] zeroed both GPU pools", flush=True)

    # ----- 7. GET graph: main-KV H2D || SWA H2D into DIFFERENT blocks -----
    get_graph = TransferOpGraph()
    op_mk_h2d = TransferOp(
        graph_id=get_graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([MK_DST_CPU], dtype=np.int64),
        dst_block_ids=np.array([MK_BACK], dtype=np.int64),
        dp_client_id=0,
        is_swa=False,
    )
    op_sw_h2d = TransferOp(
        graph_id=get_graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([SW_DST_CPU], dtype=np.int64),
        dst_block_ids=np.array([SW_BACK], dtype=np.int64),
        dp_client_id=0,
        is_swa=True,
    )
    get_graph.add_transfer_op(op_mk_h2d)
    get_graph.add_transfer_op(op_sw_h2d)
    print(f"[verify] GET graph: main-KV H2D op_id={op_mk_h2d.op_id} || "
          f"SWA H2D op_id={op_sw_h2d.op_id}", flush=True)
    te.submit_transfer_graph(get_graph)
    wait_for_op(te, {op_mk_h2d.op_id, op_sw_h2d.op_id}, timeout_s=30.0)
    torch.cuda.synchronize()
    print("[verify] GET phase done", flush=True)

    # ----- 8. byte-exact compare -----
    actual_mk = block_bytes(mk_pool, MK_BACK)
    actual_sw = block_bytes(sw_pool, SW_BACK)

    failed = False
    if actual_mk != expected_mk:
        off = next((i for i, (a, b) in enumerate(zip(actual_mk, expected_mk))
                    if a != b), -1)
        print(f"[verify] FAIL main-KV: byte mismatch at offset {off}", flush=True)
        failed = True
    else:
        print(f"[verify] OK    main-KV: {len(expected_mk)} bytes match "
              f"GPU[{MK_SRC}] -> CPU[{MK_DST_CPU}] -> GPU[{MK_BACK}]", flush=True)

    if actual_sw != expected_sw:
        off = next((i for i, (a, b) in enumerate(zip(actual_sw, expected_sw))
                    if a != b), -1)
        print(f"[verify] FAIL SWA: byte mismatch at offset {off}", flush=True)
        failed = True
    else:
        print(f"[verify] OK    SWA    : {len(expected_sw)} bytes match "
              f"GPU[{SW_SRC}] -> CPU[{SW_DST_CPU}] -> GPU[{SW_BACK}]", flush=True)

    te.shutdown()

    if failed:
        return 4
    print("[verify] PASS: end-to-end main-KV + SWA roundtrip byte-exact",
          flush=True)
    return 0


@pytest.mark.e2e
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SWA dispatch e2e needs a GPU")
def test_swa_dispatch_byte_exact():
    """pytest entry: data-plane SWA dispatch byte-exact roundtrip (main() -> 0).
    Collected by `pytest -m e2e`; also runnable standalone."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
