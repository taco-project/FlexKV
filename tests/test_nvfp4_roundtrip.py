"""Byte-exact round-trip test for FlexKV nvfp4 KV cache offload.

vLLM stores an nvfp4 KV cache as a single ``torch.uint8`` tensor whose per-head
last dim is the packed width ``full_dim = head_size//2 + head_size//16`` (fp4
data + fp8 block scales, inline). FlexKV mirrors that tensor byte-for-byte to
CPU. This test drives the real production offload path:

    KVTPClient.register_to_server  ->  KVManager.put_async (D2H)
    clear GPU  ->  KVManager.get_match + launch (H2D)  ->  verify

using uint8 GPU tensors shaped exactly like vLLM's nvfp4 output, and a
deterministic *byte* pattern (the float-hash verifier in common_utils cannot
represent uint8). A perfect round trip proves the packed nvfp4 bytes (data AND
fp8 scales) survive GPU->CPU->GPU intact.

The CPU cache is sized to honour the project constraint of cpu_cache_gb <= 1
(here far below 1 GB — a few MB — since only a handful of blocks are needed).

Run directly (single GPU):  python tests/test_nvfp4_roundtrip.py
"""
import sys
import time
import multiprocessing as mp

import torch

from flexkv.common.config import ModelConfig, CacheConfig, RankInfo
from flexkv.common.config import convert_to_block_num
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.request import KVResponseStatus
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.kvmanager import KVManager
from flexkv.server.client import KVTPClient
from flexkv.integration.config import nvfp4_kv_cache_full_dim
from flexkv.common.debug import flexkv_logger


# ---- Model shape: a small nvfp4 model (matches Qwen2.5-0.5B head_dim=64) ----
LOGICAL_HEAD_SIZE = 64                      # vLLM logical head_size
PACKED_HEAD_SIZE = nvfp4_kv_cache_full_dim(LOGICAL_HEAD_SIZE)  # 64//2 + 64//16 = 36
NUM_LAYERS = 4
NUM_KV_HEADS = 2
TOKENS_PER_BLOCK = 16
NUM_GPU_BLOCKS = 64
BLOCKS_PER_REQUEST = 16


def _byte_value(layer_id: int, kv_id: int, block_id: int) -> int:
    """Deterministic non-zero byte for a (layer, kv, block) triple."""
    return ((layer_id * 37 + kv_id * 101 + int(block_id) * 7) % 251) + 1


def run_tp_client(server_recv_port, conn):
    """Subprocess: create uint8 nvfp4-shaped GPU KV tensors, register them with
    the FlexKV server, and hand IPC handles back to the parent."""
    try:
        device_id = 0
        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=NUM_LAYERS,
            num_block=NUM_GPU_BLOCKS,
            tokens_per_block=TOKENS_PER_BLOCK,
            num_head=NUM_KV_HEADS,
            head_size=PACKED_HEAD_SIZE,
            is_mla=False,
        )
        # LAYERFIRST per-layer tensor: (kv_dim=2, num_block, tpb, num_head, head_size)
        gpu_blocks = []
        for _ in range(NUM_LAYERS):
            gpu_blocks.append(
                torch.zeros(tuple(gpu_layout.kv_shape[1:]),
                            dtype=torch.uint8, device=f"cuda:{device_id}")
            )
        tp_client = KVTPClient(server_recv_port, dp_client_id=0, pp_rank=0,
                               device_id=device_id)
        tp_client.register_to_server(gpu_blocks, gpu_layout)
        conn.send([TensorSharedHandle(t) for t in gpu_blocks])
        conn.close()
        while True:
            time.sleep(1)
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        try:
            conn.send(None)
            conn.close()
        except Exception:
            pass


def fill_blocks(gpu_tensors, block_ids):
    """Fill each (layer, kv, block) slice with its deterministic byte value."""
    for layer_id, t in enumerate(gpu_tensors):          # t: (2, nblk, tpb, nhead, hs)
        for kv_id in range(2):
            for b in block_ids:
                t[kv_id, int(b), :, :, :] = _byte_value(layer_id, kv_id, b)


def clear_blocks(gpu_tensors, block_ids):
    for t in gpu_tensors:
        for kv_id in range(2):
            for b in block_ids:
                t[kv_id, int(b), :, :, :] = 0


def verify_blocks(gpu_tensors, block_ids) -> bool:
    ok = True
    for layer_id, t in enumerate(gpu_tensors):
        for kv_id in range(2):
            for b in block_ids:
                expected = _byte_value(layer_id, kv_id, b)
                sl = t[kv_id, int(b), :, :, :]
                if not torch.all(sl == expected):
                    ok = False
                    ndiff = int((sl != expected).sum().item())
                    print(f"  MISMATCH layer={layer_id} kv={kv_id} block={int(b)}: "
                          f"expected byte {expected}, {ndiff}/{sl.numel()} bytes differ")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0

    # --- honour cpu_cache_gb <= 1 (compute blocks from a 1 GB budget) ---
    model_config = ModelConfig(
        num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS,
        head_size=PACKED_HEAD_SIZE, use_mla=False,
        dtype=torch.uint8, tp_size=1, dp_size=1,
    )
    rank_info = RankInfo(model_config=model_config, tp_rank=0,
                         pp_start_layer=0, pp_end_layer=NUM_LAYERS)
    block_bytes = rank_info.token_size_in_bytes_per_pp_stage * TOKENS_PER_BLOCK
    max_blocks_1gb = convert_to_block_num(1, block_bytes)   # blocks that fit in 1 GB
    # Only a handful are actually needed; cap well under the 1 GB ceiling to keep
    # RAM usage tiny while still satisfying "num_cpu_blocks >= num_gpu_blocks"
    # (so we expect a 100% cache hit).
    num_cpu_blocks = min(max_blocks_1gb, 256)
    assert num_cpu_blocks >= NUM_GPU_BLOCKS
    print(f"[cfg] packed head_size={PACKED_HEAD_SIZE} (logical {LOGICAL_HEAD_SIZE}), "
          f"dtype=uint8, block_bytes={block_bytes}, "
          f"1GB budget -> {max_blocks_1gb} blocks; using num_cpu_blocks={num_cpu_blocks} "
          f"({num_cpu_blocks*block_bytes/1024/1024:.2f} MB, <= 1 GB)")

    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        enable_cpu=True, enable_ssd=False,
        num_cpu_blocks=num_cpu_blocks,
    )

    kvmanager = KVManager(model_config=model_config, cache_config=cache_config,
                          dp_client_id=0)
    kvmanager.start()

    mp_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe()
    proc = mp_ctx.Process(target=run_tp_client,
                          args=(kvmanager.gpu_register_port, child_conn),
                          daemon=True)
    proc.start()
    shared = parent_conn.recv()
    parent_conn.close()
    if shared is None:
        print("FAIL: tp_client failed to register GPU blocks")
        kvmanager.shutdown()
        return 1
    gpu_tensors = [h.get_tensor() for h in shared]

    while not kvmanager.is_ready():
        time.sleep(0.5)
        flexkv_logger.info("waiting for flexkv to be ready")

    # Deterministic token ids for one request covering BLOCKS_PER_REQUEST blocks.
    torch.manual_seed(1234)
    token_ids = torch.randint(0, 100,
                              (BLOCKS_PER_REQUEST * TOKENS_PER_BLOCK,),
                              dtype=torch.int64)
    block_ids = torch.arange(0, BLOCKS_PER_REQUEST, dtype=torch.int64)
    slot_mapping = block_ids.repeat_interleave(TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK

    print("=== PUT (D2H offload of packed nvfp4 bytes) ===")
    fill_blocks(gpu_tensors, block_ids)
    torch.cuda.synchronize()
    put_id = kvmanager.put_async(token_ids=token_ids, slot_mapping=slot_mapping,
                                 token_mask=None, namespace=None)
    res = kvmanager.wait([put_id], completely=True)
    assert res[put_id].status == KVResponseStatus.SUCCESS, f"PUT failed: {res[put_id].status}"

    print("=== clear GPU, then GET (H2D reload from FlexKV) ===")
    clear_blocks(gpu_tensors, block_ids)
    torch.cuda.synchronize()
    assert torch.all(gpu_tensors[0][0, 0] == 0), "clear did not zero GPU blocks"

    get_id, matched = kvmanager.get_match(token_ids=token_ids, token_mask=None,
                                          namespace=None)
    num_matched = int(matched.sum().item())
    print(f"  get_match matched {num_matched} / {len(token_ids)} tokens")
    assert num_matched == len(token_ids), (
        f"expected full cache hit, matched only {num_matched}/{len(token_ids)}")
    kvmanager.launch(get_id, slot_mapping)
    gres = kvmanager.wait([get_id], completely=True)
    assert gres[get_id].status == KVResponseStatus.SUCCESS, f"GET failed: {gres[get_id].status}"
    torch.cuda.synchronize()

    print("=== VERIFY byte-exact round trip ===")
    ok = verify_blocks(gpu_tensors, block_ids)

    if proc.is_alive():
        proc.terminate(); proc.join(timeout=5)
    kvmanager.shutdown()

    if ok:
        print("=== NVFP4 FLEXKV ROUND-TRIP OK: all packed uint8 bytes match ===")
        return 0
    print("=== NVFP4 FLEXKV ROUND-TRIP FAILED ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
