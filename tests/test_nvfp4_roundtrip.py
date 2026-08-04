"""Byte-exact round-trip test for FlexKV nvfp4 KV cache offload.

vLLM stores an nvfp4 KV cache as a ``torch.uint8`` tensor.  Before the packed-KV
layout refactor it used ``(blocks, 2, tokens, heads, full_dim)``; current
FlashInfer uses ``(blocks, 2 * heads, tokens, full_dim)``.  This test asks the
installed vLLM backend for its real shape, builds the same NHD strides, and
drives the production offload path:

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
import os

import torch

from flexkv.common.config import ModelConfig, CacheConfig, RankInfo
from flexkv.common.config import convert_to_block_num
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.request import KVResponseStatus
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.kvmanager import KVManager
from flexkv.server.client import KVTPClient
from flexkv.common.debug import flexkv_logger


# ---- Model shape: a small nvfp4 model (matches Qwen2.5-0.5B head_dim=64) ----
LOGICAL_HEAD_SIZE = 64                      # vLLM logical head_size
NUM_LAYERS = 4
NUM_KV_HEADS = 2
TOKENS_PER_BLOCK = 16
NUM_GPU_BLOCKS = 64
BLOCKS_PER_REQUEST = 16


def _get_vllm_nvfp4_shape() -> tuple[int, ...]:
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ["VLLM_KV_CACHE_LAYOUT"] = "NHD"
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    return tuple(FlashInferBackend.get_kv_cache_shape(
        NUM_GPU_BLOCKS,
        TOKENS_PER_BLOCK,
        NUM_KV_HEADS,
        LOGICAL_HEAD_SIZE,
        "nvfp4",
    ))


def _model_layout_from_vllm_shape(
    cache_shape: tuple[int, ...],
) -> tuple[int, int, bool]:
    if len(cache_shape) == 4:
        return cache_shape[1], cache_shape[3], True
    if len(cache_shape) == 5:
        return cache_shape[3], cache_shape[4], False
    raise ValueError(f"unsupported vLLM NVFP4 cache shape: {cache_shape}")


def _make_gpu_layout(cache_shape: tuple[int, ...]) -> KVCacheLayout:
    if len(cache_shape) == 4:
        layout_type = KVCacheLayoutType.LAYERFIRST
        num_heads = cache_shape[1]
        head_size = cache_shape[3]
        packed_kv = True
    elif cache_shape[0] == NUM_GPU_BLOCKS and cache_shape[1] == 2:
        layout_type = KVCacheLayoutType.LAYERBLOCK
        num_heads = cache_shape[3]
        head_size = cache_shape[4]
        packed_kv = False
    elif cache_shape[0] == 2 and cache_shape[1] == NUM_GPU_BLOCKS:
        layout_type = KVCacheLayoutType.LAYERFIRST
        num_heads = cache_shape[3]
        head_size = cache_shape[4]
        packed_kv = False
    else:
        raise ValueError(f"unsupported vLLM NVFP4 cache shape: {cache_shape}")
    return KVCacheLayout(
        type=layout_type,
        num_layer=NUM_LAYERS,
        num_block=NUM_GPU_BLOCKS,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_head=num_heads,
        head_size=head_size,
        is_mla=False,
        packed_kv=packed_kv,
    )


def _make_vllm_tensor(cache_shape: tuple[int, ...], device_id: int) -> torch.Tensor:
    if len(cache_shape) == 4:
        physical_shape = (
            cache_shape[0], cache_shape[2], cache_shape[1], cache_shape[3])
        return torch.zeros(
            physical_shape, dtype=torch.uint8, device=f"cuda:{device_id}"
        ).permute(0, 2, 1, 3)
    return torch.zeros(cache_shape, dtype=torch.uint8, device=f"cuda:{device_id}")


def _block_slice(tensor: torch.Tensor, block_id: int) -> torch.Tensor:
    block_dim = 0 if tensor.shape[0] == NUM_GPU_BLOCKS else 1
    return tensor.select(block_dim, int(block_id))


def run_tp_client(server_recv_port, conn, cache_shape):
    """Subprocess: create uint8 nvfp4-shaped GPU KV tensors, register them with
    the FlexKV server, and hand IPC handles back to the parent."""
    try:
        device_id = 0
        gpu_layout = _make_gpu_layout(cache_shape)
        gpu_blocks = [
            _make_vllm_tensor(cache_shape, device_id)
            for _ in range(NUM_LAYERS)
        ]
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
    """Fill each physical vLLM block with a deterministic non-zero byte pattern."""
    for layer_id, tensor in enumerate(gpu_tensors):
        for block_id in block_ids:
            block = _block_slice(tensor, int(block_id))
            values = torch.arange(
                block.numel(), dtype=torch.int64, device=block.device
            ).reshape(block.shape)
            values = (values + layer_id * 37 + int(block_id) * 7) % 251 + 1
            block.copy_(values.to(torch.uint8))


def clear_blocks(gpu_tensors, block_ids):
    for tensor in gpu_tensors:
        for block_id in block_ids:
            _block_slice(tensor, int(block_id)).zero_()


def verify_blocks(gpu_tensors, expected_tensors, block_ids) -> bool:
    ok = True
    for layer_id, (tensor, expected_tensor) in enumerate(
        zip(gpu_tensors, expected_tensors, strict=True)
    ):
        for block_id in block_ids:
            got = _block_slice(tensor, int(block_id))
            expected = _block_slice(expected_tensor, int(block_id))
            if not torch.equal(got, expected):
                ok = False
                ndiff = int((got != expected).sum().item())
                print(
                    f"  MISMATCH layer={layer_id} block={int(block_id)}: "
                    f"{ndiff}/{got.numel()} bytes differ"
                )
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    torch.cuda.set_device(0)
    cache_shape = _get_vllm_nvfp4_shape()
    model_num_heads, model_head_size, packed_kv = (
        _model_layout_from_vllm_shape(cache_shape)
    )

    # --- honour cpu_cache_gb <= 1 (compute blocks from a 1 GB budget) ---
    model_config = ModelConfig(
        num_layers=NUM_LAYERS, num_kv_heads=model_num_heads,
        head_size=model_head_size, use_mla=False, packed_kv=packed_kv,
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
    print(f"[cfg] vLLM shape={cache_shape}, model num_heads={model_num_heads}, "
          f"model head_size={model_head_size}, "
          f"packed_kv={packed_kv} (logical head_size={LOGICAL_HEAD_SIZE}), "
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
                          args=(kvmanager.gpu_register_port, child_conn, cache_shape),
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
    expected_tensors = [tensor.clone() for tensor in gpu_tensors]
    torch.cuda.synchronize()
    put_id = kvmanager.put_async(token_ids=token_ids, slot_mapping=slot_mapping,
                                 token_mask=None, namespace=None)
    res = kvmanager.wait([put_id], completely=True)
    assert res[put_id].status == KVResponseStatus.SUCCESS, f"PUT failed: {res[put_id].status}"

    print("=== clear GPU, then GET (H2D reload from FlexKV) ===")
    clear_blocks(gpu_tensors, block_ids)
    torch.cuda.synchronize()
    assert torch.count_nonzero(_block_slice(gpu_tensors[0], 0)).item() == 0, (
        "clear did not zero GPU blocks")

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
    ok = verify_blocks(gpu_tensors, expected_tensors, block_ids)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
    kvmanager.shutdown()

    if ok:
        print("=== NVFP4 FLEXKV ROUND-TRIP OK: all packed uint8 bytes match ===")
        return 0
    print("=== NVFP4 FLEXKV ROUND-TRIP FAILED ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
