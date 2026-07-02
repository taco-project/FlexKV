"""P3: KVManager DSv4 three-group + SWA layerwise integration test.

Exercises the full in-process path:

  TP client registers c4 / c128 / indexer groups + SWA GPU pool
  PUT: main KV D2H (+ parallel SWA D2H / H2DISK in the put graph)
  GET: layerwise batch (fused DISK2H + multi-group H2D + SWA sidecar)
  Verify restored GPU bytes for c4 / c128 / indexer and SWA

Run:
    pytest tests/test_kvmanager_dsv4_layerwise_swa.py -v
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch
import multiprocessing as mp

from flexkv.common.config import (
    CacheConfig,
    LayerGroupSpec,
    ModelConfig,
    SWAPoolConfig,
    GLOBAL_CONFIG_FROM_ENV,
)
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.request import KVResponseStatus
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.kvmanager import KVManager
from flexkv.server.client import KVTPClient

from common_utils import (
    block_ids_2_slot_mapping,
    generate_request_pair,
    skip_if_insufficient_gpus,
)
from test_kvmanager import (
    GPUIndexerCacheVerifier,
    _mock_sglang_eventfd_client,
    shutdown_tp_client,
)

# -------------------------- DSv4 geometry (aligned with layerwise dsv4 tests) ----------
NUM_ORIGINAL_LAYERS = 8
C4_LAYER_IDS = [0, 1, 2, 3]
C128_LAYER_IDS = [4, 5, 6, 7]
C4_HEAD_SIZE = 64
C128_HEAD_SIZE = 32
INDEXER_HEAD_SIZE = 40
SWA_BYTES_PER_TOKEN = 128
C4_COMPRESS_RATIO = 4


def _dsv4_layer_groups() -> List[LayerGroupSpec]:
    return [
        LayerGroupSpec(
            num_layers=len(C4_LAYER_IDS),
            num_kv_heads=1,
            head_size=C4_HEAD_SIZE,
            layer_indices=C4_LAYER_IDS,
            dtype=torch.uint8,
            compress_ratio=C4_COMPRESS_RATIO,
        ),
        LayerGroupSpec(
            num_layers=len(C128_LAYER_IDS),
            num_kv_heads=1,
            head_size=C128_HEAD_SIZE,
            layer_indices=C128_LAYER_IDS,
            dtype=torch.uint8,
            compress_ratio=1,
        ),
        LayerGroupSpec(
            num_layers=len(C4_LAYER_IDS),
            num_kv_heads=1,
            head_size=INDEXER_HEAD_SIZE,
            layer_indices=C4_LAYER_IDS,
            dtype=torch.uint8,
            compress_ratio=C4_COMPRESS_RATIO,
        ),
    ]


def _group_layout(
    g: LayerGroupSpec,
    num_blocks: int,
    tokens_per_block: int,
) -> KVCacheLayout:
    tpb_g = tokens_per_block // g.compress_ratio
    return KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=g.num_layers,
        num_block=num_blocks,
        tokens_per_block=tpb_g,
        num_head=g.num_kv_heads,
        head_size=g.head_size,
        is_mla=True,
    )


def _swa_gpu_layout(num_blocks: int, tokens_per_block: int) -> KVCacheLayout:
    return KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=NUM_ORIGINAL_LAYERS,
        num_block=num_blocks,
        tokens_per_block=tokens_per_block,
        num_head=1,
        head_size=SWA_BYTES_PER_TOKEN,
        is_mla=True,
    )


def _make_group_gpu_blocks(
    g: LayerGroupSpec,
    num_blocks: int,
    tokens_per_block: int,
    device_id: int,
) -> List[torch.Tensor]:
    """One 4-D tensor per local layer: (num_blocks, tpb_g, num_heads, head_size)."""
    tpb_g = tokens_per_block // g.compress_ratio
    return [
        torch.empty(
            num_blocks,
            tpb_g,
            g.num_kv_heads,
            g.head_size,
            dtype=g.dtype,
            device=device_id,
        )
        for _ in range(g.num_layers)
    ]


class GPUCompressedGroupVerifier:
    """Hash-based verifier for one multi-group member (c4 or c128)."""

    def __init__(
        self,
        shared_blocks,
        group: LayerGroupSpec,
        gpu_layout: KVCacheLayout,
        tp_size: int,
        tokens_per_block: int,
    ) -> None:
        if isinstance(shared_blocks[0][0], torch.Tensor):
            self.gpu_blocks = shared_blocks
        else:
            self.gpu_blocks = [
                [h.get_tensor() for h in handles]
                for handles in shared_blocks
            ]
        self.group = group
        self.gpu_layout = gpu_layout
        self.tp_size = tp_size
        self.tokens_per_block = tokens_per_block
        self.dtype = group.dtype

    def _hash_value(self, orig_layer: int, token_ids: Sequence[int]) -> int:
        token_hash = 0
        for i, tok in enumerate(token_ids):
            token_hash += int(tok) * (i + 11)
        return ((orig_layer + 1) * 37 + token_hash) % 251 + 1

    def fill_gpu_blocks(
        self,
        block_ids: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> None:
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.int64)

        blocks_per_req = len(block_ids)
        for tp_id in range(self.tp_size):
            for local_id, orig_layer in enumerate(self.group.layer_indices):
                tensor = self.gpu_blocks[tp_id][local_id]
                for block_idx, block_id in enumerate(block_ids):
                    start = block_idx * self.tokens_per_block
                    end = start + self.tokens_per_block
                    val = self._hash_value(orig_layer, token_ids[start:end].tolist())
                    tensor[block_id, :, :] = val

    def clear_gpu_blocks(self, block_ids: torch.Tensor) -> None:
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)
        for tp_id in range(self.tp_size):
            for local_id in range(self.group.num_layers):
                self.gpu_blocks[tp_id][local_id][block_ids, :, :] = 0

    def verify_gpu_blocks(
        self,
        block_ids: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> bool:
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.int64)

        for tp_id in range(self.tp_size):
            for local_id, orig_layer in enumerate(self.group.layer_indices):
                tensor = self.gpu_blocks[tp_id][local_id]
                for block_idx, block_id in enumerate(block_ids):
                    start = block_idx * self.tokens_per_block
                    end = start + self.tokens_per_block
                    expected = self._hash_value(
                        orig_layer, token_ids[start:end].tolist(),
                    )
                    actual = tensor[block_id, :, :]
                    exp_t = torch.full_like(actual, expected)
                    if not torch.equal(actual, exp_t):
                        return False
        return True


class GPUSWACacheVerifier:
    """Per-original-layer SWA GPU pool verifier (uint8, full tpb)."""

    def __init__(
        self,
        shared_swa_blocks,
        swa_layout: KVCacheLayout,
        tp_size: int,
        tokens_per_block: int,
    ) -> None:
        if isinstance(shared_swa_blocks[0][0], torch.Tensor):
            self.gpu_blocks = shared_swa_blocks
        else:
            self.gpu_blocks = [
                [h.get_tensor() for h in handles]
                for handles in shared_swa_blocks
            ]
        self.num_layers = swa_layout.num_layer
        self.tp_size = tp_size
        self.tokens_per_block = tokens_per_block

    def _hash_value(self, orig_layer: int, token_ids: Sequence[int]) -> int:
        token_hash = sum(int(t) * (i + 3) for i, t in enumerate(token_ids))
        return ((orig_layer + 5) * 19 + token_hash) % 251 + 1

    def fill_gpu_blocks(
        self,
        block_ids: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> None:
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.int64)

        for tp_id in range(self.tp_size):
            for orig_layer in range(self.num_layers):
                tensor = self.gpu_blocks[tp_id][orig_layer]
                for block_idx, block_id in enumerate(block_ids):
                    start = block_idx * self.tokens_per_block
                    end = start + self.tokens_per_block
                    val = self._hash_value(orig_layer, token_ids[start:end].tolist())
                    tensor[block_id, :, :] = val

    def clear_gpu_blocks(self, block_ids: torch.Tensor) -> None:
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)
        for tp_id in range(self.tp_size):
            for orig_layer in range(self.num_layers):
                self.gpu_blocks[tp_id][orig_layer][block_ids, :, :] = 0

    def verify_gpu_blocks(
        self,
        block_ids: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> bool:
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.int64)

        for tp_id in range(self.tp_size):
            for orig_layer in range(self.num_layers):
                tensor = self.gpu_blocks[tp_id][orig_layer]
                for block_idx, block_id in enumerate(block_ids):
                    start = block_idx * self.tokens_per_block
                    end = start + self.tokens_per_block
                    expected = self._hash_value(
                        orig_layer, token_ids[start:end].tolist(),
                    )
                    actual = tensor[block_id, :, :]
                    exp_t = torch.full_like(actual, expected)
                    if not torch.equal(actual, exp_t):
                        return False
        return True


def run_tp_client_with_dsv4(
    dp_client_id: int,
    tp_rank: int,
    server_recv_port: str,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    num_gpu_blocks: int,
    child_conn,
) -> None:
    """Spawn-side TP client: register c4/c128/indexer + SWA pools."""
    try:
        device_id = tp_rank + dp_client_id * model_config.tp_size
        tokens_per_block = cache_config.tokens_per_block
        layer_groups = model_config.layer_groups
        assert layer_groups is not None and len(layer_groups) == 3

        group_blocks: List[List[torch.Tensor]] = []
        group_layouts: List[KVCacheLayout] = []
        for g in layer_groups:
            group_blocks.append(
                _make_group_gpu_blocks(g, num_gpu_blocks, tokens_per_block, device_id)
            )
            group_layouts.append(_group_layout(g, num_gpu_blocks, tokens_per_block))

        swa_layout = _swa_gpu_layout(num_gpu_blocks, tokens_per_block)
        swa_blocks = [
            torch.empty(
                num_gpu_blocks,
                tokens_per_block,
                1,
                SWA_BYTES_PER_TOKEN,
                dtype=torch.uint8,
                device=device_id,
            )
            for _ in range(NUM_ORIGINAL_LAYERS)
        ]

        flat_kv_caches: List[torch.Tensor] = []
        for blocks in group_blocks:
            flat_kv_caches.extend(blocks)

        # Primary layout must use the *original* layer count (8), not the c4
        # group's local layer count (4).  TransferManager / LayerwiseWorker derive
        # num_layers_per_pp_stage from this field; SWA sidecar requires it to match
        # swa_layout.num_layer.
        primary_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=NUM_ORIGINAL_LAYERS,
            num_block=num_gpu_blocks,
            tokens_per_block=tokens_per_block,
            num_head=1,
            head_size=C4_HEAD_SIZE,
            is_mla=True,
        )

        tp_client = KVTPClient(
            gpu_register_port=server_recv_port + "_gpu_register",
            dp_client_id=dp_client_id,
            pp_rank=0,
            device_id=device_id,
        )
        tp_client.register_to_server(
            kv_caches=flat_kv_caches,
            kv_layout=primary_layout,
            layer_groups=layer_groups,
            gpu_layouts=group_layouts,
            handles_per_group=group_blocks,
            swa_caches=swa_blocks,
            swa_layout=swa_layout,
        )

        if child_conn is not None:
            child_conn.send({
                "c4": [TensorSharedHandle(t) for t in group_blocks[0]],
                "c128": [TensorSharedHandle(t) for t in group_blocks[1]],
                "indexer": [TensorSharedHandle(t) for t in group_blocks[2]],
                "swa": [TensorSharedHandle(t) for t in swa_blocks],
            })
            child_conn.close()

        while True:
            time.sleep(1)
    except Exception as e:
        print(f"[TP Client {tp_rank}] Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        if child_conn is not None:
            child_conn.send(None)
            child_conn.close()


def _run_dsv4_layerwise_swa_test(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    test_config: dict,
    *,
    test_label: str = "dsv4-layerwise-swa",
    eventfd_socket_path: str = "/tmp/flexkv_layerwise_eventfd.sock",
) -> None:
    tp_size = model_config.tp_size
    tokens_per_block = cache_config.tokens_per_block
    num_gpu_blocks = test_config["num_gpu_blocks"]
    block_per_request = test_config["requests_per_block"]
    initial_write_ratio = test_config["initial_write_ratio"]
    num_requests = num_gpu_blocks // block_per_request

    skip_if_insufficient_gpus(tp_size)

    model_config.num_layers = NUM_ORIGINAL_LAYERS
    model_config.num_kv_heads = 1
    model_config.head_size = C4_HEAD_SIZE
    model_config.dtype = torch.uint8
    model_config.use_mla = True
    model_config.layer_groups = _dsv4_layer_groups()

    kvmanager = KVManager(
        model_config=model_config,
        cache_config=cache_config,
        dp_client_id=0,
    )

    # Start mock SGLang only after KVManager is constructed but before workers
    # bind the socket — long StorageEngine init otherwise exhausts mock retries.
    eventfd_thread = threading.Thread(
        target=_mock_sglang_eventfd_client,
        args=(eventfd_socket_path, 0, 1, NUM_ORIGINAL_LAYERS),
        kwargs={"max_retries": 240, "retry_interval": 0.25},
        daemon=True,
    )
    eventfd_thread.start()

    kvmanager.start()

    mp_ctx = mp.get_context("spawn")
    pipe_connections = []
    tp_client_processes = []

    for tp_rank in range(tp_size):
        parent_conn, child_conn = mp_ctx.Pipe()
        pipe_connections.append(parent_conn)
        proc = mp_ctx.Process(
            target=run_tp_client_with_dsv4,
            args=(
                0, tp_rank, kvmanager.server_recv_port,
                model_config, cache_config, num_gpu_blocks, child_conn,
            ),
            daemon=True,
        )
        tp_client_processes.append(proc)
        proc.start()

    payloads_per_tp: List[Optional[dict]] = []
    for tp_rank, parent_conn in enumerate(pipe_connections):
        payload = parent_conn.recv()
        payloads_per_tp.append(payload)
        parent_conn.close()
        assert payload is not None, f"TP client {tp_rank} failed to register"

    layer_groups = model_config.layer_groups
    assert layer_groups is not None

    c4_verifier = GPUCompressedGroupVerifier(
        [p["c4"] for p in payloads_per_tp],  # type: ignore[index]
        layer_groups[0],
        _group_layout(layer_groups[0], num_gpu_blocks, tokens_per_block),
        tp_size,
        tokens_per_block,
    )
    c128_verifier = GPUCompressedGroupVerifier(
        [p["c128"] for p in payloads_per_tp],  # type: ignore[index]
        layer_groups[1],
        _group_layout(layer_groups[1], num_gpu_blocks, tokens_per_block),
        tp_size,
        tokens_per_block,
    )
    indexer_tpb_g = tokens_per_block // layer_groups[2].compress_ratio
    indexer_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=layer_groups[2].num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=indexer_tpb_g,
        num_head=1,
        head_size=INDEXER_HEAD_SIZE,
        is_mla=True,
    )
    indexer_verifier = GPUIndexerCacheVerifier(
        [p["indexer"] for p in payloads_per_tp],  # type: ignore[index]
        indexer_layout,
        tp_size,
        layer_groups[2].dtype,
    )
    swa_verifier = GPUSWACacheVerifier(
        [p["swa"] for p in payloads_per_tp],  # type: ignore[index]
        _swa_gpu_layout(num_gpu_blocks, tokens_per_block),
        tp_size,
        tokens_per_block,
    )
    while not kvmanager.is_ready():
        time.sleep(0.5)
    print(f"[Test] KVManager ({test_label}) is ready")

    request_pairs = [
        generate_request_pair(i, block_per_request, num_gpu_blocks, tokens_per_block, 1)
        for i in range(num_requests)
    ]
    initial_write_num = int(num_requests * initial_write_ratio)

    print(f"[Test] PUT ({test_label})...")
    for token_ids, block_ids, _ in request_pairs[:initial_write_num]:
        c4_verifier.fill_gpu_blocks(block_ids, token_ids)
        c128_verifier.fill_gpu_blocks(block_ids, token_ids)
        indexer_verifier.fill_gpu_blocks(block_ids, tokens_per_block, token_ids)
        swa_verifier.fill_gpu_blocks(block_ids, token_ids)

        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
        )
        put_results = kvmanager.wait([write_request], completely=True)
        assert put_results[write_request].status == KVResponseStatus.SUCCESS

        c4_verifier.clear_gpu_blocks(block_ids)
        c128_verifier.clear_gpu_blocks(block_ids)
        indexer_verifier.clear_gpu_blocks(block_ids)
        swa_verifier.clear_gpu_blocks(block_ids)

    print(f"[Test] GET layerwise ({test_label})...")
    batch_task_ids = []
    batch_slot_mappings = []
    req_id2block_ids: Dict[int, torch.Tensor] = {}
    req_id2token_ids: Dict[int, torch.Tensor] = {}

    for i in range(min(initial_write_num, num_requests)):
        token_ids, block_ids, _ = request_pairs[i]
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
        request_id, _ = kvmanager.get_match(token_ids=token_ids, token_mask=None)
        batch_task_ids.append(request_id)
        batch_slot_mappings.append(slot_mapping)
        req_id2block_ids[request_id] = block_ids
        req_id2token_ids[request_id] = token_ids

    returned_ids = kvmanager.launch(
        task_ids=batch_task_ids,
        slot_mappings=batch_slot_mappings,
        as_batch=True,
        layerwise_transfer=True,
    )
    batch_id = returned_ids[0]
    batch_results = kvmanager.wait(batch_id, completely=True)
    kvresponse = batch_results[batch_id]
    assert kvresponse.status == KVResponseStatus.SUCCESS, (
        f"layerwise batch GET failed: {kvresponse.status}"
    )

    total_miss = 0
    for idx, orig_req_id in enumerate(batch_task_ids):
        mask = kvresponse.return_mask[idx]
        total_miss += len(mask) - mask.sum().item()
        valid_blocks = mask.sum().item() // tokens_per_block
        if valid_blocks == 0:
            continue
        blocks = req_id2block_ids[orig_req_id][:valid_blocks]
        tokens = req_id2token_ids[orig_req_id][:valid_blocks * tokens_per_block]
        assert c4_verifier.verify_gpu_blocks(blocks, tokens)
        assert c128_verifier.verify_gpu_blocks(blocks, tokens)
        assert indexer_verifier.verify_gpu_blocks(blocks, tokens_per_block, tokens)
        assert swa_verifier.verify_gpu_blocks(blocks, tokens)

    if cache_config.enable_cpu and cache_config.num_cpu_blocks >= num_gpu_blocks:
        assert total_miss == 0, f"expected 0 cache miss, got {total_miss}"

    shutdown_tp_client(tp_client_processes)
    kvmanager.shutdown()
    eventfd_thread.join(timeout=10)
    print(f"[Test] {test_label} PASSED")


@pytest.mark.parametrize(
    "model_config",
    [{"tp_size": 1, "dp_size": 1, "use_mla": True, "dtype": torch.uint8}],
    indirect=True,
)
@pytest.mark.parametrize(
    "cache_config",
    [
        {
            "tokens_per_block": 128,
            "enable_cpu": True,
            "enable_ssd": False,
            "num_cpu_blocks": 512,
        },
        {
            "tokens_per_block": 128,
            "enable_cpu": True,
            "enable_ssd": True,
            "num_cpu_blocks": 256,
            "num_ssd_blocks": 1024,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "test_config",
    [{"num_gpu_blocks": 128, "requests_per_block": 8, "initial_write_ratio": 0.5}],
    indirect=True,
)
def test_kvmanager_dsv4_layerwise_swa(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    test_config: dict,
) -> None:
    """KVManager round-trip for DSv4 three-group KV + SWA in layerwise GET mode."""
    orig_layerwise_env = os.environ.get("FLEXKV_ENABLE_LAYERWISE_TRANSFER")
    orig_layerwise_flag = GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer
    socket_path = os.environ.get(
        "FLEXKV_LAYERWISE_EVENTFD_SOCKET",
        "/tmp/flexkv_layerwise_eventfd.sock",
    )

    cache_config.swa = SWAPoolConfig(
        enabled=True,
        num_slots=max(cache_config.num_cpu_blocks, test_config["num_gpu_blocks"]),
        window_size=cache_config.tokens_per_block,
        num_swa_layers=NUM_ORIGINAL_LAYERS,
        bytes_per_token_per_layer=SWA_BYTES_PER_TOKEN,
        pin_memory=True,
    )

    try:
        os.environ["FLEXKV_ENABLE_LAYERWISE_TRANSFER"] = "1"
        GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = True

        ssd_label = "+ssd" if cache_config.enable_ssd else ""
        _run_dsv4_layerwise_swa_test(
            model_config,
            cache_config,
            test_config,
            test_label=f"dsv4-layerwise-swa{ssd_label}",
            eventfd_socket_path=socket_path,
        )
    finally:
        if orig_layerwise_env is None:
            os.environ.pop("FLEXKV_ENABLE_LAYERWISE_TRANSFER", None)
        else:
            os.environ["FLEXKV_ENABLE_LAYERWISE_TRANSFER"] = orig_layerwise_env
        GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = orig_layerwise_flag
