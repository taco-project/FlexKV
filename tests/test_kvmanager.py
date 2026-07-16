import contextlib
import time
import os
import shutil
import random

import pytest
import torch
import multiprocessing as mp
from multiprocessing import Process, Pipe

from flexkv.common.config import ModelConfig, CacheConfig, RankInfo, LayerGroupSpec
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.request import KVResponseStatus
from flexkv.kvtask import KVTaskEngine
from flexkv.kvmanager import KVManager
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.server.client import KVTPClient
import traceback

from flexkv.common.debug import flexkv_logger

# Import utilities from common_utils
from common_utils import (
    DEFAULT_MODEL_CONFIG, DEFAULT_CACHE_CONFIG, DEFAULT_TEST_CONFIG,
    generate_request_pair, block_ids_2_slot_mapping,
    skip_if_insufficient_gpus,create_gpu_kv_layout, GPUKVCacheVerifier
)


def _fp8_cuda_ops_unavailable():
    """True if fp8 dtype exists but CUDA ops (e.g. mul_cuda) are not implemented."""
    if not hasattr(torch, "float8_e4m3fn"):
        return True
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.tensor([1.0], dtype=torch.float8_e4m3fn, device="cuda")
        t.mul(1.0)
        return False
    except (NotImplementedError, RuntimeError):
        return True

def run_tp_client(dp_client_id,
                  tp_rank,
                  server_recv_port,
                  model_config,
                  cache_config,
                  num_gpu_blocks,
                  child_conn,
                  gpu_layout_type):
    """Run tp_client process"""
    try:
        device_id = tp_rank + dp_client_id * model_config.tp_size
        tp_client = KVTPClient(server_recv_port,
                               dp_client_id=dp_client_id, pp_rank=0,
                               device_id=device_id)

        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks, gpu_layout_type)

        # Create GPU blocks for this tp_rank in the tp_client process
        gpu_blocks_for_tp = []
        if gpu_layout_type == 0:
            for _ in range(model_config.num_layers):
                gpu_blocks_for_tp.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
                )
        elif gpu_layout_type == 1:
            gpu_blocks_for_tp.append(
                torch.empty(size=tuple(gpu_kv_layout.kv_shape[:]), dtype=model_config.dtype).cuda(device_id)
            )
        elif gpu_layout_type == 2:
            kv_dim = model_config.kv_dim
            for _ in range(model_config.num_layers * kv_dim):
                gpu_blocks_for_tp.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[2:]), dtype=model_config.dtype).cuda(device_id)
                )
        else:
            raise ValueError(f"Invalid GPU layout type: {gpu_layout_type}")
        tp_client.register_to_server(gpu_blocks_for_tp, gpu_kv_layout)

        # Send GPU blocks back to main process via pipe if connection provided
        if child_conn is not None:
            print(f"[TP Client {tp_rank}] Converting {len(gpu_blocks_for_tp)} GPU blocks to TensorSharedHandle")
            shared_gpu_blocks = [TensorSharedHandle(tensor) for tensor in gpu_blocks_for_tp]
            child_conn.send(shared_gpu_blocks)
            print(f"[TP Client {tp_rank}] Sent GPU blocks to main process via pipe")
            child_conn.close()

        # Keep the process running
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"[TP Client {tp_rank}] Exception occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        if child_conn is not None:
            child_conn.send(None)
            child_conn.close()

def shutdown_tp_client(tp_client_processes):
    for tp_process in tp_client_processes:
        if tp_process.is_alive():
            tp_process.terminate()
            tp_process.join(timeout=5)
            if tp_process.is_alive():
                print(f"Force killing tp_client process {tp_process.pid}")
                tp_process.kill()
                tp_process.join(timeout=2)

@pytest.mark.parametrize(
    "model_config",
    [
        {"tp_size": 1, "dp_size": 1},
        {"tp_size": 2, "dp_size": 2},
        {"dtype": torch.float32},
        {"use_mla": True},
        {"tp_size": 4, "dp_size": 1, "use_mla": True},
        {"tp_size": 4, "dp_size": 1},
        # fp8 端到端流程覆盖（仅在当前 PyTorch 支持 float8_e4m3fn 且 CUDA 具备 mul 等算子时启用）
        pytest.param(
            {"dtype": torch.float8_e4m3fn},
            marks=pytest.mark.skipif(
                _fp8_cuda_ops_unavailable(),
                reason="fp8 dtype or CUDA ops (e.g. mul_cuda) not available in this PyTorch build",
            ),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
    {'enable_cpu': True, 'enable_ssd': True, 'num_cpu_blocks': 256, 'num_ssd_blocks': 2048},
    # GDS test configs
    # {'enable_cpu': True, 'enable_gds': True, 'enable_ssd': True, \
    #     'enable_remote': False, 'num_cpu_blocks':256, 'num_ssd_blocks': 1024},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 512, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
    {'num_gpu_blocks': 512, 'requests_per_block': 16, 'initial_write_ratio': 0.4, 'namespace': ['test_namespace']},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [
    0,
    1,
    2,
])
def test_kvmanager(model_config, cache_config, test_config, gpu_layout_type, request):
    tp_size = model_config.tp_size
    dp_size = model_config.dp_size

    tokens_per_block = cache_config.tokens_per_block
    num_cpu_blocks = cache_config.num_cpu_blocks
    num_ssd_blocks = cache_config.num_ssd_blocks

    enable_cpu = cache_config.enable_cpu
    enable_ssd = cache_config.enable_ssd
    enable_remote = cache_config.enable_remote
    enable_gds = cache_config.enable_gds

    num_gpu_blocks = test_config["num_gpu_blocks"]
    block_per_request = test_config['requests_per_block']
    initial_write_ratio = test_config['initial_write_ratio']
    namespace = test_config.get('namespace', None)

    num_requests = num_gpu_blocks // block_per_request

    # Skip tests based on GPU availability and configuration
    skip_if_insufficient_gpus(tp_size * dp_size)

    if enable_gds and os.environ.get("FLEXKV_ENABLE_GDS", "0") == "0":
        pytest.skip("skip because GDS test is not enabled")

    if enable_remote:
        pytest.skip("skip because enable_remote is not supported")

    if dp_size > 1:
         #note that for now only dp_size=1 is supported
        pytest.skip("skip because server-client mode is not ready for dp_size > 1")

    kvmanager = KVManager(
        model_config=model_config,
        cache_config=cache_config,
        dp_client_id=0,
    )

    # Create pipes for each tp_client to send GPU blocks back
    mp_ctx = mp.get_context('spawn')
    pipe_connections = []
    tp_client_processes = []

    def cleanup():
        for conn in pipe_connections:
            with contextlib.suppress(OSError):
                conn.close()
        shutdown_tp_client(tp_client_processes)
        kvmanager.shutdown()

    # pytest finalizers run even when an assertion or timeout interrupts the
    # test, preventing a failed case from leaking CUDA workers/contexts into
    # the next parametrized case (the ISSUE#4 full-file cascade).
    request.addfinalizer(cleanup)
    kvmanager.start()

    for tp_rank in range(tp_size):
        parent_conn, child_conn = mp_ctx.Pipe()
        pipe_connections.append(parent_conn)

        tp_client_process = mp_ctx.Process(
            target=run_tp_client,
            args=(0, tp_rank, kvmanager.gpu_register_port, model_config, cache_config, \
                num_gpu_blocks + tp_rank, child_conn, gpu_layout_type),
            daemon=True
        )
        tp_client_processes.append(tp_client_process)
        tp_client_process.start()
        child_conn.close()

    # Collect GPU blocks from all tp_client processes
    print(f"[Main Process] Waiting to receive GPU blocks from {tp_size} TP client processes...")
    all_gpu_blocks = []

    for tp_rank, parent_conn in enumerate(pipe_connections):
        try:
            if not parent_conn.poll(INDEXER_STARTUP_TIMEOUT_S):
                process = tp_client_processes[tp_rank]
                raise TimeoutError(
                    f"TP client {tp_rank} registration timed out after "
                    f"{INDEXER_STARTUP_TIMEOUT_S}s "
                    f"(alive={process.is_alive()}, exitcode={process.exitcode})"
                )
            shared_gpu_blocks = parent_conn.recv()
            if shared_gpu_blocks is not None:
                all_gpu_blocks.append(shared_gpu_blocks)
                print(f"[Main Process] Received GPU blocks from TP client {tp_rank}")
            else:
                print(f"[Main Process] TP client {tp_rank} failed to create GPU blocks")
        except Exception as e:
            print(f"[Main Process] Error receiving from TP client {tp_rank}: {e}")
        finally:
            with contextlib.suppress(OSError):
                parent_conn.close()

    # Create GPUKVCacheVerifier with collected GPU blocks
    if all_gpu_blocks and len(all_gpu_blocks) == tp_size:
        print(f"[Main Process] Creating GPUKVCacheVerifier with GPU blocks from {len(all_gpu_blocks)} TP clients")

        # Get gpu_kv_layout from cache_config for GPUKVCacheVerifier
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks, gpu_layout_type)

        gpu_kv_verifier = GPUKVCacheVerifier(
            shared_gpu_blocks=all_gpu_blocks,
            gpu_kv_layout=gpu_kv_layout,
            tp_size=model_config.tp_size,
            tokens_per_block=cache_config.tokens_per_block,
            dtype=model_config.dtype,
            gpu_layout_type=gpu_layout_type
        )
        print("[Main Process] GPUKVCacheVerifier created successfully")
    else:
        print(f"[Main Process] Failed to collect GPU blocks from all TP clients. "
              f"Got {len(all_gpu_blocks)} out of {tp_size}")
        gpu_kv_verifier = None

    while not kvmanager.is_ready():
        time.sleep(1)
        flexkv_logger.info("waiting for flexkv to be ready")

    num_remote_blocks = cache_config.num_remote_blocks
    request_pairs = [generate_request_pair(i, block_per_request, num_gpu_blocks, tokens_per_block, dp_size)
                     for i in range(num_requests)]
    initial_write_num = int(num_requests * initial_write_ratio)
    print("writing initial data...")
    put_ids = []
    for token_ids, block_ids, dp_client_id in request_pairs[:initial_write_num]:
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
            namespace=namespace,
        )
        kvmanager.wait([write_request], completely=True)
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.clear_gpu_blocks(block_ids)

    #corner case: input token length for put is less than tokens_per_block
    write_request = kvmanager.put_async(
        token_ids=torch.randint(0, 100, size=(8,), dtype=torch.int64),
        slot_mapping=block_ids_2_slot_mapping(torch.arange(0,1, dtype=torch.int64), tokens_per_block, actual_length=8),
        token_mask=None,
        namespace=namespace,
    )
    kvmanager.wait([write_request], completely=True)
    #corner case: input token length is long enough, but the mask is less than tokens_per_block
    #my_mask = torch.zeros(16, dtype=torch.bool)
    #my_mask[0:8] = True
    #write_request = kvmanager.put_async(
    #    token_ids=torch.randint(0, 100, size=(16,), dtype=torch.int64),
    #    slot_mapping=block_ids_2_slot_mapping(torch.arange(0,1, dtype=torch.int64), tokens_per_block, actual_length=8),
    #    token_mask=my_mask,
    #)
    #kvmanager.wait_for_graph_finished(write_request)

    print(f"initial data {initial_write_num} written")
    total_cache_hit = 0
    total_cache_miss = 0
    running_get_requests = []
    running_put_requests = []
    req_id2block_ids = {}
    req_id2token_ids = {}
    flexkv_id2req_id = {}
    start_time = time.time()
    print(f"the initial {initial_write_num} write done,performing mixed read/write...")
    for i in range(initial_write_num, num_requests):
        print(f"performing mixed read/write {i} / {num_requests} ...")
        read_idx = i - initial_write_num
        token_ids, block_ids, dp_client_id = request_pairs[read_idx]
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
        request_id, _ = kvmanager.get_match(
            token_ids=token_ids,
            token_mask=None,
            namespace=namespace,
        )
        kvmanager.launch(request_id, slot_mapping)
        flexkv_id2req_id[request_id] = read_idx
        running_get_requests.append(request_id)
        req_id2block_ids[request_id] = block_ids
        req_id2token_ids[request_id] = token_ids
        token_ids, block_ids, dp_client_id = request_pairs[i]
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        request_id = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
            namespace=namespace,
        )
        req_id2block_ids[request_id] = block_ids
        flexkv_id2req_id[request_id] = i
        print(f"write flexkv request_id {request_id} to req_id {i}")
        running_put_requests.append(request_id)
        min_block_num = min(num_cpu_blocks, num_gpu_blocks)
        if (len(running_get_requests) + len(running_put_requests) >= min_block_num // block_per_request - 2 or
            i % initial_write_num == initial_write_num - 1 or
            i == num_requests - 1):
            if len(running_put_requests) > 0:
                kvmanager.wait(running_put_requests, completely=True)
                if gpu_kv_verifier is not None:
                    for req_id in running_put_requests:
                        gpu_kv_verifier.clear_gpu_blocks(req_id2block_ids[req_id])
            if len(running_get_requests) > 0:
                return_results = kvmanager.wait(running_get_requests, completely=True)
                if gpu_kv_verifier is not None:
                    for req_id, kvresponse in return_results.items():
                        assert kvresponse.status == KVResponseStatus.SUCCESS
                        valid_fetched_tokens = kvresponse.return_mask.sum().item() // \
                            tokens_per_block * tokens_per_block
                        token_ids = req_id2token_ids[req_id]
                        block_ids = req_id2block_ids[req_id]
                        assert gpu_kv_verifier.verify_kv_blocks(
                            token_ids[:valid_fetched_tokens],
                            block_ids[:valid_fetched_tokens//tokens_per_block])
                for kvresponse in return_results.values():
                    assert kvresponse.status == KVResponseStatus.SUCCESS
                    total_cache_hit += kvresponse.return_mask.sum().item()
                    total_cache_miss += len(kvresponse.return_mask) - kvresponse.return_mask.sum().item()
            running_get_requests = []
            running_put_requests = []
    if len(running_get_requests) > 0:
        return_results = kvmanager.wait(running_get_requests, completely=True)
        if gpu_kv_verifier is not None:
            for req_id, kvresponse in return_results.items():
                assert kvresponse.status == KVResponseStatus.SUCCESS
                valid_fetched_tokens = kvresponse.return_mask.sum().item() // tokens_per_block * tokens_per_block
                token_ids = req_id2token_ids[req_id]
                block_ids = req_id2block_ids[req_id]
                assert gpu_kv_verifier.verify_kv_blocks(
                    token_ids[:valid_fetched_tokens],
                    block_ids[:valid_fetched_tokens//tokens_per_block])
        running_get_requests = []
    if len(running_put_requests) > 0:
        kvmanager.wait(running_put_requests, completely=True)
        running_put_requests = []
    print("mixed read/write done")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time} s")
    print(f"Total cache hit rate: {total_cache_hit / (total_cache_hit + total_cache_miss)}")

    # =============== Test batched launched get ===============
    if not enable_gds:
        print("\n========== Testing batched launched get ==========")

        # Use the first few request_pairs that were written in initial phase
        batch_size = 6

        batched_get_task_ids = []
        batched_slot_mappings = []
        batched_req_info = []  # Store (token_ids, block_ids) for verification

        # Create multiple get_match requests
        for i in range(batch_size):
            token_ids, block_ids, dp_client_id = request_pairs[random.randint(0, num_requests - 1)]
            slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)

            request_id, return_mask = kvmanager.get_match(
                token_ids=token_ids,
                token_mask=None,
                namespace=namespace,
            )
            batched_get_task_ids.append(request_id)
            batched_slot_mappings.append(slot_mapping)
            batched_req_info.append((token_ids, block_ids, request_id))
            print(f"Created get_match request {request_id} for request_pair[{i}]")

        # Launch all get requests as a batch
        print(f"Launching {len(batched_get_task_ids)} get requests as batch...")
        batch_id = kvmanager.launch(
            task_ids=batched_get_task_ids,
            slot_mappings=batched_slot_mappings,
            as_batch=True
        )[0]
        print(f"Returned task_ids after batch launch: {batch_id}")

        # Wait for the batched get to complete
        # When as_batch=True, launch returns [batch_id], we need to wait on batch_id
        batch_results = kvmanager.wait(batch_id, completely=True)
        print(f"Batch wait returned {len(batch_results)} results")

        # Verify results
        batched_cache_hit = 0
        batched_cache_miss = 0
        kvresponse = batch_results[batch_id]
        assert kvresponse.status == KVResponseStatus.SUCCESS, \
            f"Batched get task {batch_id} failed with status {kvresponse.status}"
        for mask in kvresponse.return_mask:
            batched_cache_hit += return_mask.sum().item()
            batched_cache_miss += len(return_mask) - return_mask.sum().item()
            print(f"Task {batch_id}: cache_hit={batched_cache_hit}, cache_miss={batched_cache_miss}")

        # GPU KV cache verification for batched get
        if gpu_kv_verifier is not None:
            for idx, (token_ids, block_ids, req_id) in enumerate(batched_req_info):
                # Find the corresponding response
                # Note: when batched, the returned task_id might be the batch_id
                # We need to verify based on the actual data
                valid_fetched_tokens = kvresponse.return_mask[idx].sum().item() // tokens_per_block * tokens_per_block
                if valid_fetched_tokens > 0:
                    # Verify that GPU blocks contain correct data
                    verify_result = gpu_kv_verifier.verify_kv_blocks(
                        token_ids[:valid_fetched_tokens],
                        block_ids[:valid_fetched_tokens // tokens_per_block]
                    )

        print(f"Batched get test completed: hit={batched_cache_hit}, miss={batched_cache_miss}")

        # Since we read data that was written before, cache hit should be high
        if enable_cpu and num_cpu_blocks >= num_gpu_blocks:
            assert batched_cache_miss == 0, \
                f"Expected 0 cache miss for batched get, but got {batched_cache_miss}"
            print("  ✓ Batched launched get verification PASSED (100% cache hit)")
        else:
            print(f"  Batched launched get completed (cache hit rate: "
                    f"{batched_cache_hit / (batched_cache_hit + batched_cache_miss):.2%})")

    if enable_cpu and num_cpu_blocks >= num_gpu_blocks or \
        enable_ssd and num_ssd_blocks >= num_gpu_blocks or \
        enable_remote and num_remote_blocks >= num_gpu_blocks or \
        enable_gds and num_ssd_blocks >= num_gpu_blocks:
        assert total_cache_miss == 0
    # tp_client + kvmanager shutdown is handled by the request finalizer above,
    # so it runs even if the assertion or an earlier step fails.

    # Only verify data in direct mode
    # verify_data(gpu_blocks, dp_wise_gpu_blocks_gt, num_kv_heads, tp_size, dp_size, num_layers, use_mla)
    if total_cache_miss == 0:
        return
    elif total_cache_miss > 0:
        print(f"verify skipped, because of total_cache_miss={total_cache_miss} > 0")


class GPUIndexerCacheVerifier:
    def __init__(self,
                 shared_indexer_blocks,
                 indexer_kv_layout: KVCacheLayout,
                 tp_size: int,
                 dtype: torch.dtype) -> None:
        if not shared_indexer_blocks:
            raise ValueError("shared_indexer_blocks must not be empty")

        if isinstance(shared_indexer_blocks[0][0], torch.Tensor):
            self.gpu_blocks = shared_indexer_blocks
        else:
            imported_gpu_blocks = []
            for handles_in_one_gpu in shared_indexer_blocks:
                imported_gpu_blocks.append([handle.get_tensor() for handle in handles_in_one_gpu])
            self.gpu_blocks = imported_gpu_blocks

        self.num_layers = indexer_kv_layout.num_layer
        self.tokens_per_block = indexer_kv_layout.tokens_per_block
        self.head_size = indexer_kv_layout.head_size
        self.tp_size = tp_size
        self.dtype = dtype

    def hash_all_values(self, layer_id, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        token_hash = 0
        for i, token_id in enumerate(token_ids):
            token_hash += int(token_id) * (i + 17)
        return torch.tensor(((layer_id + 1) * 29 + token_hash) % 251 + 1, dtype=self.dtype).item()

    def fill_gpu_blocks(self, block_ids, main_kv_tokens_per_block, token_ids):
        """Fill indexer GPU blocks with deterministic hash values.

        The indexer group may use a compressed tokens_per_block dimension.
        Each block holds that group's configured entries on both GPU and CPU/SSD.
        For test purposes we broadcast one hash value per (layer, block) across
        all token positions in that block — the verifier checks ``[block_id, :, :]``
        which validates the whole block uniformly, so the round-trip integrity
        is exercised regardless of intra-block position layout.

        Args:
            block_ids: block IDs to fill (same as main KV block_ids).
            main_kv_tokens_per_block: tokens_per_block of main KV (e.g. 16).
            token_ids: full token_ids tensor from the request.
        """
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.int64)
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)

        for tp_id in range(self.tp_size):
            for layer_id in range(self.num_layers):
                gpu_tensor = self.gpu_blocks[tp_id][layer_id]
                for block_idx, block_id in enumerate(block_ids):
                    start_token_idx = block_idx * main_kv_tokens_per_block
                    end_token_idx = start_token_idx + main_kv_tokens_per_block
                    hash_value = self.hash_all_values(
                        layer_id,
                        token_ids[start_token_idx:end_token_idx],
                    )
                    # gpu_tensor shape: (num_blocks, indexer_tokens_per_block,
                    # head_size); broadcast the hash across the token dim.
                    gpu_tensor[block_id, :, :] = hash_value

    def clear_gpu_blocks(self, block_ids):
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)

        for tp_id in range(self.tp_size):
            for layer_id in range(self.num_layers):
                self.gpu_blocks[tp_id][layer_id][block_ids, :, :] = 0

    def verify_gpu_blocks(self, block_ids, main_kv_tokens_per_block, token_ids) -> bool:
        """Verify indexer GPU blocks after round-trip transfer.

        Args:
            block_ids: block IDs to verify.
            main_kv_tokens_per_block: tokens_per_block of main KV.
            token_ids: full token_ids tensor from the request.
        """
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.int64)
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)

        verification_passed = True
        errors = []

        for tp_id in range(self.tp_size):
            for layer_id in range(self.num_layers):
                gpu_tensor = self.gpu_blocks[tp_id][layer_id]
                for block_idx, block_id in enumerate(block_ids):
                    start_token_idx = block_idx * main_kv_tokens_per_block
                    end_token_idx = start_token_idx + main_kv_tokens_per_block
                    expected_hash_value = self.hash_all_values(
                        layer_id,
                        token_ids[start_token_idx:end_token_idx],
                    )
                    actual_values = gpu_tensor[block_id, :, :]
                    expected_tensor = torch.full_like(actual_values, expected_hash_value)
                    if not torch.equal(actual_values, expected_tensor):
                        verification_passed = False
                        max_abs_diff = (
                            actual_values.to(torch.int32) - expected_tensor.to(torch.int32)
                        ).abs().max().item()
                        errors.append(
                            f"Mismatch at tp={tp_id}, layer={layer_id}, block={block_id}: "
                            f"expected={expected_hash_value}, max_abs_diff={max_abs_diff}"
                        )

        if not verification_passed:
            print(f"Indexer verification failed with {len(errors)} errors:")
            for error in errors[:10]:
                print(f"  {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print("Indexer GPU blocks verification passed!")
        assert verification_passed
        return verification_passed


def run_tp_client_with_indexer(dp_client_id,
                               tp_rank,
                               server_recv_port,
                               model_config,
                               cache_config,
                               num_gpu_blocks,
                               child_conn,
                               gpu_layout_type):
    """Run tp_client process with indexer expressed as an extra LayerGroupSpec.

    Reads indexer shape from ``model_config.layer_groups[-1]`` (the indexer
    group, populated by _run_indexer_test before spawn). Registers main +
    indexer buffers via the unified ``register_to_server(layer_groups=...,
    gpu_layouts=..., handles_per_group=...)`` API.
    """
    try:
        device_id = tp_rank + dp_client_id * model_config.tp_size

        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks, gpu_layout_type)

        # Create main GPU blocks
        gpu_blocks_for_tp = []
        if gpu_layout_type == 0:
            for _ in range(model_config.num_layers):
                gpu_blocks_for_tp.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
                )
        elif gpu_layout_type == 2:
            kv_dim = model_config.kv_dim
            for _ in range(model_config.num_layers * kv_dim):
                gpu_blocks_for_tp.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[2:]), dtype=model_config.dtype).cuda(device_id)
                )
        else:
            raise ValueError(f"Invalid GPU layout type for indexer test: {gpu_layout_type}")

        # Indexer parameters come from model_config.layer_groups[-1] — the
        # indexer group appended by _run_indexer_test before spawn.
        assert model_config.layer_groups and len(model_config.layer_groups) >= 2, (
            "model_config.layer_groups must contain a main + indexer group "
            "(set by _run_indexer_test)"
        )
        indexer_group = model_config.layer_groups[-1]
        # Indexer per-block token count after compression. With
        # compress_ratio=1 (default) this matches the main KV's tpb (DSv4
        # per-token form); with compress_ratio>1 the GPU tensor shrinks to
        # tpb_g = tpb // compress_ratio, matching the shape sglang allocates
        # for the compressed group (see storage._compute_kv_shape and
        # worker._init_multi_group).
        assert cache_config.tokens_per_block % indexer_group.compress_ratio == 0, (
            f"indexer compress_ratio={indexer_group.compress_ratio} must divide "
            f"tokens_per_block={cache_config.tokens_per_block}"
        )
        indexer_tokens_per_block = cache_config.tokens_per_block // indexer_group.compress_ratio
        indexer_num_layers = indexer_group.num_layers

        # Create indexer GPU blocks (MLA-style: 3D tensors)
        indexer_blocks = []
        for _ in range(indexer_num_layers):
            indexer_blocks.append(
                torch.empty(
                    num_gpu_blocks,
                    indexer_tokens_per_block,
                    indexer_group.head_size,
                    dtype=indexer_group.dtype,
                ).cuda(device_id)
            )

        from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=indexer_num_layers,
            num_block=num_gpu_blocks,
            tokens_per_block=indexer_tokens_per_block,
            num_head=indexer_group.num_kv_heads,
            head_size=indexer_group.head_size,
            is_mla=True,
        )

        # Unified registration: main + indexer go through one register_to_server
        # call, matching the production vllm adapter path.
        tp_client = KVTPClient(
            gpu_register_port=server_recv_port + "_gpu_register",
            dp_client_id=dp_client_id, pp_rank=0,
            device_id=device_id,
        )
        tp_client.register_to_server(
            kv_caches=gpu_blocks_for_tp,
            kv_layout=gpu_kv_layout,
            layer_groups=model_config.layer_groups,
            gpu_layouts=[gpu_kv_layout, indexer_layout],
            handles_per_group=[gpu_blocks_for_tp, indexer_blocks],
        )

        # Send GPU blocks back to main process via pipe
        if child_conn is not None:
            shared_gpu_blocks = [TensorSharedHandle(tensor) for tensor in gpu_blocks_for_tp]
            shared_indexer_blocks = [TensorSharedHandle(tensor) for tensor in indexer_blocks]
            child_conn.send({
                "main": shared_gpu_blocks,
                "indexer": shared_indexer_blocks,
            })
            child_conn.close()

        # Keep the process running
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"[TP Client {tp_rank}] Exception occurred: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        if child_conn is not None:
            child_conn.send(None)
            child_conn.close()


INDEXER_STARTUP_TIMEOUT_S = 60.0


def _run_indexer_test(model_config, cache_config, test_config, gpu_layout_type,
                      request,
                      test_label="indexer", layerwise=False,
                      indexer_compress_ratio: int = 1):
    """Core test logic for KVManager with indexer shadow transfer.

    Shared by test_kvmanager_with_indexer (non-layerwise) and
    test_kvmanager_with_indexer_layerwise (layerwise mode).

    ``indexer_compress_ratio`` controls how many tokens of the main KV each
    indexer slot covers (DSv4 NSA: tpb_g = tpb // compress_ratio).  Must
    divide cache_config.tokens_per_block.
    """
    tp_size = model_config.tp_size
    tokens_per_block = cache_config.tokens_per_block
    num_gpu_blocks = test_config["num_gpu_blocks"]
    block_per_request = test_config['requests_per_block']
    initial_write_ratio = test_config['initial_write_ratio']
    num_requests = num_gpu_blocks // block_per_request

    skip_if_insufficient_gpus(tp_size)

    assert tokens_per_block % indexer_compress_ratio == 0, (
        f"indexer_compress_ratio={indexer_compress_ratio} must divide "
        f"tokens_per_block={tokens_per_block}"
    )

    # indexer as an extra LayerGroupSpec appended to
    # model_config.layer_groups.  The main group comes first (carrying the
    # full set of "real" layers); the indexer group comes last with its
    # own (num_kv_heads=1, head_size=64, dtype=uint8) shape.
    main_layer_indices = list(range(model_config.num_layers))
    main_group = LayerGroupSpec(
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_size=model_config.head_size,
        layer_indices=main_layer_indices,
        dtype=model_config.dtype,
    )
    indexer_group = LayerGroupSpec(
        num_layers=model_config.num_layers,
        num_kv_heads=1,
        head_size=64,
        layer_indices=main_layer_indices,
        dtype=torch.uint8,
        compress_ratio=indexer_compress_ratio,
    )
    model_config.layer_groups = [main_group, indexer_group]

    pipe_connections = []
    tp_client_processes = []
    kvmanager = KVManager(
        model_config=model_config,
        cache_config=cache_config,
        dp_client_id=0,
    )

    def cleanup():
        for conn in pipe_connections:
            with contextlib.suppress(OSError):
                conn.close()
        shutdown_tp_client(tp_client_processes)
        kvmanager.shutdown()

    # pytest finalizers run even when an assertion or timeout interrupts the
    # test, preventing failed cases from leaking CUDA workers into the next
    # parametrized case.
    request.addfinalizer(cleanup)
    kvmanager.start()

    mp_ctx = mp.get_context('spawn')

    for tp_rank in range(tp_size):
        parent_conn, child_conn = mp_ctx.Pipe()
        pipe_connections.append(parent_conn)

        tp_client_process = mp_ctx.Process(
            target=run_tp_client_with_indexer,
            args=(0, tp_rank, kvmanager.server_recv_port,
                  model_config, cache_config, num_gpu_blocks, child_conn,
                  gpu_layout_type),
            daemon=True
        )
        tp_client_processes.append(tp_client_process)
        tp_client_process.start()
        child_conn.close()

    all_gpu_blocks = []
    all_indexer_blocks = []
    for tp_rank, parent_conn in enumerate(pipe_connections):
        try:
            if not parent_conn.poll(INDEXER_STARTUP_TIMEOUT_S):
                process = tp_client_processes[tp_rank]
                raise TimeoutError(
                    f"TP client {tp_rank} registration timed out after "
                    f"{INDEXER_STARTUP_TIMEOUT_S}s "
                    f"(alive={process.is_alive()}, exitcode={process.exitcode})"
                )
            shared_payload = parent_conn.recv()
            if shared_payload is not None:
                if isinstance(shared_payload, dict):
                    shared_gpu_blocks = shared_payload.get("main")
                    shared_indexer_blocks = shared_payload.get("indexer")
                else:
                    shared_gpu_blocks = shared_payload
                    shared_indexer_blocks = None
                if shared_gpu_blocks is not None:
                    all_gpu_blocks.append(shared_gpu_blocks)
                    print(f"[Main Process] Received GPU blocks from TP client {tp_rank}")
                if shared_indexer_blocks is not None:
                    all_indexer_blocks.append(shared_indexer_blocks)
        except Exception as e:
            print(f"[Main Process] Error receiving from TP client {tp_rank}: {e}")
            raise
        finally:
            parent_conn.close()

    gpu_kv_verifier = None
    if all_gpu_blocks and len(all_gpu_blocks) == tp_size:
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks, gpu_layout_type)
        gpu_kv_verifier = GPUKVCacheVerifier(
            shared_gpu_blocks=all_gpu_blocks,
            gpu_kv_layout=gpu_kv_layout,
            tp_size=model_config.tp_size,
            tokens_per_block=cache_config.tokens_per_block,
            dtype=model_config.dtype,
            gpu_layout_type=gpu_layout_type,
        )

    indexer_kv_verifier = None
    indexer_cfg = (
        model_config.layer_groups[-1]
        if model_config.layer_groups and len(model_config.layer_groups) >= 2
        else None
    )
    if all_indexer_blocks and len(all_indexer_blocks) == tp_size and indexer_cfg is not None:
        # Indexer GPU layout uses tpb_g = tpb // compress_ratio (matches the
        # GPU tensor allocated in run_tp_client_with_indexer).
        indexer_gpu_tpb = cache_config.tokens_per_block // indexer_cfg.compress_ratio
        indexer_gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=indexer_cfg.num_layers,
            num_block=num_gpu_blocks,
            tokens_per_block=indexer_gpu_tpb,
            num_head=indexer_cfg.num_kv_heads,
            head_size=indexer_cfg.head_size,
            is_mla=True,
        )
        indexer_kv_verifier = GPUIndexerCacheVerifier(
            shared_indexer_blocks=all_indexer_blocks,
            indexer_kv_layout=indexer_gpu_layout,
            tp_size=model_config.tp_size,
            dtype=indexer_cfg.dtype,
        )

    ready_deadline = time.monotonic() + INDEXER_STARTUP_TIMEOUT_S
    while not kvmanager.is_ready():
        dead_clients = [
            (idx, process.exitcode)
            for idx, process in enumerate(tp_client_processes)
            if not process.is_alive()
        ]
        if dead_clients:
            raise RuntimeError(
                f"TP clients exited before KVManager became ready: "
                f"{dead_clients}")
        if time.monotonic() >= ready_deadline:
            raise TimeoutError(
                f"KVManager ({test_label}) did not become ready within "
                f"{INDEXER_STARTUP_TIMEOUT_S}s")
        time.sleep(0.2)
        flexkv_logger.info(f"waiting for flexkv ({test_label}) to be ready")
    print(f"[Test] KVManager ({test_label}) is ready")

    request_pairs = [generate_request_pair(i, block_per_request, num_gpu_blocks, tokens_per_block, 1)
                     for i in range(num_requests)]
    initial_write_num = int(num_requests * initial_write_ratio)

    print(f"[Test] Testing put flow ({test_label})...")
    for token_ids, block_ids, dp_client_id in request_pairs[:initial_write_num]:
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        if indexer_kv_verifier is not None:
            indexer_kv_verifier.fill_gpu_blocks(block_ids, tokens_per_block, token_ids)
        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
        )
        put_results = kvmanager.wait([write_request], completely=True)
        assert put_results[write_request].status == KVResponseStatus.SUCCESS
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.clear_gpu_blocks(block_ids)
        if indexer_kv_verifier is not None:
            indexer_kv_verifier.clear_gpu_blocks(block_ids)
    print(f"[Test] Initial {initial_write_num} put operations completed ({test_label})")

    print(f"[Test] Testing get flow ({test_label})...")
    total_cache_hit = 0
    total_cache_miss = 0
    running_get_requests = []
    req_id2block_ids = {}
    req_id2token_ids = {}

    batch_task_ids = []
    batch_slot_mappings = []

    for i in range(min(initial_write_num, num_requests)):
        token_ids, block_ids, dp_client_id = request_pairs[i]
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
        request_id, _ = kvmanager.get_match(
            token_ids=token_ids,
            token_mask=None,
        )
        batch_task_ids.append(request_id)
        batch_slot_mappings.append(slot_mapping)
        req_id2block_ids[request_id] = block_ids
        req_id2token_ids[request_id] = token_ids

    if layerwise:
        # Layerwise mode: launch all GETs as a single batch so that
        # merge_to_batch_graph produces a LAYERWISE op (fused DISK2H+H2D).
        returned_ids = kvmanager.launch(
            task_ids=batch_task_ids,
            slot_mappings=batch_slot_mappings,
            as_batch=True,
            layerwise_transfer=True,
        )
        batch_id = returned_ids[0]
        batch_results = kvmanager.wait(batch_id, completely=True)
        kvresponse = batch_results[batch_id]
        assert kvresponse.status == KVResponseStatus.SUCCESS, \
            f"Layerwise batch GET failed: {kvresponse.status}"
        for idx, orig_req_id in enumerate(batch_task_ids):
            mask = kvresponse.return_mask[idx]
            total_cache_hit += mask.sum().item()
            total_cache_miss += len(mask) - mask.sum().item()
            if gpu_kv_verifier is not None:
                valid_fetched_tokens = mask.sum().item() // tokens_per_block * tokens_per_block
                if valid_fetched_tokens > 0:
                    assert gpu_kv_verifier.verify_kv_blocks(
                        req_id2token_ids[orig_req_id][:valid_fetched_tokens],
                        req_id2block_ids[orig_req_id][:valid_fetched_tokens // tokens_per_block])
            if indexer_kv_verifier is not None:
                valid_fetched_blocks = mask.sum().item() // tokens_per_block
                if valid_fetched_blocks > 0:
                    assert indexer_kv_verifier.verify_gpu_blocks(
                        req_id2block_ids[orig_req_id][:valid_fetched_blocks],
                        tokens_per_block,
                        req_id2token_ids[orig_req_id][:valid_fetched_blocks * tokens_per_block])
    else:
        # Non-layerwise: launch each GET individually
        for req_id in batch_task_ids:
            kvmanager.launch(req_id, batch_slot_mappings[batch_task_ids.index(req_id)])
            running_get_requests.append(req_id)

        if running_get_requests:
            return_results = kvmanager.wait(running_get_requests, completely=True)
            for req_id, kvresponse in return_results.items():
                assert kvresponse.status == KVResponseStatus.SUCCESS
                total_cache_hit += kvresponse.return_mask.sum().item()
                total_cache_miss += len(kvresponse.return_mask) - kvresponse.return_mask.sum().item()
                if gpu_kv_verifier is not None:
                    valid_fetched_tokens = kvresponse.return_mask.sum().item() // tokens_per_block * tokens_per_block
                    if valid_fetched_tokens > 0:
                        assert gpu_kv_verifier.verify_kv_blocks(
                            req_id2token_ids[req_id][:valid_fetched_tokens],
                            req_id2block_ids[req_id][:valid_fetched_tokens // tokens_per_block])
                if indexer_kv_verifier is not None:
                    valid_fetched_blocks = kvresponse.return_mask.sum().item() // tokens_per_block
                    if valid_fetched_blocks > 0:
                        assert indexer_kv_verifier.verify_gpu_blocks(
                            req_id2block_ids[req_id][:valid_fetched_blocks],
                            tokens_per_block,
                            req_id2token_ids[req_id][:valid_fetched_blocks * tokens_per_block])
    print(f"[Test] Get flow completed ({test_label}): hit={total_cache_hit}, miss={total_cache_miss}")

    print(f"[Test] Testing try_wait flow ({test_label})...")
    if initial_write_num < num_requests:
        token_ids, block_ids, dp_client_id = request_pairs[initial_write_num]
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        if indexer_kv_verifier is not None:
            indexer_kv_verifier.fill_gpu_blocks(block_ids, tokens_per_block, token_ids)
        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
        )
        finished = {}
        for _ in range(200):
            finished = kvmanager.try_wait([write_request])
            if write_request in finished:
                break
            time.sleep(0.1)
        assert write_request in finished, "try_wait should eventually return the completed task"
        assert finished[write_request].status == KVResponseStatus.SUCCESS
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.clear_gpu_blocks(block_ids)
        if indexer_kv_verifier is not None:
            indexer_kv_verifier.clear_gpu_blocks(block_ids)
    print(f"[Test] try_wait flow completed ({test_label})")

    # Cache miss assertion: when total capacity >= GPU blocks, expect 0 miss
    enable_cpu = cache_config.enable_cpu
    enable_ssd = cache_config.enable_ssd
    num_cpu_blocks = cache_config.num_cpu_blocks
    num_ssd_blocks = cache_config.num_ssd_blocks
    if (enable_cpu and num_cpu_blocks >= num_gpu_blocks) or \
       (enable_ssd and num_ssd_blocks >= num_gpu_blocks):
        assert total_cache_miss == 0, f"Expected 0 cache miss, got {total_cache_miss}"

    print(f"[Test] {test_label} PASSED")


@pytest.mark.parametrize(
    "model_config",
    [
        # DSv4 form: main attention is MLA (kv_dim=1), indexer is also MLA-style
        # (single tensor per layer with num_kv_heads=1, head_size=64, uint8).
        {"tp_size": 1, "dp_size": 1, "use_mla": True},
    ],    indirect=True,
)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
    {'enable_cpu': True, 'enable_ssd': True, 'num_cpu_blocks': 256, 'num_ssd_blocks': 2048},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 256, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [0])
@pytest.mark.parametrize("indexer_compress_ratio", [1, 4])
def test_kvmanager_with_indexer(model_config, cache_config, test_config,
                                gpu_layout_type, indexer_compress_ratio,
                                request):
    """Test KVManager with indexer: GPU↔CPU (and optionally ↔SSD) data correctness."""
    ssd_label = "+ssd" if cache_config.enable_ssd else ""
    cr_label = f"+cr{indexer_compress_ratio}" if indexer_compress_ratio != 1 else ""
    _run_indexer_test(model_config, cache_config, test_config, gpu_layout_type,
                      request,
                      test_label=f"indexer{ssd_label}{cr_label}",
                      indexer_compress_ratio=indexer_compress_ratio)


import ctypes
import socket
import struct
import threading

# ---- Mock SGLang eventfd client for layerwise unit tests ----

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _sys_eventfd(initval: int = 0, flags: int = 0) -> int:
    """Create an eventfd file descriptor via libc."""
    fd = _libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd failed: {os.strerror(err)}")
    return fd


_EFD_SEMAPHORE = 0x1


def _send_fds_via_scm(sock: socket.socket, fds: list, extra_data: bytes = b"x"):
    """Send fds via SCM_RIGHTS (mirrors SGLang's send_fds)."""
    fds_packed = struct.pack(f"{len(fds)}i", *fds)
    ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_packed)]
    sock.sendmsg([extra_data], ancdata)


def _mock_sglang_eventfd_client(socket_path: str,
                                tp_rank: int,
                                tp_size: int,
                                num_layers: int,
                                handshake_done: threading.Event,
                                stop_requested: threading.Event,
                                errors: list,
                                num_counters: int = 3,
                                max_retries: int = 120,
                                retry_interval: float = 0.5):
    """Simulate SGLang sending eventfds to the LayerwiseTransferWorker.

    Runs in a background thread. Creates real eventfds so the C++
    LayerwiseTransferGroup receives valid file descriptors. The worker gets
    its own copies through SCM_RIGHTS; the sender copies are closed after the
    handshake.
    """
    created_fds = []
    sock = None
    try:
        # Create real eventfds
        for _ in range(num_counters * num_layers):
            created_fds.append(_sys_eventfd(0, _EFD_SEMAPHORE))

        # Retry connecting until the worker process binds the socket
        for attempt in range(max_retries):
            if stop_requested.is_set():
                return
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(socket_path)
                print(f"[MockEventfdClient] Connected to {socket_path} "
                      f"(attempt {attempt + 1})")
                break
            except (FileNotFoundError, ConnectionRefusedError):
                sock.close()
                sock = None
                if stop_requested.wait(retry_interval):
                    return

        if sock is None:
            raise TimeoutError(
                f"Failed to connect to {socket_path} after "
                f"{max_retries} attempts")

        metadata = struct.pack("iiii",
                               tp_rank, tp_size,
                               num_layers, num_counters)
        sock.sendall(metadata)

        # Send eventfds for each counter via SCM_RIGHTS
        fd_idx = 0
        for counter_id in range(num_counters):
            fds = created_fds[fd_idx:fd_idx + num_layers]
            fd_idx += num_layers
            _send_fds_via_scm(sock, fds, struct.pack("i", counter_id))

        # Wait for ACK
        sock.settimeout(30.0)
        ack = sock.recv(1)
        if ack and ack[0] == 1:
            print(f"[MockEventfdClient] Eventfd handshake OK "
                  f"(counters={num_counters}, layers={num_layers})")
            handshake_done.set()
        else:
            raise RuntimeError(f"Unexpected eventfd ACK: {ack!r}")
    except Exception as e:
        errors.append(e)
        print(f"[MockEventfdClient] Error: {e}")
        traceback.print_exc()
    finally:
        if sock is not None:
            sock.close()
        # SCM_RIGHTS duplicates descriptors into the receiving worker, so the
        # sender must close its copies or every parametrized case leaks fds.
        for fd in created_fds:
            with contextlib.suppress(OSError):
                os.close(fd)


@pytest.mark.parametrize(
    "model_config",
    [
        # DSv4 form (see test_kvmanager_with_indexer).
        {"tp_size": 1, "dp_size": 1, "use_mla": True},
    ],    indirect=True,
)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
    {'enable_cpu': True, 'enable_ssd': True, 'num_cpu_blocks': 256, 'num_ssd_blocks': 2048},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 256, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [0])
@pytest.mark.parametrize("indexer_compress_ratio", [1, 4])
def test_kvmanager_with_indexer_layerwise(model_config, cache_config, test_config,
                                          gpu_layout_type, indexer_compress_ratio,
                                          request):
    """Test KVManager with indexer in LAYERWISE mode.

    Validates the full round-trip:
      PUT: D2H + H2DISK (non-layerwise, same as normal)
      GET: LAYERWISE (fused DISK2H + H2D)
    Data correctness is verified for both the main KV cache and the
    indexer (DSA) KV cache after the round-trip.

    A background thread simulates the SGLang eventfd client so the
    LayerwiseTransferWorker can complete its initialization handshake
    without any source-code changes.
    """
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

    # Save original values
    orig_layerwise_env = os.environ.get('FLEXKV_ENABLE_LAYERWISE_TRANSFER')
    orig_socket_env = os.environ.get('FLEXKV_LAYERWISE_EVENTFD_SOCKET')
    orig_layerwise_flag = GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer

    # A unique path prevents stale workers or parametrized cases from binding
    # or connecting to one another's eventfd socket.
    socket_path = (
        f"/tmp/flexkv_layerwise_eventfd_{os.getpid()}_{time.time_ns()}.sock")
    handshake_done = threading.Event()
    stop_requested = threading.Event()
    eventfd_errors = []
    eventfd_thread = None

    try:
        # Enable layerwise transfer
        os.environ['FLEXKV_ENABLE_LAYERWISE_TRANSFER'] = '1'
        os.environ['FLEXKV_LAYERWISE_EVENTFD_SOCKET'] = socket_path
        GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = True

        # Start mock SGLang eventfd client thread BEFORE kvmanager.start()
        # so it is ready to connect once the worker process binds the socket.
        eventfd_thread = threading.Thread(
            target=_mock_sglang_eventfd_client,
            args=(socket_path, 0, 1, model_config.num_layers,
                  handshake_done, stop_requested, eventfd_errors),
            daemon=True,
        )
        eventfd_thread.start()

        ssd_label = "+ssd" if cache_config.enable_ssd else ""
        cr_label = f"+cr{indexer_compress_ratio}" if indexer_compress_ratio != 1 else ""
        _run_indexer_test(model_config, cache_config, test_config, gpu_layout_type,
                          request,
                          test_label=f"layerwise+indexer{ssd_label}{cr_label}",
                          layerwise=True,
                          indexer_compress_ratio=indexer_compress_ratio)

        eventfd_thread.join(timeout=10)
        assert not eventfd_thread.is_alive(), \
            "mock eventfd client did not exit after the layerwise test"
        assert not eventfd_errors, \
            f"mock eventfd handshake failed: {eventfd_errors!r}"
        assert handshake_done.is_set(), "mock eventfd handshake did not complete"
    finally:
        stop_requested.set()
        if eventfd_thread is not None and eventfd_thread.is_alive():
            eventfd_thread.join(timeout=2)
        # Restore original environment and config
        if orig_layerwise_env is None:
            os.environ.pop('FLEXKV_ENABLE_LAYERWISE_TRANSFER', None)
        else:
            os.environ['FLEXKV_ENABLE_LAYERWISE_TRANSFER'] = orig_layerwise_env
        if orig_socket_env is None:
            os.environ.pop('FLEXKV_LAYERWISE_EVENTFD_SOCKET', None)
        else:
            os.environ['FLEXKV_LAYERWISE_EVENTFD_SOCKET'] = orig_socket_env
        GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = orig_layerwise_flag
        with contextlib.suppress(FileNotFoundError):
            os.unlink(socket_path)
