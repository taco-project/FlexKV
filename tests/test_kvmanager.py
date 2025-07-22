import time
import os
import shutil

import pytest
import torch

from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.kvmanager import KVManager

# Import utilities from test_utils
from test_utils import (
    DEFAULT_MODEL_CONFIG, DEFAULT_CACHE_CONFIG, DEFAULT_TEST_CONFIG,
    generate_request_pair, verify_data, block_ids_2_slot_mapping,
    generate_gpu_blocks_with_ground_truth, skip_if_insufficient_gpus,
    create_kvmanager_with_mode, create_gpu_kv_layout
)


@pytest.mark.parametrize("model_config", [
    {'tp_size': 1, 'dp_size': 1},
    {'tp_size': 2, 'dp_size': 2},
    {'dtype': torch.float32},
    {'use_mla': True},
    {'tp_size': 4, 'dp_size': 1, 'use_mla': True},
], indirect=True)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'enable_remote': False, 'num_cpu_blocks': 1024},
    {'enable_cpu': True, 'enable_ssd': True, 'enable_remote': False,},
    {'enable_cpu': True, 'enable_ssd': True, 'enable_remote': False, 'ssd_cache_iouring_entries': 512},
    {'enable_cpu': True, 'enable_ssd': True, 'enable_remote': True, 'num_ssd_blocks': 256, 'num_remote_blocks': 512},
    {'enable_cpu': True, 'enable_ssd': True, 'enable_remote': True,
     'num_ssd_blocks': 256, 'num_remote_blocks': 512, 'ssd_cache_iouring_entries': 512},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 512, 'requests_per_block': 16, 'initial_write_ratio': 0.4, 'use_server_client': False},
    {'num_gpu_blocks': 512, 'requests_per_block': 16, 'initial_write_ratio': 0.4, 'use_server_client': True},
], indirect=True)
@pytest.mark.parametrize("flex_kv_layout_type", [
    KVCacheLayoutType.LAYERWISE,
    KVCacheLayoutType.BLOCKWISE,
])
def test_kvmanager(model_config, cache_config, test_config, flex_kv_layout_type):
    num_layers = model_config.num_layers
    num_kv_heads = model_config.num_kv_heads
    head_size = model_config.head_size
    tp_size = model_config.tp_size
    dp_size = model_config.dp_size
    use_mla = model_config.use_mla

    tokens_per_block = cache_config.tokens_per_block
    num_cpu_blocks = cache_config.num_cpu_blocks
    num_ssd_blocks = cache_config.num_ssd_blocks

    enable_cpu = cache_config.enable_cpu
    enable_ssd = cache_config.enable_ssd
    enable_remote = cache_config.enable_remote

    cache_config.cpu_kv_layout_type = flex_kv_layout_type
    cache_config.ssd_kv_layout_type = flex_kv_layout_type
    cache_config.remote_kv_layout_type = flex_kv_layout_type

    num_gpu_blocks = test_config["num_gpu_blocks"]
    block_per_request = test_config['requests_per_block']
    initial_write_ratio = test_config['initial_write_ratio']
    use_server_client = test_config.get('use_server_client', False)

    num_requests = num_gpu_blocks // block_per_request

    # Skip tests based on GPU availability and configuration
    skip_if_insufficient_gpus(tp_size * dp_size)

    if enable_remote:
        pytest.skip("skip because enable_remote is not supported")

    if use_server_client and dp_size > 1:
        pytest.skip("skip because server-client mode is not supported for dp_size > 1 IN THIS TEST SCRIPT now")

    if use_server_client:
        # In server-client mode, GPU blocks are created in tp_client processes
        # We only need the layout for initialization
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks)
        gpu_blocks = None  # Not used in server-client mode
        dp_wise_gpu_blocks_gt = None  # Not used in server-client mode
    else:
        # In direct mode, create GPU blocks in current process
        gpu_blocks, dp_wise_gpu_blocks_gt, gpu_kv_layout = \
            generate_gpu_blocks_with_ground_truth(model_config, cache_config, test_config)

    kvmanager = create_kvmanager_with_mode(model_config, cache_config, gpu_kv_layout, gpu_blocks, use_server_client)

    # put this after KVManager()
    num_remote_blocks = cache_config.num_remote_blocks
    assert kvmanager.is_ready()
    kvmanager.start()
    request_pairs = [generate_request_pair(i, block_per_request, num_gpu_blocks, tokens_per_block, dp_size)
                     for i in range(num_requests)]
    initial_write_num = int(num_requests * initial_write_ratio)
    print("writing initial data...")
    for token_ids, block_ids, dp_id in request_pairs[:initial_write_num]:
        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            dp_id=dp_id,
        )
        kvmanager.wait_for_graph_finished(write_request)
        if not use_server_client:
            # In direct mode, update GPU blocks for verification
            for gpu in range(dp_id * tp_size, (dp_id + 1) * tp_size):
                for i in range(num_layers):
                    gpu_blocks[gpu][i][:, block_ids, :, :, :] = 0

    #corner case: input token length for put is less than tokens_per_block
    write_request = kvmanager.put_async(
        token_ids=torch.randint(0, 100, size=(8,), dtype=torch.int64),
        slot_mapping=block_ids_2_slot_mapping(torch.arange(0,1, dtype=torch.int64), tokens_per_block, actual_length=8),
        dp_id=0,
    )
    kvmanager.wait_for_graph_finished(write_request)
    #corner case: input token length is long enough, but the mask is less than tokens_per_block
    #my_mask = torch.zeros(16, dtype=torch.bool)
    #my_mask[0:8] = True
    #write_request = kvmanager.put_async(
    #    token_ids=torch.randint(0, 100, size=(16,), dtype=torch.int64),
    #    slot_mapping=block_ids_2_slot_mapping(torch.arange(0,1, dtype=torch.int64), tokens_per_block, actual_length=8),
    #    token_mask=my_mask,
    #    dp_id=0,
    #)
    #kvmanager.wait_for_graph_finished(write_request)

    print(f"initial data {initial_write_num} written")
    total_cache_hit = 0
    total_cache_miss = 0
    running_get_requests = []
    running_put_requests = []
    start_time = time.time()
    print(f"the initial {initial_write_num} write done,performing mixed read/write...")
    for i in range(initial_write_num, num_requests):
        print(f"performing mixed read/write {i} / {num_requests} ...")
        read_idx = i - initial_write_num
        token_ids, block_ids, dp_id = request_pairs[read_idx]
        request_id = kvmanager.get_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            layer_granularity=-1,
            dp_id=dp_id,
        )
        running_get_requests.append(request_id)
        token_ids, block_ids, dp_id = request_pairs[i]
        request_id = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            dp_id=dp_id,
        )
        running_put_requests.append(request_id)
        min_block_num = min(num_cpu_blocks, num_gpu_blocks)
        if (len(running_get_requests) + len(running_put_requests) >= min_block_num // block_per_request - 2 or
            i % initial_write_num == initial_write_num - 1 or
            i == num_requests - 1):
            if len(running_put_requests) > 0:
                kvmanager.wait_for_graph_finished(running_put_requests)
            if len(running_get_requests) > 0:
                return_masks = kvmanager.wait(running_get_requests)
                for return_mask in return_masks.values():
                    total_cache_hit += return_mask.sum()
                    total_cache_miss += len(return_mask) - return_mask.sum()
            running_get_requests = []
            running_put_requests = []
    if len(running_get_requests) > 0:
        kvmanager.wait(running_get_requests)
        running_get_requests = []
    if len(running_put_requests) > 0:
        kvmanager.wait_for_graph_finished(running_put_requests)
        running_put_requests = []
    print("mixed read/write done")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time} s")
    print(f"Total cache hit rate: {total_cache_hit / (total_cache_hit + total_cache_miss)}")
    if enable_cpu and num_cpu_blocks >= num_gpu_blocks or \
        enable_ssd and num_ssd_blocks >= num_gpu_blocks or \
        enable_remote and num_remote_blocks >= num_gpu_blocks:
        assert total_cache_miss == 0
    kvmanager.shutdown()

    if total_cache_miss == 0 and not use_server_client:
        # Only verify data in direct mode
        verify_data(gpu_blocks, dp_wise_gpu_blocks_gt, num_kv_heads, tp_size, dp_size, num_layers, use_mla)
    elif total_cache_miss > 0:
        print(f"verify skipped, because of total_cache_miss={total_cache_miss}>0")
    elif use_server_client:
        print("verify skipped in server-client mode (verification happens in tp_client processes)")
