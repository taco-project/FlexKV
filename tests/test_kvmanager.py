import time
import os
import shutil

import pytest
import torch

from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.kvmanager import KVManager

DEFAULT_MODEL_CONFIG = {
    'num_layers': 4,
    'num_kv_heads': 32,
    'head_size': 128,
    'dtype': torch.float16,
    'use_mla': False,
    'tp_size': 1,
    'dp_size': 1,
}

DEFAULT_CACHE_CONFIG = {
    'tokens_per_block': 16,
    'enable_cpu': True,
    'enable_ssd': True,
    'enable_remote': False,
    'num_cpu_blocks': 128,
    'num_ssd_blocks': 512,
    'num_remote_blocks': 512,
    'remote_cache_size_mode': "block_num",
    'remote_file_size': (1024*1024*1024),
    'remote_file_num': 16,
    'remote_file_prefix': "remote_cache",
    'use_gds': False,
    'use_pinned_memory': True,
    'ssd_cache_dir': ["./ssd_cache1/", "./ssd_cache2/"],
    'remote_cache_path': ["remote_cache1", "remote_cache2"],
    'remote_config_custom': {
        "pcfs_fsid": "f_l91fz6",
        "pcfs_port": 31,
        "pcfs_ip": "172.21.16.177",
        "pcfs_parent_nodeid": 1
    }
}

DEFAULT_TEST_CONFIG = {
    'num_gpu_blocks': 512,
    'requests_per_block': 16,
    'initial_write_ratio': 0.4,
}

def generate_request_pair(idx: int, block_per_request, num_gpu_blocks, tokens_per_block, dp_size):
    start_idx = (idx * block_per_request) % num_gpu_blocks
    if start_idx + block_per_request >= num_gpu_blocks:
        start_idx = (
            (start_idx + block_per_request) % num_gpu_blocks
        )
    block_ids = torch.arange(
        start_idx,
        start_idx + block_per_request,
        dtype=torch.int64
    )
    token_ids = torch.randint(
        low=0,
        high=100,
        size=(block_per_request * tokens_per_block,),
        dtype=torch.int64
    )
    return token_ids, block_ids, idx % dp_size

def verify_data(gpu_blocks, dp_wise_gpu_blocks_gt, num_kv_heads, tp_size, dp_size, num_layers, use_mla):
    head_per_tp = num_kv_heads // tp_size
    for dp_id in range(dp_size):
        gt = dp_wise_gpu_blocks_gt[dp_id]
        for tp_id in range(tp_size):
            global_gpu_id = dp_id * tp_size + tp_id
            gpu_tensors = gpu_blocks[global_gpu_id]
            for layer in range(num_layers):
                gpu_tensor = gpu_tensors[layer].cpu()
                if not use_mla:
                    start_head = tp_id * head_per_tp
                    end_head = (tp_id + 1) * head_per_tp
                else:
                    start_head = 0
                    end_head = num_kv_heads
                gt_tensor_slice = gt[layer][:, :, :, start_head:end_head, :]
                if not torch.allclose(gpu_tensor, gt_tensor_slice):
                    print(f"Mismatch at dp_id={dp_id}, tp_id={tp_id}, global_gpu_id={global_gpu_id}, layer={layer}")
                    print(f"GPU tensor shape: {gpu_tensor.shape}, GT slice shape: {gt_tensor_slice.shape}")
                assert torch.allclose(gpu_tensor, gt_tensor_slice), \
                    f"Mismatch at dp_id={dp_id}, tp_id={tp_id}, global_gpu_id={global_gpu_id}, layer={layer}"
    print("verify done")

def block_ids_2_slot_mapping(block_ids, tokens_per_block):
    slot_mapping = block_ids.repeat_interleave(tokens_per_block) * tokens_per_block
    return slot_mapping

@pytest.fixture
def model_config(request: pytest.FixtureRequest):
    param = request.param if hasattr(request, 'param') else {}
    cfg = dict(DEFAULT_MODEL_CONFIG, **param)
    return ModelConfig(**cfg)

@pytest.fixture
def cache_config(request: pytest.FixtureRequest):
    param = request.param if hasattr(request, 'param') else {}
    cfg = dict(DEFAULT_CACHE_CONFIG, **param)
    return CacheConfig(**cfg)

@pytest.fixture
def test_config(request: pytest.FixtureRequest):
    param = request.param if hasattr(request, 'param') else {}
    cfg = dict(DEFAULT_TEST_CONFIG, **param)
    return cfg

def generate_gpu_blocks(model_config, cache_config, test_config):
    num_layers = model_config.num_layers
    num_kv_heads = model_config.num_kv_heads
    head_size = model_config.head_size
    use_mla = model_config.use_mla
    tp_size = model_config.tp_size
    dp_size = model_config.dp_size
    dtype = model_config.dtype
    tokens_per_block = cache_config.tokens_per_block
    num_gpu_blocks = test_config["num_gpu_blocks"]

    tpgroup_gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_kv_heads,
        head_size=head_size,
        is_mla=model_config.use_mla
    )
    gpu_blocks = {}
    dp_wise_gpu_blocks_gt = []
    for dp_id in range(dp_size):
        tp_group_tensors_gt = [
            torch.randn(size=tpgroup_gpu_kv_layout.kv_shape[1:], dtype=dtype)
            for _ in range(num_layers)
        ]
        head_per_tp = num_kv_heads // tp_size
        for tp_id in range(tp_size):
            global_gpu_id = dp_id * tp_size + tp_id
            device = torch.device(f"cuda:{global_gpu_id}")
            gpu_blocks[global_gpu_id] = []
            for layer_id in range(num_layers):
                if not use_mla:
                    start_head = tp_id * head_per_tp
                    end_head = (tp_id + 1) * head_per_tp
                else:
                    start_head = 0
                    end_head = num_kv_heads
                tp_tensor = tp_group_tensors_gt[layer_id][:, :, :, start_head:end_head, :].to(device)
                gpu_blocks[global_gpu_id].append(tp_tensor)
        dp_wise_gpu_blocks_gt.append(tp_group_tensors_gt)
    gpu_kv_layout = tpgroup_gpu_kv_layout.div_head(tp_size) if not use_mla else tpgroup_gpu_kv_layout
    return gpu_blocks, dp_wise_gpu_blocks_gt, gpu_kv_layout

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
    {'enable_cpu': True, 'enable_ssd': True, 'enable_remote': True, 'num_ssd_blocks': 256, 'num_remote_blocks': 512},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 512, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
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

    num_requests = num_gpu_blocks // block_per_request

    if tp_size * dp_size > torch.cuda.device_count():
        pytest.skip("skip because tp_size * dp_size > torch.cuda.device_count()")
    if enable_remote:
        pytest.skip("skip because enable_remote is not supported")

    gpu_blocks, dp_wise_gpu_blocks_gt, gpu_kv_layout = generate_gpu_blocks(model_config, cache_config, test_config)
    kvmanager = KVManager(model_config, cache_config, gpu_kv_layout, gpu_blocks)
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
        for gpu in range(dp_id * tp_size, (dp_id + 1) * tp_size):
            for i in range(num_layers):
                gpu_blocks[gpu][i][:, block_ids, :, :, :] = 0
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
    if enable_ssd:
        for dir in cache_config.ssd_cache_dir:
            if os.path.exists(dir):
                shutil.rmtree(dir)
    if total_cache_miss == 0:
        verify_data(gpu_blocks, dp_wise_gpu_blocks_gt, num_kv_heads, tp_size, dp_size, num_layers, use_mla)
    else:
        print(f"verify skipped, because of total_cache_miss={total_cache_miss}>0")
