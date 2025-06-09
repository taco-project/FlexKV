import torch
from flexkv.kvmanager import KVManager
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.debug import debuginfo
import time

num_layers = 16
num_kv_heads = 32
head_size = 128
element_size = 2
use_mla = False
tp_size = 2
dp_size = 2
tokens_per_block = 16
enable_cpu = True
enable_ssd = True
block_per_request = 16
enable_remote = False

num_gpu_blocks = 512
num_cpu_blocks = 128
num_ssd_blocks = 256 if enable_remote else 512
num_remote_blocks = 512

default_kv_layout = KVCacheLayout(
    type=KVCacheLayoutType.LAYERWISE, 
    num_layer=num_layers, 
    num_block=num_gpu_blocks, 
    tokens_per_block=tokens_per_block, 
    num_head=num_kv_heads, 
    head_size=head_size,
    is_mla=False
)

gpu_kv_layout = KVCacheLayout(
    type=KVCacheLayoutType.LAYERWISE, 
    num_layer=num_layers, 
    num_block=num_gpu_blocks, 
    tokens_per_block=tokens_per_block, 
    num_head=num_kv_heads//tp_size, 
    head_size=head_size,
    is_mla=False
)

# the ratio of gpu blocks to be written in the initial write
initial_write_ratio = 0.4

num_requests = num_gpu_blocks // block_per_request

def generate_request_pair(idx: int):
    """generate a pair of matching token_ids and block_ids"""
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

    # generate random token_ids with shape
    # (block_per_request * tokens_per_block,)
    token_ids = torch.randint(
        low=0,
        high=100,
        size=(block_per_request * tokens_per_block,),
        dtype=torch.int64
    )

    return token_ids, block_ids, idx % dp_size

def verify_data(gpu_blocks, dp_wise_gpu_blocks_gt):
    # traverse each dp group
    head_per_tp = num_kv_heads // tp_size
    for dp_id in range(dp_size):
        gt = dp_wise_gpu_blocks_gt[dp_id]  # the ground truth of the dp group
        for tp_id in range(tp_size):
            global_gpu_id = dp_id * tp_size + tp_id
            gpu_tensors = gpu_blocks[global_gpu_id]
            for layer in range(num_layers):
                gpu_tensor = gpu_tensors[layer].cpu()
                # extract the corresponding slice from ground truth
                start_head = tp_id * head_per_tp
                end_head = (tp_id + 1) * head_per_tp
                gt_tensor_slice = gt[layer][:, :, :, start_head:end_head, :]
                
                if not torch.allclose(gpu_tensor, gt_tensor_slice):
                    print(f"Mismatch at dp_id={dp_id}, tp_id={tp_id}, global_gpu_id={global_gpu_id}, layer={layer}")
                    print(f"GPU tensor shape: {gpu_tensor.shape}, GT slice shape: {gt_tensor_slice.shape}")
                assert torch.allclose(gpu_tensor, gt_tensor_slice), \
                    f"Mismatch at dp_id={dp_id}, tp_id={tp_id}, global_gpu_id={global_gpu_id}, layer={layer}"
    print("verify done")

def block_ids_2_slot_mapping(block_ids):
    slot_mapping = block_ids.repeat_interleave(tokens_per_block) * tokens_per_block
    return slot_mapping

def test_kvmanager():
    model_config = ModelConfig(num_layers=num_layers,
                               num_kv_heads=num_kv_heads,
                               head_size=head_size,
                               element_size=element_size,
                               use_mla=use_mla,
                               tp_size=tp_size,
                               dp_size=dp_size)
    cache_config = CacheConfig(raw_gpu_blocks=True,
                               enable_cpu=True,
                               enable_ssd=True,
                               enable_remote=enable_remote,
                               gpu_kv_layout=gpu_kv_layout,
                               cpu_kv_layout=default_kv_layout,
                               ssd_kv_layout=default_kv_layout,
                               remote_kv_layout=default_kv_layout,
                               use_gds=False,
                               use_pinned_memory=True,
                               tokens_per_block=tokens_per_block,
                               num_cpu_blocks=num_cpu_blocks,
                               num_ssd_blocks=num_ssd_blocks,
                               num_remote_blocks=num_remote_blocks,
                               ssd_cache_path=["ssd_cache1", "ssd_cache2"],
                               remote_cache_path=["remote_cache1", "remote_cache2"],
                               remote_config_custom={
                                    "pcfs_fsid": "f_l91fz6",
                                    "pcfs_port": 31,
                                    "pcfs_ip": "172.21.16.177",
                                    "pcfs_parent_nodeid": 144115188075855883
                               })
    gpu_blocks = {}
    dp_wise_gpu_blocks_gt = []
    for dp_id in range(dp_size):
        # generate ground truth tensors with full head dimension
        # shape: [2, num_blocks, tokens_per_block, num_head, head_size]
        tp_group_tensors_gt = [
            torch.randn(size=default_kv_layout.get_kv_shape()[1:], dtype=torch.float16)
            for _ in range(num_layers)
        ]
        
        # split tensors across TP ranks in head dimension
        head_per_tp = num_kv_heads // tp_size
        for tp_id in range(tp_size):
            global_gpu_id = dp_id * tp_size + tp_id
            device = torch.device(f"cuda:{global_gpu_id}")
            
            # split each layer tensor in head dimension for this TP rank
            # each TP rank gets [2, num_blocks, tokens_per_block, num_head//tp_size, head_size]
            gpu_blocks[global_gpu_id] = []
            for layer_id in range(num_layers):
                start_head = tp_id * head_per_tp
                end_head = (tp_id + 1) * head_per_tp
                # split in head dimension (dimension 3)
                tp_tensor = tp_group_tensors_gt[layer_id][:, :, :, start_head:end_head, :].to(device)
                gpu_blocks[global_gpu_id].append(tp_tensor)
        
        dp_wise_gpu_blocks_gt.append(tp_group_tensors_gt)
        
    kvmanager = KVManager(model_config, cache_config, gpu_blocks)

    request_pairs = [generate_request_pair(i) for i in range(num_requests)]

    # write initial data
    initial_write_num = int(num_requests * initial_write_ratio)
    print("writing initial data...")
    for token_ids, block_ids, dp_id in request_pairs[:initial_write_num]:
        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids),
            dp_id=dp_id,
        )
        kvmanager.wait_for_task_finished(write_request)
        # clear gpu blocks
        for gpu in range(dp_id * tp_size, (dp_id + 1) * tp_size):
            for i in range(num_layers):
                gpu_blocks[gpu][i][:, block_ids, :, :, :] = 0
    print(f"initial data {initial_write_num} written")
    # mixed read/write test
    total_cache_hit = 0
    total_cache_miss = 0
    running_get_requests = []
    running_put_requests = []
    start_time = time.time()
    print(f"the initial {initial_write_num} write done,performing mixed read/write...")
    for i in range(initial_write_num, num_requests):
        print(f"performing mixed read/write {i} / {num_requests} ...")
        # read from written data
        read_idx = i - initial_write_num
        token_ids, block_ids, dp_id = request_pairs[read_idx]
        request_id = kvmanager.get_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids),
            layer_granularity=-1,
            dp_id=dp_id,
        )
        running_get_requests.append(request_id)

        # write new data
        token_ids, block_ids, dp_id = request_pairs[i]
        request_id = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids),
            dp_id=dp_id,
        )
        running_put_requests.append(request_id)
        # to aviod that all seq are locked, and cannot eviction
        if len(running_get_requests) + len(running_put_requests) >= num_cpu_blocks // block_per_request - 2:
            if len(running_put_requests) > 0:
                kvmanager.wait_for_task_finished(running_put_requests)
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
        kvmanager.wait_for_task_finished(running_put_requests)
        running_put_requests = []
    print("mixed read/write done")
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time: {total_time} s")
    print(f"Total cache hit rate: {total_cache_hit / (total_cache_hit + total_cache_miss)}")
    kvmanager.shutdown()
    if(total_cache_miss == 0):
        verify_data(gpu_blocks, dp_wise_gpu_blocks_gt)
    else:
        print(f"verify skipped, because of total_cache_miss={total_cache_miss}>0")

if __name__ == "__main__":
    debuginfo.set_level("INFO")
    test_kvmanager()
