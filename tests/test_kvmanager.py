import torch
from flexkv.kvmanager import KVManager
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.debug import debuginfo
import time

num_layers = 16
num_kv_heads = 32
head_size = 128
element_size = 2
use_mla = False
tp_size = 1
tokens_per_block = 16
enable_cpu = True
enable_ssd = True
block_per_request = 16

num_gpu_blocks = 256
num_cpu_blocks = 60
num_ssd_blocks = 34
num_remote_blocks = 256

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

    return token_ids, block_ids

def verify_data(gpu_blocks, gpu_blocks_gt):
    # verify data correctness
    for i in range(num_layers):
        gpu_k = gpu_blocks[i][0, :, :, :, :]
        gpu_v = gpu_blocks[i][1, :, :, :, :]
        gpu_k_gt = gpu_blocks_gt[i][0, :, :, :, :]
        gpu_v_gt = gpu_blocks_gt[i][1, :, :, :, :]

        num_blocks = gpu_k.shape[0]
        for block_idx in range(num_blocks):
            if not torch.allclose(gpu_k[block_idx], gpu_k_gt[block_idx]):
                print(f"K mismatch at layer {i}, block {block_idx} / {num_blocks}")
        for block_idx in range(num_blocks):
            if not torch.allclose(gpu_v[block_idx], gpu_v_gt[block_idx]):
                print(f"V mismatch at layer {i}, block {block_idx} / {num_blocks}")
        assert torch.allclose(gpu_k, gpu_k_gt), f"K mismatch at layer {i}"
        assert torch.allclose(gpu_v, gpu_v_gt), f"V mismatch at layer {i}"

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
                               tp_size=tp_size)
    cache_config = CacheConfig(enable_cpu=True,
                               enable_ssd=True,
                               enable_remote=True,
                               use_gds=False,
                               use_pinned_memory=True,
                               tokens_per_block=tokens_per_block,
                               num_cpu_blocks=num_cpu_blocks,
                               num_ssd_blocks=num_ssd_blocks,
                               num_remote_blocks=num_remote_blocks,
                               ssd_cache_path=["ssd_cache1"],
                               remote_cache_path=["remote_cache1"])
    gpu_blocks = [torch.randn(2, num_gpu_blocks, tokens_per_block, num_kv_heads, head_size, dtype=torch.float16).cuda()
                  for _ in range(num_layers)]
    gpu_blocks_gt = [block.clone() for block in gpu_blocks]
    kvmanager = KVManager(model_config, cache_config, [gpu_blocks])

    request_pairs = [generate_request_pair(i) for i in range(num_requests)]

    # write initial data
    initial_write_num = num_requests // (
        num_gpu_blocks // num_cpu_blocks
    ) - 2
    write_requests = []
    print("writing initial data...")
    for token_ids, block_ids in request_pairs[:initial_write_num]:
        request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids),
        )
        write_requests.append(request)
    kvmanager.wait(write_requests)
    print("initial data written")
    # mixed read/write test
    total_data_size = 0
    all_requests = []
    start_time = time.time()
    print(f"the initial {initial_write_num} write done,performing mixed read/write...")
    for i in range(initial_write_num, num_requests):
        print(f"performing mixed read/write {i} / {num_requests} ...")
        # read from written data
        read_idx = i - initial_write_num
        token_ids, block_ids = request_pairs[read_idx]
        request = kvmanager.get_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids),
            layer_granularity=-1,
        )
        #all_requests.append(request)
        if False:
            for layer_id in range(4):
                s_time = time.time()
                kvmanager.wait_at_layer_group(request, layer_id, last_layer=(layer_id==3))
                e_time = time.time()
                print(f"for task {request}, wait at layer {layer_id} done, time: {e_time - s_time} s")
        else:
            all_requests.append(request)

        # write new data
        token_ids, block_ids = request_pairs[i]
        request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids),
        )
        all_requests.append(request)
        total_data_size += (
            block_per_request * tokens_per_block * num_kv_heads
            * head_size * element_size * num_layers * 2
        )
        # to aviod that all seq are locked, and cannot eviction
        if len(all_requests) >= num_cpu_blocks // block_per_request - 2:
            kvmanager.wait(all_requests)
            all_requests = []
    if len(all_requests) > 0:
        kvmanager.wait(all_requests)
    print("mixed read/write done")
    end_time = time.time()

    total_data_size = 2 * total_data_size / (1024 * 1024 * 1024)
    total_time = (
        end_time - start_time
    )
    print(f"total time: {total_time} s")
    throughput = total_data_size / total_time

    kvmanager.shutdown()
    verify_data(gpu_blocks, gpu_blocks_gt)

if __name__ == "__main__":
    debuginfo.set_level("INFO")
    test_kvmanager()
