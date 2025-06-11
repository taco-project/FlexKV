import random
import time
from request_generator import RequestGenerator, KVRequest
from typing import List

import cProfile
import numpy as np
import pstats
import torch

from flexkv.cache.mempool import Mempool
from flexkv.cache.radixtree import RadixTreeIndex
from flexkv.common.block import SequenceMeta
from flexkv.common.debug import debuginfo


debuginfo.set_level("INFO")

def main():
    request_generator = RequestGenerator(dataset="random",
                                         dataset_path=None,
                                         num_user_requests=100,
                                         max_num_turns=10,
                                         system_prompt_length=100,
                                         max_input_length=1000,
                                         max_output_length=1000,
                                         num_gpus=1,
                                         tp_size=1,
                                         max_parallel_per_device=16,
                                         request_rate=1000,
                                         approx_ttft=0.001,
                                         approx_tpot=0.0001,
                                         put_per_output_tokens=1000,
                                         random_seed=42)
    reqs: List[KVRequest] = request_generator.generate()
    max_sequence_length = 100 + 10 * (1000 + 1000)
    max_num_blocks = max_sequence_length * 4
    tokens_per_block = 16
    cache_index = RadixTreeIndex(tokens_per_block=tokens_per_block, max_num_blocks=max_num_blocks)
    mempool = Mempool(num_total_blocks=max_num_blocks)
    profiler = cProfile.Profile()
    num_total_evicted = 0
    num_eviction = 0
    num_total_matched_blocks = 0
    for req in reqs:
        local_vars = {
            'cache_index': cache_index,
            'mempool': mempool,
            'req': req,
        }
        sequence_meta = SequenceMeta(token_ids=req.token_ids, tokens_per_block=tokens_per_block)
        if req.request_type == "get":
            num_matched_blocks = 0
            num_total_matched_blocks += num_matched_blocks
            debuginfo.info(f"cache hit ratio: {num_matched_blocks * tokens_per_block / len(req.token_ids)}")
        elif req.request_type == "put":
            match_result = cache_index.match_prefix(sequence_meta)
            num_matched_blocks = match_result.num_matched_blocks
            ret_node = match_result.last_node
            required_blocks_num = sequence_meta.num_blocks - num_matched_blocks
            if required_blocks_num > mempool.num_free_blocks:
                cache_index.lock(ret_node)
                local_vars['required_blocks_num'] = required_blocks_num
                profiler.runctx('evicted_blocks = cache_index.evict(required_blocks_num - mempool.num_free_blocks)',
                                globals(), local_vars)
                mempool.recycle_blocks(local_vars['evicted_blocks'])
                num_total_evicted += len(local_vars['evicted_blocks'])
                num_eviction += 1
                cache_index.unlock(ret_node)
            new_block_ids = mempool.allocate_blocks(required_blocks_num)
            cache_index.insert(sequence_meta, new_block_ids)
    debuginfo.info(f"Total requests: {len(reqs)}")
    debuginfo.info(f"{num_eviction} eviction happened")
    debuginfo.info(f"Total evicted blocks: {num_total_evicted}")
    debuginfo.info(f"Total matched blocks: {num_total_matched_blocks}")
    if num_total_evicted > 0:
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats()
    else:
        debuginfo.info("No evictions")

if __name__ == "__main__":
    main()
