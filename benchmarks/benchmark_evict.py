import time
import cProfile
import pstats

import torch
from flexkv.cache.index import TokenToBlockIndex
from flexkv.cache.mempool import Mempool
import random
import numpy as np
from flexkv.common.block import SequenceMeta


def test_eviction_perf():
    """Test the performance of large-scale eviction operations."""

    # test parameters
    num_blocks = 1000000
    token_per_block = 2

    # initialize cache index
    cache_index = TokenToBlockIndex(tokens_per_block=token_per_block, max_num_blocks=num_blocks)
    mempool = Mempool(num_total_blocks=num_blocks)

    total_insert_time = 0
    total_evict_time = 0
    total_evicted = 0

    print("\nStarting eviction performance test:")
    print(f"Total blocks: {num_blocks}")

    # insert data until eviction is triggered
    blocks_inserted = 0
    blocks_evicted = 0
    start_time = time.time()

    seed = 123456
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    approxiamate_insert_times_before_eviction = 1000
    blocks_num_per_insert = num_blocks / approxiamate_insert_times_before_eviction // token_per_block * token_per_block

    # this is the history sequence meta
    history_sequence_meta = []
    inserted_blocks_len = random.randint(blocks_num_per_insert//5, blocks_num_per_insert)
    history_sequence_meta.append(
        SequenceMeta(torch.randint(0, 256, (inserted_blocks_len*token_per_block,), dtype=torch.uint8), token_per_block)
    )
    cache_index.insert(history_sequence_meta[0], mempool.allocate_blocks(inserted_blocks_len), 0)
    total_evict_time = 0
    total_evict_op_num = 0

    profiler = cProfile.Profile()
    time_start = time.time()
    while blocks_evicted < num_blocks//2:
        # generate new sequence meta
        inserted_blocks_len = random.randint(blocks_num_per_insert//5, blocks_num_per_insert)
        reuse_index = random.randint(0, len(history_sequence_meta)//2*3 + 2)
        if reuse_index < len(history_sequence_meta): # in half case, reuse the history sequence meta
            new_sequence_meta = SequenceMeta(torch.cat([history_sequence_meta[reuse_index].token_ids,
                                                        torch.randint(0,
                                                                      256,
                                                                      (inserted_blocks_len*token_per_block,),
                                                                      dtype=torch.uint8)]),
                                                        token_per_block)
        else: # in the other half case, generate a new sequence meta
            new_sequence_meta = SequenceMeta(torch.randint(0,
                                                           256,
                                                           (inserted_blocks_len*token_per_block,),
                                                           dtype=torch.uint8), token_per_block)
        prefix_block_ids = cache_index.match_prefix(new_sequence_meta)
        needed_blocks_num = new_sequence_meta.num_blocks - len(prefix_block_ids)
        if needed_blocks_num > mempool.num_free_blocks:
            total_evict_op_num += 1
            # lock the last block to avoid eviction
            if len(prefix_block_ids) > 0:
                cache_index.lock(prefix_block_ids[-1])
            local_vars = {
                'cache_index': cache_index,
                'mempool': mempool,
                'needed_blocks_num': needed_blocks_num,
                'prefix_block_ids': prefix_block_ids,
                'evicted_blocks': None,
            }
            time_start_evict = time.time()
            # evicted_blocks = cache_index.evict(needed_blocks_num - mempool.num_free_blocks)
            profiler.runctx('evicted_blocks = cache_index.evict(needed_blocks_num - mempool.num_free_blocks)',
                            globals(), local_vars)
            time_end_evict = time.time()
            evicted_blocks = local_vars['evicted_blocks']
            total_evict_time += time_end_evict - time_start_evict
            blocks_evicted += len(evicted_blocks)
            print(f"evict, need to evict {needed_blocks_num - mempool.num_free_blocks} blocks, "
                  f"sequence length {new_sequence_meta.num_blocks}, new inserted {needed_blocks_num} blocks, "
                  f"evicted {len(evicted_blocks)} blocks, time: {time_end_evict - time_start_evict:.4f}s")
            mempool.recycle_blocks(evicted_blocks)
            if len(prefix_block_ids) > 0:
                cache_index.unlock(prefix_block_ids[-1])

        new_block_ids = mempool.allocate_blocks(needed_blocks_num)
        assert len(new_block_ids) == needed_blocks_num
        cache_index.insert(new_sequence_meta, new_block_ids, len(prefix_block_ids))
        history_sequence_meta.append(new_sequence_meta)
    time_end = time.time()
    print(f"Time taken for {num_blocks} insertions: {time_end - time_start:.2f}s")
    print(f"Time taken for {num_blocks} evictions: {total_evict_time:.2f}s")
    print(f"Totally evicted {blocks_evicted} blocks, time to evict each block: {total_evict_time/blocks_evicted:.2f}s",
          f"time to evict each op: {total_evict_time/total_evict_op_num:.2f}s")

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()

if __name__ == "__main__":
    test_eviction_perf()
