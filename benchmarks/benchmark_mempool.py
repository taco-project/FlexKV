import cProfile
import pstats
from argparse import ArgumentParser

import torch

from flexkv.cache.mempool import Mempool

def main(args):
    mempool = Mempool(num_total_blocks=args.num_total_blocks)
    print(f"num_total_blocks: {mempool.num_total_blocks}")
    profiler = cProfile.Profile()
    profiler.runctx('block_ids = mempool.allocate_blocks(args.num_alloc_blocks)',
                    globals(), locals())
    print(f"allocated {args.num_alloc_blocks} blocks")
    profiler.runctx('print(f"num_free_blocks: {mempool.num_free_blocks}, num_used_blocks: {mempool.num_used_blocks}")',
                    globals(), locals())
    profiler.runctx('mempool.free_blocks(block_ids)',
                    globals(), locals())
    print(f"freed {args.num_alloc_blocks} blocks")
    profiler.runctx('print(f"num_free_blocks: {mempool.num_free_blocks}, num_used_blocks: {mempool.num_used_blocks}")',
                    globals(), locals())
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    for func in stats.stats:
        if func[2] in dir(mempool) and not func[2].startswith('__'):
            print(f"function: {func[2]:<30} "
                  f"total time: {stats.stats[func][3]:.6f}s  "
                  f"total calls: {stats.stats[func][0]}")

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Benchmark the performance of Mempool.')
    parser.add_argument('--num-total-blocks', type=int, default=10000000)
    parser.add_argument('--num-alloc-blocks', type=int, default=1000)
    args = parser.parse_args()
    main(args)
