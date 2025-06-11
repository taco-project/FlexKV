from argparse import ArgumentParser

import cProfile
import pstats
import torch

from flexkv.cache.mempool import Mempool


def main(args):
    mempool = Mempool(num_total_blocks=args.num_total_blocks)
    print(f"num_total_blocks: {mempool.num_total_blocks}")

    profiler = cProfile.Profile()
    for i in range(args.nloops):
        profiler.runctx('block_ids = mempool.allocate_blocks(args.num_alloc_blocks)',
                        globals(), locals())
        print(f"allocate {args.num_alloc_blocks} blocks")
        profiler.runctx('mempool.recycle_blocks(block_ids)',
                        globals(), locals())
        print(f"recycle {args.num_alloc_blocks} blocks")
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Benchmark the performance of Mempool.')
    parser.add_argument('--num-total-blocks', type=int, default=1000000)
    parser.add_argument('--num-alloc-blocks', type=int, default=32000)
    parser.add_argument('--nloops', type=int, default=1000)
    args = parser.parse_args()
    main(args)
