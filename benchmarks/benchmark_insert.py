from argparse import ArgumentParser

import cProfile
import pstats
import torch

from flexkv.cache.radixtree import RadixTreeIndex
from flexkv.common.block import SequenceMeta


def main(args):
    index = RadixTreeIndex(tokens_per_block=args.tokens_per_block,
                              max_num_blocks=args.max_num_blocks)
    token_ids = torch.randint(0, 10000, (args.sequence_length, ), dtype=torch.int64)
    insert_sequence_meta = SequenceMeta(token_ids=token_ids,
                                        tokens_per_block=args.tokens_per_block)

    profiler = cProfile.Profile()
    print("insert sequence of length", insert_sequence_meta.length)
    physical_block_ids = torch.arange(insert_sequence_meta.num_blocks, dtype=torch.int64)
    profiler.runctx('index.insert(insert_sequence_meta, physical_block_ids)',
                    globals(), locals())
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Benchmark the performance of Index.')
    parser.add_argument('--sequence-length', type=int, default=32000)
    parser.add_argument('--tokens-per-block', type=int, default=1)
    parser.add_argument('--max-num-blocks', type=int, default=1000000)
    args = parser.parse_args()
    main(args)
