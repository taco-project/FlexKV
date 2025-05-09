import cProfile
import pstats
from argparse import ArgumentParser

import torch

from flexkv.cache.index import TokenToBlockIndex
from flexkv.common.block import SequenceMeta


def main(args):
    index = TokenToBlockIndex(tokens_per_block=args.tokens_per_block)
    token_ids = torch.randint(0, 10000, (args.sequence_length, ), dtype=torch.int64)
    insert_token_ids = token_ids[:int(args.sequence_length*args.cache_ratio)]
    insert_sequence_meta = SequenceMeta(token_ids=insert_token_ids,
                                        tokens_per_block=args.tokens_per_block)
    match_sequence_meta = SequenceMeta(token_ids=token_ids,
                                       tokens_per_block=args.tokens_per_block)

    print("insert sequence of length", insert_sequence_meta.length)
    physical_block_ids = torch.arange(insert_sequence_meta.num_blocks, dtype=torch.int64)
    index.insert(insert_sequence_meta, physical_block_ids, 0)

    profiler = cProfile.Profile()
    print("match sequence of length", match_sequence_meta.length)
    for i in range(args.nloops):
        profiler.runctx('index.match_prefix(match_sequence_meta)',
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
    parser.add_argument('--cache-ratio', type=float, default=1)
    parser.add_argument('--nloops', type=int, default=1000)
    args = parser.parse_args()
    main(args)
