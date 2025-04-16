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
    profiler = cProfile.Profile()
    print("insert sequence of length", insert_sequence_meta.length)
    index.insert(insert_sequence_meta)
    print("match sequence of length", match_sequence_meta.length)
    profiler.runctx('index.match_prefix(match_sequence_meta)',
                    globals(), locals())
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    for func in stats.stats:
        if func[2] in dir(index) and not func[2].startswith('__')\
            or func[2].startswith('hash'):
            print(f"function: {func[2]:<30} "
                  f"total time: {stats.stats[func][3]:.6f}s  "
                  f"total calls: {stats.stats[func][0]}")

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Benchmark the performance of Index.')
    parser.add_argument('--sequence-length', type=int, default=32000)
    parser.add_argument('--tokens-per-block', type=int, default=1)
    parser.add_argument('--cache-ratio', type=float, default=1)
    args = parser.parse_args()
    main(args)
