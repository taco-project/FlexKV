from argparse import ArgumentParser
import json

import cProfile
import pstats
import torch

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from utils import generate_random_multiturn, load_config

flexkv_logger.set_level("OFF")

def main(args):
    reqs = generate_random_multiturn(num_user_requests=args.num_users,
                                     num_turns=args.num_turns,
                                     system_prompt_length=args.system_prompt_length,
                                     input_length=args.input_length,
                                     output_length=args.output_length)
    config_file = args.config
    model_config, cache_config = load_config(config_file)

    print(f"model_config: {model_config}")
    print(f"cache_config: {cache_config}")

    cache_engine = GlobalCacheEngine(cache_config, model_config)
    profiler = cProfile.Profile()
    cache_hit_ratio_list = []
    num_get_requests = 0
    num_put_requests = 0
    request_id = 0
    for req in reqs:
        fake_slot_mapping = torch.arange(req.token_mask[req.token_mask].sum(), dtype=torch.int64).numpy()
        local_vars = {
            'cache_engine': cache_engine,
            'req': req,
            'fake_slot_mapping': fake_slot_mapping,
            'request_id': request_id,
        }
        if req.request_type == "get":
            num_get_requests += 1
            if not args.only_put:
                profiler.runctx('graph, return_mask, transfer_call_back, op_callback_dict, finished_ops_ids = '
                                'cache_engine.get(request_id, req.token_ids, req.token_mask, '
                                'fake_slot_mapping, -1, -1)',
                                globals(), local_vars)
            else:
                graph, return_mask, transfer_call_back, op_callback_dict, finished_ops_ids = \
                    cache_engine.get(request_id, req.token_ids, req.token_mask,
                                   fake_slot_mapping, -1, -1)
                local_vars.update({
                    'graph': graph,
                    'return_mask': return_mask,
                    'transfer_call_back': transfer_call_back,
                    'op_callback_dict': op_callback_dict,
                    'finished_ops_ids': finished_ops_ids
                })
            profiler.runctx('transfer_call_back()', globals(), local_vars)

            return_mask = local_vars['return_mask']
            op_callback_dict = local_vars['op_callback_dict']
            cache_hit_ratio = return_mask.sum() / req.token_mask.sum()
            cache_hit_ratio_list.append(cache_hit_ratio)
            flexkv_logger.info(f"need get {req.token_mask.sum()} tokens, "
                           f"actual get {return_mask.sum()} tokens, "
                           f"cache_hit_ratio: {cache_hit_ratio}")
        elif req.request_type == "put":
            num_put_requests += 1
            if not args.only_get:
                profiler.runctx('graph, return_mask, transfer_call_back, op_callback_dict, finished_ops_ids = '
                                'cache_engine.put(request_id, req.token_ids, req.token_mask, fake_slot_mapping)',
                                globals(), local_vars)
            else:
                graph, return_mask, transfer_call_back, op_callback_dict, finished_ops_ids = \
                    cache_engine.put(request_id, req.token_ids, req.token_mask, fake_slot_mapping)
                local_vars.update({
                    'graph': graph,
                    'return_mask': return_mask,
                    'transfer_call_back': transfer_call_back,
                    'op_callback_dict': op_callback_dict,
                    'finished_ops_ids': finished_ops_ids
                })

            profiler.runctx('transfer_call_back()', globals(), local_vars)

            return_mask = local_vars['return_mask']
            flexkv_logger.info(f"need put {req.token_mask.sum()} tokens, "
                           f"actual put {return_mask.sum()} tokens")
        request_id += 1  # noqa: SIM113
    print(f"total KV Requests: {len(reqs)}")
    sorted_cache_hit_ratio_list = sorted(cache_hit_ratio_list)
    print(f"cache hit ratio: Avg={100 * sum(sorted_cache_hit_ratio_list) / len(reqs):.2f}%, "
          f"max={100 * sorted_cache_hit_ratio_list[-1]:.3f}%, "
          f"min={100 * sorted_cache_hit_ratio_list[0]:.3f}%, "
          f"median={100 * sorted_cache_hit_ratio_list[len(sorted_cache_hit_ratio_list) // 2]:.3f}%")
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    for func in stats.stats:
        if func[2] in dir(cache_engine) and not func[2].startswith('__'):
            print(f"function: {func[2]:<25} "
                  f"total time: {stats.stats[func][3]:.3f}s  "
                  f"total calls: {stats.stats[func][0]:<7}"
                  f"avg time: {1000 * stats.stats[func][3] / stats.stats[func][0]:.3f}ms")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        default="./benchmarks/example_config.yml")
    parser.add_argument("--only-get", action="store_true")
    parser.add_argument("--only-put", action="store_true")
    parser.add_argument("--num-users", type=int, default=20)
    parser.add_argument("--num-turns", type=int, default=5)
    parser.add_argument("--system-prompt-length", type=int, default=100)
    parser.add_argument("--input-length", type=int, default=1000)
    parser.add_argument("--output-length", type=int, default=64)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.only_get and args.only_put:
        raise ValueError("only-get and only-put cannot be set at the same time")
    main(args)
