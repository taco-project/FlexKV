from typing import List
import cProfile
import pstats

import torch

from request_generator import RequestGenerator, KVRequest
from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.config import CacheConfig, ModelConfig
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
                                         put_per_output_tokens=1000)
    reqs: List[KVRequest] = request_generator.generate()
    cache_config = CacheConfig(
        enable_cpu=True,
        enable_ssd=True,
        enable_remote=False,
        use_gds=False,
        tokens_per_block=16,
        num_cpu_blocks=10000,
        num_ssd_blocks=100000,
    )
    model_config = ModelConfig(
        num_layers=32,
        num_kv_heads=32,
        head_size=4096,
        element_size=4,
        use_mla=False
    )
    cache_engine = GlobalCacheEngine(cache_config, model_config)
    profiler = cProfile.Profile()
    avg_cache_hit_ratio = 0
    max_cache_hit_ratio = 0
    sum_cache_hit = 0
    for req in reqs:
        fake_slot_mapping = torch.arange(req.token_mask[req.token_mask].sum(), dtype=torch.int64)
        local_vars = {
            'cache_engine': cache_engine,
            'req': req,
            'fake_slot_mapping': fake_slot_mapping,
        }
        if req.request_type == "get":
            profiler.runctx('graph, return_mask, transfer_call_back, finished_ops_ids = '
                            'cache_engine.get(req.token_ids, req.token_mask, fake_slot_mapping)',
                            globals(), local_vars)
            profiler.runctx('transfer_call_back()', globals(), local_vars)
            cache_hit_ratio = local_vars['return_mask'].sum() / req.token_mask.sum()
            sum_cache_hit += cache_hit_ratio
            max_cache_hit_ratio = max(max_cache_hit_ratio, cache_hit_ratio)
            debuginfo.info(f"need get {req.token_mask.sum()} tokens, "
                           f"actual get {local_vars['return_mask'].sum()} tokens, "
                           f"cache_hit_ratio: {cache_hit_ratio}")
        elif req.request_type == "put":
            profiler.runctx('graph, return_mask, transfer_call_back = '
                            'cache_engine.put(req.token_ids, req.token_mask, fake_slot_mapping)',
                            globals(), local_vars)
            profiler.runctx('transfer_call_back()', globals(), local_vars)
            debuginfo.info(f"need put {req.token_mask.sum()} tokens, "
                           f"actual put {local_vars['return_mask'].sum()} tokens")
    debuginfo.info(f"Total requests: {len(reqs)}")
    avg_cache_hit_ratio = sum_cache_hit / len(reqs)
    debuginfo.info(f"Avg cache hit ratio: {avg_cache_hit_ratio}, max cache hit ratio: {max_cache_hit_ratio}")
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()


if __name__ == "__main__":
    main()
