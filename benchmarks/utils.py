import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any
import yaml

import torch
import numpy as np
from tqdm import tqdm

from flexkv.common.config import *
from flexkv.common.storage import KVCacheLayoutType


@dataclass
class KVRequest:
    user_id: int
    turn_id: int
    request_type: str  # "get" or "put"
    token_ids: np.ndarray
    token_mask: np.ndarray
    slot_mapping: Optional[np.ndarray] = None

    request_id: int = field(init=False)
    _request_id_counter: int = field(init=False, default=0)

    def __post_init__(self):
        self.request_id = KVRequest._request_id_counter
        KVRequest._request_id_counter += 1

        if isinstance(self.token_ids, torch.Tensor):
            self.token_ids = self.token_ids.numpy().astype(np.int64)
        if isinstance(self.token_mask, torch.Tensor):
            self.token_mask = self.token_mask.numpy().astype(np.int64)
        if isinstance(self.slot_mapping, torch.Tensor):
            self.slot_mapping = self.slot_mapping.numpy().astype(np.int64)

def generate_random_multiturn(num_user_requests: int,
                              num_turns: int,
                              system_prompt_length: int,
                              input_length: int,
                              output_length: int,
                              num_turns_ratio: float = 0.5,
                              input_length_ratio: float = 0.5,
                              output_length_ratio: float = 0.5) -> List[KVRequest]:
    all_requests = []
    token_id_range = 10000
    system_prompt = torch.randint(0, token_id_range, (system_prompt_length,))
    for i in range(num_user_requests):
        user_requests = []
        user_num_turns = max(random.randint(int(num_turns_ratio * num_turns), num_turns), 1)
        history = system_prompt.clone()
        for j in range(user_num_turns):
            turn_input_length = random.randint(int(input_length_ratio * input_length), input_length)
            turn_output_length = random.randint(int(output_length_ratio * output_length), output_length)
            input_tokens = torch.randint(0, token_id_range, (turn_input_length,))
            output_tokens = torch.randint(0, token_id_range, (turn_output_length,))
            request = dict(
                user_id=i,
                turn_id=j,
                input=torch.cat([history, input_tokens], dim=0),
                output=output_tokens,
            )
            history = torch.cat([history, input_tokens, output_tokens], dim=0)
            user_requests.append(request)
        all_requests.append(user_requests)
    indices = [0] * num_user_requests
    kv_requests = []
    while True:
        available_lists = [
            i for i in range(num_user_requests)
            if indices[i] < len(all_requests[i])
        ]
        if not available_lists:
            break
        user_id = random.choice(available_lists)
        request = all_requests[user_id][indices[user_id]]
        indices[user_id] += 1
        kv_requests.append(KVRequest(
            user_id=request["user_id"],
            turn_id=request["turn_id"],
            request_type="get",
            token_ids=request["input"],
            token_mask=torch.ones_like(request["input"]),
        ))
        kv_requests.append(KVRequest(
            user_id=request["user_id"],
            turn_id=request["turn_id"],
            request_type="put",
            token_ids=torch.cat([request["input"], request["output"]], dim=0),
            token_mask=torch.ones_like(torch.cat([request["input"], request["output"]], dim=0)),
        ))
    return kv_requests

def load_config(config_path: str) -> Tuple[ModelConfig, CacheConfig]:
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        print(config)
        model_config = ModelConfig()
        cache_config = CacheConfig()
        user_config = UserConfig()
        model_config.num_layers = config["num_layers"]
        model_config.num_kv_heads = config["num_kv_heads"]
        model_config.head_size = config["head_size"]
        model_config.dtype = eval(f"torch.{config['dtype']}")
        model_config.use_mla = config["use_mla"]
        model_config.tp_size = config["tp_size"]
        model_config.dp_size = config["dp_size"]
        cache_config.tokens_per_block = config["tokens_per_block"]

        if "cpu_cache_gb" in config:
            user_config.cpu_cache_gb = config["cpu_cache_gb"]
        if "ssd_cache_gb" in config:
            user_config.ssd_cache_gb = config["ssd_cache_gb"]
        if "ssd_cache_dir" in config:
            user_config.ssd_cache_dir = parse_path_list(config["ssd_cache_dir"])
        if "enable_gds" in config:
            user_config.enable_gds = config["enable_gds"]
        update_default_config_from_user_config(model_config, cache_config, user_config)
        return model_config, cache_config

if __name__ == "__main__":
    model_config, cache_config = load_config("./benchmarks/example_config.yml")
    print(model_config)
    print(cache_config)
