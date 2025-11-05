import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any

import torch
from tqdm import tqdm

from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayoutType


@dataclass
class KVRequest:
    user_id: int
    turn_id: int
    request_type: str  # "get" or "put"
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    slot_mapping: Optional[torch.Tensor] = None

    request_id: int = field(init=False)
    _request_id_counter: int = field(init=False, default=0)

    def __post_init__(self):
        self.request_id = KVRequest._request_id_counter
        KVRequest._request_id_counter += 1


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
        config = json.load(f)
        if "ModelConfig" not in config:
            print("ModelConfig not found in config, using default values")
            config["ModelConfig"] = {}
        if "CacheConfig" not in config:
            print("CacheConfig not found in config, using default values")
            config["CacheConfig"] = {}
        if "dtype" in config["ModelConfig"]:
            config["ModelConfig"]["dtype"] = eval(f"torch.{config['ModelConfig']['dtype']}")
        model_config = ModelConfig(**config["ModelConfig"])
        cache_config = CacheConfig(**config["CacheConfig"])
        return model_config, cache_config
