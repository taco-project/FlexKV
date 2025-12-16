
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from flexkv.common import request
from flexkv.common.debug import flexkv_logger
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.bindings.executor import ExecutorConfig


logger = flexkv_logger

@dataclass
class RequestWrapper:
    _request: LlmRequest

    @property
    def req_id(self):
        return self._request.request_id
    
    @property
    def all_token_ids(self):
        all_token_ids = self._request.get_tokens()
        assert len(all_token_ids) == 1, "Don't support beam search."
        return all_token_ids[0]

    @property
    def np_token_ids(self) -> np.ndarray:
        """
        Cache the list->numpy conversion once and reuse it to avoid repeated copying in hot path.
        """
        cached: Optional[np.ndarray] = getattr(self, "_np_token_ids_cache", None)
        if cached is not None:
            return cached

        tokens = self.all_token_ids
        if isinstance(tokens, np.ndarray):
            cached = tokens
        else:
            cached = np.asarray(tokens, dtype=np.int64)
        setattr(self, "_np_token_ids_cache", cached)
        return cached
    
    @property
    def num_tokens(self):
        return len(self.all_token_ids)

    @property
    def num_prompt_tokens(self):
        return self._request.prompt_len

    @property
    def num_new_matched_tokens(self):
        return self._request.num_connector_matched_tokens
        # return self._request.local_prepopulated_prompt_len

    def is_finished(self):
        return self._request.is_finished
    
    def is_finished_normal(self):
        # NORMAL = 0
        # ABNORMAL = 3
        
        # if self._request.is_finished_normal():        
        #     return NORMAL
        # else:
        #     return ABNORMAL
        return self._request.is_finished_normal

    def __repr__(self):
        return f"RequestWrapper(req_id={self.req_id}, " \
                f"num_prompt_tokens={self.num_prompt_tokens}, " \
                f"num_tokens={len(self.all_token_ids)}, " \
                f"num_new_matched_tokens={self.num_new_matched_tokens})"


def get_mapping_from_config(config):
    if hasattr(config, "mapping"):
        return config.mapping
    elif hasattr(config, "parallel_config"):
        return config.parallel_config.to_mapping()
    else:
        raise ValueError("No mapping found in config")

def get_tokens_per_block_from_config(config):
    DEFAULT_TOKENS_PER_BLOCK = 64
    if hasattr(config, "tokens_per_block"):
        return int(config.tokens_per_block)
    elif hasattr(config, "kv_cache_config"):
        return int(config.kv_cache_config.tokens_per_block)
    else:
        return DEFAULT_TOKENS_PER_BLOCK

def get_kv_cache_dtype_from_config(config) -> torch.dtype:
    DEFAULT_DTYPE = torch.bfloat16
    def _to_torch_dtype(dtype_val) -> torch.dtype:
        if isinstance(dtype_val, torch.dtype):
            return dtype_val
        if dtype_val is None:
            return DEFAULT_DTYPE

        s = dtype_val.lower()
        if s in ("float16", "fp16", "half"):
            return torch.float16
        if s in ("bfloat16", "bf16"):
            return torch.bfloat16
        if s in ("float32", "fp32"):
            return torch.float32

        return DEFAULT_DTYPE

    if hasattr(config, "pytorch_backend_config"):
        return _to_torch_dtype(config.pytorch_backend_config.kv_cache_dtype)
    elif hasattr(config, "kv_cache_config"):
        return _to_torch_dtype(config.kv_cache_config.dtype) 
    else:
        return DEFAULT_DTYPE

def get_model_path_from_config(config):
    if hasattr(config, "hf_model_dir"):
        return config.hf_model_dir
    elif hasattr(config, "model"):
        return config.model
    else:
        raise ValueError("No model path found in config")