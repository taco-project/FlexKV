
from dataclasses import dataclass
from typing import Optional

import numpy as np
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