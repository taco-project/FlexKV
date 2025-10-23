
from dataclasses import dataclass
from flexkv.common import request
from flexkv.common.debug import flexkv_logger
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.bindings.internal.args import TorchLlmArgs


logger = flexkv_logger

@dataclass
class RequestWrapper:
    _request: LlmRequest

    @property
    def req_id(self):
        logger.info(f"")
        return self._request.py_request_id
    
    @property
    def all_token_ids(self):
        all_token_ids = self._request.get_tokens()
        assert len(all_token_ids) == 1, "Don't support beam search."
        return all_token_ids[0]
    
    @property
    def num_tokens(self):
        return len(self.all_token_ids)

    @property
    def num_prompt_tokens(self):
        return self._request.py_prompt_len

    @property
    def num_new_matched_tokens(self):
        return self._request.num_new_matched_tokens

    def is_finished(self):
        return self._request.is_finished
    
    def is_finished_normal(self):
        # NORMAL = 0
        # ABNORMAL = 3
        
        # if self._request.is_finished_normal():        
        #     return NORMAL
        # else:
        #     return ABNORMAL
        return self._request.is_finished_normal()

    def __post_init__(self):
        logger.info(f"{self._request.py_prompt_len=}\n"
                    f"{self._request.py_request_id=}\n"
                    f"{self._request.all_token_ids=}\n")

def get_dp_tp_info(llm_args: TorchLlmArgs):
    if llm_args.enable_attention_dp:
        # trt 也不支持同时开 tp+dp
        tp_size = 1
        dp_size = llm_args.tensor_parallel_size
        dp_rank = llm_args.parallel_config.to_mapping().tp_rank
    else:
        tp_size = llm_args.tensor_parallel_size
        dp_size = 1 
        dp_rank = 0
    return tp_size, dp_size, dp_rank