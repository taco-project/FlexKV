import time
from dataclasses import dataclass
from collections import deque

from flexkv.common.debug import flexkv_logger

logger = flexkv_logger


@dataclass
class FlexKVStats:
    num_log_interval_requests: int
    
    # get info
    num_get_requests: int = 0
    num_get_query_tokens: int = 0
    num_gpu_matched_tokens: int = 0
    num_flexkv_matched_tokens: int = 0
    
    # put info
    num_put_requests: int = 0
    num_put_query_tokens: int = 0
    num_put_unmatched_tokens: int = 0
    
    num_failed_requests: int = 0
        
    @property
    def tatal_num_requests(self) -> int:
        return self.num_get_requests + self.num_put_requests
    
    @property
    def get_gpu_match_ratio(self) -> float:
        if self.num_get_query_tokens == 0:
            return 0.0
        return self.num_gpu_matched_tokens / self.num_get_query_tokens
    
    @property
    def get_flexkv_match_ratio(self) -> float:
        if self.num_get_query_tokens == 0:
            return 0.0
        return self.num_flexkv_matched_tokens / self.num_get_query_tokens
    
    @property
    def get_put_token_ratio(self) -> float:
        if self.num_put_unmatched_tokens == 0:
            return 0.0
        return self.num_flexkv_matched_tokens / self.num_put_unmatched_tokens
    
    def record_get(
        self,
        num_prompt_tokens: int,
        num_gpu_matched_tokens: int,
        num_flexkv_matched_tokens: int,
    ):
        self.num_get_requests += 1
        self.num_get_query_tokens += num_prompt_tokens
        self.num_gpu_matched_tokens += num_gpu_matched_tokens
        self.num_flexkv_matched_tokens += num_flexkv_matched_tokens
        if self.num_get_requests == self.num_log_interval_requests:
            self.log()
            self.clear()
        
    def record_put(
        self,
        num_all_tokens: int,
        num_unmatched_tokens: int,
    ):
        self.num_put_requests += 1
        self.num_put_query_tokens += num_all_tokens
        self.num_put_unmatched_tokens += num_unmatched_tokens
        
    def record_faild(
        self,
        num_failed_requests: int
    ):
        self.num_failed_requests += num_failed_requests
        
    def clear(self):
        self.num_get_requests = 0
        self.num_get_query_tokens = 0
        self.num_gpu_matched_tokens = 0
        self.num_flexkv_matched_tokens = 0
        self.num_put_requests = 0
        self.num_put_query_tokens = 0
        self.num_put_unmatched_tokens = 0
        self.num_failed_requests = 0
      
    def log(self):
        if self.num_put_unmatched_tokens == 0:
            get_put_token_ratio_str = "Nan"
        else:
            get_put_token_ratio_str = f"{self.get_put_token_ratio*100:.2f}%"
        logger.info(
            f"[FlexKV] Metric of Recent {self.num_log_interval_requests} Requests: "
            f"Num Failed Request: {self.num_failed_requests}, "
            f"Num Get Query Tokens: {self.num_get_query_tokens}, "
            f"GPU Hit Ratio: {self.get_gpu_match_ratio*100:.2f}%, "
            f"FlexKV Hit Ratio: {self.get_flexkv_match_ratio*100:.2f}%, "
            f"Get/Put Token Ratio: {get_put_token_ratio_str}.")
        