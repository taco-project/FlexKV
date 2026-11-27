import time
from dataclasses import dataclass, field
from collections import Counter, deque

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

    # The counts recorded since take_counts() last ran, whatever the log
    # window above has cleared since.
    untaken_counts: Counter = field(default_factory=Counter)

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
        self._add(num_get_requests=1,
                  num_get_query_tokens=num_prompt_tokens,
                  num_gpu_matched_tokens=num_gpu_matched_tokens,
                  num_flexkv_matched_tokens=num_flexkv_matched_tokens)
        if self.num_get_requests == self.num_log_interval_requests:
            self.log()
            self.clear()
        
    def record_put(
        self,
        num_all_tokens: int,
        num_unmatched_tokens: int,
    ):
        self._add(num_put_requests=1,
                  num_put_query_tokens=num_all_tokens,
                  num_put_unmatched_tokens=num_unmatched_tokens)

    def record_faild(
        self,
        num_failed_requests: int
    ):
        self._add(num_failed_requests=num_failed_requests)

    def _add(self, **counts: int) -> None:
        for name, count in counts.items():
            setattr(self, name, getattr(self, name) + count)
        self.untaken_counts.update(counts)

    def take_counts(self) -> dict:
        """The counts recorded since the last call, each handed out once: vLLM
        sums the per-step reports it gets."""
        counts = dict(self.untaken_counts)
        self.untaken_counts.clear()
        return counts

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
        