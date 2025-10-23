import numpy as np

from typing import TYPE_CHECKING, Optional, Literal, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from flexkv.integration.tensorrt_llm.utils import RequestWrapper

@dataclass
class FlexKVConnectorMetadata:
    finished_sending: List[str]
    finished_recving: List[str]

@dataclass
class FlexKVResponse:
    task_id: int
    task_type: Literal["get", "put"]
    request: RequestWrapper
    success: bool


@dataclass
class FlexKVTask(ABC):
    task_id: int = 0
    request: RequestWrapper = 0
    
    # slot mapping
    slot_mapping: Optional[np.ndarray] = None
    
    # timer
    match_start_time: float = 0
    match_end_time: float = 0
    task_launch_time: float = 0
    task_finished_time: float = 0
    
    @property
    def match_cost(self) -> float:
        return (self.match_end_time - self.match_start_time)
    
    @property
    def task_execute_cost(self) -> float:
        return (self.task_finished_time - self.task_launch_time)
    
    @property
    @abstractmethod
    def task_type(self) -> str:
        ...
    
    def __str__(self):
        return (f"FlexKVTask(task_id={self.task_id}, "
                f"request={self.request.request_id}, "
                f"match_cost {self.match_cost*1000:.2f} ms, "
                f"task execute cost {self.task_execute_cost*1000:.2f} ms)")


@dataclass(kw_only=True)
class FlexKVGetTask(FlexKVTask):
    num_computed_tokens: int
    num_new_matched_tokens: int
    
    @property
    def task_type(self) -> str:
        return "get"
    
    def __str__(self):
        return (f"FlexKVGetTask(task_id={self.task_id}, "
                f"request={self.request.request_id}, "
                f"num_computed_tokens={self.num_computed_tokens}, "
                f"num_new_matched_tokens={self.num_new_matched_tokens}, "
                f"match_cost {self.match_cost*1000:.2f} ms, "
                f"task execute cost {self.task_execute_cost*1000:.2f} ms)")

 
@dataclass(kw_only=True)
class FlexKVPutTask(FlexKVTask):
    num_matched_tokens: int
    num_unmatched_tokens: int
    
    @property
    def task_type(self) -> str:
        return "put"
    
    def __str__(self):
        return (f"FlexKVPutTask(task_id={self.task_id}, "
                f"request={self.request.request_id}, "
                f"num_matched_tokens={self.num_matched_tokens}, "
                f"num_unmatched_tokens={self.num_unmatched_tokens}, "
                f"match_cost {self.match_cost*1000:.2f} ms, "
                f"task execute cost {self.task_execute_cost*1000:.2f} ms)")
