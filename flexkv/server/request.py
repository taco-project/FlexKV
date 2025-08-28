from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from flexkv.common.config import ModelConfig
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout
from flexkv.common.request import KVResponseStatus


@dataclass
class RegisterDPClientRequest:
    model_config: ModelConfig
    client_recv_port: str


@dataclass
class RegisterTPClientRequest:
    dp_client_id: int
    tp_rank: int
    device_id: int
    client_recv_port: str
    handles: List[TensorSharedHandle]
    gpu_layout: KVCacheLayout

@dataclass
class IsReadyRequest:
    dp_client_id: int

@dataclass
class PutRequest:
    dp_client_id: int
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]
    task_id: int = -1


@dataclass
class GetRequest:
    dp_client_id: int
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]
    task_id: int = -1

@dataclass
class PutMatchRequest:
    dp_client_id: int
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]
    task_id: int = -1

@dataclass
class GetMatchRequest:
    dp_client_id: int
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]
    task_id: int = -1

@dataclass
class LaunchTaskRequest:
    dp_client_id: int
    task_ids: List[int]

@dataclass
class CancelTaskRequest:
    dp_client_id: int
    task_ids: List[int]

@dataclass
class WaitRequest:
    dp_client_id: int
    tp_rank: Optional[int]
    wait_task_ids: List[int]
    wait_timeout: float = 20.0
    completely: bool = False

# Used for async put/get
@dataclass
class TryWaitRequest:
    dp_client_id: int
    tp_rank: Optional[int]
    try_wait_task_ids: List[int]


@dataclass
class Response:
    dp_client_id: int = -1
    task_id: Optional[int] = None
    mask: Optional[Dict[int, np.ndarray]] = None
    status: Optional[Dict[int, KVResponseStatus]] = None
    is_ready: bool = False
    error_msg: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status is not None and \
               all(self.status[task_id] == KVResponseStatus.SUCCESS for task_id in self.status.keys())


@dataclass
class ShutdownRequest:
    dp_client_id: int


@dataclass
class CheckRunningRequest:
    dp_client_id: int