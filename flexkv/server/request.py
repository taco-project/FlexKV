from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from flexkv.common.config import ModelConfig
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout


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
class PutRequest:
    dp_client_id: int
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]


@dataclass
class GetRequest:
    dp_client_id: int
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]


@dataclass
class WaitRequest:
    dp_client_id: int
    tp_rank: Optional[int]
    wait_task_ids: List[int]

# Used for async put/get
@dataclass
class TryWaitRequest:
    dp_client_id: int
    tp_rank: Optional[int]
    try_wait_task_ids: List[int]


@dataclass
class Response:
    dp_client_id: int
    task_id: Optional[int] = None
    masks: Optional[Dict[int, np.ndarray]] = None
    success: bool = True
    error_msg: str = ""
