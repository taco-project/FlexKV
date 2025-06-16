from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from flexkv.common.config import ModelConfig
from flexkv.common.memory_handle import TensorSharedHandle


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


@dataclass
class PutRequest:
    dp_client_id: int
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    token_mask: Optional[torch.Tensor]


@dataclass
class GetRequest:
    dp_client_id: int
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    token_mask: Optional[torch.Tensor]


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
    masks: Optional[Dict[int, torch.Tensor]] = None
    success: bool = True
    error_msg: str = ""
