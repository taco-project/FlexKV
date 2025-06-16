from dataclasses import dataclass
from enum import Enum

import torch


class KVRequestType(Enum):
    GET = "get"
    PUT = "put"
    SHUTDOWN = "shutdown"

@dataclass
class KVRequest:
    request_type: KVRequestType
    request_id: int
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    slot_mapping: torch.Tensor
    layer_granularity: int = -1
    dp_id: int = 0
