from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

import torch
import numpy as np


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


class KVResponseStatus(Enum):
    SUCCESS = "success"
    NOTFOUND = "not_found"
    UNREADY = "unready"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class KVResponse:
    status: KVResponseStatus
    task_id: int
    return_mask: Optional[np.ndarray]

    def get_mask(self) -> torch.Tensor:
        return torch.from_numpy(self.return_mask) if self.return_mask is not None else None
