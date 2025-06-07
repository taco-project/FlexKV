
from enum import Enum
from dataclasses import dataclass
from typing import Dict

import torch

from flexkv.common.transfer import DeviceType

class cacheEngineRequestType(Enum):
    GET = "get"
    PUT = "put"

@dataclass
class cacheEngineRequest:
    request_type: cacheEngineRequestType = None
    request_id: int = None
    token_ids: torch.Tensor = None
    token_mask: torch.Tensor = None
    slot_mapping: torch.Tensor = None
    block_ids_to_unlock: Dict[DeviceType, torch.Tensor] = None
    layer_granularity: int = -1
    dp_id: int = 0