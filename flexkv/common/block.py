from enum import Enum, auto
from dataclasses import dataclass

import torch

class BlockStatus(Enum):
    UNREGISTERED = auto()
    AVAILABLE = auto()
    LOCKED = auto()
    IN_GET = auto()
    IN_PUT = auto()

@dataclass
class BlockMeta:
    hash: str = ""
    token_ids: torch.Tensor = torch.tensor([])
    last_access_time: int = 0
    reference_count: int = 0
    status: BlockStatus = BlockStatus.UNREGISTERED
