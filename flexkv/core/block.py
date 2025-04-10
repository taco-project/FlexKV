from enum import Enum
from dataclasses import dataclass


class BlockStatus(Enum):
    AVAILABLE = 0
    LOCKED = 1
    INVALID = 2


class BlockLocation(Enum):
    CPU = 0  # CPU + SSD
    SSD = 1  # only SSD


@dataclass
class BlockMeta:
    block_hash: str = ""
    last_access_time: int = 0
    cpu_block_id: int = -1
    ssd_block_id: int = -1
    prev_block: "BlockMeta" = None
    reference_count: int = 0
    status: BlockStatus = BlockStatus.AVAILABLE
    location: BlockLocation = BlockLocation.CPU
