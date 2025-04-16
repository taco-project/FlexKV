from enum import Enum, auto
from dataclasses import dataclass, field
from typing import NewType, Optional, List
import time
import torch

from flexkv.common.hash_utils import \
    HashType, hash_ctx_init, hash_ctx_update, hash_ctx_finalize

class BlockStatus(Enum):
    UNREGISTERED = auto()
    AVAILABLE = auto()
    LOCKED = auto()
    IN_GET = auto()
    IN_PUT = auto()

@dataclass
class BlockMeta:

    hash: HashType

    prev_hash: Optional[HashType]

    physical_block_id: int = -1

    last_access_time: float = 0

    lock_cnt: int = 0

    child_cnt: int = 0

    status: BlockStatus = BlockStatus.UNREGISTERED

    def __lt__(self, other: 'BlockMeta') -> bool:
        return self.last_access_time < other.last_access_time

    def __str__(self):
        return (
            f"BlockMeta(hash={self.hash}, "
            f"prev_hash={self.prev_hash if self.prev_hash else None}, "
            f"physical_block_id={self.physical_block_id}, "
            f"last_access_time={self.last_access_time}, "
            f"lock_cnt={self.lock_cnt}, "
            f"child_cnt={self.child_cnt}, "
            f"status={self.status})"
        )

    def __repr__(self):
        return self.__str__()

@dataclass
class SequenceMeta:

    token_ids: torch.Tensor

    tokens_per_block: int

    block_hashes: List[HashType] = field(default_factory=list)

    def __post_init__(self):
        assert self.token_ids.ndim == 1
        assert self.tokens_per_block > 0

    @property
    def num_blocks(self) -> int:
        return len(self.token_ids) // self.tokens_per_block

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def has_hashes(self) -> bool:
        return len(self.block_hashes) > 0

    def gen_hashes(self) -> None:
        if self.has_hashes:
            return
        hash_ctx = hash_ctx_init()
        for i in range(self.num_blocks):
            hash_ctx_update(hash_ctx,
                            self.token_ids[i*self.tokens_per_block:(i+1)*self.tokens_per_block])
            self.block_hashes.append(hash_ctx_finalize(hash_ctx))
