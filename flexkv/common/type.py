from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class MatchResultAccel:
    num_ready_matched_blocks: int = 0
    num_matched_blocks: int = 0
    last_ready_node: Optional['CRadixNode'] = None
    last_node: Optional['CRadixNode'] = None
    last_node_matched_length: int = 0
    physical_blocks: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    block_node_ids: Optional[np.ndarray] = None
    matched_pos: Optional[str] = None
    matched_node_ids: Optional[np.ndarray] = None #TODO id or ids? should we allow one req match results on multiple nodes?
    insert_to_local_cpu_index: bool = True

    def __post_init__(self) -> None:
        assert self.physical_blocks.ndim == 1


