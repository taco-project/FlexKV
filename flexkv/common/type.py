from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class MatchResultAccel:
    num_ready_matched_blocks: int = 0
    num_matched_blocks: int = 0
    # Mooncake-store only: main KV longest prefix (may exceed num_matched_blocks
    # when SWA joint hit is shorter). PUT uses this to skip existing KV keys.
    kv_matched_blocks: int = 0
    last_ready_node: Optional['CRadixNode'] = None
    last_node: Optional['CRadixNode'] = None
    last_node_matched_length: int = 0
    physical_blocks: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    block_node_ids: Optional[np.ndarray] = None
    matched_pos: Optional[str] = None
    matched_node_ids: Optional[np.ndarray] = None #TODO id or ids? should we allow one req match results on multiple nodes?
    insert_to_local_cpu_index: bool = True
    # ===== SWA (node-mounted) — passed through from CMatchResult so GET can
    # select the SWA source from the same forward match instead of re-walking
    # the tree. The deepest fully-matched ready node carrying a live SWA slot,
    # and the ready-prefix block count ending at it. None / 0 when no SWA on
    # the matched path.
    last_swa_node: Optional['CRadixNode'] = None
    swa_hit_blocks: int = 0

    def __post_init__(self) -> None:
        assert self.physical_blocks.ndim == 1


