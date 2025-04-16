from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

@dataclass
class ModelConfig:
    num_layers: int
    num_kv_heads: int
    head_size: int
    element_size: int
    use_mla: bool = False
    tp_size: int = 1

@dataclass
class CacheConfig:
    token_per_block: int = 16
    enable_cpu: bool = True
    enable_ssd: bool = False
    enable_remote: bool = False
    use_gds: bool = False
    use_pinned_memory: bool = True
    total_cpu_memory: int = 16 * 1024 * 1024 * 1024  # 16GB
    total_ssd_memory: int = 32 * 1024 * 1024 * 1024  # 32GB
    ssd_cache_path: Optional[str] = None
    eviction_policy: str = "LRU"
