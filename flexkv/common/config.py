from dataclasses import dataclass
from typing import Optional, List, Union
from enum import Enum
import torch

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
    tokens_per_block: int = 16
    enable_cpu: bool = True
    enable_ssd: bool = False
    enable_remote: bool = False
    gpu_kv_layout: str = "layerwise"
    cpu_kv_layout: str = "layerwise"
    ssd_kv_layout: str = "layerwise"
    remote_kv_layout: str = "layerwise"
    use_gds: bool = False
    use_pinned_memory: bool = True
    num_cpu_blocks: int = 1000000
    num_ssd_blocks: int = 10000000
    num_remote_blocks: int = 10000000
    ssd_cache_path: Optional[Union[str, List[str]]] = None
