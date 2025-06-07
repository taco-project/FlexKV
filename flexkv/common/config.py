from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from enum import Enum
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
import torch

@dataclass
class ModelConfig:
    num_layers: int
    num_kv_heads: int
    head_size: int
    element_size: int
    use_mla: bool = False
    tp_size: int = 1
    dp_size: int = 1

default_kv_layout = KVCacheLayout(type=KVCacheLayoutType.LAYERWISE, num_layer=1, num_block=1, tokens_per_block=1, num_head=1, head_size=1)

@dataclass
class CacheConfig:
    tokens_per_block: int = 16
    raw_gpu_blocks: bool = True
    enable_cpu: bool = True
    enable_ssd: bool = False
    enable_remote: bool = False
    gpu_kv_layout: KVCacheLayout = default_kv_layout
    cpu_kv_layout: KVCacheLayout = default_kv_layout
    ssd_kv_layout: KVCacheLayout = default_kv_layout
    remote_kv_layout: KVCacheLayout = default_kv_layout
    use_gds: bool = False
    use_pinned_memory: bool = True
    num_cpu_blocks: int = 1000000
    num_ssd_blocks: int = 10000000
    num_remote_blocks: int = 10000000
    ssd_cache_path: Optional[Union[str, List[str]]] = None
    remote_cache_path: Optional[Union[str, List[str]]] = None
    remote_config_custom: Optional[Dict[str, Any]] = None
