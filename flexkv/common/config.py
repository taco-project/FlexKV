from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union, Dict, Any

import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType


@dataclass
class ModelConfig:
    num_layers: int
    num_kv_heads: int
    head_size: int
    use_mla: bool = False
    dtype: torch.dtype = torch.float16
    tp_size: int = 1
    dp_size: int = 1

@dataclass
class CacheConfig:
    tokens_per_block: int = 16
    enable_cpu: bool = True
    enable_ssd: bool = False
    enable_remote: bool = False
    gpu_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.LAYERWISE
    cpu_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.LAYERWISE
    ssd_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.LAYERWISE
    remote_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.LAYERWISE
    use_gds: bool = False
    use_pinned_memory: bool = True
    # remote_cache_size_mode including file_size & block_num
    remote_cache_size_mode: str = "file_size"
    num_cpu_blocks: int = 1000000
    num_ssd_blocks: int = 10000000
    num_remote_blocks: Optional[int] = None
    remote_file_size: Optional[int] = None
    remote_file_num: Optional[int] = None
    remote_file_prefix: Optional[str] = None
    ssd_cache_dir: Optional[Union[str, List[str]]] = None
    ssd_cache_iouring_entries: int = 0
    ssd_cache_iouring_flags: int = 0
    remote_cache_path: Optional[Union[str, List[str]]] = None
    remote_config_custom: Optional[Dict[str, Any]] = None
