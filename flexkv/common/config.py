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
    dtype: torch.dtype = torch.bfloat16
    max_req_tokens = 163840

    # parallel configs
    tp_size: int = 1
    dp_size: int = 1

    @property
    def token_size_in_bytes(self) -> int:
        kv_dim = 1 if self.use_mla else 2
        return self.num_layers * self.num_kv_heads * self.head_size * kv_dim * self.dtype.itemsize

@dataclass
class CacheConfig:
    tokens_per_block: int = 16
    enable_cpu: bool = True
    enable_ssd: bool = False
    enable_remote: bool = False
    enable_kv_sharing: bool = False
    use_gds: bool = False
    index_accel: bool = False

    # kv cache layout configs
    gpu_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.LAYERWISE
    cpu_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE
    ssd_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE
    remote_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE

    # mempool capacity configs
    num_cpu_blocks: int = 1000000
    num_ssd_blocks: int = 10000000
    num_remote_blocks: Optional[int] = None

    # CPU-GPU transfer configs
    use_ce_transfer_h2d: bool = False
    use_ce_transfer_d2h: bool = False
    transfer_sms_h2d: int = 8
    transfer_sms_d2h: int = 8

    # ssd cache configs
    max_blocks_per_file: int = 32000  # -1 means no limit
    ssd_cache_dir: Optional[Union[str, List[str]]] = None
    ssd_cache_iouring_entries: int = 0
    ssd_cache_iouring_flags: int = 0

    # remote cache configs
    remote_cache_size_mode: str = "file_size"  # file_size or block_num
    remote_file_size: Optional[int] = None
    remote_file_num: Optional[int] = None
    remote_file_prefix: Optional[str] = None
    remote_cache_path: Optional[Union[str, List[str]]] = None
    remote_config_custom: Optional[Dict[str, Any]] = None

    # KV sharing / distributed radix tree tunables
    lt_pool_initial_capacity: int = 10000000
    refresh_batch_size: int = 128
    rebuild_interval_ms: int = 1000
    idle_sleep_ms: int = 10
    lease_ttl_ms: int = 100000
    renew_lease_ms: int = 0

    # Redis configs (for KV sharing / metadata)
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    local_ip: str = "127.0.0.1"
    redis_password: Optional[str] = None

    # Trace configs
    enable_trace: bool = True
    trace_file_path: str = "./flexkv_trace.log"
    trace_max_file_size_mb: int = 100
    trace_max_files: int = 5
    trace_flush_interval_ms: int = 1000

    #evict ratio
    evict_ratio: float = 0.0

    def __post_init__(self):
        layout_fields = ['gpu_kv_layout_type', 
                         'cpu_kv_layout_type', 
                         'ssd_kv_layout_type', 
                         'remote_kv_layout_type']
        for field in layout_fields:
            value = getattr(self, field)
            if isinstance(value, str):
                setattr(self, field, KVCacheLayoutType[value.upper()])
        
