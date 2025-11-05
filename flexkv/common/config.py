from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union, Dict, Any
from argparse import Namespace
import os

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
    enable_gds: bool = False

    # mempool capacity configs
    num_cpu_blocks: int = 1000000
    num_ssd_blocks: int = 10000000
    num_gds_blocks: int = 10000000
    num_remote_blocks: Optional[int] = None

    # ssd cache configs
    ssd_cache_dir: Optional[Union[str, List[str]]] = None

    # gds cache configs
    gds_cache_dir: Optional[Union[str, List[str]]] = None

    # remote cache configs for cfs
    remote_cache_size_mode: str = "file_size"  # file_size or block_num
    remote_file_size: Optional[int] = None
    remote_file_num: Optional[int] = None
    remote_file_prefix: Optional[str] = None
    remote_cache_path: Optional[Union[str, List[str]]] = None
    remote_config_custom: Optional[Dict[str, Any]] = None


GLOBAL_CONFIG_FROM_ENV: Namespace = Namespace(
    server_client_mode=bool(int(os.getenv('FLEXKV_SERVER_CLIENT_MODE', 0))),
    server_recv_port=os.getenv('FLEXKV_SERVER_RECV_PORT', 'flexkv_server'),

    index_accel=bool(int(os.getenv('FLEXKV_INDEX_ACCEL', 1))),
    cpu_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_CPU_LAYOUT', 'BLOCKWISE').upper()),
    ssd_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_SSD_LAYOUT', 'BLOCKWISE').upper()),
    remote_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_REMOTE_LAYOUT', 'BLOCKWISE').upper()),
    gds_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_GDS_LAYOUT', 'BLOCKWISE').upper()),

    use_ce_transfer_h2d=os.getenv('FLEXKV_USE_CE_TRANSFER_H2D', 'False').lower() == 'true',
    use_ce_transfer_d2h=os.getenv('FLEXKV_USE_CE_TRANSFER_D2H', 'False').lower() == 'true',
    transfer_sms_h2d=int(os.getenv('FLEXKV_TRANSFER_SMS_H2D', 8)),
    transfer_sms_d2h=int(os.getenv('FLEXKV_TRANSFER_SMS_D2H', 8)),

    ssd_cache_iouring_entries=int(os.getenv('FLEXKV_SSD_CACHE_IORING_ENTRIES', 512)),
    ssd_cache_iouring_flags=int(os.getenv('FLEXKV_SSD_CACHE_IORING_FLAGS', 1)),

    max_blocks_per_file=int(os.getenv('FLEXKV_MAX_BLOCKS_PER_FILE', 32000)),  # -1 means no limit

    evict_ratio=float(os.getenv('FLEXKV_EVICT_RATIO', 0.05)),
    hit_reward_seconds=int(os.getenv('FLEXKV_HIT_REWARD_SECONDS', 0)),

    enable_trace=os.getenv('FLEXKV_ENABLE_TRACE', 'False').lower() == 'true',
    trace_file_path=os.getenv('FLEXKV_TRACE_FILE_PATH', './flexkv_trace.log'),
    trace_max_file_size_mb=int(os.getenv('FLEXKV_TRACE_MAX_FILE_SIZE_MB', 100)),
    trace_max_files=int(os.getenv('FLEXKV_TRACE_MAX_FILES', 5)),
    trace_flush_interval_ms=int(os.getenv('FLEXKV_TRACE_FLUSH_INTERVAL_MS', 1000)),
)

def convert_to_block_num(size_in_GB: float, block_size_in_bytes: int) -> int:
    return int(size_in_GB * 1024 * 1024 * 1024 / block_size_in_bytes)
