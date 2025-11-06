from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union, Dict, Any
from argparse import Namespace
import os
import copy

import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType


@dataclass
class ModelConfig:
    num_layers: int = 0
    num_kv_heads: int = 0
    head_size: int = 0
    use_mla: bool = False
    dtype: torch.dtype = torch.bfloat16

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
    server_recv_port=os.getenv('FLEXKV_SERVER_RECV_PORT', 'ipc:///tmp/flexkv_server'),

    index_accel=bool(int(os.getenv('FLEXKV_INDEX_ACCEL', 1))),
    cpu_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_CPU_LAYOUT', 'BLOCKWISE').upper()),
    ssd_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_SSD_LAYOUT', 'BLOCKWISE').upper()),
    remote_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_REMOTE_LAYOUT', 'BLOCKWISE').upper()),
    gds_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_GDS_LAYOUT', 'BLOCKWISE').upper()),

    use_ce_transfer_h2d=bool(int(os.getenv('FLEXKV_USE_CE_TRANSFER_H2D', 0))),
    use_ce_transfer_d2h=bool(int(os.getenv('FLEXKV_USE_CE_TRANSFER_D2H', 0))),
    transfer_sms_h2d=int(os.getenv('FLEXKV_TRANSFER_SMS_H2D', 8)),
    transfer_sms_d2h=int(os.getenv('FLEXKV_TRANSFER_SMS_D2H', 8)),

    ssd_cache_iouring_entries=int(os.getenv('FLEXKV_SSD_CACHE_IORING_ENTRIES', 512)),
    ssd_cache_iouring_flags=int(os.getenv('FLEXKV_SSD_CACHE_IORING_FLAGS', 1)),

    max_blocks_per_file=int(os.getenv('FLEXKV_MAX_BLOCKS_PER_FILE', 32000)),  # -1 means no limit

    evict_ratio=float(os.getenv('FLEXKV_EVICT_RATIO', 0.05)),
    hit_reward_seconds=int(os.getenv('FLEXKV_HIT_REWARD_SECONDS', 0)),

    enable_trace=bool(int(os.getenv('FLEXKV_ENABLE_TRACE', 0))),
    trace_file_path=os.getenv('FLEXKV_TRACE_FILE_PATH', './flexkv_trace.log'),
    trace_max_file_size_mb=int(os.getenv('FLEXKV_TRACE_MAX_FILE_SIZE_MB', 100)),
    trace_max_files=int(os.getenv('FLEXKV_TRACE_MAX_FILES', 5)),
    trace_flush_interval_ms=int(os.getenv('FLEXKV_TRACE_FLUSH_INTERVAL_MS', 1000)),

    num_log_interval_requests=int(os.getenv('FLEXKV_NUM_LOG_INTERVAL_REQUESTS', 200)),
)

@dataclass
class UserConfig:
    cpu_cache_gb: int = 16
    ssd_cache_gb: int = 0  # 0 means disable ssd
    ssd_cache_dir: Union[str, List[str]] = "./ssd_cache"
    enable_gds: bool = False

    def __post_init__(self):
        if self.cpu_cache_gb <= 0:
            raise ValueError(f"Invalid cpu_cache_gb: {self.cpu_cache_gb}")
        if self.ssd_cache_gb < 0:
            raise ValueError(f"Invalid ssd_cache_gb: {self.ssd_cache_gb}")
        if self.ssd_cache_gb > 0 and self.ssd_cache_gb <= self.cpu_cache_gb:
            raise ValueError(f"Invalid ssd_cache_gb: {self.ssd_cache_gb}, "
                             f"must be greater than cpu_cache_gb: {self.cpu_cache_gb}.")

def parse_path_list(path_str: str) -> List[str]:
    paths = [p.strip() for p in path_str.split(';') if p.strip()]
    return paths

def load_user_config_from_file(config_file: str) -> UserConfig:
    import json
    import yaml
    # read json config file or yaml config file
    if config_file.endswith('.json'):
        with open(config_file) as f:
            config = json.load(f)
    elif config_file.endswith('.yaml'):
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file extension: {config_file}")
    if 'ssd_cache_dir' in config:
        config['ssd_cache_dir'] = parse_path_list(config['ssd_cache_dir'])
    return UserConfig(**config)

def load_user_config_from_env() -> UserConfig:
    return UserConfig(
        cpu_cache_gb=int(os.getenv('FLEXKV_CPU_CACHE_GB', 16)),
        ssd_cache_gb=int(os.getenv('FLEXKV_SSD_CACHE_GB', 256)),
        ssd_cache_dir=parse_path_list(os.getenv('FLEXKV_SSD_CACHE_DIR', "./ssd_cache")),
        enable_gds=bool(int(os.getenv('FLEXKV_ENABLE_GDS', 0))),
    )

def convert_to_block_num(size_in_GB: float, block_size_in_bytes: int) -> int:
    return int(size_in_GB * 1024 * 1024 * 1024 / block_size_in_bytes)

def update_default_config_from_user_config(model_config: ModelConfig,
                                           cache_config: CacheConfig,
                                           user_config: UserConfig) -> None:
    block_size_in_bytes = model_config.token_size_in_bytes * cache_config.tokens_per_block

    assert user_config.cpu_cache_gb > 0
    assert user_config.ssd_cache_gb >= 0

    cache_config.num_cpu_blocks = convert_to_block_num(user_config.cpu_cache_gb, block_size_in_bytes)
    cache_config.num_ssd_blocks = convert_to_block_num(user_config.ssd_cache_gb, block_size_in_bytes)
    cache_config.ssd_cache_dir = user_config.ssd_cache_dir
    cache_config.enable_ssd = user_config.ssd_cache_gb > 0
    cache_config.enable_gds = user_config.enable_gds
