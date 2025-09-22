import os
import json
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
    enable_remote: bool = False # used for indicating whether the 3rd-party remote storage is enabled
                                # has nothing to do with whether the p2p_cpu and p2p_ssd are supported
    enable_kv_sharing: bool = False # pcfs_sharing or p2p_cpu or p2p_ssd or p2p_3rd_remote
    enable_p2p_cpu: bool = False
    enable_p2p_ssd: bool = False
    enable_3rd_remote: bool = False
    use_gds: bool = False
    index_accel: bool = False # have to be True when (enable_p2p_cpu or enable_p2p_ssd) is True
    distributed_node_id: int = -1 # only used when distributed cpu/ssd and only can be set when redis_meta_client initialized

    # kv cache layout configs
    gpu_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.LAYERWISE
    cpu_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE
    ssd_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE
    remote_kv_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKWISE

    # mempool capacity configs
    num_cpu_blocks: int = 1000000
    num_ssd_blocks: int = 10000000
    num_remote_blocks: Optional[int] = None
    num_local_blocks: int = 1000000

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
    refresh_batch_size: int = 128
    rebuild_interval_ms: int = 10000
    idle_sleep_ms: int = 10
    lease_ttl_ms: int = 100000
    renew_lease_ms: int = 0

    # distributed zmq configs
    local_zmq_ip: str = "127.0.0.1"
    local_zmq_port: int = 5555

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
        
        self.enable_kv_sharing = self.enable_p2p_cpu or \
            self.enable_p2p_ssd or self.enable_3rd_remote
        self.enable_remote = self.enable_3rd_remote
        self.index_accel = self.enable_p2p_cpu or self.enable_p2p_ssd or self.index_accel
        
@dataclass
class MooncakeTransferEngineConfig:
    engine_ip: str
    engine_port: int
    metadata_backend: Union[str, None]
    metadata_server: str
    metadata_server_auth: str
    protocol: str
    device_name: str
    # redis_server: str
    # redis_db: int
    # redis_auth: str


    @staticmethod
    def from_file(file_path: str) -> "MooncakeTransferEngineConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig.from_dict(config)


    @staticmethod
    def load_from_env(env_name: str) -> "MooncakeTransferEngineConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv(env_name)
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeTransferEngineConfig.from_file(config_file_path)


    @staticmethod
    def from_dict(config: dict) -> "MooncakeTransferEngineConfig":
        """Load the config from a JSON file."""
        return MooncakeTransferEngineConfig(
            engine_ip=config.get("engine_ip", "127.0.0.1"),
            engine_port=config.get("engine_port", 5555),
            metadata_backend=config.get("metadata_backend", "redis"),
            metadata_server=config.get("metadata_server", "redis://127.0.0.1:6380"),
            metadata_server_auth=config.get("metadata_server_auth", "yourpass"),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
            # redis_server=config.get("redis_server", "redis://127.0.0.1:6379"),
            # redis_db=config.get("redis_db", 0),
            # redis_auth=config.get("redis_auth", "yourpass"),
        )