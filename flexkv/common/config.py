import os
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union, Dict, Any
from argparse import Namespace
import os
import copy

import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.debug import flexkv_logger

@dataclass
class ModelConfig:
    num_layers: int = 1
    num_kv_heads: int = 1
    head_size: int = 1
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
    enable_gds: bool = False # Requires enable_ssd=True
    enable_remote: bool = False # used for indicating whether the 3rd-party remote storage is enabled
                                # has nothing to do with whether the p2p_cpu and p2p_ssd are supported
    enable_kv_sharing: bool = False # pcfs_sharing or p2p_cpu or p2p_ssd or p2p_3rd_remote
    enable_p2p_cpu: bool = False
    enable_p2p_ssd: bool = False
    enable_3rd_remote: bool = False

    distributed_node_id: int = -1 # only used when distributed cpu/ssd and only can be set when redis_meta_client initialized
    num_tmp_cpu_blocks: int = 500 # only used when distributed ssd p2p, it controls the number blocks of temp cpu buffer which used for copy data from ssd to cpu


    # mempool capacity configs
    num_cpu_blocks: int = 1000000
    num_ssd_blocks: int = 10000000
    num_remote_blocks: Optional[int] = None
    num_local_blocks: int = 1000000

    # ssd cache configs
    ssd_cache_dir: Optional[Union[str, List[str]]] = None

    # remote cache configs for cfs
    # todo: remove this in the future
    remote_cache_size_mode: str = "file_size"  # file_size or block_num
    remote_file_size: Optional[int] = None
    remote_file_num: Optional[int] = None
    remote_file_prefix: Optional[str] = None
    remote_cache_path: Optional[Union[str, List[str]]] = None
    remote_config_custom: Optional[Dict[str, Any]] = None

    # distributed zmq configs
    local_zmq_ip: str = "127.0.0.1"
    local_zmq_port: int = 5555
    # Redis configs (for KV sharing / metadata)
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    local_ip: str = "127.0.0.1"
    redis_password: Optional[str] = None

    def __post_init__(self):
        self.enable_kv_sharing = self.enable_p2p_cpu or \
            self.enable_p2p_ssd or self.enable_3rd_remote
        self.enable_remote = self.enable_3rd_remote

GLOBAL_CONFIG_FROM_ENV: Namespace = Namespace(
    # Multi-instance configuration
    instance_num=int(os.getenv('FLEXKV_INSTANCE_NUM', 1)),
    instance_id=int(os.getenv('FLEXKV_INSTANCE_ID', 0)),
    
    # Server-client mode configuration
    server_client_mode=bool(int(os.getenv('FLEXKV_SERVER_CLIENT_MODE', 0))),
    server_recv_port=os.getenv('FLEXKV_SERVER_RECV_PORT', 'ipc:///tmp/flexkv_server'),

    index_accel=bool(int(os.getenv('FLEXKV_INDEX_ACCEL', 1))),
    cpu_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_CPU_LAYOUT', 'BLOCKFIRST').upper()),
    ssd_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_SSD_LAYOUT', 'BLOCKFIRST').upper()),
    remote_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_REMOTE_LAYOUT', 'BLOCKFIRST').upper()),
    gds_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_GDS_LAYOUT', 'BLOCKFIRST').upper()),

    enable_layerwise_transfer=bool(int(os.getenv('FLEXKV_ENABLE_LAYERWISE_TRANSFER', 0))),

    use_ce_transfer_h2d=bool(int(os.getenv('FLEXKV_USE_CE_TRANSFER_H2D', 0))),
    use_ce_transfer_d2h=bool(int(os.getenv('FLEXKV_USE_CE_TRANSFER_D2H', 0))),
    h2d_cta_num=int(os.getenv('FLEXKV_H2D_CTA_NUM', 4)),
    d2h_cta_num=int(os.getenv('FLEXKV_D2H_CTA_NUM', 4)),

    iouring_entries=int(os.getenv('FLEXKV_IOURING_ENTRIES', 512)),
    iouring_flags=int(os.getenv('FLEXKV_IOURING_FLAGS', 0)),

    max_file_size_gb=float(os.getenv('FLEXKV_MAX_FILE_SIZE_GB', -1)),  # -1 means no limit

    evict_ratio=float(os.getenv('FLEXKV_EVICT_RATIO', 0.1)),
    evict_start_threshold=float(os.getenv('FLEXKV_EVICT_START_THRESHOLD', 0.7)),
    hit_reward_seconds=int(os.getenv('FLEXKV_HIT_REWARD_SECONDS', 0)),

    enable_trace=bool(int(os.getenv('FLEXKV_ENABLE_TRACE', 0))),
    trace_file_path=os.getenv('FLEXKV_TRACE_FILE_PATH', './flexkv_trace.log'),
    trace_max_file_size_mb=int(os.getenv('FLEXKV_TRACE_MAX_FILE_SIZE_MB', 100)),
    trace_max_files=int(os.getenv('FLEXKV_TRACE_MAX_FILES', 5)),
    trace_flush_interval_ms=int(os.getenv('FLEXKV_TRACE_FLUSH_INTERVAL_MS', 1000)),

    lt_pool_initial_capacity=int(os.getenv('FLEXKV_LT_POOL_INITIAL_CAPACITY', 10000000)),
    refresh_batch_size=int(os.getenv('FLEXKV_REFRESH_BATCH_SIZE', 256)),
    rebuild_interval_ms=int(os.getenv('FLEXKV_REBUILD_INTERVAL_MS', 10000)),
    idle_sleep_ms=int(os.getenv('FLEXKV_IDLE_SLEEP_MS', 10)),
    lease_ttl_ms=int(os.getenv('FLEXKV_LEASE_TTL_MS', 30000)),
    safety_ttl_ms=int(os.getenv('FLEXKV_SAFETY_TTL_MS', 100)),
    renew_lease_ms=int(os.getenv('FLEXKV_RENEW_LEASE_MS', 4000)),
)

@dataclass
class UserConfig:
    cpu_cache_gb: int = 16
    ssd_cache_gb: int = 0  # 0 means disable ssd
    ssd_cache_dir: Union[str, List[str]] = "./ssd_cache"
    enable_gds: bool = False
    enable_p2p_cpu: bool = False
    enable_p2p_ssd: bool = False
    enable_3rd_remote: bool = False
    
    # distributed zmq configs
    local_zmq_ip: Optional[str] = None
    local_zmq_port: Optional[int] = None
    # Redis configs (for KV sharing / metadata)
    redis_host: Optional[str] = None
    redis_port: Optional[int] = None
    local_ip: Optional[str] = None
    redis_password: Optional[str] = None

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
    from dataclasses import fields

    # read json config file or yaml config file
    if config_file.endswith('.json'):
        with open(config_file) as f:
            config = json.load(f)
    elif config_file.endswith(('.yaml', '.yml')):
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file extension: {config_file}")

    if 'ssd_cache_dir' in config:
        config['ssd_cache_dir'] = parse_path_list(config['ssd_cache_dir'])

    defined_fields = {f.name for f in fields(UserConfig)}
    known_config = {k: v for k, v in config.items() if k in defined_fields}
    extra_config = {k: v for k, v in config.items() if k not in defined_fields}

    user_config = UserConfig(**known_config)

    for key, value in extra_config.items():
        setattr(user_config, f"override_{key}", value)

    return user_config

def load_user_config_from_env() -> UserConfig:
    return UserConfig(
        cpu_cache_gb=int(os.getenv('FLEXKV_CPU_CACHE_GB', 16)),
        ssd_cache_gb=int(os.getenv('FLEXKV_SSD_CACHE_GB', 0)),
        ssd_cache_dir=parse_path_list(os.getenv('FLEXKV_SSD_CACHE_DIR', "./flexkv_ssd")),
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
    cache_config.enable_p2p_cpu = user_config.enable_p2p_cpu
    cache_config.enable_p2p_ssd = user_config.enable_p2p_ssd
    cache_config.enable_3rd_remote = user_config.enable_3rd_remote
    
    # Update derived flags after setting p2p and remote configs
    cache_config.enable_kv_sharing = (cache_config.enable_p2p_cpu or 
                                      cache_config.enable_p2p_ssd or 
                                      cache_config.enable_3rd_remote)
    cache_config.enable_remote = cache_config.enable_3rd_remote

    if cache_config.num_ssd_blocks % len(cache_config.ssd_cache_dir) != 0:
        cache_config.num_ssd_blocks = \
            (cache_config.num_ssd_blocks // len(cache_config.ssd_cache_dir) + 1) * len(cache_config.ssd_cache_dir)
        flexkv_logger.warning(f"num_ssd_blocks is not a multiple of num_ssd_devices, "
                              f"adjust num_ssd_blocks to {cache_config.num_ssd_blocks}")

    # Update distributed zmq and Redis configs if provided in user_config
    if user_config.local_zmq_ip is not None:
        cache_config.local_zmq_ip = user_config.local_zmq_ip
    if user_config.local_zmq_port is not None:
        cache_config.local_zmq_port = user_config.local_zmq_port
    if user_config.redis_host is not None:
        cache_config.redis_host = user_config.redis_host
    if user_config.redis_port is not None:
        cache_config.redis_port = user_config.redis_port
    if user_config.local_ip is not None:
        cache_config.local_ip = user_config.local_ip
    if user_config.redis_password is not None:
        cache_config.redis_password = user_config.redis_password

    global_config_attrs = set(vars(GLOBAL_CONFIG_FROM_ENV).keys())
    for attr_name in dir(user_config):
        if attr_name.startswith('override_'):
            global_attr_name = attr_name[9:]  # len('override_') = 9
            if global_attr_name in global_config_attrs:
                attr_value = getattr(user_config, attr_name)
                original_value = getattr(GLOBAL_CONFIG_FROM_ENV, global_attr_name)

                original_type = type(original_value)

                try:
                    if original_type is bool:
                        if isinstance(attr_value, str):
                            attr_value = attr_value.lower() in ('true', '1', 'yes')
                        else:
                            attr_value = bool(int(attr_value))
                    elif issubclass(original_type, Enum):  # KVCacheLayoutType
                        if isinstance(attr_value, str):
                            attr_value = original_type(attr_value.upper())
                        elif not isinstance(attr_value, original_type):
                            attr_value = original_type(attr_value)
                    else:
                        attr_value = original_type(attr_value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot convert config value '{attr_value}' to type {original_type.__name__} "
                                    f"for config '{global_attr_name}': {e}") from e

                setattr(GLOBAL_CONFIG_FROM_ENV, global_attr_name, attr_value)
                flexkv_logger.info(f"Override environment variable: {'FLEXKV_' + global_attr_name.upper()} "
                                   f"to {attr_value} from config file.")
            else:
                raise ValueError(f"Unknown config name: {global_attr_name} in config file, "
                                 f"available config names: {global_config_attrs}")
        
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
