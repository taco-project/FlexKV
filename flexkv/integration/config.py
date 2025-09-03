
import json
import os
import torch
import tempfile
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from flexkv.common.debug import flexkv_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig, FullAttentionSpec
    from vllm.config import VllmConfig


logger = flexkv_logger

@dataclass
class FlexKVConfig:
    #base config
    server_recv_port: str
    
    # cache config
    cache_config: dict = field(default_factory=dict)
    
    # model config
    block_size: int = None
    num_layers: int = None
    num_kv_heads: int = None
    head_size: int = None
    dtype: torch.dtype = None
    use_mla: bool = False
    tp_size: int = 1
    
    # log config
    num_log_interval_requests: int = 200
    
    @classmethod
    def from_env(cls) -> 'FlexKVConfig':
        config_file_path = os.getenv('FLEXKV_CONFIG_PATH', None)
        logger.info(f"{config_file_path=}")
        if config_file_path is None:
            return cls(enable_flexkv=False,
                       server_recv_port="")
        
        assert config_file_path.endswith(".json"), "flexkv config must be a json file."
        
        with open(config_file_path, 'r') as f:
            config_dict: dict = json.load(f)
        logger.info(f"FlexKV Config Dict: {config_dict}")
        
        return cls(
            server_recv_port=config_dict.get("server_recv_port", f"ipc:///tmp/flexkv_test"),
            cache_config=config_dict.get("cache_config", {}),
            num_log_interval_requests=config_dict.get("num_log_interval_requests", 200),
        )
        
    def post_init_from_vllm_config(
        self, 
        vllm_config: "VllmConfig",
        ):
        self.num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.block_size = vllm_config.cache_config.block_size
        self.num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        self.head_size = vllm_config.model_config.get_head_size()
        self.dtype = vllm_config.model_config.dtype
        self.use_mla = vllm_config.model_config.is_deepseek_mla
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size