
import json
import os
import torch
import tempfile
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from transformers import AutoConfig as HFAutoConfig

from flexkv.common.debug import flexkv_logger
from flexkv.integration.tensorrt_llm.utils import get_dp_tp_info

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.bindings.executor import ExecutorConfig

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
    dp_size: int = 1
    dp_rank: int = 0
    
    # log config
    num_log_interval_requests: int = 200
    
    @classmethod
    def from_env(cls) -> 'FlexKVConfig':
        config_file_path = os.getenv('FLEXKV_CONFIG_PATH', None)
        flexkv_logger.info(f"{config_file_path=}")
        if config_file_path is None:
            return cls(enable_flexkv=False,
                       server_recv_port="")
        
        assert config_file_path.endswith(".json"), "flexkv config must be a json file."
        
        with open(config_file_path, 'r') as f:
            config_dict: dict = json.load(f)
        flexkv_logger.info(f"FlexKV Config Dict: {config_dict}")
        
        return cls(
            server_recv_port=config_dict.get("server_recv_port", f"ipc:///tmp/flexkv_test"),
            cache_config=config_dict.get("cache_config", {}),
            num_log_interval_requests=config_dict.get("num_log_interval_requests", 200),
        )
        
    def post_init_from_trt_config(
        self,
        config: ExecutorConfig,
    ):
        self.block_size = config.tokens_per_block
        # Convert dtype string to torch.dtype
        dtype_str = config.pytorch_backend_config.kv_cache_dtype
        if dtype_str == "auto":
            self.dtype = torch.bfloat16
        elif isinstance(dtype_str, str):
            # Convert string to torch.dtype
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
            }
            self.dtype = dtype_map.get(dtype_str, torch.bfloat16)
        else:
            self.dtype = dtype_str
            
        self.tp_size, self.dp_size, self.dp_rank = get_dp_tp_info(config)
        
        model_path = os.getenv('MODEL_PATH', None)
        
        try:
            hf_config = HFAutoConfig.from_pretrained(
                str(model_path), 
                trust_remote_code=True
            )
            self.num_layers = hf_config.num_hidden_layers
            self.use_mla = (hasattr(hf_config, 'kv_lora_rank') and 
                            hf_config.kv_lora_rank is not None and
                            hasattr(hf_config, 'qk_rope_head_dim') and 
                            hf_config.qk_rope_head_dim is not None)
            if self.use_mla:
                self.head_size = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
                self.num_kv_heads = 1
            else:
                self.head_size = hf_config.hidden_size // hf_config.num_key_value_heads // self.tp_size
                self.num_kv_heads = hf_config.num_key_value_heads
            
        except Exception as e:
            flexkv_logger.error(f"Failed to load config from {model_path}: {e}")
