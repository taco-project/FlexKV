
import json
import os
import torch
import tempfile
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from flexkv.common.debug import flexkv_logger

if TYPE_CHECKING:
    from tensorrt_llm.bindings.internal.args import TorchLlmArgs
    from tensorrt_llm.models import AutoConfig
    from pathlib import Path
    from transformers import AutoConfig as HFAutoConfig
    from flexkv.integration.tensorrt_llm.utils import get_dp_tp_info

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
    dp_size: int = 1
    dp_rank: int = 0
    
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
        
    def post_init_from_trt_config(
        self,
        llm_args: TorchLlmArgs,
    ):
        self.block_size = llm_args.kv_cache_config.tokens_per_block
        self.dtype = llm_args.dtype
        self.tp_size, self.dp_size, self.dp_rank = get_dp_tp_info(llm_args)
        
        model_path = Path(llm_args.model)
        assert model_path.exists(), f"Model path {model_path} does not exist."
        try:
            hf_config = HFAutoConfig.from_pretrained(
                str(model_path), 
                trust_remote_code=llm_args.trust_remote_code
            )
            self.num_layers = hf_config.num_hidden_layers
            self.num_kv_heads = getattr(hf_config, 'num_key_value_heads', 
                                        hf_config.num_attention_heads)
            self.head_size = hf_config.hidden_size // hf_config.num_attention_heads
            
            self.use_mla = (hasattr(hf_config, 'kv_lora_rank') and 
                            hf_config.kv_lora_rank is not None and
                            hasattr(hf_config, 'qk_rope_head_dim') and 
                            hf_config.qk_rope_head_dim is not None)
        except Exception as e:
            logger.error(f"Failed to load config from {model_path}: {e}")
