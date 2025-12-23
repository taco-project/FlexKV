
import json
import os
import torch
import tempfile
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from flexkv.common.debug import flexkv_logger
from flexkv.common.config import *

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig, FullAttentionSpec
    from vllm.config import VllmConfig


logger = flexkv_logger

@dataclass
class FlexKVConfig:
    enable_flexkv: bool = True

    #base config
    server_recv_port: str = ""

    gpu_register_port: str = ""

    # cache config
    cache_config: CacheConfig = field(default_factory=CacheConfig)

    # model config
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # user config
    user_config: UserConfig = field(default_factory=UserConfig)

    def __post_init__(self):
        if self.server_recv_port == "":
            self.server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        if self.gpu_register_port == "":
            self.gpu_register_port = self.server_recv_port + "_gpu_register"

    @classmethod
    def from_env(cls) -> 'FlexKVConfig':
        enable_flexkv = bool(int(os.getenv('ENABLE_FLEXKV', 1)))
        config_file_path = os.getenv('FLEXKV_CONFIG_PATH', None)
        if config_file_path is None:
            logger.info("No flexkv config file provided, please set FLEXKV_CONFIG_PATH environment variable.")
            logger.info("Loading flexkv config from environment variables.")
            user_config = load_user_config_from_env()
            return cls(enable_flexkv=enable_flexkv,
                       user_config=user_config)
        else:
            logger.info(f"Loading flexkv config from file: {config_file_path}")
            user_config = load_user_config_from_file(config_file_path) # NVMe-oF must be enabled using config file.
            return cls(enable_flexkv=enable_flexkv,
                       user_config=user_config)

    def post_init_from_vllm_config(
        self,
        vllm_config: "VllmConfig",
        ):
        self.cache_config.tokens_per_block = vllm_config.cache_config.block_size

        self.model_config.num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.model_config.head_size = vllm_config.model_config.get_head_size()
        self.model_config.dtype = vllm_config.model_config.dtype
        self.model_config.use_mla = vllm_config.model_config.is_deepseek_mla
        self.model_config.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config.dp_size = vllm_config.parallel_config.data_parallel_size
        if self.model_config.use_mla:
            self.model_config.num_kv_heads = 1
        else:
            self.model_config.num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)


    def post_init_from_sglang_config(
        self,
        sglang_config,
        tp_size: int,
        page_size: int,
    ):
        """
        Initialize FlexKVConfig fields from sglang config.
        Args:
            sglang_config: sglang.srt.configs.model_config.ModelConfig-like object
            tp_size: tensor parallel size used by sglang
            page_size: KV block size (tokens per block) used by sglang
        """
        # cache config
        self.cache_config.tokens_per_block = int(page_size)

        self.model_config.num_layers = int(getattr(sglang_config, "num_hidden_layers", 0))

        if hasattr(sglang_config, "get_num_kv_heads"):
            try:
                self.model_config.num_kv_heads = int(sglang_config.get_num_kv_heads(tp_size))
            except Exception:
                self.model_config.num_kv_heads = int(getattr(sglang_config, "num_key_value_heads", 0))
        else:
            self.model_config.num_kv_heads = int(getattr(sglang_config, "num_key_value_heads", 0))
        self.model_config.head_size = int(getattr(sglang_config, "head_dim", 0))

        self.model_config.dtype = getattr(sglang_config, "dtype", torch.bfloat16)

        attn_arch = getattr(sglang_config, "attention_arch", None)
        use_mla = False
        if hasattr(attn_arch, "name"):
            use_mla = (attn_arch.name.upper() == "MLA")
        elif isinstance(attn_arch, str):
            use_mla = (attn_arch.upper() == "MLA")
        self.model_config.use_mla = use_mla

        self.model_config.tp_size = int(tp_size)
        self.model_config.dp_size = int(getattr(sglang_config, "dp_size", 1))
        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)

    def post_init_from_trt_config(
        self,
        config,
    ):
        self.cache_config.tokens_per_block = config.tokens_per_block
        # Convert dtype string to torch.dtype
        dtype_str = config.pytorch_backend_config.kv_cache_dtype
        if dtype_str == "auto":
            self.model_config.dtype = torch.bfloat16
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
            self.model_config.dtype = dtype_map.get(dtype_str, torch.bfloat16)
        else:
            self.model_config.dtype = dtype_str
        
        # Set model config (parallel configs part)
        if config.mapping.enable_attention_dp:
            self.model_config.tp_size = 1
            self.model_config.dp_size = config.mapping.tp_size
        else:
            self.model_config.tp_size = config.mapping.tp_size
            self.model_config.dp_size = 1
            
        # self.model_config (model configs part)
        try:
            model_path = getattr(config, 'hf_model_dir', None)
            from transformers import AutoConfig as HFAutoConfig
            hf_config = HFAutoConfig.from_pretrained(
                str(model_path), 
                trust_remote_code=True
            )
            self.model_config.num_layers = hf_config.num_hidden_layers
            self.model_config.use_mla = (hasattr(hf_config, 'kv_lora_rank') and 
                            hf_config.kv_lora_rank is not None and
                            hasattr(hf_config, 'qk_rope_head_dim') and 
                            hf_config.qk_rope_head_dim is not None)
            if self.model_config.use_mla:
                self.model_config.head_size = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
                self.model_config.num_kv_heads = 1
            else:
                if hasattr(hf_config, 'num_key_value_heads'):
                    assert hf_config.num_attention_heads != hf_config.num_key_value_heads, f"{hf_config.num_attention_heads=}, {hf_config.num_key_value_heads=}"
                    self.model_config.head_size = hf_config.head_dim
                    self.model_config.num_kv_heads = hf_config.num_key_value_heads
                else:
                    self.model_config.head_size = hf_config.hidden_size // hf_config.num_attention_heads
                    self.model_config.num_kv_heads = hf_config.num_attention_heads
            
        except Exception as e:
            flexkv_logger.error(f"Failed to load config from {model_path}: {e}")
        # Update cache config with user config after model config is initialized
        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)
