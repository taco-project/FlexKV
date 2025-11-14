
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
        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)

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
            user_config = load_user_config_from_file(config_file_path)
            return cls(enable_flexkv=enable_flexkv,
                       user_config=user_config)

    def post_init_from_vllm_config(
        self,
        vllm_config: "VllmConfig",
        ):
        self.cache_config.tokens_per_block = vllm_config.cache_config.block_size

        self.model_config.num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.model_config.num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        self.model_config.head_size = vllm_config.model_config.get_head_size()
        self.model_config.dtype = vllm_config.model_config.dtype
        self.model_config.use_mla = vllm_config.model_config.is_deepseek_mla
        self.model_config.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config.dp_size = vllm_config.parallel_config.data_parallel_size

        self.__post_init__()

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

        self.__post_init__()
