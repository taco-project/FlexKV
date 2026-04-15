
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

    def _detect_indexer_config_from_hf(self, hf_config, source: str = "") -> None:
        if hf_config is None:
            return

        try:
            qk_rope_head_dim = getattr(hf_config, 'qk_rope_head_dim', None)
            if qk_rope_head_dim is None or qk_rope_head_dim <= 0:
                return

            index_head_dim = getattr(hf_config, 'index_head_dim', None)
            if index_head_dim is not None and index_head_dim > 0:
                quant_block_size = 128
                head_size = self.cache_config.tokens_per_block * (
                    index_head_dim + index_head_dim // quant_block_size * 4
                )
            else:
                head_size = qk_rope_head_dim

            # tokens_per_block is already set to sglang page_size before this
            # call, so each FlexKV block = 1 sglang page.  The indexer maps
            # 1:1 with blocks — no extra page_size grouping is needed.  For
            # NSA/DSA models, head_size stores the packed per-page buffer width
            # so the CPU layout matches the GPU indexer tensor shape.
            self.cache_config.indexer = IndexerCacheConfig(
                head_size=head_size,
                num_kv_heads=1,
                dtype=torch.uint8,
            )
            source_label = f" ({source})" if source else ""
            logger.info(
                f"Detected sparse attention indexer config{source_label}: "
                f"head_size={head_size}, dtype=uint8, "
                f"tokens_per_block={self.cache_config.tokens_per_block}")
        except Exception as e:
            logger.debug(f"Could not detect indexer config ({source}): {e}")

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
        self.model_config.head_size = vllm_config.model_config.get_head_size()
        self.model_config.dtype = vllm_config.model_config.dtype
        self.model_config.use_mla = vllm_config.model_config.is_deepseek_mla
        self.model_config.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config.dp_size = vllm_config.parallel_config.data_parallel_size
        self.model_config.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.model_config.pp_rank = getattr(vllm_config.parallel_config, 'pipeline_parallel_rank', 0)
        if self.model_config.use_mla:
            self.model_config.num_kv_heads = 1
        else:
            self.model_config.num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)
        self.server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        self.gpu_register_port = self.server_recv_port + "_gpu_register"

        hf_config = getattr(vllm_config.model_config, 'hf_config', None)
        self._detect_indexer_config_from_hf(hf_config, source="vllm")

    def post_init_from_sglang_config(
        self,
        sglang_config,
        tp_size: int,
        page_size: int,
        num_local_layers: int = 0,
        pp_size: int = 1,
        pp_rank: int = 0,
        dp_size: int = 1,
        dp_rank: int = 0,
        is_nsa_cp: bool = False,
        cp_size: int = 1,
        cp_rank: int = 0,
    ):
        """
        Initialize FlexKVConfig fields from sglang config.
        Args:
            sglang_config: sglang.srt.configs.model_config.ModelConfig-like object
            tp_size: tensor parallel size used by sglang
            page_size: KV block size (tokens per block) used by sglang
            num_local_layers: number of layers on this PP rank (0 means no PP, use total layers)
            pp_size: pipeline parallel size (default 1, no PP)
            pp_rank: pipeline parallel rank (default 0)
            dp_size: data parallel size (default 1, no DP)
            dp_rank: data parallel rank (default 0)
            is_nsa_cp: whether NSA context parallelism is enabled
            cp_size: context parallel size (default 1, no CP)
            cp_rank: context parallel rank (default 0)
        """
        # cache config: use page_size as tokens_per_block so that FlexKV's
        # CPU radix tree manages blocks at page granularity, ensuring that
        # hash generation, matching, insertion and eviction are all page-aligned.
        self.cache_config.tokens_per_block = page_size

        total_layers = int(getattr(sglang_config, "num_hidden_layers", 0))
        self.model_config.num_layers = int(num_local_layers) if num_local_layers > 0 else total_layers

        attn_arch = getattr(sglang_config, "attention_arch", None)
        use_mla = False
        if hasattr(attn_arch, "name"):
            use_mla = (attn_arch.name.upper() == "MLA")
        elif isinstance(attn_arch, str):
            use_mla = (attn_arch.upper() == "MLA")

        if use_mla:
            kv_lora_rank = int(getattr(sglang_config, "kv_lora_rank", 0))
            qk_rope_head_dim = int(getattr(sglang_config, "qk_rope_head_dim", 0))
            mla_head_size = kv_lora_rank + qk_rope_head_dim
            self.model_config.num_kv_heads = 1
            self.model_config.head_size = int(mla_head_size)
        else:
            if hasattr(sglang_config, "get_total_num_kv_heads"):
                try:
                    self.model_config.num_kv_heads = int(sglang_config.get_total_num_kv_heads())
                except Exception:
                    self.model_config.num_kv_heads = int(getattr(sglang_config, "num_key_value_heads", 0))
            elif hasattr(sglang_config, "get_num_kv_heads"):
                try:
                    per_rank = int(sglang_config.get_num_kv_heads(tp_size))
                    self.model_config.num_kv_heads = per_rank * tp_size
                except Exception:
                    self.model_config.num_kv_heads = int(getattr(sglang_config, "num_key_value_heads", 0))
            else:
                self.model_config.num_kv_heads = int(getattr(sglang_config, "num_key_value_heads", 0))
            self.model_config.head_size = int(getattr(sglang_config, "head_dim", 0))

        # Determine KV cache dtype: prioritize user_config.kv_cache_dtype (from
        # flexkv_config.yaml or FLEXKV_KV_CACHE_DTYPE env var), then fall back
        # to the sglang model dtype.  sglang's ModelConfig.dtype is the *model
        # weight* dtype (e.g. bfloat16), which may differ from the KV cache
        # dtype (e.g. fp8_e4m3 when --kv-cache-dtype fp8_e4m3 is used).
        def _parse_dtype_str(dtype_str: str) -> torch.dtype:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
                "fp8": torch.float8_e4m3fn,
                "float8": torch.float8_e4m3fn,
                "e4m3": torch.float8_e4m3fn,
                "fp8_e4m3": torch.float8_e4m3fn,
            }
            return dtype_map.get(dtype_str.lower(), torch.bfloat16)

        user_dtype_str = self.user_config.kv_cache_dtype
        if user_dtype_str is not None:
            self.model_config.dtype = _parse_dtype_str(user_dtype_str)
            logger.info(
                f"[FlexKV] Using kv_cache_dtype from user_config: "
                f"'{user_dtype_str}' -> {self.model_config.dtype}"
            )
        else:
            self.model_config.dtype = getattr(sglang_config, "dtype", torch.bfloat16)
            logger.warning(
                f"[FlexKV] No kv_cache_dtype in user_config, falling back to sglang "
                f"model dtype: {self.model_config.dtype}. If your KV cache uses a "
                f"different dtype (e.g. fp8), add 'kv_cache_dtype: fp8' to your "
                f"flexkv_config.yaml or set FLEXKV_KV_CACHE_DTYPE=fp8 environment variable."
            )

        if use_mla and getattr(sglang_config, "index_head_dim", None) is not None:
            kv_lora_rank = int(getattr(sglang_config, "kv_lora_rank", 0))
            qk_rope_head_dim = int(getattr(sglang_config, "qk_rope_head_dim", 0))
            if self.model_config.dtype == torch.float8_e4m3fn:
                assert kv_lora_rank % 128 == 0, (
                    f"kv_lora_rank {kv_lora_rank} must be multiple of 128 "
                    "for NSA FP8 KV cache layout"
                )
                self.model_config.head_size = int(
                    kv_lora_rank
                    + kv_lora_rank // 128 * 4
                    + qk_rope_head_dim * torch.bfloat16.itemsize
                )

        self.model_config.use_mla = use_mla

        self.model_config.tp_size = int(tp_size)
        self.model_config.dp_size = int(dp_size if dp_size is not None else 1)
        self.model_config.dp_rank = int(dp_rank if dp_rank is not None else 0)
        self.model_config.pp_size = int(pp_size)
        self.model_config.pp_rank = int(pp_rank)
        self.model_config.is_nsa_cp = is_nsa_cp
        self.model_config.cp_size = int(cp_size if cp_size is not None else 1)
        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)

        # Each PP rank needs its own IPC ports so that their
        # KVManager / TransferManager instances do not collide on the same
        # ZMQ endpoint.  DP ranks share the same KVServer (only DP0 creates
        # it), so they must use the same IPC port.
        _dp_rank = int(dp_rank if dp_rank is not None else 0)
        port_suffix = ""
        if int(pp_size) > 1:
            port_suffix += f"_pp{int(pp_rank)}"
        if port_suffix:
            self.server_recv_port = f"{self.server_recv_port}{port_suffix}"
            self.gpu_register_port = f"{self.server_recv_port}_gpu_register"

        rank_parts = []
        if int(tp_size) > 1:
            rank_parts.append("tp_rank=0")
        if int(pp_size) > 1:
            rank_parts.append(f"pp_rank={int(pp_rank)}")
        if int(self.model_config.dp_size) > 1:
            rank_parts.append(f"dp_rank={_dp_rank}")
        rank_label = f" [{', '.join(rank_parts)}]" if rank_parts else ""
        logger.info(
            f"[FlexKV] IPC ports configured{rank_label}: "
            f"server_recv_port={self.server_recv_port}, "
            f"gpu_register_port={self.gpu_register_port}"
        )

        hf_config = getattr(sglang_config, 'hf_config', None)
        self._detect_indexer_config_from_hf(hf_config, source="sglang")

        if self.cache_config.indexer is not None:
            logger.info(
                f"[FlexKV] Complete indexer config (sglang): "
                f"head_size={self.cache_config.indexer.head_size}, "
                f"dtype={self.cache_config.indexer.dtype}, "
                f"num_layers={self.model_config.num_layers}, "
                f"tokens_per_block={self.cache_config.tokens_per_block}"
            )

    def post_init_from_trt_config(
        self,
        config,
    ):
        self.cache_config.tokens_per_block = config.tokens_per_block
        # Convert dtype string to torch.dtype
        dtype_str = config.pytorch_backend_config.kv_cache_dtype
        flexkv_logger.info(f"[FlexKVConfig] dtype_str from TRT config: {dtype_str}")

        # Helper function to convert dtype string to torch.dtype
        def _parse_dtype_str(dtype_str: str) -> torch.dtype:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
                "fp8": torch.float8_e4m3fn,
                "float8": torch.float8_e4m3fn,
                "e4m3": torch.float8_e4m3fn,
            }
            return dtype_map.get(dtype_str.lower(), torch.bfloat16)

        if dtype_str == "auto":
            # When dtype_str is "auto", try to get kv_cache_dtype from user_config first
            # This allows users to specify kv_cache_dtype in flexkv_config.json or via environment variable
            user_dtype_str = self.user_config.kv_cache_dtype
            if user_dtype_str is not None:
                parsed_dtype = _parse_dtype_str(user_dtype_str)
                self.model_config.dtype = parsed_dtype
                flexkv_logger.info(f"[FlexKVConfig] dtype_str='auto', but found kv_cache_dtype='{user_dtype_str}' in user_config, using it -> {parsed_dtype}")
            else:
                # Try to infer from TRT config if possible (e.g., from actual tensor dtype)
                # Note: This might not be available at initialization time
                self.model_config.dtype = torch.bfloat16
                flexkv_logger.warning(
                    f"[FlexKVConfig] dtype_str='auto' and no kv_cache_dtype in user_config. "
                    f"Falling back to {self.model_config.dtype}. To specify a different dtype, add 'kv_cache_dtype' "
                    f"to your flexkv_config.json file (e.g., {{\"kv_cache_dtype\": \"fp8\"}}) "
                    f"or set FLEXKV_KV_CACHE_DTYPE environment variable."
                )
        elif isinstance(dtype_str, str):
            self.model_config.dtype = _parse_dtype_str(dtype_str)
        else:
            self.model_config.dtype = dtype_str

        # Set model config (parallel configs part)
        if config.mapping.enable_attention_dp:
            self.model_config.tp_size = 1
            self.model_config.dp_size = config.mapping.tp_size
        else:
            self.model_config.tp_size = config.mapping.tp_size
            self.model_config.dp_size = 1
        self.model_config.pp_size = getattr(config.mapping, 'pp_size', 1)
        self.model_config.pp_rank = getattr(config.mapping, 'pp_rank', 0)

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

            self._detect_indexer_config_from_hf(hf_config, source="TRT-LLM")
        except Exception as e:
            flexkv_logger.error(f"Failed to load config from {model_path}: {e}")
        # Update cache config with user config after model config is initialized
        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)
