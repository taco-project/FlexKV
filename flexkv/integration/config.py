
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
        self._hf_text_config = getattr(vllm_config.model_config, 'hf_text_config', None)
        self.model_config.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config.dp_size = vllm_config.parallel_config.data_parallel_size
        if self.model_config.use_mla:
            self.model_config.num_kv_heads = 1
        else:
            self.model_config.num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()

        # Detect heterogeneous KV cache layers (e.g., Gemma4 with different
        # head_dim/num_kv_heads for sliding vs full attention layers) BEFORE
        # computing block counts, so token_size_in_bytes is correct from the start.
        self._detect_heterogeneous_kv_layers()

        update_default_config_from_user_config(self.model_config, self.cache_config, self.user_config)
        base_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        if not GLOBAL_CONFIG_FROM_ENV.server_client_mode and GLOBAL_CONFIG_FROM_ENV.instance_num <= 1:
            # In non-server-client mode, make IPC paths unique per DP instance
            # to avoid collisions when vLLM v1 runs multiple DP instances as
            # independent processes (each reporting dp_size=1).
            # Use engine_id suffix (e.g. "_dp0") which is consistent across
            # EngineCore and Worker processes of the same DP instance.
            try:
                engine_id = vllm_config.kv_transfer_config.engine_id
                if engine_id and '_dp' in engine_id:
                    dp_part = engine_id.rsplit('_dp', 1)[-1]
                    base_port = base_port + f"_dp{dp_part}"
            except (AttributeError, TypeError):
                pass
        self.server_recv_port = base_port
        self.gpu_register_port = self.server_recv_port + "_gpu_register"


    def _detect_heterogeneous_kv_layers(self) -> None:
        """Detect heterogeneous KV cache layer groups from hf_text_config.

        Models like Gemma4 use different (num_kv_heads, head_dim) for different
        attention layer types (sliding_attention vs full_attention).  When detected,
        this sets model_config.layer_groups so that token_size_in_bytes (and thus
        num_cpu_blocks / num_ssd_blocks) are computed correctly.
        """
        hf = self._hf_text_config
        if hf is None:
            return

        layer_types = getattr(hf, 'layer_types', None)
        if not layer_types:
            return

        # Read per-type KV configs (Gemma4 naming convention)
        default_head_dim = getattr(hf, 'head_dim', None)
        default_num_kv_heads = getattr(hf, 'num_key_value_heads', None)
        global_head_dim = getattr(hf, 'global_head_dim', None)
        global_num_kv_heads = getattr(hf, 'num_global_key_value_heads', None)

        if default_head_dim is None or default_num_kv_heads is None:
            return

        # Resolve global values (fall back to default if not set)
        if global_num_kv_heads is None:
            global_num_kv_heads = default_num_kv_heads
        if global_head_dim is None:
            global_head_dim = default_head_dim

        # Only proceed if there's actual heterogeneity in KV cache shapes
        if global_head_dim == default_head_dim and global_num_kv_heads == default_num_kv_heads:
            return

        # Account for KV sharing between consecutive layers
        num_kv_shared = getattr(hf, 'num_kv_shared_layers', 0)
        group_size = max(num_kv_shared, 1)

        sliding_window = getattr(hf, 'sliding_window', None)
        tp_size = self.model_config.tp_size

        # Count unique KV layers per type (stepping by group_size for sharing)
        type_counts: dict = {}
        type_indices: dict = {}
        unique_idx = 0
        for i in range(0, len(layer_types), group_size):
            lt = layer_types[i]
            type_counts[lt] = type_counts.get(lt, 0) + 1
            type_indices.setdefault(lt, []).append(unique_idx)
            unique_idx += 1

        # Build LayerGroupSpec entries with per-GPU num_kv_heads
        from flexkv.common.config import LayerGroupSpec
        layer_groups = []
        for lt, count in type_counts.items():
            if lt == 'full_attention':
                kv_heads = global_num_kv_heads // tp_size
                h_dim = global_head_dim
                sw = None
            else:
                kv_heads = default_num_kv_heads // tp_size
                h_dim = default_head_dim
                sw = sliding_window if 'sliding' in lt else None

            layer_groups.append(LayerGroupSpec(
                num_layers=count,
                num_kv_heads=kv_heads,
                head_size=h_dim,
                layer_indices=type_indices[lt],
                sliding_window=sw,
            ))

        # Validate: total unique layers must match what vLLM reported
        total_unique = sum(g.num_layers for g in layer_groups)
        if total_unique != self.model_config.num_layers:
            logger.warning(
                f"Heterogeneous layer detection: counted {total_unique} unique "
                f"layers from hf_text_config but vLLM reports "
                f"{self.model_config.num_layers}. Skipping early layer_groups "
                f"(will be detected from GPU tensors later).")
            return

        self.model_config.layer_groups = layer_groups
        for g in layer_groups:
            logger.info(
                f"Detected layer group from hf_text_config: "
                f"{g.num_layers} layers, num_kv_heads={g.num_kv_heads}, "
                f"head_size={g.head_size}, sliding_window={g.sliding_window}")

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
