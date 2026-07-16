
import json
import os
import torch
import tempfile
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, field

from flexkv.common.debug import flexkv_logger
from flexkv.common.config import *

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig, FullAttentionSpec
    from vllm.config import VllmConfig


logger = flexkv_logger


def _dsv4_swa_transfer_enabled_from_env() -> bool:
    return bool(int(os.getenv("FLEXKV_ENABLE_SWA_TRANSFER", "1")))


def _is_nvfp4_dtype_str(dtype_str: Optional[str]) -> bool:
    """Return True if *dtype_str* selects the NVFP4 packed KV cache layout."""
    return isinstance(dtype_str, str) and dtype_str.lower() in ("nvfp4", "fp4", "e2m1")


def nvfp4_kv_cache_full_dim(head_size: int) -> int:
    """Packed last-dim size (in bytes / uint8 elements) for an NVFP4 KV cache.

    Mirrors vLLM's ``vllm.utils.torch_utils.nvfp4_kv_cache_full_dim``:
    each head packs ``head_size // 2`` bytes of fp4 data (2 fp4 values per byte)
    plus ``head_size // 16`` bytes of fp8 block scales (1 scale per 16 elements).
    """
    return head_size // 2 + head_size // 16


def _warn_nvfp4_unsupported_framework(dtype_str: Optional[str], framework: str) -> None:
    """Warn if NVFP4 KV cache is requested from a framework whose FlexKV adapter
    has not yet implemented the nvfp4 packed-layout ``head_size`` fold.

    Only the vLLM adapter (``post_init_from_vllm_config``) folds the packed width
    ``head_size//2 + head_size//16`` into ``head_size`` so the CPU/SSD mirror
    matches the framework's packed GPU tensor byte-for-byte. Without that fold the
    CPU mirror is sized for the *logical* head_size and offload/reload would be
    byte-misaligned. Until each framework's packed nvfp4 layout is verified we
    only warn here rather than silently produce a corrupt mirror.
    """
    if not _is_nvfp4_dtype_str(dtype_str):
        return
    # TODO(nvfp4): implement + verify the nvfp4 packed head_size fold for the
    # {framework} adapter (mirror post_init_from_vllm_config: guard non-MLA,
    # set model_config.head_size = nvfp4_kv_cache_full_dim(head_size)). Confirm
    # the framework stores nvfp4 KV as a single packed uint8 tensor with the same
    # (head_size//2 + head_size//16) last-dim layout as vLLM before enabling.
    logger.warning(
        f"[FlexKV {framework}] kv_cache_dtype='{dtype_str}' (NVFP4) requested, but "
        f"the {framework} FlexKV adapter does NOT yet apply the nvfp4 packed "
        f"head_size fold. The CPU/SSD mirror may be byte-misaligned with the "
        f"packed GPU tensor -> offload/reload correctness is NOT guaranteed. "
        f"NVFP4 is currently verified only through the vLLM adapter. "
        f"See TODO(nvfp4) in flexkv/integration/config.py."
    )


def _dsv4_swa_padded_bytes_per_token(swa_page_size: int,
                                     logical_bytes_per_token: int = 584) -> int:
    """Effective per-token byte width of the DSv4 SWA GPU buffer, INCLUDING the
    576-byte page-alignment padding that DeepSeekV4SingleKVPool.create_buffer
    applies.

    The GPU SWA buffer stores each page as
    ``bytes_per_page_padded = ceil_div(swa_page_size * 584, 576) * 576`` and the
    connector registers ``head_size = bytes_per_page_padded / swa_page_size``.
    For swa_page_size=256 this is 149760/256 = 585 (584 logical + 1B/token of
    spread padding). The FlexKV host SWA pool must use the SAME width so byte
    offsets line up with the GPU buffer stride. Requires the padded page bytes
    to divide evenly by swa_page_size (holds for typical DSv4 configs); raises
    otherwise so a mismatch fails loudly rather than shearing SWA KV silently.
    """
    non_padded = swa_page_size * logical_bytes_per_token
    padded = ((non_padded + 576 - 1) // 576) * 576
    if padded % swa_page_size != 0:
        raise ValueError(
            f"DSv4 SWA padded page bytes {padded} not divisible by "
            f"swa_page_size {swa_page_size}; cannot derive a per-token byte "
            f"width. Check swa_page_size / 576-alignment constants."
        )
    return padded // swa_page_size


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

    def _resolve_dtype(
        self,
        framework_dtype_str: Optional[str],
        fallback_dtype: torch.dtype,
    ) -> None:
        """Resolve KV cache dtype with unified priority logic.

        Priority:
          1. User env-var / config (``user_config.kv_cache_dtype``) — highest
          2. Framework-reported dtype (``framework_dtype_str``, e.g. from
             sglang ``--kv-cache-dtype`` or vllm ``cache_dtype``)
          3. ``fallback_dtype`` — model weight dtype or hardcoded default

        Args:
            framework_dtype_str: dtype string from framework config (None or
                ``"auto"`` means not explicitly set).
            fallback_dtype: dtype to use when nothing else is available
                (typically model weight dtype).
        """
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
            # NVFP4: vLLM stores the packed fp4 data + fp8 block scales in a
            # single uint8 tensor, so FlexKV mirrors it as a 1-byte-per-element
            # buffer. The packed per-head element count is folded into head_size
            # elsewhere (see ``nvfp4_kv_cache_full_dim`` /
            # ``post_init_from_vllm_config``).
            "nvfp4": torch.uint8,
            "fp4": torch.uint8,
            "e2m1": torch.uint8,
            "fp4_e2m1": torch.uint8,
        }

        def _parse(s: str) -> torch.dtype:
            return dtype_map.get(s.lower(), torch.bfloat16)

        user_dtype_str = self.user_config.kv_cache_dtype

        # --- Priority 1: user explicit override ---
        if user_dtype_str is not None:
            resolved = _parse(user_dtype_str)
            self.model_config.dtype = resolved
            logger.info(f"[FlexKV] Using kv_cache_dtype from user_config: '{user_dtype_str}' -> {resolved}")
            return

        # --- Priority 2: framework config ---
        if framework_dtype_str is not None and framework_dtype_str != "auto":
            resolved = _parse(framework_dtype_str)
            self.model_config.dtype = resolved
            logger.info(f"[FlexKV] Using kv_cache_dtype from framework config: '{framework_dtype_str}' -> {resolved}")
            return

        # --- Priority 3: fallback ---
        self.model_config.dtype = fallback_dtype
        logger.warning(
            f"[FlexKV] No kv_cache_dtype from user/framework config, "
            f"falling back to {fallback_dtype}. "
            f"Set FLEXKV_KV_CACHE_DTYPE env var or pass --kv-cache-dtype to be explicit."
        )

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
        ) -> RankInfo:
        parallel_config = vllm_config.parallel_config
        tp_rank = int(getattr(parallel_config, 'tensor_parallel_rank', 0))
        pp_rank = int(getattr(parallel_config, 'pipeline_parallel_rank', 0))
        dp_rank = int(getattr(parallel_config, 'data_parallel_rank', 0))
        node_rank = int(getattr(parallel_config, 'node_rank', 0))
        self.cache_config.tokens_per_block = vllm_config.cache_config.block_size

        self.model_config.num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.model_config.head_size = vllm_config.model_config.get_head_size()
        vllm_kv_cache_dtype = getattr(vllm_config.cache_config, 'cache_dtype', 'auto')
        self._resolve_dtype(
            framework_dtype_str=vllm_kv_cache_dtype if isinstance(vllm_kv_cache_dtype, str) else None,
            fallback_dtype=getattr(vllm_config.model_config, 'dtype', torch.bfloat16),
        )
        self.model_config.use_mla = vllm_config.model_config.is_deepseek_mla
        # NVFP4: vLLM stores the packed fp4 data + fp8 block scales in a single
        # uint8 tensor whose per-head last dim is head_size//2 + head_size//16.
        # _resolve_dtype has already mapped nvfp4 -> uint8; here we fold the
        # packed width into head_size so the CPU/SSD mirror matches vLLM's packed
        # GPU tensor byte-for-byte. The effective dtype string follows the same
        # user-first-else-framework priority _resolve_dtype uses.
        effective_dtype_str = (
            self.user_config.kv_cache_dtype
            if self.user_config.kv_cache_dtype is not None
            else (vllm_kv_cache_dtype if isinstance(vllm_kv_cache_dtype, str) else None)
        )
        if _is_nvfp4_dtype_str(effective_dtype_str) and not self.model_config.use_mla:
            logical_head_size = self.model_config.head_size
            packed_head_size = nvfp4_kv_cache_full_dim(logical_head_size)
            self.model_config.head_size = packed_head_size
            logger.info(
                f"[FlexKV vllm] NVFP4 KV cache detected: folding packed layout "
                f"into head_size (logical={logical_head_size} -> "
                f"packed={packed_head_size}, dtype=uint8)"
            )
        elif _is_nvfp4_dtype_str(effective_dtype_str) and self.model_config.use_mla:
            logger.warning(
                "[FlexKV vllm] kv_cache_dtype='nvfp4' requested for an MLA "
                "model. vLLM MLA backends do NOT support nvfp4 KV cache now; "
                "skipping the nvfp4 head_size fold. If vLLM rejects this "
                "config, use fp8/fp8_ds_mla for MLA instead."
            )
        self.model_config.tp_size = int(parallel_config.tensor_parallel_size)
        self.model_config.dp_size = int(parallel_config.data_parallel_size)
        self.model_config.pp_size = int(parallel_config.pipeline_parallel_size)
        # vLLM CP (context parallel) support: read cp_size from parallel_config.
        # Falls back to 1 if the attribute is not present (older vLLM versions).
        self.model_config.cp_size = max(1, int(getattr(parallel_config, 'context_parallel_size', 1)))
        self.model_config.attn_cp_size = self.model_config.cp_size
        self.model_config.nnodes = max(1, int(getattr(parallel_config, 'nnodes', 1)))


        if self.model_config.pp_size > 1:
            from vllm.distributed.utils import get_pp_indices as vllm_get_pp_indices
            pp_start_layer, pp_end_layer = vllm_get_pp_indices(
                self.model_config.num_layers, pp_rank, self.model_config.pp_size
            )
        else:
            pp_start_layer = 0
            pp_end_layer = self.model_config.num_layers
        if self.model_config.use_mla:
            self.model_config.num_kv_heads = 1
        else:
            self.model_config.num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()

        self.model_config.instance_num = int(GLOBAL_CONFIG_FROM_ENV.instance_num)
        instance_id = int(GLOBAL_CONFIG_FROM_ENV.instance_id)

        self.model_config.master_host = os.getenv("FLEXKV_MASTER_HOST", "localhost")
        self.model_config.master_ports = tuple(
            os.getenv("FLEXKV_MASTER_PORTS", "5556,5557,5558").split(",")
        )

        rank_info = RankInfo(
            model_config=self.model_config,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            node_rank=node_rank,
            instance_id=instance_id,
            pp_start_layer=pp_start_layer,
            pp_end_layer=pp_end_layer,
            # vLLM sets LOCAL_RANK env var (= rank % gpus_per_node) before
            # launching each worker process.  Use it directly as the
            # authoritative physical device index.
            local_rank=int(os.environ.get('LOCAL_RANK', -1)),
        )

        update_default_config_from_user_config(rank_info, self.cache_config, self.user_config)
        self.server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        self.gpu_register_port = self.server_recv_port + "_gpu_register"

        logger.info(f"[FlexKV vllm] {self.model_config}, {rank_info}")

        # Freeze model_config — no further mutations allowed
        self.model_config.freeze()
        return rank_info

    def post_init_from_sglang_config(
        self,
        sglang_config,
        server_args,
        page_size: int = 64,
        tp_rank: int = 0,
        pp_rank: int = 0,
        dp_rank: int = 0,
        attn_cp_rank: int = 0,
    ) -> RankInfo:
        """Populate ``self.model_config`` / ``self.cache_config`` from a
        sglang ModelConfig + ServerArgs and return the per-worker
        ``RankInfo``.

        See :meth:`post_init_from_vllm_config` for the rationale behind
        returning ``RankInfo`` instead of writing it to ``self``.

        Args:
            sglang_config: sglang.srt.configs.model_config.ModelConfig-like object
            server_args: sglang ServerArgs — source of tp_size, dp_size,
                nnodes, node_rank, enable_dp_attention, attn_cp_size,
                kv_cache_dtype,
                dist_init_addr
            page_size: KV block size (tokens per block) used by sglang
            tp_rank: physical tensor parallel rank (runtime, from process group)
            pp_rank: pipeline parallel rank (runtime, from process group)
            dp_rank: logical DP shard index for this worker.
                - plain DP (``enable_dp_attention=False``): the regular
                  ``dp_rank`` passed to the scheduler process (0, 1, …).
                - DP Attention (``enable_dp_attention=True``): the
                  ``attn_dp_rank`` derived from ``tp_rank`` via
                  ``compute_dp_attention_world_info`` (already converted
                  by the sglang scheduler before calling this method).
                In both cases this value is stored directly as
                ``RankInfo.dp_rank`` and ``ModelConfig.dp_size`` is set
                to the true ``sglang_dp_size`` so that
                ``dp_client_id = instance_id * dp_size + dp_rank`` is
                globally unique across all DP shards and instances.
            attn_cp_rank: sglang's ``attn_cp_rank`` — attention-level context
                parallel rank within the CP group.
        """
        # sglang uses attn_cp_rank; map to FlexKV's generic cp_rank here so
        # the rest of the function and all downstream code stays framework-agnostic.
        cp_rank = attn_cp_rank
        # Extract parallelism params from server_args
        sglang_tp_size = int(server_args.tp_size)  # raw sglang tp_size (composite)
        pp_size = int(server_args.pp_size)
        sglang_dp_size = int(server_args.dp_size if server_args.dp_size is not None else 1)
        nnodes = server_args.nnodes
        node_rank = server_args.node_rank
        enable_dp_attention = bool(server_args.enable_dp_attention)
        attn_cp_size = int(getattr(server_args, 'attn_cp_size', 1))
        kv_cache_dtype = getattr(server_args, 'kv_cache_dtype', None)

        dp_rank = 0 if dp_rank is None else int(dp_rank)
        cp_rank = 0 if cp_rank is None else int(cp_rank)

        attn_dp_size = sglang_dp_size if enable_dp_attention else 1
        attn_tp_size = max(1, sglang_tp_size // (attn_dp_size * attn_cp_size))
        # attn_tp_rank: derived from physical tp_rank
        attn_tp_rank = int(tp_rank) % attn_tp_size

        # cache config: use page_size as tokens_per_block so that FlexKV's
        # CPU radix tree manages blocks at page granularity, ensuring that
        # hash generation, matching, insertion and eviction are all page-aligned.
        self.cache_config.tokens_per_block = page_size

        self.model_config.num_layers = int(getattr(sglang_config, "num_hidden_layers", 0))

        from sglang.srt.configs.model_config import AttentionArch
        use_mla = getattr(sglang_config, "attention_arch", None) == AttentionArch.MLA

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
                    per_rank = int(sglang_config.get_num_kv_heads(sglang_tp_size))
                    self.model_config.num_kv_heads = per_rank * sglang_tp_size
                except Exception:
                    self.model_config.num_kv_heads = int(getattr(sglang_config, "num_key_value_heads", 0))
            else:
                self.model_config.num_kv_heads = int(getattr(sglang_config, "num_key_value_heads", 0))
            self.model_config.head_size = int(getattr(sglang_config, "head_dim", 0))

        # Resolve KV cache dtype via unified priority logic.
        self._resolve_dtype(
            framework_dtype_str=kv_cache_dtype,
            fallback_dtype=getattr(sglang_config, "dtype", torch.bfloat16),
        )
        # NVFP4 packed head_size fold is only implemented/verified for vLLM.
        _warn_nvfp4_unsupported_framework(
            self.user_config.kv_cache_dtype
            if self.user_config.kv_cache_dtype is not None else kv_cache_dtype,
            framework="sglang",
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

        # Fill FlexKV parallel config.
        #
        # model_config.tp_size = attn_tp_size (innermost TP dimension).
        #   - plain DP:       attn_tp_size == sglang_tp_size  (dp is orthogonal)
        #   - DP Attention:   attn_tp_size == sglang_tp_size / (dp_size * cp_size)
        #
        # model_config.dp_size = sglang_dp_size in BOTH modes.
        #   - plain DP:       each dp shard is an independent process; dp_size
        #                     is the true number of DP shards so that dp_client_id
        #                     = instance_id * dp_size + dp_rank is globally unique.
        #   - DP Attention:   attn_dp_size == sglang_dp_size; same formula applies.
        #
        # FlexKV does not distinguish between "plain DP" and "DP Attention" —
        # both are represented as dp_size > 1 with each shard owning its own
        # KVManager (identified by dp_client_id).  The difference is only in
        # how sglang derives dp_rank (scheduler arg vs attn_dp_rank from tp_rank).
        self.model_config.tp_size = int(attn_tp_size)
        self.model_config.dp_size = int(sglang_dp_size)
        self.model_config.cp_size = int(attn_cp_size)
        self.model_config.attn_cp_size = int(attn_cp_size)
        self.model_config.pp_size = int(pp_size)

        if pp_size > 1:
            from sglang.srt.distributed.utils import get_pp_indices as sglang_get_pp_indices
            pp_start_layer, pp_end_layer = sglang_get_pp_indices(
                self.model_config.num_layers, pp_rank, self.model_config.pp_size
            )
        else:
            pp_start_layer = 0
            pp_end_layer = self.model_config.num_layers
        self.model_config.enable_dp_attention = bool(enable_dp_attention)
        self.model_config.nnodes = max(1, int(nnodes))
        _dist_init_addr = getattr(server_args, 'dist_init_addr', None)
        if _dist_init_addr and int(nnodes) > 1:
            self.model_config.master_host = _dist_init_addr.split(":")[0]
        else:
            self.model_config.master_host = os.getenv("FLEXKV_MASTER_HOST", "localhost")
        self.model_config.master_ports = tuple(
            os.getenv("FLEXKV_MASTER_PORTS", "5556,5557,5558").split(",")
        )

        self.model_config.instance_num = int(GLOBAL_CONFIG_FROM_ENV.instance_num)
        instance_id = int(GLOBAL_CONFIG_FROM_ENV.instance_id)

        rank_info = RankInfo(
            model_config=self.model_config,
            tp_rank=attn_tp_rank,   # sglang attn_tp_rank = tp_rank % attn_tp_size
            pp_rank=pp_rank,
            dp_rank=dp_rank,        # sglang attn_dp_rank (already computed by scheduler)
            cp_rank=cp_rank,        # sglang attn_cp_rank
            node_rank=node_rank,
            instance_id=instance_id,
            pp_start_layer=pp_start_layer,
            pp_end_layer=pp_end_layer,
            # Use torch.cuda.current_device()
            # which reflects the physical GPU index set by sglang's worker launcher
            # via torch.cuda.set_device(gpu_id) before this point.
            local_rank=torch.cuda.current_device(),
        )
        update_default_config_from_user_config(rank_info, self.cache_config, self.user_config)

        # ---- SWA host pool config (DeepSeek V4 all-SWA models) ----
        # DSv4 stores its sliding-window KV in a dedicated paged pool. FlexKV's
        # SWA page size is cache_config.tokens_per_block (hard-asserted as 256 by
        # sglang model_runner_kv_cache_mixin), NOT the HF attention
        # sliding_window (128). The per-token byte size is hard-asserted to
        # 584 (qk_nope_head_dim fp8 448 + qk_rope_head_dim bf16 128 + scale 8)
        # in DeepSeekV4SingleKVPool. Without this config, cache_config.swa
        # stays None -> the cache engine never builds an SWA pool -> a SWA-aware
        # get finds no SWA, so SWA KV is never matched/reused for this all-SWA model.
        is_dsv4 = bool(getattr(sglang_config, "is_deepseek_v4_arch", False))
        if is_dsv4:
            swa_page_size = self.cache_config.tokens_per_block
            # bytes_per_token_per_layer MUST match the GPU SWA buffer's *padded*
            # per-token stride, not the logical 584. DeepSeekV4SingleKVPool packs
            # each page as bytes_per_page_padded = ceil_div(swa_page * 584, 576) *
            # 576 = ceil_div(256*584, 576)*576 = 149760, so the effective per-token
            # width the connector registers as head_size is 149760 / 256 = 585
            # (584 logical + 256B/page alignment padding spread over the tokens).
            # The FlexKV host SWA pool must use the SAME 585 so H2D/D2H byte offsets
            # line up with the GPU buffer stride; a 584 host layout would shear the
            # bytes by 1/token/page and corrupt the SWA KV. See connector
            # _register_to_server_dsv4 (swa_layout head_size)
            if self.cache_config.swa is None:
                swa_bytes_per_token = _dsv4_swa_padded_bytes_per_token(
                    swa_page_size, 584)
                self.cache_config.swa = SWAPoolConfig(
                    enabled=True,
                    num_swa_layers=self.model_config.num_layers,
                    bytes_per_token_per_layer=swa_bytes_per_token,
                )
            # Gate the SWA data plane (byte movement) behind an env switch so it
            # can be turned off for A/B or if a byte-layout issue surfaces in
            # production, degrading cleanly to full-KV-only (all build_*_chain
            # become no-ops). Default ON for DSv4 (the SWA-native arch).
            self.cache_config.enable_swa_transfer = \
                _dsv4_swa_transfer_enabled_from_env()
            logger.info(
                f"[FlexKV sglang] Constructed SWAPoolConfig for DSv4: "
                f"swa_page_size={swa_page_size}, "
                f"num_swa_layers={self.model_config.num_layers}, "
                f"bytes_per_token_per_layer="
                f"{self.cache_config.swa.bytes_per_token_per_layer} (padded), "
                f"num_slots={self.cache_config.swa.num_slots}, "
                f"enable_swa_transfer={self.cache_config.enable_swa_transfer}"
            )

        logger.info(f"[FlexKV sglang] {self.model_config}, {rank_info}")

        # Freeze model_config — no further mutations allowed
        self.model_config.freeze()
        return rank_info

    def post_init_from_trt_config(
        self,
        config,
    ) -> RankInfo:
        mapping = config.mapping
        tp_rank = mapping.tp_rank
        node_rank = mapping.node_rank
        self.cache_config.tokens_per_block = config.tokens_per_block
        # Resolve KV cache dtype via unified priority logic.
        _pytorch_backend = getattr(config, 'pytorch_backend_config', None)
        trt_dtype_str = getattr(_pytorch_backend, 'kv_cache_dtype', 'auto') if _pytorch_backend else 'auto'
        self._resolve_dtype(
            framework_dtype_str=trt_dtype_str if isinstance(trt_dtype_str, str) else None,
            fallback_dtype=torch.bfloat16,
        )
        # NVFP4 packed head_size fold is only implemented/verified for vLLM.
        _warn_nvfp4_unsupported_framework(
            self.user_config.kv_cache_dtype
            if self.user_config.kv_cache_dtype is not None
            else (trt_dtype_str if isinstance(trt_dtype_str, str) else None),
            framework="trtllm",
        )

        # Set model config (parallel configs part).
        enable_attention_dp = bool(getattr(mapping, 'enable_attention_dp', False))
        if enable_attention_dp:
            self.model_config.tp_size = 1
            self.model_config.dp_size = int(mapping.tp_size)
            dp_rank = int(mapping.tp_rank)
        else:
            self.model_config.tp_size = int(mapping.tp_size)
            self.model_config.dp_size = int(getattr(mapping, 'dp_size', 1))
            dp_rank = 0
        self.model_config.enable_dp_attention = enable_attention_dp
        self.model_config.pp_size = int(getattr(mapping, 'pp_size', 1))
        pp_rank = int(getattr(mapping, 'pp_rank', 0))
        # TRT-LLM CP size: read from mapping.cp_size (available in TRT-LLM >= 0.15).
        # Falls back to 1 if not present.
        self.model_config.cp_size = max(1, int(getattr(mapping, 'cp_size', 1)))
        self.model_config.attn_cp_size = self.model_config.cp_size
        # TRT-LLM CP rank: read from mapping.cp_rank.
        cp_rank = int(getattr(mapping, 'cp_rank', 0))

        self.model_config.nnodes = max(1, getattr(mapping, 'nnodes', 1))
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

        if self.model_config.pp_size > 1:
            layers_range = mapping.pp_layers(self.model_config.num_layers)
            pp_start_layer = layers_range[0]
            pp_end_layer = layers_range[-1] + 1
        else:
            pp_start_layer = 0
            pp_end_layer = self.model_config.num_layers

        self.model_config.instance_num = int(GLOBAL_CONFIG_FROM_ENV.instance_num)
        instance_id = int(GLOBAL_CONFIG_FROM_ENV.instance_id)


        self.model_config.use_trtllm_subprocess = True
        self.model_config.trtllm_subprocess_host = os.getenv(
            "FLEXKV_TRT_SUBPROCESS_HOST", "localhost"
        )
        self.model_config.trtllm_subprocess_ports = tuple(
            os.getenv("FLEXKV_TRT_SUBPROCESS_PORTS", "6667,6668,6669").split(",")
        )
        # Multi-node master endpoint (used when nnodes > 1).
        self.model_config.master_host = os.getenv("FLEXKV_MASTER_HOST", "localhost")
        self.model_config.master_ports = tuple(
            os.getenv("FLEXKV_MASTER_PORTS", "5556,5557,5558").split(",")
        )

        rank_info = RankInfo(
            model_config=self.model_config,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            cp_rank=cp_rank,
            node_rank=node_rank,
            instance_id=instance_id,
            pp_start_layer=pp_start_layer,
            pp_end_layer=pp_end_layer,
            local_rank=mapping.local_rank,
        )

        # Update cache config with user config after model config is initialized
        update_default_config_from_user_config(rank_info, self.cache_config, self.user_config)

        logger.info(f"[FlexKV TRT-LLM] {self.model_config}, {rank_info}")

        # Freeze model_config — no further mutations allowed
        self.model_config.freeze()
        return rank_info
