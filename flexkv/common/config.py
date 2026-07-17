import os
import json
import yaml
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional, List, Tuple, Union, Dict, Any
from argparse import Namespace
import copy

import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.debug import flexkv_logger


@dataclass
class IndexerCacheConfig:
    """Indexer-specific cache configuration, embedded inside CacheConfig."""
    # Indexer head layout
    head_size: int = 0          # qk_rope_head_dim for DSA/NSA models
    num_kv_heads: int = 1       # typically 1 for MLA-style indexer
    dtype: torch.dtype = torch.uint8  # indexer storage dtype (fp8 quantized)


@dataclass
class ModelConfig:
    num_layers: int = 1
    num_kv_heads: int = 1
    head_size: int = 1
    use_mla: bool = False
    dtype: torch.dtype = torch.bfloat16

    # ------------------------------------------------------------------
    # Parallelism sizes (global, identical for every rank)
    # ------------------------------------------------------------------
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1

    # ------------------------------------------------------------------
    # Attention-level parallel configs
    # ------------------------------------------------------------------
    # enable_dp_attention: whether DP-attention is enabled (sglang
    # ``--enable-dp-attention`` or TRT-LLM ``enable_attention_dp``).
    # When True, the physical TP group is split into
    # attn_tp × attn_cp × attn_dp.
    enable_dp_attention: bool = False

    # attn_cp_size: context-parallel size (global).
    attn_cp_size: int = 1

    # ------------------------------------------------------------------
    # Topology configs (global)
    # ------------------------------------------------------------------
    # nnodes: number of physical machines spanned by one replica
    nnodes: int = 1

    # Multi-node bootstrap: master node's IP for TransferManager rendezvous.
    # Bare host (no scheme); zmq sockets prepend ``tcp://`` at connect time.
    # Set this from the framework's own launch config (e.g. sglang's
    # ``--dist-init-addr``, TRT-LLM's launch script) so runtime layers
    # (TransferManager / KVTaskManager) never need to read env vars.
    master_host: str = "localhost"

    # Master endpoint ports (command, result, query). Set via integration
    # adapter when the launch script exposes a different port triple.
    master_ports: Tuple[str, str, str] = ("5556", "5557", "5558")

    # Whether KVTaskManager should run TransferManagerOnRemote in a
    # subprocess (currently only used by TRT-LLM to avoid MPI conflicts).
    # Set by the TRT-LLM adapter; default ``False`` for sglang/vllm.
    use_trtllm_subprocess: bool = False

    # Endpoint of that TRT-LLM subprocess TransferManagerOnRemote.
    # Only consulted when ``use_trtllm_subprocess`` is True.
    trtllm_subprocess_host: str = "localhost"
    trtllm_subprocess_ports: Tuple[str, str, str] = ("6667", "6668", "6669")

    # ------------------------------------------------------------------
    # Multi-instance deployment
    # ------------------------------------------------------------------
    instance_num: int = 1

    # ------------------------------------------------------------------
    # Freeze mechanism: after post_init, ModelConfig must not be mutated
    # ------------------------------------------------------------------
    _frozen: bool = field(default=False, init=False, repr=False)

    def freeze(self) -> None:
        """Lock the config so that any subsequent __setattr__ raises an error."""
        # ---- Topology validation ----
        if self.total_gpus % self.nnodes != 0:
            raise ValueError(
                f"[ModelConfig] cannot derive gpus_per_node: "
                f"total_gpus={self.total_gpus} not divisible by nnodes={self.nnodes}"
            )
        if self.nnodes_per_tp_group > 2:
            raise ValueError(
                f"[ModelConfig] only support 2-nodes TP for now, but got "
                f"nnodes_per_tp_group={self.nnodes_per_tp_group} "
                f"(tp_size={self.tp_size}, gpus_per_node={self.gpus_per_node})"
            )
        if self.tp_size % self.nnodes_per_tp_group != 0:
            raise ValueError(
                f"[ModelConfig] tp_size={self.tp_size} not divisible by "
                f"nnodes_per_tp_group={self.nnodes_per_tp_group}"
            )
        if self.instance_num < 1:
            raise ValueError(
                f"[ModelConfig] instance_num must be >= 1, got {self.instance_num}"
            )

        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value) -> None:
        if name == '_frozen':
            return object.__setattr__(self, name, value)
        if getattr(self, '_frozen', False):
            raise AttributeError(
                f"ModelConfig is frozen — cannot set '{name}'. "
                f"All primitive fields must be set during post_init_from_*(), "
                f"after which freeze() is called.  Derived fields (attn_tp_size, "
                f"tp_size_per_node) are @property "
                f"and cannot be set at all."
            )
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Derived topology properties
    # ------------------------------------------------------------------
    @property
    def total_gpus(self) -> int:
        """Total GPUs across all nodes for one FlexKV instance."""
        return self.dp_size * self.tp_size * self.pp_size

    @property
    def total_clients(self) -> int:
        """Total number of DPClient endpoints across all instances."""
        return self.instance_num * self.dp_size

    @property
    def gpus_per_node(self) -> int:
        """Total GPUs on this node (across all DP, PP stages and TP groups)."""
        return self.total_gpus // self.nnodes

    @property
    def nnodes_per_pp_rank(self) -> int:
        """Number of nodes spanned by one PP stage."""
        return max(self.nnodes // self.pp_size, 1)

    @property
    def nnodes_per_tp_group(self) -> int:
        """Number of nodes spanned by one TP group."""
        return self.nnodes_per_pp_rank

    @property
    def tp_size_per_node(self) -> int:
        """Number of TP ranks on this node within one TP group."""
        return self.tp_size // self.nnodes_per_tp_group

    @property
    def attn_dp_size(self) -> int:
        """Attention-level DP size (= dp_size when enable_dp_attention else 1)."""
        return max(1, self.dp_size) if self.enable_dp_attention else 1

    @property
    def attn_tp_size(self) -> int:
        """Attention-level TP size derived from tp / attn_dp / attn_cp."""
        attn_dp = self.attn_dp_size
        cp = max(1, self.attn_cp_size)
        return max(1, max(1, self.tp_size) // (attn_dp * cp))

    @property
    def attn_tp_size_per_node(self) -> int:
        """Attention-level TP size per node."""
        return self.attn_tp_size // self.nnodes_per_tp_group

    @property
    def attn_cp_size_per_node(self) -> int:
        """Attention-level CP size on this node for a single pp stage. """
        return max(1, self.attn_cp_size // self.nnodes_per_pp_rank)

    @property
    def effective_tp_size(self) -> int:
        """Effective tp-group size used for *data-plane* CPU slicing."""
        return max(1, self.attn_tp_size) * max(1, self.attn_cp_size)

    @property
    def effective_tp_size_per_node(self) -> int:
        """Per-node counterpart of :pyattr:`effective_tp_size`."""
        return self.attn_tp_size_per_node * self.attn_cp_size_per_node

    @property
    def num_kv_heads_per_node(self) -> int:
        """Number of KV heads visible to a single node."""
        if self.use_mla:
            return self.num_kv_heads
        return self.num_kv_heads * self.tp_size_per_node // max(1, self.attn_tp_size)

    @property
    def kv_dim(self) -> int:
        """KV dimension: 1 for MLA (no head split), 2 for standard (head split)."""
        return 1 if self.use_mla else 2

    @property
    def bytes_per_token_per_layer(self) -> int:
        """Raw byte footprint of a single (layer, token) KV slot."""
        return self.num_kv_heads * self.head_size * self.kv_dim * self.dtype.itemsize

    @property
    def token_size_in_bytes(self) -> int:
        """Whole-model token footprint (bytes) — all layers combined."""
        return self.num_layers * self.bytes_per_token_per_layer

    def __str__(self) -> str:
        return (
            f"ModelConfig(num_layers={self.num_layers}, num_kv_heads={self.num_kv_heads}"
            f", head_size={self.head_size}, use_mla={self.use_mla}"
            f", dtype={self.dtype}"
            f", tp_size={self.tp_size}, pp_size={self.pp_size}, dp_size={self.dp_size}"
            f", attn_cp_size={self.attn_cp_size}"
            f", nnodes={self.nnodes}, master_host={self.master_host!r}"
            f", instance_num={self.instance_num}"
        )


@dataclass(frozen=True)
class RankInfo:
    model_config: ModelConfig
    tp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0
    attn_cp_rank: int = 0
    node_rank: int = 0
    instance_id: int = 0
    pp_start_layer: int = 0
    pp_end_layer: int = -1
    @property
    def tp_rank_per_node(self) -> int:
        """TP rank index within the local node (within one TP group)."""
        return self.tp_rank % self.model_config.tp_size_per_node

    @property
    def dp_client_id(self) -> int:
        """Flat DP route label: unique int across all instances.

        Equals ``instance_id * dp_size + dp_rank``. All transfer-engine
        routing (worker maps, NVTX labels, socket paths, server-side
        client registration) keys on this single int so the legacy
        ``(instance_id, dp_rank)`` tuple (DPRoutingKey) is no longer
        needed.
        """
        return self.instance_id * self.model_config.dp_size + self.dp_rank

    @property
    def attn_tp_rank(self) -> int:
        """Attention-level TP rank derived from tp_rank / attn_tp_size."""
        return self.tp_rank % max(1, self.model_config.attn_tp_size)

    @property
    def effective_tp_rank(self) -> int:
        """Effective tp-rank in the *data-plane* segmentation space."""
        if self.model_config.use_mla:
            return self.attn_tp_rank
        attn_tp_size = max(1, self.model_config.attn_tp_size)
        return self.attn_cp_rank * attn_tp_size + self.attn_tp_rank

    @property
    def pp_size_per_node(self) -> int:
        """Number of PP stages co-located on a single node."""
        model_config = self.model_config
        return max(model_config.pp_size // model_config.nnodes, 1)

    @property
    def pp_rank_per_node(self) -> int:
        """This rank's PP index *within* its node."""
        return self.pp_rank % self.pp_size_per_node

    @property
    def dp_size_per_node(self) -> int:
        """Number of DP replicas co-located on a single node."""
        model_config = self.model_config
        return model_config.gpus_per_node // (self.pp_size_per_node * model_config.tp_size_per_node)

    @property
    def dp_rank_per_node(self) -> int:
        """This rank's DP index *within* its node (non-DP-attention layout)."""
        return self.dp_rank % self.dp_size_per_node

    @property
    def local_rank(self) -> int:
        model_config = self.model_config
        if model_config.enable_dp_attention:
            return self.pp_rank_per_node * model_config.tp_size_per_node + self.tp_rank_per_node
        return (self.dp_rank_per_node * self.pp_size_per_node + self.pp_rank_per_node) \
               * model_config.tp_size_per_node + self.tp_rank_per_node

    @property
    def num_layers_per_pp_stage(self) -> int:
        """Number of layers managed by this PP stage."""
        end = self.pp_end_layer if self.pp_end_layer >= 0 else self.model_config.num_layers
        return end - self.pp_start_layer

    @property
    def token_size_in_bytes_per_pp_stage(self) -> int:
        """Per-pp-stage token footprint (bytes) — rank-exact."""
        return (self.num_layers_per_pp_stage
                * self.model_config.bytes_per_token_per_layer)

    def __str__(self) -> str:
        """Human-readable summary of this rank including derived quantities.

        Equivalent to the retired ``FlexKVContext.describe_rank`` output
        (kept stable so log-grep patterns keep working).
        """
        return (
            f"RankInfo(tp_rank={self.tp_rank}, pp_rank={self.pp_rank}"
            f", dp_rank={self.dp_rank}, attn_cp_rank={self.attn_cp_rank}"
            f", node_rank={self.node_rank}, instance_id={self.instance_id}"
        )


@dataclass
class CacheConfig:
    tokens_per_block: int = 16
    eviction_policy: str = "lru"
    enable_cpu: bool = True
    enable_ssd: bool = False
    enable_gds: bool = False # Requires enable_ssd=True
    # When True with enable_gds, GPU<->SSD uses NIXL (GDS_MT) instead of cuFile GDS worker.
    enable_nixl: bool = False
    # Optional plugin dict for NixlAgentSession (see nixl README); only used if enable_nixl.
    nixl_extra_config: Optional[Dict[str, Any]] = None
    enable_remote: bool = False # used for indicating whether the 3rd-party remote storage is enabled
                                # has nothing to do with whether the p2p_cpu and p2p_ssd are supported
    enable_kv_sharing: bool = False # pcfs_sharing or p2p_cpu or p2p_ssd or p2p_3rd_remote
    enable_p2p_cpu: bool = False
    enable_p2p_ssd: bool = False
    enable_3rd_remote: bool = False

    distributed_node_id: int = -1 # only used when distributed cpu/ssd and only can be set when redis_meta_client initialized
    num_tmp_cpu_blocks: int = 500 # only used when distributed ssd p2p, it controls the number blocks of temp cpu buffer which used for copy data from ssd to cpu
    # When True, the main CPU KV cache is allocated from Linux HugePages via
    # ``mmap(MAP_HUGETLB)`` instead of regular CPU memory. Requires pre-reserved
    # huge pages on the host (see ``/proc/sys/vm/nr_hugepages``). Falls back
    # silently if allocation fails.
    use_hugepage_cpu_buffer: bool = False
    # When True, the temporary SSD->CPU staging buffer (used by PEER2CPUTransferWorker
    # under enable_p2p_ssd) is allocated from Linux HugePages via ``mmap(MAP_HUGETLB)``
    # instead of a pinned ``torch.empty``. Requires pre-reserved huge pages on the host
    # (see ``/proc/sys/vm/nr_hugepages``). Falls back silently if allocation fails.
    use_hugepage_tmp_buffer: bool = False
    hugepage_size_bytes: int = 2 * 1024 * 1024  # 2 MiB by default; set to 1<<30 for 1GiB


    # Indexer configuration
    indexer: Optional[IndexerCacheConfig] = None

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
    # TTL (seconds) for node:<id> key in Redis. Active nodes renew via heartbeat.
    # If a process crashes, the key auto-expires after this period.
    node_ttl_seconds: int = 30

    # Mooncake transfer engine config path (serialized via pickle to survive spawn subprocesses)
    mooncake_config_path: Optional[str] = None

    def __post_init__(self):
        self.enable_kv_sharing = self.enable_p2p_cpu or \
            self.enable_p2p_ssd or self.enable_3rd_remote
        self.enable_remote = self.enable_3rd_remote

    def __str__(self) -> str:
        return (
            f"CacheConfig(tokens_per_block={self.tokens_per_block}"
            f", enable_cpu={self.enable_cpu}, enable_ssd={self.enable_ssd}"
            f", enable_gds={self.enable_gds}, enable_remote={self.enable_remote}"
            f", enable_kv_sharing={self.enable_kv_sharing}"
            f", enable_p2p_cpu={self.enable_p2p_cpu}"
            f", enable_p2p_ssd={self.enable_p2p_ssd}"
            f", enable_3rd_remote={self.enable_3rd_remote}"
            f", num_cpu_blocks={self.num_cpu_blocks}"
            f", num_ssd_blocks={self.num_ssd_blocks})"
        )

GLOBAL_CONFIG_FROM_ENV: Namespace = Namespace(
    # Multi-instance configuration
    instance_num=int(os.getenv('FLEXKV_INSTANCE_NUM', 1)),
    instance_id=int(os.getenv('FLEXKV_INSTANCE_ID', 0)),

    # Metrics configuration
    ## Enable/disable metrics collection and HTTP server (shared by C++ and Python)
    enable_metrics=bool(int(os.getenv('FLEXKV_ENABLE_METRICS', 0))),
    ## Port for C++ metrics HTTP server (default: 8081)
    cpp_metrics_port=int(os.getenv('FLEXKV_CPP_METRICS_PORT', 8081)),
    ## Port for Python metrics HTTP server (default: 8080)
    py_metrics_port=int(os.getenv('FLEXKV_PY_METRICS_PORT', 8080)),

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
    transfer_num_cta_h2d=int(os.getenv('FLEXKV_TRANSFER_NUM_CTA_H2D', 4)),
    transfer_num_cta_d2h=int(os.getenv('FLEXKV_TRANSFER_NUM_CTA_D2H', 4)),

    transfer_segment_threshold=int(os.getenv('FLEXKV_TRANSFER_SEGMENT_THRESHOLD', 8)),
    transfer_path_opt=bool(int(os.getenv('FLEXKV_TRANSFER_PATH_OPT', 1))),
    enable_ce_memcpy2d=bool(int(os.getenv('FLEXKV_ENABLE_CE_MEMCPY2D', 1))),

    iouring_entries=int(os.getenv('FLEXKV_IOURING_ENTRIES', 512)),
    iouring_flags=int(os.getenv('FLEXKV_IOURING_FLAGS', 0)),

    max_file_size_gb=float(os.getenv('FLEXKV_MAX_FILE_SIZE_GB', -1)),  # -1 means no limit

    evict_ratio=float(os.getenv('FLEXKV_EVICT_RATIO', 0)),
    evict_start_threshold=float(os.getenv('FLEXKV_EVICT_START_THRESHOLD', 1.0)),
    hit_reward_seconds=int(os.getenv('FLEXKV_HIT_REWARD_SECONDS', 0)),
    eviction_policy=os.getenv('FLEXKV_EVICTION_POLICY', 'lru'),
    slru_protected_threshold=int(os.getenv('FLEXKV_SLRU_PROTECTED_THRESHOLD', 2)),

    enable_mps=bool(int(os.getenv('FLEXKV_ENABLE_MPS', 1))),

    enable_trace=bool(int(os.getenv('FLEXKV_ENABLE_TRACE', 0))),
    trace_file_path=os.getenv('FLEXKV_TRACE_FILE_PATH', './flexkv_trace.log'),
    trace_max_file_size_mb=int(os.getenv('FLEXKV_TRACE_MAX_FILE_SIZE_MB', 100)),
    trace_max_files=int(os.getenv('FLEXKV_TRACE_MAX_FILES', 5)),
    trace_flush_interval_ms=int(os.getenv('FLEXKV_TRACE_FLUSH_INTERVAL_MS', 1000)),

    lt_pool_initial_capacity=int(os.getenv('FLEXKV_LT_POOL_INITIAL_CAPACITY', 10000000)),
    refresh_batch_size=int(os.getenv('FLEXKV_REFRESH_BATCH_SIZE', 256)),
    rebuild_interval_ms=int(os.getenv('FLEXKV_REBUILD_INTERVAL_MS', 2000)),
    idle_sleep_ms=int(os.getenv('FLEXKV_IDLE_SLEEP_MS', 10)),
    lease_ttl_ms=int(os.getenv('FLEXKV_LEASE_TTL_MS', 30000)),
    safety_ttl_ms=int(os.getenv('FLEXKV_SAFETY_TTL_MS', 100)),
    renew_lease_ms=int(os.getenv('FLEXKV_RENEW_LEASE_MS', 4000)),

    nvcomp_batch_size=int(os.getenv('FLEXKV_NVCOMP_BATCH_SIZE', '0')),  # 0 = auto

    mla_d2h_mode=os.getenv('FLEXKV_MLA_D2H_MODE', 'sharded'),

    layerwise_notify_mode=os.getenv('FLEXKV_LAYERWISE_NOTIFY_MODE', 'hostfunc'),
)

@dataclass
class UserConfig:
    cpu_cache_gb: int = 16
    ssd_cache_gb: int = 0  # 0 means disable ssd
    ssd_cache_dir: Union[str, List[str]] = "./ssd_cache"
    enable_gds: bool = False
    enable_nixl: bool = False
    use_hugepage_cpu_buffer: bool = False
    use_hugepage_tmp_buffer: bool = False
    hugepage_size_bytes: int = 2 * 1024 * 1024
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
    node_ttl_seconds: Optional[int] = None
    kv_cache_dtype: Optional[str] = None  # Override kv_cache_dtype when TRT config uses "auto". Supported values: "fp8", "float8", "e4m3", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32", "nvfp4" (packed fp4+fp8-scale, stored as uint8)

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
        enable_nixl=bool(int(os.getenv('FLEXKV_ENABLE_NIXL', 0))),
        use_hugepage_cpu_buffer=bool(int(os.getenv('FLEXKV_USE_HUGEPAGE_CPU_BUFFER', 0))),
        use_hugepage_tmp_buffer=bool(int(os.getenv('FLEXKV_USE_HUGEPAGE_TMP_BUFFER', 0))),
        hugepage_size_bytes=int(os.getenv('FLEXKV_HUGEPAGE_SIZE_BYTES', 2 * 1024 * 1024)),
        kv_cache_dtype=os.getenv('FLEXKV_KV_CACHE_DTYPE', None),
    )

def convert_to_block_num(size_in_GB: float, block_size_in_bytes: int) -> int:
    return int(size_in_GB * 1024 * 1024 * 1024 / block_size_in_bytes)

def update_default_config_from_user_config(rank_info: RankInfo,
                                           cache_config: CacheConfig,
                                           user_config: UserConfig) -> None:
    main_block_size_in_bytes = (
        rank_info.token_size_in_bytes_per_pp_stage * cache_config.tokens_per_block
    )
    indexer_block_size_in_bytes = 0
    if cache_config.indexer is not None:
        indexer_cfg = cache_config.indexer
        # Indexer is MLA-style (single shared head set, no TP head split).
        # head_size already includes page_stride_size (main tokens_per_block ×
        # index_head_dim), so per-block bytes = num_kv_heads × head_size ×
        # dtype.itemsize — no additional tokens_per_block multiplication.
        indexer_bytes_per_token_per_layer = (
            indexer_cfg.num_kv_heads
            * indexer_cfg.head_size
            * indexer_cfg.dtype.itemsize
        )
        indexer_block_size_in_bytes = (
            rank_info.num_layers_per_pp_stage
            * indexer_bytes_per_token_per_layer
        )
    block_size_in_bytes = main_block_size_in_bytes + indexer_block_size_in_bytes

    assert user_config.cpu_cache_gb > 0
    assert user_config.ssd_cache_gb >= 0

    # MLA all_write mode: each logical KV block occupies N× physical space
    # on CPU/SSD (N GPUs each write a complete KV copy to distinct block slots).
    # To keep the physical memory budget (cpu_cache_gb / ssd_cache_gb) unchanged,
    # the logical block capacity must be divided by N.
    # This mirrors the C++ offset logic in tp_transfer_thread_group.cpp where
    # GPU i writes to cpu_startoff = i * chunk_size, requiring N slots per logical block.
    model_config = rank_info.model_config
    mla_d2h_mode = GLOBAL_CONFIG_FROM_ENV.mla_d2h_mode
    capacity_divisor = 1
    if model_config.use_mla and mla_d2h_mode == "all_write":
        num_gpus_per_node = model_config.effective_tp_size_per_node
        if num_gpus_per_node > 1:
            capacity_divisor = num_gpus_per_node
            flexkv_logger.info(
                f"[config] MLA all_write mode: logical cpu/ssd capacity "
                f"÷{num_gpus_per_node} (each block occupies {num_gpus_per_node}× "
                f"physical space, total memory budget unchanged)"
            )

    cache_config.num_cpu_blocks = convert_to_block_num(user_config.cpu_cache_gb, block_size_in_bytes) // capacity_divisor
    cache_config.num_ssd_blocks = convert_to_block_num(user_config.ssd_cache_gb, block_size_in_bytes) // capacity_divisor

    if cache_config.indexer is not None:
        flexkv_logger.info(
            f"[CacheConfig] GB->blocks conversion (with indexer): "
            f"main_block_size={main_block_size_in_bytes} B, "
            f"indexer_block_size={indexer_block_size_in_bytes} B, "
            f"total_block_size={block_size_in_bytes} B"
        )
    else:
        flexkv_logger.info(
            f"[CacheConfig] GB->blocks conversion: "
            f"block_size={block_size_in_bytes} B"
        )

    cache_config.ssd_cache_dir = user_config.ssd_cache_dir
    cache_config.enable_ssd = user_config.ssd_cache_gb > 0
    cache_config.enable_gds = user_config.enable_gds
    cache_config.enable_nixl = user_config.enable_nixl
    cache_config.use_hugepage_cpu_buffer = user_config.use_hugepage_cpu_buffer
    cache_config.use_hugepage_tmp_buffer = user_config.use_hugepage_tmp_buffer
    cache_config.hugepage_size_bytes = user_config.hugepage_size_bytes
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

    if not cache_config.enable_cpu:
        raise ValueError("enable_cpu must be True")
    if cache_config.enable_remote and not cache_config.enable_ssd:
        raise ValueError("enable_ssd must be True if enable_remote is True")
    if not cache_config.enable_cpu and not cache_config.enable_gds:
        raise ValueError("enable_gds must be True if enable_cpu is False")
    if cache_config.enable_gds and not cache_config.enable_ssd:
        raise ValueError("enable_ssd must be True if enable_gds is True")
    if cache_config.enable_kv_sharing and cache_config.enable_gds:
        raise ValueError(
            "enable_kv_sharing and enable_gds cannot be used at the same time"
        )

    if cache_config.enable_remote:
        if cache_config.remote_cache_path is None:
            if cache_config.remote_file_prefix is None:
                raise ValueError(
                    "remote_file_prefix must be provided when remote_cache_path is None"
                )
            if (cache_config.remote_file_num is None
                    or cache_config.remote_file_num <= 0):
                raise ValueError("remote_file_num must be a positive integer")
            cache_config.remote_cache_path = [
                f"{cache_config.remote_file_prefix}_{i}"
                for i in range(cache_config.remote_file_num)
            ]

        if cache_config.remote_cache_size_mode not in ("block_num", "file_size"):
            raise ValueError(
                f"remote_cache_size_mode must be 'block_num' or 'file_size', "
                f"got {cache_config.remote_cache_size_mode!r}"
            )

        if cache_config.remote_cache_size_mode == "file_size":
            if cache_config.remote_file_size is None:
                raise ValueError(
                    "remote_file_size must be set when remote_cache_size_mode == 'file_size'"
                )
            if (cache_config.remote_file_num is None
                    or cache_config.remote_file_num <= 0):
                raise ValueError("remote_file_num must be a positive integer")
            cache_config.num_remote_blocks = (
                cache_config.remote_file_size // block_size_in_bytes
                * cache_config.remote_file_num
            )
            flexkv_logger.info(
                f"num_remote_blocks derived from remote_file_size "
                f"(per-pp-stage, num_layers_per_pp_stage="
                f"{rank_info.num_layers_per_pp_stage}): "
                f"remote_file_size={cache_config.remote_file_size}, "
                f"remote_file_num={cache_config.remote_file_num}, "
                f"block_size_in_bytes={block_size_in_bytes} "
                f"-> num_remote_blocks={cache_config.num_remote_blocks}"
            )

        if (cache_config.num_remote_blocks is None
                or cache_config.num_remote_blocks <= 0):
            raise ValueError(
                "num_remote_blocks must be a positive integer "
                "(file_size mode: derived above from remote_file_size; "
                "block_num mode: set it explicitly)"
            )

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
    if user_config.node_ttl_seconds is not None:
        cache_config.node_ttl_seconds = user_config.node_ttl_seconds

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
