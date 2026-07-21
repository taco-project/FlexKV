import os
import json
import yaml
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from functools import cached_property
from typing import Optional, List, Tuple, Union, Dict, Any
from argparse import Namespace
import copy

import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.debug import flexkv_logger


@dataclass
class LayerGroupSpec:
    """One group of layers sharing the same KV cache shape.

    ``layer_indices[k]`` is the *original layer id* (index into the full
    ``model_config.num_layers`` range) that this group's k-th local layer
    (local_id = k) maps to.  Multiple groups MAY share the same original
    layer id — this expresses heterogeneous KV at one transformer block
    (e.g. DSv4: main KV bf16 and indexer uint8 both attached to the same
    layer).

    Invariants (enforced by ``ModelConfig._validate_layer_groups``):

    * ``num_layers == len(layer_indices)``
    * ``layer_indices`` has no internal duplicates
    * every element of ``layer_indices`` is in ``[0, model_config.num_layers)``
    * original layers may be omitted when they have no cached state, and may
      appear in multiple groups when they have multiple cache members
    """
    num_layers: int
    num_kv_heads: int
    head_size: int
    layer_indices: List[int]
    # Per-group storage dtype. None = inherit ModelConfig.dtype.
    # Indexer groups use a different dtype (e.g. fp8/uint8) than main KV (bf16).
    dtype: Optional[torch.dtype] = None
    # Per-group KV compression along the tokens_per_block dimension. The CPU/SSD
    # block stores ``tokens_per_block // compress_ratio`` tokens worth of data
    # for this group (the GPU tensor sglang allocates is already compressed to
    # the same shrunk shape). ``1`` = uncompressed (legacy behavior). Used by
    # DSv4-style models where different layer roles compress at different ratios
    # (e.g. CSA at 4x, HCA at 128x). ``tokens_per_block % compress_ratio == 0``
    # is enforced when the KVCacheLayout is built.
    compress_ratio: int = 1


@dataclass(frozen=True)
class LayerMemberMap:
    """Dense mapping from original layer id -> tuple of (group_idx, local_layer_id).

    ``members[i]`` is the tuple of ``(group_idx, local_layer_id)`` pairs for
    original layer ``i``, ordered by ascending ``group_idx`` (main KV before
    auxiliary groups like indexer).

    Examples:

      Single group (uniform model): every layer has 1 member.
        members = (((0, 0),), ((0, 1),), ..., ((0, N-1),))

      DSv4 (main + indexer share every layer): every layer has 2 members.
        members = (((0, 0), (1, 0)), ((0, 1), (1, 1)), ...)

      Alternating partition: each layer belongs to exactly one group.
        members = (((0, 0),), ((1, 0),), ((0, 1),), ((1, 1),), ...)
    """
    members: Tuple[Tuple[Tuple[int, int], ...], ...]

    @property
    def num_original_layers(self) -> int:
        return len(self.members)

    @property
    def total_members(self) -> int:
        return sum(len(m) for m in self.members)

    def members_of(self, original_layer_id: int) -> Tuple[Tuple[int, int], ...]:
        """Return ((group_idx, local_id), ...) for one original layer."""
        return self.members[original_layer_id]


def build_layer_member_map(
    layer_groups: List[LayerGroupSpec],
    num_original_layers: int,
) -> LayerMemberMap:
    """Construct a ``LayerMemberMap`` from ``layer_groups``.

    Members within one original layer are ordered by ascending ``group_idx``
    (i.e., the order in which groups appear in ``model_config.layer_groups``).
    By convention main KV should be group 0 so that on the stream it always
    fires before any auxiliary group (e.g., indexer).
    """
    buckets: List[List[Tuple[int, int]]] = [[] for _ in range(num_original_layers)]
    for gi, g in enumerate(layer_groups):
        for local_id, orig in enumerate(g.layer_indices):
            buckets[orig].append((gi, local_id))
    for b in buckets:
        b.sort(key=lambda x: x[0])
    return LayerMemberMap(members=tuple(tuple(b) for b in buckets))


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
    # Compatibility flag retained for framework adapters. ``tp_size`` is the
    # normalized attention TP size and ``cp_size`` is represented separately.
    enable_dp_attention: bool = False
    # Compatibility mirror of cp_size for existing SGLang connector callers.
    attn_cp_size: int = 1
    # cp_size: context-parallel size (global), default 1.
    cp_size: int = 1

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
    # Heterogeneous KV cache layers (including Indexer-as-group)
    # ------------------------------------------------------------------
    # When None, all layers share the same (num_kv_heads, head_size, dtype).
    # When set, each group carries its own shape (and optionally its own dtype),
    # and token_size_in_bytes/num_cpu_blocks are computed by summing across groups.
    layer_groups: Optional[List[LayerGroupSpec]] = None

    # ------------------------------------------------------------------
    # Freeze mechanism: after post_init, ModelConfig must not be mutated
    # ------------------------------------------------------------------
    _frozen: bool = field(default=False, init=False, repr=False)

    def freeze(self) -> None:
        """Lock the config so that any subsequent __setattr__ raises an error."""
        if self.cp_size == 1 and self.attn_cp_size != 1:
            self.cp_size = self.attn_cp_size
        elif self.attn_cp_size == 1 and self.cp_size != 1:
            self.attn_cp_size = self.cp_size
        elif self.cp_size != self.attn_cp_size:
            raise ValueError(
                f"[ModelConfig] cp_size={self.cp_size} and "
                f"attn_cp_size={self.attn_cp_size} disagree"
            )
        # ---- Topology validation ----
        if self.total_gpus % self.nnodes != 0:
            raise ValueError(
                f"[ModelConfig] cannot derive gpus_per_node: "
                f"total_gpus={self.total_gpus} not divisible by nnodes={self.nnodes}"
            )
        if self.nnodes_per_pp_rank > 2:
            raise ValueError(
                f"[ModelConfig] only support 2-nodes TP for now, but got "
                f"nnodes_per_pp_rank={self.nnodes_per_pp_rank} "
                f"(tp_size={self.tp_size}, gpus_per_node={self.gpus_per_node})"
            )
        if self.instance_num < 1:
            raise ValueError(
                f"[ModelConfig] instance_num must be >= 1, got {self.instance_num}"
            )

        # ---- LayerGroup invariants ----
        self._validate_layer_groups()

        object.__setattr__(self, '_frozen', True)

    def _validate_layer_groups(self) -> None:
        """Validate ``layer_groups`` against ``num_layers``.

        No-op when ``layer_groups`` is None (uniform model). Enforces:

        * ``g.num_layers == len(g.layer_indices)`` for every group.
        * No internal duplicate ``layer_indices`` within one group.
        * Every ``layer_indices`` entry lies in ``[0, num_layers)``.
        * ``g.compress_ratio >= 1`` for every group.

        Uncached layers (e.g. DSv4 layers 0/1 with ``compress_ratio == 0`` at
        the model level) simply do not appear in any group's ``layer_indices``
        and produce empty member lists in the resulting :class:`LayerMemberMap`;
        the union is therefore *not* required to cover every original layer.
        """
        if not self.layer_groups:
            return
        N = self.num_layers
        for gi, g in enumerate(self.layer_groups):
            if g.num_layers != len(g.layer_indices):
                raise ValueError(
                    f"[ModelConfig] layer_groups[{gi}].num_layers={g.num_layers} "
                    f"does not match len(layer_indices)={len(g.layer_indices)}"
                )
            if len(set(g.layer_indices)) != len(g.layer_indices):
                raise ValueError(
                    f"[ModelConfig] layer_groups[{gi}].layer_indices has duplicates: "
                    f"{g.layer_indices}"
                )
            if g.compress_ratio < 1:
                raise ValueError(
                    f"[ModelConfig] layer_groups[{gi}].compress_ratio must be >= 1, "
                    f"got {g.compress_ratio}"
                )
            for orig in g.layer_indices:
                if not 0 <= orig < N:
                    raise ValueError(
                        f"[ModelConfig] layer_groups[{gi}] has out-of-range "
                        f"layer index {orig} (must be in [0, {N}))"
                    )

    @cached_property
    def layer_member_map(self) -> Optional[LayerMemberMap]:
        """CSR mapping from original layer id -> [(group_idx, local_id), ...].

        Returns ``None`` when ``layer_groups`` is not set (uniform model — the
        layerwise transfer path then uses the legacy single-group code path).
        Computed once on first access and cached in ``__dict__`` via
        ``functools.cached_property`` (write bypasses the frozen
        ``__setattr__``).
        """
        if not self.layer_groups:
            return None
        return build_layer_member_map(self.layer_groups, self.num_layers)

    def __setattr__(self, name: str, value) -> None:
        if name == '_frozen':
            return object.__setattr__(self, name, value)
        # ``layer_groups`` is a derived field that is sometimes discovered late
        # (e.g. DeepSeek V4 sub-pool layout is only known once SGLang has built
        # the GPU KV pools, which happens after FlexKVConfig.from_env()/post_init
        # have already called freeze()). Allow late assignment of this field
        # specifically so multi-group registration paths work.
        if name == 'layer_groups':
            return object.__setattr__(self, name, value)
        if getattr(self, '_frozen', False):
            raise AttributeError(
                f"ModelConfig is frozen — cannot set '{name}'. "
                f"All primitive fields must be set during post_init_from_*(), "
                f"after which freeze() is called.  Derived fields (effective_tp_size, "
                f"tp_size_per_node, cp_size_per_node, nnodes_per_pp_rank) are @property "
                f"and cannot be set at all."
            )
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Derived topology properties
    # ------------------------------------------------------------------
    @property
    def total_gpus(self) -> int:
        """Total GPU worker registration slots across all nodes for one FlexKV instance.

        Unified formula: dp_size × tp_size × cp_size × pp_size."""
        return self.dp_size * self.tp_size * self.cp_size * self.pp_size

    @property
    def total_clients(self) -> int:
        """Total number of DPClient endpoints across all instances."""
        return self.instance_num * self.dp_size

    @property
    def gpus_per_node(self) -> int:
        """GPU worker registration slots on this node (across all DP shards, PP stages and TP groups)."""
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
        return max(1, self.tp_size // self.nnodes_per_pp_rank)

    @property
    def attn_dp_size(self) -> int:
        """Attention-level DP size (= dp_size when enable_dp_attention else 1)."""
        return max(1, self.dp_size) if self.enable_dp_attention else 1

    @property
    def attn_tp_size(self) -> int:
        """Compatibility alias for the normalized attention TP size."""
        return max(1, self.tp_size)

    @property
    def attn_tp_size_per_node(self) -> int:
        """Attention-level TP size per node."""
        return self.tp_size_per_node

    @property
    def attn_cp_size_per_node(self) -> int:
        """Compatibility alias for per-node context parallel size."""
        return self.cp_size_per_node

    @property
    def cp_size_per_node(self) -> int:
        """CP size on this node for a single PP stage.

        Used for multi-node scenarios where the CP group spans multiple nodes.
        """
        return max(1, self.cp_size // self.nnodes_per_pp_rank)

    @property
    def effective_tp_size(self) -> int:
        """Number of CPU block slices = tp_size × cp_size."""
        return max(1, self.tp_size) * max(1, self.cp_size)

    @property
    def effective_tp_size_per_node(self) -> int:
        """Per-node counterpart of :pyattr:`effective_tp_size`."""
        return self.tp_size_per_node * self.cp_size_per_node

    @property
    def num_kv_heads_per_node(self) -> int:
        """Number of KV heads visible to a single node."""
        if self.use_mla:
            return self.num_kv_heads
        return self.num_kv_heads * self.tp_size_per_node // max(1, self.tp_size)

    @property
    def kv_dim(self) -> int:
        """KV dimension: 1 for MLA (no head split), 2 for standard (head split)."""
        return 1 if self.use_mla else 2

    @property
    def bytes_per_token_per_layer(self) -> int:
        """Raw byte footprint of a single (layer, token) KV slot.

        NOTE: assumes uniform (num_kv_heads, head_size, dtype) across layers.
        Not meaningful when ``layer_groups`` is set — callers in that path must
        use ``token_size_in_bytes`` instead.
        """
        return self.num_kv_heads * self.head_size * self.kv_dim * self.dtype.itemsize

    @property
    def token_size_in_bytes(self) -> int:
        """Whole-model per-token KV footprint (bytes) across all layers/groups."""
        kv_dim = 1 if self.use_mla else 2
        if self.layer_groups:
            # layer_groups store per-GPU num_kv_heads; multiply by tp_size
            # to get full-model per-token size (matching CPU/SSD block sizing).
            # Each group may carry its own dtype (None = inherit ModelConfig.dtype),
            # so indexer-as-group (fp8/uint8) and main KV (bf16) sum correctly.
            # ``compress_ratio`` shrinks the per-token contribution. This
            # integer property is suitable for aggregate metrics; exact block
            # allocation must use block_size_in_bytes_for_cache so division is
            # applied to tokens_per_block before rounding.
            return sum(
                g.num_layers * g.num_kv_heads * g.head_size * kv_dim
                * (g.dtype or self.dtype).itemsize
                // g.compress_ratio
                for g in self.layer_groups
            ) * self.tp_size
        return self.num_layers * self.num_kv_heads * self.head_size * kv_dim * self.dtype.itemsize

    def __str__(self) -> str:
        layer_groups_str = (
            f", layer_groups={len(self.layer_groups)}groups"
            if self.layer_groups else ""
        )
        return (
            f"ModelConfig(num_layers={self.num_layers}, num_kv_heads={self.num_kv_heads}"
            f", head_size={self.head_size}, use_mla={self.use_mla}"
            f", dtype={self.dtype}"
            f", tp_size={self.tp_size}, pp_size={self.pp_size}, dp_size={self.dp_size}"
            f", cp_size={self.cp_size}"
            f", total_gpus={self.total_gpus}"
            f", nnodes={self.nnodes}, master_host={self.master_host!r}"
            f", instance_num={self.instance_num}"
            f"{layer_groups_str}"
        )


@dataclass(frozen=True)
class RankInfo:
    model_config: ModelConfig
    tp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0
    cp_rank: int = 0
    # Compatibility mirror for origin/main's SGLang-facing API.
    attn_cp_rank: int = 0
    node_rank: int = 0
    instance_id: int = 0
    pp_start_layer: int = 0
    pp_end_layer: int = -1
    local_rank: int = -1

    def __post_init__(self) -> None:
        if self.cp_rank == 0 and self.attn_cp_rank != 0:
            object.__setattr__(self, "cp_rank", self.attn_cp_rank)
        elif self.attn_cp_rank == 0 and self.cp_rank != 0:
            object.__setattr__(self, "attn_cp_rank", self.cp_rank)
        elif self.cp_rank != self.attn_cp_rank:
            raise ValueError(
                f"cp_rank={self.cp_rank} and attn_cp_rank={self.attn_cp_rank} disagree"
            )
        if self.local_rank < 0:
            model_config = self.model_config
            tp_cp_rank = (
                self.cp_rank * model_config.tp_size_per_node
                + self.tp_rank_per_node
            )
            if model_config.enable_dp_attention:
                local_rank = (
                    self.pp_rank_per_node
                    * model_config.cp_size_per_node
                    * model_config.tp_size_per_node
                    + tp_cp_rank
                )
            else:
                local_rank = (
                    (
                        self.dp_rank_per_node * self.pp_size_per_node
                        + self.pp_rank_per_node
                    )
                    * model_config.cp_size_per_node
                    * model_config.tp_size_per_node
                    + tp_cp_rank
                )
            object.__setattr__(self, "local_rank", local_rank)

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
        """Compatibility alias for the normalized attention TP rank."""
        return self.tp_rank

    @property
    def effective_tp_rank(self) -> int:
        """Effective tp-rank in the *data-plane* segmentation space.

        For MLA models, every CP rank holds the same KV pages (the MLA latent
        is not split along the sequence axis from a KV perspective), so
        ``cp_rank`` must NOT participate in slice indexing — otherwise a CP rank
        would write to a non-existent CPU slice and corrupt block accounting.
        For non-MLA models, CP shards along the sequence dimension and each
        ``(cp_rank, tp_rank)`` pair owns a unique slice.
        """
        if self.model_config.use_mla:
            return self.tp_rank
        return self.cp_rank * max(1, self.model_config.tp_size) + self.tp_rank

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
        return max(
            1,
            model_config.gpus_per_node
            // (
                self.pp_size_per_node
                * model_config.tp_size_per_node
                * model_config.cp_size_per_node
            ),
        )

    @property
    def dp_rank_per_node(self) -> int:
        """This rank's DP index within its node."""
        return self.dp_rank % self.dp_size_per_node

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
            f", dp_rank={self.dp_rank}, cp_rank={self.cp_rank}"
            f", node_rank={self.node_rank}, instance_id={self.instance_id}"
            f", local_rank={self.local_rank}, effective_tp_rank={self.effective_tp_rank}"
        )

@dataclass
class SWAPoolConfig:
    """Configuration for SWA (Sliding Window Attention) host pool(s).

    SWA is managed at PAGE granularity: one pool slot stores exactly one
    ``tokens_per_block`` page of SWA KV, and all SWA IO moves a whole page
    (one slot) at a time.
    """
    enabled: bool = False
    num_slots: int = 1024              # Number of CPU SWA pool slots
    num_ssd_slots: int = 0             # Number of SSD SWA pool slots (0 = no SSD SWA tier)
    num_remote_slots: int = 0          # Number of REMOTE SWA pool slots (0 = no REMOTE SWA tier)
    num_swa_layers: int = 61           # Number of SWA layers (all 61 for DSv4)
    bytes_per_token_per_layer: int = 584  # nope_fp8(448) + rope_bf16(128) + scale(8)
    # True when the SWA page also carries heterogeneous sidecar groups (for
    # example DeepSeek-V4 attention/indexer compress states).  Layerwise GET
    # still fuses SWA/state H2D into LAYERWISE via launch_swa_mg_h2d_layer_;
    # this flag drives multi-group layout / registration, not the fuse choice.
    multi_group: bool = False
    evict_ratio: float = 0.1           # Fraction of pool to evict when full
    pin_memory: bool = True            # Use pinned memory for async DMA

    def for_ssd_tier(self) -> "SWAPoolConfig":
        """Derive the SSD-tier SWA config (same slot geometry, num_ssd_slots slots).

        SSD SWA slots are not pinned host memory; pin_memory is forced off."""
        return replace(self, num_slots=self.num_ssd_slots, pin_memory=False)

    def for_remote_tier(self) -> "SWAPoolConfig":
        """Derive the REMOTE-tier SWA config (same slot geometry, num_remote_slots).

        REMOTE SWA slots are not pinned host memory; pin_memory is forced off."""
        return replace(self, num_slots=self.num_remote_slots, pin_memory=False)

    def for_cache_tier(self, device_type) -> Optional["SWAPoolConfig"]:
        """Return the SWA config this cache tier should own, or None.

        CPU uses the primary pool config. SSD and REMOTE use their tier-specific
        slot counts and are disabled when that tier's slot count is zero.
        """
        if not self.enabled:
            return None
        device_name = getattr(device_type, "name", str(device_type))
        if device_name == "CPU":
            return self
        if device_name == "SSD" and self.num_ssd_slots > 0:
            return self.for_ssd_tier()
        if device_name == "REMOTE" and self.num_remote_slots > 0:
            return self.for_remote_tier()
        return None


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

    # Stored for deferred recomputation when layer_groups become known
    _user_cpu_cache_gb: float = 0
    _user_ssd_cache_gb: float = 0

    # SWA pool config (DeepSeek V4)
    swa: Optional['SWAPoolConfig'] = None

    # Gate for the SWA peer-op DATA-PLANE transfer (SWA_H2D/SWA_D2H ops built into
    # the transfer graph). Default False: the SWA control plane (node-mounted match /
    # capacity / lock) works regardless, but the actual async SWA byte transfer
    # requires the dedicated SWA transfer worker (data plane). Keep this False
    # until that worker is registered, otherwise SWA ops would hit "Unsupported
    # transfer type" in the transfer engine. Flip to True once the worker lands.
    enable_swa_transfer: bool = False

    # Fuse SWA (including heterogeneous state sidecars) into the layerwise H2D
    # worker.  When disabled, layerwise main-KV restore stays enabled while SWA
    # uses its standalone H2D worker and completes before the main layerwise op.
    swa_multi_layer: bool = True

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

    ce_segment_threshold=int(os.getenv('FLEXKV_CE_SEGMENT_THRESHOLD', 8)),
    ce_path_opt=bool(int(os.getenv('FLEXKV_CE_PATH_OPT', 1))),
    enable_ce_memcpy2d=bool(int(os.getenv('FLEXKV_ENABLE_CE_MEMCPY2D', 1))),
    ce_gather_threads=int(os.getenv('FLEXKV_CE_GATHER_THREADS', 4)),
    ce_gather_nt=bool(int(os.getenv('FLEXKV_CE_GATHER_NT', 1))),

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
    # DeepSeek-V4 SWA sidecar policy. None/True enables attention and indexer
    # compress-state I/O together with SWA; False keeps the legacy SWA-only
    # path. None is intentionally distinct from False so old configs default
    # to the correctness-preserving state restore path.
    swa_multi_group: Optional[bool] = None
    # Fuse SWA/state H2D into the main layerwise restore worker. Disable this to
    # keep SWA/state on the standalone predecessor worker as a compatibility or
    # debugging fallback.
    swa_multi_layer: bool = True

    def __post_init__(self):
        if self.cpu_cache_gb <= 0:
            raise ValueError(f"Invalid cpu_cache_gb: {self.cpu_cache_gb}")
        if self.ssd_cache_gb < 0:
            raise ValueError(f"Invalid ssd_cache_gb: {self.ssd_cache_gb}")
        if self.ssd_cache_gb > 0 and self.ssd_cache_gb <= self.cpu_cache_gb:
            raise ValueError(f"Invalid ssd_cache_gb: {self.ssd_cache_gb}, "
                             f"must be greater than cpu_cache_gb: {self.cpu_cache_gb}.")
        if self.swa_multi_group is not None and not isinstance(
            self.swa_multi_group, bool
        ):
            raise ValueError(
                "swa_multi_group must be a boolean when configured, "
                f"got {self.swa_multi_group!r}"
            )
        if not isinstance(self.swa_multi_layer, bool):
            raise ValueError(
                "swa_multi_layer must be a boolean, "
                f"got {self.swa_multi_layer!r}"
            )

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
    swa_multi_group_env = os.getenv('FLEXKV_SWA_MULTI_GROUP')
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
        swa_multi_group=(
            None
            if swa_multi_group_env is None
            else bool(int(swa_multi_group_env))
        ),
        swa_multi_layer=bool(int(os.getenv('FLEXKV_SWA_MULTI_LAYER', 1))),
    )

def convert_to_block_num(size_in_GB: float, block_size_in_bytes: int) -> int:
    return int(size_in_GB * 1024 * 1024 * 1024 / block_size_in_bytes)


def block_size_in_bytes_for_cache(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    rank_info: Optional["RankInfo"] = None,
) -> int:
    """Bytes per CPU/SSD block for pool sizing.

    When ``layer_groups`` is set, sum exact per-group bytes after applying each
    group's block compression. Otherwise fall back to the uniform per-PP-stage
    estimate from ``rank_info``.
    """
    if model_config.layer_groups is not None:
        # Match KVCacheLayout._compute_kv_shape exactly.  Computing a rounded
        # per-token size first and multiplying it by tokens_per_block loses
        # bytes for compressed groups whenever the group contribution is not
        # divisible by compress_ratio.
        for gi, group in enumerate(model_config.layer_groups):
            if cache_config.tokens_per_block % group.compress_ratio != 0:
                raise ValueError(
                    f"layer_groups[{gi}].compress_ratio={group.compress_ratio} "
                    f"does not divide tokens_per_block="
                    f"{cache_config.tokens_per_block}"
                )
        return model_config.tp_size * sum(
            group.num_layers
            * model_config.kv_dim
            * (cache_config.tokens_per_block // group.compress_ratio)
            * group.num_kv_heads
            * group.head_size
            * (group.dtype or model_config.dtype).itemsize
            for group in model_config.layer_groups
        )
    if rank_info is None:
        raise ValueError(
            "rank_info is required when model_config.layer_groups is None")
    return rank_info.token_size_in_bytes_per_pp_stage * cache_config.tokens_per_block


def recompute_cache_block_counts(
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> bool:
    """Recompute ``num_cpu_blocks`` / ``num_ssd_blocks`` from stored GB budgets.

    No-op when ``layer_groups`` is unset (initial uniform estimate is final).
    Returns True if any block count changed.
    """
    if model_config.layer_groups is None:
        return False

    block_size_in_bytes = block_size_in_bytes_for_cache(
        model_config, cache_config)
    capacity_divisor = 1
    if (model_config.use_mla
            and GLOBAL_CONFIG_FROM_ENV.mla_d2h_mode == "all_write"):
        capacity_divisor = max(
            1, model_config.effective_tp_size_per_node)

    changed = False

    if cache_config._user_cpu_cache_gb > 0:
        old_cpu = cache_config.num_cpu_blocks
        new_cpu = (
            convert_to_block_num(
                cache_config._user_cpu_cache_gb, block_size_in_bytes)
            // capacity_divisor
        )
        if new_cpu != old_cpu:
            flexkv_logger.info(
                f"Recomputed num_cpu_blocks with layer_groups: "
                f"{old_cpu} -> {new_cpu} "
                f"(block_size={block_size_in_bytes} B)")
            cache_config.num_cpu_blocks = new_cpu
            changed = True

    if cache_config._user_ssd_cache_gb > 0:
        old_ssd = cache_config.num_ssd_blocks
        new_ssd = (
            convert_to_block_num(
                cache_config._user_ssd_cache_gb, block_size_in_bytes)
            // capacity_divisor
        )
        if new_ssd != old_ssd:
            flexkv_logger.info(
                f"Recomputed num_ssd_blocks with layer_groups: "
                f"{old_ssd} -> {new_ssd} "
                f"(block_size={block_size_in_bytes} B)")
            cache_config.num_ssd_blocks = new_ssd
            changed = True
            if (cache_config.num_ssd_blocks
                    % len(cache_config.ssd_cache_dir) != 0):
                cache_config.num_ssd_blocks = (
                    (cache_config.num_ssd_blocks
                     // len(cache_config.ssd_cache_dir) + 1)
                    * len(cache_config.ssd_cache_dir)
                )

    return changed


def update_default_config_from_user_config(rank_info: RankInfo,
                                           cache_config: CacheConfig,
                                           user_config: UserConfig) -> None:
    block_size_in_bytes = block_size_in_bytes_for_cache(
        rank_info.model_config, cache_config, rank_info)

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

    # Store original GB values for deferred recomputation (when layer_groups become known)
    cache_config._user_cpu_cache_gb = user_config.cpu_cache_gb
    cache_config._user_ssd_cache_gb = user_config.ssd_cache_gb

    cache_config.num_cpu_blocks = (
        convert_to_block_num(user_config.cpu_cache_gb, block_size_in_bytes)
        // capacity_divisor
    )
    cache_config.num_ssd_blocks = (
        convert_to_block_num(user_config.ssd_cache_gb, block_size_in_bytes)
        // capacity_divisor
    )

    flexkv_logger.info(
        f"[CacheConfig] GB->blocks conversion: "
        f"block_size={block_size_in_bytes} B; "
        f"cpu_cache_gb={user_config.cpu_cache_gb} -> num_cpu_blocks={cache_config.num_cpu_blocks}, "
        f"ssd_cache_gb={user_config.ssd_cache_gb} -> num_ssd_blocks={cache_config.num_ssd_blocks}"
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
    cache_config.swa_multi_layer = user_config.swa_multi_layer

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
