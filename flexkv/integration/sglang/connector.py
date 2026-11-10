"""Wrapper around FlexKV ``KVManager`` for sglang.

The public surface is small (see "Public API" below). The class owns:

* the FlexKV ``KVManager`` (server-client mode when ``dp_size > 1`` or
  multi-instance; in-process otherwise — handled by FlexKV itself);
* the per-rank ``KVTPClient`` that registers this rank's GPU KV cache
  with the FlexKV TransferManager;
* an optional ``FlexKVLayerDoneCounter`` plus the UDS-side handshake
  that wires its eventfds into the FlexKV layerwise transfer worker.

Cross-rank sync uses :class:`FlexKVComm`. Only the **sync leader**
(rank 0 of every PP × CP × TP axis) talks to ``KVManager``; other
ranks block on broadcast / barrier.

Modes:
  * **MP / synchronous** (default): ``retrieve_kv`` fires ``launch``
    and blocks on ``wait`` so the device slots are ready by the time
    sglang's prefill runs.
  * **Layerwise** (``FLEXKV_ENABLE_LAYERWISE_TRANSFER=1``): ``launch``
    is fired with ``layerwise_transfer=True`` and the per-layer hook
    registered via ``register_layer_transfer_counter`` blocks each
    forward layer on its own eventfd.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import math
import os
import signal
import socket
import struct
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from flexkv.integration.sglang.comm import (
    CMD_LAYERWISE,
    FlexKVComm,
    FlexKVLayerDoneCounter,
    FlexKVScatterChannel,
    send_fds,
)

from flexkv.common.config import (
    GLOBAL_CONFIG_FROM_ENV,
    LayerGroupSpec,
    recompute_cache_block_counts,
)
from flexkv.common.request import KVResponseStatus
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.integration.config import FlexKVConfig
from flexkv.kvmanager import KVManager
from flexkv.server.client import KVTPClient
from flexkv.transfer.layer_eventfd import build_layerwise_eventfd_socket_path
from flexkv.transfer_manager import TransferManagerOnRemote


def _radixshmem_distributed(model_config, cache_config) -> bool:
    """Whether this node's radix-server is part of a cluster (peer pulls
    possible). The server is the operator's process and knows; every TP rank
    asks it the same question once it is ready, so the prefetch gate below is
    the same in all ranks (the PREFETCH_START scatter needs that). The attach
    brings FlexKV's geometry, so the answer does not depend on which process
    configured the server first."""
    from flexkv.server.shm_radix_bootstrap import radix_server_is_distributed
    return radix_server_is_distributed(model_config, cache_config, label="FlexKVConnector")


logger = logging.getLogger(__name__)

_SGLANG_REQ_ID_UNSET = object()
_INFO_STATUSES = {"hit", "miss", "skipped", "success"}
_WARNING_STATUSES = {"cancelled", "failed", "missing", "not_found", "timeout"}
# Minimum gap between idle engine nudges in non-chunked mode (seconds).
_IDLE_DRIVE_INTERVAL_S = 0.1


@dataclass
class _CacheOpContext:
    operation: str
    sglang_req_id: Optional[str]
    start_ns: int
    task_id: int = -1


@dataclass
class _PreparedLayerwiseLoad:
    task_id: int
    producer_id: int
    slot_mapping: torch.Tensor
    swa_slot_mapping: Optional[torch.Tensor]
    context: _CacheOpContext
    start_ns: int


@dataclass(frozen=True)
class _IndexerBufferGroup:
    """Physical indexer buffers plus their original layer ids."""

    buffers: Tuple[torch.Tensor, ...]
    layer_indices: Tuple[int, ...]
    logical_layer_count: int


def _is_sparse_indexer_group(group: Optional[_IndexerBufferGroup]) -> bool:
    """True when only a strict subset of logical indexer layers is cached."""
    return bool(group is not None and len(group.buffers) < group.logical_layer_count)


def _status_value(response: Any) -> str:
    status = getattr(response, "status", None)
    value = getattr(status, "value", status)
    return str(value).lower() if value is not None else "missing"


def _is_terminal_status(status: str) -> bool:
    return status in {"success", "failed", "cancelled", "not_found"}


def _pack_dsv4_indexer_full_pages(
    buffers: Sequence[torch.Tensor],
    *,
    full_page_size: int,
    compress_ratio: int,
    physical_page_size: int,
) -> Tuple[List[torch.Tensor], int]:
    """View indexer rows so one GPU row is one FlexKV (full) page.

    V4.1 ratio-1/2 indexers page at 64 or 128 tokens, while FlexKV addresses
    every main-KV group by the full ``page_size`` id. SGLang's PD path already
    groups ``(full_page_size // ratio) / physical_page_size`` indexer rows
    into one transfer item; this view lets the existing DSv4
    ``LayerGroupSpec`` path do the same without a new transfer mode.

    Returns ``(buffers, logical_tokens_per_full_page)``.
    """
    if compress_ratio < 1 or full_page_size % compress_ratio != 0:
        raise RuntimeError(
            f"FlexKV DSv4 indexer compress_ratio={compress_ratio} does not "
            f"divide page_size={full_page_size}"
        )
    slots_per_full_page = full_page_size // compress_ratio
    if physical_page_size < 1 or slots_per_full_page % physical_page_size != 0:
        raise RuntimeError(
            f"FlexKV DSv4 ratio-{compress_ratio} indexer page_size="
            f"{physical_page_size} does not tile a full page of "
            f"{slots_per_full_page} slots"
        )
    pages_per = slots_per_full_page // physical_page_size
    if pages_per <= 1:
        return list(buffers), slots_per_full_page
    packed: List[torch.Tensor] = []
    for buf in buffers:
        if buf.ndim != 2:
            raise RuntimeError(
                "FlexKV DSv4 indexer buffers must be 2D page rows, "
                f"got shape={tuple(buf.shape)}"
            )
        usable_rows = (buf.shape[0] // pages_per) * pages_per
        if usable_rows == 0:
            raise RuntimeError(
                "FlexKV DSv4 indexer is shorter than one full page: "
                f"rows={buf.shape[0]}, pages_per_full_page={pages_per}"
            )
        packed.append(
            buf[:usable_rows].reshape(
                usable_rows // pages_per, pages_per * buf.shape[1]
            )
        )
    return packed, slots_per_full_page


class FlexKVHostReleaseShim:
    """HiCache-compatible host-release hook for ``release_host_resources``.

    Scheduler already does::

        host_pool = getattr(tree_cache, "token_to_kv_pool_host", None)
        if host_pool is not None:
            host_pool.destroy()

    FlexKV has no local pinned HostKVCache; ``destroy()`` forwards to
    ``FlexKVConnector.shutdown()`` so unpin stays on that same path.
    """

    def __init__(self, connector: FlexKVConnector) -> None:
        self._connector = connector

    def destroy(self) -> None:
        print(
            f"[FLEXKV] HostReleaseShim.destroy begin {self._connector._label}",
            flush=True,
        )
        self._connector.shutdown()
        print(
            f"[FLEXKV] HostReleaseShim.destroy done {self._connector._label}",
            flush=True,
        )


class FlexKVConnector:
    """A FlexKV-side façade used by :class:`FlexKVRadixCache`.

    This class manages connection lifecycle and provides a small,
    sgl-friendly contract over FlexKV's task-based API:

      * ``lookup_kv`` — page-aligned hit count + a held task id.
      * ``retrieve_kv`` — synchronous load (launch + wait).
      * ``start_load_kv_layerwise`` — layerwise async load.
      * ``store_kv`` — page-aligned write back.
      * ``check_completed_stores`` — drain async store completions.
      * ``prefetch_async`` / ``check_prefetch_progress`` /
        ``pop_prefetch_loaded_tokens`` / ``cancel_prefetch`` —
        opportunistic SSD/Remote → CPU staging (wired from the scheduler
        when ``--enable-flexkv`` is set).
      * ``release_pending`` — cancel a held task whose load won't run.
      * ``reset`` / ``shutdown``.
    """

    _chunked_namespace_supported = True
    # Read by SGLang's layer-broadcast scheduler: only a strict subset of
    # indexer layers is cached, so inactive indexer broadcasts may be skipped.
    sparse_indexer = False

    @property
    def supports_cache_namespace(self) -> bool:
        """All enabled lookup/store/prefetch paths accept ``namespace``.

        A chunked-prefetch connector must opt in separately after forwarding
        the namespace through every chunk and its session identity.
        """
        return not getattr(self, "_chunked_prefetch", False) or getattr(
            self, "_chunked_namespace_supported", False
        ) is True

    def __init__(
        self,
        *,
        sgl_model_config: Any,
        server_args: Any,
        page_size: int,
        kvcache: Any,
        tp_rank: int,
        dp_rank: Optional[int],
        pp_rank: int,
        attn_cp_rank: int,
        pp_group: Any = None,
        attn_tp_group: Any = None,
        attn_cp_group: Any = None,
        layer_split_plan: Any = None,
        indexer_layer_ids: Optional[List[int]] = None,
        draft_kvcache: Any = None,
    ) -> None:
        self.page_size = int(page_size)

        # 1. Resolve FlexKV config from env + sglang server args.
        self.flexkv_config = FlexKVConfig.from_env()
        from flexkv.common.gdr_config import publish_gdr_config
        publish_gdr_config(self.flexkv_config.user_config.gdr)
        self.rank_info = self.flexkv_config.post_init_from_sglang_config(
            sglang_config=sgl_model_config,
            server_args=server_args,
            page_size=self.page_size,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,  # None is resolved (or rejected) by the config
            attn_cp_rank=attn_cp_rank,
        )
        self.model_config = self.flexkv_config.model_config
        self.cache_config = self.flexkv_config.cache_config
        self._chunked_prefetch = bool(self.cache_config.enable_chunked_prefetch)
        if self._chunked_prefetch:
            from flexkv.prefetch.types import PrefetchOptions
            opts = dict(self.cache_config.prefetch_options or {})
            opts.setdefault(
                "policy",
                getattr(server_args, "hicache_storage_prefetch_policy", "wait_complete"),
            )
            self._chunked_prefetch_options = PrefetchOptions(**opts)
            self._chunked_prefetch_options.validate()
            self.cache_config.prefetch_options = opts
            # wait_complete needs the original whole-task path. Disable the
            # chunk runtime before KVManager is created, including its threads.
            if self._chunked_prefetch_options.policy == "wait_complete":
                self._chunked_prefetch = False
                self.cache_config.enable_chunked_prefetch = False
        self._prefetch_sessions = {}
        self._prefetch_result_sessions = {}
        self._prefetch_loaded_spans = {}
        self._label = f"[model_config={self.model_config}, rank_info={self.rank_info}]"

        # 2. Cross-rank sync context.
        world_rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self._sync_ctx = FlexKVComm(
            rank_info=self.rank_info,
            world_rank=world_rank,
            pp_group=pp_group,
            attn_tp_group=attn_tp_group,
            attn_cp_group=attn_cp_group,
            enable_collective_sync=GLOBAL_CONFIG_FROM_ENV.enable_collective_sync,
        )
        # Master switch: FLEXKV_ENABLE_COLLECTIVE_SYNC=0 disables all
        # cross-rank scatter/barrier/all_reduce. Every rank then acts
        # independently (TP-only deployments that share one FlexKV
        # instance per rank do not need collective sync).
        if not GLOBAL_CONFIG_FROM_ENV.enable_collective_sync:
            self._sync_ctx.needs_sync = False
            logger.info(
                "[FlexKV] Collective sync disabled "
                "(FLEXKV_ENABLE_COLLECTIVE_SYNC=0): "
                "scatter/barrier/all_reduce are skipped."
            )

        self._agree_cache_geometry(
            json.loads(os.environ["FLEXKV_GDR_CONFIG"]), "GDR configuration"
        )

        # 3. Align block counts across all ranks (MIN reduce) so each
        # rank's KVManager registers compatible sizes.
        for attr in ("num_cpu_blocks", "num_ssd_blocks", "num_remote_blocks"):
            orig = getattr(self.cache_config, attr, None)
            if orig is None or orig <= 0:
                continue
            aligned = self._sync_ctx.all_reduce_min(int(orig))
            if aligned != orig:
                logger.info(
                    "[FlexKV] Block count MIN alignment '%s': %d -> %d",
                    attr,
                    orig,
                    aligned,
                )
            setattr(self.cache_config, attr, aligned)

        # 4. Extract GPU buffers. DeepSeek V4 uses heterogeneous compressed
        # pools and is registered through FlexKV's multi-group API; regular
        # MLA/MHA models keep the single-layout path.
        self._kvcache = kvcache
        self._swa_kv_pool = getattr(kvcache, "swa_kv_pool", None)
        self._is_dsv4 = getattr(sgl_model_config, "is_deepseek_v4_arch", False)
        self._dsv4_layer_groups: List[Dict[str, Any]] = []
        self._dsv4_state_groups: List[Dict[str, Any]] = []
        self._deduplicate_indexer_group = os.getenv(
            "FLEXKV_DEDUP_INDEXER_GROUP", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

        # Encoder bounded replay rebuilds the window from the full KV, so the
        # model has no paged SWA pool to register. The host SWA pool must go
        # with it: the cache engine plans SWA ops from that config alone, and
        # they would reach a transfer engine with no SWA GPU handle behind it.
        if self._swa_kv_pool is None and self.cache_config.swa is not None:
            self.cache_config.swa = None
            self.cache_config.enable_swa_transfer = False
            logger.info(
                "[FlexKV] No paged SWA KV pool on this rank "
                "(encoder SWA bounded replay); disabling the FlexKV SWA pool %s",
                self._label,
            )

        self._indexer_layer_ids = indexer_layer_ids
        self._draft_kvcache = draft_kvcache
        self._draft_groups = []
        self._draft_namespace = None
        self._agree_cache_geometry(
            [bool(getattr(kvcache, "layer_shard_enabled", False)),
             self._deduplicate_indexer_group, draft_kvcache is not None,
             layer_split_plan is not None,
             bool(getattr(kvcache, "index_k_with_scale_buffer", None)),
             os.environ.get("FLEXKV_L3_PREFETCH_MAX_BLOCKS", "0").strip().lower()],
            "cache feature selection",
        )
        kv_caches = []
        indexer_group = None
        buffer_error = None
        try:
            kv_caches, indexer_group = self._resolve_kv_buffers(kvcache)
            if indexer_group is not None and not self._is_dsv4:
                main = kv_caches[0]
                if main.ndim not in (3, 4):
                    raise ValueError("Indexer requires 3D/4D main KV buffers")
                blocks = main.shape[0] // self.page_size if main.ndim == 3 else main.shape[0]
                if indexer_group.buffers[0].shape[0] != blocks:
                    raise ValueError("Indexer page count must match main KV blocks 1:1")
        except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
            buffer_error = str(exc)
        self._require_cache_preflight(buffer_error, "GPU buffer validation")
        if indexer_group is not None and self.model_config.pp_size == 1:
            sample = indexer_group.buffers[0]
            self._agree_cache_geometry(
                [list(sample.shape), str(sample.dtype), sample.element_size(),
                 list(indexer_group.layer_indices), indexer_group.logical_layer_count],
                "Indexer buffer geometry",
            )
        self._configure_h2d_holders(layer_split_plan)
        draft_error = None
        try:
            self._draft_groups = self._build_draft_groups(draft_kvcache, kv_caches)
        except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
            draft_error = str(exc)
        self._require_cache_preflight(draft_error, "draft buffer validation")
        if self._swa_kv_pool is not None and self.cache_config.swa is not None:
            buffers = self._swa_kv_pool.kv_buffer
            snapshot_bytes = sum(
                buffer[0].numel() * buffer.element_size() for buffer in buffers
            )
            for group in self._dsv4_state_groups:
                snapshot_bytes += sum(
                    int(group["sub_page_size"]) * buffer[0].numel() * buffer.element_size()
                    for buffer in group["buffers"]
                )
            self.cache_config.swa.snapshot_bytes = snapshot_bytes

        # Heterogeneous groups change the bytes represented by one logical
        # FlexKV block. Recompute CPU/SSD capacities before KVManager starts.
        self._apply_layer_groups_for_cache_sizing(kv_caches, indexer_group)
        if self._draft_groups or getattr(self.model_config, "layer_shard_enabled", False):
            layout = [(g.num_layers, g.num_kv_heads, g.head_size, g.compress_ratio,
                       str(g.dtype), g.layer_indices) for g in self.model_config.layer_groups]
            identity = ["hy4-target-draft-v1", getattr(server_args, "model_path", None),
                        getattr(server_args, "speculative_draft_model_path", None),
                        self.page_size, self.model_config.nnodes,
                        self.model_config.layer_shard_enabled,
                        self.model_config.h2d_layer_holders, layout]
            self._draft_namespace = self._agree_cache_geometry(identity, "target/draft layout")
        if self._is_dsv4 and self.cache_config.swa is not None:
            _, _, swa_layer_groups, _, _ = self._build_dsv4_swa_registration()
            self.cache_config.swa.layer_groups = swa_layer_groups
        self._label = f"[model_config={self.model_config}, rank_info={self.rank_info}]"

        # 5. On multi-node setups, every node beyond node 0 needs a
        # TransferManagerOnRemote process (FlexKV side) before any rank
        # on that node can register GPU buffers.
        self._remote_process = None
        needs_remote_transfer_manager = self.model_config.local_dp_size is None
        if (
            needs_remote_transfer_manager
            and self.model_config.nnodes > 1
            and self.rank_info.node_rank > 0
            and self.rank_info.local_rank == 0
        ):
            self._remote_process = TransferManagerOnRemote.create_process(
                master_host=self.model_config.master_host,
                master_ports=self.model_config.master_ports,
            )
            logger.info(
                "[FlexKV] Launched TransferManagerOnRemote on node_rank=%d %s",
                self.rank_info.node_rank,
                self._label,
            )

        # 6. Bring up KVManager on the sync leader only.
        self.kv_manager: Optional[KVManager] = None
        if self._sync_ctx.is_sync_leader:
            self.kv_manager = KVManager(
                model_config=self.model_config,
                cache_config=self.cache_config,
                dp_client_id=self.rank_info.dp_client_id,
                local_dp_client_id=self.rank_info.local_dp_client_id,
                server_recv_port=self.flexkv_config.server_recv_port,
                gpu_register_port=self.flexkv_config.gpu_register_port,
            )
            self.kv_manager.start()

        # 7. Per-rank TP client registers this rank's GPU buffers.
        self.tp_client = KVTPClient(
            self.flexkv_config.gpu_register_port,
            dp_client_id=self.rank_info.dp_client_id,
            pp_rank=self.rank_info.pp_rank,
            intra_client_id=self.rank_info.intra_client_id,
            device_id=self.rank_info.local_rank,
            gdr_local_rank=(self.rank_info.intra_client_id
                            % self.model_config.effective_tp_size_per_node),
        )
        self._register_with_retry(kv_caches, indexer_group)

        # 8. Layerwise transfer plumbing.
        self.enable_layerwise = bool(
            int(os.environ.get("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "0"))
        )
        self._profile_store_stages = os.getenv(
            "FLEXKV_PROFILE_STORE_STAGES", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._layerwise_socket = build_layerwise_eventfd_socket_path(
            dp_client_id=self.rank_info.dp_client_id,
            pp_rank=self.rank_info.pp_rank,
            model_config=self.model_config,
        )
        self._layerwise_eventfd_connect_max_retries = max(
            360,
            int(os.environ.get("FLEXKV_LAYERWISE_EVENTFD_CONNECT_MAX_RETRIES", "0")),
        )
        self.layer_done_counter: Optional[FlexKVLayerDoneCounter] = None
        if self.enable_layerwise:
            self.layer_done_counter = FlexKVLayerDoneCounter(
                self.rank_info.num_layers_per_pp_stage,
                world_rank=world_rank,
            )
            self._send_eventfds_to_worker()
        self._layerwise_generation = 0

        # 9. Wait for the KVManager (and its remote subprocess) to be ready.
        if self._sync_ctx.is_sync_leader:
            self._wait_kv_manager_ready()

        if self._chunked_prefetch:
            status = {"error": None}
            if self._sync_ctx.is_sync_leader:
                try:
                    caps = self.kv_manager.prefetch_capabilities()
                    if not caps.stop_and_drain or caps.protocol_version != 1:
                        raise RuntimeError("server lacks chunked prefetch stop/drain support")
                    if self._swa_kv_pool is not None and not caps.partial_swa_checkpoints:
                        raise RuntimeError("server lacks SWA checkpoint prefetch support")
                except Exception as exc:
                    status["error"] = str(exc)
            if self._sync_ctx.needs_sync:
                status = self._sync_ctx.scatter(
                    status, channel=FlexKVScatterChannel.PREFETCH_START
                )
            if status["error"]:
                raise RuntimeError(status["error"])

        # 10. Per-rank in-flight tracking.
        # Loads
        self._pending_lookups: Dict[str, int] = {}  # rid -> fkv_task_id
        self._pending_lookup_contexts: Dict[str, _CacheOpContext] = {}
        self._inflight_loads: Dict[int, int] = {}  # producer_id -> rid hashlike
        self._completed_layerwise: List[int] = []
        self._launched_load_tids: List[int] = []  # leader-only, for periodic drain
        self._prepared_layerwise_loads: Dict[str, _PreparedLayerwiseLoad] = {}
        self._layerwise_batch_launch_error: Optional[str] = None
        self._launched_load_contexts: Dict[int, _CacheOpContext] = {}
        # Stores
        self._inflight_stores: Dict[str, int] = {}  # rid -> fkv_task_id
        self._inflight_store_contexts: Dict[str, _CacheOpContext] = {}
        # Prefetches
        self._ongoing_prefetches: Dict[str, int] = {}  # rid -> fkv_task_id
        self._prefetch_contexts: Dict[str, _CacheOpContext] = {}
        self._prefetch_planned_tokens: Dict[str, int] = {}
        self._prefetch_loaded_tokens: Dict[str, int] = {}
        self._prefetch_enabled = bool(
            self.cache_config.enable_ssd
            or self.cache_config.enable_remote
            or self.cache_config.enable_kv_sharing
            or (GLOBAL_CONFIG_FROM_ENV.enable_radixshmem
                and _radixshmem_distributed(self.model_config, self.cache_config))
        )
        self._shutdown_done = False
        self._l3_prefetch_window = None
        if os.environ.get("FLEXKV_L3_PREFETCH_MAX_BLOCKS", "0").strip().lower() not in ("", "0"):
            from flexkv.integration.sglang.prefetch_window import PrefetchWindow
            window_error = None
            try:
                self._l3_prefetch_window = PrefetchWindow.from_connector(
                    self, scatter_channels=FlexKVScatterChannel
                )
            except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
                window_error = str(exc)
            self._require_cache_preflight(window_error, "L3 window configuration")
            window = self._l3_prefetch_window
            self._agree_cache_geometry(
                None if window is None else
                [window.max_blocks, window.max_inflight, window.max_entries,
                 window.max_queued_blocks, window.poll_interval, window.wait_s,
                 window.unconsumed_ttl], "L3 window limits",
            )

        # Keep FlexKV lifecycle self-contained (avoid scattering hooks in
        # scheduler / tokenizer / http_server):
        #   * Ignore process-group Ctrl+C so this scheduler process stays
        #     alive until ShutdownReq → process exit → atexit unpin.
        #   * atexit runs kv_manager.shutdown() (cudaHostUnregister).
        #   * Stretch the parent's scheduler-exit wait so kill_process_tree
        #     does not SIGKILL mid-unpin (generic env, not FlexKV-named in TM).
        with suppress(Exception):
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Graceful shutdown timeout hierarchy (see FlexKV config.py):
        #   tokenizer wait (this env)                   = 1200s
        #     > TM parent-side wait (FLEXKV_TRANSFER_MANAGER_SHUTDOWN_TIMEOUT_S) = 900s
        #       > per-worker wait   (FLEXKV_WORKER_SHUTDOWN_TIMEOUT_S)           = 600s
        # If you tune one, keep the ordering: worker < TM < scheduler-wait.
        # This must be >= FLEXKV_TRANSFER_MANAGER_SHUTDOWN_TIMEOUT_S + buffer,
        # otherwise tokenizer's kill_process_tree fires mid-unpin.
        os.environ.setdefault("SGLANG_SCHEDULER_SHUTDOWN_WAIT_S", "1200")
        # Compat with older server.sh / docs.
        os.environ.setdefault(
            "SGLANG_FLEXKV_SHUTDOWN_WAIT_S",
            os.environ["SGLANG_SCHEDULER_SHUTDOWN_WAIT_S"],
        )
        atexit.register(self.shutdown)

        logger.info(
            "[FlexKV] Connector ready %s: layerwise=%s, prefetch=%s",
            self._label,
            self.enable_layerwise,
            self._prefetch_enabled,
        )

    def _new_op_context(
        self,
        operation: str,
        tracking_key: Optional[str],
        sglang_req_id: Any = _SGLANG_REQ_ID_UNSET,
        task_id: int = -1,
    ) -> _CacheOpContext:
        if sglang_req_id is _SGLANG_REQ_ID_UNSET:
            sglang_req_id = tracking_key
        return _CacheOpContext(
            operation=operation,
            sglang_req_id=sglang_req_id,
            start_ns=time.perf_counter_ns(),
            task_id=task_id,
        )

    def _pop_context(
        self,
        attr: str,
        key: Any,
        operation: str,
        task_id: int,
    ) -> _CacheOpContext:
        context = getattr(self, attr, {}).pop(key, None)
        if context is None:
            context = self._new_op_context(operation, str(key), task_id=task_id)
        context.operation = operation
        context.task_id = task_id
        return context

    def _log_cache_op(
        self,
        context: _CacheOpContext,
        action: str,
        status: str,
        *,
        task_id: Optional[int] = None,
        start_ns: Optional[int] = None,
        **fields: Any,
    ) -> None:
        if not self._sync_ctx.is_sync_leader:
            return
        if status in _WARNING_STATUSES:
            level = logging.WARNING
        elif action == "launch" or status in _INFO_STATUSES:
            level = logging.INFO
        else:
            level = logging.DEBUG
        if not logger.isEnabledFor(level):
            return

        if start_ns is None and action == "complete":
            start_ns = context.start_ns
        elapsed_s = (
            (time.perf_counter_ns() - start_ns) / 1e9 if start_ns is not None else None
        )

        values = [
            ("operation", context.operation),
            ("act", action),
            ("status", status),
        ]
        for name in ("direction", "blocks", "op_id", "graph_id"):
            if name in fields:
                values.append((name, fields.pop(name)))
        values.extend(
            [
                ("sglang_req_id", context.sglang_req_id),
                (
                    "flexkv_task_id",
                    context.task_id if task_id is None else task_id,
                ),
            ]
        )
        for name in (
            "child_task_ids",
            "flexkv_batch_task_id",
            "flexkv_batch_task_ids",
            "flexkv_task_ids",
        ):
            if name in fields:
                values.append((name, fields.pop(name)))
        if "transfer_mode" in fields:
            values.append(("mode", fields.pop("transfer_mode")))
        error = fields.pop("error", None)
        values.extend((name, value) for name, value in fields.items())
        if elapsed_s is not None:
            values.append((f"{context.operation}_time", elapsed_s))
        if error is not None:
            values.append(("error", error))

        formatted = []
        for name, value in values:
            if value is None:
                continue
            if name in {"sglang_req_id", "error"}:
                value = json.dumps(str(value), ensure_ascii=True)
            elif isinstance(value, bool):
                value = str(value).lower()
            elif name.endswith("_time"):
                value = f"{value:.4f}s"
            elif isinstance(value, float):
                value = f"{value:.3f}"
            elif isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value) or "-"
            formatted.append(f"{name}={value}")

        logger.log(
            level,
            "[%s][FlexKV-SGLang] %s",
            logging.getLevelName(level),
            " ".join(formatted),
        )

    # ------------------------------------------------------------------
    # Public API — lookup / load
    # ------------------------------------------------------------------

    def lookup_kv(
        self,
        token_ids: Sequence[int],
        token_mask: torch.Tensor,
        rid: Optional[str] = None,
        sglang_req_id: Any = _SGLANG_REQ_ID_UNSET,
        namespace: Optional[List[str]] = None,
    ) -> Tuple[int, int]:
        """Page-aligned prefix lookup against FlexKV.

        Args:
          token_ids: full token id sequence we'd like to check.
          token_mask: 1-D bool tensor or array, True for "this token is
            *not* already on GPU and is a candidate for load-back".
          rid: if set and hit > 0, the held FlexKV task id is stashed
            under this key so a later ``retrieve_kv(rid, slots)`` call
            can resolve it. If not set, the held task is cancelled when
            hit > 0 and the caller didn't ask to track it.
          sglang_req_id: business request ID used only for logs. This can be
            ``None`` when ``rid`` is an internal tracking key.

          namespace: cache identity components shared with store and prefetch.

        Returns:
          ``(fkv_task_id, hit_count)``. ``hit_count`` is page-aligned
          and may be smaller than the raw FlexKV match if the page
          floor truncated it.
        """
        context = self._new_op_context("lookup", rid, sglang_req_id)
        fkv_task_id = -1
        hit_length = 0
        lookup_error: Optional[Exception] = None

        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            tids_np = np.asarray(token_ids, dtype=np.int64)
            mask_np = self._as_numpy_mask(token_mask)
            try:
                # A DSv4 prefix is reusable only when the same host node also
                # owns its SWA window and compress-state sidecars.
                res = self.kv_manager.get_match(
                    token_ids=tids_np,
                    token_mask=mask_np,
                    cpu_only=getattr(self, "_chunked_prefetch", False),
                    swa_aware=self._swa_kv_pool is not None,
                    namespace=namespace,
                )
            except Exception as exc:  # noqa: BLE001
                lookup_error = exc
                res = None
            if res is None:
                fkv_task_id = -1
                hit_length = 0
            else:
                fkv_task_id, matched_mask = res
                hit_length = int(matched_mask.sum()) if matched_mask is not None else 0

        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                {
                    "task_id": fkv_task_id,
                    "hit": hit_length,
                },
                channel=FlexKVScatterChannel.LOOKUP,
            )
            fkv_task_id = payload["task_id"]
            hit_length = payload["hit"]

        # Page-align: FlexKV transfers whole pages.
        if hit_length > 0 and self.page_size > 1:
            aligned = (hit_length // self.page_size) * self.page_size
            if aligned < hit_length:
                logger.debug(
                    "[FlexKV] lookup_kv: page-aligning hit %d -> %d (page=%d)",
                    hit_length,
                    aligned,
                    self.page_size,
                )
            hit_length = aligned

        context.task_id = fkv_task_id
        lookup_status = (
            "failed"
            if lookup_error is not None
            else ("hit" if hit_length > 0 else "miss")
        )
        self._log_cache_op(
            context,
            "complete",
            lookup_status,
            tokens=len(token_ids),
            hit_tokens=hit_length,
            hit_blocks=hit_length // self.page_size,
            error=str(lookup_error) if lookup_error is not None else None,
        )

        # Decide what to do with the held task. Three cases:
        #   1. hit_length > 0 and rid given → stash for retrieve_kv later.
        #   2. hit_length > 0 and rid is None → cancel; caller can't use it.
        #   3. hit_length == 0 → no work to do; FlexKV already marked the
        #      empty graph COMPLETED inside get_match, cancel would warn.
        if hit_length > 0 and rid is not None and fkv_task_id >= 0:
            # SGLang can call match_prefix more than once for the same request
            # before init_load_back consumes the held lookup.  Replacing the
            # rid entry without cancelling the old FlexKV task loses its only
            # handle while that task still pins its matched CPU radix leaf.
            # Under remote-hit pressure this leaked one leaf lock per duplicate
            # lookup until every historical leaf became unevictable.
            replaced_task_id = self._pending_lookups.pop(rid, -1)
            replaced_context = self._pending_lookup_contexts.pop(rid, None)
            if (
                replaced_task_id >= 0
                and replaced_task_id != fkv_task_id
                and self._sync_ctx.is_sync_leader
            ):
                assert self.kv_manager is not None
                self.kv_manager.cancel([replaced_task_id])
            if replaced_context is not None:
                replaced_context.operation = "load"
                self._log_cache_op(
                    replaced_context,
                    "complete",
                    "cancelled",
                    reason="lookup_replaced",
                    replacement_task_id=fkv_task_id,
                )
            self._pending_lookups[rid] = fkv_task_id
            self._pending_lookup_contexts[rid] = context
            # The held GET has pinned its CPU prefix; transfer the result lease.
            self._release_prefetch_result(rid)
        elif hit_length > 0 and fkv_task_id >= 0 and self._sync_ctx.is_sync_leader:
            assert self.kv_manager is not None
            self.kv_manager.cancel([fkv_task_id])

        if rid is not None:
            self._release_prefetch_result(rid)
        return fkv_task_id, hit_length

    def sync_load_ready(self, local_ready: bool) -> bool:
        """Agree on allocation before any rank prepares a restore collective."""
        if self._sync_ctx.needs_sync:
            return bool(self._sync_ctx.all_reduce_min(int(local_ready)))
        return bool(local_ready)

    def release_pending(self, rid: str) -> None:
        """Cancel the task held by an earlier ``lookup_kv(rid=...)`` that
        won't be followed by a ``retrieve_kv`` (e.g. allocation failed)."""
        prepared = getattr(self, "_prepared_layerwise_loads", {}).get(rid)
        if prepared is not None:
            if getattr(self, "_layerwise_batch_launch_error", None):
                raise RuntimeError("Cannot cancel a layerwise batch with unknown launch status")
            if self._sync_ctx.is_sync_leader:
                self.kv_manager.cancel([prepared.task_id])
            self.layer_done_counter.rollback_prepared_transfer(
                prepared.task_id, prepared.producer_id, transfer_added=True
            )
            del self._prepared_layerwise_loads[rid]
            self._log_cache_op(prepared.context, "complete", "cancelled",
                               reason="prepared_layerwise_cancel")
        fkv_task_id = self._pending_lookups.pop(rid, -1)
        context = getattr(self, "_pending_lookup_contexts", {}).pop(rid, None)
        if fkv_task_id >= 0 and self._sync_ctx.is_sync_leader:
            assert self.kv_manager is not None
            self.kv_manager.cancel([fkv_task_id])
        if context is not None:
            context.operation = "load"
            self._log_cache_op(
                context,
                "complete",
                "cancelled",
                reason="lookup_released",
            )

    def retrieve_kv(
        self,
        rid: str,
        slot_mapping: torch.Tensor,
    ) -> int:
        """Synchronous load: ``launch`` + ``wait``.

        Returns the number of slots actually loaded. The caller is
        responsible for having allocated ``slot_mapping`` of length
        equal to ``hit_length`` from a prior ``lookup_kv``.
        """
        fkv_task_id = self._pending_lookups.pop(rid, -1)
        if fkv_task_id < 0:
            return 0
        context = self._pop_context(
            "_pending_lookup_contexts", rid, "load", fkv_task_id
        )
        load_start_ns = time.perf_counter_ns()

        slot_mapping_cpu = self._to_cpu_int64(slot_mapping)
        swa_slot_mapping = self._build_swa_slot_mapping(slot_mapping)
        swa_slots = 0 if swa_slot_mapping is None else int(swa_slot_mapping.numel())

        # Cross-node PP receivers must send their slot mapping back to
        # the TransferManagerOnRemote so the remote side knows where to
        # land the H2D copies on its own GPUs.
        if self._sync_ctx.should_send_slot_mapping_to_remote:
            self._send_slot_mapping_to_remote(fkv_task_id, slot_mapping_cpu)

        n = slot_mapping_cpu.numel()
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                self.kv_manager.launch(
                    task_ids=[fkv_task_id],
                    slot_mappings=[slot_mapping_cpu],
                    swa_slot_mappings=[swa_slot_mapping],
                    as_batch=True,
                    layerwise_transfer=False,
                )
            except Exception as exc:  # noqa: BLE001
                self._log_cache_op(
                    context,
                    "complete",
                    "failed",
                    start_ns=load_start_ns,
                    direction="H2D",
                    transfer_mode="no-layerwise",
                    stage="launch",
                    error=str(exc),
                )
                raise
            self._log_cache_op(
                context,
                "launch",
                "running",
                direction="H2D",
                transfer_mode="no-layerwise",
                slots=n,
                swa_slots=swa_slots,
            )
            try:
                resp = self.kv_manager.wait([fkv_task_id], timeout=30.0) or {}
            except Exception as exc:  # noqa: BLE001
                self._log_cache_op(
                    context,
                    "complete",
                    "failed",
                    start_ns=load_start_ns,
                    direction="H2D",
                    transfer_mode="no-layerwise",
                    error=str(exc),
                )
                raise
            status = _status_value(resp.get(fkv_task_id))
            success = status == KVResponseStatus.SUCCESS.value
            self._log_cache_op(
                context,
                "complete",
                status,
                start_ns=load_start_ns,
                direction="H2D",
                transfer_mode="no-layerwise",
                slots=n,
                swa_slots=swa_slots,
            )
            if not success:
                n = 0
        if self._sync_ctx.needs_sync:
            self._sync_ctx.barrier()
        return n

    def start_load_kv_layerwise(
        self,
        rid: str,
        slot_mapping: torch.Tensor,
        *,
        defer_launch: bool = False,
    ) -> Tuple[int, int]:
        """Layerwise load. Fires ``launch(layerwise_transfer=True)`` and
        returns ``(n_slots, producer_id)`` after registering the producer
        with the layer hook so the KV pool blocks on the right eventfds."""
        assert self.enable_layerwise and self.layer_done_counter is not None, (
            "start_load_kv_layerwise called but layerwise transfer is "
            "disabled. Set FLEXKV_ENABLE_LAYERWISE_TRANSFER=1."
        )
        if getattr(self, "_layerwise_batch_launch_error", None):
            raise RuntimeError("Previous layerwise batch has unknown launch status")
        if rid in getattr(self, "_prepared_layerwise_loads", {}):
            raise RuntimeError(f"Duplicate prepared layerwise load: rid={rid}")
        fkv_task_id = self._pending_lookups.pop(rid, -1)
        context = self._pop_context(
            "_pending_lookup_contexts", rid, "load", fkv_task_id
        )
        load_start_ns = time.perf_counter_ns()

        slot_mapping_cpu = self._to_cpu_int64(slot_mapping)
        swa_slot_mapping = self._build_swa_slot_mapping(slot_mapping)
        swa_slots = 0 if swa_slot_mapping is None else int(swa_slot_mapping.numel())
        n = slot_mapping_cpu.numel()

        # Counter selection must be a single global decision.  Choosing from
        # local timing state lets CP ranks cross a batch boundary at different
        # moments under overlap scheduling, so they can wait on eventfds that
        # the leader's FlexKV launch will never signal.
        generation = getattr(self, "_layerwise_generation", 0)
        producer_id = -1
        expected_signal_count = -1
        selected_new_counter = False
        selection_error = None
        if self._sync_ctx.is_sync_leader:
            try:
                if fkv_task_id < 0:
                    raise RuntimeError(f"No pending FlexKV lookup for rid={rid}")
                producer_id = self.layer_done_counter.update_producer()
                selected_new_counter = (
                    self.layer_done_counter.consumer_index != producer_id
                )
                expected_signal_count = self.layer_done_counter.expected_transfer_count(
                    producer_id,
                    allow_unclaimed_active=selected_new_counter,
                )
            except Exception as exc:  # noqa: BLE001
                selection_error = str(exc)

        manifest = {
            "cmd": CMD_LAYERWISE,
            "task_id": fkv_task_id,
            "producer_id": producer_id,
            "generation": generation,
            "slot_count": n,
            "expected_signal_count": expected_signal_count,
            "error": selection_error,
        }
        if self._sync_ctx.needs_sync:
            manifest = self._sync_ctx.scatter(
                manifest,
                channel=FlexKVScatterChannel.LAYERWISE_MANIFEST,
            )

        local_ok = 1
        local_error = None
        registered = False
        transfer_added = False
        try:
            if not isinstance(manifest, dict) or manifest.get("cmd") != CMD_LAYERWISE:
                raise RuntimeError("Invalid FlexKV layerwise manifest")
            if manifest.get("error"):
                raise RuntimeError(str(manifest["error"]))
            if int(manifest["generation"]) != generation:
                raise RuntimeError(
                    "FlexKV layerwise generation mismatch: "
                    f"local={generation}, leader={manifest.get('generation')}"
                )
            if int(manifest["task_id"]) != fkv_task_id or fkv_task_id < 0:
                raise RuntimeError(
                    "FlexKV layerwise task mismatch: "
                    f"local={fkv_task_id}, leader={manifest.get('task_id')}"
                )
            if int(manifest["slot_count"]) != n:
                raise RuntimeError(
                    "FlexKV layerwise slot-count mismatch: "
                    f"local={n}, leader={manifest.get('slot_count')}"
                )
            producer_id = int(manifest["producer_id"])
            expected_signal_count = int(manifest["expected_signal_count"])
            if not 0 <= producer_id < self.layer_done_counter.num_counters:
                raise RuntimeError(f"Invalid FlexKV producer_id={producer_id}")
            if expected_signal_count <= 0:
                raise RuntimeError(
                    f"Invalid FlexKV expected signal count {expected_signal_count}"
                )
            if not self._sync_ctx.is_sync_leader:
                local_expected = self.layer_done_counter.expected_transfer_count(
                    producer_id
                )
                if local_expected != expected_signal_count:
                    raise RuntimeError(
                        "FlexKV layerwise batch boundary differs across ranks: "
                        f"local_expected={local_expected}, "
                        f"leader_expected={expected_signal_count}"
                    )
        except Exception as exc:  # noqa: BLE001
            local_ok = 0
            local_error = str(exc)

        # Include remote slot metadata and local counter registration in the
        # same all-rank preflight. Nothing reaches KVManager.launch until every
        # rank reports that it is ready to consume the leader's counter.
        if local_ok and self._sync_ctx.should_send_slot_mapping_to_remote:
            try:
                self._send_slot_mapping_to_remote(fkv_task_id, slot_mapping_cpu)
            except Exception as exc:  # noqa: BLE001
                local_ok = 0
                local_error = f"remote slot metadata failed: {exc}"

        if local_ok:
            try:
                if self._sync_ctx.is_sync_leader:
                    self.layer_done_counter.register_task(fkv_task_id, producer_id)
                else:
                    self.layer_done_counter.register_task_with_explicit_counter_id(
                        fkv_task_id, producer_id
                    )
                registered = True
                self.layer_done_counter.set_consumer(fkv_task_id)
                transfer_added = True
                local_signal_count = self.layer_done_counter.events[
                    producer_id
                ].pending_transfers()
                if local_signal_count != expected_signal_count:
                    raise RuntimeError(
                        "FlexKV layerwise signal-count changed during registration: "
                        f"local={local_signal_count}, leader={expected_signal_count}"
                    )
            except Exception as exc:  # noqa: BLE001
                local_ok = 0
                local_error = f"counter registration failed: {exc}"

        self._layerwise_generation = generation + 1
        all_ranks_ok = (
            self._sync_ctx.all_reduce_min(local_ok)
            if self._sync_ctx.needs_sync
            else local_ok
        )
        if all_ranks_ok == 0:
            cleanup_ok = 1
            cleanup_error = None
            try:
                if registered or transfer_added:
                    self.layer_done_counter.rollback_prepared_transfer(
                        fkv_task_id,
                        producer_id,
                        transfer_added=transfer_added,
                    )
                elif self._sync_ctx.is_sync_leader and selected_new_counter:
                    self.layer_done_counter.abort_unregistered_selection(producer_id)
            except Exception as exc:  # noqa: BLE001
                cleanup_ok = 0
                cleanup_error = str(exc)
            all_cleanup_ok = (
                self._sync_ctx.all_reduce_min(cleanup_ok)
                if self._sync_ctx.needs_sync
                else cleanup_ok
            )
            if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
                try:
                    if fkv_task_id >= 0:
                        self.kv_manager.cancel([fkv_task_id])
                except Exception as exc:  # noqa: BLE001
                    self._log_cache_op(
                        context,
                        "cleanup",
                        "failed",
                        start_ns=load_start_ns,
                        direction="H2D",
                        transfer_mode="layerwise",
                        stage="prelaunch_cancel",
                        error=str(exc),
                    )
            if all_cleanup_ok == 0:
                raise RuntimeError(
                    "FlexKV layerwise counter rollback failed on at least one rank: "
                    f"{cleanup_error or 'remote rank failure'}"
                )
            self._log_cache_op(
                context,
                "complete",
                "failed",
                start_ns=load_start_ns,
                direction="H2D",
                transfer_mode="layerwise",
                stage="preflight",
                rank=self._sync_ctx.world_rank,
                generation=generation,
                producer_id=producer_id,
                error=local_error or "another rank rejected the pre-launch state",
            )
            return 0, -1

        self._log_cache_op(
            context,
            "register",
            "ready",
            direction="H2D",
            transfer_mode="layerwise",
            rank=self._sync_ctx.world_rank,
            generation=generation,
            producer_id=producer_id,
            expected_signal_count=expected_signal_count,
        )

        if defer_launch:
            if not hasattr(self, "_prepared_layerwise_loads"):
                self._prepared_layerwise_loads = {}
            self._prepared_layerwise_loads[rid] = _PreparedLayerwiseLoad(
                fkv_task_id, producer_id, slot_mapping_cpu, swa_slot_mapping,
                context, load_start_ns,
            )
            return n, producer_id

        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                launched_task_ids = self.kv_manager.launch(
                    task_ids=[fkv_task_id],
                    slot_mappings=[slot_mapping_cpu],
                    swa_slot_mappings=[swa_slot_mapping],
                    as_batch=True,
                    layerwise_transfer=True,
                    counter_id=producer_id,
                )
            except Exception as exc:  # noqa: BLE001
                self._log_cache_op(
                    context,
                    "complete",
                    "failed",
                    start_ns=load_start_ns,
                    direction="H2D",
                    transfer_mode="layerwise",
                    stage="launch",
                    error=str(exc),
                )
                raise
            # A layerwise launch is always merged into a batch task. Track the
            # IDs returned by launch(), not the lookup task ID that the merge
            # removes from FlexKV's task table.
            self._launched_load_tids.extend(launched_task_ids)
            for launched_task_id in launched_task_ids:
                self._launched_load_contexts[launched_task_id] = replace(
                    context,
                    task_id=launched_task_id,
                    start_ns=load_start_ns,
                )
            self._log_cache_op(
                context,
                "launch",
                "running",
                direction="H2D",
                flexkv_batch_task_ids=launched_task_ids,
                transfer_mode="layerwise",
                slots=n,
                swa_slots=swa_slots,
                producer_id=producer_id,
            )

        return n, producer_id

    def flush_layerwise_loads(self, request_ids: List[str]) -> None:
        """Submit the admitted compute batch through the existing graph merger.

        Preparation retains each request's destinations and runs the existing
        all-rank preflight, but submits no DMA. The scheduler calls this once
        after admission and before preparing/executing the forward batch.
        """
        prepared = getattr(self, "_prepared_layerwise_loads", {})
        if not prepared:
            return
        if getattr(self, "_layerwise_batch_launch_error", None):
            raise RuntimeError("Previous layerwise batch has unknown launch status")
        members = [prepared[rid] for rid in request_ids if rid in prepared]
        task_ids = [m.task_id for m in members]
        producer_id = members[0].producer_id if members else -1
        entries = [
            (rid, prepared[rid].task_id, prepared[rid].producer_id,
             int(prepared[rid].slot_mapping.numel()),
             0 if prepared[rid].swa_slot_mapping is None else
             int(prepared[rid].swa_slot_mapping.numel()))
            for rid in request_ids if rid in prepared
        ]
        manifest = {"entries": entries}
        if self._sync_ctx.needs_sync:
            manifest = self._sync_ctx.scatter(
                manifest, channel=FlexKVScatterChannel.LAYERWISE_BATCH
            )
        local_error = None
        try:
            if manifest != {"entries": entries}:
                raise RuntimeError("Layerwise batch members differ across ranks")
            if len(set(request_ids)) != len(request_ids):
                raise RuntimeError("Duplicate request in admitted layerwise batch")
            if set(prepared) - set(request_ids):
                raise RuntimeError("Prepared restore was not admitted to this batch")
            if any(m.producer_id != producer_id for m in members):
                raise RuntimeError("Layerwise batch spans multiple producer counters")
            self.layer_done_counter.validate_prepared_batch(task_ids, producer_id)
        except Exception as exc:
            local_error = str(exc)
        ready = int(local_error is None)
        if self._sync_ctx.needs_sync:
            ready = self._sync_ctx.all_reduce_min(ready)
        if not ready:
            # Nothing was submitted. Retain prepared records for cancellation
            # or reset; never let the model consume an incomplete batch.
            raise RuntimeError(local_error or "Another rank rejected the layerwise batch")

        self.layer_done_counter.coalesce_prepared_batch(task_ids, producer_id)
        result = {"error": None, "task_ids": []}
        # From this point, failure may mean the server accepted the launch.
        # Retain mappings and refuse reset unless launch was acknowledged.
        self._layerwise_batch_launch_error = "Layerwise batch launch is unresolved"
        if self._sync_ctx.is_sync_leader:
            try:
                result["task_ids"] = self.kv_manager.launch(
                    task_ids=task_ids,
                    slot_mappings=[m.slot_mapping for m in members],
                    swa_slot_mappings=[m.swa_slot_mapping for m in members],
                    as_batch=True, layerwise_transfer=True, counter_id=producer_id,
                )
                if len(result["task_ids"]) != 1:
                    raise RuntimeError("Layerwise merge did not return one batch task")
                self._launched_load_tids.extend(result["task_ids"])
                for tid in result["task_ids"]:
                    self._launched_load_contexts[tid] = replace(
                        members[0].context, task_id=tid,
                        start_ns=min(m.start_ns for m in members),
                    )
            except Exception as exc:
                result["error"] = str(exc) or type(exc).__name__
                actual_ids = getattr(exc, "launched_task_ids", None) or result["task_ids"]
                for tid in actual_ids or ():
                    if tid not in self._launched_load_tids:
                        self._launched_load_tids.append(tid)
                    self._launched_load_contexts[tid] = replace(
                        members[0].context, task_id=tid,
                        start_ns=min(m.start_ns for m in members),
                    )
                result["task_ids"] = list(actual_ids or ())
        if self._sync_ctx.needs_sync:
            result = self._sync_ctx.scatter(
                result, channel=FlexKVScatterChannel.LAYERWISE_BATCH, blocking=True
            )
        if result["error"]:
            self._layerwise_batch_launch_error = result["error"]
            raise RuntimeError(f"Layerwise batch launch failed: {result['error']}")
        self._layerwise_batch_launch_error = None
        prepared.clear()
        for m in members:
            self._log_cache_op(
                m.context, "launch", "running", direction="H2D",
                flexkv_batch_task_ids=result["task_ids"], transfer_mode="layerwise",
                slots=int(m.slot_mapping.numel()), producer_id=producer_id,
                merged_requests=len(members), expected_signal_count=1,
            )

    def drain_launched_loads(self, threshold: int = 1) -> None:
        """Periodic non-blocking sweep on long-lived launched tasks so the
        FlexKV pipe doesn't accumulate. No-op on non-leader ranks."""
        if not self._sync_ctx.is_sync_leader or self.kv_manager is None:
            return
        if not self._launched_load_tids:
            # Without chunked prefetch the engine only advances when a client
            # polls it, so a cancelled graph would keep its CPU blocks pinned
            # until the next request arrives. Nudge it at a bounded rate: the
            # scheduler calls this every idle tick, which would otherwise be
            # thousands of synchronous RPCs per second. Chunked prefetch runs
            # the TaskRuntime thread, which already drains graphs on its own.
            if not getattr(self, "_chunked_prefetch", False):
                now = time.monotonic()
                last = getattr(self, "_last_idle_drive_ts", None)
                if last is None or now - last >= _IDLE_DRIVE_INTERVAL_S:
                    self._last_idle_drive_ts = now
                    self.kv_manager.try_wait(task_ids=[])
            return
        if len(self._launched_load_tids) < threshold:
            return
        task_ids = list(dict.fromkeys(self._launched_load_tids))
        contexts = getattr(self, "_launched_load_contexts", {})
        try:
            responses = (
                self.kv_manager.wait(task_ids, timeout=0.0, completely=True) or {}
            )
        except Exception as exc:  # noqa: BLE001
            context = next(
                (contexts[task_id] for task_id in task_ids if task_id in contexts),
                None,
            )
            context = context or self._new_op_context("load", "-")
            self._log_cache_op(
                context,
                "poll",
                "failed",
                task_id=-1,
                direction="H2D",
                flexkv_batch_task_ids=task_ids,
                transfer_mode="layerwise",
                error=str(exc),
            )
            return
        for task_id, response in responses.items():
            status = _status_value(response)
            if not _is_terminal_status(status):
                continue
            context = self._pop_context(
                "_launched_load_contexts", task_id, "load", task_id
            )
            self._log_cache_op(
                context,
                "complete",
                status,
                direction="H2D",
                transfer_mode="layerwise",
            )
        self._launched_load_tids = [
            task_id
            for task_id in task_ids
            if task_id not in responses
            or not _is_terminal_status(_status_value(responses[task_id]))
        ]

    # ------------------------------------------------------------------
    # Public API — store
    # ------------------------------------------------------------------

    @property
    def is_store_sync_leader(self) -> bool:
        """Whether this rank owns the authoritative FlexKV store decision."""
        return bool(self._sync_ctx.is_sync_leader)

    @property
    def supports_async_store_slot_mapping(self) -> bool:
        """Whether a leader-only pinned slot copy is valid for this topology.

        PP stages need their own stage-local mapping, and SWA needs a GPU-side
        full-to-SWA translation. Keep those paths on the synchronous
        implementation until they have an explicit sideband.
        """
        # TODO: add stage-local mapping sidebands for PP and carry
        # the GPU full-to-SWA translation in the asynchronous store protocol.
        return not bool(getattr(self._sync_ctx, "is_pp_active", False)) and (
            self._swa_kv_pool is None
        )

    def sync_ready_store_rids(self, ready_rids: List[str]) -> List[str]:
        """Fan out the leader's ready pinned-copy set to every cache rank."""
        payload = list(ready_rids) if self._sync_ctx.is_sync_leader else []
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                payload,
                channel=FlexKVScatterChannel.STORE_READY,
            )
        return list(payload)

    def store_kv(
        self,
        rid: str,
        token_ids: List[int],
        kv_indices: torch.Tensor,
        sglang_req_id: Any = _SGLANG_REQ_ID_UNSET,
        namespace: Optional[List[str]] = None,
    ) -> int:
        """Schedule a write back from GPU into FlexKV.

        The sync leader runs ``put_match`` and ``launch``. Its result is
        fanned out to every PP/CP/TP rank so every rank retains ownership
        of the source GPU slots until the store completes. A remote PP
        stage leader also forwards its local slot mapping.

        Returns the FlexKV task id of the in-flight store, or -1 if
        nothing needed to be written.
        """
        context = self._new_op_context("store", rid, sglang_req_id)
        with self._store_profile_scope("flexkv.connector.store.tokens_to_numpy"):
            token_ids_np = np.asarray(token_ids, dtype=np.int64)
        n = len(token_ids_np)
        if n != len(kv_indices):
            raise ValueError(
                f"store_kv: token_ids has {n} entries but kv_indices "
                f"has {len(kv_indices)} entries"
            )

        # Page-align inputs *before* put_match so the FlexKV allocator
        # only reserves slots that line up with the slot_mapping we send.
        if self.page_size > 1:
            aligned_len = (n // self.page_size) * self.page_size
            if aligned_len == 0:
                self._log_cache_op(
                    context,
                    "complete",
                    "skipped",
                    direction="D2H",
                    reason="partial_block",
                    tokens=0,
                )
                return -1
            if aligned_len < n:
                token_ids_np = token_ids_np[:aligned_len]
                kv_indices = kv_indices[:aligned_len]
                n = aligned_len

        store_start = {
            "rid": rid,
            "task_id": -1,
            "active": False,
            "slot_count": n,
            "unmatched_count": 0,
            "unmatched_mask": [],
            "error": "",
        }
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            match_error: Optional[Exception] = None
            try:
                with self._store_profile_scope("flexkv.connector.store.put_match"):
                    res = self.kv_manager.put_match(
                        token_ids=token_ids_np, token_mask=None, namespace=namespace
                    )
            except Exception as exc:  # noqa: BLE001
                match_error = exc
                res = None
            if res is None:
                self._log_cache_op(
                    context,
                    "complete",
                    "failed",
                    direction="D2H",
                    stage="put_match",
                    reason="no_response" if match_error is None else None,
                    error=str(match_error) if match_error is not None else None,
                )
            else:
                fkv_task_id, unmatched_mask = res
                # TP/CP followers only need the task id to retain their local
                # radix-node ownership.  Broadcasting one Python bool per
                # prompt token made the common TP-only write-through path
                # serialize/deserialise tens of thousands of objects per
                # request.  Only cross-node PP receivers need the mask to
                # construct their stage-local slot mapping.
                needs_remote_mask = bool(
                    getattr(self._sync_ctx, "is_cross_node_pp", False)
                )
                unmatched_count = int(np.count_nonzero(unmatched_mask))
                mask_list = (
                    unmatched_mask.tolist()
                    if needs_remote_mask and hasattr(unmatched_mask, "tolist")
                    else list(unmatched_mask)
                    if needs_remote_mask
                    else []
                )
                store_start.update(
                    task_id=int(fkv_task_id),
                    unmatched_count=unmatched_count,
                    unmatched_mask=mask_list,
                )
                context.task_id = int(fkv_task_id)
                if unmatched_count > 0:
                    with self._store_profile_scope(
                        "flexkv.connector.store.filter_slot_mapping"
                    ):
                        filtered = kv_indices[unmatched_mask]
                    with self._store_profile_scope(
                        "flexkv.connector.store.slot_mapping_to_cpu"
                    ):
                        slot_mapping_cpu = self._to_cpu_int64(filtered)
                        swa_slot_mapping = self._build_swa_slot_mapping(filtered)
                    swa_slots = (
                        0
                        if swa_slot_mapping is None
                        else int(swa_slot_mapping.numel())
                    )
                    try:
                        with self._store_profile_scope(
                            "flexkv.connector.store.kvmanager_launch"
                        ):
                            self.kv_manager.launch(
                                task_ids=[fkv_task_id],
                                slot_mappings=[slot_mapping_cpu],
                                swa_slot_mappings=[swa_slot_mapping],
                                as_batch=False,
                                layerwise_transfer=False,
                            )
                    except Exception as exc:  # noqa: BLE001
                        store_start["error"] = str(exc)
                        self._log_cache_op(
                            context,
                            "complete",
                            "failed",
                            direction="D2H",
                            transfer_mode="no-layerwise",
                            stage="launch",
                            error=str(exc),
                        )
                    else:
                        store_start["active"] = True
                        self._log_cache_op(
                            context,
                            "launch",
                            "running",
                            direction="D2H",
                            transfer_mode="no-layerwise",
                            tokens=unmatched_count,
                            slots=int(slot_mapping_cpu.numel()),
                            swa_slots=swa_slots,
                        )
                else:
                    self._log_cache_op(
                        context,
                        "complete",
                        "skipped",
                        direction="D2H",
                        reason="already_cached",
                        tokens=0,
                    )

        if self._sync_ctx.needs_sync:
            with self._store_profile_scope("flexkv.connector.store.scatter_start"):
                store_start = self._sync_ctx.scatter(
                    store_start,
                    channel=FlexKVScatterChannel.STORE_START,
                )
        if store_start.get("rid") != rid:
            raise RuntimeError(
                "[FlexKV] store-start rid mismatch: "
                f"local={rid!r}, leader={store_start.get('rid')!r}"
            )
        if store_start.get("error"):
            raise RuntimeError(
                f"[FlexKV] store launch failed: {store_start['error']}"
            )
        if not store_start.get("active"):
            return -1

        fkv_task_id = int(store_start["task_id"])
        slot_count = int(store_start.get("slot_count", -1))
        mask_list = store_start.get("unmatched_mask", [])
        if fkv_task_id < 0 or slot_count != len(kv_indices):
            raise RuntimeError(
                "[FlexKV] invalid store-start payload: "
                f"task_id={fkv_task_id}, slot_count={slot_count}, "
                f"slot_len={len(kv_indices)}"
            )

        # Cross-node PP needs the local PP stage's physical slot mapping.
        if self._sync_ctx.should_send_slot_mapping_to_remote:
            if len(mask_list) != len(kv_indices):
                raise RuntimeError(
                    "[FlexKV] invalid cross-node store mask: "
                    f"mask_len={len(mask_list)}, slot_len={len(kv_indices)}"
                )
            unmatched_mask = torch.as_tensor(
                mask_list, dtype=torch.bool, device=kv_indices.device
            )
            filtered = kv_indices[unmatched_mask]
            self._send_slot_mapping_to_remote(
                fkv_task_id, self._to_cpu_int64(filtered)
            )

        # This is deliberately done on every rank. SGLang uses the returned
        # positive task id to keep radix-node locks (and GPU slots) alive.
        context.task_id = fkv_task_id
        self._inflight_stores[rid] = fkv_task_id
        self._inflight_store_contexts[rid] = context
        return fkv_task_id

    def check_completed_stores(self) -> List[str]:
        """Return rids whose stores have completed since the last call."""
        completed_rids: List[str] = []
        completed_by_rid: Dict[str, Any] = {}

        if (self._sync_ctx.is_sync_leader and self.kv_manager is not None
                and self._inflight_stores):
            fk_to_rid = {v: k for k, v in self._inflight_stores.items()}
            try:
                completed_dict = (
                    self.kv_manager.wait(
                        list(fk_to_rid), timeout=0.0, completely=True
                    )
                    or {}
                )
            except Exception as exc:  # noqa: BLE001
                rid = next(iter(self._inflight_stores))
                context = getattr(self, "_inflight_store_contexts", {}).get(rid)
                context = context or self._new_op_context(
                    "store", rid, task_id=self._inflight_stores[rid]
                )
                self._log_cache_op(
                    context,
                    "poll",
                    "failed",
                    task_id=-1,
                    flexkv_task_ids=list(fk_to_rid),
                    error=str(exc),
                )
                completed_dict = {}
            for fk_tid, response in completed_dict.items():
                status = _status_value(response)
                if not _is_terminal_status(status):
                    continue
                rid = fk_to_rid[fk_tid]
                completed_rids.append(rid)
                completed_by_rid[rid] = response

        if self._sync_ctx.needs_sync:
            completed_rids = self._sync_ctx.scatter(
                completed_rids,
                channel=FlexKVScatterChannel.STORE_COMPLETION,
            )
        for rid in completed_rids:
            fk_tid = self._inflight_stores.pop(rid, None)
            if fk_tid is None:
                continue
            if self._sync_ctx.is_sync_leader:
                context = self._pop_context(
                    "_inflight_store_contexts", rid, "store", fk_tid
                )
                response = completed_by_rid.get(rid)
                self._log_cache_op(
                    context,
                    "complete",
                    _status_value(response),
                    direction="D2H",
                    transfer_mode="no-layerwise",
                )
            else:
                getattr(self, "_inflight_store_contexts", {}).pop(rid, None)
        return completed_rids

    def wait_store(self, rid: str, timeout: float = 30.0) -> bool:
        """Block until a single store task identified by ``rid`` finishes."""
        fkv_task_id = self._inflight_stores.get(rid, -1)
        if fkv_task_id < 0:
            return True
        context = getattr(self, "_inflight_store_contexts", {}).get(rid)
        context = context or self._new_op_context(
            "store", rid, task_id=fkv_task_id
        )
        if not self._sync_ctx.is_sync_leader or self.kv_manager is None:
            self._inflight_stores.pop(rid, None)
            getattr(self, "_inflight_store_contexts", {}).pop(rid, None)
            return True
        try:
            resp = self.kv_manager.wait(
                [fkv_task_id], timeout=timeout, completely=True
            ) or {}
        except Exception as exc:  # noqa: BLE001
            self._log_cache_op(
                context,
                "poll",
                "failed",
                direction="D2H",
                transfer_mode="no-layerwise",
                error=str(exc),
            )
            return False
        status = _status_value(resp.get(fkv_task_id))
        if not _is_terminal_status(status):
            self._log_cache_op(
                context,
                "poll",
                status,
                direction="D2H",
                transfer_mode="no-layerwise",
            )
            return False
        self._inflight_stores.pop(rid, None)
        self._pop_context(
            "_inflight_store_contexts", rid, "store", fkv_task_id
        )
        success = status == KVResponseStatus.SUCCESS.value
        self._log_cache_op(
            context,
            "complete",
            status,
            direction="D2H",
            transfer_mode="no-layerwise",
        )
        return success

    # ------------------------------------------------------------------
    # Public API — prefetch
    # ------------------------------------------------------------------

    def prefetch_async(
        self,
        rid: str,
        token_ids: List[int],
        sglang_req_id: Any = _SGLANG_REQ_ID_UNSET,
        namespace: Optional[List[str]] = None,
        *, candidate_start_token: int = 0,
    ) -> int:
        if not self._prefetch_enabled or not rid:
            return -1
        if getattr(self, "_chunked_prefetch", False):
            return self._start_chunked_prefetch(rid, token_ids, candidate_start_token, namespace)
        context = self._new_op_context("prefetch", rid, sglang_req_id)
        task_id = -1
        planned_tokens = 0
        prefetch_error: Optional[Exception] = None
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                prefetch_result = self.kv_manager.prefetch_async(
                    token_ids=np.asarray(token_ids, dtype=np.int64),
                    namespace=namespace,
                    swa_aware=self._swa_kv_pool is not None,
                )
                # KVManager currently returns
                # ``(task_id, actual_prefetch_tokens)`` even though older
                # versions annotated this API as returning an int.  Accept
                # both shapes so the SGLang connector remains compatible
                # across FlexKV versions and never scatters a tuple as the
                # task id.
                task_id = (
                    prefetch_result[0]
                    if isinstance(prefetch_result, tuple)
                    else prefetch_result
                )
                if isinstance(prefetch_result, tuple) and len(prefetch_result) > 1:
                    planned_tokens = int(prefetch_result[1])
            except Exception as exc:  # noqa: BLE001
                prefetch_error = exc
                task_id = -1
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                {"task_id": task_id, "planned_tokens": planned_tokens},
                channel=FlexKVScatterChannel.PREFETCH_START,
            )
            task_id = payload["task_id"]
            planned_tokens = int(payload["planned_tokens"])
        if task_id >= 0:
            context.task_id = task_id
            self._ongoing_prefetches[rid] = task_id
            self._prefetch_contexts[rid] = context
            self._prefetch_planned_tokens[rid] = planned_tokens
            self._log_cache_op(
                context,
                "launch",
                "running",
                tokens=len(token_ids),
            )
        elif self._sync_ctx.is_sync_leader:
            self._log_cache_op(
                context,
                "complete",
                "failed",
                reason="task_not_created" if prefetch_error is None else None,
                error=str(prefetch_error) if prefetch_error is not None else None,
            )
        return task_id

    def check_prefetch_progress(self, rid: str) -> bool:
        if getattr(self, "_chunked_prefetch", False):
            return self._check_chunked_prefetch(rid)
        if not self._prefetch_enabled:
            return True
        task_id = self._ongoing_prefetches.get(rid, -1)
        if task_id < 0:
            return True
        done = False
        status = "running"
        loaded_tokens = 0
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                completed = self.kv_manager.try_wait(task_ids=[task_id]) or {}
            except Exception as exc:  # noqa: BLE001
                context = getattr(self, "_prefetch_contexts", {}).get(rid)
                context = context or self._new_op_context(
                    "prefetch", rid, task_id=task_id
                )
                self._log_cache_op(
                    context,
                    "poll",
                    "failed",
                    error=str(exc),
                )
                completed = {}
            if task_id in completed:
                response = completed[task_id]
                status = _status_value(response)
                done = _is_terminal_status(status)
                if done and status == KVResponseStatus.SUCCESS.value:
                    # Current FlexKV finalizes the authoritative reusable L3
                    # prefix in KVResponse.return_mask after REMOTE2H commits
                    # into the CPU tree.  Launch-time prefetch_async returns
                    # only a task id, so using that result for accounting
                    # silently reports real storage hits as host hits.
                    return_mask = getattr(response, "return_mask", None)
                    if return_mask is None:
                        # Compatibility with older managers that returned the
                        # planned token count at launch and no completion mask.
                        loaded_tokens = self._prefetch_planned_tokens.get(rid, 0)
                    elif isinstance(return_mask, list):
                        loaded_tokens = sum(
                            int(np.count_nonzero(mask)) for mask in return_mask
                        )
                    else:
                        loaded_tokens = int(np.count_nonzero(return_mask))
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                {
                    "done": done,
                    "status": status,
                    "loaded_tokens": loaded_tokens,
                },
                channel=FlexKVScatterChannel.PREFETCH_PROGRESS,
            )
            done = payload["done"]
            status = payload["status"]
            loaded_tokens = int(payload["loaded_tokens"])
        if done:
            self._ongoing_prefetches.pop(rid, None)
            self._prefetch_planned_tokens.pop(rid, None)
            self._prefetch_loaded_tokens[rid] = loaded_tokens
            context = self._pop_context("_prefetch_contexts", rid, "prefetch", task_id)
            self._log_cache_op(
                context,
                "complete",
                status,
            )
        return done

    def _start_chunked_prefetch(self, rid, token_ids, candidate_start_token=0, namespace=None):
        if rid in self._prefetch_sessions:
            return self._prefetch_sessions[rid].session_id
        self._release_prefetch_result(rid)
        payload = {"handle": None, "error": None}
        if self._sync_ctx.is_sync_leader:
            try:
                payload["handle"] = self.kv_manager.start_prefetch(
                    np.asarray(token_ids, dtype=np.int64),
                    replace(
                        self._chunked_prefetch_options,
                        candidate_start_token=candidate_start_token,
                        swa_aware=self._swa_kv_pool is not None,
                    ),
                    namespace=namespace,
                )
            except Exception as exc:
                payload["error"] = str(exc)
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                payload, channel=FlexKVScatterChannel.PREFETCH_START
            )
        if payload["error"]:
            logger.warning("[FlexKV] prefetch admission failed: %s", payload["error"])
            return -1
        handle = payload["handle"]
        self._prefetch_sessions[rid] = handle
        return handle.session_id

    def _check_chunked_prefetch(self, rid):
        handle = self._prefetch_sessions.get(rid)
        if handle is None:
            return True
        payload = {"snapshot": None, "error": None}
        if self._sync_ctx.is_sync_leader:
            try:
                payload["snapshot"] = self.kv_manager.progress_prefetch(
                    [handle], demand_handles=[handle]
                )[handle]
            except Exception as exc:
                payload["error"] = str(exc)
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                payload, channel=FlexKVScatterChannel.PREFETCH_PROGRESS
            )
        if payload["error"]:
            # Do not let any rank compute with a stale/incomplete prefix.
            raise RuntimeError("FlexKV prefetch progress failed: " + payload["error"])
        snapshot = payload["snapshot"]
        if not snapshot.terminal:
            return False
        self._prefetch_sessions.pop(rid)
        self._prefetch_result_sessions[rid] = handle
        self._prefetch_loaded_tokens[rid] = snapshot.loaded_tokens if snapshot.lease_valid else 0
        self._prefetch_loaded_spans[rid] = snapshot.l3_loaded_spans if snapshot.lease_valid else ()
        return True

    def _release_prefetch_result(self, rid):
        handle = getattr(self, "_prefetch_result_sessions", {}).pop(rid, None)
        if handle is not None and self._sync_ctx.is_sync_leader:
            self.kv_manager.release_prefetch(handle)

    def pop_prefetch_loaded_span(self, rid):
        spans = getattr(self, "_prefetch_loaded_spans", {}).pop(rid, ())
        tokens = self.pop_prefetch_loaded_tokens(rid)
        return tokens, spans[0][0] if spans else None

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        """Return the successfully materialized REMOTE2H prefix once."""
        return int(self._prefetch_loaded_tokens.pop(rid, 0))

    def cancel_prefetch(self, rid: str) -> None:
        if getattr(self, "_chunked_prefetch", False):
            handle = self._prefetch_sessions.pop(rid, None)
            if handle is not None and self._sync_ctx.is_sync_leader:
                # release seals the session, whose resource ledger survives
                # this request-facing tracking entry until every graph drains.
                self.kv_manager.release_prefetch(handle)
            self._release_prefetch_result(rid)
            self._prefetch_loaded_spans.pop(rid, None)
            self._prefetch_loaded_tokens.pop(rid, None)
            self.release_pending(rid)
            return
        lookup_task = self._pending_lookups.pop(rid, None)
        if lookup_task is not None and self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            self.kv_manager.cancel([lookup_task])
        lookup_context = getattr(self, "_pending_lookup_contexts", {}).pop(rid, None)
        if lookup_context is not None:
            lookup_context.operation = "load"
            self._log_cache_op(
                lookup_context,
                "complete",
                "cancelled",
                reason="caller_cancelled",
            )
        # Running tasks keep ownership until their graph drains (see
        # KVTaskManager._cancel_task); unlaunched tasks roll back immediately.
        task_id = self._ongoing_prefetches.pop(rid, -1)
        if task_id >= 0 and self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            self.kv_manager.cancel([task_id])
        self._prefetch_planned_tokens.pop(rid, None)
        self._prefetch_loaded_tokens.pop(rid, None)
        context = getattr(self, "_prefetch_contexts", {}).pop(rid, None)
        if context is not None:
            self._log_cache_op(
                context,
                "complete",
                "cancelled",
                task_id=task_id,
                reason="caller_cancelled",
            )

    # ------------------------------------------------------------------
    # Layerwise transfer hooks
    # ------------------------------------------------------------------

    def register_layer_transfer_counter(self, kvcache: Any) -> None:
        """Register the FlexKVLayerDoneCounter onto sglang's KV pool so
        each forward layer blocks on its eventfd. No-op when layerwise
        is disabled."""
        if (
            self.layer_done_counter is None
            or kvcache is None
            or not hasattr(kvcache, "register_layer_transfer_counter")
        ):
            return
        kvcache.register_layer_transfer_counter(self.layer_done_counter)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        if getattr(self, "_layerwise_batch_launch_error", None):
            raise RuntimeError(
                "Cannot reset after an unacknowledged layerwise batch launch: "
                + self._layerwise_batch_launch_error
            )
        for rid in list(getattr(self, "_prepared_layerwise_loads", {})):
            self.release_pending(rid)
        if getattr(self, "_chunked_prefetch", False):
            for rid in list(self._prefetch_sessions):
                self.cancel_prefetch(rid)
            for rid in list(self._prefetch_result_sessions):
                self._release_prefetch_result(rid)
            self._prefetch_loaded_spans.clear()
        # Layerwise launch merges the lookup task into a new batch task. Wait
        # for those returned batch task IDs before the scheduler clears GPU
        # slot ownership or the eventfd producer ring is reused. ``completely``
        # is required: an early task-end response is not sufficient to prove
        # that the worker has stopped writing layer completion eventfds.
        load_reset_status = {"ok": True, "error": ""}
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            launched = list(dict.fromkeys(self._launched_load_tids))
            if launched:
                try:
                    responses = self.kv_manager.wait(
                        launched, timeout=30.0, completely=True
                    )
                    responses = responses or {}
                    for task_id, response in responses.items():
                        status = _status_value(response)
                        context = self._pop_context(
                            "_launched_load_contexts", task_id, "load", task_id
                        )
                        self._log_cache_op(
                            context,
                            "complete",
                            status,
                            direction="H2D",
                            transfer_mode="layerwise",
                            source="reset",
                        )
                    failed = [
                        task_id
                        for task_id in launched
                        if task_id not in responses
                        or responses[task_id].status != KVResponseStatus.SUCCESS
                    ]
                    if failed:
                        load_reset_status = {
                            "ok": False,
                            "error": (
                                "layerwise load tasks did not finish during reset: "
                                f"{failed}"
                            ),
                        }
                except Exception as exc:  # noqa: BLE001
                    load_reset_status = {
                        "ok": False,
                        "error": f"waiting for layerwise loads during reset failed: {exc}",
                    }
        if self._sync_ctx.needs_sync:
            load_reset_status = self._sync_ctx.scatter(
                load_reset_status,
                channel=FlexKVScatterChannel.RESET,
                blocking=True,
            )
        if not load_reset_status["ok"]:
            raise RuntimeError(f"[FlexKV] {load_reset_status['error']}")

        # Drop pending lookups (cancel their held tasks on the leader).
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            pending = [tid for tid in self._pending_lookups.values() if tid >= 0]
            if pending:
                try:
                    self.kv_manager.cancel(pending)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[FlexKV] reset cancel: %s", exc)
        for context in self._pending_lookup_contexts.values():
            context.operation = "load"
            self._log_cache_op(
                context, "complete", "cancelled", reason="connector_reset"
            )
        self._pending_lookup_contexts.clear()
        self._pending_lookups.clear()
        for context in self._prefetch_contexts.values():
            self._log_cache_op(
                context, "complete", "cancelled", reason="connector_reset"
            )
        self._prefetch_contexts.clear()
        self._ongoing_prefetches.clear()
        getattr(self, "_prefetch_planned_tokens", {}).clear()
        getattr(self, "_prefetch_loaded_tokens", {}).clear()
        self._inflight_loads.clear()
        self._completed_layerwise.clear()
        self._launched_load_tids.clear()
        getattr(self, "_launched_load_contexts", {}).clear()
        # Store completion protects source GPU slots. Followers must not
        # return from reset and let SGLang free those slots before the leader
        # has observed a terminal (completely drained) response.
        store_reset_status = {"ok": True, "error": ""}
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            inflight = list(self._inflight_stores.items())
            task_ids = [fk_tid for _, fk_tid in inflight if fk_tid >= 0]
            store_responses: Dict[int, Any] = {}
            wait_error = ""
            if task_ids:
                try:
                    store_responses = self.kv_manager.wait(
                        task_ids, timeout=30.0, completely=True
                    ) or {}
                except Exception as exc:  # noqa: BLE001
                    wait_error = str(exc)
            unsafe = []
            for rid, fk_tid in inflight:
                if fk_tid < 0:
                    continue
                response = store_responses.get(fk_tid)
                status = _status_value(response)
                if response is None or not _is_terminal_status(status):
                    unsafe.append(fk_tid)
                context = self._pop_context(
                    "_inflight_store_contexts", rid, "store", fk_tid
                )
                self._log_cache_op(
                    context,
                    "complete",
                    status,
                    direction="D2H",
                    transfer_mode="no-layerwise",
                    source="reset",
                    error=wait_error or None,
                )
            if unsafe or wait_error:
                store_reset_status = {
                    "ok": False,
                    "error": (
                        "store tasks were not safely drained during reset: "
                        f"task_ids={unsafe}, error={wait_error or 'missing response'}"
                    ),
                }
        if self._sync_ctx.needs_sync:
            store_reset_status = self._sync_ctx.scatter(
                store_reset_status,
                channel=FlexKVScatterChannel.STORE_RESET,
                blocking=True,
            )
        if not store_reset_status["ok"]:
            raise RuntimeError(f"[FlexKV] {store_reset_status['error']}")
        self._inflight_stores.clear()
        getattr(self, "_inflight_store_contexts", {}).clear()
        if self.layer_done_counter is not None:
            self.layer_done_counter.reset()

    def shutdown(self) -> None:
        """Idempotent: unpin FlexKV CPU host buffers via kv_manager.shutdown()."""
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        # print+flush: survives abrupt exit better than logger buffering into | tee.
        logger.info("[FlexKV] connector shutdown begin %s", self._label)
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            logger.info(
                f"[FLEXKV] kv_manager.shutdown begin {self._label}",
            )
            try:
                self.kv_manager.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FlexKV] kv_manager.shutdown: %s", exc)
            self.kv_manager = None
        else:
            logger.info(
                f"[FLEXKV] connector shutdown skip kv_manager "
                f"(leader={self._sync_ctx.is_sync_leader}) {self._label}",
            )
        if self._remote_process is not None:
            try:
                self._remote_process.terminate()
                self._remote_process.join(timeout=5.0)
                if self._remote_process.is_alive():
                    self._remote_process.kill()
                    self._remote_process.join()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FlexKV] remote process shutdown: %s", exc)
            self._remote_process = None
        tp_client = getattr(self, "tp_client", None)
        if tp_client is not None:
            tp_client.retire_gdr_owners()
        logger.info("[FlexKV] connector shutdown done %s", self._label)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _alias_empty_indexer_buffers(
        indexer_buffers: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Replace skip-topk zero-row placeholders with their source layer.

        SGLang intentionally allocates a ``(0, page_stride)`` tensor for a
        skip-topk layer because that layer reuses the preceding layer's
        indexer cache.  FlexKV keeps one physical indexer slot per model layer
        (matching HiCache sizing), so its pointer table must alias the same
        preceding buffer instead of exporting a null ``data_ptr()``.
        """
        resolved: List[torch.Tensor] = []
        previous: Optional[torch.Tensor] = None
        alias_count = 0
        for layer_idx, buffer in enumerate(indexer_buffers):
            if buffer.ndim != 2:
                raise RuntimeError(
                    f"FlexKV indexer layer {layer_idx} expects a 2D buffer, "
                    f"got shape={tuple(buffer.shape)}"
                )
            if buffer.shape[0] == 0:
                if previous is None:
                    raise RuntimeError(
                        "FlexKV cannot alias a leading skip-topk indexer layer: "
                        f"layer={layer_idx} has shape={tuple(buffer.shape)}"
                    )
                if buffer.shape[1] != previous.shape[1]:
                    raise RuntimeError(
                        "FlexKV skip-topk indexer placeholder width mismatch: "
                        f"layer={layer_idx} width={buffer.shape[1]} "
                        f"source_width={previous.shape[1]}"
                    )
                resolved.append(previous)
                alias_count += 1
                continue
            if previous is not None and buffer.shape != previous.shape:
                raise RuntimeError(
                    "FlexKV active indexer buffers must share one shape: "
                    f"layer={layer_idx} shape={tuple(buffer.shape)} "
                    f"previous={tuple(previous.shape)}"
                )
            previous = buffer
            resolved.append(buffer)
        if alias_count:
            logger.info(
                "[FlexKV] aliased %d skip-topk indexer placeholders to the "
                "preceding active layer",
                alias_count,
            )
        return resolved

    def _require_cache_preflight(self, error: Optional[str], label: str) -> None:
        ready = self._sync_ctx.all_reduce_min(int(error is None))
        if not ready:
            raise ValueError(f"FlexKV {label}: {error or 'rejected on another cache rank'}")

    def _agree_cache_geometry(self, geometry: Any, label: str) -> str:
        fingerprint = hashlib.sha256(
            json.dumps(geometry, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        value = int(fingerprint[:15], 16)
        low = self._sync_ctx.all_reduce_min(value)
        high = -self._sync_ctx.all_reduce_min(-value)
        if low != high:
            raise ValueError(f"FlexKV {label} differs across cache ranks")
        return fingerprint

    def _resolve_cp_kv_buffers(self, pool: Any):
        """Validate original owner buffers before introducing scratch aliases."""
        if self._is_dsv4 or self._swa_kv_pool is not None:
            raise ValueError("Physical CP requires an NSA/MLA pool without SWA")
        owner_of = getattr(pool, "_get_layer_owner_rank", None)
        shard_size = int(getattr(pool, "layer_shard_size", 0))
        shard_rank = int(getattr(pool, "layer_shard_rank", -1))
        start = int(getattr(pool, "start_layer", 0))
        main = list(getattr(pool, "kv_buffer", ()))
        count = int(getattr(pool, "layer_num", 0))
        if (not callable(owner_of) or not 0 <= shard_rank < shard_size
                or count != self.rank_info.num_layers_per_pp_stage or len(main) != count):
            raise ValueError("Physical CP requires a complete owner map and KV layer array")
        owners = [owner_of(start + layer) for layer in range(count)]
        if any(type(owner) is not int or not 0 <= owner < shard_size for owner in owners):
            raise ValueError("Physical CP layer owner is outside the CP group")
        scratch = getattr(pool, "remote_kv_buffer", None)
        if (scratch is None or scratch.ndim not in (3, 4)
                or scratch.numel() == 0 or not scratch.is_contiguous()
                or scratch.shape[1] != 1
                or scratch.shape[scratch.ndim - 1] * scratch.element_size() % 8
                or (scratch.ndim == 4 and scratch.shape[2] != self.page_size)):
            raise ValueError("Physical CP requires contiguous 3D/4D receive scratch")
        if scratch.ndim == 3 and scratch.shape[0] % self.page_size:
            raise ValueError("Physical CP token-major scratch must contain whole pages")
        blocks = scratch.shape[0] // self.page_size if scratch.ndim == 3 else scratch.shape[0]
        spans = []

        def add_span(buffer, label):
            pointer = int(buffer.data_ptr())
            if pointer <= 0 or pointer % 8:
                raise ValueError(f"Physical CP {label} has an invalid/unaligned pointer")
            spans.append((pointer, pointer + buffer.numel() * buffer.element_size(), label))

        def validate(buffer, sample, owned, label):
            shape = tuple(sample.shape) if owned else (0, *tuple(sample.shape)[1:])
            if (buffer is None or tuple(buffer.shape) != shape
                    or buffer.dtype != sample.dtype or buffer.device != sample.device
                    or not buffer.is_contiguous()):
                raise ValueError(f"Physical CP {label} has inconsistent shape/dtype/device/stride")
            if owned:
                add_span(buffer, label)

        add_span(scratch, "KV scratch")
        resolved = []
        for layer, buffer in enumerate(main):
            owned = owners[layer] == shard_rank
            validate(buffer, scratch, owned, f"KV layer {layer}")
            resolved.append(buffer if owned else scratch)
        original = getattr(pool, "index_k_with_scale_buffer", None)
        if original is None or len(original) != count:
            raise ValueError("Physical NSA CP requires an Indexer buffer for every layer")
        skip = getattr(pool, "skip_topk_layers", None)
        if skip is None:
            skip = [False] * count
        if len(skip) != count or any(type(value) is not bool for value in skip):
            raise ValueError("Indexer skip_topk_layers must contain one bool per layer")
        active = [i for i, shared in enumerate(skip) if not shared]
        explicit_ids = getattr(self, "_indexer_layer_ids", None)
        if (not active or (explicit_ids is not None and
                (any(type(i) is not int for i in explicit_ids) or list(explicit_ids) != active))):
            raise ValueError("Indexer active IDs disagree with skip_topk_layers")
        index_scratch = getattr(pool, "remote_index_k_with_scale_buffer", None)
        if (index_scratch is None or index_scratch.ndim != 2
                or index_scratch.shape[0] != blocks or index_scratch.shape[1] <= 0
                or not index_scratch.is_contiguous()
                or index_scratch.shape[1] * index_scratch.element_size() % 8
                or index_scratch.device != scratch.device):
            raise ValueError("Physical CP Indexer scratch must map KV pages 1:1")
        add_span(index_scratch, "Indexer scratch")
        indices = active if self._deduplicate_indexer_group else list(range(count))
        resolved_index = {}
        for layer in active:
            owned = owners[layer] == shard_rank
            validate(original[layer], index_scratch, owned, f"Indexer layer {layer}")
            resolved_index[layer] = original[layer] if owned else index_scratch
        for layer in indices:
            if layer not in resolved_index:
                # Shared indexers are never model-owned storage. A scratch row
                # preserves the legacy full layout when dedup is not requested.
                resolved_index[layer] = index_scratch
        spans.sort()
        if any(previous[1] > current[0] for previous, current in zip(spans, spans[1:])):
            raise ValueError("Physical CP owned buffers overlap each other or receive scratch")
        return resolved, _IndexerBufferGroup(
            buffers=tuple(resolved_index[layer] for layer in indices),
            layer_indices=tuple(indices), logical_layer_count=count,
        )

    def _configure_h2d_holders(self, plan: Any) -> None:
        config = self.model_config
        pool = self._kvcache
        physical = bool(getattr(pool, "layer_shard_enabled", False))
        if not physical and plan is None:
            return
        holders = None
        error = None
        try:
            eligible = (
                self._sync_ctx.needs_sync and config.kv_dim == 1
                and config.num_kv_heads == 1 and not self._is_dsv4
                and self._swa_kv_pool is None
                and GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode != "all_write"
                and all(g.num_kv_heads == 1 for g in (config.layer_groups or []))
            )
            if physical:
                eligible = (
                    eligible and config.pp_size == 1
                    and int(pool.layer_shard_size) == config.effective_tp_size > 1
                    and int(pool.layer_shard_rank) == self.rank_info.effective_tp_rank
                    and (config.nnodes == 1 or
                         (config.dp_size == 1 and config.tp_size == 1 and config.instance_num == 1))
                )
                if not eligible:
                    raise ValueError("Physical CP topology is incompatible with owner-only transfers")
                holders = [int(pool._get_layer_owner_rank(int(getattr(pool, "start_layer", 0)) + i))
                           for i in range(config.num_layers)]
            else:
                eligible = (
                    eligible and GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer
                    and config.nnodes == 1 and config.pp_size == 1
                    and plan.cp_size == config.effective_tp_size > 1
                    and plan.cp_rank == self.rank_info.effective_tp_rank
                    and plan.num_layers == config.num_layers
                    and int(getattr(pool, "start_layer", 0)) == 0
                )
                if eligible:
                    plan.validate()
                    holders = [int(plan.holder_of(i)) for i in range(config.num_layers)]
            if holders is not None and any(not 0 <= owner < config.effective_tp_size for owner in holders):
                raise ValueError("CP holder is outside the cache-sharing group")
        except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
            error = str(exc)
        self._require_cache_preflight(error, "CP owner plan")
        self._agree_cache_geometry([physical, holders], "CP owner plan")
        config = replace(config, h2d_layer_holders=holders, layer_shard_enabled=physical)
        config.freeze()
        self.model_config = self.flexkv_config.model_config = config
        self.rank_info = replace(self.rank_info, model_config=config)
        if physical and config.nnodes > 1:
            self.cache_config.mooncake_store_cp_nodes = config.nnodes

    def _build_draft_groups(self, pool: Any, kv_caches: List[torch.Tensor]):
        """Pack replicated NEXTN KV and Indexer into the target's cache block."""
        if pool is None:
            return []
        if (self._is_dsv4 or self._swa_kv_pool is not None
                or self.model_config.kv_dim != 1 or self.model_config.pp_size != 1
                or getattr(pool, "layer_shard_enabled", False)
                or int(getattr(pool, "layer_num", 0)) != 1):
            raise ValueError("Draft piggyback requires a replicated one-layer NSA/MLA pool")
        main = list(getattr(pool, "kv_buffer", ()))
        index = list(getattr(pool, "index_k_with_scale_buffer", ()))
        if len(main) != 1 or len(index) != 1:
            raise ValueError("NEXTN piggyback requires one KV and one Indexer buffer")
        draft, target = main[0], kv_caches[0]
        if (draft.ndim not in (3, 4) or draft.ndim != target.ndim
                or draft.device != target.device or draft.numel() == 0
                or draft.shape[0] != target.shape[0] or draft.shape[1] != 1
                or (draft.ndim == 4 and draft.shape[2] != self.page_size)
                or (draft.ndim == 3 and draft.shape[0] % self.page_size)):
            raise ValueError("Draft and target must share KV slot geometry")
        if draft.shape[draft.ndim - 1] * draft.element_size() % 8:
            raise ValueError("Draft KV rows must be 8-byte aligned")
        blocks = draft.shape[0] // self.page_size if draft.ndim == 3 else draft.shape[0]
        sidecar = index[0]
        if (sidecar is None or sidecar.ndim != 2 or sidecar.shape[0] != blocks
                or sidecar.shape[1] <= 0 or sidecar.shape[1] * sidecar.element_size() % 8
                or sidecar.device != draft.device):
            raise ValueError("Draft Indexer pages must map target KV blocks 1:1")
        for tensor in (draft, sidecar):
            if not tensor.is_contiguous() or tensor.data_ptr() <= 0 or tensor.data_ptr() % 8:
                raise ValueError("Draft buffers must have contiguous aligned physical storage")
        holders = getattr(self.model_config, "h2d_layer_holders", None)
        if holders:
            owners = sorted(set(holders))
            if owners != list(range(self.model_config.effective_tp_size)):
                raise ValueError("Draft piggyback needs a target layer for every CP owner")
            representatives = sorted(holders.index(owner) for owner in owners)
        else:
            representatives = [0]
        result = []
        for tensor, tpb, ratio in ((draft, self.page_size, 1), (sidecar, 1, self.page_size)):
            spec = LayerGroupSpec(
                num_layers=len(representatives), num_kv_heads=1,
                head_size=int(tensor.shape[tensor.ndim - 1]),
                layer_indices=list(representatives), compress_ratio=ratio, dtype=tensor.dtype,
            )
            layout = KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST, num_layer=len(representatives),
                num_block=blocks, tokens_per_block=tpb, num_head=1,
                head_size=spec.head_size, kv_dim=1, num_kv_heads=1,
            )
            result.append((spec, layout, [tensor] * len(representatives)))
        return result

    def _cache_namespace(self, namespace):
        fingerprint = getattr(self, "_draft_namespace", None)
        if fingerprint is None:
            return namespace
        return [json.dumps(["target-draft-v1", fingerprint, namespace],
                           ensure_ascii=True, separators=(",", ":"))]

    @staticmethod
    def _radix_key_input(token_ids, namespace, extra_key=None):
        if hasattr(token_ids, "token_ids"):
            if extra_key is None:
                extra_key = getattr(token_ids, "extra_key", None)
            token_ids = token_ids.token_ids
        if extra_key is not None:
            namespace = [json.dumps(["sglang-extra-key-v1", extra_key, namespace],
                                    ensure_ascii=True, separators=(",", ":"))]
        return token_ids, namespace

    @staticmethod
    def _compact_indexer_buffers(
        indexer_buffers: Sequence[torch.Tensor],
        skip_topk_layers: Sequence[bool],
        *,
        layer_shard_enabled: bool,
    ) -> _IndexerBufferGroup:
        """Drop static ``skip_topk`` aliases from the FlexKV indexer group.

        This first implementation supports ordinary TP/CP, where every cache
        rank exposes the same physical Index-K buffers.  DSA cache layer split
        also represents non-owned *active* layers as zero-row tensors; those
        require an owner-aware registration protocol and must not be inferred
        from tensor shape here.
        """
        buffers = list(indexer_buffers)
        if any(type(value) is not bool for value in skip_topk_layers):
            raise ValueError("Indexer skip_topk_layers must contain booleans")
        skip_mask = list(skip_topk_layers)
        if layer_shard_enabled:
            raise RuntimeError(
                "FLEXKV_DEDUP_INDEXER_GROUP does not yet support "
                "CP DSA cache layer split; disable "
                "--enable-dsa-cache-layer-split or disable indexer dedup"
            )
        if len(skip_mask) != len(buffers):
            raise RuntimeError(
                "FlexKV indexer skip-topk metadata length mismatch: "
                f"mask={len(skip_mask)} buffers={len(buffers)}"
            )

        active_buffers: List[torch.Tensor] = []
        active_layer_indices: List[int] = []
        sample: Optional[torch.Tensor] = None
        for layer_idx, (buffer, skip_topk) in enumerate(zip(buffers, skip_mask, strict=True)):
            if skip_topk:
                continue
            if buffer is None or buffer.ndim != 2:
                raise RuntimeError(
                    f"FlexKV active indexer layer {layer_idx} requires a 2D buffer")
            if (not buffer.is_contiguous() or buffer.shape[1] <= 0
                    or buffer.shape[1] * buffer.element_size() % 8):
                raise RuntimeError("Indexer rows must be contiguous and 8-byte aligned")
            if buffer.shape[0] == 0:
                raise RuntimeError(
                    "FlexKV active indexer layer has no physical buffer: "
                    f"layer={layer_idx} shape={tuple(buffer.shape)}"
                )
            if sample is None:
                sample = buffer
            elif (
                buffer.shape != sample.shape
                or buffer.dtype != sample.dtype
                or buffer.device != sample.device
            ):
                raise RuntimeError(
                    "FlexKV active indexer buffers must share shape, dtype, "
                    f"and device: layer={layer_idx} shape={tuple(buffer.shape)} "
                    f"dtype={buffer.dtype} device={buffer.device}; "
                    f"expected_shape={tuple(sample.shape)} "
                    f"expected_dtype={sample.dtype} expected_device={sample.device}"
                )
            active_buffers.append(buffer)
            active_layer_indices.append(layer_idx)

        if not active_buffers:
            raise RuntimeError("FlexKV indexer dedup found no active Index-K layers")
        spans = sorted(
            (buffer.data_ptr(), buffer.data_ptr() + buffer.numel() * buffer.element_size())
            for buffer in active_buffers)
        if any(start <= 0 or start % 8 for start, _ in spans):
            raise RuntimeError("Indexer buffer address must be nonzero and 8-byte aligned")
        if any(previous[1] > current[0] for previous, current in zip(spans, spans[1:])):
            raise RuntimeError("FlexKV active indexer buffers unexpectedly alias or overlap")
        return _IndexerBufferGroup(
            buffers=tuple(active_buffers),
            layer_indices=tuple(active_layer_indices),
            logical_layer_count=len(buffers),
        )

    def _resolve_kv_buffers(
        self, kvcache: Any
    ) -> Tuple[List[torch.Tensor], Optional[_IndexerBufferGroup]]:
        """Resolve the GPU buffers and describe heterogeneous DSv4 pools."""
        for attr in kvcache.__dict__:
            print(f"attr: {attr}")
            print(f"value: {getattr(kvcache, attr)}")
            print("-"*100)
        self.sparse_indexer = False
        if getattr(kvcache, "layer_shard_enabled", False):
            resolved, cp_indexer_group = self._resolve_cp_kv_buffers(kvcache)
            self.sparse_indexer = _is_sparse_indexer_group(cp_indexer_group)
            return resolved, cp_indexer_group
        indexer_buffers = getattr(kvcache, "index_k_with_scale_buffer", None)
        if indexer_buffers:
            if self._deduplicate_indexer_group:
                indexer_group = self._compact_indexer_buffers(
                    indexer_buffers,
                    getattr(kvcache, "skip_topk_layers", ()),
                    layer_shard_enabled=bool(
                        getattr(kvcache, "layer_shard_enabled", False)
                    ),
                )
                explicit_ids = getattr(self, "_indexer_layer_ids", None)
                if explicit_ids is not None and (
                    any(type(i) is not int for i in explicit_ids)
                    or tuple(explicit_ids) != indexer_group.layer_indices
                ):
                    raise ValueError("Indexer active IDs disagree with skip_topk_layers")
                logger.info(
                    "[FlexKV] compacted indexer group: logical_layers=%d "
                    "physical_layers=%d skipped_layers=%d layer_indices=%s",
                    indexer_group.logical_layer_count,
                    len(indexer_group.buffers),
                    indexer_group.logical_layer_count - len(indexer_group.buffers),
                    list(indexer_group.layer_indices),
                )
            else:
                aliased = self._alias_empty_indexer_buffers(list(indexer_buffers))
                indexer_group = _IndexerBufferGroup(
                    buffers=tuple(aliased),
                    layer_indices=tuple(range(len(aliased))),
                    logical_layer_count=len(aliased),
                )
        else:
            indexer_group = None
        self.sparse_indexer = _is_sparse_indexer_group(indexer_group)
        if not self._is_dsv4:
            if hasattr(kvcache, "kv_buffer"):
                return list(kvcache.kv_buffer), indexer_group
            if hasattr(kvcache, "k_buffer"):
                return (
                    list(kvcache.k_buffer) + list(kvcache.v_buffer),
                    indexer_group,
                )
            raise AttributeError(
                f"Unsupported KV cache type {type(kvcache).__name__}: "
                "expected kv_buffer, k_buffer/v_buffer, or DSv4 multi-pools."
            )

        if getattr(kvcache, "_unified_kv", False):
            raise NotImplementedError(
                "FlexKV DSv4 integration currently requires the split "
                "c4/c128/SWA GPU layout; unified_kv is not supported yet."
            )

        compression_ratios = list(kvcache.compression_ratios)
        stage_start = int(getattr(kvcache, "_stage_start", 0))
        stage_end = int(getattr(kvcache, "_stage_end", len(compression_ratios)))
        sources_by_ratio = getattr(kvcache, "sources_by_ratio", None) or {}
        # FlexKV's transfer workers and layerwise eventfds use PP-stage-local
        # layer coordinates. Keep the global ids only for indexing SGLang's
        # full-model state-pool arrays.

        def stage_global_layer_ids(ratio: int) -> List[int]:
            if ratio in sources_by_ratio:
                return [
                    i
                    for i in sources_by_ratio[ratio]
                    if stage_start <= i < stage_end
                ]
            return [
                i
                for i in range(stage_start, stage_end)
                if compression_ratios[i] == ratio
            ]

        def stage_local_layer_ids(ratio: int) -> List[int]:
            return [i - stage_start for i in stage_global_layer_ids(ratio)]

        def add_kv_group(
            name: str,
            ratio: int,
            layer_ids: List[int],
            buffers: List[torch.Tensor],
            sub_page_size: int,
            bytes_per_token: int,
        ) -> None:
            if buffers:
                self._dsv4_layer_groups.append(
                    {
                        "name": name,
                        "ratio": ratio,
                        "layer_ids": layer_ids,
                        "buffers": list(buffers),
                        "sub_page_size": int(sub_page_size),
                        "bytes_per_token": int(bytes_per_token),
                        "dtype": buffers[0].dtype,
                    }
                )

        kv_pools = getattr(kvcache, "kv_pools", None)
        if not isinstance(kv_pools, dict):
            kv_pools = {}
        else:
            kv_pools = dict(kv_pools)
        for ratio, attr in ((4, "c4_kv_pool"), (128, "c128_kv_pool")):
            if kv_pools.get(ratio) is None:
                pool = getattr(kvcache, attr, None)
                if pool is not None:
                    kv_pools[ratio] = pool

        for ratio in (4, 128, 1, 2):
            pool = kv_pools.get(ratio)
            if pool is None:
                continue
            layer_ids = stage_local_layer_ids(ratio)
            buffers = list(getattr(pool, "kv_buffer", None) or [])
            if not buffers or not layer_ids:
                continue
            if len(buffers) != len(layer_ids):
                raise RuntimeError(
                    f"FlexKV DSv4 group c{ratio}: {len(buffers)} GPU buffers "
                    f"vs {len(layer_ids)} source layers"
                )
            add_kv_group(
                f"c{ratio}",
                ratio,
                layer_ids,
                buffers,
                pool.page_size,
                pool.get_bytes_per_token(),
            )

        index_pools = getattr(kvcache, "index_pools", None)
        if not isinstance(index_pools, dict):
            index_pools = {}
        else:
            index_pools = dict(index_pools)
        if index_pools.get(4) is None:
            c4_indexer = getattr(kvcache, "c4_indexer_kv_pool", None)
            if c4_indexer is not None:
                index_pools[4] = c4_indexer

        for ratio in (4, 1, 2):
            indexer_pool = index_pools.get(ratio)
            if indexer_pool is None:
                continue
            raw_buffers = list(
                getattr(indexer_pool, "index_k_with_scale_buffer", None) or []
            )
            if not raw_buffers:
                continue
            layer_ids = stage_local_layer_ids(ratio)
            if not layer_ids:
                continue
            if len(raw_buffers) != len(layer_ids):
                raise RuntimeError(
                    f"FlexKV DSv4 group c{ratio}_indexer: {len(raw_buffers)} "
                    f"GPU buffers vs {len(layer_ids)} source layers"
                )
            packed, logical_page = _pack_dsv4_indexer_full_pages(
                raw_buffers,
                full_page_size=self.page_size,
                compress_ratio=ratio,
                physical_page_size=int(indexer_pool.page_size),
            )
            sample = packed[0]
            if sample.shape[1] % logical_page != 0:
                raise RuntimeError(
                    f"FlexKV DSv4 group c{ratio}_indexer page stride "
                    f"{sample.shape[1]} is not divisible by {logical_page}"
                )
            add_kv_group(
                f"c{ratio}_indexer",
                ratio,
                list(layer_ids),
                packed,
                logical_page,
                sample.shape[1] // logical_page,
            )

        # Compress states share the SWA physical-page mapping. Only paged
        # (non-request-scoped) pools tile a SWA page: classic DSv4 ratio-4
        # attention/indexer state. DSv4.1 ratio-2 pair state and ratio-128
        # online state are request-scoped rings and must not ride this path.
        # The FlexKV user option is tri-state: omitted/None enables the
        # correctness-preserving default; explicit false keeps SWA-only I/O.
        swa_multi_group = getattr(
            getattr(self.flexkv_config, "user_config", None),
            "swa_multi_group",
            None,
        )
        if getattr(self, "_swa_kv_pool", None) is None:
            logger.info(
                "[FlexKV-DSv4] no paged SWA pool; skipping SWA and "
                "compress-state sidecars"
            )
        elif swa_multi_group is not False:
            attn_state_pools = getattr(kvcache, "compress_state_pools", None) or []
            indexer_state_pools = (
                getattr(kvcache, "indexer_compress_state_pools", None) or []
            )
            for ratio in (4, 128, 1, 2):
                global_ids = stage_global_layer_ids(ratio)
                local_ids = stage_local_layer_ids(ratio)
                self._append_dsv4_state_group(
                    f"c{ratio}_attention_state",
                    local_ids,
                    [attn_state_pools[i] for i in global_ids]
                    if global_ids
                    else [],
                )
                self._append_dsv4_state_group(
                    f"c{ratio}_indexer_state",
                    local_ids,
                    [indexer_state_pools[i] for i in global_ids]
                    if global_ids
                    else [],
                )
        else:
            logger.info(
                "[FlexKV-DSv4] swa_multi_group=false; registering SWA without "
                "compress-state sidecars"
            )

        if self._dsv4_state_groups:
            if self.cache_config.swa is None:
                raise RuntimeError(
                    "FlexKV DSv4 state sidecars require an SWA cache config"
                )
            self.cache_config.swa.multi_group = True

        if not self._dsv4_layer_groups:
            raise RuntimeError("DSv4 KV cache has no buffers on this PP stage")
        flat_buffers = [
            buffer for group in self._dsv4_layer_groups for buffer in group["buffers"]
        ]
        logger.info(
            "[FlexKV-DSv4] resolved groups=%s state_groups=%s",
            [group["name"] for group in self._dsv4_layer_groups],
            [group["name"] for group in self._dsv4_state_groups],
        )
        return flat_buffers, None

    def _append_dsv4_state_group(
        self,
        name: str,
        layer_ids: List[int],
        pools: List[Any],
    ) -> None:
        if not pools:
            return
        if any(pool is None for pool in pools):
            if all(pool is None for pool in pools):
                return
            raise RuntimeError(
                f"FlexKV DSv4 state group {name!r} has a mix of missing and "
                f"present compress-state pools for layers {layer_ids}"
            )
        if any(bool(getattr(pool, "request_scoped", False)) for pool in pools):
            logger.info(
                "[FlexKV-DSv4] skipping request-scoped state group %s "
                "(not SWA-page mapped)",
                name,
            )
            return
        if len(pools) != len(layer_ids):
            raise RuntimeError(
                f"FlexKV DSv4 state group {name!r}: {len(pools)} pools vs "
                f"{len(layer_ids)} layers"
            )
        ring_sizes = {int(pool.ring_size) for pool in pools}
        if len(ring_sizes) != 1:
            raise RuntimeError(
                f"FlexKV DSv4 state group {name!r} has mixed ring sizes: "
                f"{sorted(ring_sizes)}"
            )
        ring_size = ring_sizes.pop()
        swa_page_size = int(getattr(self._kvcache, "swa_page_size", self.page_size))
        if swa_page_size != self.page_size or swa_page_size % ring_size != 0:
            raise RuntimeError(
                f"FlexKV DSv4 state group {name!r} has incompatible "
                f"page_size={swa_page_size}, ring_size={ring_size}"
            )
        buffers = [pool.kv_score_buffer.kv_score for pool in pools]
        sample = buffers[0]
        if sample.ndim != 2 or not sample.is_contiguous():
            raise RuntimeError(
                f"FlexKV DSv4 state group {name!r} expects contiguous 2D "
                f"buffers, got shape={tuple(sample.shape)}"
            )
        if any(
            buf.shape != sample.shape or buf.dtype != sample.dtype for buf in buffers
        ):
            raise RuntimeError(
                f"FlexKV DSv4 state group {name!r} has inconsistent buffers"
            )
        self._dsv4_state_groups.append(
            {
                "name": name,
                "layer_ids": list(layer_ids),
                "buffers": buffers,
                "ratio": swa_page_size // ring_size,
                "sub_page_size": ring_size,
                "head_size": int(sample.shape[1]),
                "dtype": sample.dtype,
            }
        )

    def _build_dsv4_layer_group_specs(self) -> List[LayerGroupSpec]:
        specs = []
        for group in self._dsv4_layer_groups:
            sample = group["buffers"][0]
            sub_page_size = group["sub_page_size"]
            if sample.shape[1] % sub_page_size != 0:
                raise RuntimeError(
                    f"FlexKV DSv4 group {group['name']!r} has page bytes "
                    f"{sample.shape[1]} not divisible by {sub_page_size}"
                )
            specs.append(
                LayerGroupSpec(
                    num_layers=len(group["layer_ids"]),
                    num_kv_heads=1,
                    head_size=sample.shape[1] // sub_page_size,
                    layer_indices=list(group["layer_ids"]),
                    compress_ratio=group["ratio"],
                    dtype=group["dtype"],
                )
            )
        return specs

    def _build_indexer_layer_group_specs(
        self,
        kv_caches: List[torch.Tensor],
        indexer_group: _IndexerBufferGroup,
    ) -> List[LayerGroupSpec]:
        kv_tensor = kv_caches[0]
        if kv_tensor.ndim not in (3, 4):
            raise ValueError(f"Expected 3D/4D KV cache tensor, got {kv_tensor.shape}")
        num_kv_heads, head_size = kv_tensor.shape[1], kv_tensor.shape[kv_tensor.ndim - 1]
        if indexer_group.logical_layer_count != self.rank_info.num_layers_per_pp_stage:
            raise RuntimeError(
                "FlexKV indexer logical layer count mismatch: "
                f"indexer={indexer_group.logical_layer_count} "
                f"pp_stage={self.rank_info.num_layers_per_pp_stage}"
            )
        indexer_buffers = indexer_group.buffers
        return [
            LayerGroupSpec(
                num_layers=self.rank_info.num_layers_per_pp_stage,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                layer_indices=list(range(self.rank_info.num_layers_per_pp_stage)),
                compress_ratio=1,
                dtype=kv_caches[0].dtype,
            ),
            LayerGroupSpec(
                num_layers=len(indexer_buffers),
                num_kv_heads=1,
                # GLM DSA stores one flattened index+scale row per complete
                # KV page: [num_pages, page_size * (head + scale)]. The
                # physical GPU layout therefore has tokens_per_block=1.
                # Describe that row as one page so the worker does not
                # multiply it by page_size a second time.
                head_size=indexer_buffers[0].shape[1],
                layer_indices=list(indexer_group.layer_indices),
                compress_ratio=self.page_size,
                dtype=indexer_buffers[0].dtype,
            ),
        ]

    def _apply_layer_groups_for_cache_sizing(
        self,
        kv_caches: List[torch.Tensor],
        indexer_group: Optional[_IndexerBufferGroup],
    ) -> None:
        draft_groups = getattr(self, "_draft_groups", ())
        error = None
        model_config = self.model_config
        try:
            layer_groups = None
            if self._is_dsv4:
                layer_groups = self._build_dsv4_layer_group_specs()
            elif indexer_group is not None:
                layer_groups = self._build_indexer_layer_group_specs(kv_caches, indexer_group)
            elif draft_groups:
                raise ValueError("Draft piggyback requires target Indexer groups")
            if layer_groups is not None:
                layer_groups += [entry[0] for entry in draft_groups]
                if model_config.layer_groups is None:
                    model_config = replace(model_config, layer_groups=layer_groups)
                    model_config.freeze()
                elif draft_groups and model_config.layer_groups != layer_groups:
                    raise ValueError("Explicit layer_groups conflict with draft piggyback layout")
        except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
            error = str(exc)
        self._require_cache_preflight(error, "cache layer groups")
        self.model_config = self.flexkv_config.model_config = model_config
        self.rank_info = replace(self.rank_info, model_config=model_config)
        if not model_config.layer_groups:
            return

        old_cpu_blocks = self.cache_config.num_cpu_blocks
        if not recompute_cache_block_counts(self.model_config, self.cache_config):
            return
        logger.info(
            "[FlexKV] recomputed cache capacity for %d groups: %s -> %s CPU blocks",
            len(self.model_config.layer_groups),
            old_cpu_blocks,
            self.cache_config.num_cpu_blocks,
        )
        for attr in ("num_cpu_blocks", "num_ssd_blocks", "num_remote_blocks"):
            value = getattr(self.cache_config, attr, None)
            if value is not None and value > 0:
                setattr(
                    self.cache_config,
                    attr,
                    self._sync_ctx.all_reduce_min(int(value)),
                )

    def _build_swa_slot_mapping(
        self, full_indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Translate full-pool indices and keep only the mapped SWA tail."""
        if self._swa_kv_pool is None:
            return None
        translate = getattr(self._kvcache, "translate_loc_from_full_to_swa", None)
        if translate is None:
            return None
        swa_indices = translate(full_indices).to(device="cpu", dtype=torch.int64)
        mapped = swa_indices > 0
        if not bool(mapped.any()):
            return None
        unmapped = (~mapped).nonzero(as_tuple=False)
        if unmapped.numel() > 0:
            swa_indices = swa_indices[int(unmapped[-1].item()) + 1 :]
        if swa_indices.numel() == 0:
            return None
        if not bool((swa_indices > 0).all()):
            raise RuntimeError("SWA slot mapping contains the reserved slot 0")
        return swa_indices

    @staticmethod
    def _as_numpy_mask(mask) -> np.ndarray:
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            return mask.detach().cpu().numpy()
        return np.asarray(mask)

    @staticmethod
    def _to_cpu_int64(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.to(torch.int64)

    def _store_profile_scope(self, name: str):
        if not getattr(self, "_profile_store_stages", False):
            return nullcontext()
        return torch.profiler.record_function(name)

    def _wait_kv_manager_ready(self, poll_interval: float = 10.0) -> None:
        assert self.kv_manager is not None
        timeout = float(os.environ.get("FLEXKV_READY_TIMEOUT_S", "360"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("FLEXKV_READY_TIMEOUT_S must be finite and positive")
        start = time.monotonic()
        while not self.kv_manager.is_ready():
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError(f"FlexKV did not become ready within {timeout}s {self._label}")
            time.sleep(min(poll_interval, remaining))
            logger.info(
                "[FlexKV] Waiting for FlexKV ready %s (waited %.0fs)",
                self._label, time.monotonic() - start,
            )
        logger.info("[FlexKV] FlexKV is ready %s", self._label)

    def _register_with_retry(
        self,
        kv_caches: List[torch.Tensor],
        indexer_group: Optional[_IndexerBufferGroup] = None,
        max_retries: int = 360,
    ) -> None:
        """Retry GPU registration. On node_rank>0, the
        TransferManagerOnRemote may not be ready immediately; retry up
        to ~6 minutes."""
        for attempt in range(max_retries):
            try:
                self._register_to_server(kv_caches, indexer_group)
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                if attempt % 30 == 0:
                    logger.info(
                        "[FlexKV] GPU register retry %s attempt=%d/%d error=%s",
                        self._label,
                        attempt + 1,
                        max_retries,
                        exc,
                    )
                time.sleep(1.0)

    def _register_to_server(
        self,
        kv_caches: List[torch.Tensor],
        indexer_group: Optional[_IndexerBufferGroup] = None,
    ) -> None:
        assert len(kv_caches) > 0
        if self._is_dsv4:
            self._register_dsv4_to_server(kv_caches)
        else:
            self._register_standard_to_server(kv_caches, indexer_group)

    def _register_standard_to_server(
        self,
        kv_caches: List[torch.Tensor],
        indexer_group: Optional[_IndexerBufferGroup] = None,
    ) -> None:
        kv_tensor = kv_caches[0]
        if kv_tensor.ndim == 4:
            num_blocks, physical_num_kv_heads, tokens_per_block, head_size = kv_tensor.shape
            if tokens_per_block != self.page_size:
                raise ValueError(
                    f"KV page axis {tokens_per_block} does not match page_size={self.page_size}")
            if any(t.shape != kv_tensor.shape or not t.is_contiguous() for t in kv_caches):
                raise ValueError("Page-major KV tensors must have matching contiguous layouts")
        elif kv_tensor.ndim == 3:
            num_tokens, physical_num_kv_heads, head_size = kv_tensor.shape
            num_blocks = num_tokens // self.page_size
            tokens_per_block = self.page_size
        else:
            raise ValueError(f"Expected 3D/4D KV cache tensor, got {kv_tensor.shape}")

        kv_dim = self.model_config.kv_dim
        global_num_kv_heads = self.model_config.num_kv_heads
        # FlexKV uses num_kv_heads == 1 for KV shared across all TP ranks.
        # TODO: represent partial KV-head replication when 1 < global heads < TP.

        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=self.rank_info.num_layers_per_pp_stage,
            num_block=num_blocks,
            tokens_per_block=tokens_per_block,
            num_head=physical_num_kv_heads,
            head_size=head_size,
            kv_dim=kv_dim,
            num_kv_heads=global_num_kv_heads,
        )

        indexer_layout = None
        if indexer_group is not None:
            indexer_buffers = indexer_group.buffers
            indexer_tensor = indexer_buffers[0]
            assert indexer_tensor.ndim == 2, (
                f"Expected 2D indexer tensor (num_pages, page_stride_size), "
                f"got shape={indexer_tensor.shape}"
            )
            indexer_layout = KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=len(indexer_buffers),
                num_block=indexer_tensor.shape[0],
                tokens_per_block=1,
                num_head=1,
                head_size=indexer_tensor.shape[1],
                # Indexer: single-region single-head sidecar (kv_dim=1, num_kv_heads=1).
                kv_dim=1,
                num_kv_heads=1,
            )

        if indexer_group is not None and indexer_layout is not None:
            indexer_buffers = indexer_group.buffers
            draft_groups = getattr(self, "_draft_groups", ())
            handles = [list(kv_caches), list(indexer_buffers)] + [g[2] for g in draft_groups]
            self.tp_client.register_to_server(
                kv_caches=[tensor for group in handles for tensor in group],
                kv_layout=gpu_layout,
                layer_groups=self._build_indexer_layer_group_specs(
                    kv_caches, indexer_group
                ) + [g[0] for g in draft_groups],
                gpu_layouts=[gpu_layout, indexer_layout] + [g[1] for g in draft_groups],
                handles_per_group=handles,
            )
        else:
            self.tp_client.register_to_server(
                kv_caches=kv_caches,
                kv_layout=gpu_layout,
            )
        logger.info("[FlexKV] Registered KV caches to server %s", self._label)

    def _build_dsv4_swa_registration(self):
        """SWA GPU pool geometry for DSv4: (caches, layout, layer_groups,
        gpu_layouts, handles_per_group), all None when the model has no SWA
        pool. Shared by the GPU registration and by the radixshmem bootstrap,
        which must know the sidecar groups before the TE sees a registration."""
        swa_caches = None
        swa_layout = None
        swa_layer_groups = None
        swa_gpu_layouts = None
        swa_handles_per_group = None
        swa_pool = self._swa_kv_pool
        swa_buffers = list(getattr(swa_pool, "kv_buffer", []))
        if swa_buffers:
            sample = swa_buffers[0]
            sub_page_size = int(swa_pool.page_size)
            if sample.ndim != 2 or sample.shape[1] % sub_page_size != 0:
                raise RuntimeError(
                    "FlexKV DSv4 SWA buffers must be 2D with a page-aligned stride"
                )
            head_size = sample.shape[1] // sub_page_size
            swa_layout = KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=len(swa_buffers),
                num_block=sample.shape[0],
                tokens_per_block=sub_page_size,
                num_head=1,
                head_size=head_size,
                # SWA: single-region single-head sidecar (kv_dim=1, num_kv_heads=1).
                kv_dim=1,
                num_kv_heads=1,
            )
            swa_caches = list(swa_buffers)

            if self._dsv4_state_groups:
                swa_layer_ids = list(range(len(swa_buffers)))
                swa_layer_groups = [
                    LayerGroupSpec(
                        num_layers=len(swa_buffers),
                        num_kv_heads=1,
                        head_size=head_size,
                        layer_indices=swa_layer_ids,
                        compress_ratio=1,
                        dtype=sample.dtype,
                    )
                ]
                swa_gpu_layouts = [swa_layout]
                swa_handles_per_group = [list(swa_buffers)]

                for state_group in self._dsv4_state_groups:
                    state_buffers = state_group["buffers"]
                    ring_size = int(state_group["sub_page_size"])
                    state_pages = min(
                        int(buffer.shape[0]) // ring_size for buffer in state_buffers
                    )
                    if state_pages < sample.shape[0]:
                        raise RuntimeError(
                            f"FlexKV DSv4 state group {state_group['name']!r} "
                            f"has {state_pages} pages, SWA has {sample.shape[0]}"
                        )
                    state_layout = KVCacheLayout(
                        type=KVCacheLayoutType.LAYERFIRST,
                        num_layer=len(state_group["layer_ids"]),
                        num_block=state_pages,
                        tokens_per_block=ring_size,
                        num_head=1,
                        head_size=int(state_group["head_size"]),
                        # State sidecar: single-region single-head (kv_dim=1, num_kv_heads=1).
                        kv_dim=1,
                        num_kv_heads=1,
                    )
                    swa_layer_groups.append(
                        LayerGroupSpec(
                            num_layers=len(state_group["layer_ids"]),
                            num_kv_heads=1,
                            head_size=int(state_group["head_size"]),
                            layer_indices=list(state_group["layer_ids"]),
                            compress_ratio=int(state_group["ratio"]),
                            dtype=state_group["dtype"],
                        )
                    )
                    swa_gpu_layouts.append(state_layout)
                    swa_handles_per_group.append(list(state_buffers))
                    swa_caches.extend(state_buffers)

        return (swa_caches, swa_layout, swa_layer_groups, swa_gpu_layouts,
                swa_handles_per_group)

    def _register_dsv4_to_server(self, kv_caches: List[torch.Tensor]) -> None:
        """Register DSv4 main KV groups plus SWA/state sidecars."""
        layer_groups: List[LayerGroupSpec] = []
        gpu_layouts: List[KVCacheLayout] = []
        handles_per_group: List[List[torch.Tensor]] = []
        all_buffers: List[torch.Tensor] = []

        for group in self._dsv4_layer_groups:
            buffers = group["buffers"]
            sample = buffers[0]
            if sample.ndim != 2:
                raise RuntimeError(
                    f"FlexKV DSv4 group {group['name']!r} expects 2D page "
                    f"buffers, got shape={tuple(sample.shape)}"
                )
            if any(buf.shape != sample.shape for buf in buffers):
                raise RuntimeError(
                    f"FlexKV DSv4 group {group['name']!r} has mixed shapes"
                )
            sub_page_size = int(group["sub_page_size"])
            if self.page_size % int(group["ratio"]) != 0:
                raise RuntimeError(
                    f"FlexKV page_size={self.page_size} is not divisible by "
                    f"DSv4 ratio={group['ratio']}"
                )
            if sample.shape[1] % sub_page_size != 0:
                raise RuntimeError(
                    f"FlexKV DSv4 group {group['name']!r} page stride "
                    f"{sample.shape[1]} is not divisible by {sub_page_size}"
                )
            head_size = sample.shape[1] // sub_page_size
            layer_groups.append(
                LayerGroupSpec(
                    num_layers=len(group["layer_ids"]),
                    num_kv_heads=1,
                    head_size=head_size,
                    layer_indices=list(group["layer_ids"]),
                    compress_ratio=int(group["ratio"]),
                    dtype=group["dtype"],
                )
            )
            gpu_layouts.append(
                KVCacheLayout(
                    type=KVCacheLayoutType.LAYERFIRST,
                    num_layer=len(group["layer_ids"]),
                    num_block=sample.shape[0],
                    tokens_per_block=sub_page_size,
                    num_head=1,
                    head_size=head_size,
                    kv_dim=self.model_config.kv_dim,
                    num_kv_heads=self.model_config.num_kv_heads,
                )
            )
            handles_per_group.append(list(buffers))
            all_buffers.extend(buffers)

        if len(all_buffers) != len(kv_caches):
            raise RuntimeError(
                f"FlexKV DSv4 flattened {len(all_buffers)} buffers, expected "
                f"{len(kv_caches)}"
            )

        # The primary layout owns the full PP-stage layer-id namespace. Group
        # layouts remain local because several groups cover disjoint layer sets.
        first_layout = gpu_layouts[0]
        primary_layout = KVCacheLayout(
            type=first_layout.type,
            num_layer=self.rank_info.num_layers_per_pp_stage,
            num_block=first_layout.num_block,
            tokens_per_block=first_layout.tokens_per_block,
            num_head=first_layout.num_head,
            head_size=first_layout.head_size,
            kv_dim=first_layout.kv_dim,
            num_kv_heads=first_layout.num_kv_heads,
        )

        (swa_caches, swa_layout, swa_layer_groups, swa_gpu_layouts,
         swa_handles_per_group) = self._build_dsv4_swa_registration()

        self.tp_client.register_to_server(
            kv_caches=all_buffers,
            kv_layout=primary_layout,
            layer_groups=layer_groups,
            gpu_layouts=gpu_layouts,
            handles_per_group=handles_per_group,
            swa_caches=swa_caches,
            swa_layout=swa_layout,
            swa_layer_groups=swa_layer_groups,
            swa_gpu_layouts=swa_gpu_layouts,
            swa_handles_per_group=swa_handles_per_group,
        )
        logger.info(
            "[FlexKV-DSv4] registered %d main groups, SWA=%s, state_groups=%d",
            len(layer_groups),
            swa_caches is not None,
            len(self._dsv4_state_groups),
        )

    def _send_slot_mapping_to_remote(
        self, task_id: int, slot_mapping_cpu: torch.Tensor
    ) -> None:
        np_arr = slot_mapping_cpu.numpy()
        self.tp_client.set_slot_mapping(task_id, np_arr)

    def _send_eventfds_to_worker(self, retry_interval: float = 1.0) -> None:
        """UDS handshake with the FlexKV layerwise transfer worker.

        Sends per-counter eventfd FDs over a unix domain socket using
        ``SCM_RIGHTS``. Retries connect (worker may not yet be up) and
        retries the whole connect+send sequence on send error.
        """
        max_retries = self._layerwise_eventfd_connect_max_retries
        max_send_retries = 3
        last_error: Optional[BaseException] = None

        assert self.layer_done_counter is not None

        for send_attempt in range(max_send_retries):
            sock: Optional[socket.socket] = None
            try:
                # Phase 1: connect (worker may not yet be up).
                for attempt in range(max_retries):
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    try:
                        sock.connect(self._layerwise_socket)
                        logger.info(
                            "[FlexKV] Eventfd connected %s socket=%s attempts=%d",
                            self._label,
                            self._layerwise_socket,
                            attempt + 1,
                        )
                        break
                    except (FileNotFoundError, ConnectionRefusedError) as exc:
                        sock.close()
                        sock = None
                        if attempt == max_retries - 1:
                            raise RuntimeError(
                                f"[FlexKV] Failed to connect to eventfd socket "
                                f"{self._layerwise_socket} after {max_retries} attempts"
                            ) from exc
                        time.sleep(retry_interval)
                assert sock is not None

                # Phase 2: send 16-byte metadata + per-counter FDs + read ACK.
                num_counters = self.layer_done_counter.num_counters
                metadata = struct.pack(
                    "iiii",
                    self.rank_info.effective_tp_rank,
                    self.model_config.effective_tp_size_per_node,
                    self.rank_info.num_layers_per_pp_stage,
                    num_counters,
                )
                sock.sendall(metadata)
                for counter_id in range(num_counters):
                    fds = self.layer_done_counter.events[counter_id].load_event_fds
                    send_fds(sock, fds, struct.pack("i", counter_id))
                sock.settimeout(30.0)
                try:
                    ack = sock.recv(1)
                except socket.timeout as exc:
                    raise RuntimeError(
                        "Timed out waiting for ACK from FlexKV layerwise worker"
                    ) from exc
                if not ack or ack[0] != 1:
                    raise RuntimeError(
                        f"FlexKV layerwise worker NACK'd eventfd transfer (ack={ack!r})"
                    )
                logger.info(
                    "[FlexKV] Eventfd handshake complete %s counters=%d layers=%d",
                    self._label,
                    num_counters,
                    self.rank_info.num_layers_per_pp_stage,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "[FlexKV] Eventfd handshake send_attempt=%d/%d failed: %s",
                    send_attempt + 1,
                    max_send_retries,
                    exc,
                )
            finally:
                if sock is not None:
                    sock.close()
                time.sleep(retry_interval)

        raise RuntimeError(
            f"[FlexKV] Failed to send eventfds to {self._layerwise_socket} "
            f"after {max_send_retries} attempts: {last_error}"
        )
