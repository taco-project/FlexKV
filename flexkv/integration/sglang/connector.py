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

Queued prefetch retains its result lease while an exact, immutable CPU query
prepares a held GET on the existing state owner. Scheduler-thread progress
publishes readiness without waiting for that GET; lookup consumes it only if
tokens, namespace and the complete GPU-prefix mask still match.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import signal
import socket
import struct
import time
from concurrent.futures import Future
from contextlib import suppress
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
# Leader-side pacing of non-blocking store-completion polls (one RPC each).
# The first poll after a launch is not delayed; each poll that finds nothing
# doubles the gap from MIN up to MAX (2, 4, 8, 16, 32 ms by default), and any
# completion restarts at MIN. A scheduler tick longer than the gap (a forward)
# still polls every tick. Store completion only releases lock refs (off the
# TTFT path), so idle polls can back off; MIN matches TaskRuntime's 2 ms
# state-advance period. Setting MIN to 0 disables pacing.
_STORE_POLL_MIN_INTERVAL_S = max(
    0.0, float(os.getenv("FLEXKV_STORE_POLL_MIN_INTERVAL_MS", "2")) / 1000.0
)
_STORE_POLL_MAX_INTERVAL_S = max(
    _STORE_POLL_MIN_INTERVAL_S,
    float(os.getenv("FLEXKV_STORE_POLL_MAX_INTERVAL_MS", "32")) / 1000.0,
)
# Arrival lookups (L2-only): the leader issues the admission GET when the
# request is queued, so admission does not pay one synchronous get_match RPC
# per request on the serial scheduler path. Unconsumed positive GETs pin CPU
# blocks, so outstanding lookups are bounded in count and age.
_ARRIVAL_LOOKUP_TTL_S = float(os.getenv("FLEXKV_ARRIVAL_LOOKUP_TTL_S", "120"))
_ARRIVAL_LOOKUP_MAX = int(os.getenv("FLEXKV_ARRIVAL_LOOKUP_MAX", "64"))
_ARRIVAL_LOOKUP_WAIT_S = 30.0


def _arrival_lookup_opt_in(prefetch_enabled: bool) -> bool:
    # Opt-in: an admission that never reaches lookup_kv (full L1 hit)
    # leaves its held GET pinning CPU blocks until TTL expiry.
    return not prefetch_enabled and os.getenv("FLEXKV_ARRIVAL_LOOKUP", "0") == "1"


@dataclass
class _CacheOpContext:
    operation: str
    sglang_req_id: Optional[str]
    start_ns: int
    task_id: int = -1


@dataclass
class _StoreAdmission:
    """Owned CPU snapshots retained through reservation, launch, and reset."""

    stores: List[dict]
    prepared: list
    contexts: List[_CacheOpContext]
    positions: list
    requests: list
    future: Optional[Future] = None
    matches: Optional[list] = None
    starts: Optional[list] = None
    asynchronous: bool = False
    launch_started: bool = False
    resetting: bool = False
    result: Optional[List[int]] = None
    error: str = ""
    retain_slots: bool = False
    reset: bool = False



@dataclass(frozen=True)
class _PrefetchLookup:
    """Packed query identity and its unlaunched, asynchronously admitted GET."""

    token_ids: np.ndarray
    token_mask: np.ndarray
    namespace: Optional[Tuple[str, ...]]
    future: Future

@dataclass(frozen=True)
class _IndexerBufferGroup:
    """Physical indexer buffers plus their original layer ids."""

    buffers: Tuple[torch.Tensor, ...]
    layer_indices: Tuple[int, ...]
    logical_layer_count: int


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
    ) -> None:
        self.page_size = int(page_size)

        # 1. Resolve FlexKV config from env + sglang server args.
        self.flexkv_config = FlexKVConfig.from_env()
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
        self._prepared_lookups: Dict[str, _PrefetchLookup] = {}
        # Leader-only: rid -> (issue time, first masked token, arrival GET).
        self._arrival_lookups: Dict[str, Tuple[float, int, _PrefetchLookup]] = {}
        self._arrival_retirements: List[Future] = []
        # Leader-only: store rid -> (namespace, tokens, page-aligned length) for
        # stores that may still commit blocks an arrival GET could not see.
        self._arrival_store_guard: Dict[str, Tuple[Optional[Tuple[str, ...]], Any, int]] = {}
        self._prefetch_loaded_spans = {}
        self._queued_prefetch = (
            self._chunked_prefetch or self.model_config.dp_size > 1
            or self.model_config.instance_num > 1
            or GLOBAL_CONFIG_FROM_ENV.server_client_mode
        )
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

        kv_caches, indexer_group = self._resolve_kv_buffers(kvcache)
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
        )
        self._register_with_retry(kv_caches, indexer_group)

        # 8. Layerwise transfer plumbing.
        self.enable_layerwise = bool(
            int(os.environ.get("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "0"))
        )
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
        self._launched_load_contexts: Dict[int, _CacheOpContext] = {}
        # Stores
        self._inflight_stores: Dict[str, int] = {}  # rid -> fkv_task_id
        self._inflight_store_contexts: Dict[str, _CacheOpContext] = {}
        self._pending_store_batch: Optional[_StoreAdmission] = None
        self._store_poll_next_ts = 0.0  # leader-only, see _advance_store_poll
        self._store_poll_interval_s = _STORE_POLL_MIN_INTERVAL_S
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
        # With a lower tier, prefetch must finish before a lookup can see its
        # blocks, so the early GET is only valid for an L2-only deployment.
        self._arrival_lookup_enabled = _arrival_lookup_opt_in(self._prefetch_enabled)
        self._shutdown_done = False

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
            arrival = self._pop_arrival_lookup(rid)
            try:
                prepared = getattr(self, "_prepared_lookups", {}).get(rid)
                if prepared is not None and self._lookup_identity_matches(
                    prepared, tids_np, mask_np, namespace,
                ):
                    # Consume the speculative match here, waiting only if it
                    # has not finished. It must not add an admission gate that
                    # leaves a runnable prefill batch underfilled.
                    try:
                        res = prepared.future.result(timeout=_ARRIVAL_LOOKUP_WAIT_S)
                    except Exception:
                        # A late GET may still hold CPU blocks: retire it
                        # together with its prefetch lease, never drop it.
                        self._release_prefetch_result(rid)
                        raise
                    self._prepared_lookups.pop(rid, None)
                elif arrival is not None and self._lookup_identity_matches(
                    arrival, tids_np, mask_np, namespace,
                ):
                    # Same query as a synchronous GET now: L1 prefix unchanged
                    # and no overlapping store could commit since issue.
                    lookup, arrival = arrival, None
                    try:
                        res = lookup.future.result(timeout=_ARRIVAL_LOOKUP_WAIT_S)
                    except Exception:
                        self._retire_arrival_lookup(lookup)
                        raise
                else:
                    if prepared is not None:
                        self._release_prefetch_result(rid)
                    # A DSv4 prefix must own its SWA/compress-state sidecars.
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
            if arrival is not None:
                # Superseded (identity changed or a prepared lookup won).
                self._retire_arrival_lookup(arrival)
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
        """Agree on GPU allocation before any rank enters a restore launch."""
        if self._sync_ctx.needs_sync:
            return bool(self._sync_ctx.all_reduce_min(int(local_ready)))
        return bool(local_ready)

    @staticmethod
    def _lookup_identity_matches(lookup, token_ids, token_mask, namespace) -> bool:
        return (
            lookup.namespace == (tuple(namespace) if namespace is not None else None)
            and np.array_equal(lookup.token_ids, token_ids)
            and np.array_equal(lookup.token_mask, token_mask)
        )

    # ---- arrival lookups (leader-only, L2-only deployments) ----------------
    #
    # Correctness: an arrival GET returns what a synchronous GET at admission
    # would, provided (a) the admission query is identical (checked in
    # lookup_kv: same tokens, L1 prefix/mask and namespace) and (b) no block
    # in the masked range was committed in between. Positive hits are pinned
    # by the held GET, so (b) only concerns stores, and a store can only add
    # a hit if it covers the first masked page with the same tokens. Such a
    # store is either live at issue time (issue skipped) or submitted later
    # (pending arrival GETs invalidated in _prepare_store_batch).

    def _issue_arrival_lookup(self, rid, token_ids, namespace, start) -> None:
        if self.kv_manager is None:
            return
        self._drop_arrival_lookup(rid)
        now = time.monotonic()
        self._expire_arrival_lookups(now)
        self._reap_arrival_retirements()
        if len(self._arrival_lookups) >= _ARRIVAL_LOOKUP_MAX:
            return
        start = int(start)
        if start < 0 or start % self.page_size or start >= len(token_ids):
            return
        tokens = np.array(token_ids, dtype=np.int64, copy=True, order="C")
        ns = tuple(namespace) if namespace is not None else None
        if self._live_store_covers(ns, tokens, start):
            return
        tokens.setflags(write=False)
        mask = np.zeros(tokens.size, dtype=np.bool_)
        mask[start:] = True
        mask.setflags(write=False)
        lookup = _PrefetchLookup(tokens, mask, ns, Future())
        try:
            future = self.kv_manager.get_match_async(
                token_ids=tokens,
                token_mask=mask,
                cpu_only=getattr(self, "_chunked_prefetch", False),
                swa_aware=self._swa_kv_pool is not None,
                namespace=list(ns) if ns is not None else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[FlexKV] arrival lookup not issued rid=%s: %s", rid, exc)
            return
        future.add_done_callback(
            lambda done, target=lookup.future: self._finish_prefetch_lookup(target, done)
        )
        self._arrival_lookups[rid] = (now, start, lookup)

    def _pop_arrival_lookup(self, rid) -> Optional[_PrefetchLookup]:
        lookups = getattr(self, "_arrival_lookups", None)
        if not lookups or rid is None:
            return None
        entry = lookups.pop(rid, None)
        return None if entry is None else entry[2]

    def _drop_arrival_lookup(self, rid) -> None:
        lookup = self._pop_arrival_lookup(rid)
        if lookup is not None:
            self._retire_arrival_lookup(lookup)

    def _expire_arrival_lookups(self, now: float) -> None:
        lookups = self._arrival_lookups
        while lookups:
            rid, (issued, _, lookup) = next(iter(lookups.items()))
            if now - issued < _ARRIVAL_LOOKUP_TTL_S:
                break
            del lookups[rid]
            self._retire_arrival_lookup(lookup)

    def _retire_arrival_lookup(self, lookup: _PrefetchLookup) -> None:
        """Cancel the held task of an unconsumed GET once it is admitted.

        Runs from completion callbacks: no scheduler state or collectives.
        """
        retired = Future()
        manager = self.kv_manager

        def cancelled(future):
            try:
                future.result()
                retired.set_result(None)
            except Exception as exc:  # noqa: BLE001
                retired.set_exception(exc)

        def admitted(future):
            try:
                result = future.result()
            except Exception:  # noqa: BLE001
                result = None  # Failed admission created no held task.
            if result is not None:
                task_id, mask = result
                if task_id >= 0 and mask is not None and np.any(mask):
                    try:
                        manager.cancel_async([task_id]).add_done_callback(cancelled)
                    except Exception as exc:  # noqa: BLE001
                        retired.set_exception(exc)
                    return
            # Zero-hit GETs are already complete and must not be cancelled.
            retired.set_result(None)

        lookup.future.add_done_callback(admitted)
        self._arrival_retirements.append(retired)

    def _reap_arrival_retirements(self) -> None:
        pending = []
        for future in self._arrival_retirements:
            if not future.done():
                pending.append(future)
            elif future.exception() is not None:
                logger.warning(
                    "[FlexKV] arrival lookup cancel failed: %s", future.exception()
                )
        self._arrival_retirements = pending

    def _drain_arrival_lookups(self, timeout: float = 30.0) -> None:
        lookups = getattr(self, "_arrival_lookups", None)
        if lookups is None:
            return
        while lookups:
            _, (_, _, lookup) = lookups.popitem()
            self._retire_arrival_lookup(lookup)
        self._arrival_store_guard.clear()
        retirements, self._arrival_retirements = self._arrival_retirements, []
        for future in retirements:
            future.result(timeout=timeout)

    def _live_store_covers(self, ns, tokens: np.ndarray, start: int) -> bool:
        guard = self._arrival_store_guard
        if not guard:
            return False
        live = set(self._inflight_stores)
        pending = getattr(self, "_pending_store_batch", None)
        if pending is not None:
            live.update(store["rid"] for store in pending.stores)
        end = start + self.page_size
        page = tokens[start:end]
        covered = False
        for rid in list(guard):
            if rid not in live:
                del guard[rid]  # Completed (committed) or abandoned.
                continue
            store_ns, store_tokens, n = guard[rid]
            if (
                not covered and store_ns == ns and n >= end
                and np.array_equal(np.asarray(store_tokens[start:end], dtype=np.int64), page)
            ):
                covered = True
        return covered

    def _guard_arrival_lookups(self, stores, prepared) -> None:
        """Record new stores and invalidate arrival GETs they could stale."""
        lookups = self._arrival_lookups
        guard = self._arrival_store_guard
        # Only one admission is pending at a time and none is set yet, so a
        # guarded store that is not inflight has committed or been abandoned.
        # Prune here too: issue-time pruning stops when lookups are saturated.
        for rid in [rid for rid in guard if rid not in self._inflight_stores]:
            del guard[rid]
        starts = {start for _, start, _ in lookups.values()}
        for store, (_, _, n) in zip(stores, prepared):
            if not n:
                continue
            ns = store.get("namespace")
            ns = tuple(ns) if ns is not None else None
            tokens = store["token_ids"]
            self._arrival_store_guard[store["rid"]] = (ns, tokens, n)
            for start in starts:
                end = start + self.page_size
                if n < end:
                    continue
                page = np.asarray(tokens[start:end], dtype=np.int64)
                for rid, (_, lookup_start, lookup) in list(lookups.items()):
                    if (
                        lookup_start == start and lookup.namespace == ns
                        and np.array_equal(lookup.token_ids[start:end], page)
                    ):
                        del lookups[rid]
                        self._retire_arrival_lookup(lookup)

    def release_pending(self, rid: str) -> None:
        """Cancel the task held by an earlier ``lookup_kv(rid=...)`` that
        won't be followed by a ``retrieve_kv`` (e.g. allocation failed)."""
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
            self._send_slot_mapping_to_remote(fkv_task_id, slot_mapping_cpu, swa_slot_mapping)

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
    ) -> Tuple[int, int]:
        """Layerwise load. Fires ``launch(layerwise_transfer=True)`` and
        returns ``(n_slots, producer_id)`` after registering the producer
        with the layer hook so the KV pool blocks on the right eventfds."""
        assert self.enable_layerwise and self.layer_done_counter is not None, (
            "start_load_kv_layerwise called but layerwise transfer is "
            "disabled. Set FLEXKV_ENABLE_LAYERWISE_TRANSFER=1."
        )
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
                self._send_slot_mapping_to_remote(fkv_task_id, slot_mapping_cpu, swa_slot_mapping)
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
            # Queued (server-mode) legacy prefetch has no TaskRuntime either:
            # its server only self-drives while a release is draining.
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
        """Whether the ordinary adapter's leader-only pinned copy is safe.

        Hybrid stores carry explicit per-rank Full/SWA snapshots through
        store_kv_batch; the ordinary adapter still stages only the leader.
        """
        return not bool(getattr(self._sync_ctx, "is_pp_active", False)) and (
            self._swa_kv_pool is None
        )

    def sync_ready_store_rids(
        self, ready_rids: List[str], *, require_all_ranks: bool = False,
    ) -> List[str]:
        """Select the identical ready queue prefix, optionally on every rank."""
        payload = list(ready_rids) if self._sync_ctx.is_sync_leader else []
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                payload, channel=FlexKVScatterChannel.STORE_READY,
            )
            # An empty leader queue makes the MIN below necessarily 0. Every
            # rank decides from the same scattered payload, so the collective
            # sequence stays aligned while idle ticks skip a redundant reduce.
            if require_all_ranks and payload:
                common = 0
                for local, leader in zip(ready_rids, payload):
                    if local != leader:
                        break
                    common += 1
                payload = payload[:self._sync_ctx.all_reduce_min(common)]
        return list(payload)

    def store_kv(
        self,
        rid: str,
        token_ids: Sequence[int],
        kv_indices: torch.Tensor,
        sglang_req_id: Any = _SGLANG_REQ_ID_UNSET,
        namespace: Optional[List[str]] = None,
    ) -> int:
        """Ordinary adapter entry point; snapshot before the shared batch path."""
        swa_indices = None
        if getattr(self, "_swa_kv_pool", None) is not None:
            translate = getattr(self._kvcache, "translate_loc_from_full_to_swa", None)
            if translate is not None:
                swa_indices = self._to_cpu_int64(translate(kv_indices))
        return self.store_kv_batch([dict(
            rid=rid, token_ids=token_ids,
            kv_indices=self._to_cpu_int64(kv_indices),
            swa_indices=swa_indices, sglang_req_id=sglang_req_id,
            namespace=namespace,
        )])[0]

    def store_kv_batch(self, stores: Sequence[dict]) -> List[int]:
        """Synchronously reserve and launch using the shared admission protocol."""
        if not stores:
            return []
        batch = self._prepare_store_batch(stores)
        self._pending_store_batch = batch
        self._match_store_batch(batch, asynchronous=False)
        try:
            return self._complete_store_batch(batch)
        except Exception:
            if not batch.retain_slots:
                self._pending_store_batch = None
            raise

    def submit_store_kv_batch(self, stores: Sequence[dict]) -> _StoreAdmission:
        """Queue CPU reservation and launch without blocking on their RPCs.

        All ranks call submit/poll in the same order on the scheduler thread.
        The owner only processes retained CPU token and slot snapshots, never
        reads GPU storage or enters a distributed collective. Callers retain
        source locks through native transfer completion or successful reset.
        ``ValueError`` means every rank rejected the batch before any
        reservation, so callers may release its sources immediately.
        """
        batch = self._prepare_store_batch(stores)
        self._pending_store_batch = batch
        self._match_store_batch(batch, asynchronous=True)
        return batch

    def poll_store_kv_batch(self, batch: _StoreAdmission) -> Optional[List[int]]:
        """Return None while reservation is pending; otherwise finalize once."""
        if batch.reset:
            raise RuntimeError("[FlexKV] store admission was reset")
        if batch.result is not None:
            return batch.result
        if batch is not getattr(self, "_pending_store_batch", None):
            raise ValueError("store admission does not belong to this connector")
        ready = (
            int(batch.future.done()) if self._sync_ctx.is_sync_leader else 1
        )
        if self._sync_ctx.needs_sync:
            ready = self._sync_ctx.all_reduce_min(ready)
        if not ready:
            return None
        return self._complete_store_batch(batch)

    def _prepare_store_batch(self, stores: Sequence[dict]) -> _StoreAdmission:
        contexts = []
        prepared = []
        local_error = ""
        try:
            if getattr(self, "_pending_store_batch", None) is not None:
                raise ValueError("a store admission is already pending")
            seen = set()
            for store in stores:
                if store.get("validation_error"):
                    raise ValueError(store["validation_error"])
                rid = store["rid"]
                if rid in seen or rid in self._inflight_stores:
                    raise ValueError(f"duplicate inflight store rid: {rid!r}")
                seen.add(rid)
                tokens, slots = store["token_ids"], store["kv_indices"]
                swa = store.get("swa_indices")
                if len(tokens) != len(slots):
                    raise ValueError("store token and slot lengths differ")
                for mapping in (slots, swa):
                    if mapping is not None and (
                        mapping.device.type != "cpu" or mapping.dtype != torch.int64
                        or mapping.ndim != 1 or len(mapping) != len(slots)
                    ):
                        raise ValueError("store sidebands must be equal-length CPU int64 vectors")
                n = len(tokens) // self.page_size * self.page_size
                prepared.append((slots[:n], None if swa is None else swa[:n], n))
                contexts.append(self._new_op_context(
                    "store", rid, store.get("sglang_req_id", _SGLANG_REQ_ID_UNSET),
                ))
        except Exception as exc:
            local_error = str(exc)
        if self._sync_ctx.needs_sync:
            valid = self._sync_ctx.all_reduce_min(int(not local_error))
        else:
            valid = int(not local_error)
        if not valid:
            raise ValueError(local_error or "another rank rejected the store batch")
        if getattr(self, "_arrival_lookup_enabled", False) and self._sync_ctx.is_sync_leader:
            self._guard_arrival_lookups(stores, prepared)
        return _StoreAdmission(
            list(stores), prepared, contexts,
            [i for i, p in enumerate(prepared) if p[2]], [],
        )

    def _match_store_batch(self, batch: _StoreAdmission, *, asynchronous: bool) -> None:
        if not self._sync_ctx.is_sync_leader:
            return
        batch.future = Future()
        batch.asynchronous = asynchronous
        batch.starts = [
            dict(rid=s["rid"], task_id=-1, active=False, slot_count=p[2],
                 unmatched_count=0, unmatched_mask=[])
            for s, p in zip(batch.stores, batch.prepared)
        ]
        try:
            batch.requests = [
                dict(
                    token_ids=np.ascontiguousarray(
                        batch.stores[i]["token_ids"], dtype=np.int64,
                    )[:batch.prepared[i][2]],
                    token_mask=None, namespace=batch.stores[i].get("namespace"),
                )
                for i in batch.positions
            ]
            if not batch.requests or self.kv_manager is None:
                batch.future.set_result([])
            elif asynchronous:
                reservation = self.kv_manager.put_match_batch_async(batch.requests)
                reservation.add_done_callback(
                    lambda done: self._continue_store_batch(batch, done)
                )
            else:
                batch.future.set_result(self.kv_manager.put_match_batch(batch.requests))
        except Exception as exc:
            batch.future.set_exception(exc)

    def _prepare_store_launch(self, batch: _StoreAdmission):
        if len(batch.matches) != len(batch.positions):
            raise RuntimeError("store batch match response length mismatch")
        task_ids, mappings, swa_mappings, active_positions, skipped = [], [], [], [], []
        for i, (task_id, mask, needs_transfer) in zip(batch.positions, batch.matches):
            slots, swa, n = batch.prepared[i]
            mask = np.asarray(mask, dtype=np.bool_)
            if mask.ndim != 1 or len(mask) != n:
                raise ValueError("store match mask length mismatch")
            count = int(np.count_nonzero(mask))
            if not needs_transfer:
                skipped.append(int(task_id))
                continue
            mappings.append(slots.numpy()[mask])
            swa_mappings.append(
                None if swa is None else self._trim_swa_slot_mapping(
                    swa.numpy()[mask] if count else swa.numpy()
                )
            )
            # An empty Full mask can still own L2->L3 work and a SWA snapshot.
            task_ids.append(int(task_id))
            active_positions.append(i)
            batch.starts[i].update(
                task_id=int(task_id), unmatched_count=count,
                unmatched_mask=mask if getattr(self._sync_ctx, "is_cross_node_pp", False) else [],
            )
        return task_ids, mappings, swa_mappings, active_positions, skipped

    def _continue_store_batch(self, batch: _StoreAdmission, reservation: Future) -> None:
        # Owner callbacks only touch this batch's retained CPU snapshots. No
        # distributed operation or scheduler-owned bookkeeping runs here.
        try:
            batch.matches = reservation.result()
            if batch.resetting:
                batch.future.set_result(batch.matches)
                return
            task_ids, mappings, swa_mappings, active_positions, skipped = (
                self._prepare_store_launch(batch)
            )

            def launched(done):
                try:
                    done.result()
                    for i in active_positions:
                        batch.starts[i]["active"] = True
                    batch.future.set_result(batch.matches)
                except Exception as exc:
                    batch.future.set_exception(exc)

            def launch():
                try:
                    if batch.resetting or not task_ids:
                        batch.future.set_result(batch.matches)
                        return
                    batch.launch_started = True
                    self.kv_manager.launch_async(
                        task_ids=task_ids, slot_mappings=mappings,
                        swa_slot_mappings=swa_mappings,
                        layerwise_transfer=False,
                    ).add_done_callback(launched)
                except Exception as exc:
                    batch.future.set_exception(exc)

            def retired(done):
                try:
                    done.result()
                    launch()
                except Exception as exc:
                    batch.future.set_exception(exc)

            if skipped:
                self.kv_manager.cancel_async(skipped).add_done_callback(retired)
            else:
                launch()
        except Exception as exc:
            batch.future.set_exception(exc)

    def _complete_store_batch(self, batch: _StoreAdmission) -> List[int]:
        if batch.error:
            raise RuntimeError(f"[FlexKV] store batch failed: {batch.error}")
        try:
            result = self._finalize_store_batch(batch)
        except Exception as exc:
            if not batch.error:
                # Failure after launch/publication cannot be retried as a
                # second launch. Keep the snapshots until reset proves drain.
                batch.error = str(exc)
                batch.retain_slots = True
            raise
        batch.result = result
        self._pending_store_batch = None
        return result

    def _finalize_store_batch(self, batch: _StoreAdmission) -> List[int]:
        stores, prepared, contexts = batch.stores, batch.prepared, batch.contexts

        starts = batch.starts if batch.starts is not None else [
            dict(rid=s["rid"], task_id=-1, active=False, slot_count=p[2],
                 unmatched_count=0, unmatched_mask=[])
            for s, p in zip(stores, prepared)
        ]
        payload = {"stores": starts, "error": ""}
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                batch.matches = batch.future.result()
                if not batch.asynchronous:
                    task_ids, mappings, swa_mappings, active_positions, skipped = (
                        self._prepare_store_launch(batch)
                    )
                    if skipped:
                        self.kv_manager.cancel(skipped)
                    if task_ids:
                        batch.launch_started = True
                        self.kv_manager.launch(
                            task_ids=task_ids, slot_mappings=mappings,
                            swa_slot_mappings=swa_mappings, as_batch=False,
                            layerwise_transfer=False, confirm_launch=True,
                        )
                        for i in active_positions:
                            starts[i]["active"] = True
            except Exception as exc:
                payload["error"] = str(exc)
                allocated = [int(match[0]) for match in batch.matches or ()]
                recovery_task_ids = getattr(exc, "task_ids", None)
                if recovery_task_ids:
                    allocated = list(recovery_task_ids)
                    batch.matches = [(tid, None, None) for tid in allocated]
                if getattr(exc, "retain_slots", False):
                    payload["retain_slots"] = True
                if allocated:
                    try:
                        self.kv_manager.cancel(allocated)
                        if batch.launch_started or recovery_task_ids:
                            responses = self.kv_manager.wait(
                                allocated, timeout=30.0, completely=True,
                            ) or {}
                            if any(
                                tid not in responses
                                or not _is_terminal_status(_status_value(responses[tid]))
                                for tid in allocated
                            ):
                                raise RuntimeError("store launch drain incomplete; retain GPU slots")
                    except Exception as cleanup_error:
                        payload["error"] += f"; cleanup failed: {cleanup_error}"
                        payload["retain_slots"] = True
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                payload, channel=FlexKVScatterChannel.STORE_START,
            )
        if payload["error"]:
            batch.error = payload["error"]
            batch.retain_slots = bool(payload.get("retain_slots"))
            if payload.get("retain_slots"):
                for start, context in zip(payload["stores"], contexts):
                    if start["task_id"] >= 0:
                        context.task_id = start["task_id"]
                        self._inflight_stores[start["rid"]] = start["task_id"]
                        self._inflight_store_contexts[start["rid"]] = context
            raise RuntimeError(f"[FlexKV] store batch failed: {payload['error']}")
        starts = payload["stores"]
        if len(starts) != len(stores):
            raise RuntimeError("[FlexKV] store-start batch length mismatch")
        result = []
        for store, prepared_store, context, start in zip(stores, prepared, contexts, starts):
            rid = store["rid"]
            slots, swa, n = prepared_store
            if start["rid"] != rid or start["slot_count"] != n:
                raise RuntimeError("[FlexKV] store-start rid/slot mismatch")
            if not start["active"]:
                self._log_cache_op(
                    context, "complete", "skipped", direction="D2H",
                    reason="already_cached" if n else "partial_block", tokens=0,
                )
                result.append(-1)
                continue
            task_id = int(start["task_id"])
            if task_id < 0:
                raise RuntimeError("[FlexKV] invalid store task id")
            if self._sync_ctx.should_send_slot_mapping_to_remote:
                mask = np.asarray(start["unmatched_mask"], dtype=np.bool_)
                if len(mask) != n:
                    raise RuntimeError("[FlexKV] invalid cross-node store mask")
                swa_mapping = None if swa is None else self._trim_swa_slot_mapping(
                    swa[mask] if start["unmatched_count"] else swa
                )
                self._send_slot_mapping_to_remote(task_id, slots[mask], swa_mapping)
            context.task_id = task_id
            self._inflight_stores[rid] = task_id
            self._inflight_store_contexts[rid] = context
            self._note_store_launched()
            self._log_cache_op(
                context, "launch", "running", direction="D2H",
                transfer_mode="no-layerwise", tokens=start["unmatched_count"],
            )
            result.append(task_id)
        return result

    def _reset_store_poll(self) -> None:
        self._store_poll_next_ts = 0.0
        self._store_poll_interval_s = _STORE_POLL_MIN_INTERVAL_S

    def _note_store_launched(self) -> None:
        """A new store should be observed soon even if older ones backed off."""
        self._store_poll_interval_s = _STORE_POLL_MIN_INTERVAL_S
        self._store_poll_next_ts = min(
            getattr(self, "_store_poll_next_ts", 0.0),
            time.monotonic() + _STORE_POLL_MIN_INTERVAL_S,
        )

    def _store_poll_due(self) -> bool:
        return time.monotonic() >= getattr(self, "_store_poll_next_ts", 0.0)

    def _advance_store_poll(self, progressed: bool) -> None:
        """Schedule the next leader poll: restart at MIN on progress, otherwise
        use the current gap and double it (capped at MAX) for the one after."""
        if progressed:
            gap = _STORE_POLL_MIN_INTERVAL_S
        else:
            gap = getattr(self, "_store_poll_interval_s", _STORE_POLL_MIN_INTERVAL_S)
        self._store_poll_next_ts = time.monotonic() + gap
        self._store_poll_interval_s = min(
            _STORE_POLL_MAX_INTERVAL_S, max(gap * 2, _STORE_POLL_MIN_INTERVAL_S)
        )

    def check_completed_stores(self) -> List[str]:
        """Return rids whose stores have completed since the last call."""
        completed_rids: List[str] = []
        completed_by_rid: Dict[str, Any] = {}

        # Every rank changes _inflight_stores only from collective payloads
        # (STORE_START / STORE_COMPLETION / STORE_RESET), so all ranks agree
        # that it is empty and skip the STORE_COMPLETION scatter together.
        if not self._inflight_stores:
            self._reset_store_poll()
            return completed_rids

        # Pacing is leader-local: when it skips the RPC, followers still get
        # the (empty) scatter, so the per-tick collective stays in lockstep.
        if (self._sync_ctx.is_sync_leader and self.kv_manager is not None
                and self._store_poll_due()):
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
            self._advance_store_poll(bool(completed_rids))

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
        if not self._inflight_stores:
            self._reset_store_poll()
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
        token_ids: Sequence[int],
        sglang_req_id: Any = _SGLANG_REQ_ID_UNSET,
        namespace: Optional[List[str]] = None,
        *, candidate_start_token: int = 0,
    ) -> int:
        if not self._prefetch_enabled or not rid:
            if (
                rid and getattr(self, "_arrival_lookup_enabled", False)
                and self._sync_ctx.is_sync_leader
            ):
                self._issue_arrival_lookup(rid, token_ids, namespace, candidate_start_token)
            return -1
        if self._uses_queued_prefetch():
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
        if self._uses_queued_prefetch():
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

    def _uses_queued_prefetch(self):
        return bool(getattr(self, "_queued_prefetch", getattr(self, "_chunked_prefetch", False)))

    def _start_chunked_prefetch(self, rid, token_ids, candidate_start_token=0, namespace=None):
        if rid in self._prefetch_sessions:
            return self._prefetch_sessions[rid]["ticket"]
        self._release_prefetch_result(rid)
        ticket = getattr(self, "_next_prefetch_ticket", 0)
        self._next_prefetch_ticket = ticket + 1
        entry = dict(ticket=ticket, admission=None, handle=None, progress=None,
                     demand=False, error=None, lookup=None,
                     candidate_start_token=candidate_start_token)
        payload = {"ticket": ticket, "error": None}
        if self._sync_ctx.is_sync_leader:
            try:
                # One packed snapshot owns the caller's mutable array. Both
                # admissions reuse this same read-only NumPy buffer.
                tokens = np.array(token_ids, dtype=np.int64, copy=True, order="C")
                tokens.setflags(write=False)
                entry["token_ids"] = tokens
                entry["namespace"] = tuple(namespace) if namespace is not None else None
                kwargs = dict(
                    token_ids=tokens,
                    namespace=list(entry["namespace"]) if namespace is not None else None,
                )
                if self._chunked_prefetch:
                    kwargs["options"] = replace(
                        self._chunked_prefetch_options,
                        candidate_start_token=candidate_start_token,
                        swa_aware=self._swa_kv_pool is not None,
                    )
                else:
                    kwargs["swa_aware"] = self._swa_kv_pool is not None
                entry["admission"] = self.kv_manager.prefetch_control_async(
                    "start" if self._chunked_prefetch else "legacy_start", **kwargs,
                )
            except Exception as exc:
                payload["error"] = str(exc)
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                payload, channel=FlexKVScatterChannel.PREFETCH_START,
            )
        if payload["error"]:
            raise RuntimeError("FlexKV prefetch submission failed: " + payload["error"])
        entry["ticket"] = payload["ticket"]
        self._prefetch_sessions[rid] = entry
        # A ticket represents pending admission, not a cache miss. No scheduler
        # thread waits for the server to assign the actual handle.
        return entry["ticket"]

    @staticmethod
    def _finish_prefetch_lookup(target, completed):
        try:
            result = completed.result()
        except Exception as exc:
            target.set_exception(exc)
        else:
            target.set_result(result)

    def _prepare_completed_prefetch_lookups(self, progress, lookups):
        # Run on completion, not the next scheduler tick. These immutable
        # records own every possible late GET before cancellation can retire
        # its prefetch lease. No scheduler dictionaries or collectives here.
        try:
            snapshots = progress.result()
        except Exception as exc:
            for _, lookup in lookups:
                lookup.future.set_exception(exc)
            return
        for handle, lookup in lookups:
            try:
                if not self._queued_prefetch_terminal(snapshots.get(handle)):
                    lookup.future.set_result(None)
                    continue
                future = self.kv_manager.get_match_async(
                    token_ids=lookup.token_ids,
                    token_mask=lookup.token_mask,
                    cpu_only=self._chunked_prefetch,
                    swa_aware=self._swa_kv_pool is not None,
                    namespace=list(lookup.namespace) if lookup.namespace is not None else None,
                )
                future.add_done_callback(
                    lambda done, target=lookup.future: self._finish_prefetch_lookup(target, done)
                )
            except Exception as exc:
                lookup.future.set_exception(exc)

    def _poll_queued_prefetch(self):
        for entry in self._prefetch_sessions.values():
            admission = entry["admission"]
            if entry["handle"] is None and not entry["error"] and admission.done():
                try:
                    entry["handle"] = admission.result()
                except Exception as exc:
                    entry["error"] = str(exc)
        pending = getattr(self, "_prefetch_progress_batch", None)
        if pending is not None:
            future, entries = pending
            if not future.done():
                return
            try:
                snapshots = future.result()
                for entry in entries:
                    entry["progress"] = snapshots.get(entry["handle"])
            except Exception as exc:
                for entry in entries:
                    entry["error"] = str(exc)
            self._prefetch_progress_batch = None
        entries = [
            entry for entry in self._prefetch_sessions.values()
            if entry["handle"] is not None and not entry["error"]
            and not self._queued_prefetch_terminal(entry["progress"])
        ]
        if entries:
            handles = [entry["handle"] for entry in entries]
            if self._chunked_prefetch:
                kwargs = dict(
                    handles=handles,
                    demand_handles=[e["handle"] for e in entries if e["demand"]],
                )
            else:
                kwargs = dict(task_ids=handles)
            lookups = []
            for entry in entries:
                previous = entry["lookup"]
                if previous is None:
                    mask = np.ones(entry["token_ids"].size, dtype=np.bool_)
                    mask[:entry["candidate_start_token"]] = False
                    mask.setflags(write=False)
                else:
                    mask = previous.token_mask
                lookup = _PrefetchLookup(
                    entry["token_ids"], mask, entry["namespace"], Future(),
                )
                entry["lookup"] = lookup
                lookups.append((entry["handle"], lookup))
            try:
                future = self.kv_manager.prefetch_control_async(
                    "progress" if self._chunked_prefetch else "legacy_progress", **kwargs,
                )
            except Exception as exc:
                for _, lookup in lookups:
                    lookup.future.set_exception(exc)
                raise
            self._prefetch_progress_batch = (future, entries)
            future.add_done_callback(
                lambda done: self._prepare_completed_prefetch_lookups(done, lookups)
            )
        self._reap_prefetch_retirements()

    def _queued_prefetch_terminal(self, snapshot):
        if snapshot is None:
            return False
        return (
            snapshot.terminal if self._chunked_prefetch
            else _is_terminal_status(_status_value(snapshot))
        )

    def _check_chunked_prefetch(self, rid):
        entry = self._prefetch_sessions.get(rid)
        if entry is None:
            return True
        entry["demand"] = True
        payload = dict(done=False, error=None, handle=None, loaded_tokens=0, spans=())
        if self._sync_ctx.is_sync_leader:
            try:
                self._poll_queued_prefetch()
                payload["error"] = entry["error"]
                snapshot = entry["progress"]
                if (
                    self._queued_prefetch_terminal(snapshot)
                    and entry["lookup"] is not None
                ):
                    payload.update(done=True, handle=entry["handle"])
                    if self._chunked_prefetch:
                        if snapshot.lease_valid:
                            payload.update(loaded_tokens=snapshot.loaded_tokens, spans=snapshot.l3_loaded_spans)
                    elif _status_value(snapshot) == KVResponseStatus.SUCCESS.value:
                        mask = getattr(snapshot, "return_mask", None)
                        payload["loaded_tokens"] = (
                            sum(int(np.count_nonzero(m)) for m in mask)
                            if isinstance(mask, list) else int(np.count_nonzero(mask))
                        )
            except Exception as exc:
                payload["error"] = str(exc)
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                payload, channel=FlexKVScatterChannel.PREFETCH_PROGRESS,
            )
        if payload["error"]:
            raise RuntimeError("FlexKV prefetch progress failed: " + payload["error"])
        if not payload["done"]:
            return False
        self._prefetch_sessions.pop(rid)
        if self._sync_ctx.is_sync_leader:
            self._prepared_lookups[rid] = entry["lookup"]
        self._prefetch_result_sessions[rid] = payload["handle"]
        self._prefetch_loaded_tokens[rid] = payload["loaded_tokens"]
        self._prefetch_loaded_spans[rid] = payload["spans"]
        return True

    def _retire_prefetch_admission(self, admission, lookup=None):
        """Drain late GET admission/cancellation before releasing its source.

        Completion callbacks only enqueue work on the existing state owner;
        they never call a synchronous manager API or a rank collective.
        """
        retired = Future()
        manager, chunked = self.kv_manager, self._chunked_prefetch

        def released(future):
            try:
                future.result()
                retired.set_result(None)
            except Exception as exc:
                retired.set_exception(exc)

        def admitted(future):
            try:
                handle = future.result()
            except Exception:
                retired.set_result(None)  # No session was created.
                return
            try:
                release = manager.prefetch_control_async(
                    "release" if chunked else "legacy_release",
                    **({"handle": handle} if chunked else {"task_ids": [handle]}),
                )
                release.add_done_callback(released)
            except Exception as exc:
                retired.set_exception(exc)

        def lookup_cancelled(future):
            try:
                future.result()
            except Exception as exc:
                # Retain the source lease if held-GET cleanup is not proven.
                retired.set_exception(exc)
                return
            admission.add_done_callback(admitted)

        def lookup_admitted(future):
            try:
                result = future.result()
            except Exception:
                result = None  # Failed admission created no held task.
            if result is not None:
                task_id, mask = result
                if task_id >= 0 and mask is not None and np.any(mask):
                    try:
                        manager.cancel_async([task_id]).add_done_callback(lookup_cancelled)
                    except Exception as exc:
                        retired.set_exception(exc)
                    return
            # Zero-hit GETs are already complete and must not be cancelled.
            admission.add_done_callback(admitted)

        if lookup is None:
            admission.add_done_callback(admitted)
        else:
            lookup.future.add_done_callback(lookup_admitted)
        self._prefetch_retirements = [
            future for future in getattr(self, "_prefetch_retirements", ())
            if not future.done() or future.exception() is not None
        ]
        self._prefetch_retirements.append(retired)

    def _reap_prefetch_retirements(self):
        """Settle finished retirements without failing the calling request.

        A retirement only releases a prefetch lease, so a failure here leaks
        nothing the scheduler could repair: either the server already freed the
        resources, or it retained them and its own TTL reap will.

        Two reasons this must not propagate. First, retirements are reaped from
        _poll_queued_prefetch(), which runs inside _check_chunked_prefetch(rid)
        for an unrelated rid -- raising reports one request's cleanup failure as
        another request's prefetch failure, and that error is then scattered to
        every CP rank and kills the engine. Second, the pending list has to be
        rebuilt before raising: the previous version raised beforehand, so a
        failed Future stayed in the list forever and every later poll re-raised
        it, leaving the connector permanently broken.
        """
        pending, failures = [], []
        for future in getattr(self, "_prefetch_retirements", ()):
            if not future.done():
                pending.append(future)
                continue
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(exc)
        self._prefetch_retirements = pending
        if not failures:
            return
        # Rate limited: a transport-level fault fails every queued retirement
        # at once, and this runs on the scheduler's hot path once per tick.
        total = getattr(self, "_retirement_failures", 0) + len(failures)
        self._retirement_failures = total
        now = time.monotonic()
        if total - len(failures) < 5 or now - getattr(self, "_retirement_logged_at", 0) >= 60:
            self._retirement_logged_at = now
            logger.warning(
                "[FlexKV] prefetch lease retirement failed (lease dropped, "
                "total=%d): %s", total, failures[0],
            )

    def _release_prefetch_result(self, rid):
        lookup = getattr(self, "_prepared_lookups", {}).pop(rid, None)
        handle = getattr(self, "_prefetch_result_sessions", {}).pop(rid, None)
        if handle is not None and self._sync_ctx.is_sync_leader:
            admission = Future()
            admission.set_result(handle)
            self._retire_prefetch_admission(admission, lookup)

    def _drain_queued_prefetch(self):
        for future in getattr(self, "_prefetch_retirements", ()):
            future.result(timeout=30)
        self._prefetch_retirements = []
        pending = getattr(self, "_prefetch_progress_batch", None)
        if pending is not None:
            pending[0].result(timeout=30)
            self._prefetch_progress_batch = None
        deadline = time.monotonic() + 30
        while not self.kv_manager.prefetch_control_async("drained").result(timeout=30):
            if time.monotonic() >= deadline:
                raise TimeoutError("prefetch reset drain timed out; resources retained")
            time.sleep(0.002)

    def pop_prefetch_loaded_span(self, rid):
        spans = getattr(self, "_prefetch_loaded_spans", {}).pop(rid, ())
        tokens = self.pop_prefetch_loaded_tokens(rid)
        return tokens, spans[0][0] if spans else None

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        """Return the successfully materialized REMOTE2H prefix once."""
        return int(self._prefetch_loaded_tokens.pop(rid, 0))

    def cancel_prefetch(self, rid: str) -> None:
        self._drop_arrival_lookup(rid)
        if self._uses_queued_prefetch():
            entry = self._prefetch_sessions.pop(rid, None)
            if entry is not None and self._sync_ctx.is_sync_leader:
                self._retire_prefetch_admission(entry["admission"], entry["lookup"])
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

    def _drain_pending_store_batch(self) -> None:
        batch = getattr(self, "_pending_store_batch", None)
        if batch is None:
            return
        batch.resetting = True
        status = {"ok": True, "error": ""}
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                try:
                    # Reservation may already have filled matches while launch
                    # is still queued. Fence the complete owner pipeline first.
                    batch.matches = batch.future.result(timeout=30)
                except Exception as exc:
                    if not batch.future.done() or getattr(exc, "retain_slots", False):
                        raise
                    if batch.matches is None:
                        batch.matches = [
                            (tid, None, None) for tid in getattr(exc, "task_ids", ())
                        ]
                task_ids = [int(match[0]) for match in batch.matches]
                if task_ids:
                    self.kv_manager.cancel(task_ids)
                    responses = self.kv_manager.wait(
                        task_ids, timeout=30.0, completely=True,
                    ) or {}
                    if any(
                        tid not in responses
                        or not _is_terminal_status(_status_value(responses[tid]))
                        for tid in task_ids
                    ):
                        raise RuntimeError("store reservation drain incomplete; retain GPU slots")
            except Exception as exc:
                status = {"ok": False, "error": str(exc)}
        if self._sync_ctx.needs_sync:
            status = self._sync_ctx.scatter(
                status, channel=FlexKVScatterChannel.STORE_RESET, blocking=True,
            )
        if not status["ok"]:
            raise RuntimeError("[FlexKV] store admission reset failed: " + status["error"])
        batch.reset = True
        self._pending_store_batch = None

    def reset(self) -> None:
        self._drain_pending_store_batch()
        if self._uses_queued_prefetch():
            for rid in list(self._prefetch_sessions):
                self.cancel_prefetch(rid)
            for rid in list(self._prefetch_result_sessions):
                self._release_prefetch_result(rid)
            self._prefetch_loaded_spans.clear()
            prefetch_reset = {"ok": True, "error": ""}
            if self._sync_ctx.is_sync_leader:
                try:
                    self._drain_queued_prefetch()
                except Exception as exc:
                    prefetch_reset = {"ok": False, "error": str(exc)}
            if self._sync_ctx.needs_sync:
                prefetch_reset = self._sync_ctx.scatter(
                    prefetch_reset, channel=FlexKVScatterChannel.RESET, blocking=True,
                )
            if not prefetch_reset["ok"]:
                raise RuntimeError("FlexKV prefetch reset failed: " + prefetch_reset["error"])
        # Layerwise launch merges the lookup task into a new batch task. Wait
        # for those returned batch task IDs before the scheduler clears GPU
        # slot ownership or the eventfd producer ring is reused. ``completely``
        # is required: an early task-end response is not sufficient to prove
        # that the worker has stopped writing layer completion eventfds.
        load_reset_status = {"ok": True, "error": ""}
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                self._drain_arrival_lookups()
            except Exception as exc:  # noqa: BLE001
                load_reset_status = {
                    "ok": False,
                    "error": f"releasing arrival lookups during reset failed: {exc}",
                }
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
        self._reset_store_poll()
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
        skip_mask = [bool(value) for value in skip_topk_layers]
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
            if buffer.ndim != 2:
                raise RuntimeError(
                    f"FlexKV indexer layer {layer_idx} expects a 2D buffer, "
                    f"got shape={tuple(buffer.shape)}"
                )
            if skip_topk:
                continue
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
        active_ptrs = [buffer.data_ptr() for buffer in active_buffers]
        if len(set(active_ptrs)) != len(active_ptrs):
            raise RuntimeError(
                "FlexKV active indexer buffers unexpectedly alias each other"
            )
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
        _, num_kv_heads, head_size = kv_caches[0].shape
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
        if self.model_config.layer_groups is None:
            layer_groups = None
            if self._is_dsv4:
                layer_groups = self._build_dsv4_layer_group_specs()
            elif indexer_group is not None:
                layer_groups = self._build_indexer_layer_group_specs(
                    kv_caches, indexer_group
                )
            else:
                return

            # FlexKV freezes ModelConfig at the end of its SGLang adapter.
            # Preserve that immutability contract by replacing the config
            # rather than mutating the frozen instance in place.
            model_config = replace(self.model_config, layer_groups=layer_groups)
            model_config.freeze()
            self.model_config = model_config
            self.flexkv_config.model_config = model_config
            self.rank_info = replace(self.rank_info, model_config=model_config)

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
        return self._trim_swa_slot_mapping(swa_indices)

    @staticmethod
    def _trim_swa_slot_mapping(
        swa_indices: torch.Tensor | np.ndarray,
    ) -> Optional[torch.Tensor | np.ndarray]:
        """Keep the mapped CPU suffix without dispatching Torch operations."""
        unmapped = np.flatnonzero(np.asarray(swa_indices) <= 0)
        if unmapped.size:
            swa_indices = swa_indices[int(unmapped[unmapped.size - 1]) + 1 :]
        return swa_indices if len(swa_indices) else None

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
        assert (
            kv_caches[0].ndim == 3
        ), f"Expected 3D KV cache tensor, got shape={kv_caches[0].shape}"

        kv_dim = self.model_config.kv_dim
        num_blocks, physical_num_kv_heads, head_size = kv_caches[0].shape
        global_num_kv_heads = self.model_config.num_kv_heads
        # FlexKV uses num_kv_heads == 1 for KV shared across all TP ranks.
        # TODO: represent partial KV-head replication when 1 < global heads < TP.

        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=self.rank_info.num_layers_per_pp_stage,
            num_block=num_blocks // self.page_size,
            tokens_per_block=self.page_size,
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
            self.tp_client.register_to_server(
                kv_caches=list(kv_caches) + list(indexer_buffers),
                kv_layout=gpu_layout,
                layer_groups=self._build_indexer_layer_group_specs(
                    kv_caches, indexer_group
                ),
                gpu_layouts=[gpu_layout, indexer_layout],
                handles_per_group=[list(kv_caches), list(indexer_buffers)],
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
        self, task_id: int, slot_mapping_cpu: torch.Tensor,
        swa_slot_mapping: Optional[torch.Tensor] = None,
    ) -> None:
        block_size = self.cache_config.tokens_per_block
        full_blocks = slot_mapping_cpu.numpy()[::block_size] // block_size
        swa_blocks = (
            None if swa_slot_mapping is None
            else swa_slot_mapping.numpy()[::block_size] // block_size
        )
        self.tp_client.set_slot_mapping(task_id, full_blocks, swa_blocks)

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
