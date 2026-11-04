# cython: boundscheck=True, wraparound=True
"""GET / PUT / PREFETCH planning on the radixshmem CPU (and SSD) tiers.

`RadixShmemCacheEngine` is `GlobalCacheEngine` with the CPU tier backed by a
radix-server: `CacheEngineRadixShmem`, a `RadixClient` on the shared index and
SlotStore that `shm_radix_bootstrap` brings up. `KVTaskEngine` picks this class
when ``FLEXKV_ENABLE_RADIXSHMEM=1``.

With ``FLEXKV_RADIXSHMEM_SSD_SERVER_NAME`` set and the SSD tier on, the SSD
tier is indexed by a second, index-only radix-server on the node: its slot ids
are this FlexKV's SSD block ids, and being a cluster member it also finds
blocks in a peer's SSD (`ShmRadixMatch.remote_*`).

Why a subclass rather than more branches in `GlobalCacheEngine`:

* The tree lives in shared memory and admits a block only once it holds data,
  so a PUT inserts from the graph-completion callback (`StagedRadixInsert`)
  instead of inserting unready nodes at plan time.
* A match is a pinned query (`ShmRadixMatch`), not a locked `RadixNode`; the
  pin drops when the graph completes or the plan is aborted.
* Peer reuse is never spliced into a GET. A prefetch (``ignore_gpu``) brings
  everything this node does not hold in CPU into the local CPU tree first:
  a peer's CPU run through `RadixClient.pull_async` (the plan carries the
  job, `KVTaskEngine` completes the task once both the job and the graph are
  done), then the SSD tail -- this node's SSD with DISK2H, a peer's SSD with
  PEERSSD2H. A GET reads the local CPU tier, falling back to the local SSD.

Writes go through: a PUT stores the new blocks in the local CPU tier and the
blocks the local SSD lacks in the local SSD.

Not supported here: the REMOTE tier, GDS, the Redis-backed P2P paths
(``enable_p2p_cpu`` / ``enable_p2p_ssd``) and kv sharing. Peer reuse follows
the radix-servers instead (on whenever they are part of a cluster).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import nvtx

from flexkv.cache.cache_engine import (
    DEFAULT_CACHE_STRATEGY,
    CacheStrategy,
    GlobalCacheEngine,
    TransferPlanHandle,
    _synchronized_cache_tree,
)
from flexkv.cache.radix_shmem_engine import (
    COMPONENT_FULL,
    COMPONENT_MASK_FULL,
    COMPONENT_MASK_SWA,
    COMPONENT_SWA,
    CacheEngineRadixShmem,
    ShmRadixMatch,
    StagedRadixInsert,
)
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.pool import PoolId
from flexkv.server.shm_radix_bootstrap import (PREFETCH_MAX_INFLIGHT, PREFETCH_TIMEOUT_MS,
                                               attach_ssd_index_client, expected_geometry,
                                               radix_server_name, radix_ssd_server_name)
from flexkv.common.transfer import (
    DeviceType,
    TransferOp,
    TransferOpGraph,
    TransferType,
    add_virtual_op_for_multiple_finished_ops,
)
from flexkv.integration.dynamo.collector import KVEventCollector

Action = Callable[[], None]


@dataclass
class RadixGetPlan:
    """What a radixshmem GET or PREFETCH decided.

    ``on_complete`` runs when the graph completes (match-pin release, staged
    SSD reads published into the CPU tree); ``on_abort`` when the plan is
    cancelled before its graph ever launched or the graph failed.
    A PREFETCH plan may carry the `PullJob` of a peer CPU pull next to its
    graph (SSD reads), and states the local CPU hit it started from and the
    hit it plans to reach (blocks).
    """

    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int] = field(default_factory=list)
    num_gpu_blocks_to_transfer: int = 0
    on_complete: List[Action] = field(default_factory=list)
    on_abort: List[Action] = field(default_factory=list)
    is_prefetch: bool = False
    prefetch_job: Optional[Any] = None
    prefetch_local_hit_blocks: int = 0
    prefetch_planned_hit_blocks: int = 0
    prefetch_ssd_insert: Optional[StagedRadixInsert] = None

    @classmethod
    def empty(cls) -> RadixGetPlan:
        return cls(transfer_graph=TransferOpGraph.create_empty_graph())


@dataclass
class RadixPutPlan:
    """What a radixshmem PUT decided. ``op_callbacks`` publish each tier as soon
    as the ops writing it complete (CPU at the D2H, SSD at the H2DISK);
    ``on_complete`` / ``on_abort`` settle whatever is left and drop the match
    pins (`StagedRadixInsert` settles once, so a published tier stays)."""

    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int] = field(default_factory=list)
    num_gpu_blocks_to_transfer: int = 0
    skipped_gpu_blocks: int = 0
    on_complete: List[Action] = field(default_factory=list)
    on_abort: List[Action] = field(default_factory=list)
    op_callbacks: Dict[int, Action] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> RadixPutPlan:
        return cls(transfer_graph=TransferOpGraph.create_empty_graph())


@dataclass
class _SsdStage:
    """SSD blocks ``[start, end)`` being read into fresh CPU slots by this graph.

    ``full_slots`` / ``swa_slots`` are the CPU staging slots (SWA: the window
    ending at ``end``); ``full_ops`` / ``swa_ops`` the reads filling them.
    ``inserts`` publish the staged slots into the CPU tree (FULL first) or,
    when the CPU tree cannot hold them contiguously, give them back."""
    start: int
    end: int
    full_slots: np.ndarray
    swa_slots: Optional[np.ndarray]
    full_ops: List[TransferOp]
    swa_ops: List[TransferOp]
    local_blocks: int
    remote_blocks: int
    remote_rank: int
    on_complete: List[Action]
    on_abort: List[Action]
    full_insert: Optional[StagedRadixInsert] = None


class RadixPlanHandle(TransferPlanHandle):
    """`TransferPlanHandle` plus what `KVTaskEngine` needs of a PREFETCH plan.

    `KVTaskEngine` reads ``prefetch_job`` off the handle (``getattr``, so the
    base handle needs nothing) and completes a job-backed task once both the
    job and the graph are done; ``prefetch_ssd_end()`` is how far the staged
    SSD reads were published into the CPU tree (blocks, 0 = none).
    """

    __slots__ = ("prefetch_job", "prefetch_local_hit_blocks",
                 "prefetch_planned_hit_blocks", "_ssd_insert")

    def __init__(self,
                 complete: Action,
                 abort: Action,
                 *,
                 prefetch_job: Optional[Any] = None,
                 prefetch_local_hit_blocks: int = 0,
                 prefetch_planned_hit_blocks: int = 0,
                 ssd_insert: Optional[StagedRadixInsert] = None):
        super().__init__(complete, abort)
        self.prefetch_job = prefetch_job
        self.prefetch_local_hit_blocks = prefetch_local_hit_blocks
        self.prefetch_planned_hit_blocks = prefetch_planned_hit_blocks
        self._ssd_insert = ssd_insert

    def prefetch_ssd_end(self) -> int:
        ins = self._ssd_insert
        if ins is None or ins.landed <= 0:
            return 0
        return int(ins.path_end)


def _noop() -> None:
    return None


def _check_cache_config(cache_config: CacheConfig) -> None:
    if not cache_config.enable_cpu:
        raise ValueError("radix_shmem needs enable_cpu=True: it backs the CPU tier")
    if cache_config.enable_remote:
        raise ValueError(
            "radix_shmem backs the CPU and SSD tiers only; enable_remote must be off "
            f"(got enable_remote={cache_config.enable_remote})")
    if cache_config.enable_ssd:
        if radix_ssd_server_name() is None:
            raise ValueError(
                "radix_shmem with the SSD tier needs FLEXKV_RADIXSHMEM_SSD_SERVER_NAME: the "
                "name of this node's index-only SSD radix-server (`radix-server --name "
                "/flexkv_ssd` without --data-bytes)")
        if cache_config.enable_gds:
            raise ValueError("radix_shmem with the SSD tier does not support GDS")
        if GLOBAL_CONFIG_FROM_ENV.instance_num > 1:
            raise ValueError(
                "radix_shmem with the SSD tier needs one FlexKV per node "
                f"(FLEXKV_INSTANCE_NUM={GLOBAL_CONFIG_FROM_ENV.instance_num}): the SSD index "
                "names this FlexKV's own SSD files")
    if cache_config.enable_p2p_cpu or cache_config.enable_p2p_ssd or cache_config.enable_kv_sharing:
        raise ValueError(
            "radix_shmem does its own peer reuse (etcd + RDMA inside the "
            "radix-server, on whenever it is started with cluster flags); "
            "enable_p2p_cpu / enable_p2p_ssd must be off")


class RadixShmemCacheEngine(GlobalCacheEngine):
    """`GlobalCacheEngine` whose CPU (and SSD) tiers are radix-servers. See the
    module docstring for what differs from the built-in planners."""

    # A GET must never adopt an in-flight prefetch's staging slots: here they
    # are unpinned once published and may be recycled under the GET.
    supports_prefetch_inherit = False

    def __init__(self,
                 cache_config: CacheConfig,
                 model_config: ModelConfig,
                 redis_meta=None,
                 event_collector: Optional[KVEventCollector] = None):
        _check_cache_config(cache_config)
        self._prefetch_jobs: List[Any] = []
        super().__init__(cache_config, model_config, redis_meta, event_collector)

    def _build_cpu_cache_engine(self,
                                cache_config: CacheConfig,
                                event_collector: Optional[KVEventCollector]):
        """Attach to this node's radix-server as a `RadixClient`.

        The server (index + SlotStore + peer transfer) is the operator's
        `radix-server` process. The attach brings FlexKV's geometry
        (idempotent: the KVManager already handed it over and adopted the slot
        counts into `cache_config`) and waits for the server to be ready. Peer
        reuse follows the server: on whenever it is part of a cluster.
        """
        return CacheEngineRadixShmem(
            radix_server_name(),
            geometry=expected_geometry(self.model_config, cache_config),
            tokens_per_block=cache_config.tokens_per_block,
            num_total_blocks=cache_config.num_cpu_blocks,
            swa_config=cache_config.swa,
            event_collector=event_collector,
            metrics_collector=self._metrics_collector,
        )

    def _build_ssd_cache_engine(self,
                                cache_config: CacheConfig,
                                event_collector: Optional[KVEventCollector]):
        """Attach to this node's index-only SSD radix-server.

        Its slot ids are the SSD blocks of this FlexKV's TE, whose files carry
        nothing an earlier run left: an index that survived a FlexKV restart
        (the server outlives FlexKV) would hand out stale blocks, so a
        non-empty index is cleared here, before the TE writes anything."""
        client = attach_ssd_index_client(cache_config, label="RadixShmemCacheEngine-SSD")
        engine = CacheEngineRadixShmem(
            client.name,
            tokens_per_block=cache_config.tokens_per_block,
            num_total_blocks=cache_config.num_ssd_blocks,
            swa_config=cache_config.swa,
            metrics_collector=self._metrics_collector,
            client=client,
            device_type=DeviceType.SSD,
        )
        stale = int(client.total_blocks())
        if stale > 0:
            flexkv_logger.warning(
                f"radixshmem SSD index {engine.shm_name} holds {stale} blocks from an "
                f"earlier FlexKV run; clearing it (their SSD files are not trusted)")
            engine.reset()
        flexkv_logger.info(
            f"radixshmem SSD tier: index {engine.shm_name}, {engine.num_total_blocks} blocks, "
            f"SWA={'on' if engine.swa_enabled else 'off'}, peers="
            f"{'on' if engine.peer_enabled else 'off'} (rank {engine.rank})")
        return engine

    def _update_mempool_metrics(self) -> None:
        if self._metrics_collector is None:
            return
        for label, tier in (("cpu", self.cpu_cache_engine), ("ssd", self.ssd_cache_engine)):
            if tier is None:
                continue
            pool = getattr(tier, "mempool", tier)
            self._metrics_collector.update_mempool_stats(
                label, pool.num_total_blocks, pool.num_free_blocks)

    @_synchronized_cache_tree
    def get(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            dp_client_id: int,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None,
            swa_aware: bool = False) \
                -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        req = self._prepare_request(token_ids, token_mask, slot_mapping, namespace)
        if req.block_end_idx == 0:
            return self._empty_result(token_mask)

        if temp_cache_strategy.ignore_gpu:
            plan = self._plan_prefetch(
                request_id, req.sequence_meta, req.block_end_idx, dp_client_id,
                swa_aware=swa_aware)
        else:
            plan = self._plan_get(
                request_id, req.sequence_meta, req.block_start_idx, req.block_end_idx,
                req.gpu_block_ids, dp_client_id, swa_aware=swa_aware,
                use_ssd=not temp_cache_strategy.ignore_ssd)

        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            plan.transfer_graph, plan.finished_ops_ids, dp_client_id)

        tpb = self.tokens_per_block
        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        if plan.is_prefetch:
            return_mask[plan.prefetch_local_hit_blocks * tpb:
                        plan.prefetch_planned_hit_blocks * tpb] = True
        else:
            return_mask[req.block_start_idx * tpb:
                        (req.block_start_idx + plan.num_gpu_blocks_to_transfer) * tpb] = True

        handle = RadixPlanHandle(
            complete=partial(self._run_actions, plan.on_complete, "completion"),
            abort=partial(self._run_actions, plan.on_abort, "abort"),
            prefetch_job=plan.prefetch_job,
            prefetch_local_hit_blocks=plan.prefetch_local_hit_blocks,
            prefetch_planned_hit_blocks=plan.prefetch_planned_hit_blocks,
            ssd_insert=plan.prefetch_ssd_insert,
        )
        if self._metrics_collector is not None:
            self._update_mempool_metrics()
        return transfer_graph, return_mask, handle, {}, task_end_op_id

    @_synchronized_cache_tree
    def put(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            dp_client_id: int,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None) \
                -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        req = self._prepare_request(token_ids, token_mask, slot_mapping, namespace)
        assert req.block_start_idx == 0
        assert not temp_cache_strategy.ignore_gpu
        if req.block_end_idx == 0:
            return self._empty_result(token_mask)

        plan = self._plan_put(request_id, req.sequence_meta, req.block_start_idx,
                              req.block_end_idx, req.gpu_block_ids, dp_client_id,
                              use_ssd=not temp_cache_strategy.ignore_ssd)

        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            plan.transfer_graph, plan.finished_ops_ids, dp_client_id)

        tpb = self.tokens_per_block
        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        mask_lo = (req.block_start_idx + plan.skipped_gpu_blocks) * tpb
        return_mask[mask_lo:mask_lo + plan.num_gpu_blocks_to_transfer * tpb] = True

        handle = RadixPlanHandle(
            complete=partial(self._run_actions, plan.on_complete, "completion"),
            abort=partial(self._run_actions, plan.on_abort, "abort"),
        )
        op_callbacks = {op_id: partial(self._run_actions, [action], "tier commit")
                        for op_id, action in plan.op_callbacks.items()}
        if self._metrics_collector is not None:
            self._update_mempool_metrics()
        return transfer_graph, return_mask, handle, op_callbacks, task_end_op_id

    @staticmethod
    def _empty_result(token_mask: np.ndarray) \
            -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        return (TransferOpGraph.create_empty_graph(),
                np.zeros_like(token_mask, dtype=np.bool_),
                RadixPlanHandle(complete=_noop, abort=_noop),
                {}, -1)

    @_synchronized_cache_tree
    def _run_actions(self, actions: List[Action], what: str) -> None:
        """Completion / abort of a plan. Every action runs even if one fails:
        a skipped one would leave a pin or staged slots behind for the life of
        the region."""
        for action in actions:
            try:
                action()
            except Exception:
                flexkv_logger.error(
                    f"radixshmem plan {what} action failed", exc_info=True)

    def _match_cpu(self,
                   sequence_meta: SequenceMeta,
                   swa_aware: bool = False,
                   swa_query_end: Optional[int] = None) -> ShmRadixMatch:
        """Local CPU match for the planners (GET and PUT alike).

        radixshmem's `match` never leaves this node: peer blocks reach the local
        tree through `_plan_prefetch`, so a GET sees them as an ordinary local
        hit once the prefetch has completed.

        ``swa_aware`` turns the match into a joint FULL|SWA query capped at
        ``swa_query_end``: its ``num_matched_blocks`` is then the common hit both
        components can serve, and the window (``swa_slots``) ends exactly there.
        """
        assert self.cpu_cache_engine is not None
        if swa_aware:
            return self.cpu_cache_engine.match(
                sequence_meta,
                component_mask=COMPONENT_MASK_FULL | COMPONENT_MASK_SWA,
                query_end=swa_query_end,
            )
        return self.cpu_cache_engine.match(sequence_meta)

    def _ssd_swa_active(self, swa_active: bool) -> bool:
        return (swa_active and self.ssd_cache_engine is not None
                and self.ssd_cache_engine.swa_enabled)

    def _stage_from_ssd(self,
                        request_id: int,
                        sequence_meta: SequenceMeta,
                        transfer_graph: TransferOpGraph,
                        start: int,
                        end: int,
                        dp_client_id: int,
                        *,
                        swa_active: bool,
                        local_only: bool,
                        publish: bool) -> Optional[_SsdStage]:
        """Read SSD blocks ``[start, S)`` (S = the SSD hit, capped at ``end``)
        into fresh CPU slots: this node's SSD with DISK2H, a peer's with
        PEERSSD2H. With ``swa_active`` the hit is the joint FULL|SWA one and
        the SWA window ending at S is read too.

        ``publish``: on completion insert the staged slots into the CPU tree
        (the caller guarantees the tree reaches ``start`` by then); otherwise
        give them back. The SSD match pin (peer blocks included) is held until
        then. Returns None, with nothing held, when the SSD has nothing past
        ``start`` or the CPU tier cannot stage it."""
        ssd = self.ssd_cache_engine
        cpu = self.cpu_cache_engine
        if ssd is None or end <= start:
            return None
        ssd_swa = self._ssd_swa_active(swa_active)
        if swa_active and not ssd_swa:
            return None  # an SWA-aware restore needs the window; the SSD keeps none
        mask = (COMPONENT_MASK_FULL | COMPONENT_MASK_SWA) if ssd_swa else COMPONENT_MASK_FULL
        match = ssd.match(sequence_meta, component_mask=mask, query_end=end,
                          local_only=local_only)
        stop = min(match.num_matched_blocks, end)
        if stop <= start or (ssd_swa and len(match.swa_slots) == 0):
            match.release()
            return None

        n = stop - start
        full_slots = cpu.take(num_required_blocks=n)
        swa_slots: Optional[np.ndarray] = None
        if len(full_slots) == n and ssd_swa:
            swa_slots = cpu.take(num_required_blocks=len(match.swa_slots), component=COMPONENT_SWA)
            if len(swa_slots) != len(match.swa_slots):
                cpu.recycle(swa_slots, component=COMPONENT_SWA)
                swa_slots = None
        if len(full_slots) < n or (ssd_swa and swa_slots is None):
            cpu.recycle(full_slots)
            match.release()
            flexkv_logger.warning(
                f"radixshmem {request_id}: no CPU room to stage {n} SSD blocks; skipping the SSD")
            if self._metrics_collector is not None:
                self._metrics_collector.record_allocation_failure("local")
            return None

        nl = match.num_local_blocks
        local_lo, local_hi = start, min(stop, nl)
        remote_lo = max(start, nl)
        full_ops: List[TransferOp] = []
        local_blocks = max(0, local_hi - local_lo)
        remote_blocks = stop - remote_lo if stop > remote_lo else 0
        if local_blocks > 0:
            full_ops += self._make_sliced_host_ops(
                transfer_graph,
                transfer_type=TransferType.DISK2H,
                src_block_ids=match.local_slots[local_lo:local_hi],
                dst_block_ids=full_slots[:local_blocks],
                dp_client_id=dp_client_id,
            )
        if remote_blocks > 0:
            op = TransferOp(
                graph_id=transfer_graph.graph_id,
                transfer_type=TransferType.PEERSSD2H,
                src_block_ids=match.remote_slots[remote_lo - nl:stop - nl],
                dst_block_ids=full_slots[local_blocks:],
                dp_client_id=dp_client_id,
                src_block_node_ids=np.full(remote_blocks, match.remote_rank, dtype=np.int64),
            )
            transfer_graph.add_transfer_op(op)
            full_ops.append(op)

        swa_ops: List[TransferOp] = []
        if swa_slots is not None:
            remote_swa = match.swa_rank != ssd.rank
            op = TransferOp(
                graph_id=transfer_graph.graph_id,
                transfer_type=TransferType.PEERSSD2H if remote_swa else TransferType.DISK2H,
                src_block_ids=match.swa_slots,
                dst_block_ids=swa_slots,
                dp_client_id=dp_client_id,
                pool_id=PoolId.SWA,
                src_block_node_ids=(np.full(len(swa_slots), match.swa_rank, dtype=np.int64)
                                    if remote_swa else None),
            )
            transfer_graph.add_transfer_op(op)
            swa_ops.append(op)

        on_complete: List[Action] = []
        on_abort: List[Action] = []
        full_insert = StagedRadixInsert(engine=cpu, sequence_meta=sequence_meta,
                                        slots=full_slots, path_end=stop,
                                        label=f"{request_id} SSD->CPU",
                                        holds=[match.release] if swa_slots is None else [])
        inserts = [full_insert]
        if swa_slots is not None:
            inserts.append(StagedRadixInsert(engine=cpu, sequence_meta=sequence_meta,
                                             slots=swa_slots, path_end=stop,
                                             label=f"{request_id} SSD->CPU SWA",
                                             holds=[match.release], component=COMPONENT_SWA))
        for staged in inserts:  # FULL first: the SWA insert needs its path
            on_complete.append(staged.publish if publish else staged.abort)
            on_abort.append(staged.abort)
        if publish:
            def _report() -> None:
                flexkv_logger.info(
                    "[FlexKV-IO] operation=ssd_stage act=publish flexkv_task_id=%d "
                    "start=%d end=%d local_blocks=%d peer_blocks=%d peer_rank=%d "
                    "landed=%d swa_landed=%s",
                    request_id, start, stop, local_blocks, remote_blocks,
                    match.remote_rank if remote_blocks > 0 else -1, full_insert.landed,
                    inserts[1].landed if len(inserts) > 1 else "-")
            on_complete.append(_report)
        return _SsdStage(start=start, end=stop, full_slots=full_slots, swa_slots=swa_slots,
                         full_ops=full_ops, swa_ops=swa_ops, local_blocks=local_blocks,
                         remote_blocks=remote_blocks,
                         remote_rank=match.remote_rank if remote_blocks > 0 else -1,
                         on_complete=on_complete, on_abort=on_abort,
                         full_insert=full_insert if publish else None)

    def _plan_get(self,
                  request_id: int,
                  sequence_meta: SequenceMeta,
                  block_mask_start: int,
                  block_mask_end: int,
                  gpu_block_ids: np.ndarray,
                  dp_client_id: int,
                  swa_aware: bool = False,
                  use_ssd: bool = True) -> RadixGetPlan:
        """GET: local CPU match, then the local SSD for what follows; H2D (plus
        the SWA H2D chain when SWA-aware).

        Peer blocks are not spliced in. `_plan_prefetch` (sglang's prefetch
        hooks run it ahead of scheduling) brings a peer's CPU run and the SSD
        tail into this node's CPU tree, so by the time the request is matched
        here the local CPU hit normally covers them. Blocks read from the local
        SSD here are published into the CPU tree too when it reaches them.

        Block index: 0      start      cpu_end           ssd_end      end
            GPU    : (cached) |   H2D      |   H2D             |  (compute)
            CPU    :  hit ----------------- | staged <- DISK2H  |
        """
        nvtx_range = nvtx.start_range(
            message=f"CacheEngine.plan_get_radixshmem[{request_id}]", color="cyan")
        swa_active = swa_aware and self.swa_op_constructor.enabled
        cpu_match = self._match_cpu(
            sequence_meta, swa_aware=swa_active, swa_query_end=block_mask_end)
        cpu_end = min(cpu_match.num_matched_blocks, block_mask_end)

        transfer_graph = TransferOpGraph()
        stage: Optional[_SsdStage] = None
        if use_ssd and self.ssd_cache_engine is not None:
            stage_start = max(cpu_end, block_mask_start)
            stage = self._stage_from_ssd(
                request_id, sequence_meta, transfer_graph, stage_start, block_mask_end,
                dp_client_id, swa_active=swa_active, local_only=True,
                publish=cpu_end >= block_mask_start)

        end = stage.end if stage is not None else cpu_end
        if end <= block_mask_start:
            cpu_match.release()
            if self._metrics_collector is not None and block_mask_end > block_mask_start:
                self._metrics_collector.record_cache_miss(
                    block_mask_end - block_mask_start)
            nvtx.end_range(nvtx_range)
            return RadixGetPlan.empty()

        cpu_part = max(0, cpu_end - block_mask_start)
        ssd_part = stage.end - stage.start if stage is not None else 0
        if self._metrics_collector is not None:
            self._metrics_collector.record_cache_hit("cpu", cpu_part)
            self._metrics_collector.record_cache_hit("ssd", ssd_part)
            if block_mask_end > end:
                self._metrics_collector.record_cache_miss(block_mask_end - end)

        src = cpu_match.local_range(block_mask_start, cpu_end) if cpu_part > 0 \
            else np.empty(0, dtype=np.int64)
        if stage is not None:
            src = np.concatenate([src, stage.full_slots])
        h2d_ops = self._build_get_h2d_ops(
            transfer_graph=transfer_graph,
            cpu_blocks=src,
            gpu_blocks=gpu_block_ids[:end - block_mask_start],
            resident_num_blocks=cpu_part,
            split=False,
            dp_client_id=dp_client_id,
            staged_predecessors=stage.full_ops if stage is not None else [],
        )
        finished_ops_ids = [op.op_id for op in h2d_ops]

        if swa_active:
            if stage is not None and stage.swa_slots is not None:
                swa_h2d_id = self.swa_op_constructor.build_swa_op(
                    transfer_graph, TransferType.H2D, stage.swa_slots,
                    np.zeros(len(stage.swa_slots), dtype=np.int64),
                    dp_client_id=dp_client_id)
                if swa_h2d_id is not None:
                    for op in stage.swa_ops:
                        transfer_graph.add_dependency(swa_h2d_id, op.op_id)
                    finished_ops_ids.append(swa_h2d_id)
            elif len(cpu_match.swa_slots) > 0:
                swa_h2d_id = self.swa_op_constructor.build_get_chain(
                    transfer_graph,
                    gpu_slot_ids=np.zeros(len(cpu_match.swa_slots), dtype=np.int64),
                    cpu_slot_ids=cpu_match.swa_slots,
                    dp_client_id=dp_client_id,
                )
                if swa_h2d_id is not None:
                    finished_ops_ids.append(swa_h2d_id)

        on_complete: List[Action] = []
        on_abort: List[Action] = []
        if stage is not None:
            on_complete += stage.on_complete
            on_abort += stage.on_abort
            flexkv_logger.info(
                "[FlexKV-IO] operation=get act=ssd_read flexkv_task_id=%d "
                "cpu_blocks=%d ssd_blocks=%d swa=%s publish=%s",
                request_id, cpu_part, ssd_part, stage.swa_slots is not None,
                cpu_end >= block_mask_start)
        on_complete.append(cpu_match.release)
        on_abort.append(cpu_match.release)

        nvtx.end_range(nvtx_range)
        return RadixGetPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            num_gpu_blocks_to_transfer=end - block_mask_start,
            on_complete=on_complete,
            on_abort=on_abort,
        )

    def _prefetch_inflight(self) -> int:
        """Peer pulls this engine started that have not finished yet."""
        live = [job for job in self._prefetch_jobs
                if not job.done() and not getattr(job, "cancelled", False)]
        self._prefetch_jobs = live
        return len(live)

    def _plan_prefetch(self,
                       request_id: int,
                       sequence_meta: SequenceMeta,
                       block_mask_end: int,
                       dp_client_id: int,
                       swa_aware: bool = False) -> RadixGetPlan:
        """PREFETCH: bring everything this node does not hold in CPU into the
        local CPU tree, CPU before SSD.

        Block index: 0         L0            P             S        end
            CPU    : local hit | peer CPU run |                        (pull_async)
            SSD    :                          | local DISK2H | peer PEERSSD2H

        1. A peer's CPU run ``[L0, P)``: `RadixClient.pull_async` queries the
           cluster, stages local slots, has the radix-server RDMA-read them and
           publishes them into the local tree. The plan carries the job.
        2. The SSD tail ``[max(L0, P), S)``: a cluster-wide SSD match (this
           node's run, then at most one peer's), read into fresh CPU slots by
           this plan's graph and published into the CPU tree once both the
           graph and the pull are done (`KVTaskEngine` waits for both): the
           insert needs ``[0, max(L0, P))`` on the tree first. A pull that
           lands short only wastes the SSD read -- radixshmem refuses the
           disconnected insert and recycles the slots.

        The local CPU prefix stays pinned until the plan settles, so staging
        cannot evict it. Nothing to do yields an empty plan: the later GET
        says what is here.
        """
        assert self.cpu_cache_engine is not None
        engine = self.cpu_cache_engine
        swa_active = swa_aware and self.swa_op_constructor.enabled
        mask = (COMPONENT_MASK_FULL | COMPONENT_MASK_SWA) if swa_active else COMPONENT_MASK_FULL
        plan = RadixGetPlan.empty()
        plan.is_prefetch = True

        cpu_match = self._match_cpu(sequence_meta, swa_aware=swa_active,
                                    swa_query_end=block_mask_end)
        local_hit = min(cpu_match.num_matched_blocks, block_mask_end)
        planned = local_hit

        job = None
        if engine.peer_enabled:
            inflight = self._prefetch_inflight()
            if inflight >= PREFETCH_MAX_INFLIGHT:
                flexkv_logger.debug(
                    f"radixshmem prefetch {request_id}: {inflight} peer pulls in flight "
                    f"(limit {PREFETCH_MAX_INFLIGHT}); skipping the peer walk")
            else:
                job = engine.prefetch(
                    sequence_meta,
                    component_mask=mask,
                    query_end=block_mask_end,
                    timeout_ms=PREFETCH_TIMEOUT_MS,
                )
        if job is not None:
            self._prefetch_jobs.append(job)
            plan.prefetch_job = job
            local_hit = min(local_hit, int(job.local_hit))
            planned = max(planned, int(job.planned_hit))
            if self._metrics_collector is not None and job.planned_hit > job.local_hit:
                self._metrics_collector.record_cache_hit("peer", job.planned_hit - job.local_hit)

        stage = None
        if self.ssd_cache_engine is not None and planned < block_mask_end:
            stage = self._stage_from_ssd(
                request_id, sequence_meta, plan.transfer_graph, planned, block_mask_end,
                dp_client_id, swa_active=swa_active, local_only=False, publish=True)
        if stage is None and job is None:
            cpu_match.release()
            plan.prefetch_local_hit_blocks = plan.prefetch_planned_hit_blocks = local_hit
            return plan

        if stage is not None:
            plan.finished_ops_ids = [op.op_id for op in stage.full_ops + stage.swa_ops]
            plan.on_complete += stage.on_complete
            plan.on_abort += stage.on_abort
            plan.prefetch_ssd_insert = stage.full_insert
            if self._metrics_collector is not None:
                self._metrics_collector.record_cache_hit("ssd", stage.local_blocks)
                self._metrics_collector.record_cache_hit("peer_ssd", stage.remote_blocks)
            flexkv_logger.info(
                "[FlexKV-IO] operation=prefetch act=ssd_read flexkv_task_id=%d "
                "cpu_local_blocks=%d cpu_peer_planned=%d ssd_start=%d ssd_end=%d "
                "ssd_local_blocks=%d ssd_peer_blocks=%d ssd_peer_rank=%d swa=%s",
                request_id, local_hit, planned - local_hit, stage.start, stage.end,
                stage.local_blocks, stage.remote_blocks, stage.remote_rank,
                stage.swa_slots is not None)
            planned = stage.end
        plan.on_complete.append(cpu_match.release)
        plan.on_abort.append(cpu_match.release)
        plan.prefetch_local_hit_blocks = local_hit
        plan.prefetch_planned_hit_blocks = planned
        return plan

    def _plan_put(self,
                  request_id: int,
                  sequence_meta: SequenceMeta,
                  block_mask_start: int,
                  block_mask_end: int,
                  gpu_block_ids: np.ndarray,
                  dp_client_id: int,
                  use_ssd: bool = True) -> RadixPutPlan:
        """PUT: local only, like every PUT; written through to the local SSD.

        The difference from ``GlobalCacheEngine._put_impl_cpu_ssd`` is WHEN the
        trees learn about the slots: radixshmem accepts them only once they
        hold data, so each tier is inserted from the completion of the ops
        writing it (CPU at the D2H, SSD at the H2DISK); an abort hands the
        slots back instead.

        Block index:  0        cpu_tot                    block_mask_end
            GPU     : (skipped) |          fragment           |
                                     |  D2H into new slots
            CPU     : (cached) -+--- new -------------------- |
                             ssd_tot   |  H2DISK (from cached + new CPU)
            SSD     : (cached) |  new ----------------------- |

        Like the built-in planner the SSD is written only along with new CPU
        blocks: a prefix the CPU tier already holds (for instance pulled from a
        peer) is not copied into this node's SSD.
        """
        assert self.cpu_cache_engine is not None
        cpu_engine = self.cpu_cache_engine
        ssd_engine = self.ssd_cache_engine if use_ssd else None
        cpu_match = self._match_cpu(sequence_meta)
        ssd_match: Optional[ShmRadixMatch] = None

        def _release_matches() -> RadixPutPlan:
            cpu_match.release()
            if ssd_match is not None:
                ssd_match.release()
            return RadixPutPlan.empty()

        num_skipped = len(cpu_match.local_range(block_mask_start, block_mask_end))
        num_cpu_new = block_mask_end - block_mask_start - num_skipped
        if num_cpu_new <= 0:
            return _release_matches()

        cpu_new = cpu_engine.take(num_required_blocks=num_cpu_new)
        if len(cpu_new) < num_cpu_new:
            flexkv_logger.warning(
                f"radixshmem PUT {request_id} skipped: CPU "
                f"{len(cpu_new)}/{num_cpu_new} slots available"
            )
            cpu_engine.recycle(cpu_new)
            if self._metrics_collector is not None:
                self._metrics_collector.record_allocation_failure("local")
            return _release_matches()

        swa_new: Optional[np.ndarray] = None
        ssd_new: Optional[np.ndarray] = None
        ssd_swa_new: Optional[np.ndarray] = None
        try:
            if self.swa_op_constructor.enabled:
                k = min(block_mask_end, self.cache_config.swa.window_blocks)
                swa_take = cpu_engine.take(num_required_blocks=k, component=COMPONENT_SWA)
                if len(swa_take) == k:
                    swa_new = swa_take
                else:
                    cpu_engine.recycle(swa_take, component=COMPONENT_SWA)
                    flexkv_logger.warning(
                        f"radixshmem PUT {request_id}: no {k}-slot SWA window "
                        f"available; storing Full KV only"
                    )

            # The SSD part: [ssd_tot, end), sourced from the cached CPU slots
            # below cpu_tot (pinned by cpu_match) and the new ones above it.
            ssd_tot = 0
            if ssd_engine is not None:
                ssd_match = ssd_engine.match(sequence_meta)
                ssd_tot = min(ssd_match.num_matched_blocks, block_mask_end)
                num_ssd_new = block_mask_end - ssd_tot
                if num_ssd_new > 0:
                    take = ssd_engine.take(num_required_blocks=num_ssd_new)
                    if len(take) == num_ssd_new:
                        ssd_new = take
                    else:
                        ssd_engine.recycle(take)
                        flexkv_logger.warning(
                            f"radixshmem PUT {request_id}: SSD {len(take)}/{num_ssd_new} "
                            f"blocks available; storing in CPU only")
                if ssd_new is not None and swa_new is not None and ssd_engine.swa_enabled:
                    swa_take = ssd_engine.take(num_required_blocks=len(swa_new),
                                               component=COMPONENT_SWA)
                    if len(swa_take) == len(swa_new):
                        ssd_swa_new = swa_take
                    else:
                        ssd_engine.recycle(swa_take, component=COMPONENT_SWA)
                        flexkv_logger.warning(
                            f"radixshmem PUT {request_id}: no {len(swa_new)}-slot SSD SWA "
                            f"window available; the SSD stores Full KV only")

            transfer_graph = TransferOpGraph()
            finished_ops_ids: List[int] = []

            fragment_gpu_blocks = gpu_block_ids[num_skipped:]
            op_d2h = TransferOp(
                graph_id=transfer_graph.graph_id,
                transfer_type=TransferType.D2H,
                src_block_ids=fragment_gpu_blocks,
                dst_block_ids=cpu_new,
                dp_client_id=dp_client_id,
            )
            transfer_graph.add_transfer_op(op_d2h)
            finished_ops_ids.append(op_d2h.op_id)

            swa_d2h_id: Optional[int] = None
            swa_h2disk_id: Optional[int] = None
            if swa_new is not None:
                swa_ops = self.swa_op_constructor.build_put_chain(
                    transfer_graph,
                    gpu_slot_ids=np.zeros(len(swa_new), dtype=np.int64),
                    cpu_slot_ids=swa_new,
                    ssd_slot_ids=ssd_swa_new,
                    dp_client_id=dp_client_id,
                    return_op_ids=True,
                )
                assert swa_ops.d2h_id is not None
                swa_d2h_id = swa_ops.d2h_id
                swa_h2disk_id = swa_ops.h2disk_id
                finished_ops_ids.append(swa_d2h_id)

            h2disk_ops: List[TransferOp] = []
            if ssd_new is not None:
                cpu_tot = block_mask_start + num_skipped
                cached_src = cpu_match.local_range(ssd_tot, cpu_tot) if ssd_tot < cpu_tot \
                    else np.empty(0, dtype=np.int64)
                new_src = cpu_new[max(0, ssd_tot - cpu_tot):]
                h2disk_ops = self._make_sliced_host_ops(
                    transfer_graph,
                    transfer_type=TransferType.H2DISK,
                    src_block_ids=np.concatenate([cached_src, new_src]),
                    dst_block_ids=ssd_new,
                    dp_client_id=dp_client_id,
                )
                for op in h2disk_ops:
                    transfer_graph.add_dependency(op.op_id, op_d2h.op_id)

            on_complete: List[Action] = []
            on_abort: List[Action] = []
            op_callbacks: Dict[int, Action] = {}

            def _arm_tier(engine: CacheEngineRadixShmem,
                          parts: List[Tuple[Optional[np.ndarray], Any, str]],
                          writers: List[Optional[int]]) -> None:
                """Publish one tier's staged slots (FULL before SWA: the SWA
                insert needs the FULL path) once every op writing that tier has
                completed -- they complete in any order -- or at graph end."""
                staged = [StagedRadixInsert(engine=engine, sequence_meta=sequence_meta,
                                            slots=slots, path_end=block_mask_end,
                                            label=label, component=component)
                          for slots, component, label in parts if slots is not None]

                def _publish() -> None:
                    for ins in staged:
                        ins.publish()
                pending = {w for w in writers if w is not None}
                for op_id in list(pending):
                    def _on_write(op_id: int = op_id) -> None:
                        pending.discard(op_id)
                        if not pending:
                            _publish()
                    op_callbacks[op_id] = _on_write
                on_complete.append(_publish)
                on_abort.extend(ins.abort for ins in staged)

            # CPU readable at its D2H(s); the SSD at its H2DISK(s).
            _arm_tier(cpu_engine,
                      [(cpu_new, COMPONENT_FULL, f"PUT {request_id} CPU"),
                       (swa_new, COMPONENT_SWA, f"PUT {request_id} CPU SWA")],
                      [op_d2h.op_id, swa_d2h_id])
            if ssd_new is not None:
                _arm_tier(ssd_engine,
                          [(ssd_new, COMPONENT_FULL, f"PUT {request_id} SSD"),
                           (ssd_swa_new, COMPONENT_SWA, f"PUT {request_id} SSD SWA")],
                          [op.op_id for op in h2disk_ops] + [swa_h2disk_id])
            # The H2DISK reads cached CPU slots: drop the pins at graph end.
            on_complete.append(cpu_match.release)
            on_abort.append(cpu_match.release)
            if ssd_match is not None:
                on_complete.append(ssd_match.release)
                on_abort.append(ssd_match.release)

            if ssd_new is not None:
                flexkv_logger.info(
                    "[FlexKV-IO] operation=put act=ssd_write flexkv_task_id=%d "
                    "cpu_new=%d ssd_start=%d ssd_new=%d swa=%s",
                    request_id, len(cpu_new), ssd_tot, len(ssd_new), ssd_swa_new is not None)

            return RadixPutPlan(
                transfer_graph=transfer_graph,
                finished_ops_ids=finished_ops_ids,
                num_gpu_blocks_to_transfer=len(fragment_gpu_blocks),
                skipped_gpu_blocks=num_skipped,
                on_complete=on_complete,
                on_abort=on_abort,
                op_callbacks=op_callbacks,
            )
        except BaseException:
            for eng, slots, comp in ((cpu_engine, cpu_new, COMPONENT_FULL),
                                     (cpu_engine, swa_new, COMPONENT_SWA),
                                     (ssd_engine, ssd_new, COMPONENT_FULL),
                                     (ssd_engine, ssd_swa_new, COMPONENT_SWA)):
                if eng is None or slots is None or len(slots) == 0:
                    continue
                try:
                    eng.recycle(slots, component=comp)
                except Exception as e:  # noqa: BLE001 - report, keep unwinding
                    flexkv_logger.error(
                        f"radixshmem PUT {request_id}: could not return {len(slots)} "
                        f"{comp} slots after a planning failure: {e!r}")
            try:
                _release_matches()
            except Exception as e:  # noqa: BLE001
                flexkv_logger.error(
                    f"radixshmem PUT {request_id}: could not release the match pins "
                    f"after a planning failure: {e!r}")
            raise
