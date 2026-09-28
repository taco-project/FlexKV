# cython: boundscheck=True, wraparound=True
"""GET / PUT / PREFETCH planning on the radixshmem CPU tier.

`RadixShmemCacheEngine` is `GlobalCacheEngine` with the CPU tier backed by a
radix-server: `CacheEngineRadixShmem`, a `RadixClient` on the shared index and
SlotStore that `shm_radix_bootstrap` brings up. `KVTaskEngine` picks this class
when ``FLEXKV_ENABLE_RADIXSHMEM=1``.

Why a subclass rather than more branches in `GlobalCacheEngine`:

* The tree lives in shared memory and admits a block only once it holds data,
  so a PUT inserts from the graph-completion callback (`StagedRadixInsert`)
  instead of inserting unready nodes at plan time.
* A match is a pinned query (`ShmRadixMatch`), not a locked `RadixNode`; the
  pin drops when the graph completes or the plan is aborted.
* Peer reuse is never spliced into a GET. A prefetch (``ignore_gpu``) starts a
  `RadixClient.get_async` that pulls a peer's run into the local tree; the plan
  has no ops and `KVTaskEngine` completes the task from the job it finds on the
  returned `RadixPlanHandle`.

Not supported here: the SSD and REMOTE tiers, the Redis-backed P2P paths
(``enable_p2p_cpu`` / ``enable_p2p_ssd``) and kv sharing. Peer reuse follows
the radix-server instead (on whenever it is part of a cluster).
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
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.server.shm_radix_bootstrap import (PREFETCH_MAX_INFLIGHT, PREFETCH_TIMEOUT_MS,
                                               expected_geometry, radix_server_name)
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

    ``on_complete`` runs when the graph completes (match-pin release);
    ``on_abort`` when the plan is cancelled before its graph ever launched.
    A PREFETCH plan has an empty graph and carries the `GetJob` instead: the
    pull it started, the local hit it started from and the hit it plans to
    reach (blocks). `KVTaskEngine` completes such a task from the job.
    """

    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int] = field(default_factory=list)
    num_gpu_blocks_to_transfer: int = 0
    on_complete: List[Action] = field(default_factory=list)
    on_abort: List[Action] = field(default_factory=list)
    prefetch_job: Optional[Any] = None
    prefetch_local_hit_blocks: int = 0
    prefetch_planned_hit_blocks: int = 0

    @classmethod
    def empty(cls) -> RadixGetPlan:
        return cls(transfer_graph=TransferOpGraph.create_empty_graph())


@dataclass
class RadixPutPlan:
    """What a radixshmem PUT decided. ``on_complete`` publishes the staged
    slots (`StagedRadixInsert.publish`); ``on_abort`` returns them
    (`StagedRadixInsert.abort`). Both drop the match pin."""

    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int] = field(default_factory=list)
    num_gpu_blocks_to_transfer: int = 0
    skipped_gpu_blocks: int = 0
    on_complete: List[Action] = field(default_factory=list)
    on_abort: List[Action] = field(default_factory=list)

    @classmethod
    def empty(cls) -> RadixPutPlan:
        return cls(transfer_graph=TransferOpGraph.create_empty_graph())


class RadixPlanHandle(TransferPlanHandle):
    """`TransferPlanHandle` plus the prefetch job a PREFETCH plan rides on.

    `KVTaskEngine` reads ``prefetch_job`` off the handle (``getattr``, so the
    base handle needs nothing) and completes a job-backed task from the job
    rather than from graph completion.
    """

    __slots__ = ("prefetch_job", "prefetch_local_hit_blocks",
                 "prefetch_planned_hit_blocks")

    def __init__(self,
                 complete: Action,
                 abort: Action,
                 *,
                 prefetch_job: Optional[Any] = None,
                 prefetch_local_hit_blocks: int = 0,
                 prefetch_planned_hit_blocks: int = 0):
        super().__init__(complete, abort)
        self.prefetch_job = prefetch_job
        self.prefetch_local_hit_blocks = prefetch_local_hit_blocks
        self.prefetch_planned_hit_blocks = prefetch_planned_hit_blocks


def _noop() -> None:
    return None


def _check_cache_config(cache_config: CacheConfig) -> None:
    if not cache_config.enable_cpu:
        raise ValueError("radix_shmem needs enable_cpu=True: it backs the CPU tier")
    if cache_config.enable_ssd or cache_config.enable_remote:
        raise ValueError(
            "radix_shmem backs the CPU tier only; enable_ssd and enable_remote "
            f"must be off (got enable_ssd={cache_config.enable_ssd}, "
            f"enable_remote={cache_config.enable_remote})")
    if cache_config.enable_p2p_cpu or cache_config.enable_p2p_ssd or cache_config.enable_kv_sharing:
        raise ValueError(
            "radix_shmem does its own peer reuse (etcd + RDMA inside the "
            "radix-server, on whenever it is started with cluster flags); "
            "enable_p2p_cpu / enable_p2p_ssd must be off")


class RadixShmemCacheEngine(GlobalCacheEngine):
    """`GlobalCacheEngine` whose CPU tier is a radix-server. See the module
    docstring for what differs from the built-in planners."""

    def __init__(self,
                 cache_config: CacheConfig,
                 model_config: ModelConfig,
                 redis_meta=None,
                 event_collector: Optional[KVEventCollector] = None):
        _check_cache_config(cache_config)
        # GetJobs this engine started and has not yet seen finish; pruned on
        # every prefetch and used for back-pressure (PREFETCH_MAX_INFLIGHT).
        self._prefetch_jobs: List[Any] = []
        super().__init__(cache_config, model_config, redis_meta, event_collector)

    # ------------------------------------------------------------------ tier

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

    def _update_mempool_metrics(self) -> None:
        if self._metrics_collector is None or self.cpu_cache_engine is None:
            return
        tier = self.cpu_cache_engine
        # A radixshmem tier reports its own counts (no Mempool object).
        pool = getattr(tier, "mempool", tier)
        self._metrics_collector.update_mempool_stats(
            "cpu", pool.num_total_blocks, pool.num_free_blocks)

    # ------------------------------------------------------------- entrances

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
            # Prefetch: pull a peer's run into this node's tree.
            plan = self._plan_prefetch(
                request_id, req.sequence_meta, req.block_end_idx, swa_aware=swa_aware)
        else:
            plan = self._plan_get(
                request_id, req.sequence_meta, req.block_start_idx, req.block_end_idx,
                req.gpu_block_ids, dp_client_id, swa_aware=swa_aware)

        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            plan.transfer_graph, plan.finished_ops_ids, dp_client_id)

        tpb = self.tokens_per_block
        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        if plan.prefetch_job is not None:
            # The planned pull [local hit, planned hit); KVTaskEngine rewrites
            # it to what actually landed when the job completes.
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
        # the mask should have a prefix of True
        assert req.block_start_idx == 0
        assert not temp_cache_strategy.ignore_gpu
        if req.block_end_idx == 0:
            return self._empty_result(token_mask)

        plan = self._plan_put(request_id, req.sequence_meta, req.block_start_idx,
                              req.block_end_idx, req.gpu_block_ids, dp_client_id)

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
        if self._metrics_collector is not None:
            self._update_mempool_metrics()
        return transfer_graph, return_mask, handle, {}, task_end_op_id

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

    # ----------------------------------------------------------------- match

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

    # ------------------------------------------------------------------- GET

    def _plan_get(self,
                  request_id: int,
                  sequence_meta: SequenceMeta,
                  block_mask_start: int,
                  block_mask_end: int,
                  gpu_block_ids: np.ndarray,
                  dp_client_id: int,
                  swa_aware: bool = False) -> RadixGetPlan:
        """GET: local match, one H2D (plus the SWA H2D chain when SWA-aware).

        Peer blocks are not spliced in. `_plan_prefetch` (sglang's prefetch
        hooks run it ahead of scheduling) pulls a peer's run into this node's
        tree, so by the time the request is matched here the local hit already
        covers them; whatever is not local is a miss.

        Slots join the tree only once they hold data, so nothing here mutates
        the tree: the match pin is the only state, released at graph completion
        or abort.
        """
        nvtx_range = nvtx.start_range(
            message=f"CacheEngine.plan_get_radixshmem[{request_id}]", color="cyan")
        swa_active = swa_aware and self.swa_op_constructor.enabled
        cpu_match = self._match_cpu(
            sequence_meta, swa_aware=swa_active, swa_query_end=block_mask_end)

        end = min(cpu_match.num_matched_blocks, block_mask_end)
        if end <= block_mask_start:
            # Nothing to restore; drop the query's pin now.
            cpu_match.release()
            if self._metrics_collector is not None and block_mask_end > block_mask_start:
                self._metrics_collector.record_cache_miss(
                    block_mask_end - block_mask_start)
            nvtx.end_range(nvtx_range)
            return RadixGetPlan.empty()

        if self._metrics_collector is not None:
            self._metrics_collector.record_cache_hit("cpu", end - block_mask_start)
            if block_mask_end > end:
                self._metrics_collector.record_cache_miss(block_mask_end - end)

        transfer_graph = TransferOpGraph()
        op_h2d = TransferOp(
            graph_id=transfer_graph.graph_id,
            transfer_type=TransferType.H2D,
            src_block_ids=cpu_match.local_range(block_mask_start, end),
            dst_block_ids=gpu_block_ids[:end - block_mask_start],
            dp_client_id=dp_client_id,
        )
        transfer_graph.add_transfer_op(op_h2d)
        finished_ops_ids = [op_h2d.op_id]

        if swa_active and len(cpu_match.swa_slots) > 0:
            swa_h2d_id = self.swa_op_constructor.build_get_chain(
                transfer_graph,
                gpu_slot_ids=np.zeros(len(cpu_match.swa_slots), dtype=np.int64),
                cpu_slot_ids=cpu_match.swa_slots,
                dp_client_id=dp_client_id,
            )
            if swa_h2d_id is not None:
                finished_ops_ids.append(swa_h2d_id)

        nvtx.end_range(nvtx_range)
        return RadixGetPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            num_gpu_blocks_to_transfer=end - block_mask_start,
            on_complete=[cpu_match.release],
            on_abort=[cpu_match.release],
        )

    # -------------------------------------------------------------- PREFETCH

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
                       swa_aware: bool = False) -> RadixGetPlan:
        """PREFETCH: start a peer pull.

        `RadixClient.get_async` queries the cluster, stages local slots for a
        peer's run, has the radix-server RDMA-read it and publishes the blocks
        into this node's tree when the transfer completes. Nothing moves through
        the TE, so the plan's graph is empty and the task completes when the job
        does (`KVTaskEngine` polls it). A node without peers, or one with too
        many pulls in flight, returns an empty plan: the later local GET says
        what is here.
        """
        assert self.cpu_cache_engine is not None
        engine = self.cpu_cache_engine
        if not engine.peer_enabled:
            return RadixGetPlan.empty()
        inflight = self._prefetch_inflight()
        if inflight >= PREFETCH_MAX_INFLIGHT:
            flexkv_logger.debug(
                f"radixshmem prefetch {request_id}: {inflight} peer pulls in flight "
                f"(limit {PREFETCH_MAX_INFLIGHT}); skipping the peer walk")
            return RadixGetPlan.empty()
        swa_active = swa_aware and self.swa_op_constructor.enabled
        mask = (COMPONENT_MASK_FULL | COMPONENT_MASK_SWA) if swa_active else COMPONENT_MASK_FULL
        job = engine.prefetch(
            sequence_meta,
            component_mask=mask,
            query_end=block_mask_end,
            timeout_ms=PREFETCH_TIMEOUT_MS,
        )
        plan = RadixGetPlan.empty()
        if job is None:
            return plan
        self._prefetch_jobs.append(job)
        plan.prefetch_job = job
        plan.prefetch_local_hit_blocks = int(job.local_hit)
        plan.prefetch_planned_hit_blocks = int(job.planned_hit)
        if self._metrics_collector is not None and job.planned_hit > job.local_hit:
            self._metrics_collector.record_cache_hit("peer", job.planned_hit - job.local_hit)
        return plan

    # ------------------------------------------------------------------- PUT

    def _plan_put(self,
                  request_id: int,
                  sequence_meta: SequenceMeta,
                  block_mask_start: int,
                  block_mask_end: int,
                  gpu_block_ids: np.ndarray,
                  dp_client_id: int) -> RadixPutPlan:
        """PUT: local only, like every PUT.

        The one difference from ``GlobalCacheEngine._put_impl_local`` is WHEN
        the tree learns about the slots. radixshmem accepts them only once they
        hold data, so the insert moves into the graph-completion callback; an
        abort before launch hands the slots back instead.

        Block index:  0        cpu_tot                    block_mask_end
            GPU     : (skipped) |          fragment           |
                                     |  D2H into new slots
            CPU     : (cached) -+
        """
        assert self.cpu_cache_engine is not None
        cpu_engine = self.cpu_cache_engine
        cpu_match = self._match_cpu(sequence_meta)

        def _release_match() -> RadixPutPlan:
            # Nothing will consume the matched prefix; drop the query's pin now.
            cpu_match.release()
            return RadixPutPlan.empty()

        num_skipped = len(cpu_match.local_range(block_mask_start, block_mask_end))
        # Window blocks the CPU tier does not already hold.
        num_cpu_new = block_mask_end - block_mask_start - num_skipped
        # Same policy as _put_impl_local: a fully-matched CPU prefix ends the PUT.
        # This also skips the SWA sidecar, so a window lost to SWA-pool eviction
        # is not republished until the Full path ages out (accepted limitation).
        if num_cpu_new <= 0:
            return _release_match()

        cpu_new = cpu_engine.take(num_required_blocks=num_cpu_new)
        if len(cpu_new) < num_cpu_new:
            flexkv_logger.warning(
                f"radixshmem PUT {request_id} skipped: CPU "
                f"{len(cpu_new)}/{num_cpu_new} slots available"
            )
            cpu_engine.recycle(cpu_new)
            if self._metrics_collector is not None:
                self._metrics_collector.record_allocation_failure("local")
            return _release_match()

        swa_new: Optional[np.ndarray] = None
        try:
            if self.swa_op_constructor.enabled:
                k = min(block_mask_end, self.cache_config.swa.window_blocks)
                swa_take = cpu_engine.take(num_required_blocks=k, component=COMPONENT_SWA)
                if len(swa_take) == k:
                    swa_new = swa_take
                else:
                    # All-or-none contract says this is empty; recycle defensively
                    # in case it ever is not.
                    cpu_engine.recycle(swa_take, component=COMPONENT_SWA)
                    flexkv_logger.warning(
                        f"radixshmem PUT {request_id}: no {k}-slot SWA window "
                        f"available; storing Full KV only"
                    )

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

            if swa_new is not None:
                swa_ops = self.swa_op_constructor.build_put_chain(
                    transfer_graph,
                    gpu_slot_ids=np.zeros(len(swa_new), dtype=np.int64),
                    cpu_slot_ids=swa_new,
                    dp_client_id=dp_client_id,
                    return_op_ids=True,
                )
                assert swa_ops.d2h_id is not None
                finished_ops_ids.append(swa_ops.d2h_id)

            on_complete: List[Action] = []
            on_abort: List[Action] = []

            def _arm(slots: np.ndarray, hold: Optional[Action], label: str,
                     component=COMPONENT_FULL) -> None:
                staged = StagedRadixInsert(engine=cpu_engine,
                                           sequence_meta=sequence_meta,
                                           slots=slots,
                                           path_end=block_mask_end,
                                           label=label,
                                           holds=[] if hold is None else [hold],
                                           component=component)
                on_complete.append(staged.publish)
                on_abort.append(staged.abort)

            # The match pin travels with the LAST publish: SWA's insert refuses paths
            # the Full tree does not reach yet, so it runs after FULL and releases.
            _arm(cpu_new, cpu_match.release if swa_new is None else None,
                 f"PUT {request_id} CPU")
            if swa_new is not None:
                _arm(swa_new, cpu_match.release, f"PUT {request_id} CPU SWA",
                     component=COMPONENT_SWA)

            return RadixPutPlan(
                transfer_graph=transfer_graph,
                finished_ops_ids=finished_ops_ids,
                num_gpu_blocks_to_transfer=len(fragment_gpu_blocks),
                skipped_gpu_blocks=num_skipped,
                on_complete=on_complete,
                on_abort=on_abort,
            )
        except BaseException:
            # Planning failed after slots were taken and before any handle could
            # own them: hand them back, drop the match pin, then re-raise. (A
            # StagedRadixInsert armed above is garbage now; nothing calls it.)
            for slots, comp in ((cpu_new, COMPONENT_FULL), (swa_new, COMPONENT_SWA)):
                if slots is None or len(slots) == 0:
                    continue
                try:
                    if comp == COMPONENT_FULL:
                        cpu_engine.recycle(slots)
                    else:
                        cpu_engine.recycle(slots, component=comp)
                except Exception as e:  # noqa: BLE001 - report, keep unwinding
                    flexkv_logger.error(
                        f"radixshmem PUT {request_id}: could not return {len(slots)} "
                        f"{comp} slots after a planning failure: {e!r}")
            try:
                cpu_match.release()
            except Exception as e:  # noqa: BLE001
                flexkv_logger.error(
                    f"radixshmem PUT {request_id}: could not release the match pin "
                    f"after a planning failure: {e!r}")
            raise
