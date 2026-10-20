"""Radixshmem chunk planner: peer pulls via ``pull_async`` sub-ranges.

Implements ``ChunkPlannerBase`` so ``PrefetchCoordinator`` can drive
wait_complete / timeout / best_effort without REMOTE2H graphs. Each chunk is
one ``RadixClient.pull_async`` over a growing hash prefix; completions are
polled (no transfer-engine pipe).
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flexkv.common.block import SequenceMeta
from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import TransferOpGraph
from flexkv.prefetch.base_planner import ChunkPlannerBase
from flexkv.prefetch.types import PrefetchCapacityExhausted
from flexkv.server.shm_radix_bootstrap import PREFETCH_MAX_INFLIGHT, PREFETCH_TIMEOUT_MS


@dataclass
class RadixPrefetchContext:
    tokens: np.ndarray
    namespace: Optional[list]
    client_id: int
    future: Optional[Future] = None
    sequence: Optional[SequenceMeta] = None
    match: Any = None  # ShmRadixMatch pin for the reusable prefix
    reusable_end: int = 0
    pinned_bytes: int = 0
    released: bool = False
    cancel_query: threading.Event = field(default_factory=threading.Event)
    component_mask: int = 0


@dataclass
class RadixPrefetchChunk:
    begin: int
    end: int
    nbytes: int
    graph: TransferOpGraph
    pending: Any = None
    op_id: int = -1
    done: bool = False
    failed: bool = False
    completion: Any = None
    resolved: bool = False
    job: Any = None  # PullJob
    block_begin: int = 0
    block_end: int = 0
    published_end: int = 0


class _FakeCompletion:
    """Mirrors CompletedOp fields the coordinator's on_completion expects."""

    __slots__ = ("graph_id", "op_id", "failed")

    def __init__(self, graph_id: int, *, failed: bool = False):
        self.graph_id = graph_id
        self.op_id = -1
        self.failed = failed


class RadixShmemChunkPlanner(ChunkPlannerBase):
    """Prefetch backend that stages peer KV through radixshmem ``pull_async``."""

    def __init__(self, engine, max_pinned_bytes: int, *, chunk_timeout_ms: Optional[int] = None):
        self._init_common(
            engine,
            max_pinned_bytes,
            apply_kv_replicas=False,
            executor_name="flexkv-radix-prefetch-query",
        )
        self.chunk_timeout_ms = int(
            chunk_timeout_ms if chunk_timeout_ms is not None else PREFETCH_TIMEOUT_MS
        )
        self._inflight: Dict[int, RadixPrefetchChunk] = {}
        from flexkv.cache.radix_shmem_engine import COMPONENT_MASK_FULL

        self._default_mask = COMPONENT_MASK_FULL

    def begin(self, tokens, namespace, client_id, options):
        if options.swa_aware:
            raise ValueError(
                "radixshmem chunked prefetch does not support SWA models; "
                "use wait_complete or disable FLEXKV_ENABLE_CHUNKED_PREFETCH"
            )
        tokens = self.validate_prefetch_tokens(tokens, options.candidate_start_token)
        context = RadixPrefetchContext(
            tokens.copy(),
            list(namespace) if namespace else None,
            client_id,
            component_mask=self._default_mask,
        )
        context.future = self.executor.submit(self._query, context)
        return context

    def _cpu(self):
        return self.cache.cpu_cache_engine

    def _query(
        self, context: RadixPrefetchContext
    ) -> Tuple[Optional[SequenceMeta], int, int]:
        if context.cancel_query.is_set():
            return None, 0, 0
        sequence = SequenceMeta(context.tokens, self.block_size, context.namespace)
        sequence.gen_hashes()
        cpu = self._cpu()
        local_hit, cluster_hit = self._probe_hits(cpu, sequence, context)
        return sequence, local_hit, cluster_hit

    def _probe_hits(
        self, cpu, sequence: SequenceMeta, context: RadixPrefetchContext
    ) -> Tuple[int, int]:
        hashes = sequence.block_hashes.view(np.uint64)
        local_q = cpu.client.query(
            hashes, mask=context.component_mask, local_only=True, lock=False, update_meta=False
        )
        local_hit = int(local_q.common_hit) if int(local_q.status) == 0 else 0
        if local_q.finalize is not None:
            local_q.finalize()
        if not getattr(cpu, "peer_enabled", False):
            return local_hit, local_hit
        if context.cancel_query.is_set():
            return local_hit, local_hit
        cluster_q = cpu.client.query(
            hashes, mask=context.component_mask, local_only=False, lock=False, update_meta=True
        )
        cluster_hit = int(cluster_q.common_hit) if int(cluster_q.status) == 0 else local_hit
        if cluster_q.finalize is not None:
            cluster_q.finalize()
        return local_hit, max(local_hit, cluster_hit)

    def resolve(self, context: RadixPrefetchContext) -> Optional[Tuple[int, int]]:
        if context.future is None or not context.future.done():
            return None
        context.sequence, local_blocks, cluster_blocks = context.future.result()
        if context.sequence is None:
            return 0, 0
        start = local_blocks * self.block_size
        target = max(start, cluster_blocks * self.block_size)
        if not self._pin_prefix(context, start):
            raise PrefetchCapacityExhausted("prefetch pinned-prefix capacity exhausted")
        return start, target

    def _pin_prefix(
        self, context: RadixPrefetchContext, end: int, reserved_to_release: int = 0
    ) -> bool:
        if end <= 0:
            context.reusable_end = 0
            return True
        cpu = self._cpu()
        prefix = SequenceMeta(
            context.tokens[:end], self.block_size, context.namespace
        )
        matched = cpu.match(prefix, component_mask=context.component_mask)
        ready = int(matched.num_matched_blocks) * self.block_size
        if ready < end:
            matched.release()
            # Peer publish may still be catching up; pin what is resident.
            end = ready
            if end <= 0:
                return True
            matched = cpu.match(
                SequenceMeta(context.tokens[:end], self.block_size, context.namespace),
                component_mask=context.component_mask,
            )
        blocks = int(matched.num_matched_blocks)
        amount = blocks * self.block_bytes
        delta = amount - context.pinned_bytes
        reserved = self.engine._prefetch.reserved_bytes - reserved_to_release
        if self.pinned_bytes + delta + reserved > self.max_pinned_bytes:
            matched.release()
            return False
        if context.match is not None:
            context.match.release()
        context.match = matched
        context.reusable_end = end
        self.pinned_bytes += delta
        context.pinned_bytes = amount
        return True

    def _prefetch_inflight_jobs(self) -> int:
        return sum(1 for c in self._inflight.values() if c.job is not None and not c.done)

    def reserve(self, context, begin, target, options, available_bytes):
        # Growing radix-tree prefixes must not overlap in flight: each pull is
        # hashes[:block_end], so concurrent chunks would race on the same path.
        if self._inflight:
            return None
        if self._prefetch_inflight_jobs() >= PREFETCH_MAX_INFLIGHT:
            raise PrefetchCapacityExhausted("radixshmem prefetch inflight limit reached")
        remaining_pin = (
            self.max_pinned_bytes
            - self.pinned_bytes
            - self.engine._prefetch.reserved_bytes
        )
        available_bytes = min(available_bytes, remaining_pin)
        blocks = min(
            options.chunk_max_blocks,
            available_bytes // self.block_bytes if self.block_bytes > 0 else 0,
            (target - begin) // self.block_size,
        )
        if blocks <= 0:
            return None
        end = begin + blocks * self.block_size
        block_begin = begin // self.block_size
        block_end = end // self.block_size
        graph = TransferOpGraph.create_empty_graph()
        chunk = RadixPrefetchChunk(
            begin=begin,
            end=end,
            nbytes=blocks * self.block_bytes,
            graph=graph,
            pending=context.sequence,
            block_begin=block_begin,
            block_end=block_end,
        )
        return chunk

    def submit_batch(self, chunks: List[RadixPrefetchChunk]):
        cpu = self._cpu()
        for chunk in chunks:
            if chunk.job is not None:
                continue
            sequence = chunk.pending
            if not isinstance(sequence, SequenceMeta):
                raise RuntimeError("radixshmem prefetch chunk missing sequence metadata")
            # Prefix pull to chunk.end: local_hit covers [0, begin) already in
            # the tree, so only the missing remote suffix is transferred.
            job = cpu.prefetch(
                sequence,
                component_mask=self._default_mask,
                query_end=chunk.block_end,
                timeout_ms=self.chunk_timeout_ms,
                hash_start=0,
                hash_end=chunk.block_end,
            )
            if job is None:
                chunk.done = True
                chunk.failed = True
                chunk.published_end = chunk.begin
            else:
                chunk.job = job
                if job.done():
                    chunk.done = True
                    chunk.published_end = self._clamp_published(chunk, job.published_hit)
            self._inflight[chunk.graph.graph_id] = chunk

    def _clamp_published(self, chunk: RadixPrefetchChunk, common_hit_blocks: int) -> int:
        """Map absolute common_hit (blocks from radix root) into this chunk's span."""
        abs_tokens = int(common_hit_blocks) * self.block_size
        return min(max(abs_tokens, chunk.begin), chunk.end)

    def poll_completions(self) -> List[_FakeCompletion]:
        done = []
        for graph_id, chunk in list(self._inflight.items()):
            if getattr(chunk, "_completion_emitted", False):
                continue
            job = chunk.job
            if job is not None and not job.done() and not chunk.done:
                continue
            failed = bool(chunk.failed)
            if job is not None and job.done() and not chunk.done:
                try:
                    result = job.wait(0)
                    # common_hit is absolute from the radix root for hashes[:block_end].
                    chunk.published_end = self._clamp_published(chunk, result.common_hit)
                    result.finalize()
                except Exception as exc:  # noqa: BLE001
                    flexkv_logger.warning(
                        "radixshmem prefetch chunk graph=%s failed: %s", graph_id, exc
                    )
                    failed = True
                    chunk.published_end = chunk.begin
            chunk.done = True
            chunk.failed = failed
            chunk._completion_emitted = True  # type: ignore[attr-defined]
            done.append(_FakeCompletion(graph_id, failed=failed))
        return done

    def complete(self, chunk, completion):
        if completion.op_id == -1:
            chunk.done = True
            chunk.failed = bool(getattr(completion, "failed", False))

    def commit(self, context, chunk: RadixPrefetchChunk):
        if chunk.resolved:
            raise RuntimeError("prefetch chunk resolved twice")
        chunk.resolved = True
        self._inflight.pop(chunk.graph.graph_id, None)
        if chunk.failed:
            return chunk.begin
        end = int(chunk.published_end or chunk.begin)
        if end < chunk.begin:
            end = chunk.begin
        if end > chunk.end:
            end = chunk.end
        if end > chunk.begin:
            if not self._pin_prefix(context, end, reserved_to_release=chunk.nbytes):
                return chunk.begin
        return end

    def discard(self, context, chunk: RadixPrefetchChunk):
        if chunk.resolved:
            return
        chunk.resolved = True
        self._inflight.pop(chunk.graph.graph_id, None)
        job = chunk.job
        if job is not None and not job.done() and not getattr(job, "cancelled", False):
            try:
                job.cancel()
            except Exception:  # noqa: BLE001
                pass

    def result_end(self, context: RadixPrefetchContext, committed: int) -> int:
        return min(committed, context.reusable_end) if context.reusable_end else committed

    def release(self, context: RadixPrefetchContext) -> None:
        if context.released:
            return
        context.released = True
        self.cancel_query(context)
        if context.match is not None:
            context.match.release()
            context.match = None
        self.pinned_bytes -= context.pinned_bytes
        context.pinned_bytes = 0
