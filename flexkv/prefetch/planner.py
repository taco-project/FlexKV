"""Mooncake chunk planner using existing transfer graphs and deferred publication.

Only detached CPU staging is allocated, lazily per window. Full sequence hashes
are constructed once. Completion and publication are deliberately separate.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, field, replace
import threading
from typing import Any, Optional, Tuple

import numpy as np

from flexkv.cache.cache_engine import (
    DeferredCacheInsert,
    DeferredPublishResult,
    MooncakeLoadResult,
)
from flexkv.common.block import SequenceMeta
from flexkv.common.transfer import DeviceType, TransferOp, TransferOpGraph, TransferType
from flexkv.prefetch.types import PrefetchCapacityExhausted


@dataclass
class PrefetchContext:
    tokens: np.ndarray
    namespace: Optional[list[str]]
    client_id: int
    future: Optional[Future] = None
    sequence: Optional[SequenceMeta] = None
    node: Any = None
    swa_node: Any = None
    swa_aware: bool = False
    checkpoints: Tuple[int, ...] = ()
    reusable_end: int = 0
    pinned_bytes: int = 0
    released: bool = False
    cancel_query: threading.Event = field(default_factory=threading.Event)


@dataclass
class PrefetchChunk:
    begin: int
    end: int
    nbytes: int
    graph: TransferOpGraph
    pending: DeferredCacheInsert
    op_id: int
    done: bool = False
    failed: bool = False
    completion: Any = None
    resolved: bool = False
    swa_op_id: Optional[int] = None
    swa_completion: Any = None


class MooncakeChunkPlanner:
    def __init__(self, engine, max_pinned_bytes):
        self.engine = engine
        self.cache = engine.cache_engine
        self.block_size = engine.cache_config.tokens_per_block
        from flexkv.common.config import (
            block_size_in_bytes_for_cache,
            RankInfo,
            GLOBAL_CONFIG_FROM_ENV,
        )

        self.block_bytes = block_size_in_bytes_for_cache(
            engine.model_config, engine.cache_config, RankInfo(engine.model_config)
        )
        # Full KV and its SWA/state sidecar use the same rank replication mode.
        replicas = 1
        if (
            engine.model_config.num_kv_heads == 1
            and GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode == "all_write"
        ):
            replicas = engine.model_config.effective_tp_size_per_node
        self.block_bytes *= replicas
        swa = engine.cache_config.swa
        self.swa_slot_bytes = 0
        if swa is not None and swa.enabled:
            if swa.multi_group and swa.snapshot_bytes is None:
                raise ValueError(
                    "multi-group SWA prefetch requires registered snapshot geometry"
                )
            self.swa_slot_bytes = replicas * (
                swa.snapshot_bytes
                or (
                    swa.num_swa_layers * self.block_size * swa.bytes_per_token_per_layer
                )
            )
        self.max_pinned_bytes = max_pinned_bytes
        self.pinned_bytes = 0
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="flexkv-prefetch-query"
        )
        self._query_client = None

    def begin(self, tokens, namespace, client_id, options):
        if options.swa_aware and not self.cache.swa_op_constructor.enabled:
            raise ValueError("SWA prefetch requires enabled SWA storage and transfer")
        tokens = np.asarray(tokens)
        if tokens.ndim != 1 or tokens.dtype != np.int64:
            raise ValueError("prefetch token_ids must be a one-dimensional int64 array")
        if options.candidate_start_token > len(tokens):
            raise ValueError("candidate_start_token exceeds request length")
        context = PrefetchContext(
            tokens.copy(),
            list(namespace) if namespace else None,
            client_id,
            swa_aware=options.swa_aware,
        )
        context.future = self.executor.submit(self._query, context)
        return context

    def _query(self, context: PrefetchContext) -> Tuple[Optional[SequenceMeta], int]:
        # Hashing and network queries occur without the radix lock. The query
        # client belongs exclusively to this executor, not foreground GET/PUT.
        from flexkv.external.mooncake_store_keys import PoolKind
        from flexkv.external.mooncake_store_utils import (
            MooncakeStoreClient,
            MooncakeStoreConfig,
        )

        if context.cancel_query.is_set():
            return None, 0
        sequence = SequenceMeta(context.tokens, self.block_size, context.namespace)
        if self._query_client is None:
            config = MooncakeStoreConfig.from_file(
                self.engine.cache_config, override_global_segment_size=0
            )
            self._query_client = MooncakeStoreClient(config, query_only=True)
        remote = self.cache.remote_cache_engine
        matched = 0
        checkpoints = []
        for begin in range(0, sequence.num_blocks, 256):
            if context.cancel_query.is_set():
                break
            hashes = sequence.block_hashes[begin : begin + 256]
            keys = [remote._build_pool_key(str(h), PoolKind.KV) for h in hashes]
            if context.swa_aware:
                swa_keys = [
                    remote._build_pool_key(str(h), PoolKind.SWA) for h in hashes
                ]
                exists = self._query_client.batch_exists_impl(keys + swa_keys)
                if len(exists) != 2 * len(keys):
                    raise RuntimeError("invalid Mooncake checkpoint metadata result")
                count = next(
                    (i for i, found in enumerate(exists[: len(keys)]) if found != 1),
                    len(keys),
                )
                checkpoints.extend(
                    (begin + i + 1) * self.block_size
                    for i, found in enumerate(exists[len(keys) : len(keys) + count])
                    if found == 1
                )
            else:
                count = int(self._query_client.batch_exists(keys))
            if not 0 <= count <= len(keys):
                raise RuntimeError("invalid Mooncake metadata result")
            matched += count
            if count != len(keys):
                break
        context.checkpoints = tuple(checkpoints)
        if context.swa_aware:
            matched = checkpoints[-1] // self.block_size if checkpoints else 0
        return sequence, matched

    def resolve(self, context: PrefetchContext) -> Optional[Tuple[int, int]]:
        if not context.future.done():
            return None
        context.sequence, matched = context.future.result()
        if context.sequence is None:
            return 0, 0
        with self.cache._cache_tree_lock:
            local = self.cache.cpu_cache_engine.match(context.sequence)
            ready = int(local.num_ready_matched_blocks)
            start = (
                min(ready, int(local.swa_hit_blocks)) if context.swa_aware else ready
            )
            start *= self.block_size
            if not self._pin_prefix(context, start):
                raise PrefetchCapacityExhausted(
                    "prefetch pinned-prefix capacity exhausted"
                )
        return start, max(start, matched * self.block_size)

    def _pin_prefix(
        self, context: PrefetchContext, end: int, reserved_to_release: int = 0
    ) -> bool:
        cpu = self.cache.cpu_cache_engine
        prefix = copy(context.sequence)
        prefix.token_ids = prefix.token_ids[:end]
        prefix.block_hashes = prefix.block_hashes[: end // self.block_size]
        local = cpu.match(prefix)
        if int(local.num_ready_matched_blocks) * self.block_size != end:
            raise RuntimeError("prefetch published prefix is not resident")
        node = local.last_ready_node
        # Locks protect whole nodes/ancestors, including a ready suffix beyond
        # our requested boundary. Charge that actual span, without splitting
        # existing nodes just for accounting.
        blocks, ancestor = 0, node
        while ancestor is not None:
            blocks += ancestor.size()
            ancestor = ancestor.parent
        swa_node = local.last_swa_node if context.swa_aware else None
        amount = blocks * self.block_bytes + (
            self.swa_slot_bytes if swa_node is not None else 0
        )
        delta = amount - context.pinned_bytes
        reserved = self.engine._prefetch.reserved_bytes - reserved_to_release
        if self.pinned_bytes + delta + reserved > self.max_pinned_bytes:
            return False
        if node is not None:
            cpu.lock_node(node)
        if context.node is not None:
            cpu.unlock(context.node)
        # Full-KV locks and snapshot locks are independent: SWA has its own LRU.
        if swa_node is not None:
            cpu._pin_swa_node(swa_node)
        if context.swa_node is not None:
            context.swa_node.dec_swa_lock_ref()
            cpu.index.unlock(context.swa_node)
        context.node = node
        context.swa_node = swa_node
        context.reusable_end = (
            int(local.swa_hit_blocks) * self.block_size if context.swa_aware else end
        )
        self.pinned_bytes += delta
        context.pinned_bytes = amount
        return True

    def reserve(self, context, begin, target, options, available_bytes):
        block_bytes = self.block_bytes
        # Reserve enough pin budget for *all* detached buffers too: multiple
        # inflight chunks must not collectively exceed the future pin budget.
        remaining_pin = (
            self.max_pinned_bytes
            - self.pinned_bytes
            - self.engine._prefetch.reserved_bytes
        )
        available_bytes = min(available_bytes, remaining_pin)
        # Leave room for a complete SWA/state snapshot in a checkpoint chunk.
        snapshot_bytes = self.swa_slot_bytes if context.swa_aware else 0
        available_bytes -= snapshot_bytes
        blocks = min(
            options.chunk_max_blocks,
            available_bytes // block_bytes,
            (target - begin) // self.block_size,
        )
        if blocks <= 0:
            return None
        end = begin + blocks * self.block_size
        checkpoints = [point for point in context.checkpoints if begin < point <= end]
        if checkpoints:
            end = checkpoints[-1]
            blocks = (end - begin) // self.block_size
        needs_snapshot = context.swa_aware and end in context.checkpoints
        with self.cache._cache_tree_lock:
            cpu = self.cache.cpu_cache_engine
            try:
                physical = cpu.take(blocks, protected_node=context.node, strict=True)
            except RuntimeError:
                return None
            swa_slot = -1
            if needs_snapshot:
                swa_slot = cpu._alloc_swa_slot(protected_node=context.node)
                if swa_slot < 0:
                    cpu.recycle(physical)
                    return None
            try:
                graph = TransferOpGraph()
                start_block, end_block = (
                    begin // self.block_size,
                    end // self.block_size,
                )
                op = TransferOp(
                    graph_id=graph.graph_id,
                    transfer_type=TransferType.REMOTE2H,
                    src_block_ids=np.arange(start_block, end_block, dtype=np.int64),
                    dst_block_ids=physical,
                    dp_client_id=context.client_id,
                    mooncake_store_block_hashes=context.sequence.block_hashes[
                        start_block:end_block
                    ],
                )
                graph.add_transfer_op(op)
                pending = DeferredCacheInsert(
                    device_type=DeviceType.CPU,
                    sequence_meta=context.sequence,
                    physical_blocks=physical,
                    staged_start_block=start_block,
                    remote_start_block=start_block,
                    requested_end_block=end_block,
                    load_result=MooncakeLoadResult(),
                    publish_result=DeferredPublishResult(),
                )
                swa_op_id = None
                if needs_snapshot:
                    swa_op_id = self.cache.swa_op_constructor.build_swa_op(
                        graph,
                        TransferType.REMOTE2H,
                        src_slot_ids=np.array([0], dtype=np.int64),
                        dst_slot_ids=np.array([swa_slot], dtype=np.int64),
                        dp_client_id=context.client_id,
                        mooncake_tail_hashes=[
                            str(context.sequence.block_hashes[end_block - 1])
                        ],
                    )
                    pending = replace(
                        pending,
                        swa_slot=swa_slot,
                        swa_anchor_block=end_block - 1,
                        swa_load_result=MooncakeLoadResult(),
                    )
                return PrefetchChunk(
                    begin,
                    end,
                    blocks * block_bytes + (snapshot_bytes if needs_snapshot else 0),
                    graph,
                    pending,
                    op.op_id,
                    swa_op_id=swa_op_id,
                )
            except BaseException:
                cpu.recycle(physical)
                if swa_slot >= 0:
                    cpu._free_swa_slot(swa_slot)
                raise

    def submit_batch(self, chunks):
        graphs = [chunk.graph for chunk in chunks]
        for handle in self.engine.transfer_handles:
            handle.submit_batch(graphs)

    @staticmethod
    def complete(chunk, completion):
        if completion.op_id == -1:
            chunk.done = True
            chunk.failed = completion.failed
        elif completion.op_id == chunk.op_id:
            chunk.completion = completion
        elif completion.op_id == chunk.swa_op_id:
            chunk.swa_completion = completion

    def commit(self, context, chunk):
        if chunk.resolved:
            raise RuntimeError("prefetch chunk resolved twice")
        completion = chunk.completion
        expected = (chunk.end - chunk.begin) // self.block_size
        bitmap = None if completion is None else completion.block_results
        if chunk.failed or bitmap is None or len(bitmap) != expected:
            self.discard(context, chunk)
            return chunk.begin
        chunk.pending.load_result.record(completion)
        swa_completion = chunk.swa_completion
        # One peer op restores exactly one snapshot slot.
        if (
            swa_completion is not None
            and swa_completion.block_results is not None
            and len(swa_completion.block_results) == 1
        ):
            chunk.pending.swa_load_result.record(swa_completion)
        with self.cache._cache_tree_lock:
            self.cache._commit_deferred_insert(chunk.pending)
            chunk.resolved = True
            result = chunk.pending.publish_result
            published = 0 if result.failed else (result.published_remote_blocks or 0)
            end = chunk.begin + published * self.block_size
            if not self._pin_prefix(context, end, reserved_to_release=chunk.nbytes):
                # Concurrent publication may extend the node we would lock.
                # The new bytes stay cached, but only the existing lease is
                # guaranteed for this result; never overrun the pin budget.
                return chunk.begin
            return end

    def discard(self, context, chunk):
        if chunk.resolved:
            return
        with self.cache._cache_tree_lock:
            self.cache.cpu_cache_engine.recycle(chunk.pending.physical_blocks)
            if chunk.pending.swa_slot >= 0:
                self.cache.cpu_cache_engine._free_swa_slot(chunk.pending.swa_slot)
            chunk.resolved = True

    @staticmethod
    def result_end(context: PrefetchContext, committed: int) -> int:
        """Only a resident Full+SWA boundary can resume an SWA request."""
        return min(committed, context.reusable_end) if context.swa_aware else committed

    def release(self, context: PrefetchContext) -> None:
        if context.released:
            return
        context.released = True
        context.cancel_query.set()
        context.future.cancel()
        with self.cache._cache_tree_lock:
            if context.node is not None:
                self.cache.cpu_cache_engine.unlock(context.node)
                context.node = None
            if context.swa_node is not None:
                context.swa_node.dec_swa_lock_ref()
                self.cache.cpu_cache_engine.index.unlock(context.swa_node)
                context.swa_node = None
            self.pinned_bytes -= context.pinned_bytes
            context.pinned_bytes = 0

    @staticmethod
    def cancel_query(context: PrefetchContext) -> None:
        context.cancel_query.set()
        context.future.cancel()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
