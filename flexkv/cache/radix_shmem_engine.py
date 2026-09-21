# SPDX-License-Identifier: Apache-2.0
# cython: boundscheck=True, wraparound=True
"""
The radixshmem CPU tier: one process's `shmradix.RadixClient` on the node's
radix-server (index shm + SlotStore + RDMA engine, see
`flexkv.server.shm_radix_bootstrap`). Used only by
`flexkv.cache.radix_shmem_planner.RadixShmemCacheEngine`.

Contracts:

1. Publish after transfer: `take()` -> transfer -> `insert()`. A block is
   servable by being in the tree, so `insert()` runs from the completion
   callback (`StagedRadixInsert`). Slots neither inserted nor recycled leak.
2. A match is a pin: `ShmRadixMatch.release()` must run on every path, or the
   prefix stays pinned for the life of the region.
3. `match()` is local only. Peer blocks arrive through `prefetch()`
   (`RadixClient.pull_async`), which publishes them into the local tree.

A region may carry an SWA pool next to FULL. `FULL|SWA` queries return the
joint hit plus the W-block window ending there; SWA slots are addressed by
`component=` and publish only after the Full path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType

if TYPE_CHECKING:  # these pull in the C++ extension; keep them off import time
    from flexkv.common.block import SequenceMeta
    from flexkv.common.config import SWAPoolConfig
    from flexkv.integration.dynamo.collector import KVEventCollector

try:
    import shmradix
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "shmradix is not installed; install it from the radixshmem repo "
        "(pip install -e radixshmem/python)") from e

COMPONENT_MASK_FULL = int(shmradix.COMPONENT_MASK_FULL)
COMPONENT_MASK_SWA = int(shmradix.COMPONENT_MASK_SWA)
COMPONENT_FULL = shmradix.ComponentType.FULL
COMPONENT_SWA = shmradix.ComponentType.SWA


def _empty_i64() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


@dataclass
class ShmRadixMatch:
    """One local prefix query, pinned until `release()`.

    `local_slots[i]` holds block `i` of the `num_matched_blocks`-block hit. On
    a `FULL|SWA` query the hit is the joint one and `swa_slots` cover
    `[swa_start, num_matched_blocks)`; both are empty otherwise.
    """
    num_matched_blocks: int = 0
    local_slots: np.ndarray = field(default_factory=_empty_i64)
    swa_start: int = 0
    swa_slots: np.ndarray = field(default_factory=_empty_i64)
    finalize: Optional[Callable[[], None]] = None

    def local_range(self, first: int, last: int) -> np.ndarray:
        """Slots of block range [first, last), clipped to the hit."""
        return self.local_slots[first:last]

    def release(self) -> None:
        """Drop the pin. Idempotent."""
        finalize, self.finalize = self.finalize, None
        if finalize is not None:
            finalize()


class StagedRadixInsert:
    """Staged slots waiting for their transfer. Exactly one of `publish()`
    (graph completed) or `abort()` (graph never ran) must run; both then drop
    `holds`, the match pin `publish` needs while it inserts.

    A Full+SWA PUT arms two instances, FULL first: SWA's insert refuses paths
    the Full tree does not reach yet.
    """

    def __init__(self,
                 engine: CacheEngineRadixShmem,
                 sequence_meta: SequenceMeta,
                 slots: np.ndarray,
                 path_end: int,
                 label: str,
                 holds: Sequence[Callable[[], None]] = (),
                 component: shmradix.ComponentType = COMPONENT_FULL) -> None:
        self._engine = engine
        self._sequence_meta = sequence_meta
        self._slots = slots
        self._path_end = path_end
        self._label = label
        self._holds = list(holds)
        self._component = component
        self._settled = False

    def publish(self) -> None:
        if self._settled:
            return
        self._settled = True
        try:
            self._engine.insert(self._sequence_meta, self._slots,
                                num_insert_blocks=self._path_end,
                                component=self._component)
        except Exception as e:
            flexkv_logger.error(
                f"radixshmem {self._label}: insert of {len(self._slots)} "
                f"staged slots failed: {e}; returning them to the mempool")
            self._engine.recycle(self._slots, component=self._component)
        finally:
            self._release_holds()

    def abort(self) -> None:
        if self._settled:
            return
        self._settled = True
        try:
            self._engine.recycle(self._slots, component=self._component)
        except Exception as e:
            flexkv_logger.error(
                f"radixshmem {self._label}: recycle of {len(self._slots)} "
                f"staged slots failed: {e}")
        finally:
            self._release_holds()

    def _release_holds(self) -> None:
        holds, self._holds = self._holds, []
        for release in holds:
            try:
                release()
            except Exception as e:  # keep releasing: a held ref pins for good
                flexkv_logger.error(
                    f"radixshmem {self._label}: ref release failed: {e}")


class CacheEngineRadixShmem:
    """One process's attachment to the node's radix-server. Several instances
    (one per DP scheduler process) share a server and operate concurrently."""

    def __init__(self,
                 shm_name: str,
                 *,
                 tokens_per_block: int,
                 num_total_blocks: int,
                 peer_enabled: bool = False,
                 swa_config: Optional[SWAPoolConfig] = None,
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector=None):
        """`shm_name` is the base index name (`radix_index_name`); the server
        must be running or starting. `peer_enabled` only takes effect on a
        clustered region. `num_total_blocks` is FlexKV's expectation; the
        region's capacity is authoritative."""
        from flexkv.server.shm_radix_bootstrap import attach_radix_client

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector
        cpu_swa = swa_config.for_cache_tier(DeviceType.CPU) if swa_config is not None else None
        self.swa_enabled = cpu_swa is not None and cpu_swa.num_slots > 0

        self._client = attach_radix_client(shm_name)
        self._tree = self._client  # index ops pass through the client
        self.shm_name = self._client.info.index_name  # node-suffixed when distributed
        self.is_distributed = bool(self._client.is_distributed())
        self.peer_enabled = bool(peer_enabled) and self.is_distributed
        if peer_enabled and not self.is_distributed:
            flexkv_logger.warning(
                f"radixshmem peer reuse is enabled for {self.shm_name} but the "
                f"attached region has world_size=1; prefetch stays local-only")

        region_tpb = int(self._client.block_size())
        if region_tpb != int(tokens_per_block):
            raise ValueError(
                f"radix-server {self.shm_name} has tokens_per_block={region_tpb}, "
                f"FlexKV is configured with {tokens_per_block}")
        self.tokens_per_block = int(tokens_per_block)
        capacity = int(self._client.mempool_total())
        if num_total_blocks > 0 and capacity != int(num_total_blocks):
            flexkv_logger.warning(
                f"radix-server {self.shm_name} has {capacity} FULL slots, FlexKV "
                f"expected {num_total_blocks}; the index is authoritative")
        self.num_total_blocks = capacity

    # ---------- attachment / lifecycle ----------

    @property
    def client(self):
        return self._client

    @property
    def num_free_blocks(self) -> int:
        return int(self._tree.mempool_free())

    def reset(self) -> None:
        """Clear the tree. Invalidates outstanding slot ids and matches."""
        self._tree.reset()

    def close(self) -> None:
        client, self._client, self._tree = self._client, None, None
        if client is not None:
            client.close()

    # ---------- queries ----------

    @staticmethod
    def _hashes(sequence_meta: SequenceMeta, query_end: Optional[int]) -> np.ndarray:
        sequence_meta.gen_hashes()
        hashes = sequence_meta.block_hashes.view(np.uint64)  # int64 -> uint64, same width
        return hashes if query_end is None else hashes[:query_end]

    def match(self,
              sequence_meta: SequenceMeta,
              *,
              component_mask: int = COMPONENT_MASK_FULL,
              query_end: Optional[int] = None) -> ShmRadixMatch:
        """Pinned local prefix match. `query_end` caps the queried path so the
        SWA window ends where the caller's restore will."""
        hashes = self._hashes(sequence_meta, query_end)
        # A refused query has zeroed fields and an unarmed finalize.
        qr = self._tree.query(hashes, mask=component_mask,
                              local_only=True, lock=True)
        common_hit = int(qr.common_hit)

        fragments = list(qr.full_fragments)  # local_only: at most one
        if len(fragments) > 1:
            self._finalize_and_raise(
                qr, f"radixshmem local query returned {len(fragments)} fragments; "
                    f"a local_only query yields at most one")
        local_slots = (np.asarray(fragments[0][2], dtype=np.int64)
                       if fragments else _empty_i64())
        if len(local_slots) != common_hit:
            self._finalize_and_raise(
                qr, f"radixshmem local query covers {len(local_slots)} blocks "
                    f"of a {common_hit}-block hit")

        swa_slots = np.asarray(qr.swa_slots, dtype=np.int64)
        swa_start = int(qr.swa_start) if len(swa_slots) > 0 else 0
        if len(swa_slots) > 0 and swa_start + len(swa_slots) != common_hit:
            self._finalize_and_raise(
                qr, f"radixshmem returned {len(swa_slots)} SWA slots at "
                    f"swa_start={swa_start} for a {common_hit}-block joint hit")

        return ShmRadixMatch(num_matched_blocks=common_hit,
                             local_slots=local_slots,
                             swa_start=swa_start,
                             swa_slots=swa_slots,
                             finalize=qr.finalize)

    @staticmethod
    def _finalize_and_raise(qr, message: str) -> None:
        if qr.finalize is not None:  # never propagate with the prefix pinned
            qr.finalize()
        raise RuntimeError(message)

    def prefetch(self,
                 sequence_meta: SequenceMeta,
                 *,
                 component_mask: int = COMPONENT_MASK_FULL,
                 query_end: Optional[int] = None,
                 timeout_ms: int = 30000) -> Any:
        """Start `pull_async` for the prefix; None when the region has no peers.

        The server pulls one peer's run into local slots and the client publishes
        it into the local tree on completion. `lock=False`: nothing stays pinned.
        `block=False`: a saturated client completes the job with the local hit.
        """
        if not self.peer_enabled:
            return None
        hashes = self._hashes(sequence_meta, query_end)
        job = self._client.pull_async(hashes, component_mask, lock=False,
                                      timeout_ms=int(timeout_ms), block=False)
        flexkv_logger.debug(
            f"radixshmem prefetch on {self.shm_name}: mask={component_mask:#x} "
            f"blocks={len(hashes)} local_hit={job.local_hit} "
            f"planned_hit={job.planned_hit} job={job.job_id}")
        return job

    # ---------- slots ----------

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int,
               component: shmradix.ComponentType = COMPONENT_FULL) -> None:
        """Attach transferred slots: `physical_block_ids[i]` is block
        `num_insert_blocks - len(physical_block_ids) + i`. Ownership passes to
        radixshmem (`auto_recycle=True`); do not recycle these slots again."""
        sequence_meta.gen_hashes()
        hashes = sequence_meta.block_hashes.view(np.uint64)

        slots = np.ascontiguousarray(physical_block_ids, dtype=np.int32)
        num_slots = len(slots)
        if num_slots == 0:
            return

        path_end = min(int(num_insert_blocks), len(hashes))
        start = path_end - num_slots
        if start < 0:  # caller bug; raising keeps slot ownership with the caller
            raise ValueError(
                f"radixshmem insert of {num_slots} slots overruns the "
                f"{path_end}-block path on {self.shm_name}")

        # SWA inserts are right-aligned by radixshmem and refuse a non-zero start.
        tree_start = start if component == COMPONENT_FULL else 0
        result = self._tree.insert(hashes[:path_end], slots, start=tree_start,
                                   auto_recycle=True, component=component)

        landed = num_slots - len(result.unused_slots)
        if result.error == shmradix.InsertError.FULL_PATH_MISSING:
            flexkv_logger.warning(
                f"radixshmem {component} insert on {self.shm_name}: full path "
                f"[0, {path_end}) was evicted before the window published "
                f"(slots were auto-recycled)")
        elif result.error != shmradix.InsertError.OK:
            flexkv_logger.warning(
                f"radixshmem {component} insert on {self.shm_name} returned "
                f"{result.error}: {landed}/{num_slots} blocks landed at "
                f"start={start} (unused slots were auto-recycled)")
        if landed <= 0:
            return

        if self.peer_enabled and component == COMPONENT_FULL:
            self._tree.flush()  # make the new blocks visible cluster-wide

        if (self.event_collector is not None and component == COMPONENT_FULL
                and result.error == shmradix.InsertError.OK):
            # Error-free, the only unused slots are a redundant prefix, so what
            # landed is the tail of the path.
            self.event_collector.publish_stored(
                block_hashes=sequence_meta.block_hashes[path_end - landed:path_end],
                block_size=self.tokens_per_block,
                medium="CPU")

    def take(self,
             num_required_blocks: int,
             component: shmradix.ComponentType = COMPONENT_FULL) -> np.ndarray:
        """Allocate up to `num_required_blocks` slots, evicting unpinned LRU
        blocks as needed; fewer come back when the pool cannot supply them
        (the SWA pool is all-or-none)."""
        slots = np.asarray(self._tree.allocate_slots(num_required_blocks,
                                                     component=component),
                           dtype=np.int64)
        if (self._metrics_collector is not None and len(slots) > 0
                and component == COMPONENT_FULL):  # SWA has its own pool
            self._metrics_collector.record_allocation("cpu", len(slots))
        return slots

    def recycle(self,
                physical_blocks: np.ndarray,
                component: shmradix.ComponentType = COMPONENT_FULL) -> None:
        if physical_blocks is None or len(physical_blocks) == 0:
            return
        self._tree.recycle_slots(np.ascontiguousarray(physical_blocks, dtype=np.int32),
                                 component=component)
