# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWA Pool Manager — ties together SWACacheEngine + SWAStorage + DMA.

This module provides `SWAPoolManager`, the unified coordinator that the
GlobalCacheEngine and KVTaskEngine use to manage SWA pages.  It handles:

  1. Index management (allocate / match / evict via SWACacheEngine)
  2. CPU buffer storage (SWAStorage — pinned memory)
  3. Async GPU↔CPU DMA transfers (SWACudaDMA — stream-based)
  4. Integration with main KV RadixTree (endpoint hash linkage)

The manager is created once per GlobalCacheEngine and exposes high-level
put/get/match operations that the rest of the system can call without
knowing about the internal SWA machinery.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from flexkv.swa.swa_cache_engine import SWACacheEngine, SWAMatchResult, HashType
from flexkv.swa.swa_storage import SWAStorage, SWAStorageConfig
from flexkv.common.config import SWAPoolConfig


@dataclass
class SWATransferRequest:
    """Descriptor for a pending SWA transfer (put or get)."""
    request_id: int
    endpoint_hash: HashType
    slot_id: int
    direction: str  # "put" or "get"
    gpu_tensor: Optional[object] = None  # torch.Tensor on GPU
    start_time: float = 0.0
    completed: bool = False


class SWAPoolManager:
    """Unified SWA Pool Manager.

    Manages the full lifecycle of SWA pages:
      - Allocate/match/evict index entries (SWACacheEngine)
      - Store/retrieve page data (SWAStorage)
      - Coordinate async GPU↔CPU transfers (when CUDA available)
      - Provide match results to the main KV matching pipeline

    Args:
        swa_config: SWAPoolConfig from CacheConfig.swa
        tokens_per_block: Block size from main KV cache (for hash consistency)
    """

    def __init__(self, swa_config: SWAPoolConfig, tokens_per_block: int = 16):
        if not swa_config.enabled:
            raise ValueError("SWAPoolManager created with disabled SWAPoolConfig")

        self._config = swa_config
        self._tokens_per_block = tokens_per_block

        # Core components
        self._engine = SWACacheEngine(
            num_slots=swa_config.num_slots,
            evict_ratio=swa_config.evict_ratio,
        )
        self._storage = SWAStorage(
            SWAStorageConfig.from_pool_config(swa_config),
            pin_memory=_TORCH_AVAILABLE and torch.cuda.is_available(),
        )

        # Transfer tracking
        self._pending_transfers: Dict[int, SWATransferRequest] = {}
        self._next_transfer_id = 0

        # Statistics
        self._stats = {
            "puts": 0,
            "gets": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    @property
    def config(self) -> SWAPoolConfig:
        return self._config

    @property
    def engine(self) -> SWACacheEngine:
        return self._engine

    @property
    def storage(self) -> SWAStorage:
        return self._storage

    @property
    def stats(self) -> Dict[str, int]:
        return self._stats.copy()

    # ------------------------------------------------------------------
    # High-level operations (synchronous CPU path)
    # ------------------------------------------------------------------

    def put(self,
            endpoint_hash: HashType,
            swa_data: np.ndarray,
            ) -> bool:
        """Store an SWA page to CPU cache.

        Called when a request is evicted from GPU and its SWA ring buffer
        needs to be preserved for later restoration.

        Args:
            endpoint_hash: Hash identifying the sequence endpoint.
            swa_data: Raw SWA page bytes (uint8 array, page_size_bytes).

        Returns:
            True if stored successfully, False if allocation failed.
        """
        slot = self._engine.allocate(endpoint_hash)
        if slot is None:
            return False

        self._storage.write_slot(slot, swa_data)
        self._engine.set_ready(endpoint_hash, True)
        self._stats["puts"] += 1
        return True

    def get(self, endpoint_hash: HashType) -> Optional[np.ndarray]:
        """Retrieve an SWA page from CPU cache.

        Called when a request is restored to GPU and needs its SWA ring
        buffer data back.

        Args:
            endpoint_hash: Hash identifying the sequence endpoint.

        Returns:
            SWA page data (uint8 array) or None if not cached.
        """
        result = self._engine.match(endpoint_hash)
        if not result.hit:
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1
        self._stats["gets"] += 1
        data = self._storage.read_slot(result.physical_block)
        if _TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            return data.numpy()
        return data

    def match(self, endpoint_hash: HashType) -> SWAMatchResult:
        """Check if an SWA page exists for the given endpoint.

        This is a lightweight check without data retrieval, used during
        the main KV prefix matching to determine SWA availability.

        Args:
            endpoint_hash: Hash identifying the sequence endpoint.

        Returns:
            SWAMatchResult with hit status.
        """
        result = self._engine.match(endpoint_hash)
        if result.hit:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1
        return result

    def remove(self, endpoint_hash: HashType) -> None:
        """Remove an SWA page (e.g., when request finishes normally)."""
        self._engine.remove(endpoint_hash)

    def lock(self, endpoint_hash: HashType) -> None:
        """Lock an SWA page to prevent eviction during transfer."""
        self._engine.lock(endpoint_hash)

    def unlock(self, endpoint_hash: HashType) -> None:
        """Unlock an SWA page after transfer completes."""
        self._engine.unlock(endpoint_hash)

    # ------------------------------------------------------------------
    # Async GPU↔CPU transfer operations
    # ------------------------------------------------------------------

    def put_async(self,
                  endpoint_hash: HashType,
                  gpu_swa_tensor,  # torch.Tensor on GPU
                  ) -> Tuple[int, bool]:
        """Initiate async GPU→CPU transfer for SWA page eviction.

        Args:
            endpoint_hash: Sequence endpoint hash.
            gpu_swa_tensor: GPU tensor containing SWA ring buffer data.

        Returns:
            (transfer_id, success) - transfer_id for polling, success=False if alloc failed.
        """
        slot = self._engine.allocate(endpoint_hash)
        if slot is None:
            return -1, False

        self._engine.lock(endpoint_hash)
        transfer_id = self._next_transfer_id
        self._next_transfer_id += 1

        if _TORCH_AVAILABLE and torch.cuda.is_available() and gpu_swa_tensor.is_cuda:
            # Async D2H copy
            cpu_view = self._storage.get_slot_view(slot)
            if isinstance(cpu_view, torch.Tensor):
                cpu_view.copy_(gpu_swa_tensor.view(-1).to(torch.uint8), non_blocking=True)
                # Record event for completion check
                event = torch.cuda.Event()
                event.record()
                self._pending_transfers[transfer_id] = SWATransferRequest(
                    request_id=transfer_id,
                    endpoint_hash=endpoint_hash,
                    slot_id=slot,
                    direction="put",
                    gpu_tensor=event,
                    start_time=time.time(),
                )
                return transfer_id, True
            else:
                # Fallback: sync copy
                data = gpu_swa_tensor.view(-1).to(torch.uint8).cpu().numpy()
                self._storage.write_slot(slot, data)
        else:
            # No CUDA: direct write (data is already on CPU)
            if _TORCH_AVAILABLE and isinstance(gpu_swa_tensor, torch.Tensor):
                data = gpu_swa_tensor.view(-1).to(torch.uint8).numpy()
            else:
                data = np.asarray(gpu_swa_tensor, dtype=np.uint8).ravel()
            self._storage.write_slot(slot, data)

        # Synchronous completion
        self._engine.set_ready(endpoint_hash, True)
        self._engine.unlock(endpoint_hash)
        self._stats["puts"] += 1

        self._pending_transfers[transfer_id] = SWATransferRequest(
            request_id=transfer_id,
            endpoint_hash=endpoint_hash,
            slot_id=slot,
            direction="put",
            start_time=time.time(),
            completed=True,
        )
        return transfer_id, True

    def get_async(self,
                  endpoint_hash: HashType,
                  gpu_swa_tensor,  # torch.Tensor on GPU (destination)
                  ) -> Tuple[int, bool]:
        """Initiate async CPU→GPU transfer for SWA page restoration.

        Args:
            endpoint_hash: Sequence endpoint hash.
            gpu_swa_tensor: GPU tensor to receive SWA ring buffer data.

        Returns:
            (transfer_id, hit) - hit=False means page not in cache.
        """
        result = self._engine.match(endpoint_hash)
        if not result.hit:
            self._stats["misses"] += 1
            return -1, False

        self._stats["hits"] += 1
        self._engine.lock(endpoint_hash)
        transfer_id = self._next_transfer_id
        self._next_transfer_id += 1

        slot = result.physical_block

        if _TORCH_AVAILABLE and torch.cuda.is_available() and gpu_swa_tensor.is_cuda:
            # Async H2D copy
            cpu_view = self._storage.get_slot_view(slot)
            if isinstance(cpu_view, torch.Tensor):
                gpu_swa_tensor.view(-1).copy_(cpu_view.to(torch.uint8), non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                self._pending_transfers[transfer_id] = SWATransferRequest(
                    request_id=transfer_id,
                    endpoint_hash=endpoint_hash,
                    slot_id=slot,
                    direction="get",
                    gpu_tensor=event,
                    start_time=time.time(),
                )
                self._stats["gets"] += 1
                return transfer_id, True
            else:
                # Fallback: sync copy
                data = self._storage.read_slot(slot)
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                gpu_swa_tensor.view(-1).copy_(data)
        else:
            # No CUDA path — just return data via sync
            pass

        self._engine.unlock(endpoint_hash)
        self._stats["gets"] += 1

        self._pending_transfers[transfer_id] = SWATransferRequest(
            request_id=transfer_id,
            endpoint_hash=endpoint_hash,
            slot_id=slot,
            direction="get",
            start_time=time.time(),
            completed=True,
        )
        return transfer_id, True

    def poll_transfers(self) -> List[int]:
        """Poll for completed async transfers.

        Returns:
            List of completed transfer_ids.
        """
        completed = []
        for tid, req in list(self._pending_transfers.items()):
            if req.completed:
                completed.append(tid)
                continue

            # Check CUDA event
            if _TORCH_AVAILABLE and req.gpu_tensor is not None:
                event = req.gpu_tensor
                if hasattr(event, 'query') and event.query():
                    req.completed = True
                    # Finalize put: mark ready and unlock
                    if req.direction == "put":
                        self._engine.set_ready(req.endpoint_hash, True)
                        self._stats["puts"] += 1
                    self._engine.unlock(req.endpoint_hash)
                    completed.append(tid)

        # Remove completed from pending
        for tid in completed:
            del self._pending_transfers[tid]

        return completed

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    def compute_endpoint_hash(self, token_ids: np.ndarray) -> HashType:
        """Compute endpoint hash for SWA lookup from token sequence.

        Uses a simple hash of the last tokens_per_block tokens, consistent
        with the main KV RadixTree's block hashing approach.

        Args:
            token_ids: Full token sequence.

        Returns:
            Integer hash suitable as SWA lookup key.
        """
        tpb = self._tokens_per_block
        if len(token_ids) == 0:
            return 0

        # Use hash of the trailing block (last tpb tokens)
        if len(token_ids) >= tpb:
            # Align to block boundary
            n_blocks = len(token_ids) // tpb
            last_block = token_ids[(n_blocks - 1) * tpb: n_blocks * tpb]
        else:
            last_block = token_ids

        # Simple polynomial hash (matches SequenceMeta.gen_hashes logic)
        h = 0
        for t in last_block:
            h = h * 31 + int(t) + 1
            h &= 0x7FFFFFFFFFFFFFFF  # Keep positive 63-bit
        return h

    def get_num_cached(self) -> int:
        """Number of SWA pages currently in CPU cache."""
        return self._engine.num_cached

    def get_num_free_slots(self) -> int:
        """Number of available SWA slots."""
        return self._engine.num_free_slots

    def reset(self) -> None:
        """Clear all SWA state."""
        self._engine.reset()
        self._pending_transfers.clear()
        self._stats = {k: 0 for k in self._stats}
