"""SWA Host Pool — CPU-side pinned memory for SWA page storage.

SWA is managed at PAGE granularity: each slot stores exactly one swa_page of
window KV = swa_page_size tokens x num_swa_layers layers x bytes_per_token_per_layer
(``window_size`` in SWAPoolConfig carries the physical swa_page_size). All SWA
IO addresses a whole slot (= one page) at a time. Slot allocation uses a simple
free-list (stack). When the pool is full, the caller (cache engine) triggers SWA-LRU
eviction before retrying.
"""
import time
from typing import Optional, Union

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from flexkv.common.config import SWAPoolConfig


class SWAHostPool:
    """Fixed-size CPU buffer pool for SWA page snapshots."""

    def __init__(self, config: SWAPoolConfig):
        self._config = config
        self._slot_size = config.slot_size_bytes
        self._num_slots = config.num_slots

        # Allocate buffer
        if _TORCH_AVAILABLE:
            use_pin = config.pin_memory and torch.cuda.is_available()
            self._buffer = torch.zeros(
                (self._num_slots, self._slot_size),
                dtype=torch.uint8,
                pin_memory=use_pin,
            )
            self._use_torch = True
        else:
            self._buffer = np.zeros((self._num_slots, self._slot_size), dtype=np.uint8)
            self._use_torch = False

        # Free-list (stack-based)
        self._free_slots = list(range(self._num_slots - 1, -1, -1))

    # --- Allocation --------------------------------------------------------

    def allocate(self) -> Optional[int]:
        """Allocate a slot. Returns slot_id or None if pool is full."""
        if not self._free_slots:
            return None
        return self._free_slots.pop()

    def free(self, slot_id: int) -> None:
        """Return a slot to the free list."""
        self._free_slots.append(slot_id)

    def reset(self) -> None:
        """Return every slot to the free list (all SWA state dropped).

        Called when the owning radix tree is reset: the tree bulk-deletes all
        nodes without buffering their slots, so the pool must be re-armed as
        fully free to avoid permanently leaking those slots.
        """
        self._free_slots = list(range(self._num_slots - 1, -1, -1))

    # --- Data Access -------------------------------------------------------

    def write(self, slot_id: int, data: Union['torch.Tensor', np.ndarray, bytes]) -> None:
        """Write SWA data into a slot."""
        if isinstance(data, bytes):
            # np.frombuffer on bytes yields a read-only array; copy so the
            # resulting tensor is writable and torch.from_numpy stays quiet.
            arr = np.frombuffer(data, dtype=np.uint8).copy()
        elif _TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            arr = data.detach().cpu().view(-1).to(torch.uint8)
        else:
            arr = np.asarray(data, dtype=np.uint8).ravel()

        n = min(len(arr), self._slot_size)
        if self._use_torch:
            if isinstance(arr, np.ndarray):
                self._buffer[slot_id, :n] = torch.from_numpy(arr[:n])
            else:
                self._buffer[slot_id, :n] = arr[:n]
        else:
            if _TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
                self._buffer[slot_id, :n] = arr[:n].numpy()
            else:
                self._buffer[slot_id, :n] = arr[:n]

    def read(self, slot_id: int) -> Union['torch.Tensor', np.ndarray]:
        """Read SWA data from a slot (returns a view, zero-copy)."""
        return self._buffer[slot_id]

    def read_copy(self, slot_id: int) -> Union['torch.Tensor', np.ndarray]:
        """Read SWA data from a slot (returns a copy)."""
        if self._use_torch:
            return self._buffer[slot_id].clone()
        else:
            return self._buffer[slot_id].copy()

    # --- Properties --------------------------------------------------------

    @property
    def buffer(self):
        """The backing slot buffer ``[num_slots, slot_size_bytes]`` (uint8).

        Pinned torch tensor when CUDA is available, else a numpy array. The SWA
        transfer worker (data plane) shares this and addresses bytes by slot row.
        """
        return self._buffer

    @property
    def num_free(self) -> int:
        return len(self._free_slots)

    @property
    def num_used(self) -> int:
        return self._num_slots - self.num_free

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def slot_size_bytes(self) -> int:
        return self._slot_size

    @property
    def config(self) -> SWAPoolConfig:
        return self._config
