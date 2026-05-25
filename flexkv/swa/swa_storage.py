# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SWA page storage: CPU-pinned buffer for SWA page data.

Each SWA "slot" stores a full ring-buffer snapshot:
  window_size x bytes_per_token_per_layer x num_swa_layers bytes.

The buffer is a flat contiguous allocation indexed by physical_block ID
from the SWA CacheEngine's mempool.
"""

from dataclasses import dataclass

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from flexkv.common.config import SWAPoolConfig


@dataclass
class SWAStorageConfig:
    """Describes the geometry of the SWA storage buffer."""
    num_slots: int              # Max concurrent SWA pages
    window_size: int            # Tokens per SWA page (e.g. 128)
    bytes_per_token_per_layer: int  # e.g. 584 for DeepSeek V4
    num_swa_layers: int         # e.g. 61

    @property
    def page_size_bytes(self) -> int:
        """Size of one SWA page in bytes."""
        return self.window_size * self.bytes_per_token_per_layer * self.num_swa_layers

    @property
    def total_size_bytes(self) -> int:
        """Total buffer size in bytes."""
        return self.num_slots * self.page_size_bytes

    @classmethod
    def from_pool_config(cls, cfg: SWAPoolConfig) -> "SWAStorageConfig":
        """Create SWAStorageConfig from SWAPoolConfig."""
        return cls(
            num_slots=cfg.num_slots,
            window_size=cfg.window_size,
            bytes_per_token_per_layer=cfg.bytes_per_token_per_layer,
            num_swa_layers=cfg.num_swa_layers,
        )


class SWAStorage:
    """CPU storage buffer for SWA pages.

    Provides a flat pinned-memory buffer where each slot stores one
    complete SWA page.  Slots are addressed by physical_block ID from
    the SWA CacheEngine's mempool.

    Args:
        config: Storage geometry configuration.
        pin_memory: If True, allocate pinned CPU memory for DMA.
    """

    def __init__(self, config: SWAStorageConfig, pin_memory: bool = True):
        self._config = config
        self._page_size = config.page_size_bytes
        self._num_slots = config.num_slots

        if _TORCH_AVAILABLE:
            should_pin = pin_memory and torch.cuda.is_available()
            self._buffer = torch.zeros(
                self._num_slots, self._page_size,
                dtype=torch.uint8,
                pin_memory=should_pin,
            )
        else:
            # Fallback to numpy for environments without torch/CUDA
            self._buffer_np = np.zeros(
                (self._num_slots, self._page_size), dtype=np.uint8
            )
            self._buffer = None

    @property
    def buffer(self):
        """Raw buffer tensor [num_slots, page_size_bytes] or numpy array."""
        if self._buffer is not None:
            return self._buffer
        return self._buffer_np

    def get_slot_view(self, slot_id: int):
        """Get a view into the buffer for a specific slot.

        Args:
            slot_id: Physical block ID from SWA mempool.

        Returns:
            Tensor or numpy view of shape [page_size_bytes] (uint8).
        """
        if slot_id < 0 or slot_id >= self._num_slots:
            raise IndexError(
                f"slot_id {slot_id} out of range [0, {self._num_slots})"
            )
        if self._buffer is not None:
            return self._buffer[slot_id]
        return self._buffer_np[slot_id]

    def write_slot(self, slot_id: int, data) -> None:
        """Write data into a slot.

        Args:
            slot_id: Target slot.
            data: Tensor or numpy array of shape [page_size_bytes].
        """
        if slot_id < 0 or slot_id >= self._num_slots:
            raise IndexError(
                f"slot_id {slot_id} out of range [0, {self._num_slots})"
            )
        if self._buffer is not None:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            self._buffer[slot_id].copy_(data)
        else:
            if _TORCH_AVAILABLE and hasattr(data, 'numpy'):
                data = data.numpy()
            self._buffer_np[slot_id] = data

    def read_slot(self, slot_id: int):
        """Read data from a slot (returns a copy).

        Args:
            slot_id: Source slot.

        Returns:
            Copy of the slot data.
        """
        if slot_id < 0 or slot_id >= self._num_slots:
            raise IndexError(
                f"slot_id {slot_id} out of range [0, {self._num_slots})"
            )
        if self._buffer is not None:
            return self._buffer[slot_id].clone()
        return self._buffer_np[slot_id].copy()

    def get_slot_address(self, slot_id: int) -> int:
        """Get the byte offset of a slot in the buffer (for DMA ops)."""
        return slot_id * self._page_size

    @property
    def page_size_bytes(self) -> int:
        return self._page_size

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def config(self) -> SWAStorageConfig:
        return self._config

    @property
    def data_ptr(self) -> int:
        """Base data pointer of the buffer (for C++/CUDA interop)."""
        if self._buffer is not None:
            return self._buffer.data_ptr()
        return self._buffer_np.ctypes.data
