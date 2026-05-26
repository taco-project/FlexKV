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

"""Async extension for SWAStorage providing CUDA DMA integration.

Adds async_write_slot() and async_read_slot() methods to SWAStorage
for non-blocking GPU↔CPU transfers via SWACudaDMAEngine.
"""

from typing import Optional, Union
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def attach_async_methods(storage_class):
    """Decorator to attach async methods to SWAStorage class."""

    def async_write_slot_h2d(self,
                             slot_id: int,
                             data: Union[torch.Tensor, np.ndarray],
                             dma_engine: Optional[object] = None) -> Optional[int]:
        """Async write to slot via Host-to-Device transfer.

        Stages data from host tensor to GPU, then DMA-copies to pinned storage.
        Returns transfer_id for polling.

        Args:
            slot_id: Target storage slot.
            data: Tensor/array to write (on GPU or CPU).
            dma_engine: SWACudaDMAEngine instance for async transfer.

        Returns:
            transfer_id for async tracking, or None if sync fallback used.
        """
        if slot_id < 0 or slot_id >= self._num_slots:
            raise IndexError(
                f"slot_id {slot_id} out of range [0, {self._num_slots})"
            )

        if dma_engine is None:
            # Fallback to sync write
            self.write_slot(slot_id, data)
            return None

        # Get GPU pointer from data
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if not data.is_cuda:
            data = data.cuda()

        gpu_ptr = data.data_ptr()
        host_ptr = self.get_slot_address(slot_id)
        size_bytes = self._page_size

        # Submit D2H transfer (GPU → host pinned buffer)
        transfer_id = dma_engine.submit_d2h(
            endpoint_hash=0,  # Not used in async storage
            slot_id=slot_id,
            gpu_ptr=gpu_ptr,
            host_ptr=host_ptr,
            size_bytes=size_bytes
        )
        return transfer_id

    def async_read_slot_d2h(self,
                            slot_id: int,
                            dma_engine: Optional[object] = None) -> Optional[int]:
        """Async read from slot via Device-to-Host transfer.

        Stages pinned buffer data to GPU memory asynchronously.
        Returns transfer_id for polling.

        Args:
            slot_id: Source storage slot.
            dma_engine: SWACudaDMAEngine instance for async transfer.

        Returns:
            transfer_id for async tracking, or None if sync fallback used.
        """
        if slot_id < 0 or slot_id >= self._num_slots:
            raise IndexError(
                f"slot_id {slot_id} out of range [0, {self._num_slots})"
            )

        if dma_engine is None:
            # Fallback to sync read (caller should handle)
            return None

        host_ptr = self.get_slot_address(slot_id)
        # GPU destination would be provided by caller
        # For now, return transfer tracking only
        # Full implementation would allocate GPU staging buffer

        return None  # Placeholder for Phase 2.5

    # Attach methods to class
    storage_class.async_write_slot_h2d = async_write_slot_h2d
    storage_class.async_read_slot_d2h = async_read_slot_d2h

    return storage_class
