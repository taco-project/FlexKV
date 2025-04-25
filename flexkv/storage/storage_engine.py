from typing import Dict, Optional, List, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import torch

from .allocator import CPUAllocator, GPUAllocator, SSDAllocator, StorageAllocator
from ..common.storage import AccessibleHandle, AccessHandleType
from ..common.transfer import DeviceType
from ..common.config import ModelConfig, CacheConfig
class StorageEngine:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 all_gpu_blocks: List[List[torch.Tensor]] = None):
        """Initialize storage engine"""
        self._allocators: Dict[Tuple[DeviceType, int], StorageAllocator] = {}
        self._model_config = model_config
        self._cache_config = cache_config
        self._all_gpu_blocks = all_gpu_blocks
        self._block_size = (
            self._model_config.num_kv_heads * self._model_config.head_size * self._cache_config.tokens_per_block
        )
        if all_gpu_blocks:
            for i in range(len(all_gpu_blocks)):
                self.add_one_gpu_allocator(all_gpu_blocks[i], i)
        if self._cache_config.enable_cpu:
            self.create_allocator(
                device_type=DeviceType.CPU,
                tensor_shape=(self._model_config.num_layers, 2, self._cache_config.num_cpu_blocks, self._block_size),
                dtype=torch.float16,
                pin_memory=self._cache_config.use_pinned_memory
            )
        if self._cache_config.enable_ssd:
            self.create_allocator(
                device_type=DeviceType.SSD,
                tensor_shape=(self._model_config.num_layers, 2, self._cache_config.num_ssd_blocks, self._block_size),
                dtype=torch.float16,
                file_path=self._cache_config.ssd_cache_path
            )

    def add_one_gpu_allocator(self, gpu_blocks: List[torch.Tensor], device_id: int = 0):
        print(f"gpu {device_id} has {gpu_blocks[0].shape[1]} blocks")
        self.create_allocator(
            device_type=DeviceType.GPU, # NOTE can we use gpu_blocks[i][0].shape[1] as block_num?
            tensor_shape=(self._model_config.num_layers, 2, gpu_blocks[0].shape[1], self._block_size),
            dtype=torch.float16,
            device_id=device_id,
            raw_data=gpu_blocks
        )

    def create_allocator(self,
                        device_type: DeviceType,
                        tensor_shape: Tuple[int, ...],
                        dtype: torch.dtype,
                        device_id: int = 0,
                        raw_data: Union[List[torch.Tensor], str] = None,
                        **kwargs) -> bool:
        """
        Create and add an allocator for specified device

        Args:
            device_type: Type of the device (CPU, GPU, etc.)
            tensor_shape: Shape of tensors to be stored
            dtype: Data type of tensors
            device_id: Device ID (default 0)
            raw_data: Optional raw data to be used for initialization
            **kwargs: Additional arguments for specific allocator types
                     (e.g., pin_memory for CPU, file_path for Disk)

        Returns:
            bool: True if allocator created successfully, False otherwise
        """
        key = (device_type, device_id)
        if key in self._allocators:
            return False

        if device_type == DeviceType.CPU:
            pin_memory = kwargs.get('pin_memory', False)
            if raw_data is not None:
                allocator = CPUAllocator.from_raw_data(
                    data=raw_data,
                    shape=tensor_shape,
                    pin_memory=pin_memory
                )
            else:
                allocator = CPUAllocator(
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    pin_memory=pin_memory
                )
        elif device_type == DeviceType.GPU:
            if raw_data is not None:
                allocator = GPUAllocator.from_raw_data(
                    data=raw_data,
                    shape=tensor_shape,
                    device_id=device_id
                )
            else:
                allocator = GPUAllocator(
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    device_id=device_id
            )
        elif device_type == DeviceType.SSD:
            file_path = kwargs.get('file_path')
            if raw_data is not None:
                allocator = SSDAllocator.from_raw_data(
                    shape=tensor_shape,
                    dtype=dtype,
                    file_path=raw_data
                )
            else:
                if not file_path:
                    raise ValueError("file_path is required for SSD allocator")
                allocator = SSDAllocator(
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    file_path=file_path
                )
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

        self._allocators[key] = allocator
        return True


    def get_allocator_handle(self,
                           device_type: DeviceType,
                           device_id: int = 0) -> Optional[AccessibleHandle]:
        """
        Get accessible handle for specified blocks

        Args:
            device_type: Type of the device to get handle from
            device_id: Device ID
        """
        key = (device_type, device_id)
        if key not in self._allocators:
            return None

        allocator = self._allocators[key]

        return allocator.get_accessible_handle()

    def has_allocator(self,
                     device_type: DeviceType,
                     device_id: int = 0) -> bool:
        """Check if allocator exists for given device type and id"""
        return (device_type, device_id) in self._allocators
