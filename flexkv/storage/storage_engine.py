from typing import Dict, Optional, List, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import torch

from .allocator import CPUAllocator, GPUAllocator, SSDAllocator, StorageAllocator
from ..common.storage import AccessibleHandle, KVCacheLayout, KVCacheLayoutType
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
        self._gpu_layout = None
        if all_gpu_blocks:
            for i in range(len(all_gpu_blocks)):
                self.add_gpu_blocks(all_gpu_blocks[i], i)
        if self._cache_config.enable_cpu:
            if not self._cache_config.cpu_kv_layout == "layerwise":
                raise ValueError("Only layerwise layout is supported for CPU")
            self._cpu_layout = KVCacheLayout(
                type=KVCacheLayoutType(self._cache_config.cpu_kv_layout),
                num_layer=self._model_config.num_layers,
                num_block=self._cache_config.num_cpu_blocks,
                tokens_per_block=self._cache_config.tokens_per_block,
                num_head=self._model_config.num_kv_heads,
                head_size=self._model_config.head_size,
                is_mla=False
            )
            self.allocate(
                device_type=DeviceType.CPU,
                layout=self._cpu_layout,
                dtype=torch.float16,
                pin_memory=self._cache_config.use_pinned_memory,
                num_chunks=self._model_config.num_layers
            )
        if self._cache_config.enable_ssd:
            if not self._cache_config.ssd_kv_layout == "layerwise":
                raise ValueError("Only layerwise layout is supported for SSD")
            self._ssd_layout = KVCacheLayout(
                type=KVCacheLayoutType(self._cache_config.ssd_kv_layout),
                num_layer=self._model_config.num_layers,
                num_block=self._cache_config.num_ssd_blocks,
                tokens_per_block=self._cache_config.tokens_per_block,
                num_head=self._model_config.num_kv_heads,
                head_size=self._model_config.head_size,
                is_mla=False
            )
            self.allocate(
                device_type=DeviceType.SSD,
                layout=self._ssd_layout,
                dtype=torch.float16,
                file_path=self._cache_config.ssd_cache_path
            )

    def add_gpu_blocks(self, gpu_blocks: List[torch.Tensor], device_id: int = 0):
        if not self._cache_config.gpu_kv_layout == "layerwise":
            raise ValueError("Only layerwise layout is supported for GPU")
        self.num_gpu_blocks = gpu_blocks[0].shape[1]
        if self._gpu_layout is None:
            self._gpu_layout = KVCacheLayout(
                type=KVCacheLayoutType(self._cache_config.gpu_kv_layout),
                num_layer=self._model_config.num_layers,
                num_block=self.num_gpu_blocks,
                tokens_per_block=self._cache_config.tokens_per_block,
                num_head=self._model_config.num_kv_heads,
                head_size=self._model_config.head_size,
                is_mla=False
            )
        self.allocate(
            device_type=DeviceType.GPU,
            layout=self._gpu_layout,
            dtype=torch.float16,
            device_id=device_id,
            raw_data=gpu_blocks
        )

    def allocate(self,
                 device_type: DeviceType,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_chunks: int = 1,
                 device_id: int = 0,
                 raw_data: Union[List[torch.Tensor], str] = None,
                 **kwargs) -> bool:
        """
        Create and add an allocator for specified device

        Args:
            device_type: Type of the device (CPU, GPU, etc.)
            layout: Layout of kv cache
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
                    layout=layout,
                )
            else:
                allocator = CPUAllocator(
                    layout=layout,
                    dtype=dtype,
                    pin_memory=pin_memory,
                    num_chunks=num_chunks
                )
        elif device_type == DeviceType.GPU:
            if raw_data is not None:
                allocator = GPUAllocator.from_raw_data(
                    data=raw_data,
                    layout=layout,
                    device_id=device_id
                )
            else:
                allocator = GPUAllocator(
                    layout=layout,
                    dtype=dtype,
                    device_id=device_id,
                    num_chunks=num_chunks
                )
        elif device_type == DeviceType.SSD:
            file_path = kwargs.get('file_path')
            if raw_data is not None:
                allocator = SSDAllocator.from_raw_data(
                    layout=layout,
                    dtype=dtype,
                    file_path=raw_data
                )
            else:
                if not file_path:
                    raise ValueError("file_path is required for SSD allocator")
                allocator = SSDAllocator(
                    layout=layout,
                    dtype=dtype,
                    file_path=file_path
                )
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

        allocator.allocate()
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
