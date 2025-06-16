from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, List, Tuple, Union

import torch

from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.memory_handle import KVCacheTensorHandle, TensorSharedHandle
from flexkv.common.storage import StorageHandle, KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import DeviceType
from flexkv.storage.allocator import CPUAllocator, GPUAllocator, SSDAllocator, RemoteAllocator


class StorageEngine:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig):
        """Initialize storage engine"""
        self._storage_handles: Dict[Tuple[DeviceType, int], StorageHandle] = {}
        self._model_config = model_config
        self._cache_config = cache_config
        self._gpu_layout: Optional[KVCacheLayout] = None
        if self._cache_config.enable_cpu:
            if not self._cache_config.cpu_kv_layout.type == KVCacheLayoutType.LAYERWISE:
                raise ValueError("Only layerwise layout is supported for CPU")
            self._cpu_layout: Optional[KVCacheLayout] = self._cache_config.cpu_kv_layout
            self.allocate(
                device_type=DeviceType.CPU,
                layout=self._cpu_layout,
                dtype=self._model_config.dtype,
                pin_memory=self._cache_config.use_pinned_memory,
                num_chunks=self._model_config.num_layers
            )
        if self._cache_config.enable_ssd:
            if not self._cache_config.ssd_kv_layout.type == KVCacheLayoutType.LAYERWISE:
                raise ValueError("Only layerwise layout is supported for SSD")
            self._ssd_layout: Optional[KVCacheLayout] = self._cache_config.ssd_kv_layout
            self.allocate(
                device_type=DeviceType.SSD,
                layout=self._ssd_layout,
                dtype=self._model_config.dtype,
                file_path=self._cache_config.ssd_cache_path
            )
        if self._cache_config.enable_remote:
            if not self._cache_config.remote_kv_layout.type == KVCacheLayoutType.LAYERWISE:
                raise ValueError("Only layerwise layout is supported for remote")
            self._remote_layout: Optional[KVCacheLayout] = self._cache_config.remote_kv_layout
            self.allocate(
                device_type=DeviceType.REMOTE,
                layout=self._remote_layout,
                dtype=self._model_config.dtype,
                file_path=self._cache_config.remote_cache_path,
                remote_config_custom = self._cache_config.remote_config_custom
            )

    def register_gpu_blocks(self,
                            gpu_blocks: List[TensorSharedHandle],
                            device_id: int = 0,
                            dtype: torch.dtype = torch.float16) -> None:
        if self._gpu_layout is None:
            self._gpu_layout = self._cache_config.gpu_kv_layout
        self.allocate(
            device_type=DeviceType.GPU,
            layout=self._gpu_layout,
            dtype=dtype,
            device_id=device_id,
            raw_data=gpu_blocks
        )

    def allocate(self,
                 device_type: DeviceType,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_chunks: int = 1,
                 device_id: int = 0,
                 raw_data: Optional[Union[List[TensorSharedHandle], List[str], str]] = None,
                 **kwargs: Any) -> bool:
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
        if key in self._storage_handles:
            return False

        storage_handle: StorageHandle
        if device_type == DeviceType.CPU:
            pin_memory = kwargs.get('pin_memory', False)
            if raw_data is not None:
                assert isinstance(raw_data, list) and \
                    all(isinstance(x, torch.Tensor) for x in raw_data), \
                    "raw_data for CPUAllocator must be List[Tensor]"
                storage_handle = CPUAllocator.from_raw_data(
                    data=raw_data,  # type: ignore
                    layout=layout,
                    dtype=dtype,
                    num_chunks=num_chunks,
                    pin_memory=pin_memory
                )
            else:
                storage_handle = CPUAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    num_chunks=num_chunks,
                    pin_memory=pin_memory
                )
        elif device_type == DeviceType.GPU:
            if raw_data is not None:
                assert isinstance(raw_data, list) and \
                    (all(isinstance(x, TensorSharedHandle) for x in raw_data) or \
                     all(isinstance(x, torch.Tensor) for x in raw_data)), \
                    "raw_data for GPUAllocator must be List[TensorWrapper] or List[Tensor]"
                storage_handle = GPUAllocator.from_raw_data(
                    data=raw_data,  # type: ignore
                    layout=layout,
                    dtype=dtype,
                    num_chunks=num_chunks,
                    device_id=device_id
                )
            else:
                storage_handle = GPUAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    num_chunks=num_chunks,
                    device_id=device_id
                )
        elif device_type == DeviceType.SSD:
            file_path = kwargs.get('file_path')
            if raw_data is not None:
                assert isinstance(raw_data, str) or \
                    (isinstance(raw_data, list) and all(isinstance(x, str) for x in raw_data)), \
                    "raw_data for SSDAllocator must be str or List[str]"
                storage_handle = SSDAllocator.from_raw_data(
                    data=raw_data,  # type: ignore
                    layout=layout,
                    dtype=dtype
                )
            else:
                if not file_path:
                    raise ValueError("file_path is required for SSD allocator")
                storage_handle = SSDAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    file_path=file_path
                )
        elif device_type == DeviceType.REMOTE:
            file_path = kwargs.get('file_path')
            remote_config_custom = kwargs.get('remote_config_custom')
            if raw_data is not None:
                if (isinstance(raw_data, str) or \
                    (isinstance(raw_data, list) and all(isinstance(x, str) for x in raw_data))):
                    if not isinstance(remote_config_custom, dict):
                        raise TypeError("remote_config_custom for RemoteAllocator.from_raw_data must be dict[str, Any]")
                    storage_handle = RemoteAllocator.from_raw_data(
                        data=raw_data,  # type: ignore
                        layout=layout,
                        dtype=dtype,
                        remote_config_custom=remote_config_custom
                    )
                else:
                    raise TypeError("raw_data for RemoteAllocator must be str or List[str]")
            else:
                if not file_path:
                    raise ValueError("file_path is required for remote allocator")
                if not isinstance(remote_config_custom, dict):
                    raise TypeError("remote_config_custom for RemoteAllocator must be dict[str, Any]")
                storage_handle = RemoteAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    file_path=file_path,
                    remote_config_custom=remote_config_custom
                )
        else:
            raise ValueError(f"Unsupported device type: {device_type}")
        self._storage_handles[key] = storage_handle
        return True

    def get_storage_handle(self,
                           device_type: DeviceType,
                           device_id: int = 0) -> StorageHandle:
        """
        Get accessible handle for specified blocks

        Args:
            device_type: Type of the device to get handle from
            device_id: Device ID
        """
        key = (device_type, device_id)
        if key not in self._storage_handles:
            raise ValueError(f"Storage handle not found for device type: {device_type}, device id: {device_id}")

        storage_handle = self._storage_handles[key]
        return storage_handle

    def has_storage_handle(self,
                           device_type: DeviceType,
                           device_id: int = 0) -> bool:
        """Check if storage handle exists for given device type and id"""
        return (device_type, device_id) in self._storage_handles
