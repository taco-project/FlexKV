import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Union, Dict, Any, BinaryIO

import numpy as np
import torch

from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import StorageHandle, AccessHandleType, KVCacheLayout, KVCacheLayoutType


class BaseStorageAllocator(ABC):
    @classmethod
    @abstractmethod
    def allocate(self,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_chunks: int = 1,
                 **kwargs: Any
                 ) -> StorageHandle:
        pass

    @classmethod
    @abstractmethod
    def free(cls, accessible_handle: StorageHandle) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_raw_data(cls,
        data: Any,
        layout: KVCacheLayout,
        dtype: torch.dtype,
        **kwargs: Any) -> StorageHandle:
        pass

class GPUAllocator(BaseStorageAllocator):
    @classmethod
    def allocate(cls,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_chunks: int = 1,
                 **kwargs: Any) -> StorageHandle:
        device_id = kwargs.get("device_id", torch.cuda.current_device())
        device = f"cuda:{device_id}"

        total_size = layout.get_total_elements()
        total_size_per_chunk = total_size // num_chunks
        physical_chunks = []
        for _ in range(num_chunks):
            physical_chunks.append(
                torch.empty(
                    size=(total_size_per_chunk,),
                    dtype=dtype,
                    device=device,
                )
            )
        return StorageHandle(
            handle_type=AccessHandleType.TENSOR,
            data=physical_chunks,
            kv_layout=layout,
            dtype=dtype,
            gpu_device_id=device_id,
        )

    @classmethod
    def free(cls, accessible_handle: StorageHandle) -> None:
        pass

    @classmethod
    def from_raw_data(cls,
        data: Union[List[TensorSharedHandle], List[torch.Tensor]],  # type: ignore
        layout: KVCacheLayout,
        dtype: torch.dtype,
        **kwargs: Any) -> StorageHandle:
        device_id = kwargs.get("device_id")
        if device_id is None:
            raise ValueError("device_id is required for GPU allocator")
        return StorageHandle(
            handle_type=AccessHandleType.TENSOR \
                if isinstance(data[0], torch.Tensor) else AccessHandleType.TENSOR_HANDLE,
            data=data,
            kv_layout=layout,
            dtype=dtype,
            gpu_device_id=device_id,
        )

class CPUAllocator(BaseStorageAllocator):
    @classmethod
    def allocate(cls,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_chunks: int = 1,
                 **kwargs: Any) -> StorageHandle:
        pin_memory = kwargs.get("pin_memory", True)
        total_size = layout.get_total_elements()
        total_size_per_chunk = total_size // num_chunks
        physical_chunks = []
        for _ in range(num_chunks):
            physical_chunks.append(
                torch.empty(
                    size=(total_size_per_chunk,),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
            )
        return StorageHandle(
            handle_type=AccessHandleType.TENSOR,
            data=physical_chunks,
            kv_layout=layout,
            dtype=dtype,
        )

    @classmethod
    def free(cls, accessible_handle: StorageHandle) -> None:
        pass

    @classmethod
    def from_raw_data(cls,
                      data: List[torch.Tensor],  # type: ignore
                      layout: KVCacheLayout,
                      dtype: torch.dtype,
                      **kwargs: Any) -> StorageHandle:
        return StorageHandle(
            handle_type=AccessHandleType.TENSOR,
            data=data,
            kv_layout=layout,
            dtype=dtype,
        )

class SSDAllocator(BaseStorageAllocator):
    @classmethod
    def allocate(cls,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_chunks: int = 1,
                 **kwargs: Any) -> StorageHandle:
        file_path = kwargs.get("file_path")
        if file_path is None:
            raise ValueError("file_path is required for SSD allocator")

        if isinstance(file_path, str):
            file_path = [file_path]
        total_size = layout.get_total_elements() * dtype.itemsize
        total_size_per_file = total_size // len(file_path)
        for path in file_path:
            with open(path, "wb+", buffering=0) as file:
                cls._init_file(file, total_size_per_file)

        return StorageHandle(
            handle_type=AccessHandleType.FILE,
            data=file_path,
            kv_layout=layout,
            dtype=dtype,
        )

    @classmethod
    def _init_file(cls, file: BinaryIO, total_size_per_file: int) -> None:
        try:
            os.truncate(file.fileno(), total_size_per_file)
        except OSError as e:
            raise RuntimeError(f"Failed to initialize file: {e}") from e
        file.flush()
        os.fsync(file.fileno())

    @classmethod
    def from_raw_data(cls,
                      data: Union[str, List[str]],  # type: ignore
                      layout: KVCacheLayout,
                      dtype: torch.dtype,
                      **kwargs: Any) -> StorageHandle:
        if isinstance(data, str):
            data = [data]
        return StorageHandle(
            handle_type=AccessHandleType.FILE,
            data=data,
            kv_layout=layout,
            dtype=dtype,
        )

class RemoteAllocator(BaseStorageAllocator):
    @classmethod
    def allocate(cls,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_chunks: int = 1,
                 **kwargs: Any) -> StorageHandle:
        file_path = kwargs.get("file_path")
        if file_path is None:
            raise ValueError("file_path is required for Remote allocator")
        remote_config_custom = kwargs.get("remote_config_custom")
        if remote_config_custom is None:
            raise ValueError("remote_config_custom is required for Remote allocator")
        if isinstance(file_path, str):
            file_path = [file_path]

        return StorageHandle(
            handle_type=AccessHandleType.FILE,
            data=file_path,
            kv_layout=layout,
            dtype=dtype,
            remote_config_custom = remote_config_custom,
        )

    @classmethod
    def free(cls, accessible_handle: StorageHandle) -> None:
        pass

    @classmethod
    def from_raw_data(cls,
                      data: Union[str, List[str]],  # type: ignore
                      layout: KVCacheLayout,
                      dtype: torch.dtype,
                      **kwargs: Any) -> StorageHandle:
        remote_config_custom = kwargs.get("remote_config_custom")
        if remote_config_custom is None:
            raise ValueError("remote_config_custom is required for Remote allocator")
        if isinstance(data, str):
            data = [data]

        return StorageHandle(
            handle_type=AccessHandleType.FILE,
            data=data,
            kv_layout=layout,
            dtype=dtype,
            remote_config_custom = remote_config_custom,
        )
