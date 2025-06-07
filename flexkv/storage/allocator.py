from abc import ABC, abstractmethod
from flexkv.common.storage import AccessibleHandle, AccessHandleType, KVCacheLayout, KVCacheLayoutType
import torch
from typing import Tuple, Optional, List, Union, Dict, Any
import numpy as np
import os

class StorageAllocator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def allocate(self) -> int:
        pass

    @abstractmethod
    def free(self):
        pass

    @abstractmethod
    def get_accessible_handle(self) -> AccessibleHandle:
        pass

    @classmethod
    @abstractmethod
    def from_raw_data(cls) -> 'StorageAllocator':
        pass

class GPUAllocator(StorageAllocator):
    def __init__(
        self,
        layout: KVCacheLayout,
        dtype: torch.dtype,
        device_id: int,
        num_chunks: int = 1
    ):
        self.layout = layout
        self.dtype = dtype
        self.device_id = device_id
        self.num_chunks = num_chunks

        self.physical_chunks = []
        self._need_free = True

    @property
    def total_size(self) -> int:
        return self.layout.get_total_elements()

    @property
    def total_size_in_bytes(self) -> int:
        return self.total_size * self.dtype.itemsize

    def allocate(self):
        if self.physical_chunks:
            return
        if self.device_id is None:
            device = "cuda"
        else:
            device = f"cuda:{self.device_id}"
        total_size_per_chunk = self.total_size // self.num_chunks
        for _ in range(self.num_chunks):
            self.physical_chunks.append(
                torch.empty(
                    size=(total_size_per_chunk,),
                    dtype=self.dtype,
                    device=device,
                )
            )

    def free(self):
        if not self._need_free:
            return
        self.physical_chunks.clear()

    def get_accessible_handle(self) -> AccessibleHandle:
        if not self.physical_chunks:
            raise RuntimeError(f"Physical blocks are not allocated on GPU {self.device_id}")
        return AccessibleHandle(
            handle_type=AccessHandleType.TENSOR,
            data=self.physical_chunks,
            kv_layout=self.layout,
            dtype=self.dtype,
            gpu_device_id=self.device_id,
        )

    @classmethod
    def from_raw_data(cls,
        data: Union[List[torch.Tensor], List[KVCacheTensorHandle]],
        layout: KVCacheLayout,
        dtype: torch.dtype,
        device_id: Optional[int] = None) -> 'GPUAllocator':
        allocator = cls(
            layout=layout,
            dtype=dtype,
            device_id=device_id,
            num_chunks=len(data),
        )
        allocator._need_free = False
        allocator.physical_chunks = data
        return allocator

class CPUAllocator(StorageAllocator):
    def __init__(
        self,
        layout: KVCacheLayout,
        dtype: torch.dtype,
        pin_memory: bool = True,
        num_chunks: int = 1,
    ):
        self.layout = layout
        self.dtype = dtype
        self.pin_memory = pin_memory
        self.num_chunks = num_chunks

        self.physical_chunks = []
        self._need_free = True

    @property
    def total_size(self) -> int:
        return self.layout.get_total_elements()

    @property
    def total_size_in_bytes(self) -> int:
        return self.total_size * self.dtype.itemsize

    def allocate(self):
        if self.physical_chunks:
            return
        total_size_per_chunk = self.total_size // self.num_chunks
        for _ in range(self.num_chunks):
            self.physical_chunks.append(
                torch.empty(
                    size=(total_size_per_chunk,),
                    dtype=self.dtype,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
            )

    def free(self):
        if not self._need_free:
            return
        self.physical_chunks.clear()

    def get_accessible_handle(self) -> AccessibleHandle:
        if not self.physical_chunks:
            raise RuntimeError("Physical blocks are not allocated on CPU")
        return AccessibleHandle(
            handle_type=AccessHandleType.TENSOR,
            data=self.physical_chunks,
            kv_layout=self.layout,
            dtype=self.dtype,
        )

    @classmethod
    def from_raw_data(cls,
        data: List[torch.Tensor],
        layout: KVCacheLayout) -> 'CPUAllocator':
        allocator = cls(
            layout=layout,
            dtype=data[0].dtype,
            pin_memory=data[0].is_pinned(),
            num_chunks=len(data),
        )
        allocator._need_free = False
        allocator.physical_chunks = data
        return allocator

class SSDAllocator(StorageAllocator):
    def __init__(
        self,
        layout: KVCacheLayout,
        dtype: torch.dtype,
        file_path: Union[str, List[str]],
    ):
        if isinstance(file_path, str):
            file_path = [file_path]
        self.file_path = file_path
        self.layout = layout
        self.dtype = dtype
        self.num_files = len(file_path)

        self._has_allocated = False

    @property
    def total_size(self) -> int:
        return self.layout.get_total_elements()

    @property
    def total_size_in_bytes(self) -> int:
        return self.total_size * self.dtype.itemsize

    def allocate(self):
        if self._has_allocated:
            return
        for path in self.file_path:
            with open(path, "wb+", buffering=0) as file:
                self._init_file(file)
        self._has_allocated = True

    def _init_file(self, file):
        file_size = self.total_size_in_bytes // self.num_files
        try:
            os.truncate(file.fileno(), file_size)
        except OSError as e:
            raise RuntimeError(f"Failed to initialize file: {e}") from e
        file.flush()
        os.fsync(file.fileno())

    def free(self):
        pass

    def get_accessible_handle(self) -> AccessibleHandle:
        if not self._has_allocated:
            raise RuntimeError("SSD file is not allocated")
        return AccessibleHandle(
            handle_type=AccessHandleType.FILE,
            data=self.file_path,
            kv_layout=self.layout,
            dtype=self.dtype,
        )

    @classmethod
    def from_raw_data(cls,
        layout: KVCacheLayout,
        dtype: torch.dtype,
        file_path: Union[str, List[str]]) -> 'SSDAllocator':
        allocator = cls(
            layout=layout,
            dtype=dtype,
            file_path=file_path,
        )
        allocator._has_allocated = True
        return allocator

class RemoteAllocator(StorageAllocator):
    def __init__(
        self,
        layout: KVCacheLayout,
        dtype: torch.dtype,
        file_path: Union[str, List[str]],
        remote_config_custom: Dict[str, Any]
    ):
        if isinstance(file_path, str):
            file_path = [file_path]
        self.file_path = file_path
        self.layout = layout
        self.dtype = dtype
        self.num_files = len(file_path)
        self.remote_config_custom = remote_config_custom

        self._has_allocated = False

    @property
    def total_size(self) -> int:
        return self.layout.get_total_elements()

    @property
    def total_size_in_bytes(self) -> int:
        return self.total_size * self.dtype.itemsize

    def allocate(self):
        if self._has_allocated:
            return

        self._has_allocated = True

    def _init_file(self, file):
        pass
    
    def free(self):
        pass

    def get_accessible_handle(self) -> AccessibleHandle:
        if not self._has_allocated:
            raise RuntimeError("Remote file is not allocated")
        return AccessibleHandle(
            handle_type=AccessHandleType.FILE,
            data=self.file_path,
            kv_layout=self.layout,
            dtype=self.dtype,
            remote_config_custom = self.remote_config_custom,
        )

    @classmethod
    def from_raw_data(cls,
        layout: KVCacheLayout,
        dtype: torch.dtype,
        file_path: Union[str, List[str]],
        remote_config_custom: Dict[str, Any]) -> 'RemoteAllocator':
        allocator = cls(
            layout=layout,
            dtype=dtype,
            file_path=file_path,
            remote_config_custom = remote_config_custom,
        )
        allocator._has_allocated = True
        return allocator
