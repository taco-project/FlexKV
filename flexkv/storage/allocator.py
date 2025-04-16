from abc import ABC, abstractmethod
from flexkv.common.storage import AccessibleHandle, AccessHandleType
import torch
from typing import Tuple, Optional, List
import numpy as np
class StorageAllocator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def allocate(self, size: int) -> int:
        pass

    @abstractmethod
    def free(self, offset: int):
        pass

    @abstractmethod
    def get_accessible_handle(self) -> AccessibleHandle:
        pass

    @classmethod
    @abstractmethod
    def from_raw_data() -> 'StorageAllocator':
        pass

    def _get_layer_ptrs(self, layer_blocks: List[torch.Tensor]) -> torch.Tensor:
        num_layers = len(layer_blocks)
        layer_ptrs = torch.zeros(
            num_layers,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        for lay_id in range(num_layers):
            layer_ptrs[lay_id] = layer_blocks[lay_id][0].data_ptr()
        return layer_ptrs

class GPUAllocator(StorageAllocator):
    def __init__(
        self,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device_id: Optional[int] = None, 
    ):
        self.tensor_shape = tensor_shape
        self.dtype = dtype
        self.device_id = device_id
        self.physical_blocks = []
        self.allocate()
        self.layer_ptrs = None

    def allocate(self):
        if self.physical_blocks:
            return
        if self.device_id is None:
            device = "cuda"
        else:
            device = f"cuda:{self.device_id}"
        for _ in range(self.tensor_shape[0]):
            self.physical_blocks.append(
                torch.empty(
                    size=self.tensor_shape[1:],
                    dtype=self.dtype,
                    device=device,
                )
            )

    def free(self):
        for block in self.physical_blocks:
            block.free()
    
    def get_accessible_handle(self) -> AccessibleHandle:
        if not self.layer_ptrs:
            self.layer_ptrs = self._get_layer_ptrs(self.physical_blocks)
        return AccessibleHandle(
            handle_type=AccessHandleType.TENSOR_LIST,
            data=self.layer_ptrs,
            kv_shape=self.tensor_shape,
            dtype=self.dtype,
            device_id=self.device_id,
        )

    @classmethod
    def from_raw_data(cls, 
        data: List[torch.Tensor],
        shape: Tuple[int, ...],
        device_id: Optional[int] = None) -> 'GPUAllocator':
        allocator = cls(
            tensor_shape=shape,
            dtype=data[0].dtype,
            device_id=device_id,
        )
        allocator.physical_blocks = data
        #allocator.layer_ptrs = allocator._get_layer_ptrs(data)
        return allocator
        
    
class CPUAllocator(StorageAllocator):
    def __init__(
        self,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype,
        pin_memory: bool = False
    ):
        assert len(tensor_shape) == 4
        self.tensor_shape = tensor_shape
        self.num_layers = tensor_shape[0]
        self.num_blocks = tensor_shape[2]
        self.block_size = tensor_shape[3]
        self.dtype = dtype
        self.pin_memory = pin_memory
        self.physical_blocks = []
        self.allocate()
        self.layer_ptrs = None
    
    def allocate(self):
        if self.physical_blocks:
            return
        for _ in range(self.tensor_shape[0]):
            self.physical_blocks.append(
                torch.empty(
                    size=self.tensor_shape[1:],
                    dtype=self.dtype,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
            )

    def free(self):
        for block in self.physical_blocks:
            block.free()
    
    def get_accessible_handle(self) -> AccessibleHandle:
        if not self.layer_ptrs:
            self.layer_ptrs = self._get_layer_ptrs(self.physical_blocks)
        return AccessibleHandle(
            handle_type=AccessHandleType.TENSOR_LIST,
            data=self.layer_ptrs,
            kv_shape=self.tensor_shape,
            dtype=self.dtype,
        )

    @classmethod
    def from_raw_data(cls, 
        data: List[torch.Tensor],
        shape: Tuple[int, ...],
        pin_memory: bool = False) -> 'CPUAllocator':
        allocator = cls(
            tensor_shape=shape,
            dtype=data[0].dtype,
            pin_memory=pin_memory,
        )
        allocator.physical_blocks = data
        #allocator.layer_ptrs = allocator._get_layer_ptrs(data)
        return allocator

class SSDAllocator(StorageAllocator):
    def __init__(
        self,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        file_path: str = "ssd.cache",
    ):
        self.file_path = file_path
        self.tensor_shape = tensor_shape
        self.dtype = dtype
        self.total_file_size = np.prod(tensor_shape) * dtype.itemsize

    def allocate(self):
        self.file = open(self.file_path, "wb+", buffering=0)
        self.file.write(b"\x00" * self.total_file_size)
        self.file.close()

    def free(self):
        pass

    def get_accessible_handle(self) -> AccessibleHandle:
        return AccessibleHandle(
            handle_type=AccessHandleType.FILE,
            data=self.file_path,
            kv_shape=self.tensor_shape,
            dtype=self.dtype,
        )

    @classmethod
    def from_raw_data(cls,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        file_path: str = "ssd.cache") -> 'SSDAllocator':
        allocator = cls(
            tensor_shape=shape,
            dtype=dtype,
            file_path=file_path)
        return allocator