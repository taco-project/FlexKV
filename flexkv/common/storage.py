from enum import Enum, auto
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple
import torch
from pathlib import Path

class AccessHandleType(Enum):
    TENSOR_LIST = "gpu_handle"     # List of tensors in memory
    TENSOR_LIST_LIST = "cpu_handle" # List of list of tensors in memory
    FILE = "disk_handle"           # File handle

@dataclass
class AccessibleHandle:
    handle_type: AccessHandleType
    # The actual handle data
    data: Union[List[torch.Tensor], List[List[torch.Tensor]], str]
    kv_shape: Tuple[int, ...]
    dtype: torch.dtype
    # Optional metadata
    device_id: Optional[int] = None
    
    @classmethod
    def from_tensor_list(cls, 
                         tensors: List[torch.Tensor], 
                         kv_shape: Tuple[int, ...],
                         device_id: Optional[int] = None) -> 'AccessibleHandle':
        return cls(
            handle_type=AccessHandleType.TENSOR_LIST,
            data=tensors,
            kv_shape=kv_shape,
            dtype=tensors[0].dtype,
            device_id=device_id
        )
    
    @classmethod
    def from_file(cls, 
                  file_path: Union[str, Path], 
                  kv_shape: Tuple[int, ...],
                  dtype: torch.dtype,
                  device_id: Optional[int] = None) -> 'AccessibleHandle':
        return cls(
            handle_type=AccessHandleType.FILE,
            data=Path(file_path),
            kv_shape=kv_shape,
            dtype=dtype,
            device_id=device_id
        )
    
    def get_tensors(self) -> Optional[List[torch.Tensor]]:
        """Get tensor list if handle type is TENSOR_LIST"""
        return self.data if self.handle_type == AccessHandleType.TENSOR_LIST else None
    
    def get_file_path(self) -> Optional[Path]:
        """Get file path if handle type is FILE"""
        return self.data if self.handle_type == AccessHandleType.FILE else None

    def get_handle_data(self) -> Optional[Union[List[torch.Tensor], List[List[torch.Tensor]], Path]]:
        """Get handle data"""
        return self.data

    def get_kv_shape(self) -> Tuple[int, ...]:
        """Get kv shape"""
        return self.kv_shape
    
    @classmethod        
    def build_from_handle_data(
        cls,
        handle_type: AccessHandleType,
        handle_data: Union[List[torch.Tensor], List[List[torch.Tensor]], Path],
        kv_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device_id: Optional[int] = None
    ) -> 'AccessibleHandle':
        """Build from handle data"""
        return cls(
            handle_type=handle_type,
            data=handle_data,
            kv_shape=kv_shape,
            dtype=dtype,
            device_id=device_id
        )
