from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Union, List, Optional, Any, Dict

import torch

from flexkv.common.memory_handle import TensorSharedHandle


class AccessHandleType(Enum):
    TENSOR = auto()  # single tensor or tensor list
    FILE = auto()  # single file or file list
    TENSOR_HANDLE = auto()  # single tensor handle or tensor handle list

class KVCacheLayoutType(Enum):
    LAYERWISE = "layerwise"
    BLOCKWISE = "blockwise"

@dataclass
class KVCacheLayout:
    type: KVCacheLayoutType
    num_layer: int
    num_block: int
    tokens_per_block: int
    num_head: int
    head_size: int
    is_mla: bool = False
    _kv_shape: Optional[torch.Size] = None

    @property
    def kv_shape(self) -> torch.Size:
        if self._kv_shape is None:
            self._compute_kv_shape()
        assert self._kv_shape is not None
        return self._kv_shape

    def __post_init__(self) -> None:
        self._compute_kv_shape()

    def _compute_kv_shape(self) -> None:
        if self._kv_shape is None:
            if self.type == KVCacheLayoutType.LAYERWISE:
                if self.is_mla:
                    self._kv_shape = torch.Size([self.num_layer,
                                                self.num_block,
                                                self.tokens_per_block,
                                                self.num_head,
                                                self.head_size])
                else:
                    self._kv_shape = torch.Size([self.num_layer,
                                                2,
                                                self.num_block,
                                                self.tokens_per_block,
                                                self.num_head,
                                                self.head_size])
            elif self.type == KVCacheLayoutType.BLOCKWISE:
                if self.is_mla:
                    self._kv_shape = torch.Size([self.num_block,
                                                self.num_layer,
                                                self.tokens_per_block,
                                                self.num_head,
                                                self.head_size])
                else:
                    self._kv_shape = torch.Size([self.num_block,
                                                self.num_layer,
                                                2,
                                                self.tokens_per_block,
                                                self.num_head,
                                                self.head_size])
            else:
                raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_total_elements(self) -> int:
        return self.kv_shape.numel()

@dataclass
class StorageHandle:
    handle_type: AccessHandleType
    # The actual handle data
    data: Union[List[torch.Tensor], List[str], List[TensorSharedHandle]]
    kv_layout: KVCacheLayout
    dtype: torch.dtype
    # Optional metadata
    gpu_device_id: Optional[int] = None
    remote_config_custom: Optional[Dict[str, Any]] = None

    def get_tensor_list(self) -> List[torch.Tensor]:
        assert isinstance(self.data, list) and \
                (all(isinstance(x, torch.Tensor) for x in self.data) or \
                all(isinstance(x, TensorSharedHandle) for x in self.data)), \
                "handle data must be List[Tensor] or List[TensorWrapper]"
        if self.handle_type == AccessHandleType.TENSOR:
            return self.data  # type: ignore
        elif self.handle_type == AccessHandleType.TENSOR_HANDLE:
            assert all(isinstance(x, TensorSharedHandle) for x in self.data), \
                "All elements must be TensorSharedHandle for TENSOR_HANDLE type"
            return [x.get_tensor() for x in self.data]  # type: ignore
        else:
            raise ValueError(f"Invalid data type: {type(self.data)}, expected list of tensors")

    def get_file_list(self) -> List[str]:
        if self.handle_type == AccessHandleType.FILE:
            assert isinstance(self.data, list) and \
                all(isinstance(x, str) for x in self.data), \
                "handle data must be List[str]"
            return self.data  # type: ignore
        else:
            raise ValueError(f"Invalid data type: {type(self.data)}, expected list of strings")

    def get_tensor_handle_list(self) -> List[TensorSharedHandle]:
        assert isinstance(self.data, list) and \
                (all(isinstance(x, torch.Tensor) for x in self.data) or \
                all(isinstance(x, TensorSharedHandle) for x in self.data)), \
                "handle data must be List[Tensor] or List[TensorWrapper]"
        if self.handle_type == AccessHandleType.TENSOR_HANDLE:
            assert all(isinstance(x, TensorSharedHandle) for x in self.data), \
                "All elements must be TensorSharedHandle for TENSOR_HANDLE type"
            return self.data  # type: ignore
        elif self.handle_type == AccessHandleType.TENSOR:
            assert all(isinstance(x, torch.Tensor) for x in self.data), \
                "All elements must be torch.Tensor for TENSOR type"
            return [TensorSharedHandle(x) for x in self.data]  # type: ignore
        else:
            raise ValueError(f"Invalid data type: {type(self.data)}, expected list of tensors")
