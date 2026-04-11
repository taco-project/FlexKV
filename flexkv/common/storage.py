from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Union, List, Optional, Any, Dict, TYPE_CHECKING

import torch

from flexkv.common.memory_handle import TensorSharedHandle

if TYPE_CHECKING:
    from flexkv.common.config import LayerGroupSpec


class AccessHandleType(Enum):
    TENSOR = auto()  # single tensor or tensor list
    FILE = auto()  # single file or file list
    TENSOR_HANDLE = auto()  # single tensor handle or tensor handle list
    GDS_MANAGER = auto()

# NOTE: currently, we assume that the layout type of GPU should always be LAYERFIRST
# and the layout type of CPU, SSD, remote should be the same, either laywise or BLOCKFIRST
class KVCacheLayoutType(Enum):
    LAYERFIRST = "LAYERFIRST"
    BLOCKFIRST = "BLOCKFIRST"

@dataclass
class KVCacheLayout:
    type: KVCacheLayoutType
    num_layer: int
    num_block: int
    tokens_per_block: int
    num_head: int
    head_size: int
    is_mla: bool
    _kv_shape: Optional[torch.Size] = None
    # Multi-group support: when set, the layout represents a heterogeneous block
    # where different layer groups have different (num_kv_heads, head_size).
    layer_groups: Optional[List[LayerGroupSpec]] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KVCacheLayout):
            return NotImplemented
        return (self.type == other.type and
                self.num_layer == other.num_layer and
                self.num_block == other.num_block and
                self.tokens_per_block == other.tokens_per_block and
                self.num_head == other.num_head and
                self.head_size == other.head_size and
                self.is_mla == other.is_mla and
                self.layer_groups == other.layer_groups and
                self.kv_shape == other.kv_shape)

    @property
    def _kv_dim(self) -> int:
        return 2 if not self.is_mla else 1

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
            if self.layer_groups is not None and self.type == KVCacheLayoutType.BLOCKFIRST:
                # Multi-group: elements per block = sum of per-group contributions
                # Each group: num_layers * kv_dim * tokens_per_block * num_kv_heads * head_size
                elements_per_block = sum(
                    g.num_layers * self._kv_dim * self.tokens_per_block * g.num_kv_heads * g.head_size
                    for g in self.layer_groups
                )
                self._kv_shape = torch.Size([self.num_block, elements_per_block])
            elif self.type == KVCacheLayoutType.LAYERFIRST:  # for Layerwise transfer
                self._kv_shape = torch.Size([self.num_layer,
                                             self._kv_dim,
                                             self.num_block,
                                             self.tokens_per_block,
                                             self.num_head,
                                             self.head_size])
            elif self.type == KVCacheLayoutType.BLOCKFIRST:
                self._kv_shape = torch.Size([self.num_block,
                                             self.num_layer,
                                             self._kv_dim,
                                             self.tokens_per_block,
                                             self.num_head,
                                             self.head_size])
            else:
                raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def div_block(self, num_chunks: int, padding: bool = False) -> 'KVCacheLayout':
        if padding:
            num_blocks = (self.num_block + num_chunks - 1) // num_chunks
        else:
            assert self.num_block % num_chunks == 0, \
                f"num_block {self.num_block} must be divisible by num_chunks {num_chunks}"
            num_blocks = self.num_block // num_chunks
        new_layout = KVCacheLayout(
            type=self.type,
            num_layer=self.num_layer,
            num_block=num_blocks,
            tokens_per_block=self.tokens_per_block,
            num_head=self.num_head,
            head_size=self.head_size,
            is_mla=self.is_mla,
            layer_groups=self.layer_groups,
        )
        return new_layout

    def div_layer(self, num_chunks: int) -> 'KVCacheLayout':
        assert self.num_layer % num_chunks == 0, \
            f"num_layer {self.num_layer} must be divisible by num_chunks {num_chunks}"
        new_layout = KVCacheLayout(
            type=self.type,
            num_layer=self.num_layer // num_chunks,
            num_block=self.num_block,
            tokens_per_block=self.tokens_per_block,
            num_head=self.num_head,
            head_size=self.head_size,
            is_mla=self.is_mla,
        )
        return new_layout

    def div_head(self, num_chunks: int) -> 'KVCacheLayout':
        assert self.num_head % num_chunks == 0, \
            f"num_head {self.num_head} must be divisible by num_chunks {num_chunks}"
        new_layout = KVCacheLayout(
            type=self.type,
            num_layer=self.num_layer,
            num_block=self.num_block,
            tokens_per_block=self.tokens_per_block,
            num_head=self.num_head // num_chunks,
            head_size=self.head_size,
            is_mla=self.is_mla,
        )
        return new_layout

    def get_chunk_size(self) -> int:
        if self.layer_groups is not None:
            raise ValueError("get_chunk_size() is not valid for multi-group layout; "
                             "use per-group strides instead")
        return self.tokens_per_block * self.num_head * self.head_size

    def get_layer_stride(self) -> int:
        if self.layer_groups is not None:
            raise ValueError("get_layer_stride() is not valid for multi-group layout; "
                             "use per-group strides instead")
        if self.type == KVCacheLayoutType.LAYERFIRST:
            return self.kv_shape[1:].numel()
        elif self.type == KVCacheLayoutType.BLOCKFIRST:
            return self.kv_shape[2:].numel()
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_block_stride(self) -> int:
        if self.layer_groups is not None:
            # For multi-group BLOCKFIRST, kv_shape is [num_block, elements_per_block]
            return self.kv_shape[1]
        if self.type == KVCacheLayoutType.LAYERFIRST:
            return self.kv_shape[3:].numel()
        elif self.type == KVCacheLayoutType.BLOCKFIRST:
            return self.kv_shape[1:].numel()
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_kv_stride(self) -> int:
        if self.layer_groups is not None:
            raise ValueError("get_kv_stride() is not valid for multi-group layout; "
                             "use per-group strides instead")
        if self.type == KVCacheLayoutType.LAYERFIRST:
            return self.kv_shape[2:].numel()
        elif self.type == KVCacheLayoutType.BLOCKFIRST:
            return self.kv_shape[3:].numel()
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_group_strides(self) -> List[Dict[str, int]]:
        """Compute per-group stride info for multi-group BLOCKFIRST layout.

        Returns a list of dicts, one per group, each containing:
            - num_layers: number of layers in this group
            - offset_elements: element offset of this group within a block
            - layer_stride: elements per layer (kv_dim * tpb * num_kv_heads * head_size)
            - kv_stride: elements per KV half (tpb * num_kv_heads * head_size)
            - chunk_size: elements per K or V chunk (tpb * num_kv_heads * head_size)
        """
        if self.layer_groups is None:
            raise ValueError("get_group_strides() requires layer_groups to be set")
        if self.type != KVCacheLayoutType.BLOCKFIRST:
            raise ValueError("get_group_strides() only supports BLOCKFIRST layout")

        result = []
        offset = 0
        for g in self.layer_groups:
            chunk = self.tokens_per_block * g.num_kv_heads * g.head_size
            kv_stride = chunk  # tpb * num_kv_heads * head_size
            layer_stride = self._kv_dim * kv_stride
            group_elements = g.num_layers * layer_stride
            result.append({
                'num_layers': g.num_layers,
                'num_kv_heads': g.num_kv_heads,
                'head_size': g.head_size,
                'layer_indices': g.layer_indices,
                'offset_elements': offset,
                'layer_stride': layer_stride,
                'kv_stride': kv_stride,
                'chunk_size': chunk,
            })
            offset += group_elements
        return result

    def get_total_elements(self) -> int:
        return self.kv_shape.numel()

    def get_elements_per_block(self) -> int:
        return self.get_total_elements() // self.num_block


@dataclass
class StorageHandle:
    handle_type: AccessHandleType
    # The actual handle data
    data: Union[List[torch.Tensor],
                torch.Tensor,
                List[str],
                List[TensorSharedHandle],  # for shared gpu tensors
                Dict[int, List[str]]  # for ssd files: ssd_device_id -> file_paths
                ]
    kv_layout: KVCacheLayout
    dtype: torch.dtype
    # Optional metadata
    num_blocks_per_file: Optional[int] = None
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
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected TENSOR or TENSOR_HANDLE")

    def get_tensor(self) -> torch.Tensor:
        assert isinstance(self.data, torch.Tensor), \
            "handle data must be torch.Tensor"
        if self.handle_type == AccessHandleType.TENSOR:
            return self.data
        else:
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected TENSOR")

    def get_file_list(self) -> Union[List[str], Dict[int, List[str]]]:
        if self.handle_type == AccessHandleType.FILE:
            return self.data  # type: ignore
        else:
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected FILE")

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
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected TENSOR_HANDLE or TENSOR")
