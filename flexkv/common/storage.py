from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Union, List, Optional, Any, Dict

import torch

from flexkv.common.transfer import DeviceType


class AccessHandleType(Enum):
    TENSOR = auto()  # single tensor or tensor list
    FILE = auto()  # single file or file list

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
    kv_shape: torch.Size = None

    def __post_init__(self):
        if self.kv_shape is None:
            if self.type == KVCacheLayoutType.LAYERWISE:
                if self.is_mla:
                    self.kv_shape = torch.Size([self.num_layer,
                                                self.num_block,
                                                self.tokens_per_block,
                                                self.num_head,
                                                self.head_size])
                else:
                    self.kv_shape = torch.Size([self.num_layer,
                                                2,
                                                self.num_block,
                                                self.tokens_per_block,
                                                self.num_head,
                                                self.head_size])
            elif self.type == KVCacheLayoutType.BLOCKWISE:
                if self.is_mla:
                    self.kv_shape = torch.Size([self.num_block,
                                                self.num_layer,
                                                self.tokens_per_block,
                                                self.num_head,
                                                self.head_size])
                else:
                    self.kv_shape = torch.Size([self.num_block,
                                                self.num_layer,
                                                2,
                                                self.tokens_per_block,
                                                self.num_head,
                                                self.head_size])
            else:
                raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_kv_shape(self) -> torch.Size:
        return self.kv_shape

    def get_total_elements(self) -> int:
        return self.kv_shape.numel()

@dataclass
class AccessibleHandle:
    handle_type: AccessHandleType
    # The actual handle data
    data: Union[List[torch.Tensor], List[List[torch.Tensor]], str]
    kv_layout: KVCacheLayout
    dtype: torch.dtype
    # Optional metadata
    gpu_device_id: Optional[int] = None
    remote_config_custom: Optional[Dict[str, Any]] = None

    @classmethod
    def from_tensor(cls,
                    tensor: Union[torch.Tensor, List[torch.Tensor]],
                    kv_layout: KVCacheLayout,
                    gpu_device_id: Optional[int] = None) -> 'AccessibleHandle':
        if isinstance(tensor, torch.Tensor):
            tensor_list = [tensor]
        return cls(
            handle_type=AccessHandleType.TENSOR,
            data=tensor_list,
            kv_layout=kv_layout,
            dtype=tensor_list[0].dtype,
            gpu_device_id=gpu_device_id,
            remote_config_custom=None
        )

    @classmethod
    def from_file(cls,
                  file_path: Union[str, Path, List[Union[str, Path]]],
                  kv_layout: KVCacheLayout,
                  dtype: torch.dtype,
                  remote_config_custom: Optional[Dict[str, Any]] = None
                  ) -> 'AccessibleHandle':
        if isinstance(file_path, (str, Path)):
            file_path_list = [str(file_path)]
        if isinstance(file_path[0], Path):
            file_path_list = [str(fp) for fp in file_path]
        return cls(
            handle_type=AccessHandleType.FILE,
            data=file_path_list,
            kv_layout=kv_layout,
            dtype=dtype,
            gpu_device_id=None,
            remote_config_custom=remote_config_custom
        )
