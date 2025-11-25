import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Union, Dict, Any, BinaryIO
try:
    from flexkv.c_ext import Pcfs
except ImportError:
    Pcfs = None

import numpy as np
import torch

from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import StorageHandle, AccessHandleType, KVCacheLayout, KVCacheLayoutType
from flexkv.common.debug import flexkv_logger


class BaseStorageAllocator(ABC):
    @classmethod
    @abstractmethod
    def allocate(cls,
                 layout: KVCacheLayout,  # TODO: do we need to pass layout/dtype here?
                 dtype: torch.dtype,
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
                 **kwargs: Any) -> StorageHandle:
        device_id = kwargs.get("device_id", torch.cuda.current_device())
        device = f"cuda:{device_id}"
        num_chunks = kwargs.get("num_chunks", 1)

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
                 **kwargs: Any) -> StorageHandle:
        total_size = layout.get_total_elements()
        # although the kv layout may have multiple dimensions, we only have one-dim CPU tensor
        flexkv_logger.info(f"CPU allocate total_size: {2 * total_size/1024/1024/1024} GB")
        physical_tensor = torch.empty(
                            size=(total_size,),
                            dtype=dtype,
                            device="cpu",
                            pin_memory=False,
                        )
        return StorageHandle(
            handle_type=AccessHandleType.TENSOR,
            data=physical_tensor,
            kv_layout=layout,
            dtype=dtype,
        )

    @classmethod
    def free(cls, accessible_handle: StorageHandle) -> None:
        pass

    @classmethod
    def from_raw_data(cls,
                      data: torch.Tensor,  # type: ignore
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
                 **kwargs: Any) -> StorageHandle:
        cache_dir = kwargs.get("cache_dir")
        file_prefix = kwargs.get("file_prefix", "flexkv_ssd_cache")
        cfg_max_blocks_per_file = kwargs.get("max_blocks_per_file", -1)
        if cfg_max_blocks_per_file == -1:
            cfg_max_blocks_per_file = int(1e9)
        if cache_dir is None:
            raise ValueError("cache_dir is required for SSD allocator")
        if isinstance(cache_dir, str):
            cache_dir = [cache_dir]
        for dir in cache_dir:
            if not os.path.exists(dir):
                os.makedirs(dir)
            if not os.path.isdir(dir):
                raise ValueError("cache_dir must be a directory")
        if not isinstance(file_prefix, str):
            raise ValueError("file_prefix must be a string")

        num_ssd_devices = len(cache_dir)
        if layout.num_block % num_ssd_devices != 0:
            raise ValueError(f"num_ssd_blocks ({layout.num_block}) must be a multiple of "
                             f"num_ssd_devices ({num_ssd_devices})")

        total_blocks_per_device = layout.num_block // num_ssd_devices
        block_size = layout.get_elements_per_block() * dtype.itemsize

        fsys_max_blocks_per_file = cls.get_file_size_limit(cache_dir[0]) // block_size
        num_blocks_per_file = min(fsys_max_blocks_per_file, cfg_max_blocks_per_file)

        num_files_per_device = (total_blocks_per_device + num_blocks_per_file - 1) // num_blocks_per_file
        real_file_size = num_blocks_per_file * block_size

        ssd_files: Dict[int, List[str]] = {}
        for i in range(num_ssd_devices):
            ssd_files[i] = []
            for j in range(num_files_per_device):
                file_path = os.path.join(cache_dir[i], f"{file_prefix}_{i}_{j}.bin")
                with open(file_path, "wb+", buffering=0) as file:
                    cls._create_file(file, real_file_size)
                ssd_files[i].append(file_path)

        return StorageHandle(
            handle_type=AccessHandleType.FILE,
            data=ssd_files,
            kv_layout=layout,
            dtype=dtype,
            num_blocks_per_file=num_blocks_per_file,
        )

    @classmethod
    def _create_file(cls, file: BinaryIO, total_size_per_file: int) -> None:
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
        raise NotImplementedError

    @staticmethod
    def get_file_size_limit(file_path: str) -> int:
        st = os.statvfs(file_path)
        return st.f_frsize * st.f_bavail

class RemoteAllocator(BaseStorageAllocator):
    @classmethod
    def allocate(cls,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 **kwargs: Any) -> StorageHandle:
        file_path = kwargs.get("file_path")
        if file_path is None:
            raise ValueError("file_path is required for Remote allocator")
        remote_config_custom = kwargs.get("remote_config_custom")
        if remote_config_custom is None:
            raise ValueError("remote_config_custom is required for Remote allocator")
        if isinstance(file_path, str):
            file_path = [file_path]

        if not remote_config_custom:
            raise RuntimeError("remote_config_custom is not provided")
        pcfs_fsid = remote_config_custom.get("pcfs_fsid")
        pcfs_port = remote_config_custom.get("pcfs_port")
        pcfs_ip = remote_config_custom.get("pcfs_ip")
        pcfs_parent_nodeid = remote_config_custom.get("pcfs_parent_nodeid")
        if None in (pcfs_fsid, pcfs_port, pcfs_ip, pcfs_parent_nodeid):
            raise RuntimeError("Some required PCFS config fields are missing")
        if Pcfs is None:
            raise RuntimeError("Pcfs class not available. Please build with FLEXKV_ENABLE_CFS=1")
        pcfs = Pcfs(pcfs_fsid, pcfs_port, pcfs_ip, False, pcfs_parent_nodeid)
        if not pcfs.init():
            raise RuntimeError(f"PCFS init failed: fsid={pcfs_fsid}, ip={pcfs_ip}")
        for file in file_path:
            total_size = layout.get_total_elements() * dtype.itemsize
            file_size = total_size // len(file_path)
            need_create = True
            print(f"file_size in init:{file_size}")
            nodeid = pcfs.lookup_or_create_file(file, file_size, need_create)
            if nodeid == 0:
                raise RuntimeError(f"lookup or create file failed for file: {file}")

            # destroy pcfs & close file, not used
            close_res = pcfs.close(nodeid, 1000)
            if not close_res:
                raise RuntimeError(f"close file failed for file: {file}")
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
