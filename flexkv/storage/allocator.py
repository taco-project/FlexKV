import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Union, Dict, Any, BinaryIO
try:
    from flexkv.c_ext import Pcfs
    from flexkv.c_ext import get_fm_extents, FileExtent
except ImportError:
    Pcfs = None
    FileExtent = None

import numpy as np
import torch

from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import StorageHandle, AccessHandleType, KVCacheLayout, KVCacheLayoutType
from flexkv.common.debug import flexkv_logger
from flexkv.cache.redis_meta import RedisMeta


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
        cfg_max_file_size_gb = kwargs.get("max_file_size_gb", -1)
        cfg_max_blocks_per_file = int(1e9)
        enable_nvmet: bool = kwargs.get('enable_nvmet', False)
        md_dev: str = kwargs.get('md_dev', '/dev/md0')
        nvmf_targets: Optional[Dict[int, Dict[str, str]]] = kwargs.get('nvmf_targets', None)
        redis_meta: Optional[RedisMeta] = kwargs.get('redis_meta', None)

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
        if enable_nvmet:
            assert FileExtent is not None, 'FileExtent class unavailable'
            assert nvmf_targets is not None, 'nvmf_targets is invalid for NVMe-oF target offload'
            assert redis_meta is not None, 'redis_meta is required for NVMe-oF target offload'

        num_ssd_devices = len(cache_dir)
        if layout.num_block % num_ssd_devices != 0:
            raise ValueError(f"num_ssd_blocks ({layout.num_block}) must be a multiple of "
                             f"num_ssd_devices ({num_ssd_devices})")

        total_blocks_per_device = layout.num_block // num_ssd_devices
        block_size = layout.get_elements_per_block() * dtype.itemsize

        if cfg_max_file_size_gb != -1:
            cfg_max_blocks_per_file = int(cfg_max_file_size_gb * 1024 * 1024 * 1024 // block_size)
        else:
            # when we don't set max_file_size_gb, we will create a file, size is exactly the required capacity
            cfg_max_blocks_per_file = total_blocks_per_device

        fsys_max_blocks_per_file = cls.get_file_size_limit(cache_dir[0]) // block_size
        num_blocks_per_file = min(fsys_max_blocks_per_file, cfg_max_blocks_per_file, total_blocks_per_device)

        num_files_per_device = (total_blocks_per_device + num_blocks_per_file - 1) // num_blocks_per_file
        real_file_size = num_blocks_per_file * block_size

        # Publish RAID geomoetry
        if enable_nvmet:
            if md_dev is not None:
                assert num_ssd_devices == 1, 'Use a single SSD cache dir for RAID0' # Only support RAID0
                redis_meta.publish_nvme_geometry(cls._get_md_member_devs(md_dev),
                                                 num_files_per_device,
                                                 cls._get_md_chunk_size(md_dev))
            else:
                redis_meta.publish_nvme_geometry(cls._get_backing_devs(cache_dir),
                                                 num_files_per_device)

        ssd_files: Dict[int, List[str]] = {}
        all_file_extents: Dict[Tuple[int, int], List[FileExtent]] = {}
        # TODO: Parallelize file allocation + flush & sync + layout detection
        for i in range(num_ssd_devices):
            ssd_files[i] = []
            for j in range(num_files_per_device):
                file_path = os.path.join(cache_dir[i], f"{file_prefix}_{i}_{j}.bin")
                with open(file_path, "wb+", buffering=0) as file:
                    extent: FileExtent = cls._create_file(file, real_file_size, get_layout=enable_nvmet)
                    if enable_nvmet:
                        all_file_extents[(i, j)] = extent
                ssd_files[i].append(file_path)
        if enable_nvmet:
            redis_meta.publish_file_extents_batch(all_file_extents)

        total_num_files = num_files_per_device * num_ssd_devices
        real_total_size = total_num_files * real_file_size
        flexkv_logger.info(f"SSD allocator create total {total_num_files} files in {cache_dir}, "
                           f"each file has {real_file_size/1024/1024/1024:.2f} GB, total size {real_total_size/1024/1024/1024:.2f} GB")
        return StorageHandle(
            handle_type=AccessHandleType.FILE,
            data=ssd_files,
            kv_layout=layout,
            dtype=dtype,
            num_blocks_per_file=num_blocks_per_file,
            nvmf_targets=nvmf_targets,
        )

    @classmethod
    def _create_file(cls, file: BinaryIO, total_size_per_file: int, *,
                     get_layout: bool = False) -> Optional[List[FileExtent]]:
        '''Physically allocate file space, flush to disk and optionally detect layout.

        Args:
            file (BinaryIO): opened file handle
            total_size_per_file (int): total size to allocate for this file
            get_layout (bool): whether to collect layout metadata

        Returns:
            Optional[List[FileExtent]]
        '''
        fd = file.fileno()
        try:
            os.posix_fallocate(fd, 0, total_size_per_file) # Eager file allocation
        except OSError as e:
            raise RuntimeError(f"Failed to initialize file: {e}") from e
        file.flush()
        os.fsync(fd)

        if not get_layout:
            return None
        
        extents: List[FileExtent] = get_fm_extents(fd, max_extents=256)
        return extents

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

    @staticmethod
    def _get_md_chunk_size(md_dev: str) -> int:
        '''Get chunk size of md device. Only support RAID0.

        Args:
            md_dev (str): md device path (not partition path), e.g. /dev/md0 instead of /dev/md0p1

        Returns:
            int: chunk size in bytes
        '''
        sysfs_path = f'/sys/block/{os.path.basename(md_dev)}/md/chunk_size'
        with open(sysfs_path, 'r') as f:
            return int(f.read().strip())

    @staticmethod
    def _get_md_member_devs(md_dev: str) -> List[str]:
        '''Return member device names in md's internal order.
        WARN: 

        Args:
            md_dev (str): md device path (not partition path), e.g. /dev/md0 instead of /dev/md0p1

        Returns:
            List[str]: list of member device names
        '''
        import glob

        md_name = os.path.basename(md_dev)
        base = f"/sys/block/{md_name}/md"

        entries: Dict[int, str] = {}

        # /sys/block/md0/md/rdX symlinks
        for rd_path in glob.glob(os.path.join(base, "rd*")):
            rd_name = os.path.basename(rd_path)
            if not rd_name.startswith("rd"):
                continue
            try:
                idx = int(rd_name[2:]) # rd0 -> 0, rd1 -> 1, ...
            except ValueError:
                continue

            target = os.readlink(rd_path) # e.g. dev-nvme2n1
            # dev-nvme2n1 -> nvme2n1
            dev_name = target.split("-", 1)[1] if "-" in target else target
            entries[idx] = dev_name

        if not entries:
            # fall back to check /sys/block/md0/md/dev-nvmeXn1/slot
            for slot_path in glob.glob(os.path.join(base, "dev-nvme*n1/slot")):
                # slot_path: /sys/block/md0/md/dev-nvmeXn1/slot
                dev_dir = os.path.dirname(slot_path)
                dev_name = os.path.basename(dev_dir)[len("dev-"):] # strip dev-
                with open(slot_path, "r") as f:
                    idx = int(f.read().strip())
                entries[idx] = dev_name

        if not entries:
            raise RuntimeError(f"No md members found under {base}")

        return [dev_name for _, dev_name in sorted(entries.items())]

    @staticmethod
    def _get_backing_devs(cache_dir: List[str]) -> List[str]:
        '''
        Returns:
            List[str]: list of backing NVMe device names for given cache dir, e.g. ['nvme0n1', 'nvme1n1']
        '''
        import subprocess

        def _get_mount_source(path: str) -> Optional[str]:
            try:
                _path = os.path.realpath(path)
                cmd = ['findmnt', '-n', '-o', 'SOURCE', '--target', _path]
                # Example output:
                #     /dev/vda2
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                source = result.stdout.strip()
                # Filter out non-device mounts, e.g. overlay, tmpfs, etc.)
                if source.startswith('/dev/'):
                    return source
                return None
            except (subprocess.CalledProcessError, OSError):
                return None

        def _resolve_physical_nvme(device_path: str) -> Optional[str]:
            import re

            try:
                cmd = ['lsblk', '--inverse', '-p', '-n', '-r', '-o', 'NAME,TYPE', device_path]
                # Example output:
                #     /dev/vda2 part
                #     /dev/vda disk
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                nvme_pattern = re.compile(r'/dev/nvme\d+n\d+$') # Matches nvme0n1, ignores p1
                for line in result.stdout.splitlines():
                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    current_path, current_type = parts[0], parts[1]
                    if current_type == 'disk' and nvme_pattern.match(current_path):
                        return os.path.basename(current_path)

                return None
            except subprocess.CalledProcessError:
                return None

        results: List[str] = []
        dev_cache: Dict[str, str] = {}
        for dir in cache_dir:
            # 1. Find logical device
            logical_dev: Optional[str] = _get_mount_source(dir)
            assert logical_dev is not None, f'Failed in finding logical device for {dir}'
            # 2. Check cache
            if logical_dev in dev_cache:
                results.append(dev_cache[logical_dev])
                continue
            # 3. Resolve physical device
            physical_dev: Optional[str] = _resolve_physical_nvme(logical_dev)
            assert physical_dev is not None, f'Failed in resolving physical NVMe device for {dir}'
            dev_cache[logical_dev] = physical_dev
            results.append(physical_dev)

        return results

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
