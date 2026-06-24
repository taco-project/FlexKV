import ctypes
import mmap
import os
import weakref
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Dict, Any, BinaryIO
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

# ---------------------------------------------------------------------------
# HugePage helpers (standalone, reusable outside of BaseStorageAllocator)
# ---------------------------------------------------------------------------
DEFAULT_HUGE_PAGE_SIZE = 2 * 1024 * 1024  # 2 MiB
DEFAULT_HUGETLBFS_DIR = "/mnt/hugepages"

_MAP_SHARED = 0x01
_MAP_PRIVATE = 0x02
_MAP_ANONYMOUS = 0x20
_MAP_HUGETLB = 0x40000
_MAP_HUGE_SHIFT = 26
_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_FAILED = ctypes.c_void_p(-1).value  # (void*)-1
_HUGETLBFS_MAGIC = 0x958458F6

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.restype = ctypes.c_int
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.ftruncate.restype = ctypes.c_int
_libc.ftruncate.argtypes = [ctypes.c_int, ctypes.c_long]
_libc.close.restype = ctypes.c_int
_libc.close.argtypes = [ctypes.c_int]


class _StatFS(ctypes.Structure):
    _fields_ = [
        ("f_type", ctypes.c_long),
        ("f_bsize", ctypes.c_long),
        ("f_blocks", ctypes.c_ulong),
        ("f_bfree", ctypes.c_ulong),
        ("f_bavail", ctypes.c_ulong),
        ("f_files", ctypes.c_ulong),
        ("f_ffree", ctypes.c_ulong),
        ("f_fsid", ctypes.c_int * 2),
        ("f_namelen", ctypes.c_long),
        ("f_frsize", ctypes.c_long),
        ("f_flags", ctypes.c_long),
        ("f_spare", ctypes.c_long * 4),
    ]


_libc.statfs.restype = ctypes.c_int
_libc.statfs.argtypes = [ctypes.c_char_p, ctypes.POINTER(_StatFS)]

@dataclass
class _HugePageMapping:
    finalizer: Any
    aligned: int
    path: str | None = None


_live_hugepage_mappings: "Dict[int, _HugePageMapping]" = {}


@dataclass(frozen=True)
class HugePageTensorHandle:
    path: str
    num_elements: int
    dtype: torch.dtype
    aligned: int

    def get_tensor(self) -> torch.Tensor:
        return _materialize_shareable_hugepage_tensor(
            path=self.path,
            num_elements=self.num_elements,
            dtype=self.dtype,
            aligned=self.aligned,
        )


def _cleanup_hugepage_mapping(addr: int, aligned: int, fd: int,
                              path: str | None, data_ptr: int) -> None:
    _munmap_huge(addr, aligned)
    if fd >= 0:
        _libc.close(fd)
    if path is not None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    _live_hugepage_mappings.pop(data_ptr, None)


def _cleanup_hugepage_mmap(mm: mmap.mmap, path: str | None, data_ptr: int) -> None:
    try:
        mm.close()
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        _live_hugepage_mappings.pop(data_ptr, None)


def _statfs_type(path: str) -> int:
    statfs_buf = _StatFS()
    if _libc.statfs(path.encode(), ctypes.byref(statfs_buf)) != 0:
        err = ctypes.get_errno()
        raise RuntimeError(
            f"HugePage: statfs({path}) failed: {os.strerror(err)} (errno={err})"
        )
    return int(statfs_buf.f_type)


def _ensure_hugetlbfs_mount(mnt_dir: str) -> None:
    if not os.path.isdir(mnt_dir):
        raise RuntimeError(f"HugePage: hugetlbfs directory does not exist: {mnt_dir}")
    fs_type = _statfs_type(mnt_dir)
    if fs_type != _HUGETLBFS_MAGIC:
        raise RuntimeError(
            f"HugePage: {mnt_dir} is not a hugetlbfs mount "
            f"(f_type=0x{fs_type:x}, expected=0x{_HUGETLBFS_MAGIC:x})"
        )


def _create_hugetlbfs_file(aligned: int) -> tuple[str, int]:
    mnt_dir = os.environ.get("FLEXKV_HUGETLBFS_DIR", DEFAULT_HUGETLBFS_DIR)
    _ensure_hugetlbfs_mount(mnt_dir)
    path = os.path.join(mnt_dir, f"flexkv_hugepage_{os.getpid()}_{id(object()):x}")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o600)
    try:
        ctypes.set_errno(0)
        if _libc.ftruncate(fd, aligned) != 0:
            err = ctypes.get_errno()
            raise RuntimeError(
                f"HugePage: ftruncate({path}, {aligned}) failed: "
                f"{os.strerror(err)} (errno={err})"
            )
    except Exception:
        os.close(fd)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    return path, fd


def _wrap_mmap_tensor(mm: mmap.mmap,
                      aligned: int,
                      num_elements: int,
                      dtype: torch.dtype,
                      cleanup_path: str | None) -> torch.Tensor:
    num_bytes = num_elements * dtype.itemsize
    tensor = torch.frombuffer(mm, dtype=torch.uint8, count=num_bytes).view(dtype)[:num_elements]
    ptr = tensor.data_ptr()
    finalizer = weakref.finalize(tensor, _cleanup_hugepage_mmap, mm, cleanup_path, ptr)
    _live_hugepage_mappings[ptr] = _HugePageMapping(
        finalizer=finalizer,
        aligned=aligned,
        path=cleanup_path,
    )
    return tensor


def _materialize_shareable_hugepage_tensor(path: str,
                                           num_elements: int,
                                           dtype: torch.dtype,
                                           aligned: int) -> torch.Tensor:
    fd = os.open(path, os.O_RDWR)
    try:
        mm = mmap.mmap(
            fd,
            aligned,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(fd)
    return _wrap_mmap_tensor(mm, aligned, num_elements, dtype, cleanup_path=None)


def materialize_worker_tensor(data: Union[torch.Tensor, HugePageTensorHandle]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, HugePageTensorHandle):
        return data.get_tensor()
    raise TypeError(f"Unsupported worker tensor type: {type(data)}")


def get_worker_hugepage_handle(tensor: torch.Tensor,
                               num_elements: int,
                               dtype: torch.dtype) -> HugePageTensorHandle | None:
    mapping = _live_hugepage_mappings.get(tensor.data_ptr())
    if mapping is None or mapping.path is None:
        return None
    return HugePageTensorHandle(
        path=mapping.path,
        num_elements=num_elements,
        dtype=dtype,
        aligned=mapping.aligned,
    )


def _align_to_page(num_bytes: int, page_size_bytes: int) -> int:
    """Round *num_bytes* up to the next multiple of *page_size_bytes*."""
    return (num_bytes + page_size_bytes - 1) & ~(page_size_bytes - 1)


def _read_hugepages_free(page_size_bytes: int) -> int:
    """Return the number of free huge pages for *page_size_bytes*."""
    try:
        size_kb = page_size_bytes // 1024
        cur_kb = 0
        free = 0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("Hugepagesize:"):
                    cur_kb = int(line.split()[1])
                elif line.startswith("HugePages_Free:"):
                    free = int(line.split()[1])
        if cur_kb != size_kb:
            return 0  # wrong page size pool
        return free
    except Exception:
        return 0


def _mmap_huge(num_bytes: int, page_size_bytes: int) -> Tuple[int, int, int]:
    if num_bytes <= 0:
        raise ValueError(f"HugePage: num_bytes must be > 0, got {num_bytes}")
    if page_size_bytes <= 0 or (page_size_bytes & (page_size_bytes - 1)) != 0:
        raise ValueError(
            f"HugePage: page_size_bytes must be a power of two, got {page_size_bytes}"
        )

    aligned = _align_to_page(num_bytes, page_size_bytes)
    page_shift = page_size_bytes.bit_length() - 1

    # 1) Anonymous MAP_HUGETLB — no hugetlbfs mount needed.
    free_pages = _read_hugepages_free(page_size_bytes)
    if free_pages and aligned > free_pages * page_size_bytes:
        flexkv_logger.warning(
            f"HugePage: requested {aligned // page_size_bytes} pages "
            f"({aligned / (1024**3):.3f} GiB) but only {free_pages} free "
            f"(page_size={page_size_bytes // (1024*1024)} MiB). "
            f"The kernel may fall back to regular pages or overcommit."
        )

    ctypes.set_errno(0)
    huge_flags = _MAP_PRIVATE | _MAP_ANONYMOUS | _MAP_HUGETLB | (page_shift << _MAP_HUGE_SHIFT)
    ret = _libc.mmap(None, aligned, _PROT_READ | _PROT_WRITE, huge_flags, -1, 0)
    if ret is not None and ret != _MAP_FAILED:
        return int(ret), aligned, -1

    # 2) Fallback: file-backed hugetlbfs. Reject non-hugetlbfs mounts so we
    # never silently succeed on regular 4 KiB pages.
    fd = -1
    try:
        path, fd = _create_hugetlbfs_file(aligned)
        try:
            os.unlink(path)
        except OSError:
            pass

        ctypes.set_errno(0)
        ret = _libc.mmap(
            None,
            aligned,
            _PROT_READ | _PROT_WRITE,
            _MAP_SHARED,
            fd,
            0,
        )
        if ret is None or ret == _MAP_FAILED:
            err = ctypes.get_errno()
            raise RuntimeError(
                f"HugePage: mmap({path}, {aligned}) failed: "
                f"{os.strerror(err)} (errno={err})"
            )
        return int(ret), aligned, fd
    except Exception:  # noqa: BLE001
        if fd >= 0:
            os.close(fd)
        raise


def _munmap_huge(addr: int, length: int) -> None:
    if _libc.munmap(ctypes.c_void_p(addr), length) != 0:
        err = ctypes.get_errno()
        flexkv_logger.warning(
            f"HugePage: munmap({hex(addr)}, {length}) failed: "
            f"{os.strerror(err)} (errno={err})"
        )


def alloc_hugepage_tensor(num_elements: int,
                          dtype: torch.dtype,
                          page_size_bytes: int = DEFAULT_HUGE_PAGE_SIZE,
                          shareable: bool = False) -> torch.Tensor:
    """Allocate ``num_elements`` values of ``dtype`` on HugePage-backed memory.

    Returns a 1-D ``torch.Tensor`` that zero-copy wraps the mmap'd region.
    The tensor's ``data_ptr()`` can be passed to ``cudaHostRegister`` or to
    other RDMA-style registration APIs.

    Use ``free_hugepage_tensor(tensor)`` to explicitly release the mapping;
    otherwise it will be released when the tensor (and all references to it)
    are garbage-collected.

    Raises:
        RuntimeError: if the mmap fails or if the resulting VMA is not backed
            by huge pages of the requested size (i.e. no silent fallback).
    """
    num_bytes = num_elements * dtype.itemsize

    if shareable:
        aligned = _align_to_page(num_bytes, page_size_bytes)
        path, fd = _create_hugetlbfs_file(aligned)
        try:
            mm = mmap.mmap(
                fd,
                aligned,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(fd)
        return _wrap_mmap_tensor(mm, aligned, num_elements, dtype, cleanup_path=path)

    addr, aligned, fd = _mmap_huge(num_bytes, page_size_bytes)

    # Zero-copy wrap: build a numpy uint8 array pointing at the raw memory,
    # then view it as the requested dtype via ``torch.frombuffer``. The numpy
    # array keeps a reference (``_base_keepalive``) so Python's GC cannot free
    # the underlying bytes while the tensor is still live.
    buf_type = (ctypes.c_uint8 * aligned)
    raw = buf_type.from_address(addr)
    np_arr = np.frombuffer(raw, dtype=np.uint8, count=num_bytes)
    tensor = torch.frombuffer(np_arr, dtype=torch.uint8, count=num_bytes) \
        .view(dtype)[:num_elements]

    ptr = tensor.data_ptr()
    finalizer = weakref.finalize(tensor, _cleanup_hugepage_mapping,
                                 addr, aligned, fd, None, ptr)
    _live_hugepage_mappings[ptr] = _HugePageMapping(
        finalizer=finalizer,
        aligned=aligned,
        path=None,
    )
    return tensor


def free_hugepage_tensor(tensor: torch.Tensor) -> None:
    """Release the HugePage mapping previously created by :func:`alloc_hugepage_tensor`.

    No-op if ``tensor`` is not known to be HugePage-backed.
    The caller must ensure no other references to the tensor's memory remain
    in active use (e.g. ``cudaHostUnregister`` should be called first, and
    any Python reference to ``tensor`` should be dropped after this call).
    """
    if not isinstance(tensor, torch.Tensor):
        return
    ptr = tensor.data_ptr()
    mapping = _live_hugepage_mappings.pop(ptr, None)
    if mapping is None:
        return
    mapping.finalizer()


class HugePageAllocator(BaseStorageAllocator):
    """CPU KV-cache allocator backed by hugetlbfs HugePages.

    Unlike :class:`CPUAllocator` (which relies on ``torch.empty`` on top of 4KiB
    pages), this allocator maps a hugetlbfs file and wraps the resulting buffer
    into a 1-D ``torch.Tensor`` (zero-copy).

    Benefits:
        * Reduced TLB pressure for large KV caches (2MiB / 1GiB pages).
        * The returned tensor's ``data_ptr()`` can still be passed to
          ``cudaHostRegister`` for pinned H2D/D2H transfers.

    Prerequisites:
        * The kernel must have huge pages reserved, e.g. for 2MiB pages::

              echo N > /proc/sys/vm/nr_hugepages
              # or, per-size on recent kernels:
              echo N > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

          For 1GiB pages the kernel usually needs ``hugepagesz=1G`` at boot
          and a corresponding ``hugepages=N`` reservation.

    kwargs:
        page_size_bytes (int): Huge page size in bytes. Supported values:
            ``2 * 1024 * 1024`` (default) or ``1024 * 1024 * 1024``.
    """

    @classmethod
    def allocate(cls,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 **kwargs: Any) -> StorageHandle:
        page_size_bytes = int(kwargs.get("page_size_bytes", DEFAULT_HUGE_PAGE_SIZE))
        total_elements = layout.get_total_elements()
        element_size = dtype.itemsize

        flexkv_logger.info(
            f"HugePage allocate total_size: "
            f"{total_elements * element_size / 1024 / 1024 / 1024:.4f} GB "
            f"(page_size={page_size_bytes // (1024 * 1024)}MiB)"
        )
        try:
            physical_tensor = alloc_hugepage_tensor(
                total_elements,
                dtype,
                page_size_bytes,
                shareable=True,
            )
        except Exception as e:  # noqa: BLE001
            flexkv_logger.warning(
                f"HugePage allocation failed ({e}); falling back to regular CPU memory."
            )
            return CPUAllocator.allocate(layout, dtype, **kwargs)
        worker_data = get_worker_hugepage_handle(physical_tensor, total_elements, dtype)
        return StorageHandle(
            handle_type=AccessHandleType.TENSOR,
            data=physical_tensor,
            kv_layout=layout,
            dtype=dtype,
            worker_data=worker_data,
        )

    @classmethod
    def free(cls, accessible_handle: StorageHandle) -> None:
        if accessible_handle.handle_type != AccessHandleType.TENSOR:
            return
        tensor = accessible_handle.data
        if isinstance(tensor, torch.Tensor):
            free_hugepage_tensor(tensor)

    @classmethod
    def from_raw_data(cls,
                      data: torch.Tensor,  # type: ignore
                      layout: KVCacheLayout,
                      dtype: torch.dtype,
                      **kwargs: Any) -> StorageHandle:
        # We assume the caller already backs ``data`` with huge pages (or does
        # not care). We do not take ownership of any mmap here.
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

        if cfg_max_file_size_gb != -1:
            cfg_max_blocks_per_file = int(cfg_max_file_size_gb * 1024 * 1024 * 1024 // block_size)
        else:
            # when we don't set max_file_size_gb, we will create a file, size is exactly the required capacity
            cfg_max_blocks_per_file = total_blocks_per_device

        fsys_max_blocks_per_file = cls.get_file_size_limit(cache_dir[0]) // block_size
        num_blocks_per_file = min(fsys_max_blocks_per_file, cfg_max_blocks_per_file, total_blocks_per_device)

        num_files_per_device = (total_blocks_per_device + num_blocks_per_file - 1) // num_blocks_per_file
        real_file_size = num_blocks_per_file * block_size

        ssd_files: Dict[int, List[str]] = {}
        total_num_files = num_files_per_device * num_ssd_devices
        real_total_size = total_num_files * real_file_size
        flexkv_logger.info(f"SSD allocator creating {total_num_files} files in {cache_dir}, "
                           f"each file {real_file_size/1024/1024/1024:.2f} GB, "
                           f"total {real_total_size/1024/1024/1024:.2f} GB")
        file_count = 0
        for i in range(num_ssd_devices):
            ssd_files[i] = []
            for j in range(num_files_per_device):
                file_path = os.path.join(cache_dir[i], f"{file_prefix}_{i}_{j}.bin")
                with open(file_path, "wb+", buffering=0) as file:
                    cls._create_file(file, real_file_size)
                ssd_files[i].append(file_path)
                file_count += 1
                if file_count % max(1, total_num_files // 10) == 0 or file_count == total_num_files:
                    flexkv_logger.info(
                        f"SSD allocator progress: {file_count}/{total_num_files} files created "
                        f"({file_count * 100 // total_num_files}%)"
                    )
        flexkv_logger.info(f"SSD allocator done: {total_num_files} files in {cache_dir}, "
                           f"each file has {real_file_size/1024/1024/1024:.2f} GB, total size {real_total_size/1024/1024/1024:.2f} GB")
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
