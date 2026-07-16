import multiprocessing as mp
import os
import time
from typing import Callable, Any, Optional, Tuple, Union
from dataclasses import dataclass
import ctypes

import torch
import torch.multiprocessing.reductions as reductions
import zmq

from flexkv.common.debug import flexkv_logger


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


# CUmemFabricHandle mirrors the driver API layout: an opaque 64-byte struct.
# vLLM's CuMemAllocator (enable_sleep_mode) allocates KV cache via cuMemCreate
# + cuMemMap, which cannot be shared through the legacy cudaIpcGetMemHandle
# path. Fabric handles are byte-serializable so they travel over the same ZMQ
# transport we already use for cudaIpc handles — no auxiliary UDS channel is
# required on GPUs that support CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC.
class CUmemFabricHandle_t(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 64)]


# Load CUDA runtime library
try:
    cudart = ctypes.CDLL("libcudart.so")
except:
    try:
        cudart = ctypes.CDLL("libcudart.so.12")
    except:
        cudart = ctypes.CDLL("libcudart.so.11")

# Load CUDA driver library. CuMemAllocator's cuMemCreate + cuMemMap live in
# libcuda, not libcudart, so we need both. Setting to None on ImportError
# keeps the legacy code path working on systems without the driver stub.
try:
    libcuda = ctypes.CDLL("libcuda.so")
except OSError:
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        libcuda = None


# CUDA IPC handle size (64 bytes on Linux)
CUDA_IPC_HANDLE_SIZE = 64
CU_MEM_FABRIC_HANDLE_SIZE = 64

# CUDA runtime error codes
cudaSuccess = 0
cudaErrorInvalidValue = 11

# CUDA driver error codes / enums used by the VMM path.
CUDA_SUCCESS = 0

# cuPointerGetAttribute selectors we consume. Values match cuda.h; the
# CUDA API guarantees stability so hardcoding is safe. IS_LEGACY_CUDA_IPC_CAPABLE
# is the most authoritative "can I call cudaIpcGetMemHandle on this pointer?"
# probe — it returns 1 for cudaMalloc allocations and 0 for VMM ranges.
CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10
CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12
CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14

# CUmemAllocationHandleType bitmask values.
CU_MEM_HANDLE_TYPE_NONE = 0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
CU_MEM_HANDLE_TYPE_WIN32 = 0x2
CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x4
CU_MEM_HANDLE_TYPE_FABRIC = 0x8

# CUmemLocationType.
CU_MEM_LOCATION_TYPE_DEVICE = 1

# CUmemAccess_flags.
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

# CUmemAllocationType.
CU_MEM_ALLOCATION_TYPE_PINNED = 1

# CUmemAllocationGranularity_flags.
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemAllocationProp_st_allocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", CUmemAllocationProp_st_allocFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


# Configure argtypes/restype so ctypes marshals pointers correctly on 64-bit
# systems (the default int-return-type would truncate CUresult values above
# INT_MAX in principle; the CUDA driver stays under that but explicit types
# make the intent clear).
if libcuda is not None:
    _CUresult = ctypes.c_int
    _CUdeviceptr = ctypes.c_ulonglong  # 64-bit device address

    libcuda.cuPointerGetAttribute.restype = _CUresult
    libcuda.cuPointerGetAttribute.argtypes = [
        ctypes.c_void_p, ctypes.c_int, _CUdeviceptr
    ]
    libcuda.cuMemRetainAllocationHandle.restype = _CUresult
    libcuda.cuMemRetainAllocationHandle.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p
    ]
    libcuda.cuMemGetAllocationPropertiesFromHandle.restype = _CUresult
    libcuda.cuMemGetAllocationPropertiesFromHandle.argtypes = [
        ctypes.POINTER(CUmemAllocationProp), ctypes.c_void_p
    ]
    libcuda.cuMemExportToShareableHandle.restype = _CUresult
    libcuda.cuMemExportToShareableHandle.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong
    ]
    libcuda.cuMemImportFromShareableHandle.restype = _CUresult
    libcuda.cuMemImportFromShareableHandle.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    ]
    libcuda.cuMemAddressReserve.restype = _CUresult
    libcuda.cuMemAddressReserve.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        _CUdeviceptr, ctypes.c_ulonglong,
    ]
    libcuda.cuMemAddressFree.restype = _CUresult
    libcuda.cuMemAddressFree.argtypes = [_CUdeviceptr, ctypes.c_size_t]
    libcuda.cuMemMap.restype = _CUresult
    libcuda.cuMemMap.argtypes = [
        _CUdeviceptr, ctypes.c_size_t, ctypes.c_ulonglong,
        ctypes.c_void_p, ctypes.c_ulonglong,
    ]
    libcuda.cuMemUnmap.restype = _CUresult
    libcuda.cuMemUnmap.argtypes = [_CUdeviceptr, ctypes.c_size_t]
    libcuda.cuMemSetAccess.restype = _CUresult
    libcuda.cuMemSetAccess.argtypes = [
        _CUdeviceptr, ctypes.c_size_t,
        ctypes.POINTER(CUmemAccessDesc), ctypes.c_size_t,
    ]
    libcuda.cuMemGetAllocationGranularity.restype = _CUresult
    libcuda.cuMemGetAllocationGranularity.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(CUmemAllocationProp),
        ctypes.c_int,
    ]
    libcuda.cuMemRelease.restype = _CUresult
    libcuda.cuMemRelease.argtypes = [ctypes.c_void_p]


# FDs returned by cuMemExportToShareableHandle stay valid only while the
# exporting process holds them open. We stash them in a module-level list so
# importers that arrive later (or restart) can still pidfd_getfd them.
# List is intentionally simple (no eviction) — FlexKV registers a bounded
# number of KV cache tensors at startup and never re-exports.
_EXPORTED_VMM_FDS: list[int] = []

# Linux 5.6+ syscalls used for cross-process FD passing without SCM_RIGHTS.
# Numbers are architecture-specific; these are for x86_64 (the FlexKV
# supported target). ppc64/aarch64 add-ons can extend this dict later.
_SYS_pidfd_open = 434
_SYS_pidfd_getfd = 438

try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.syscall.restype = ctypes.c_long
    _libc.syscall.argtypes = None  # varargs; caller supplies proper types
except OSError:
    _libc = None


def _pidfd_open(pid: int) -> int:
    if _libc is None:
        raise RuntimeError("libc not available for pidfd_open syscall")
    ret = _libc.syscall(
        ctypes.c_long(_SYS_pidfd_open), ctypes.c_int(pid), ctypes.c_uint(0)
    )
    if ret < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"pidfd_open({pid}) failed: errno={err}")
    return int(ret)


def _pidfd_getfd(pidfd: int, target_fd: int) -> int:
    if _libc is None:
        raise RuntimeError("libc not available for pidfd_getfd syscall")
    ret = _libc.syscall(
        ctypes.c_long(_SYS_pidfd_getfd),
        ctypes.c_int(pidfd),
        ctypes.c_int(target_fd),
        ctypes.c_uint(0),
    )
    if ret < 0:
        err = ctypes.get_errno()
        raise OSError(
            err,
            f"pidfd_getfd(pidfd={pidfd}, target_fd={target_fd}) failed: "
            f"errno={err}",
        )
    return int(ret)


# --- SCM_RIGHTS FD-serving side channel -------------------------------------
#
# CUDA POSIX-FD shareable handles reference a driver-specific kernel object
# accessed via ``/dev/nvidiactl``. They cannot be duplicated by opening
# ``/proc/<pid>/fd/<n>`` (which gives you a fresh ``/dev/nvidiactl`` handle,
# not a copy of the underlying CUDA resource) and cannot be duplicated via
# ``pidfd_getfd`` unless the caller has ``CAP_SYS_PTRACE`` — which Docker
# drops from its default profile. The only portable way to move them across
# process boundaries is SCM_RIGHTS on an ``AF_UNIX`` socket.
#
# To keep the fix confined to ``memory_handle.py`` (no changes needed to the
# KVManager / KVTPClient wire format), each exporter process spins up a
# background thread that accepts connections on a well-known UDS path
# derived from its PID. Importers include the exporter PID and an
# opaque FD key in the 64-byte VMM handle payload; on ``get_tensor`` they
# connect to the UDS and receive the FD via SCM_RIGHTS. FDs stay registered
# for the lifetime of the exporter process (KV cache tensors are permanent
# from FlexKV's perspective) so there is no unregistration API to design.

import socket
import threading
import array

_VMM_FD_SERVER_STATE: dict = {
    "lock": threading.Lock(),
    "sock": None,      # socket.socket
    "path": None,      # str
    "thread": None,    # threading.Thread
    "fd_by_key": {},   # int -> int (key -> fd we exported)
    "next_key": 0,
}


def _vmm_fd_server_path(pid: Optional[int] = None) -> str:
    if pid is None:
        pid = os.getpid()
    # ``FLEXKV_VMM_FD_SOCK_DIR`` lets deployments override the location for
    # sandboxed environments where /tmp is restricted. Default /tmp keeps
    # the naming consistent with FlexKV's other ipc:// paths.
    root = os.environ.get("FLEXKV_VMM_FD_SOCK_DIR", "/tmp")
    return os.path.join(root, f"flexkv_vmm_fd_{pid}.sock")


def _vmm_fd_server_ensure_started() -> None:
    """Idempotently start the SCM_RIGHTS FD-serving thread for this process."""
    state = _VMM_FD_SERVER_STATE
    with state["lock"]:
        if state["sock"] is not None:
            return
        path = _vmm_fd_server_path()
        # Best-effort cleanup of a stale socket left by a crashed prior run
        # under the same PID (unlikely but possible after PID recycling).
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        # Restrict access: only the owning UID should be able to obtain FDs.
        # ``chmod 0600`` blocks other users on the host from stealing KV cache
        # buffers even if they know the socket path.
        os.chmod(path, 0o600)
        srv.listen(16)
        state["sock"] = srv
        state["path"] = path

        thread = threading.Thread(
            target=_vmm_fd_server_loop,
            name="flexkv-vmm-fd-server",
            daemon=True,
        )
        state["thread"] = thread
        thread.start()
        flexkv_logger.info(
            f"[VMM] FD-serving thread started at {path} (SCM_RIGHTS)."
        )


def _vmm_fd_server_loop() -> None:
    """Accept-and-serve loop: read a 4-byte key, reply with the matching FD."""
    state = _VMM_FD_SERVER_STATE
    srv = state["sock"]
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        try:
            raw = conn.recv(4)
            if len(raw) != 4:
                conn.close()
                continue
            key = int.from_bytes(raw, "little", signed=False)
            fd = state["fd_by_key"].get(key)
            if fd is None:
                # Signal "unknown key" by sending a single zero byte with
                # no ancillary FD.
                conn.sendmsg([b"\x00"])
            else:
                conn.sendmsg(
                    [b"\x01"],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                      array.array("i", [fd]))],
                )
        except Exception as exc:
            flexkv_logger.warning(f"[VMM] FD-serving loop error: {exc}")
        finally:
            conn.close()


def _vmm_fd_server_register(fd: int) -> int:
    """Register ``fd`` for later retrieval and return its opaque key."""
    _vmm_fd_server_ensure_started()
    state = _VMM_FD_SERVER_STATE
    with state["lock"]:
        key = state["next_key"]
        state["next_key"] = (key + 1) & 0xFFFFFFFF
        state["fd_by_key"][key] = fd
        return key


def _fetch_remote_fd(pid: int, key: int) -> int:
    """Connect to the exporter's SCM_RIGHTS server and dup the target FD."""
    path = _vmm_fd_server_path(pid)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"VMM FD server socket not found for pid {pid} at {path}. "
            f"The exporter process may have crashed or been misconfigured."
        )
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
        sock.sendall(int(key).to_bytes(4, "little", signed=False))
        msg, ancdata, flags, _ = sock.recvmsg(
            1, socket.CMSG_LEN(4 * array.array("i").itemsize)
        )
        if not msg:
            raise RuntimeError("empty response from VMM FD server")
        if msg[0] == 0:
            raise KeyError(
                f"exporter (pid {pid}) has no FD registered under key {key}"
            )
        if len(ancdata) != 1:
            raise RuntimeError(
                f"expected exactly one SCM_RIGHTS cmsg, got {len(ancdata)}"
            )
        _, _, data = ancdata[0]
        fds = array.array("i")
        fds.frombytes(data[:fds.itemsize])
        return int(fds[0])
    finally:
        sock.close()


def _pack_vmm_fd(pid: int, key: int) -> bytes:
    """Encode ``(exporter_pid, fd_key)`` in the 64-byte handle envelope.

    ``fd_key`` is the opaque token issued by _vmm_fd_server_register, not
    the raw FD — the importer swaps it for a real FD by talking to the
    exporter's SCM_RIGHTS server.
    """
    buf = bytearray(64)
    buf[0:4] = int(pid).to_bytes(4, "little", signed=False)
    buf[4:8] = int(key).to_bytes(4, "little", signed=False)
    return bytes(buf)


def _unpack_vmm_fd(handle_bytes: bytes) -> Tuple[int, int]:
    pid = int.from_bytes(handle_bytes[0:4], "little", signed=False)
    key = int.from_bytes(handle_bytes[4:8], "little", signed=False)
    return pid, key


def _check_cu(where: str, result: int) -> None:
    """Raise on non-CUDA_SUCCESS return codes from the driver API."""
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"{where} failed with CUresult={result}")


def _is_vmm_pointer(data_ptr: int) -> bool:
    """Return True when ``data_ptr`` points into a cuMemCreate/cuMemMap range.

    We query ``CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE`` which is
    documented as "1 if this pointer maps to an allocation suitable for
    cudaIpcGetMemHandle, 0 otherwise". cudaMalloc returns 1; VMM ranges
    return 0. Inverting the answer gives us the VMM discriminator we want.
    If the attribute query itself fails we conservatively answer False so
    the caller falls back to the existing cudaIpc path (which then errors
    out cleanly on a truly non-shareable allocation).
    """
    if libcuda is None:
        return False
    legacy_ok = ctypes.c_uint(0)
    result = libcuda.cuPointerGetAttribute(
        ctypes.byref(legacy_ok),
        CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
        ctypes.c_ulonglong(data_ptr),
    )
    if result != CUDA_SUCCESS:
        return False
    return legacy_ok.value == 0


@dataclass
class TensorSharedHandle:
    rebuild_func: Optional[Callable]
    rebuild_args: Optional[Tuple[Any]]
    device: torch.device
    # For direct CUDA IPC
    use_direct_ipc: bool = False
    ipc_handle: Optional[bytes] = None
    tensor_shape: Optional[Tuple[int, ...]] = None
    tensor_dtype: Optional[torch.dtype] = None
    tensor_numel: Optional[int] = None
    offset: int = 0

    # VMM (cuMemCreate/cuMemMap) sharing metadata. Populated only when the
    # source tensor lives in a virtual-memory allocation such as vLLM's
    # CuMemAllocator (enable_sleep_mode). ``handle_type`` disambiguates which
    # importer path to run on ``get_tensor``:
    #   'torch_reduce'  – legacy PyTorch reduce_tensor  (use_direct_ipc=False)
    #   'cuda_ipc'      – legacy cudaIpcGetMemHandle    (use_direct_ipc=True)
    #   'vmm_fabric'    – cuMemExport(FABRIC) bytes, no auxiliary channel
    #   'vmm_posix_fd'  – FD-based sharing; requires an out-of-band UDS
    #                     (not yet implemented — see _init_from_tensor)
    handle_type: str = "torch_reduce"
    vmm_allocation_size: int = 0
    vmm_granularity: int = 0

    def __init__(
        self,
        data: Union[torch.Tensor, bytes],
        device_id: int = -1,
        force_direct_ipc: bool = False,
        *,
        tensor_shape: Optional[Tuple[int, ...]] = None,  # only used when data is bytes
        tensor_dtype: Optional[
            Union[torch.dtype, str]
        ] = None,  # only used when data is bytes
        offset: int = 0,  # offset in bytes from base pointer (for memory pool allocations)
        handle_type: Optional[str] = None,  # only used when data is bytes
        vmm_allocation_size: int = 0,       # only used when data is bytes + VMM
        vmm_granularity: int = 0,           # only used when data is bytes + VMM
    ):
        """
        Now we support three ways to construct TensorSharedHandle:
        If data is a tensor that is managed by torch, we will use the reduce_tensor method to export the TensorSharedHandle.
        If data is a tensor that is allocated by cudamalloc, we will use the cudaIpcGetMemHandle method to export the TensorSharedHandle.
        If data is a tensor that is allocated by CUDA VMM (cuMemCreate/cuMemMap — used by vLLM's CuMemAllocator when
        enable_sleep_mode is on), we will use cuMemExportToShareableHandle with a fabric handle so the resulting bytes
        can still be sent over ZMQ. When fabric is unavailable we fall back to POSIX FDs, which require an out-of-band
        channel — this path is currently not wired up and raises a clear error instead of silently corrupting state.
        If data is bytes-like, it means the memory has already been shared by CUDA IPC / VMM, we will skip the export
        process to construct the TensorSharedHandle.
        """

        self.use_direct_ipc = False
        self.ipc_handle = None
        self.tensor_shape = None
        self.tensor_dtype = None
        self.tensor_numel = None
        self.handle_type = "torch_reduce"
        self.vmm_allocation_size = 0
        self.vmm_granularity = 0

        if isinstance(data, torch.Tensor):
            self._init_from_tensor(data, device_id, force_direct_ipc)
            return

        elif isinstance(data, bytes):
            self._init_from_ipc_handle(
                bytes(data), device_id, tensor_shape, tensor_dtype,
                offset=offset,
                handle_type=handle_type,
                vmm_allocation_size=vmm_allocation_size,
                vmm_granularity=vmm_granularity,
            )
            return
        else:
            raise ValueError(
                f"Unsupported data type {type(data)} for TensorSharedHandle, expected torch.Tensor / bytes-like"
            )

    def _init_from_tensor(
        self,
        tensor: torch.Tensor,
        device_id: int,
        force_direct_ipc: bool,
    ) -> None:
        if not tensor.is_cuda:
            raise ValueError("Only support CUDA tensor sharing")

        # VMM allocations (vLLM CuMemAllocator etc.) cannot be exported via
        # cudaIpcGetMemHandle at all — the legacy IPC API rejects them. We
        # must detect this up front and route to the driver-API export path,
        # otherwise both existing legacy attempts below would fail and we'd
        # surface a confusing "Both PyTorch and direct CUDA IPC export failed"
        # error message that gives no hint that sleep_mode is the culprit.
        if libcuda is not None and _is_vmm_pointer(tensor.data_ptr()):
            self._export_vmm(tensor, device_id)
            return

        if not force_direct_ipc:
            ## Try PyTorch's built-in method first
            try:
                (
                    self.rebuild_func,
                    self.rebuild_args,
                    tensor_device_id,
                ) = self._export_tensor_handle(tensor)
                if device_id == -1:
                    self.device = tensor_device_id
                else:
                    self.device = torch.device(f"cuda:{device_id}")
                    tmp_list = list(self.rebuild_args)
                    tmp_list[6] = device_id
                    self.rebuild_args = tuple(tmp_list)
                self.handle_type = "torch_reduce"
                return
            except RuntimeError as e:
                flexkv_logger.warning(f"PyTorch CUDA IPC export failed: {e}")
                flexkv_logger.info("Attempting direct CUDA IPC export...")

        try:
            ## Try direct CUDA IPC export
            self.ipc_handle = self._export_cuda_ipc_handle(tensor)
            self.use_direct_ipc = True
            self.handle_type = "cuda_ipc"
            self.tensor_shape = tuple(tensor.shape)
            self.tensor_dtype = tensor.dtype
            self.tensor_numel = tensor.numel()
            self.device = (
                tensor.device if device_id == -1 else torch.device(f"cuda:{device_id}")
            )
            self.rebuild_func = None
            self.rebuild_args = None
            self.offset = 0    ## only used when constructing from direct ipc handle
            flexkv_logger.info(
                f"Tensor exported via direct CUDA IPC: tensor.device={tensor.device}, passed device_id={device_id}, final self.device={self.device}"
            )
        except Exception as e:
            raise RuntimeError(f"Both PyTorch and direct CUDA IPC export failed: {e}")

    def _export_vmm(self, tensor: torch.Tensor, device_id: int) -> None:
        """Export a VMM-backed tensor via cuMemExportToShareableHandle.

        Prefers CU_MEM_HANDLE_TYPE_FABRIC because the resulting 64-byte
        handle is byte-serializable and can travel over the existing ZMQ
        transport unchanged. Only falls back to POSIX FD when the
        allocation explicitly does not permit fabric, and even then only
        emits a clear NotImplementedError since FD passing requires an
        auxiliary UDS channel that the FlexKV control plane does not
        currently expose.
        """
        assert libcuda is not None
        data_ptr = tensor.data_ptr()

        base_ptr = ctypes.c_ulonglong()
        alloc_size = ctypes.c_size_t()
        res = libcuda.cuPointerGetAttribute(
            ctypes.byref(base_ptr),
            CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
            ctypes.c_ulonglong(data_ptr),
        )
        _check_cu("cuPointerGetAttribute(RANGE_START_ADDR)", res)
        res = libcuda.cuPointerGetAttribute(
            ctypes.byref(alloc_size),
            CU_POINTER_ATTRIBUTE_RANGE_SIZE,
            ctypes.c_ulonglong(data_ptr),
        )
        _check_cu("cuPointerGetAttribute(RANGE_SIZE)", res)

        mem_handle = ctypes.c_void_p()
        res = libcuda.cuMemRetainAllocationHandle(
            ctypes.byref(mem_handle), ctypes.c_void_p(base_ptr.value)
        )
        _check_cu("cuMemRetainAllocationHandle", res)

        try:
            prop = CUmemAllocationProp()
            res = libcuda.cuMemGetAllocationPropertiesFromHandle(
                ctypes.byref(prop), mem_handle
            )
            _check_cu("cuMemGetAllocationPropertiesFromHandle", res)

            allowed = int(prop.requestedHandleTypes)
            if allowed & CU_MEM_HANDLE_TYPE_FABRIC:
                fabric = CUmemFabricHandle_t()
                res = libcuda.cuMemExportToShareableHandle(
                    ctypes.byref(fabric),
                    mem_handle,
                    ctypes.c_int(CU_MEM_HANDLE_TYPE_FABRIC),
                    ctypes.c_ulonglong(0),
                )
                _check_cu("cuMemExportToShareableHandle(FABRIC)", res)
                self.ipc_handle = ctypes.string_at(
                    ctypes.byref(fabric), CU_MEM_FABRIC_HANDLE_SIZE
                )
                self.handle_type = "vmm_fabric"
            elif allowed & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR:
                # POSIX FD path — the driver hands us an integer FD in the
                # exporting process. CUDA POSIX FDs reference an internal
                # kernel object (backed by /dev/nvidiactl) and cannot be
                # cloned by re-opening ``/proc/<pid>/fd/<n>`` or by
                # ``pidfd_getfd`` without CAP_SYS_PTRACE. To move them
                # cross-process we register the FD with a small
                # SCM_RIGHTS-serving thread (see _vmm_fd_server_*) and
                # ship (pid, key) as bytes; the importer connects to the
                # UDS, sends the key, and receives the FD back. The
                # exporter keeps its FD open for the lifetime of the
                # process (see _EXPORTED_VMM_FDS below) so importers may
                # arrive arbitrarily later.
                fd_out = ctypes.c_int(-1)
                res = libcuda.cuMemExportToShareableHandle(
                    ctypes.byref(fd_out),
                    mem_handle,
                    ctypes.c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                    ctypes.c_ulonglong(0),
                )
                _check_cu("cuMemExportToShareableHandle(POSIX_FD)", res)
                _EXPORTED_VMM_FDS.append(int(fd_out.value))
                key = _vmm_fd_server_register(int(fd_out.value))
                self.ipc_handle = _pack_vmm_fd(os.getpid(), key)
                self.handle_type = "vmm_posix_fd"
            else:
                raise RuntimeError(
                    "VMM allocation has no shareable handle type set "
                    f"(requestedHandleTypes=0x{allowed:x})."
                )

            granularity = ctypes.c_size_t()
            res = libcuda.cuMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(prop),
                ctypes.c_int(CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            )
            _check_cu("cuMemGetAllocationGranularity", res)

            self.use_direct_ipc = True  # importer must go through get_tensor
            self.tensor_shape = tuple(tensor.shape)
            self.tensor_dtype = tensor.dtype
            self.tensor_numel = tensor.numel()
            self.device = (
                tensor.device
                if device_id == -1
                else torch.device(f"cuda:{device_id}")
            )
            self.rebuild_func = None
            self.rebuild_args = None
            self.offset = int(data_ptr - base_ptr.value)
            self.vmm_allocation_size = int(alloc_size.value)
            self.vmm_granularity = int(granularity.value)
            flexkv_logger.info(
                f"Tensor exported via CUDA VMM ({self.handle_type}): "
                f"device={self.device}, base=0x{base_ptr.value:x}, "
                f"offset={self.offset}, size={self.vmm_allocation_size}, "
                f"granularity={self.vmm_granularity}"
            )
        finally:
            libcuda.cuMemRelease(mem_handle)

    def _init_from_ipc_handle(
        self,
        ipc_handle: Optional[bytes],
        device_id: int,
        tensor_shape: Optional[Tuple[int, ...]],
        tensor_dtype: Optional[Union[torch.dtype, str]],
        offset: int = 0,
        handle_type: Optional[str] = None,
        vmm_allocation_size: int = 0,
        vmm_granularity: int = 0,
    ) -> None:
        if ipc_handle is None:
            raise ValueError("ipc_handle is required when constructing from external handle")
        if tensor_shape is None:
            raise ValueError("tensor_shape is required when constructing from external handle")
        if tensor_dtype is None:
            raise ValueError("tensor_dtype is required when constructing from external handle")
        if device_id == -1:
            raise ValueError("device_id must be provided when constructing from external handle")

        resolved_shape = tuple(int(dim) for dim in tensor_shape)
        resolved_dtype = self._ensure_torch_dtype(tensor_dtype)

        self.use_direct_ipc = True # must set to true when constructing from direct ipc handle
        self.ipc_handle = bytes(ipc_handle)
        self.tensor_shape = resolved_shape
        self.tensor_dtype = resolved_dtype
        numel = 1
        for dim in resolved_shape:
            numel *= dim
        self.tensor_numel = numel
        self.device = torch.device(f"cuda:{device_id}")
        self.rebuild_func = None
        self.rebuild_args = None
        self.offset = offset
        # Default to cuda_ipc so pre-existing callers that construct from raw
        # bytes without specifying handle_type keep the legacy behavior.
        self.handle_type = handle_type or "cuda_ipc"
        self.vmm_allocation_size = int(vmm_allocation_size)
        self.vmm_granularity = int(vmm_granularity)
        if self.handle_type in ("vmm_fabric", "vmm_posix_fd") and (
            self.vmm_allocation_size <= 0 or self.vmm_granularity <= 0
        ):
            raise ValueError(
                "VMM handle_type requires vmm_allocation_size and "
                "vmm_granularity to be set (both must be positive)."
            )
        
  
        # flexkv_logger.info(
        #     f"TensorSharedHandle constructed from external IPC handle {self.ipc_handle.hex()} on device {self.device} \
        #         with shape {self.tensor_shape} and dtype {self.tensor_dtype}, ptr offset={offset}"
        # )

    @staticmethod
    def _ensure_torch_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            normalized = dtype.strip().lower()
            mapping = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "fp8": torch.float8_e4m3fn,
                "e4m3": torch.float8_e4m3fn,
                "float8": torch.float8_e4m3fn,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "int8": torch.int8,
                "uint8": torch.uint8,
                "int16": torch.int16,
                "int32": torch.int32,
                "int64": torch.int64,
                "bool": torch.bool,
            }
            if normalized in mapping:
                return mapping[normalized]
        raise ValueError(f"Unsupported tensor dtype: {dtype}")

    def get_tensor(self) -> torch.Tensor:
        if self.handle_type == "vmm_fabric":
            return self._import_vmm_handle(
                self.ipc_handle,
                self.tensor_shape,
                self.tensor_dtype,
                self.device,
                offset=self.offset,
                allocation_size=self.vmm_allocation_size,
                granularity=self.vmm_granularity,
                handle_type=CU_MEM_HANDLE_TYPE_FABRIC,
            )
        if self.handle_type == "vmm_posix_fd":
            return self._import_vmm_handle(
                self.ipc_handle,
                self.tensor_shape,
                self.tensor_dtype,
                self.device,
                offset=self.offset,
                allocation_size=self.vmm_allocation_size,
                granularity=self.vmm_granularity,
                handle_type=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            )
        if self.use_direct_ipc:
            return self._import_cuda_ipc_handle(
                self.ipc_handle, self.tensor_shape, self.tensor_dtype,
                self.device, offset=self.offset,
            )
        return self._import_tensor_handle(
            self.rebuild_func, self.rebuild_args, self.device
        )

    ## Import CUDA VMM allocation
    @staticmethod
    def _import_vmm_handle(
        shareable_bytes: bytes,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        offset: int,
        allocation_size: int,
        granularity: int,
        handle_type: int,
    ) -> torch.Tensor:
        """Rehydrate a VMM allocation exported via cuMemExportToShareableHandle.

        The exporter side sent us just the 64-byte fabric handle. Here we
        cuMemImportFromShareableHandle -> cuMemAddressReserve ->
        cuMemMap -> cuMemSetAccess, then wrap the virtual address in a
        zero-copy torch tensor. The imported allocation stays mapped for
        the lifetime of the process; freeing it would invalidate every
        tensor we handed out, and FlexKV's KVManager already owns the
        registration lifetime.
        """
        if libcuda is None:
            raise RuntimeError(
                "libcuda not available; cannot import a VMM shareable handle."
            )

        if not torch.cuda.is_initialized():
            torch.cuda.init()
        device_id = device.index if device.index is not None else 0
        torch.cuda.set_device(device_id)
        _ = torch.zeros(1, device=device)  # force context

        # Both fabric and POSIX-FD paths reuse the same import primitive; the
        # only difference is what we point osHandle at.
        mem_handle = ctypes.c_void_p()
        if handle_type == CU_MEM_HANDLE_TYPE_FABRIC:
            fabric = CUmemFabricHandle_t()
            ctypes.memmove(
                ctypes.byref(fabric), shareable_bytes,
                CU_MEM_FABRIC_HANDLE_SIZE,
            )
            res = libcuda.cuMemImportFromShareableHandle(
                ctypes.byref(mem_handle),
                ctypes.byref(fabric),
                ctypes.c_int(handle_type),
            )
            _check_cu("cuMemImportFromShareableHandle(FABRIC)", res)
        elif handle_type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR:
            pid, key = _unpack_vmm_fd(shareable_bytes)
            # If we're being replayed inside the same process (unlikely
            # since CUDA rejects same-process import, but we handle it
            # anyway for defense in depth) look up the FD directly.
            if pid == os.getpid():
                local_fd = _VMM_FD_SERVER_STATE["fd_by_key"].get(key)
                if local_fd is None:
                    raise KeyError(
                        f"no VMM FD registered locally under key {key}"
                    )
                close_local_fd = False
            else:
                local_fd = _fetch_remote_fd(pid, key)
                close_local_fd = True
            try:
                link_target = os.readlink(f"/proc/self/fd/{local_fd}")
            except OSError:
                link_target = "?"
            flexkv_logger.info(
                f"[VMM import] pid={pid} key={key} -> local_fd={local_fd} "
                f"({link_target})"
            )
            try:
                # NCCL and CUDA sample code (vectorAddMMAP, cudaVMM_p2p)
                # pass the FD directly as ``osHandle`` (cast to ``void*``),
                # NOT a pointer to an int. This is the value-shaped
                # convention documented by CUDA for the POSIX FD handle
                # type: for FABRIC osHandle points to a 64-byte struct,
                # for POSIX FD osHandle IS the FD.
                res = libcuda.cuMemImportFromShareableHandle(
                    ctypes.byref(mem_handle),
                    ctypes.c_void_p(local_fd),
                    ctypes.c_int(handle_type),
                )
                _check_cu(
                    "cuMemImportFromShareableHandle(POSIX_FD)", res
                )
            finally:
                # cuMemImportFromShareableHandle dups the FD internally, so
                # we can close our local copy once the driver has consumed
                # it. Skip if the FD was already owned by this process.
                if close_local_fd:
                    os.close(local_fd)
        else:
            raise ValueError(f"unsupported vmm handle_type {handle_type}")

        try:
            # Reserve a VA range of exactly the allocation size, aligned to
            # the exporter's granularity so cuMemMap will accept it.
            va = ctypes.c_ulonglong(0)
            res = libcuda.cuMemAddressReserve(
                ctypes.byref(va),
                ctypes.c_size_t(allocation_size),
                ctypes.c_size_t(granularity),
                ctypes.c_ulonglong(0),
                ctypes.c_ulonglong(0),
            )
            _check_cu("cuMemAddressReserve", res)

            res = libcuda.cuMemMap(
                ctypes.c_ulonglong(va.value),
                ctypes.c_size_t(allocation_size),
                ctypes.c_ulonglong(0),
                mem_handle,
                ctypes.c_ulonglong(0),
            )
            _check_cu("cuMemMap", res)

            # Grant this device read/write access to the mapped range.
            access = CUmemAccessDesc()
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = device_id
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            res = libcuda.cuMemSetAccess(
                ctypes.c_ulonglong(va.value),
                ctypes.c_size_t(allocation_size),
                ctypes.byref(access),
                ctypes.c_size_t(1),
            )
            _check_cu("cuMemSetAccess", res)
        except Exception:
            libcuda.cuMemRelease(mem_handle)
            raise
        # NOTE: the mem_handle refcount is retained by cuMemMap for as long
        # as the mapping lives, so we can safely release our own ref now.
        libcuda.cuMemRelease(mem_handle)

        data_ptr = va.value + int(offset)
        flexkv_logger.info(
            f"Imported VMM tensor: device={device}, base=0x{va.value:x}, "
            f"offset={offset}, size={allocation_size}"
        )
        return TensorSharedHandle._create_tensor_from_cuda_ptr(
            data_ptr, shape, dtype, device
        )

    ## Export tensor handle
    @staticmethod
    def _export_tensor_handle(
        tensor: torch.Tensor,
    ) -> Tuple[Callable, Tuple[Any], torch.device]:
        device = tensor.device
        rebuild_func, rebuild_args = reductions.reduce_tensor(tensor)
        return rebuild_func, rebuild_args, device

    ## Import tensor handle
    @staticmethod
    def _import_tensor_handle(
        rebuild_func: Callable, rebuild_args: Tuple[Any], device: torch.device
    ) -> torch.Tensor:
        try:
            tensor = rebuild_func(*rebuild_args)
            assert isinstance(tensor, torch.Tensor)

            if tensor.device != device:
                flexkv_logger.warning(
                    f"Tensor device {tensor.device} is not the same as the target device {device}"
                )
                tensor = tensor.to(device)

            return tensor

        except Exception as e:
            flexkv_logger.error("Import tensor handle failed: %s", e)
            return torch.empty(0)

    @staticmethod
    def _create_tensor_from_cuda_ptr(
        data_ptr: int, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device, strides: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Helper function to create a PyTorch tensor from a CUDA memory pointer.
        
        This function handles the special case of bfloat16 by using uint16 as an intermediate
        type, since PyTorch's __cuda_array_interface__ doesn't support "<e" typestr directly.
        
        Args:
            data_ptr: CUDA memory pointer (integer address)
            shape: Tensor shape
            dtype: PyTorch dtype
            device: Target CUDA device
            strides: Optional strides (None for C-contiguous)
        
        Returns:
            PyTorch tensor pointing to the CUDA memory (zero-copy)
        """
        # Typestr mapping for __cuda_array_interface__
        TYPESTR_MAP = {
            torch.float32: "<f4",
            torch.float64: "<f8",
            torch.float16: "<f2",
            torch.int32: "<i4",
            torch.int64: "<i8",
            torch.int16: "<i2",
            torch.uint8: "|u1",
            torch.int8: "|i1",
            torch.bool: "|b1",
            torch.uint16: "<u2"
        }
        
        # For bfloat16, PyTorch's __cuda_array_interface__ doesn't support "<e" directly.
        # We need to use uint16 ("<u2") and then view as bfloat16 to get correct data.
        # This is a zero-copy operation - view() only changes the type interpretation.
        if dtype == torch.bfloat16:
            class CudaArrayInterface:
                def __init__(self, ptr, shape, strides=None):
                    self.__cuda_array_interface__ = {
                        "data": (ptr, False),  # (data_ptr, read_only)
                        "shape": tuple(shape),
                        "typestr": "<u2",  # uint16 (bfloat16 is stored as 16-bit)
                        "version": 3,
                        "strides": strides,  # None for C-contiguous
                        "descr": [("", "")],
                    }
            
            cuda_interface = CudaArrayInterface(data_ptr, shape, strides)
            # Create as uint16 first, then view as bfloat16 (zero-copy type reinterpretation)
            tensor_u16 = torch.as_tensor(cuda_interface, dtype=torch.uint16, device=device)
            return tensor_u16.view(torch.bfloat16)

        elif hasattr(torch, 'float8_e4m3fn') and dtype == torch.float8_e4m3fn:
            class CudaArrayInterface:
                def __init__(self, ptr, shape, strides=None):
                    self.__cuda_array_interface__ = {
                        "data": (ptr, False),
                        "shape": tuple(shape),
                        "typestr": "|u1",  # uint8，跳过不支持的 |f1 
                        "version": 3,
                        "strides": strides,
                        "descr": [("", "")],
                    }
            cuda_interface = CudaArrayInterface(data_ptr, shape, strides)
            # 先作为 uint8 认领，再 view 为 fp8
            return torch.as_tensor(cuda_interface, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
        else:
            # For other dtypes, use standard typestr mapping
            if dtype not in TYPESTR_MAP:
                raise ValueError(f"Unsupported dtype for CUDA pointer: {dtype}")
            
            class CudaArrayInterface:
                def __init__(self, ptr, shape, typestr, strides=None):
                    self.__cuda_array_interface__ = {
                        "data": (ptr, False),  # (data_ptr, read_only)
                        "shape": tuple(shape),
                        "typestr": typestr,
                        "version": 3,
                        "strides": strides,  # None for C-contiguous
                        "descr": [("", "")],
                    }
            
            flexkv_logger.debug(f"Creating {dtype} tensor from CUDA ptr")
            cuda_interface = CudaArrayInterface(data_ptr, shape, TYPESTR_MAP[dtype], strides)
            return torch.as_tensor(cuda_interface, dtype=dtype, device=device)

    ## Export CUDA IPC handle
    @staticmethod
    def _export_cuda_ipc_handle(tensor: torch.Tensor) -> bytes:
        """
        Use CUDA IPC API to export the tensor's IPC handle
        """
        # Get device pointer
        data_ptr = tensor.data_ptr()
        device = tensor.device

        flexkv_logger.debug(
            f"Exporting CUDA IPC handle: device={device}, data_ptr={hex(data_ptr)}"
        )

        # Ensure we're on the correct device
        torch.cuda.set_device(device)

        # Create IPC handle buffer
        # ipc_handle = ctypes.create_string_buffer(CUDA_IPC_HANDLE_SIZE)
        ipc_handle = cudaIpcMemHandle_t()

        # Call cudaIpcGetMemHandle
        result = cudart.cudaIpcGetMemHandle(
            ctypes.byref(ipc_handle), ctypes.c_void_p(data_ptr)
        )

        if result != cudaSuccess:
            error_msg = f"cudaIpcGetMemHandle failed with error code {result} for device {device}, ptr={hex(data_ptr)}"
            flexkv_logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Return handle as bytes
        # handle_bytes = bytes(ipc_handle.raw)
        handle_bytes = ctypes.string_at(ctypes.byref(ipc_handle), 64)
        flexkv_logger.debug(
            f"IPC handle exported successfully, first 16 bytes: {handle_bytes.hex()}"
        )
        return handle_bytes

    ## Import CUDA IPC handle
    @staticmethod
    def _import_cuda_ipc_handle(
        ipc_handle: bytes,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Using CUDA IPC API to import the tensor from the IPC handle
        
        Args:
            ipc_handle: CUDA IPC memory handle (bytes)
            shape: Tensor shape
            dtype: Tensor dtype
            device: Target CUDA device
            offset: Offset in bytes from the base pointer (for memory pool allocations)
        """
        # Ensure CUDA is initialized in this process
        if not torch.cuda.is_initialized():
            flexkv_logger.info("Initializing CUDA in subprocess")
            torch.cuda.init()

        # Set device and create a dummy tensor to ensure context is created
        device_id = device.index if device.index is not None else 0
        torch.cuda.set_device(device_id)

        # Force CUDA context creation
        _ = torch.zeros(1, device=device)

        # Create IPC handle buffer
        ipc_handle_buf = ctypes.create_string_buffer(ipc_handle, CUDA_IPC_HANDLE_SIZE)

        # Rebuild IPC handle
        handle = cudaIpcMemHandle_t()
        ctypes.memmove(ctypes.byref(handle), ipc_handle, 64)

        # Open IPC memory handle to get base pointer
        base_ptr = ctypes.c_void_p()
        result = cudart.cudaIpcOpenMemHandle(
            ctypes.byref(base_ptr),
            handle,
            ctypes.c_int(1),  # cudaIpcMemLazyEnablePeerAccess = 1
        )
        # Print GPU memory address for comparison with C++ side

        if result != cudaSuccess:
            error_msg = f"cudaIpcOpenMemHandle failed with error code {result} for device {device_id}"
            flexkv_logger.error(error_msg)
            flexkv_logger.error(f"IPC handle bytes (full): {ipc_handle.hex()}")
            flexkv_logger.error(f"Current CUDA device: {torch.cuda.current_device()}")
            flexkv_logger.error(f"Target device: {device_id}")
            raise RuntimeError(error_msg)

        # Calculate the actual data pointer: base_ptr + offset
        data_ptr = base_ptr.value + offset
        if offset > 0:
            data_ptr_hex = hex(data_ptr)
            base_ptr_hex = hex(base_ptr.value)
            flexkv_logger.info(
                f"_import_cuda_ipc_handle: Opened IPC handle: device={device}, base_gpu_ptr={base_ptr_hex}, offset={offset}, actual data_ptr={data_ptr_hex}"
            )

        # Create tensor from pointer using helper function
        tensor = TensorSharedHandle._create_tensor_from_cuda_ptr(
            data_ptr, shape, dtype, device
        )

        flexkv_logger.info(f"Imported tensor with shape {shape} from CUDA IPC handle, dtype={tensor.dtype}, offset={offset}")
        return tensor


def _zmq_test_worker() -> None:
    context = zmq.Context()
    socket = context.socket(zmq.SocketType.PULL)
    socket.connect("tcp://127.0.0.1:5555")
    handle = socket.recv_pyobj()
    tensor = handle.get_tensor()
    print(f"Process {os.getpid()}: tensor: {tensor}")
    tensor[:] = 1
    print(f"Process {os.getpid()}: tensor modified")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    gpu_tensor = torch.zeros(10, dtype=torch.int64, device="cuda:0")
    print(f"Process {os.getpid()}: tensor: {gpu_tensor}")
    gpu_handle = TensorSharedHandle(gpu_tensor, force_direct_ipc=True)

    context = zmq.Context()
    socket = context.socket(zmq.SocketType.PUSH)
    socket.bind("tcp://127.0.0.1:5555")

    process = mp.Process(target=_zmq_test_worker, daemon=True)
    process.start()

    time.sleep(1)
    socket.send_pyobj(gpu_handle)

    process.join()
    print(f"Process {os.getpid()}: tensor: {gpu_tensor}")
