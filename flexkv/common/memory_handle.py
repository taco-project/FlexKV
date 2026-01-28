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


# Load CUDA runtime library
try:
    cudart = ctypes.CDLL("libcudart.so")
except:
    try:
        cudart = ctypes.CDLL("libcudart.so.12")
    except:
        cudart = ctypes.CDLL("libcudart.so.11")

# CUDA IPC handle size (64 bytes on Linux)
CUDA_IPC_HANDLE_SIZE = 64

# CUDA error codes
cudaSuccess = 0
cudaErrorInvalidValue = 11


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
    ):
        """
        Now we support three ways to construct TensorSharedHandle:
        If data is a tensor that is managed by torch, we will use the reduce_tensor method to export the TensorSharedHandle.
        If data is a tensor that is allocated by cudamalloc, we will use the cudaIpcGetMemHandle method to export the TensorSharedHandle.
        If data is bytes-like, it means the memory has already been shared by CUDA IPC, we will skip the export process to construct the TensorSharedHandle.
        """

        self.use_direct_ipc = False
        self.ipc_handle = None
        self.tensor_shape = None
        self.tensor_dtype = None
        self.tensor_numel = None

        if isinstance(data, torch.Tensor):
            self._init_from_tensor(data, device_id, force_direct_ipc)
            return

        elif isinstance(data, bytes):
            self._init_from_ipc_handle(
                bytes(data), device_id, tensor_shape, tensor_dtype, offset=offset
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
                flexkv_logger.debug(
                    f"Tensor exported via PyTorch CUDA IPC for device {self.device}"
                )
                return
            except RuntimeError as e:
                flexkv_logger.warning(f"PyTorch CUDA IPC export failed: {e}")
                flexkv_logger.info("Attempting direct CUDA IPC export...")

        try:
            ## Try direct CUDA IPC export
            self.ipc_handle = self._export_cuda_ipc_handle(tensor)
            self.use_direct_ipc = True
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

    def _init_from_ipc_handle(
        self,
        ipc_handle: Optional[bytes],
        device_id: int,
        tensor_shape: Optional[Tuple[int, ...]],
        tensor_dtype: Optional[Union[torch.dtype, str]],
        offset: int = 0,
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
        
  
        flexkv_logger.info(
            f"TensorSharedHandle constructed from external IPC handle {self.ipc_handle.hex()} on device {self.device} \
                with shape {self.tensor_shape} and dtype {self.tensor_dtype}, ptr offset={offset}"
        )

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
        if self.use_direct_ipc:
            return self._import_cuda_ipc_handle(
                self.ipc_handle, self.tensor_shape, self.tensor_dtype, self.device, offset=self.offset
            )
        else:
            return self._import_tensor_handle(
                self.rebuild_func, self.rebuild_args, self.device
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
