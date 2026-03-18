import multiprocessing as mp
import os
import time
from typing import Callable, Any, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.multiprocessing.reductions as reductions
import zmq

from flexkv.common.debug import flexkv_logger
from flexkv.common import gpu_runtime


@dataclass
class TensorSharedHandle:
    rebuild_func: Optional[Callable]
    rebuild_args: Optional[Tuple[Any]]
    device: torch.device
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
        tensor_shape: Optional[Tuple[int, ...]] = None,
        tensor_dtype: Optional[
            Union[torch.dtype, str]
        ] = None,
        offset: int = 0,
    ):
        """
        Now we support three ways to construct TensorSharedHandle:
        If data is a tensor that is managed by torch, we will use the reduce_tensor method
            to export the TensorSharedHandle.
        If data is a tensor that is allocated by cudamalloc/musaMalloc, we will use the
            IPC API to export the TensorSharedHandle.
        If data is bytes-like, it means the memory has already been shared by IPC,
            we will skip the export process to construct the TensorSharedHandle.
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
        if not gpu_runtime.is_gpu_tensor(tensor):
            raise ValueError("Only support GPU tensor sharing (CUDA or MUSA)")

        if not force_direct_ipc:
            try:
                (
                    self.rebuild_func,
                    self.rebuild_args,
                    tensor_device_id,
                ) = self._export_tensor_handle(tensor)
                if device_id == -1:
                    self.device = tensor_device_id
                else:
                    self.device = gpu_runtime.get_device(device_id)
                    tmp_list = list(self.rebuild_args)
                    tmp_list[6] = device_id
                    self.rebuild_args = tuple(tmp_list)
                flexkv_logger.debug(
                    f"Tensor exported via PyTorch CUDA IPC for device {self.device}"
                )
                return
            except RuntimeError as e:
                flexkv_logger.warning(f"PyTorch IPC export failed: {e}")
                flexkv_logger.info("Attempting direct IPC export...")

        try:
            self.ipc_handle = self._export_ipc_handle(tensor)
            self.use_direct_ipc = True
            self.tensor_shape = tuple(tensor.shape)
            self.tensor_dtype = tensor.dtype
            self.tensor_numel = tensor.numel()
            self.device = (
                tensor.device if device_id == -1 else gpu_runtime.get_device(device_id)
            )
            self.rebuild_func = None
            self.rebuild_args = None
            self.offset = 0
            flexkv_logger.info(
                f"Tensor exported via direct IPC: tensor.device={tensor.device}, passed device_id={device_id}, final self.device={self.device}"
            )
        except Exception as e:
            raise RuntimeError(f"Both PyTorch and direct IPC export failed: {e}") from e

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

        self.use_direct_ipc = True
        self.ipc_handle = bytes(ipc_handle)
        self.tensor_shape = resolved_shape
        self.tensor_dtype = resolved_dtype
        numel = 1
        for dim in resolved_shape:
            numel *= dim
        self.tensor_numel = numel
        self.device = gpu_runtime.get_device(device_id)
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
        if self.use_direct_ipc:
            return self._import_ipc_handle(
                self.ipc_handle, self.tensor_shape, self.tensor_dtype, self.device,
                offset=self.offset
            )
        else:
            return self._import_tensor_handle(
                self.rebuild_func, self.rebuild_args, self.device
            )

    @staticmethod
    def _export_tensor_handle(
        tensor: torch.Tensor,
    ) -> Tuple[Callable, Tuple[Any], torch.device]:
        device = tensor.device
        rebuild_func, rebuild_args = reductions.reduce_tensor(tensor)
        return rebuild_func, rebuild_args, device

    @staticmethod
    def _export_ipc_handle(tensor: torch.Tensor) -> bytes:
        """Export an IPC handle for the tensor using the active backend (CUDA or MUSA)."""
        data_ptr = tensor.data_ptr()
        device = tensor.device

        flexkv_logger.debug(f"Exporting IPC handle: device={device}, data_ptr={hex(data_ptr)}")
        gpu_runtime.set_device(device.index if device.index is not None else 0)

        handle_bytes = gpu_runtime.ipc_get_mem_handle(data_ptr)
        flexkv_logger.debug(f"IPC handle exported successfully, first 16 bytes: {handle_bytes.hex()}")
        return handle_bytes

    @staticmethod
    def _import_ipc_handle(
        ipc_handle: bytes,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        offset: int = 0,
    ) -> torch.Tensor:
        """Import a tensor from an IPC handle using the active backend (CUDA or MUSA)."""
        flexkv_logger.debug(f"Attempting to import IPC handle for device {device}")

        if not gpu_runtime.is_initialized():
            flexkv_logger.info("Initializing GPU runtime in subprocess")
            gpu_runtime.init_runtime()

        device_id = device.index if device.index is not None else 0
        gpu_runtime.set_device(device_id)

        _ = torch.zeros(1, device=device)
        flexkv_logger.debug(
            f"GPU context created for device {device_id}, current_device={gpu_runtime.current_device()}"
        )

        dev_ptr_value = gpu_runtime.ipc_open_mem_handle(ipc_handle)

        data_ptr = dev_ptr_value + offset
        if offset > 0:
            flexkv_logger.info(
                f"_import_ipc_handle: base_gpu_ptr={hex(dev_ptr_value)}, offset={offset}, actual data_ptr={hex(data_ptr)}"
            )

        flexkv_logger.debug(f"import IPC handle: device={device}, dev_ptr={hex(data_ptr)}")

        tensor = TensorSharedHandle._create_tensor_from_gpu_ptr(
            data_ptr, shape, dtype, device
        )

        flexkv_logger.debug(f"Imported tensor with shape {shape} from IPC handle, offset={offset}")
        return tensor

    @staticmethod
    def _create_tensor_from_gpu_ptr(
        data_ptr: int, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device,
        strides: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Create a PyTorch tensor from a GPU memory pointer (CUDA or MUSA).

        Handles bfloat16 and fp8 via intermediate uint types since
        __cuda_array_interface__ doesn't support their typestr directly.
        """
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
            torch.uint16: "<u2",
        }

        if dtype == torch.bfloat16:
            class _ArrayInterface:
                def __init__(self, ptr, shape, strides=None):
                    self.__cuda_array_interface__ = {
                        "data": (ptr, False),
                        "shape": tuple(shape),
                        "typestr": "<u2",
                        "version": 3,
                        "strides": strides,
                        "descr": [("", "")],
                    }
            array_iface = _ArrayInterface(data_ptr, shape, strides)
            tensor_u16 = torch.as_tensor(array_iface, dtype=torch.uint16, device=device)
            return tensor_u16.view(torch.bfloat16)

        elif hasattr(torch, 'float8_e4m3fn') and dtype == torch.float8_e4m3fn:
            class _ArrayInterface:
                def __init__(self, ptr, shape, strides=None):
                    self.__cuda_array_interface__ = {
                        "data": (ptr, False),
                        "shape": tuple(shape),
                        "typestr": "|u1",
                        "version": 3,
                        "strides": strides,
                        "descr": [("", "")],
                    }
            array_iface = _ArrayInterface(data_ptr, shape, strides)
            return torch.as_tensor(array_iface, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)

        else:
            if dtype not in TYPESTR_MAP:
                raise ValueError(f"Unsupported dtype for GPU pointer: {dtype}")

            class _ArrayInterface:
                def __init__(self, ptr, shape, typestr, strides=None):
                    self.__cuda_array_interface__ = {
                        "data": (ptr, False),
                        "shape": tuple(shape),
                        "typestr": typestr,
                        "version": 3,
                        "strides": strides,
                        "descr": [("", "")],
                    }
            array_iface = _ArrayInterface(data_ptr, shape, TYPESTR_MAP[dtype], strides)
            return torch.as_tensor(array_iface, dtype=dtype, device=device)

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

    device_str = gpu_runtime.get_device_string(0)
    gpu_tensor = torch.zeros(10, dtype=torch.int64, device=device_str)
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
