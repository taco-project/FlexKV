import base64
import multiprocessing as mp
import os
import pickle
import time
from typing import Callable, Any, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.multiprocessing.reductions as reductions
import zmq

from flexkv.common.debug import flexkv_logger


@dataclass
class TensorSharedHandle:
    rebuild_func: Callable
    rebuild_args: Tuple[Any]
    device: torch.device

    def __init__(self, tensor: torch.Tensor, device_id: int = -1):
        if not tensor.is_cuda:
            raise ValueError("Only support CUDA tensor sharing")
        self.rebuild_func, self.rebuild_args, tensor_device_id = self._export_tensor_handle(tensor)
        if device_id == -1:
            self.device = tensor_device_id
        else:
            self.device = torch.device(f"cuda:{device_id}")
            tmp_list = list(self.rebuild_args)
            tmp_list[6] = device_id
            self.rebuild_args = tuple(tmp_list)

    def get_tensor(self) -> torch.Tensor:
        tensor = self._import_tensor_handle(self.rebuild_func, self.rebuild_args, self.device)
        return tensor

    @staticmethod
    def _export_tensor_handle(tensor: torch.Tensor) -> Tuple[Callable, Tuple[Any], torch.device]:

        device = tensor.device
        rebuild_func, rebuild_args = reductions.reduce_tensor(tensor)

        return rebuild_func, rebuild_args, device

    @staticmethod
    def _import_tensor_handle(rebuild_func: Callable, rebuild_args: Tuple[Any], device: torch.device) -> torch.Tensor:
        try:
            tensor = rebuild_func(*rebuild_args)
            assert isinstance(tensor, torch.Tensor)

            if tensor.device != device:
                flexkv_logger.warning(f"Tensor device {tensor.device} is not the same as the target device {device}")
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
    mp.set_start_method('spawn', force=True)

    gpu_tensor = torch.zeros(10, dtype=torch.int64, device="cuda:0")
    print(f"Process {os.getpid()}: tensor: {gpu_tensor}")
    gpu_handle = TensorSharedHandle(gpu_tensor)

    context = zmq.Context()
    socket = context.socket(zmq.SocketType.PUSH)
    socket.bind("tcp://127.0.0.1:5555")

    process = mp.Process(target=_zmq_test_worker, daemon=True)
    process.start()

    time.sleep(1)
    socket.send_pyobj(gpu_handle)

    process.join()
    print(f"Process {os.getpid()}: tensor: {gpu_tensor}")
