import base64
import multiprocessing as mp
import os
import pickle
import time
from typing import List, Callable, Any

import torch
import torch.multiprocessing.reductions as reductions
import zmq

from flexkv.common.debug import init_logger
from flexkv.common.storage import KVCacheLayout


logger = init_logger(__name__)

class KVCacheTensorHandle:
    def __init__(
        self, 
        rebuild_func: Callable, 
        rebuild_args: List[Any], 
        device_id: int, 
        layer_id: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout
    ):
        self.rebuild_func = rebuild_func
        self.rebuild_args = rebuild_args
        self.device_id = device_id
        self.layer_id = layer_id
        self.dtype = dtype
        self.kv_layout = kv_layout
        

    def rebuild_tensor(self) -> torch.Tensor:
        return self.rebuild_func(*self.rebuild_args)

def export_layer_tensor_handle(
    tensor: torch.Tensor, 
    layer_id: int, 
    kv_layout: KVCacheLayout = None
) -> KVCacheTensorHandle:
    if not tensor.is_cuda:
        raise ValueError("Invalid tensor: not a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError("Invalid tensor: not a contiguous tensor")

    rebuild_func, rebuild_args = reductions.reduce_tensor(tensor)

    return KVCacheTensorHandle(rebuild_func, rebuild_args, tensor.device.index, layer_id, tensor.dtype, kv_layout)

def import_layer_tensor_handle(handle: KVCacheTensorHandle) -> tuple[torch.Tensor, int]:
    try:
        # handle = KVCacheTensorHandle.loads(tensor_desc_bytes)
        tensor = handle.rebuild_tensor()

        if not tensor.is_cuda:
            tensor = tensor.to(f"cuda:{handle.device_id}")

        return tensor, handle.layer_id

    except Exception as e:
        logger.error("Import tensor handle failed: %s", e)
        return None, -1

def _zmq_test_worker():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://127.0.0.1:5555")
    handle = socket.recv_pyobj()
    tensor, layer_id = import_layer_tensor_handle(handle)
    tensor[:] = 1
    print(f"Process {os.getpid()}: layer {layer_id} tensor modified")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    layer_tensor = torch.zeros(10, dtype=torch.int64, device="cuda:0")
    print(f"Process {os.getpid()}: layer {0}: {layer_tensor}")
    handle = export_layer_tensor_handle(layer_tensor, 0)

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://127.0.0.1:5555")

    process = mp.Process(target=_zmq_test_worker, daemon=True)
    process.start()

    time.sleep(1)
    socket.send_pyobj(handle)

    process.join()
    print(f"Process {os.getpid()}: layer {0}: {layer_tensor}")
