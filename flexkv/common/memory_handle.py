import torch
from typing import List, Optional, Callable, Any

import torch.multiprocessing.reductions as reductions
import base64
import pickle
import traceback

# from flexkv.c_ext import export_handle, import_handle

class KVCacheTensorHandle:
    def __init__(self, rebuild_func: Callable, rebuild_args: List[Any], device_id: int, layer_id: int):
        self.rebuild_func = rebuild_func
        self.rebuild_args = rebuild_args
        self.device_id = device_id
        self.layer_id = layer_id

    def dumps(self) -> bytes:
        tensor_desc = {
            "rebuild_func": base64.b64encode(pickle.dumps(self.rebuild_func)).decode("ascii"),
            "rebuild_args": base64.b64encode(pickle.dumps(self.rebuild_args)).decode("ascii"),
            "device_id": self.device_id,
            "layer_id": self.layer_id
        }
        return pickle.dumps(tensor_desc)

    @classmethod
    def loads(cls, tensor_desc_bytes: bytes) -> "KVCacheTensorHandle":
        tensor_desc = pickle.loads(tensor_desc_bytes)
        rebuild_func = pickle.loads(base64.b64decode(tensor_desc["rebuild_func"]))
        rebuild_args = pickle.loads(base64.b64decode(tensor_desc["rebuild_args"]))
        return KVCacheTensorHandle(rebuild_func, rebuild_args, tensor_desc["device_id"], tensor_desc["layer_id"])

    def rebuild_tensor(self) -> torch.Tensor:
        return self.rebuild_func(*self.rebuild_args)

def export_layer_tensor_handle(tensor: torch.Tensor, layer_id: int) -> bytes:
    if not tensor.is_cuda:
        raise ValueError("Invalid tensor: not a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError("Invalid tensor: not a contiguous tensor")

    rebuild_func, rebuild_args = reductions.reduce_tensor(tensor)

    return KVCacheTensorHandle(rebuild_func, rebuild_args, tensor.device.index, layer_id).dumps()

def import_layer_tensor_handle(tensor_desc_bytes: bytes) -> torch.Tensor:
    try:
        handle = KVCacheTensorHandle.loads(tensor_desc_bytes)
        tensor = handle.rebuild_tensor()

        if not tensor.is_cuda:
            tensor = tensor.to(f"cuda:{handle.device_id}")

        return tensor

    except Exception as e:
        print("Import tensor handle failed: %s", e)
        return None

def _test_import_and_modify_tensor(handle: bytes):
    tensor = import_layer_tensor_handle(handle)
    tensor[:] = 1

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    layer_tensor = torch.zeros(10, dtype=torch.int64, device="cuda:0")
    print(layer_tensor)
    handle = export_layer_tensor_handle(layer_tensor, 0)

    process = mp.Process(
            target=_test_import_and_modify_tensor,
            args=(handle,),
            daemon=True
        )
    process.start()
    process.join()
    print(layer_tensor)
