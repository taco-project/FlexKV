import torch
from typing import List, Optional
from flexkv.c_ext import export_handle, import_handle

class CUDAIPCHandle:
    def __init__(self, handle_data: bytes, shape: torch.Size, dtype: torch.dtype, device_id: int):
        self.handle_data = handle_data
        self.shape = shape
        self.dtype = dtype
        self.device_id = device_id

def export_cuda_tensor(tensor: torch.Tensor) -> CUDAIPCHandle:
    """Export a CUDA tensor to a shareable handle"""
    if not tensor.is_cuda:
        raise ValueError("Only CUDA tensors can be shared")
    
    ptr = tensor.data_ptr()
    handle_data = export_handle(ptr, tensor.numel() * tensor.element_size(), 
                                device_id=tensor.device.index)
    
    return CUDAIPCHandle(
        handle_data=handle_data,
        shape=tensor.shape,
        dtype=tensor.dtype,
        device_id=tensor.device.index
    )

def import_cuda_tensor(handle: CUDAIPCHandle) -> torch.Tensor:
    """Import a CUDA tensor from a handle"""
    with torch.cuda.device(handle.device_id): # is this necessary?
        ptr = import_handle(handle.handle_data, device_id=handle.device_id)
        
        tensor = torch.from_file(
            ptr,
            shape=handle.shape,
            dtype=handle.dtype,
            device=f"cuda:{handle.device_id}"
        )
        return tensor