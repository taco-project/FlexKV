from flexkv.c_ext import  transfer_kv_blocks_ssd
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv import c_ext
import torch
import numpy as np
from typing import Union, List


def write_data_to_ssd():
    
    cpu_layout = KVCacheLayout(
        KVCacheLayoutType.BLOCKWISE,  ## test block wise layout
        num_layer = 20,
        num_block = 100,
        tokens_per_block= 64,
        num_head = 8,
        head_size = 32,
        is_mla=True
    )
    total_size = cpu_layout.get_total_elements()
    block_stride = cpu_layout.get_block_stride() 

    cpu_tensor = torch.arange(
        start=0,
        end=total_size,
        dtype=torch.int32,       # 或者使用 torch.long
        device="cpu",
        pin_memory=False,
    )
    # cpu_tensor = torch.ones(
    #     size=(total_size,),
    #     dtype=torch.int32,
    #     device="cpu",
    #     pin_memory=False,
    # )
    ssd_layout = KVCacheLayout(
        KVCacheLayoutType.BLOCKWISE,  ## test block wise layout
        num_layer = 20,
        num_block = 100,
        tokens_per_block= 64,
        num_head = 8,
        head_size = 32,
        is_mla=True
    )        
    with open("/data0/flexkv_ssd_cache_0_0.bin", "wb") as f:
        f.write(cpu_tensor.numpy().tobytes())

 

    result_tensor =  torch.zeros(
        size=(total_size,),
        dtype=torch.uint32,
        device="cpu",
        pin_memory=False,
    )
    with open("/data0/flexkv_ssd_cache_0_0.bin", "rb") as f:
        data = f.read()

    loaded = np.frombuffer(data, dtype=np.int32)  # 保持 dtype 一致
    loaded_tensor = torch.from_numpy(loaded)

    print("loaded tensor:", loaded_tensor[:10])
    

    
    
write_data_to_ssd()