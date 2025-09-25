from flexkv.c_ext import  transfer_kv_blocks_ssd
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv import c_ext
import torch
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
        dtype=torch.uint32,       # 或者使用 torch.long
        device="cpu",
        pin_memory=False,
    )
    
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
        pass  # 创建一个空的二进制文件

    ssd_files = {0:"/data0/flexkv_ssd_cache_0_0.bin"}
    num_blocks_per_file = 1000
    num_files = sum(len(file_list) for file_list in ssd_files.values())

    ssd_kv_layout_per_file = ssd_layout.div_block(num_files, padding=True)

    
    chunk_size_in_bytes = cpu_layout.get_chunk_size() * torch.uint32.itemsize
    block_stride_in_bytes = cpu_layout.get_block_stride() * torch.uint32.itemsize
    cpu_kv_stride_in_bytes = cpu_layout.get_kv_stride() * torch.uint32.itemsize
    cpu_layer_stride_in_bytes = cpu_layout.get_layer_stride() * torch.uint32.itemsize
    ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * torch.uint32.itemsize
    ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * torch.uint32.itemsize

    ioctx = c_ext.SSDIOCTX(ssd_files, len(ssd_files), 32,0)
    
    layer_id_list = torch.arange(0, 20, dtype=torch.int32)
    ssd_block_id_list = torch.arange(0,100,dtype=torch.int32)
    cpu_block_id_list = torch.arange(0,100,dtype=torch.int32)

    transfer_kv_blocks_ssd(
        ioctx=ioctx,
        cpu_layer_id_list=layer_id_list,
        cpu_tensor_ptr=cpu_tensor.data_ptr(),
        ssd_block_ids=ssd_block_id_list,
        cpu_block_ids=cpu_block_id_list,
        cpu_layer_stride_in_bytes=cpu_layer_stride_in_bytes,
        cpu_kv_stride_in_bytes=cpu_kv_stride_in_bytes,
        ssd_layer_stride_in_bytes=ssd_layer_stride_in_bytes,
        ssd_kv_stride_in_bytes=ssd_kv_stride_in_bytes,
        chunk_size_in_bytes=chunk_size_in_bytes,
        block_stride_in_bytes=block_stride_in_bytes,
        is_read=False,
        num_blocks_per_file=num_blocks_per_file,
        round_robin=1,
        num_threads_per_device=32,
        is_mla=True,
    )

    result_tensor =  torch.zeros(
        size=(total_size,),
        dtype=torch.uint32,
        device="cpu",
        pin_memory=False,
    )
    
    transfer_kv_blocks_ssd(
        ioctx=ioctx,
        cpu_layer_id_list=layer_id_list,
        cpu_tensor_ptr=result_tensor.data_ptr(),
        ssd_block_ids=ssd_block_id_list,
        cpu_block_ids=cpu_block_id_list,
        cpu_layer_stride_in_bytes=cpu_layer_stride_in_bytes,
        cpu_kv_stride_in_bytes=cpu_kv_stride_in_bytes,
        ssd_layer_stride_in_bytes=ssd_layer_stride_in_bytes,
        ssd_kv_stride_in_bytes=ssd_kv_stride_in_bytes,
        chunk_size_in_bytes=chunk_size_in_bytes,
        block_stride_in_bytes=block_stride_in_bytes,
        is_read=True,
        num_blocks_per_file=num_blocks_per_file,
        round_robin=1,
        num_threads_per_device=32,
        is_mla=True,
    )
    
    print(result_tensor[:10])
    
    
write_data_to_ssd()