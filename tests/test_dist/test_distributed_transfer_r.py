import torch
import os
from torch.multiprocessing import Queue as MPQueue, Pipe as MPPipe
import numpy as np

from flexkv.transfer.utils import group_blocks_by_node_and_type
from flexkv.transfer.worker import PEER2CPUTransferWorker, WorkerTransferOp
from flexkv.mooncakeEngineWrapper import RDMATaskInfo, MoonCakeTransferEngineWrapper

from flexkv.common.ring_buffer import SharedOpPool
from flexkv.common.storage import StorageHandle, AccessHandleType, KVCacheLayout, KVCacheLayoutType
from flexkv.common.config import MooncakeTransferEngineConfig, CacheConfig
from flexkv.common.transfer import TransferOp, TransferType
from flexkv.cache.redis_meta import RedisMeta



def test_worker():
    finished_ops_queue: MPQueue[int] = MPQueue()
    parent_conn, child_conn = MPPipe()  # create pipe
    ## initialize the mooncake transfer engine
    os.environ["MC_REDIS_PASSWORD"] = "yourpass"
    
    cache_config = CacheConfig(
        tokens_per_block=64,
        enable_cpu=True,
        enable_kv_sharing=True,
        enable_p2p_cpu=True
    )
    redis_meta = RedisMeta(
        "172.160.18",
        6379,
        "yourpass",
        "127.0.0.1"
    )
    redis_meta.init_meta()
    node_id = redis_meta.get_node_id()
    print(f"[Node B] node id: {node_id}")
    
    cache_config.distributed_node_id = node_id

    # define the kv cache layout
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

    physical_tensor = torch.arange(
        start=0,
        end=total_size,
        dtype=torch.bfloat16,       # 或者使用 torch.long
        device="cpu",
        pin_memory=False,
    )
    
    cpu_handle = StorageHandle(
            handle_type=AccessHandleType.TENSOR,
            data=physical_tensor,
            kv_layout=cpu_layout,
            dtype=torch.bfloat16,
        )    
    pin_buffer = SharedOpPool(2048, 100)
    ssd_layout = KVCacheLayout(
        KVCacheLayoutType.BLOCKWISE,  ## test block wise layout
        num_layer = 20,
        num_block = 100,
        tokens_per_block= 64,
        num_head = 8,
        head_size = 32,
        is_mla=True
    )
    ssd_files = {0:"/data0/flexkv_ssd_cache_0_0.bin"}

    worker = PEER2CPUTransferWorker(
        worker_id=0,
        transfer_conn= child_conn,
        finished_ops_queue = finished_ops_queue,
        op_buffer_tensor = pin_buffer.get_buffer(),
        cpu_blocks = cpu_handle.get_tensor(),
        cpu_kv_layout=cpu_handle.kv_layout,
        ssd_kv_layout=ssd_layout,
        remote_kv_layout=cpu_handle.kv_layout, # TODO: get remote kv_layout
        dtype=cpu_handle.dtype,
        cache_config=cache_config,
        ssd_files=ssd_files,
        num_blocks_per_file=1000
    )

    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("exiting scripts")
    
    
if __name__ == "__main__":
    os.environ["MC_REDIS_PASSWORD"] = "yourpass"
    os.environ["MOONCAKE_CONFIG_PATH"] = "./mooncake_config_r.json"
    test_worker()