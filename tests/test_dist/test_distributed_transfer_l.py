import torch
from torch.multiprocessing import Queue as MPQueue, Pipe as MPPipe
import numpy as np
import os
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
    
    cache_config = CacheConfig(
        tokens_per_block=64,
        enable_cpu=True,
        enable_kv_sharing=True,
        enable_p2p_cpu=True
    )
    
    ## step1: initialize the redis meta
    redis_meta = RedisMeta(
        "172.160.18",
        6379,
        "yourpass",
        "127.0.0.1"
    )
    redis_meta.init_meta()
    node_id = redis_meta.get_node_id()
    print(f"[Node A] node id: {node_id}")

    cache_config.distributed_node_id = node_id
    ## step2: define the kv cache layout and allocate the physical buffer
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
    ## initial physical tensor and regist it to mooncake engine
    physical_tensor = torch.zeros(
                    size=(total_size,),
                    dtype=torch.uint32,
                    device="cpu",
                    pin_memory=False,
                )
    
    ## step5: initialize other obj that used in worker
    cpu_handle = StorageHandle(
            handle_type=AccessHandleType.TENSOR,
            data=physical_tensor,
            kv_layout=cpu_layout,
            dtype=torch.bfloat16,
        )    
    
    ## initialize SharedOpPool
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

    ## step 6: initialize PEER2CPUTransferWorker
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
    
    ## test remote cpu to local cpu
    cpu_src_block_ids = np.array([0, 1, 10, 11, 12, 13])
    cpu_dst_block_ids = np.array([0, 1, 2, 3, 4, 6])
    remote_block_node_ids = np.array([1, 1, 1, 1, 1, 1])

    transfer_op = TransferOp(
        graph_id = 1,
        transfer_type = TransferType.PEERH2H,
        src_block_ids=cpu_src_block_ids,
        dst_block_ids=cpu_dst_block_ids,
        layer_id = 0,
        layer_granularity=20,
        src_block_node_ids=remote_block_node_ids,
    )
    # copy blocks into pin buffer
    transfer_op.src_slot_id = pin_buffer.allocate_slot(transfer_op.src_block_ids)
    transfer_op.dst_slot_id = pin_buffer.allocate_slot(transfer_op.dst_block_ids)
    
    ## initialize worker transfer_op
    worker_transfer_op_cpu = WorkerTransferOp(transfer_op)
    
    ## step8: read data from remote node
    worker.launch_transfer(worker_transfer_op_cpu)
    
    
    #### test remote ssd 2 cpu
    ssd_src_block_ids = np.array([0, 1, 2, 4, 5, 6])
    ssd_dst_block_ids = np.array([7, 8, 9, 10, 11, 12])
    remote_block_node_ids = np.array([1, 1, 1, 1, 1, 1])
    
    transfer_op = TransferOp(
        graph_id = 1,
        transfer_type = TransferType.PEERSSD2H,
        src_block_ids=ssd_src_block_ids,
        dst_block_ids=ssd_dst_block_ids,
        layer_id = 0,
        layer_granularity=20,
        src_block_node_ids=remote_block_node_ids,
    )
    
    worker_transfer_op_ssd = WorkerTransferOp(transfer_op)
    worker.launch_transfer(worker_transfer_op_ssd)

    return physical_tensor, block_stride
    
if __name__ == "__main__":
    os.environ["MC_REDIS_PASSWORD"] = "yourpass"
    os.environ["MOONCAKE_CONFIG_PATH"] = "./mooncake_config_l.json"
    test_worker()
    ret_tensor, block_stride = test_worker()
    print(block_stride)
    blocks = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
    for i in blocks:
        print(ret_tensor[i*block_stride:(i+1)*block_stride])
