import torch
from torch.multiprocessing import Queue as MPQueue, Pipe as MPPipe
import numpy as np

from flexkv.transfer.utils import group_blocks_by_node_and_type
from flexkv.transfer.worker import RemoteCPU2CPUTransferWorker, WorkerTransferOp
from flexkv.mooncakeEngineWrapper import RDMATaskInfo, MoonCakeTransferEngineWrapper

from flexkv.common.ring_buffer import SharedOpPool
from flexkv.common.storage import StorageHandle, AccessHandleType, KVCacheLayout, KVCacheLayoutType
from flexkv.common.config import MooncakeTransferEngineConfig, CacheConfig
from flexkv.common.transfer import TransferOp, TransferType
from flexkv.cache.redis_meta import RedisMeta

# def test_group_blocks_by_node_and_type():

#     src_block_ids = torch.tensor([0, 8, 11, 12])
#     dst_block_ids = torch.tensor([0, 1, 2, 3])
#     remote_block_node_ids = [0,0,1,1]
#     remote_block_src_types = ["ssd", "cpu", "cpu", "cpu"]
    
#     grouped = group_blocks_by_node_and_type(src_block_ids, dst_block_ids, remote_block_node_ids, remote_block_src_types)
#     print("Grouped result:", grouped)

def test_worker():
    finished_ops_queue: MPQueue[int] = MPQueue()
    parent_conn, child_conn = MPPipe()  # create pipe
    
    cache_config = CacheConfig(
        tokens_per_block=64,
        enable_cpu=True,
        enable_kv_sharing=True,
    )
    
    ## step1: initialize the mooncake transfer engine
    # config_path = "./mooncake_config_l.json"
    # mooncake_config = MooncakeTransferEngineConfig.from_file(config_path)
    # mooncake_transfer_engine = MoonCakeTransferEngineWrapper(mooncake_config)
    
    ## step1: initialize the redis meta
    
    redis_meta = RedisMeta(
        "172.160.18",
        6379,
        "yourpass",
        "127.0.0.1"
    )
    redis_meta.init_meta()
    node_id = redis_meta.get_node_id()
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
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=False,
                )
    
    ## step3: regist local cpu buffer to transfer engine
    # mooncake_transfer_engine.regist_buffer(physical_tensor.data_ptr(), total_size)
    
    # ## step4: regist node info to redis_server
    # mooncake_transfer_engine.register_node_meta(
    #     node_id = 0,
    #     ip = mooncake_config.engine_ip,
    #     port = mooncake_config.engine_port,
    #     cpu_buffer_base_ptr = physical_tensor.data_ptr(), ## record cpu base ptr into redis
    #     ssd_buffer_base_ptr =  physical_tensor.data_ptr() ## temp value
    # ) ## node id 0 recieve the data from mode id 1

    ## step5: initialize other obj that used in worker
    cpu_handle = StorageHandle(
            handle_type=AccessHandleType.TENSOR,
            data=physical_tensor,
            kv_layout=cpu_layout,
            dtype=torch.bfloat16,
        )    
    
    ## initialize SharedOpPool
    pin_buffer = SharedOpPool(2048, 100)

    ## step 6: initialize RemoteCPU2CPUTransferWorker
    worker = RemoteCPU2CPUTransferWorker(
        worker_id=0,
        transfer_conn= child_conn,
        finished_ops_queue = finished_ops_queue,
        op_buffer_tensor = pin_buffer.get_buffer(),
        cpu_blocks = cpu_handle.get_tensor(),
        cpu_kv_layout=cpu_handle.kv_layout,
        remote_kv_layout=cpu_handle.kv_layout, # TODO: get remote kv_layout
        dtype=cpu_handle.dtype,
        cache_config=cache_config
    )
    ## step7: construct the transfer_op
    src_block_ids = np.array([0, 8, 10, 11])
    dst_block_ids = np.array([0, 1, 2, 4])
    remote_block_ids = np.array([1,1,1,1])
    remote_block_type = np.array(["CPU", "CPU", "CPU", "CPU"])

    transfer_op = TransferOp(
        graph_id = 1,
        transfer_type = TransferType.DIST2H,
        src_block_ids=src_block_ids,
        dst_block_ids=dst_block_ids,
        layer_id = 0,
        layer_granularity=20,
        remote_block_node_ids=remote_block_ids,
    )
    # copy blocks into pin buffer
    transfer_op.src_slot_id = pin_buffer.allocate_slot(transfer_op.src_block_ids)
    transfer_op.dst_slot_id = pin_buffer.allocate_slot(transfer_op.dst_block_ids)
    
    ## initialize worker transfer_op
    worker_transfer_op = WorkerTransferOp(transfer_op)
    
    ## step8: read data from remote node
    worker.launch_transfer(worker_transfer_op)
    
    return physical_tensor, block_stride
    
if __name__ == "__main__":
    ret_tensor, block_stride = test_worker()
    print(block_stride)
    for i in range(6):
        print(ret_tensor[i*block_stride:(i+1)*block_stride])
