import torch
import os
from torch.multiprocessing import Queue as MPQueue, Pipe as MPPipe
import numpy as np

from flexkv.transfer.utils import group_blocks_by_node_and_type
from flexkv.transfer.worker import RemoteCPU2CPUTransferWorker, WorkerTransferOp
from flexkv.mooncakeEngineWrapper import RDMATaskInfo, MoonCakeTransferEngineWrapper

from flexkv.common.ring_buffer import SharedOpPool
from flexkv.common.storage import StorageHandle, AccessHandleType, KVCacheLayout, KVCacheLayoutType
from flexkv.common.config import MooncakeTransferEngineConfig
from flexkv.common.transfer import TransferOp, TransferType
from flexkv.cache.redis_meta import RedisMeta



def test_worker():
  
    ## initialize the mooncake transfer engine
    config_path = "./mooncake_config_r.json"
    os.environ["MC_REDIS_PASSWORD"] = "yourpass"

    redis_meta = RedisMeta(
        "172.160.18",
        6379,
        "yourpass",
        "127.0.0.1"
    )
    redis_meta.init_meta()
    
    mooncake_config = MooncakeTransferEngineConfig.from_file(config_path)
    mooncake_transfer_engine = MoonCakeTransferEngineWrapper(mooncake_config,redis_meta)

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
    ## initial physical tensor and regist it to mooncake engine
    # physical_tensor = torch.ones(
    #                 size=(total_size,),
    #                 dtype=torch.bfloat16,
    #                 device="cpu",
    #                 pin_memory=False,
    #             )
    physical_tensor = torch.arange(
        start=0,
        end=total_size,
        dtype=torch.bfloat16,       # 或者使用 torch.long
        device="cpu",
        pin_memory=False,
    )
    mooncake_transfer_engine.regist_buffer(physical_tensor.data_ptr(), total_size)
    # mooncake_transfer_engine.register_node_meta(
    #     node_id = 1,
    #     ip = mooncake_config.engine_ip,
    #     port = mooncake_config.engine_port,
    #     cpu_buffer_base_ptr = physical_tensor.data_ptr(), ## record cpu base ptr into redis
    #     ssd_buffer_base_ptr = physical_tensor.data_ptr() ## temp value
    # ) ## node id 0 recieve the data from mode id 1
    
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("exiting scripts")
    
    
if __name__ == "__main__":
    test_worker()