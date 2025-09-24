import os
import numpy as np
import redis
import json
import time
from typing import Optional, Union, List, Dict, Tuple, Any

from engine import TransferEngine
from flexkv.common.debug import flexkv_logger
from flexkv.common.config import MooncakeTransferEngineConfig
from flexkv.cache.redis_meta import RedisMeta
from urllib.parse import urlparse

class RDMATaskInfo:
    def __init__(
        self, task_id: int, peer_engine_addr: str, peer_zmq_addr: str, src_ptr: int, dst_ptr: int, 
        src_block_ids: int, dst_block_ids: int, data_size: int
    ):
        self.task_id = task_id
        self.peer_engine_addr = peer_engine_addr
        self.src_ptr = src_ptr
        self.dst_ptr = dst_ptr
        self.peer_zmq_addr = peer_zmq_addr
        self.src_block_ids = src_block_ids
        self.dst_block_ids = dst_block_ids
        self.data_size = data_size

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "peer_engine_addr": self.peer_engine_addr,
            "peer_zmq_addr": self.peer_zmq_addr,
            "src_ptr": self.src_ptr,
            "dst_ptr": self.dst_ptr,
            "src_block_ids": self.src_block_ids,
            "dst_block_ids": self.dst_block_ids,
            "data_size": self.data_size,
        }

    def from_dict(self, data: dict) -> "RDMATaskInfo":
        return RDMATaskInfo(
            task_id = data.get("task_id", 0),
            peer_engine_addr=data.get("peer_engine_addr", ""),
            peer_zmq_addr = data.get("peer_zmq_addr", ""),
            src_ptr=int(data.get("src_ptr", 0)),
            dst_ptr=int(data.get("dst_ptr", 0)),
            src_block_ids=data.get("src_block_ids"),
            dst_block_ids=data.get("dst_block_ids"),
            data_size=int(data.get("data_size", 0)),
        )


class NodeMetaInfo:
    """Node information for flexkv sub-nodes"""

    def __init__(
        self,
        node_id: int,
        engine_addr: Optional[str] = None,
        zmq_addr: Optional[str] = None,
        cpu_bufer_base_ptr: Optional[int] = None,
        ssd_bufer_base_ptr: Optional[int] = None,
    ):
        self.node_id = node_id
        self.engine_addr = engine_addr
        self.zmq_addr = zmq_addr
        self.cpu_bufer_base_ptr = cpu_bufer_base_ptr
        self.ssd_bufer_base_ptr = ssd_bufer_base_ptr

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_id": self.node_id,
            "engine_addr": self.engine_addr,
            "zmq_addr": self.zmq_addr,
            "cpu_bufer_base_ptr": self.cpu_bufer_base_ptr,
            "ssd_bufer_base_ptr": self.ssd_bufer_base_ptr,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeMetaInfo":
        return cls(
            node_id=data.get("node_id"),
            engine_addr=data.get("engine_addr"),
            zmq_addr=data.get("zmq_addr"),
            cpu_bufer_base_ptr=data.get("cpu_bufer_base_ptr"),
            ssd_bufer_base_ptr=data.get("ssd_bufer_base_ptr"),
        )


class MoonCakeTransferEngineWrapper:
    def __init__(
        self, config: MooncakeTransferEngineConfig, redis_meta: RedisMeta
    ):
        if config is None:
            mooncake_config_path = os.environ["MOONCAKE_CONFIG_PATH"]
            self.config = MooncakeTransferEngineConfig.from_file(mooncake_config_path)
        else:
            self.config = config
        self.engine_ip = config.engine_ip
        self.engien_port = config.engine_port
        self.mooncake_addr = f"{self.engine_ip}:{self.engien_port}"
        flexkv_logger.info(f"Mooncake listen on: {self.mooncake_addr}")

        supported_backend = ["redis"]
        self.metadata_backend = self.config.metadata_backend.lower()
        if self.metadata_backend not in supported_backend:
            raise ValueError(
                "Mooncake Configuration error. Currently only support "
                f" {supported_backend} metadata_backend."
            )

        # transfer engine initialize
        self.engine = TransferEngine()

        self.engine.initialize_ext(
            self.mooncake_addr,
            self.config.metadata_server,
            self.config.protocol,
            self.config.device_name,
            self.metadata_backend,
        ) 

        # Persistent NodeMetaInfo Pool for skip redis operation when getting 
        # NodeMetaInfo according to node_id
        # assuming that every flexkv progress has unique node id
        self.node_metas: Dict[int, NodeMetaInfo] = {}

        # init redis meta
        self.redis_meta_client = redis_meta
        assert self.redis_meta_client != None

        

    ## redis operations which calling redis_meta functions
    def regist_node_meta(
        self,
        cpu_buffer_base_ptr: int,
        ssd_buffer_base_ptr: int,
        zmq_addr: str,
    ) -> None:
        """Register the node to the redis."""
        self.redis_meta_client.regist_node_meta(self.redis_meta_client.get_node_id(), self.mooncake_addr, zmq_addr,
                                                cpu_buffer_base_ptr, ssd_buffer_base_ptr)
        #NOTE: maybe useless
        node_meta_info = NodeMetaInfo(
            self.redis_meta_client.get_node_id(),
            self.mooncake_addr,
            zmq_addr,
            cpu_buffer_base_ptr,
            ssd_buffer_base_ptr
        )
        self.node_metas[self.redis_meta_client.get_node_id()] = node_meta_info
        flexkv_logger.info(f"Registered node {self.redis_meta_client.get_node_id()} to Redis.")


    def unregist_node_meta(self, node_id: int = None) -> None:
        self.redis_meta_client.unregist_node_meta(self.redis_meta_client.get_node_id())
        flexkv_logger.info(f"Unregistered node {self.redis_meta_client.get_node_id()} from Redis.")
 

    def get_node_meta(self, node_id: int) -> Optional[NodeMetaInfo]:
        """Get the node meta info by node id."""
        if not node_id in self.node_metas:
            ## fetch from redis
            node_redis_data = self.redis_meta_client.get_node_meta(node_id)
            if not node_redis_data:
                flexkv_logger.error(f"Node {node_id} meta not found in Redis.")
                return None

            node_meta = NodeMetaInfo.from_dict(node_redis_data)

            self.node_metas[node_id] = node_meta
            flexkv_logger.info(f"Fetched node {node_id} meta from Redis.")

        return self.node_metas[node_id]

    # mooncake operations
    def regist_buffer(self, buffer_ptr: int, buffer_size: int) -> int:
        """Register the buffer to the mooncake engine."""
        ret = self.engine.register_memory(buffer_ptr, buffer_size)
        return ret if ret == 0 else -1

    def unregist_buffer(self, buffer_ptr: int) -> None:
        """Unregister the buffer to the mooncake engine."""
        ret = self.engine.unregister_memory(buffer_ptr)
        return ret if ret == 0 else -1

    def transfer_sync_impl(self, task: RDMATaskInfo) -> int:
        """Transfer the data synchronously."""

        ret = self.engine.transfer_sync_read(
            task.peer_engine_addr, task.dst_ptr, task.src_ptr, task.data_size
        )
        return ret if ret == 0 else -1
    
    def transfer_sync_write(self, task: RDMATaskInfo):
        ret = self.engine.transfer_sync_write(
            task.peer_engine_addr, task.src_ptr, task.dst_ptr, task.data_size
        )
        return ret if ret == 0 else -1
    
    def transfer_sync_write_with_notify(self, task: RDMATaskInfo, notify_name: str, notify_msg: str):
        ret = self.engine.transfer_sync_write_with_notify(
            task.peer_engine_addr,
            task.src_ptr,
            task.dst_ptr,
            task.data_size,
            notify_name,
            notify_msg
        )
        return ret if ret == 0 else -1
    
    def wait_notify(self, peer_addr: str, task_id: int):
        """
        Wait for the notify from the remote peer. Currently, this operation will block the main thread.
        Note that because the remote ssd task is executed sequentially, here we just wait for the notify
        with a timeout. It should be pointed out that the when the tasks are executed in parallel, 
        we need to modify the implementation. Maybe we could use a map to store the tasks and their notify status,
        and design a seperate thread to poll the notifies and update the map.
        
        Input:
        peer_addr: the remote peer engine address
        task_id: the task id to wait for
        Output:
        True if the notify is received, False otherwise.
        """
        # TODO: modify the implementation to support parallel tasks.
        timeout = 5.0 # timeout after 5 seconds
        start_time = time.time()
        while True:
            notifies = self.engine.get_notifies()
            if notifies:
                if notifies.name == peer_addr and notifies.msg == str(task_id):
                    flexkv_logger.info(f"Received notify: {notifies.name}, {notifies.msg}")
                    return True
            if time.time() - start_time > timeout:
                #TODO: how to cancle the transfer task
                flexkv_logger.warning(f"Timeout waiting for notify: {peer_addr}, task={task_id}")
                return False
        
            time.sleep(0.01) # sleep for 10 ms to avoid busy waiting
            
    # helper function
    def get_engine_addr(self):
        return self.mooncake_addr

if __name__ == "__main__":
    pass