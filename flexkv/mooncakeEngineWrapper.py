import os
import time
from typing import List

import engine
from engine import TransferEngine
from flexkv.common.debug import flexkv_logger
from flexkv.common.config import MooncakeTransferEngineConfig
from flexkv.transfer.utils import RDMATaskInfo

class MoonCakeTransferEngineWrapper:
    def __init__(
        self, config: MooncakeTransferEngineConfig
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
        notify = engine.TransferNotify(notify_name, notify_msg)
        ret = self.engine.transfer_sync(
            task.peer_engine_addr, task.src_ptr, task.dst_ptr, task.data_size, engine.TransferOpcode.Write, notify)
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
            found = False
            notifies = self.engine.get_notifies()
            if notifies:
                for notify in notifies:
                    if notify.name == peer_addr or notify.msg == str(task_id):
                        flexkv_logger.info(f"Received notify: {notify.name}, {notify.msg}")
                        found= True
                        break
            if found:
                break
                    
            if time.time() - start_time > timeout:
                #TODO: how to cancle the transfer task
                flexkv_logger.warning(f"Timeout waiting for notify: {peer_addr}, task={task_id}")
                return False
        
            time.sleep(0.01) # sleep for 10 ms to avoid busy waiting
            
        return True
    
    # helper function
    def get_engine_addr(self):
        return self.mooncake_addr

if __name__ == "__main__":
    pass