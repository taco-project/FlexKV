import os
import time


import engine
from engine import TransferEngine
from flexkv.common.debug import flexkv_logger
from flexkv.common.config import MooncakeTransferEngineConfig
from flexkv.transfer.utils import RDMATaskInfo
from flexkv.transfer.zmqHelper import NotifyMsg, NotifyStatus
from typing import List


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

    def transfer_sync_read(self, peer_engine_addr: str, src_ptr: int, dst_ptr: int, data_size: int) -> int:
        """Transfer the data synchronously."""

        ret = self.engine.transfer_sync_read(
            peer_engine_addr, dst_ptr, src_ptr, data_size
        )
        return ret if ret == 0 else -1
    
    def transfer_sync_write(self, peer_engine_addr: int, src_ptr: int, dst_ptr: int, data_size: int) -> int:
        ret = self.engine.transfer_sync_write(
            peer_engine_addr, src_ptr, dst_ptr, data_size
        )
        return ret if ret == 0 else -1
    
    def batch_transfer_sync_read(self, peer_engine_addr: str, src_ptr_list: List[int], dst_ptr_list: List[int], data_size_list: List[int])-> int:
        ret = self.engine.batch_transfer_sync_read(peer_engine_addr, dst_ptr_list, src_ptr_list, data_size_list)
        return ret if ret == 0 else -1
    
    def batch_transfer_sync_write(self, peer_engine_addr: str, src_ptr_list: List[int], dst_ptr_list: List[int], data_size_list: List[int])-> int:
        ret = self.engine.batch_transfer_sync_write(peer_engine_addr, src_ptr_list, dst_ptr_list, data_size_list)
        return ret if ret == 0 else -1
    
    def transfer_sync_write_with_notify(self, peer_engine_addr: str, src_ptr: int, dst_ptr: int, data_size: int, notify_name: str, msg : NotifyMsg):
        notify = engine.TransferNotify(notify_name, msg.to_string())
        ret = self.engine.transfer_sync(
            peer_engine_addr, src_ptr, dst_ptr, data_size, engine.TransferOpcode.Write, notify)
        return ret if ret == 0 else -1
    
    def transfer_failure_notify(self, peer_engine_addr: str, src_ptr: int, dst_ptr: int, notify_name: str, notify_msg: NotifyMsg):
        notify = engine.TransferNotify(notify_name, notify_msg.to_string())
        ret = self.engine.transfer_sync(
           peer_engine_addr, src_ptr, dst_ptr, 0, engine.TransferOpcode.Write, notify)
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
        transfer_status = False
        while True:
            found = False
            notifies = self.engine.get_notifies()
            if notifies:
                for notify in notifies:
                    msg = NotifyMsg.from_string(notify.msg)
                    if notify.name == peer_addr and msg.task_id == task_id:
                        flexkv_logger.info(f"Received notify: {notify.name}, {notify.msg}")
                        if msg.status == NotifyStatus.SUCCESS:
                            transfer_status = True
                        found= True
                        break
            if found:
                break
                    
            if time.time() - start_time > timeout:
                #TODO: how to cancle the transfer task
                flexkv_logger.warning(f"Timeout waiting for notify: {peer_addr}, task={task_id}")
                return False
        
            time.sleep(0.01) # sleep for 10 ms to avoid busy waiting
            
        return transfer_status
    
    # helper function
    def get_engine_addr(self):
        return self.mooncake_addr

if __name__ == "__main__":
    pass