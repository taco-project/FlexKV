import os
import zmq
import threading
import json
import time
from enum import Enum
from flexkv.transfer.utils import RemoteSSD2HMetaInfo
from flexkv.common.debug import flexkv_logger

class NotifyStatus(Enum):
    SUCCESS = 0
    FAIL = 1

class NotifyMsg:
    mooncake_engine_addr: str
    task_id: int
    status: NotifyStatus
    def __init__(self, mooncake_engine_addr: str,  task_id: int, status: NotifyStatus):
        self.mooncake_engine_addr = mooncake_engine_addr
        self.task_id = task_id
        self.status = status
        
    def to_string(self):
        return json.dumps({
            "mooncake_engine_addr": self.mooncake_engine_addr,
            "task_id": self.task_id,
            "status": self.status.name
        })
    @classmethod
    def from_string(cls, s):
        data = json.loads(s)
        status = NotifyStatus[data["status"]] 
        return cls(mooncake_engine_addr = data["mooncake_engine_addr"], task_id = data["task_id"], status = status)


class SSDZMQServer:
    def __init__(self, ip: str, port: int, ssd_handle_loop = None):
        if ssd_handle_loop is None:
            raise ValueError("ssd_handle_loop must be provided externally")
        
        self.zmq_context = zmq.Context()
        ## listening the ssd meta info
        self.meta_addr = f"tcp://{ip}:{port}"
        self.listen_socket = self.zmq_context.socket(zmq.REP)
        self.listen_socket.bind(self.meta_addr)
        
        self.shutdown_event = threading.Event()
        self.ssd_handle_loop = ssd_handle_loop
        self.thread = threading.Thread(target=self.ssd_handle_loop, daemon=True)
        self.start()
        print(f"[Server] Listening on {self.meta_addr}")

    def get_addr(self):
        return self.meta_addr
    
    def start(self):
        self.thread.start()

    def shutdown(self):
        self.shutdown_event.set()
        try:
            self.listen_socket.close(0)
            self.zmq_context.term()
        except Exception as e:
            flexkv_logger.error(f"Error when closing ZMQ: {e}")    
        self.thread.join()
        
    def send_transfer_status(self, peer_status_addr: str, status_msg: NotifyMsg):
        req_socket = self.zmq_context.socket(zmq.PUSH)
        req_socket.setsockopt(zmq.SNDTIMEO, 1000)  ## time out 1s
        try:
            req_socket.connect(peer_status_addr)
            req_socket.send(status_msg.to_string().encode("utf-8"))
            req_socket.close()
            flexkv_logger.info(
                f"Send meta transfer status to {peer_status_addr} for task {status_msg.task_id}"
            )
        except zmq.Again:
            # timeout
            flexkv_logger.error(
                f"Failed to send transfer status to {peer_status_addr}: timeout after {1000} ms"
            )
            req_socket.close()
            return False

        except Exception as e:
            # other exception
            req_socket.close()
            flexkv_logger.error(f"Error sending transfer status to {peer_status_addr}: {e}")
            return False   
    
                
class SSDZMQClient:
    def __init__(self, ip: str, port: int):
        self.zmq_context = zmq.Context()
        self.zmq_status_addr = f"tcp://{ip}:{port}"
        self.status_socket = self.zmq_context.socket(zmq.PULL)
        self.status_socket.bind(self.zmq_status_addr) 
    
    def get_addr(self):
        return self.zmq_status_addr
    
    def send_meta_info(self, meta_info: RemoteSSD2HMetaInfo, peer_zmq_addr: str):
        req_socket = self.zmq_context.socket(zmq.REQ)
        req_socket.setsockopt(zmq.SNDTIMEO, 1000)  ## time out 1s
        try:
            req_socket.connect(peer_zmq_addr)
            req_socket.send(json.dumps(meta_info.to_dict()).encode("utf-8"))
            reply = req_socket.recv()  
            
            if reply != b"OK":
                return False
            flexkv_logger.info(f"Meta info sent to {peer_zmq_addr}, waiting for transfer status")
            req_socket.close()
            return True

        except zmq.Again:
            # timeout
            flexkv_logger.error(
                f"Failed to send meta info to {peer_zmq_addr}: timeout after {1000} ms"
            )
            req_socket.close()
            return False

        except Exception as e:
            # other exception
            req_socket.close()
            flexkv_logger.error(f"Error sending meta info to {peer_zmq_addr}: {e}")
            return False

    def wait_transfer_notify(self, peer_addr: str, task_id: int):
        timeout = 5.0 # timeout after 5 seconds
        start_time = time.time()
        transfer_status = False
        while True:
            notify = self.status_socket.recv().decode("utf-8")
            msg = NotifyMsg.from_string(notify)
            
            if msg.mooncake_engine_addr == peer_addr and msg.task_id == task_id:
                flexkv_logger.info(f"Received notify: {msg.mooncake_engine_addr}, {msg.task_id}, {msg.status}")
                if msg.status == NotifyStatus.SUCCESS:
                    transfer_status = True
                break
            if time.time() - start_time > timeout:
                flexkv_logger.warning(f"Timeout waiting for notify: {peer_addr}, task={task_id}")
                return False
            time.sleep(0.01) # sleep for 10 ms to avoid busy waiting
        return transfer_status

    def shutdown(self):
        try:
            self.status_socket.close(0)
            self.zmq_context.term()
        except Exception as e:
            flexkv_logger.error(f"Error when closing ZMQ: {e}")    
