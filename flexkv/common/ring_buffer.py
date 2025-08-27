import torch
import threading
import time
import random

from collections import OrderedDict,deque
import numpy as np
from flexkv.common.transfer import TransferOp
from flexkv.common.debug import flexkv_logger

class PinnedMemoryRing:
    def __init__(self, max_task_num: int, max_block_num: int, dtype = np.int64):
        self.max_task_num = max_task_num
        self.max_block_num = max_block_num
        self.dtype = dtype
        self.time_out = 1  ## waiting time for get free slot (1s)
        # create the buffer tensor
        self.src_buffer_o = torch.empty((self.max_task_num, self.max_block_num), dtype = torch.int64)
        self.dst_buffer_o = torch.empty((self.max_task_num, self.max_block_num), dtype = torch.int64)
        # move tensor to share memory
        self.src_buffer = self.src_buffer_o.share_memory_()
        self.dst_buffer = self.dst_buffer_o.share_memory_()
        
        flexkv_logger.info(f"[PinnedMemoryRing] block ids src_buffer data_ptr: {self.src_buffer.data_ptr()}")
        flexkv_logger.info(f"[PinnedMemoryRing] block ids dst_buffer data_ptr: {self.dst_buffer.data_ptr()}")

        self.op_slot_map = OrderedDict() ## {op_id : ring buffer slot}
        self.slot_in_use = [False]*max_task_num
        self.free_slots = deque(range(max_task_num))

        self.valid_length = [0]*max_task_num
        
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
    def allocate_and_write(self, op_id: int, op: TransferOp):
        """
        Allocating a slot for the op and copy src block ids and dst block ids to the buffer.
        Params:
            op_id: the id index of the op
            op: the actual op object, which contains the block ids
        Returns:
            slot: the slot which is assigned to the current op
            num_blocks: the valid number of blocks in the current op
        """
        # firstly, determine whether the length of block ids exceeds the limit
        num_blocks = op.src_descriptor.physical_block_ids.size(0)
        if num_blocks > self.max_block_num:
            raise ValueError(f"block_ids too large: {num_blocks} > {self.max_block_num}")
        
        assert op.src_descriptor.physical_block_ids.size(0) == op.dst_descriptor.physical_block_ids.size(0), \
            f"the number of src block ids ({op.src_descriptor.physical_block_ids.size(0)}) is not eaqual to" \
            f"the number of dst block ids ({op.dst_descriptor.physical_block_ids.size(0)})"

        # get the slot of empty buffer
        with self.condition:
            while not self.free_slots:
                if not self.condition.wait(timeout=self.time_out):
                    raise TimeoutError("Timeout waiting for a free slot in the ring buffer")

            slot = self.free_slots.popleft()  # O(1) 
          
        # update status managers
        self.slot_in_use[slot] = True
        self.op_slot_map[op_id] = slot
        self.valid_length[slot] = num_blocks
        # print("----> ring buffer src blocks: ", op.src_descriptor.physical_block_ids)
        # print("----> ring buffer dst blocks: ", op.dst_descriptor.physical_block_ids)
        
        # do copy
        self.src_buffer[slot, :num_blocks] = op.src_descriptor.physical_block_ids
        self.dst_buffer[slot, :num_blocks] = op.dst_descriptor.physical_block_ids
        
        # set the rest value of this buffer to -1
        if num_blocks < self.max_block_num:
            self.src_buffer[slot, num_blocks:] = -1  # 
            self.dst_buffer[slot, num_blocks:] = -1  # 

        return slot, num_blocks
    
    def mark_free(self, op_id: int):
        """
        Free the relevant resources of corresponding op, called when op transfer completed.
        Input: 
            op_id: the index of current op
        Output:
            None
        """
        with self.condition:
            if op_id not in self.op_slot_map:
                raise KeyError(f"Task {op_id} not found in buffer")
            
            slot = self.op_slot_map[op_id]
            if not self.slot_in_use[slot]:
                raise RuntimeError(f"Slot {slot} is already free, double free detected!")
           
            self.slot_in_use[slot] = False
            self.valid_length[slot] = 0
            self.free_slots.append(slot)
            del self.op_slot_map[op_id]
            
            self.condition.notify()
    
    def get_src_block_ids(self, slot: int):
        if slot < 0 or slot >= self.max_task_num:
            raise IndexError(f"Invalid slot index {slot}")
        return self.src_buffer[slot, :self.valid_length[slot]]    
    
    def get_dst_block_ids(self, slot: int):
        if slot < 0 or slot >= self.max_task_num:
            raise IndexError(f"Invalid slot index {slot}")
        return self.dst_buffer[slot, :self.valid_length[slot]]    

    def get_src_buffer(self):
        return self.src_buffer
    
    def get_dst_buffer(self):
        return self.dst_buffer
    
    def get_buffer_size(self):
        return self.max_task_num, self.max_block_num
    
    def status(self):
        """
        Current status logger
        """
        with self.lock:
            used = sum(self.slot_in_use)
            free = self.max_task_num - used
            return {"used_slots": used,
                    "free_slots": free,
                    "capacity": self.max_task_num}


def producer(manager, task_id, data):
    try:
        print(f"Producer {task_id} trying to allocate...")
        slot = manager.allocate_and_write(task_id, data)
        print(f"Producer {task_id} got slot {slot}")

        time.sleep(random.uniform(0.1, 2.0))

        manager.mark_free(task_id)
        print(f"Producer {task_id} released slot {slot}")
    except Exception as e:
        print(f"Producer {task_id} encountered an error: {e}")

if __name__ == "__main__":
    manager = PinnedMemoryRing(4, 10)
    

 