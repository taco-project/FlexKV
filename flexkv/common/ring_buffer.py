import torch
import threading
import time
import random

from collections import OrderedDict,deque
import numpy as np
from flexkv.common.transfer import TransferOp
from flexkv.common.debug import flexkv_logger
from flexkv.common.hash_utils import hash_array, hash_array_with_prefix


class SharedOpPool:
    def __init__(self, max_op_num: int, max_block_num: int, dtype = np.int64):
        self.max_op_num = max_op_num
        self.max_block_num = max_block_num
        self.dtype = dtype
        # create the buffer tensor
        self.buffer_o = torch.empty((self.max_op_num, self.max_block_num), dtype = torch.int64)
        # move tensor to share memory
        self.buffer = self.buffer_o.share_memory_()

        self.free_slots = deque(range(max_op_num))
        self.slot_map = dict() # {slot_hash: slot_id}

        self.slot_ref_count = np.zeros(max_op_num, dtype=np.int32)
        self.slot_hashes = [0]*max_op_num

        self.lock = threading.Lock()

    def allocate_slot(self, block_ids: np.ndarray, device_type_prefix: int = 0):
        """
        Allocating a slot for the given block ids
        Params:
            block_ids: the block ids of src address or dst address
            device_type_prefix: optional prefix to distinguish different device types
        Returns:
            slot_id: the slot which is assigned to the given block ids, -1 if failed
        """
        # firstly, determine whether the length of block ids exceeds the limit
        num_blocks = block_ids.size
        if num_blocks > self.max_block_num or num_blocks == 0:
            return -1

        # Use prefix to avoid hash collisions between different device types
        if device_type_prefix != 0:
            slot_hash = hash_array_with_prefix(block_ids, device_type_prefix)
        else:
            slot_hash = hash_array(block_ids)
        reuse = False

        # get the slot of empty buffer
        with self.lock:
            if slot_hash in self.slot_map:
                slot_id = self.slot_map[slot_hash]
                reuse = True
            else:
                if not self.free_slots:
                    flexkv_logger.info("No empty slot in SharedOpPool")
                    return -1

                slot_id = self.free_slots.popleft()
                self.slot_map[slot_hash] = slot_id

        # update status managers
        self.slot_ref_count[slot_id] += 1
        self.slot_hashes[slot_id] = slot_hash

        # do copy
        if not reuse:
            self.buffer[slot_id, :num_blocks] = torch.from_numpy(block_ids).to(torch.int64)

        return slot_id

    def free_slot(self, slot_id: int):
        """
        Free the relevant resources of corresponding op, called when op transfer completed.
        Input:
            op_id: the index of current op
        Output:
            None
        """
        with self.lock:
            slot_hash = self.slot_hashes[slot_id]
            if slot_hash not in self.slot_map:
                raise RuntimeError(f"Slot {slot_id} is not in use, double free detected!")
            self.slot_ref_count[slot_id] -= 1
            assert self.slot_ref_count[slot_id] >= 0, f"Slot {slot_id} ref count is negative"
            if self.slot_ref_count[slot_id] == 0:
                self.free_slots.append(slot_id)
                del self.slot_map[slot_hash]

    def get_buffer(self):
        return self.buffer

    def get_buffer_size(self):
        return self.max_op_num, self.max_block_num

    def status(self):
        """
        Current status logger
        """
        with self.lock:
            used = len(self.slot_map)
            free = self.max_op_num - used
            return {"used_slots": used,
                    "free_slots": free,
                    "capacity": self.max_op_num}


if __name__ == "__main__":
    manager = SharedOpPool(4, 10)
