import torch
import time
from typing import List

"""

class SSD_storage:
    def __init__(
        self, num_blocks, block_size, num_layers, num_kv_heads, head_size, dtype
    ):
        self.cache = open("ssd.cache", "wb+", buffering=0)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.layer_stride = (
            2 * block_size * num_kv_heads * head_size * dtype.itemsize
        )
        self.dtype = dtype
        self.block_buffer_size = (
            num_layers
            * 2
            * block_size
            * num_kv_heads
            * head_size
            * self.dtype.itemsize
        )
        self.buffer_size = self.num_blocks * self.block_buffer_size
        self.cache.write(b"\x00" * self.buffer_size)

    def get(self, block_id, target_tensor_list: List[torch.tensor]) -> None:
        assert block_id < self.num_blocks
        assert len(target_tensor_list) == self.num_layers
        assert target_tensor_list[0].shape[0] == 2
        assert target_tensor_list[0].shape[1] == 1
        assert target_tensor_list[0].shape[2] == self.block_size
        assert target_tensor_list[0].shape[3] == self.num_kv_heads
        assert target_tensor_list[0].shape[4] == self.head_size
        assert target_tensor_list[0].dtype == self.dtype

        offset = block_id * self.block_buffer_size
        stride = 0
        for layer_tensor in target_tensor_list:
            self.cache.seek(offset)
            raw_data = self.cache.read(self.layer_stride)
            data_tensor = torch.frombuffer(raw_data, dtype=self.dtype)
            layer_tensor.copy_(data_tensor.view_as(layer_tensor))
            stride += self.layer_stride
        return

    # vllm shape:
    # list of layer_num x [2,block_num,block_size,head_num,head_size]
    def put(self, block_id, cpu_tensor_list: List[torch.tensor]) -> None:
        assert block_id < self.num_blocks
        assert len(cpu_tensor_list) == self.num_layers
        assert cpu_tensor_list[0].shape[0] == 2
        assert cpu_tensor_list[0].shape[1] == 1
        assert cpu_tensor_list[0].shape[2] == self.block_size
        assert cpu_tensor_list[0].shape[3] == self.num_kv_heads
        assert cpu_tensor_list[0].shape[4] == self.head_size
        assert cpu_tensor_list[0].dtype == self.dtype
        offset = block_id * self.block_buffer_size
        stride = 0
        for layer_tensor in cpu_tensor_list:
            raw_data = layer_tensor.view(torch.uint8).numpy().tobytes()
            self.cache.seek(offset + stride)
            self.cache.write(raw_data)
            stride += self.layer_stride
        return None


if __name__ == "__main__":
    ssd = SSD_storage(10, 16, 32, 8, 128, torch.bfloat16)
    # vllm shape:
    # list of layer_num x [2,block_num,block_size,head_num,head_size]
    cpu_block_list = [
        torch.zeros((2, 1, 16, 8, 128), dtype=torch.bfloat16) for i in range(32)
    ]
    [i.fill_(2) for i in cpu_block_list]
    target_block_list = [
        torch.zeros((2, 1, 16, 8, 128), dtype=torch.bfloat16) for i in range(32)
    ]
    s = time.time()
    for i in range(100):
        ssd.put(0, cpu_block_list)
    print(f"put 100 block cost time:{time.time() - s}")
    ssd.get(1, target_block_list)
    s = time.time()
    for i in range(100):
        ssd.get(0, target_block_list)
    print(f"get 100 block cost time:{time.time() - s}")
"""
