import abc
from collections import deque
from typing import List, Tuple

import torch

from flexkv.core.utils import DeviceType
import time


class BasicMemPool(abc.ABC):
    def __init__(
        self,
        num_blocks: int,
    ):
        self.num_blocks = num_blocks

        self.free_ids = deque(range(self.num_blocks))
        self._preserve_blocks()

    @abc.abstractmethod
    def _preserve_blocks(self) -> None:
        pass

    # Transfer needs all physical blocks
    @abc.abstractmethod
    def get_physical_blocks(self):
        pass

    def allocate_blocks(self, num: int) -> List[int]:
        if num > self.num_blocks:
            raise ValueError("Exceed max num of blocks")
        if self.num_free_blocks < num:
            raise ValueError("Not enough free blocks")
        ids = []
        for _ in range(num):
            ids.append(self.free_ids.popleft())
        return ids

    def free_blocks(self, block_ids: List[int]):
        self.free_ids.extend(block_ids)

    def reset(self) -> None:
        self.free_ids.clear()

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_ids)


class CPUMemPool(BasicMemPool):
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        pin_memory: bool = False,
    ):
        self.num_layers = num_layers
        self.block_size = block_size
        self.dtype = dtype
        self.pin_memory = pin_memory
        self.physical_blocks: List[torch.Tensor] = []
        super().__init__(num_blocks)

    def _preserve_blocks(self) -> None:
        for _ in range(self.num_layers):
            self.physical_blocks.append(
                torch.empty(
                    self.layer_cache_shape,
                    dtype=self.dtype,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
            )

    def get_physical_blocks(self) -> List[torch.Tensor]:
        return self.physical_blocks

    @property
    def layer_cache_shape(self) -> Tuple[int, ...]:
        return (2, self.num_blocks, self.block_size)


class file_storage:
    def __init__(
        self,
        file_path: str = "ssd.cache",
        num_layers: int = 1,
        num_blocks: int = 1,
        block_size: int = 1024,
        dtype: torch.dtype = torch.float16,
    ):
        self.file_path = file_path
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.block_size_in_bytes = block_size * dtype.itemsize
        self.dtype = dtype
        self.size = num_layers * 2 * block_size * dtype.itemsize * num_blocks

        self.file = open(file_path, "wb+", buffering=0)
        self.file.write(b"\x00" * self.size)
        self.file.close()

    def write(self, src: torch.tensor, offset: int) -> None:
        raw_data = src.view(torch.uint8).numpy().tobytes()
        self.file.seek(int(offset))
        self.file.write(raw_data)
        self.file.flush()

    def read_to_block(self, dst: torch.tensor, offset: int):

        time_start = time.time()
        self.file.seek(int(offset))
        raw_data = self.file.read(self.block_size_in_bytes)
        time_middle = time.time()
        #print(f"ssd read offset{offset}, raw_data_length:{len(raw_data)}")
        dst.copy_(torch.frombuffer(raw_data, dtype=self.dtype).view_as(dst))
        time_end = time.time()
        print(f"ssd read time: {time_middle - time_start:.6f} s"
              f"tensor transfer time: {time_end - time_middle:.6f} s")
        print(f"ssd read bandwidth: "
              f"{self.block_size_in_bytes/(time_end-time_start)/1e9:.4f} GB/s")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def __del__(self) -> None:
        self.file.close()


class SSDMemPool(BasicMemPool):
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.block_size = block_size
        self.dtype = dtype
        super().__init__(num_blocks)

    def _preserve_blocks(self) -> None:
        self.file = file_storage(
            num_layers=self.num_layers,
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            dtype=self.dtype,
        )

    def get_physical_blocks(self) -> List[torch.Tensor]:
        return self.file
