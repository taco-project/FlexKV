import torch
import time
import numpy as np
from flexkv.core.kv_manager import KVManager
from flexkv.core.debug_utils import debuginfo
import random
from typing import List, Tuple, Dict
import json
from pathlib import Path
from time import sleep

class SSDBenchmark:
    def __init__(self):
        self.tokens_per_block = 16
        self.num_layers = 32
        self.num_kv_heads = 8
        self.head_size = 128

        # use small cpu size
        self.block_per_request = 32
        self.cpu_block_num = 128
        # 16 * 32 * 8 * 128 * 2 * 2 = 2 MB
        self.ssd_block_num = 2048
        self.gpu_block_num = 512
        assert self.ssd_block_num % self.gpu_block_num == 0


        self.block_size = (
            self.tokens_per_block * self.num_kv_heads * self.head_size
        )
        self.seq_len = self.block_per_request * self.tokens_per_block
        self.num_requests = self.ssd_block_num // self.block_per_request

        # use for verification
        self.gpu_blocks_gt = None

    def init_kvmanager(self) -> Tuple[KVManager, List[torch.Tensor]]:
        """initialize KVManager and return GPU blocks for verification"""
        gpu_blocks = [
            torch.randn(
                2, self.gpu_block_num,
                self.tokens_per_block, self.num_kv_heads,
                self.head_size, dtype=torch.float16
            ).cuda()
            for _ in range(self.num_layers)
        ]
        self.gpu_blocks = gpu_blocks
        # save a copy for verification
        self.gpu_blocks_gt = [block.clone() for block in gpu_blocks]

        kvpool = KVManager(
            num_cpu_blocks=self.cpu_block_num,
            num_ssd_blocks=self.ssd_block_num,
            tokens_per_block=self.tokens_per_block,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=torch.float16,
            gpu_physical_blocks=gpu_blocks,
        )

        return kvpool

    def generate_request_pair(self, idx: int):
        """generate a pair of matching token_ids and block_ids"""
        start_idx = (idx * self.block_per_request) % self.gpu_block_num
        if start_idx + self.block_per_request >= self.gpu_block_num:
            start_idx = (
                (start_idx + self.block_per_request) % self.gpu_block_num
            )
        block_ids = torch.arange(
            start_idx,
            start_idx + self.block_per_request,
            dtype=torch.int64
        )

        # generate random token_ids with shape
        # (block_per_request * tokens_per_block,)
        token_ids = torch.randint(
            low=0,
            high=100,
            size=(self.block_per_request * self.tokens_per_block,),
            dtype=torch.int64
        )

        return token_ids, block_ids

    def verify_data(self):
        # verify data correctness
        for i in range(self.num_layers):
            gpu_k = self.gpu_blocks[i][0, :, :, :, :]
            gpu_v = self.gpu_blocks[i][1, :, :, :, :]
            gpu_k_gt = self.gpu_blocks_gt[i][0, :, :, :, :]
            gpu_v_gt = self.gpu_blocks_gt[i][1, :, :, :, :]

            assert torch.allclose(gpu_k, gpu_k_gt), f"K mismatch at layer {i}"
            assert torch.allclose(gpu_v, gpu_v_gt), f"V mismatch at layer {i}"

    def benchmark_mixed(self) -> dict:
        """test mixed read/write performance"""
        kvpool = self.init_kvmanager()
        print("generating request pairs...")
        request_pairs = [
            self.generate_request_pair(i)
            for i in range(self.num_requests)
        ]

        # write initial data
        initial_write_num = self.num_requests // (
            self.ssd_block_num // self.cpu_block_num
        )
        write_requests = []
        print("writing initial data...")
        for token_ids, block_ids in request_pairs[:initial_write_num]:
            request = kvpool.async_put(
                token_ids=token_ids,
                token_mask=None,
                gpu_physical_block_ids=block_ids.pin_memory(),
            )
            write_requests.append(request)
        kvpool.wait_until_finished(write_requests)
        print("initial data written")
        # mixed read/write test
        total_data_size = 0
        all_requests = []
        start_time = time.time()
        print("performing mixed read/write...")

        for i in range(initial_write_num, self.num_requests):
            print(f"performing mixed read/write {i} / {self.num_requests} ...")
            # read from written data
            read_idx = i - initial_write_num
            token_ids, block_ids = request_pairs[read_idx]

            request = kvpool.async_get(
                token_ids=token_ids,
                token_mask=None,
                gpu_physical_block_ids=block_ids.pin_memory(),
            )
            all_requests.append(request)
            sleep(0.2)
            # write new data
            token_ids, block_ids = request_pairs[i]
            request = kvpool.async_put(
                token_ids=token_ids,
                token_mask=None,
                gpu_physical_block_ids=block_ids.pin_memory(),
            )

            all_requests.append(request)
            total_data_size += (
                self.block_per_request * self.block_size *
                self.num_layers * 2 * 2
            )

        kvpool.wait_until_finished(all_requests)
        print("mixed read/write done")
        end_time = time.time()

        total_data_size = 2 * total_data_size / (1024 * 1024 * 1024)
        total_time = (
            end_time - start_time -
            0.2 * (self.num_requests - initial_write_num)
        )
        #print(f"total time: {total_time} s")
        throughput = total_data_size / total_time

        kvpool.shutdown()

        self.verify_data()

        return {
            "total_data_gb": total_data_size,
            "total_time_s": total_time,
            "w/r throughput_gb_s": throughput,
            "num_requests": self.num_requests - initial_write_num,
            "blocks_per_request": self.block_per_request,
        }

if __name__ == "__main__":
    debuginfo.set_level("INFO")
    benchmark = SSDBenchmark()
    result = benchmark.benchmark_mixed()
    print(result)
