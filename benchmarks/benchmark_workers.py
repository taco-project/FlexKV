import time
import json
import multiprocessing as mp
from dataclasses import dataclass
from typing import Tuple
from argparse import ArgumentParser
from tqdm import tqdm
import copy

import torch

from flexkv.common.transfer import TransferOp, TransferType
from flexkv.transfer.worker import GPUCPUTransferWorker, CPUSSDDiskTransferWorker, WorkerHandle, tpGPUCPUTransferWorker, \
    GDSTransferWorker, tpGDSTransferWorker
from flexkv.storage.allocator import CPUAllocator, GPUAllocator, SSDAllocator
from flexkv.common.storage import KVCacheLayoutType, KVCacheLayout
from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from utils import load_config

# flexkv_logger.set_level("OFF")

@dataclass
class BenchmarkConfig:
    transfer_type: TransferType = TransferType.H2D
    num_layers_to_transfer: int = -1
    num_blocks_to_transfer: int = 16
    shuffle_ids: bool = False
    warmup_round: int = 1
    benchmark_round: int = 10
    bidirectional: bool = False
    gpu_layout_type: int = 0

def make_configs(args: dict) -> Tuple[ModelConfig, CacheConfig, BenchmarkConfig]:
    config_file = args.config
    try:
        model_config, cache_config = load_config(config_file)
        if args.transfer_type == "H2D" or args.transfer_type == "D2H":
            cache_config.enable_ssd = False
        elif args.transfer_type == "H2DISK" or args.transfer_type == "DISK2H":
            assert cache_config.enable_ssd, "SSD cache must be enabled for DISK2H or H2DISK benchmark"
        elif args.transfer_type == "DISK2D" or args.transfer_type == "D2DISK":
            assert cache_config.enable_ssd, "SSD cache must be enabled for DISK2D or D2DISK benchmark"
        bench_config = BenchmarkConfig(
            transfer_type=TransferType(args.transfer_type),
            num_layers_to_transfer=args.num_layers,
            num_blocks_to_transfer=args.num_blocks,
            shuffle_ids=args.shuffle_ids,
            warmup_round=args.warmup_round,
            benchmark_round=args.benchmark_round,
            bidirectional=args.bi,
            gpu_layout_type=args.gpu_layout_type
        )
        cache_config.num_ssd_blocks = max(cache_config.num_ssd_blocks, bench_config.num_blocks_to_transfer)
        return model_config, cache_config, bench_config
    except Exception as e:
        raise ValueError(f"Failed to load config file {config_file}: {e}") from None

def create_cpu_gpu_worker(
                  model_config: ModelConfig,
                  cache_config: CacheConfig,
                  num_gpu_blocks: int,
                  gpu_layout_type: int = 0) -> Tuple[WorkerHandle, mp.Queue]:
    mp.set_start_method('spawn', force=True)
    cpu_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_cpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )
    if gpu_layout_type == 0 or gpu_layout_type == 2:
        layout_type = KVCacheLayoutType.LAYERFIRST
    elif gpu_layout_type == 1:
        layout_type = KVCacheLayoutType.BLOCKFIRST
    else:
        raise ValueError(f"Invalid GPU layout type: {gpu_layout_type}")
    
    if gpu_layout_type == 0:
        num_chunks = model_config.num_layers
    elif gpu_layout_type == 1:
        num_chunks = 1
    elif gpu_layout_type == 2:
        num_chunks = model_config.num_layers * 2
    else:
        raise ValueError(f"Invalid GPU layout type: {gpu_layout_type}")
    
    gpu_layout = KVCacheLayout(
        type=layout_type,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )
    gpu_layout = gpu_layout.div_head(model_config.tp_size) if not model_config.use_mla else gpu_layout
    cpu_handle = CPUAllocator.allocate(
        layout=cpu_layout,
        dtype=model_config.dtype,
        pin_memory=True
    )
    gpu_handles = []
    for tp_id in range(model_config.tp_size):
        gpu_handles.append(GPUAllocator.allocate(
            layout=gpu_layout,
            dtype=model_config.dtype,
            num_chunks=num_chunks,
        ))
    finished_ops_queue = mp.Queue()
    # Create a shared memory buffer for transfer operations
    # max_op_num=4, max_block_num should be larger than num_blocks_to_transfer
    max_block_num = max(1024, cache_config.num_cpu_blocks)
    op_buffer_tensor = torch.empty((4, max_block_num), dtype=torch.int64).share_memory_()

    if model_config.tp_size == 1:
        worker_handle = GPUCPUTransferWorker.create_worker(
            mp_ctx=mp.get_context('spawn'),
            finished_ops_queue=finished_ops_queue,
            op_buffer_tensor=op_buffer_tensor,
            gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
            cpu_blocks=cpu_handle.get_tensor(),
            gpu_kv_layout=gpu_handles[0].kv_layout,
            cpu_kv_layout=cpu_handle.kv_layout,
            dtype=model_config.dtype,
            gpu_device_id=0,
            use_ce_transfer_h2d=False,
            use_ce_transfer_d2h=False,
            transfer_sms_h2d=8,
            transfer_sms_d2h=8,
        )
    else:
        worker_handle = tpGPUCPUTransferWorker.create_worker(
            mp_ctx=mp.get_context('spawn'),
            finished_ops_queue=finished_ops_queue,
            op_buffer_tensor=op_buffer_tensor,
            gpu_blocks=[handle.get_tensor_handle_list() for handle in gpu_handles],
            cpu_blocks=cpu_handle.get_tensor(),
            gpu_kv_layout=gpu_handles[0].kv_layout,
            cpu_kv_layout=cpu_handle.kv_layout,
            dtype=model_config.dtype,
            tp_group_size=model_config.tp_size,
            dp_group_id=0,
            use_ce_transfer_h2d=False,
            use_ce_transfer_d2h=False,
            transfer_sms_h2d=8,
            transfer_sms_d2h=8,
        )
    return (
        worker_handle,
        finished_ops_queue,
    )

def create_cpu_ssd_worker(
                  model_config: ModelConfig,
                  cache_config: CacheConfig) -> Tuple[WorkerHandle, mp.Queue]:
    mp.set_start_method('spawn', force=True)
    cpu_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_cpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla
    )
    ssd_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.ssd_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_ssd_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla
    )
    cpu_handle = CPUAllocator.allocate(
        layout=cpu_layout,
        dtype=model_config.dtype,
        pin_memory=True
    )
    ssd_handle = SSDAllocator.allocate(
        layout=ssd_layout,
        dtype=model_config.dtype,
        num_chunks=model_config.num_layers,
        cache_dir=cache_config.ssd_cache_dir,
        max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb,
    )
    finished_ops_queue = mp.Queue()
    # Create a shared memory buffer for transfer operations
    # max_op_num=4, max_block_num should be larger than num_blocks_to_transfer
    max_block_num = max(1024, cache_config.num_cpu_blocks)
    op_buffer_tensor = torch.empty((4, max_block_num), dtype=torch.int64).share_memory_()

    worker_handle = CPUSSDDiskTransferWorker.create_worker(
                mp_ctx=mp.get_context('spawn'),
                finished_ops_queue=finished_ops_queue,
                op_buffer_tensor=op_buffer_tensor,
                cpu_blocks=cpu_handle.get_tensor(),
                ssd_files=ssd_handle.get_file_list(),
                cpu_kv_layout=cpu_handle.kv_layout,
                ssd_kv_layout=ssd_handle.kv_layout,
                dtype=model_config.dtype,
                num_blocks_per_file=ssd_handle.num_blocks_per_file,
                cache_config=cache_config
            )
    return (
        worker_handle,
        finished_ops_queue,
    )

def create_gpu_ssd_worker(
                  model_config: ModelConfig,
                  cache_config: CacheConfig,
                  num_gpu_blocks: int,
                  gpu_layout_type: int = 0) -> Tuple[WorkerHandle, mp.Queue]:
    mp.set_start_method('spawn', force=True)
    
    if gpu_layout_type == 0 or gpu_layout_type == 2:
        layout_type = KVCacheLayoutType.LAYERFIRST
    elif gpu_layout_type == 1:
        layout_type = KVCacheLayoutType.BLOCKFIRST
    else:
        raise ValueError(f"Invalid GPU layout type: {gpu_layout_type}")
    
    if gpu_layout_type == 0:
        num_chunks = model_config.num_layers
    elif gpu_layout_type == 1:
        num_chunks = 1
    elif gpu_layout_type == 2:
        num_chunks = model_config.num_layers * 2 
    else:
        raise ValueError(f"Invalid GPU layout type: {gpu_layout_type}")
    
    gpu_layout = KVCacheLayout(
        type=layout_type,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )
    ssd_layout = KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.ssd_layout_type,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_ssd_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )
    gpu_layout = gpu_layout.div_head(model_config.tp_size) if not model_config.use_mla else gpu_layout
    
    gpu_handles = []
    for tp_id in range(model_config.tp_size):
        gpu_handles.append(GPUAllocator.allocate(
            layout=gpu_layout,
            dtype=model_config.dtype,
            num_chunks=num_chunks,
        ))
    
    ssd_handle = SSDAllocator.allocate(
        layout=ssd_layout,
        dtype=model_config.dtype,
        num_chunks=model_config.num_layers,
        cache_dir=cache_config.ssd_cache_dir,
        max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb,
    )
    
    finished_ops_queue = mp.Queue()
    max_block_num = max(1024, cache_config.num_ssd_blocks)
    op_buffer_tensor = torch.empty((4, max_block_num), dtype=torch.int64).share_memory_()

    if model_config.tp_size == 1:
        worker_handle = GDSTransferWorker.create_worker(
            mp_ctx=mp.get_context('spawn'),
            finished_ops_queue=finished_ops_queue,
            op_buffer_tensor=op_buffer_tensor,
            gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
            ssd_files=ssd_handle.get_file_list(),
            num_blocks_per_file=ssd_handle.num_blocks_per_file,
            gpu_kv_layout=gpu_handles[0].kv_layout,
            ssd_kv_layout=ssd_handle.kv_layout,
            dtype=model_config.dtype,
            gpu_device_id=0,
        )
    else:
        worker_handle = tpGDSTransferWorker.create_worker(
            mp_ctx=mp.get_context('spawn'),
            finished_ops_queue=finished_ops_queue,
            op_buffer_tensor=op_buffer_tensor,
            gpu_blocks=[handle.get_tensor_handle_list() for handle in gpu_handles],
            ssd_files=ssd_handle.get_file_list(),
            num_blocks_per_file=ssd_handle.num_blocks_per_file,
            gpu_kv_layout=gpu_handles[0].kv_layout,
            ssd_kv_layout=ssd_handle.kv_layout,
            dtype=model_config.dtype,
            tp_group_size=model_config.tp_size,
            dp_group_id=0,
        )
    return (
        worker_handle,
        finished_ops_queue,
    )

def launch_transfer(worker_handle: WorkerHandle,
                    finished_ops_queue: mp.Queue,
                    transfer_op: TransferOp):
    worker_handle.submit_transfer(transfer_op)

def sync_all(finished_ops_queue: mp.Queue, num_ops: int):
    for _ in range(num_ops):
        finished_ops_queue.get()

REVERSE_TYPE_MAP = {
    TransferType.D2H: TransferType.H2D,
    TransferType.H2D: TransferType.D2H,
    TransferType.DISK2H: TransferType.H2DISK,
    TransferType.H2DISK: TransferType.DISK2H,
    TransferType.DISK2D: TransferType.D2DISK,
    TransferType.D2DISK: TransferType.DISK2D,
    }

def bench_worker(args):
    model_config, cache_config, bench_config = make_configs(args)
    print(f"model_config: {model_config}")
    print(f"cache_config: {cache_config}")
    print(f"bench_config: {bench_config}")
    if model_config.tp_size > torch.cuda.device_count():
        raise ValueError(f"TP size {model_config.tp_size} is greater than "
                         f"the number of GPUs {torch.cuda.device_count()}")
    warmup_round = bench_config.warmup_round
    benchmark_round = bench_config.benchmark_round
    transfer_type = bench_config.transfer_type
    num_layers_to_transfer = bench_config.num_layers_to_transfer
    if num_layers_to_transfer == -1:
        num_layers_to_transfer = model_config.num_layers
    num_blocks_to_transfer = bench_config.num_blocks_to_transfer
    shuffle_ids = bench_config.shuffle_ids
    bidirectional = bench_config.bidirectional
    gpu_layout_type = bench_config.gpu_layout_type

    if transfer_type == TransferType.H2D or transfer_type == TransferType.D2H:
        worker_handle, finished_ops_queue = create_cpu_gpu_worker(model_config, cache_config, num_blocks_to_transfer, gpu_layout_type)
    elif transfer_type == TransferType.H2DISK or transfer_type == TransferType.DISK2H:
        worker_handle, finished_ops_queue = create_cpu_ssd_worker(model_config, cache_config)
    elif transfer_type == TransferType.DISK2D or transfer_type == TransferType.D2DISK:
        worker_handle, finished_ops_queue = create_gpu_ssd_worker(model_config, cache_config, num_blocks_to_transfer, gpu_layout_type)
    else:
        raise ValueError(f"Unsupported transfer type: {transfer_type} for benchmark, "
                         f"currently only support {TransferType.H2D.name}, {TransferType.D2H.name}, "
                         f"{TransferType.H2DISK.name}, {TransferType.DISK2H.name}, "
                         f"{TransferType.DISK2D.name}, {TransferType.D2DISK.name}")
    reverse_worker_handle = None
    reverse_finished_ops_queue = None
    if bidirectional:
        if transfer_type == TransferType.H2D or transfer_type == TransferType.D2H:
            reverse_worker_handle, reverse_finished_ops_queue = \
                create_cpu_gpu_worker(model_config, cache_config, num_blocks_to_transfer, gpu_layout_type)
        elif transfer_type == TransferType.H2DISK or transfer_type == TransferType.DISK2H:
            reverse_worker_handle, reverse_finished_ops_queue = \
                create_cpu_ssd_worker(model_config, cache_config)
        elif transfer_type == TransferType.DISK2D or transfer_type == TransferType.D2DISK:
            reverse_worker_handle, reverse_finished_ops_queue = \
                create_gpu_ssd_worker(model_config, cache_config, num_blocks_to_transfer, gpu_layout_type)

    if shuffle_ids:
        block_ids = torch.randperm(num_blocks_to_transfer).numpy()
    else:
        block_ids = torch.arange(num_blocks_to_transfer).numpy()

    transfer_op = TransferOp(
        transfer_type=transfer_type,
        layer_id=0,
        layer_granularity=num_layers_to_transfer,
        src_block_ids=block_ids,
        dst_block_ids=block_ids,
        graph_id=0,
        dp_id=0,
        successors=[],
        predecessors=[],
    )

    reverse_transfer_op = None
    if bidirectional:
        reverse_type = REVERSE_TYPE_MAP.get(transfer_type)
        if reverse_type is None:
            raise ValueError(f"Bidirectional test not supported for transfer type: {transfer_type}")

        reverse_block_ids = torch.randperm(num_blocks_to_transfer).numpy()

        reverse_transfer_op = TransferOp(
            transfer_type=reverse_type,
            layer_id=0,
            layer_granularity=num_layers_to_transfer,
            src_block_ids=reverse_block_ids,
            dst_block_ids=reverse_block_ids,
            graph_id=1,
            dp_id=0,
            successors=[],
            predecessors=[],
        )
    if transfer_type == TransferType.DISK2H or transfer_type == TransferType.H2DISK:
        tmp_op = copy.deepcopy(transfer_op)
        tmp_op.transfer_type = TransferType.H2DISK
        tmp_op.src_block_ids = transfer_op.dst_block_ids
        tmp_op.dst_block_ids = transfer_op.src_block_ids
        launch_transfer(worker_handle, finished_ops_queue, tmp_op)
        sync_all(finished_ops_queue, 1)
    elif transfer_type == TransferType.DISK2D:
        tmp_op = copy.deepcopy(transfer_op)
        tmp_op.transfer_type = TransferType.D2DISK
        tmp_op.src_block_ids = transfer_op.dst_block_ids
        tmp_op.dst_block_ids = transfer_op.src_block_ids
        launch_transfer(worker_handle, finished_ops_queue, tmp_op)
        sync_all(finished_ops_queue, 1)

    for _ in range(warmup_round):
        if bidirectional:
            launch_transfer(reverse_worker_handle, reverse_finished_ops_queue, reverse_transfer_op)
        launch_transfer(worker_handle, finished_ops_queue, transfer_op)
    sync_all(finished_ops_queue, warmup_round)
    if bidirectional:
        sync_all(reverse_finished_ops_queue, warmup_round)

    pbar = tqdm(total=benchmark_round, desc="Benchmarking")
    start_time = time.time()
    for _ in range(benchmark_round):
        if bidirectional:
            launch_transfer(reverse_worker_handle, reverse_finished_ops_queue, reverse_transfer_op)
        launch_transfer(worker_handle, finished_ops_queue, transfer_op)
        pbar.update(1)
    pbar.close()
    sync_all(finished_ops_queue, benchmark_round)
    end_time = time.time()
    if bidirectional:
        sync_all(reverse_finished_ops_queue, benchmark_round)
    total_data_size_GB = (
        num_blocks_to_transfer *
        cache_config.tokens_per_block *
        model_config.token_size_in_bytes *
        num_layers_to_transfer /
        (model_config.num_layers * 1024 * 1024 * 1024)
    )
    avg_time = (end_time - start_time) / benchmark_round
    print(f"Total data size: {total_data_size_GB} GB")
    print(f"Avg Time taken: {avg_time} seconds")
    print(f"Avg Bandwidth: {total_data_size_GB / avg_time} GB/s")
    worker_handle.shutdown()
    if bidirectional:
        reverse_worker_handle.shutdown()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--transfer-type",
                        type=str,
                        default=TransferType.H2D.name)
    parser.add_argument("--num-layers",
                        type=int,
                        default=-1)
    parser.add_argument("--num-blocks",
                        type=int,
                        default=16)
    parser.add_argument("--config",
                        type=str,
                        default="./benchmarks/example_config.yml")
    parser.add_argument("--shuffle-ids",
                        action="store_true")
    parser.add_argument("--warmup-round",
                        type=int,
                        default=1)
    parser.add_argument("--benchmark-round",
                        type=int,
                        default=10)
    parser.add_argument("--bi",
                        action="store_true",
                        help="benchmark bidirectional bandwidth")
    parser.add_argument("--gpu-layout-type",
                        type=int,
                        default=0,
                        choices=[0, 1, 2],
                        help="GPU KV cache layout type")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    bench_worker(args)
