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
    use_ce_transfer: bool = False
    transfer_num_cta: int = 4

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
            gpu_layout_type=args.gpu_layout_type,
            use_ce_transfer=args.use_ce,
            transfer_num_cta=args.cta,
        )
        cache_config.num_ssd_blocks = max(cache_config.num_ssd_blocks, bench_config.num_blocks_to_transfer)
        return model_config, cache_config, bench_config
    except Exception as e:
        raise ValueError(f"Failed to load config file {config_file}: {e}") from None

def create_cpu_gpu_worker(
                  model_config: ModelConfig,
                  cache_config: CacheConfig,
                  num_gpu_blocks: int,
                  gpu_layout_type: int = 0,
                  use_ce_transfer: bool = False,
                  transfer_num_cta: int = 4) -> Tuple[WorkerHandle, mp.Queue]:
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
        torch.cuda.set_device(tp_id)
        gpu_handles.append(GPUAllocator.allocate(
            layout=gpu_layout,
            dtype=model_config.dtype,
            num_chunks=num_chunks,
            device_id=tp_id,
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
            use_ce_transfer_h2d=use_ce_transfer,
            use_ce_transfer_d2h=use_ce_transfer,
            transfer_num_cta_h2d=transfer_num_cta,
            transfer_num_cta_d2h=transfer_num_cta,
        )
    else:
        worker_handle = tpGPUCPUTransferWorker.create_worker(
            mp_ctx=mp.get_context('spawn'),
            finished_ops_queue=finished_ops_queue,
            op_buffer_tensor=op_buffer_tensor,
            gpu_blocks=[handle.get_tensor_handle_list() for handle in gpu_handles],
            cpu_blocks=cpu_handle.get_tensor(),
            gpu_kv_layouts=[gpu_handles[tp_id].kv_layout for tp_id in range(model_config.tp_size)],
            cpu_kv_layout=cpu_handle.kv_layout,
            dtype=model_config.dtype,
            tp_group_size=model_config.tp_size,
            dp_group_id=0,
            use_ce_transfer_h2d=use_ce_transfer,
            use_ce_transfer_d2h=use_ce_transfer,
            transfer_num_cta_h2d=transfer_num_cta,
            transfer_num_cta_d2h=transfer_num_cta,
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
        torch.cuda.set_device(tp_id)
        gpu_handles.append(GPUAllocator.allocate(
            layout=gpu_layout,
            dtype=model_config.dtype,
            num_chunks=num_chunks,
            device_id=tp_id,
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
            gpu_kv_layouts=[gpu_handles[tp_id].kv_layout for tp_id in range(model_config.tp_size)],
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
    if model_config.tp_size > torch.cuda.device_count():
        raise ValueError(f"TP size {model_config.tp_size} is greater than "
                         f"the number of GPUs {torch.cuda.device_count()}")

    # Determine block counts: sweep mode or single run
    if args.sweep_blocks:
        block_counts = [int(x.strip()) for x in args.sweep_blocks.split(",")]
        print(f"Sweep mode: {len(block_counts)} block counts: {block_counts}")
    else:
        block_counts = [bench_config.num_blocks_to_transfer]

    transfer_type = bench_config.transfer_type
    gpu_layout_type = bench_config.gpu_layout_type
    warmup_round = bench_config.warmup_round
    benchmark_round = bench_config.benchmark_round

    # In sweep mode, create the worker once with the maximum block count to
    # avoid paying memory-pin overhead on every step.
    max_blocks = max(block_counts)

    num_layers_to_transfer = bench_config.num_layers_to_transfer
    if num_layers_to_transfer == -1:
        num_layers_to_transfer = model_config.num_layers

    if transfer_type == TransferType.H2D or transfer_type == TransferType.D2H:
        worker_handle, finished_ops_queue = create_cpu_gpu_worker(
            model_config, cache_config, max_blocks,
            gpu_layout_type, bench_config.use_ce_transfer,
            bench_config.transfer_num_cta)
    elif transfer_type == TransferType.H2DISK or transfer_type == TransferType.DISK2H:
        worker_handle, finished_ops_queue = create_cpu_ssd_worker(model_config, cache_config)
    elif transfer_type == TransferType.DISK2D or transfer_type == TransferType.D2DISK:
        worker_handle, finished_ops_queue = create_gpu_ssd_worker(
            model_config, cache_config, max_blocks, gpu_layout_type)
    else:
        raise ValueError(f"Unsupported transfer type: {transfer_type}")

    results = []

    for num_blocks_to_transfer in block_counts:
        if bench_config.shuffle_ids:
            block_ids = torch.randperm(num_blocks_to_transfer).numpy()
        else:
            block_ids = torch.arange(num_blocks_to_transfer).numpy()

        transfer_op = TransferOp(
            transfer_type=transfer_type,
            src_block_ids=block_ids,
            dst_block_ids=block_ids,
            graph_id=0,
            dp_client_id=0,
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
            launch_transfer(worker_handle, finished_ops_queue, transfer_op)
        sync_all(finished_ops_queue, warmup_round)

        desc = f"Blocks={num_blocks_to_transfer}" if len(block_counts) > 1 else "Benchmarking"
        pbar = tqdm(total=benchmark_round, desc=desc)
        start_time = time.time()
        for _ in range(benchmark_round):
            launch_transfer(worker_handle, finished_ops_queue, transfer_op)
            pbar.update(1)
        pbar.close()
        sync_all(finished_ops_queue, benchmark_round)
        end_time = time.time()

        total_data_size_GB = (
            num_blocks_to_transfer *
            cache_config.tokens_per_block *
            model_config.token_size_in_bytes *
            num_layers_to_transfer /
            (model_config.num_layers * 1024 * 1024 * 1024)
        )
        avg_time = (end_time - start_time) / benchmark_round
        bw = total_data_size_GB / avg_time
        results.append({
            "num_blocks": num_blocks_to_transfer,
            "total_gb": total_data_size_GB,
            "avg_time_s": avg_time,
            "bw_gbps": bw,
        })
        if len(block_counts) == 1:
            print(f"Total data size: {total_data_size_GB:.2f} GB")
            print(f"Avg Time taken: {avg_time:.6f} seconds")
            print(f"Avg Bandwidth: {bw:.2f} GB/s")
        else:
            print(f"  -> {total_data_size_GB:.2f} GB | {avg_time*1000:.3f} ms | {bw:.2f} GB/s")

    worker_handle.shutdown()

    # Summary table in sweep mode
    if len(block_counts) > 1:
        print("\n" + "=" * 70)
        print(f"  Sweep Summary: {transfer_type.name} | "
              f"CE={'on' if bench_config.use_ce_transfer else 'off'} | "
              f"CTA={bench_config.transfer_num_cta}")
        print("=" * 70)
        hdr = "{:>10s}  {:>10s}  {:>12s}  {:>12s}".format(
            "Blocks", "Total GB", "Avg ms", "BW GB/s")
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for r in results:
            print("  {:>10d}  {:>10.2f}  {:>12.3f}  {:>12.2f}".format(
                r["num_blocks"], r["total_gb"],
                r["avg_time_s"] * 1000, r["bw_gbps"]))

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
    parser.add_argument("--use-ce",
                        action="store_true",
                        help="Use CE (cudaMemcpyAsync) transfer path instead of CUDA kernel")
    parser.add_argument("--cta",
                        type=int,
                        default=4,
                        help="transfer_num_cta for kernel path (default: 4)")
    parser.add_argument("--sweep-blocks",
                        type=str,
                        default=None,
                        help="Comma-separated block counts to sweep "
                             "(e.g. '64,128,256,512,1024,2048,4096')")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    bench_worker(args)
