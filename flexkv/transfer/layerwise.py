import copy
import torch.multiprocessing as mp
import threading
import time
import os
import socket
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.multiprocessing import Queue as MPQueue, Pipe as MPPipe
from multiprocessing.connection import Connection
from threading import Thread
from typing import List, Any, Dict, Union, Optional, Tuple

import ctypes
import numpy as np
import nvtx
import torch

from flexkv import c_ext

from flexkv.c_ext import LayerwiseTransferGroup
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferOp, TransferType, PartitionBlockType
from flexkv.common.transfer import get_nvtx_range_color
from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV

from flexkv.transfer.worker_op import WorkerTransferOp, WorkerLayerwiseTransferOp
from flexkv.transfer.worker import TransferWorkerBase, cudaHostRegister


def _recv_fds(sock: socket.socket, num_fds: int) -> Tuple[List[int], bytes]:
    """Receive multiple fds + extra_data via Unix domain socket (SCM_RIGHTS)."""
    data_buf = bytearray(256)
    anc_buf_size = socket.CMSG_SPACE(num_fds * struct.calcsize("i"))
    
    nbytes, ancdata, flags, addr = sock.recvmsg_into([data_buf], anc_buf_size, anc_buf_size)
    data = bytes(data_buf[:nbytes])

    fds = []
    for level, ctype, cdata in ancdata:
        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
            num_received = len(cdata) // struct.calcsize("i")
            fds = list(struct.unpack(f"{num_received}i", cdata[:num_received * struct.calcsize("i")]))
            break
    if not fds:
        raise RuntimeError("did not receive fds via SCM_RIGHTS")
    return fds, data

class LayerwiseTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 gpu_blocks: List[List[TensorSharedHandle]],
                 cpu_blocks: torch.Tensor,
                 ssd_files: Dict[int, List[str]],
                 gpu_kv_layouts: List[KVCacheLayout],
                 cpu_kv_layout: KVCacheLayout,
                 ssd_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 tp_group_size: int,
                 dp_group_id: int,
                 num_blocks_per_file: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 h2d_cta_num: int = 4,
                 d2h_cta_num: int = 4) -> None:
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        assert len(gpu_blocks) == tp_group_size, f"len(gpu_blocks) = {len(gpu_blocks)}, tp_group_size = {tp_group_size}"
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            blocks_in_one_gpu = []
            for handle in handles_in_one_gpu:
                blocks_in_one_gpu.append(handle.get_tensor())
            imported_gpu_blocks.append(blocks_in_one_gpu)
        self.gpu_blocks = imported_gpu_blocks
        self.dtype = dtype # note this should be quantized data type
        self.is_mla = gpu_kv_layouts[0].is_mla

        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.dp_group_id = dp_group_id

        # initialize GPU storage
        self.num_layers = gpu_kv_layouts[0].num_layer
        # here the chunk size doesn't include the layer info
        self.gpu_chunk_sizes_in_bytes = [gpu_kv_layout.get_chunk_size() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_kv_strides_in_bytes = [gpu_kv_layout.get_kv_stride() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_block_strides_in_bytes = [gpu_kv_layout.get_block_stride() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_layer_strides_in_bytes = [gpu_kv_layout.get_layer_stride() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]

        num_blocks_first_gpu = len(imported_gpu_blocks[0]) if imported_gpu_blocks else 0
        if num_blocks_first_gpu == 1:
            self.gpu_block_type_ = 1  # TRTLLM
        elif num_blocks_first_gpu == self.num_layers:
            self.gpu_block_type_ = 0  # VLLM
        elif num_blocks_first_gpu == self.num_layers * 2:
            self.gpu_block_type_ = 2  # SGLANG
        else:
            raise ValueError(f"Invalid GPU block type: {num_blocks_first_gpu}")

        # initialize CPU storage
        flexkv_logger.info(f"Pinning CPU Memory: "
                           f"{cpu_blocks.numel() * cpu_blocks.element_size() / (1024 ** 3):.2f} GB")
        cudaHostRegister(cpu_blocks)
        self.cpu_blocks = cpu_blocks

        self.cpu_chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
        self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize
        self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize

        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h
        self.h2d_cta_num = h2d_cta_num
        self.d2h_cta_num = d2h_cta_num

        # initialize SSD storage
        self.enable_ssd = len(ssd_files) > 0
        self.ssd_files = ssd_files
        if self.enable_ssd:
            self.num_blocks_per_file = num_blocks_per_file
            self.num_files = sum(len(file_list) for file_list in ssd_files.values())
            self.round_robin = 1

            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)
            self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            self.ssd_block_stride_in_bytes = ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize
        else:
            self.num_blocks_per_file = 0
            self.round_robin = 1
            self.ssd_kv_stride_in_bytes = 0
            self.ssd_layer_stride_in_bytes = 0
            self.ssd_block_stride_in_bytes = 0

        gpu_kv_strides_tensor = torch.tensor(self.gpu_kv_strides_in_bytes, dtype=torch.int64)
        gpu_block_strides_tensor = torch.tensor(self.gpu_block_strides_in_bytes, dtype=torch.int64)
        gpu_chunk_sizes_tensor = torch.tensor(self.gpu_chunk_sizes_in_bytes, dtype=torch.int64)
        gpu_layer_strides_tensor = torch.tensor(self.gpu_layer_strides_in_bytes, dtype=torch.int64)

        layer_eventfds_tensor = self._receive_eventfds_from_sglang(tp_group_size)

        # Create LayerwiseTransferGroup which handles both SSD->CPU and CPU->GPU transfers
        self.layerwise_transfer_group = LayerwiseTransferGroup(
            self.num_gpus, self.gpu_blocks, cpu_blocks, ssd_files,
            dp_group_id, self.num_layers,
            gpu_kv_strides_tensor, gpu_block_strides_tensor,
            gpu_layer_strides_tensor, gpu_chunk_sizes_tensor,
            GLOBAL_CONFIG_FROM_ENV.iouring_entries,
            GLOBAL_CONFIG_FROM_ENV.iouring_flags,
            layer_eventfds_tensor, tp_group_size)

    def _receive_eventfds_from_sglang(self, tp_group_size: int, 
                                       max_retries: int = 180, 
                                       retry_interval: float = 1.0) -> torch.Tensor:
        """Receive eventfds from SGLang via Unix socket (FlexKV as server)."""
        socket_path = os.environ.get('FLEXKV_LAYERWISE_EVENTFD_SOCKET', '/tmp/flexkv_layerwise_eventfd.sock')
        
        def cleanup_socket():
            try:
                if os.path.exists(socket_path):
                    os.unlink(socket_path)
            except OSError:
                pass
        
        cleanup_socket()
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_sock.bind(socket_path)
            server_sock.listen(tp_group_size)
            os.chmod(socket_path, 0o777)
            flexkv_logger.info(f"[LayerwiseWorker] Listening on {socket_path}, waiting for {tp_group_size} connections")
        except Exception as e:
            flexkv_logger.error(f"[LayerwiseWorker] Failed to bind/listen: {e}")
            server_sock.close()
            return torch.empty(0, dtype=torch.int32)
        
        server_sock.settimeout(max_retries * retry_interval)
        all_rank_eventfds: Dict[int, Dict[int, List[int]]] = {}
        num_layers, num_counters = self.num_layers, 3
        
        try:
            for conn_idx in range(tp_group_size):
                try:
                    conn, _ = server_sock.accept()
                except socket.timeout:
                    flexkv_logger.warning(f"[LayerwiseWorker] Timeout, received {conn_idx}/{tp_group_size}")
                    break
                
                with conn:
                    metadata = conn.recv(16)
                    if len(metadata) < 16:
                        flexkv_logger.error(f"[LayerwiseWorker] Incomplete metadata: {len(metadata)} bytes")
                        continue
                    
                    tp_rank, _, recv_num_layers, recv_num_counters = struct.unpack("iiii", metadata)
                    if conn_idx == 0:
                        num_layers, num_counters = recv_num_layers, recv_num_counters
                    
                    rank_eventfds = {}
                    for _ in range(recv_num_counters):
                        fds, extra_data = _recv_fds(conn, recv_num_layers)
                        counter_id = struct.unpack("i", extra_data[:4])[0]
                        rank_eventfds[counter_id] = fds
                    
                    all_rank_eventfds[tp_rank] = rank_eventfds
                    flexkv_logger.info(f"[LayerwiseWorker] Received eventfds from tp_rank={tp_rank}")
        except Exception as e:
            flexkv_logger.error(f"[LayerwiseWorker] Error in accept loop: {e}")
        finally:
            server_sock.close()
            cleanup_socket()
        
        if not all_rank_eventfds:
            flexkv_logger.warning("[LayerwiseWorker] No connections received")
            return torch.empty(0, dtype=torch.int32)
        
        # Build tensor: [num_counters, tp_size, num_layers]
        eventfds_list = []
        for counter_id in range(num_counters):
            for tp_rank in range(tp_group_size):
                fds = all_rank_eventfds.get(tp_rank, {}).get(counter_id, [-1] * num_layers)
                eventfds_list.extend(fds)
        
        tensor = torch.tensor(eventfds_list, dtype=torch.int32)
        flexkv_logger.info(f"[LayerwiseWorker] Eventfds tensor: {tensor.shape}, counters={num_counters}, tp={tp_group_size}, layers={num_layers}")
        return tensor

    def _transfer_impl(self,
                      src_block_ids_h2d: torch.Tensor,
                      dst_block_ids_h2d: torch.Tensor,
                      src_block_ids_disk2h: Optional[torch.Tensor],
                      dst_block_ids_disk2h: Optional[torch.Tensor],
                      layer_granularity: int,
                      counter_id: int = 0,
                      **kwargs: Any) -> None:
        assert src_block_ids_h2d.dtype == torch.int64
        assert dst_block_ids_h2d.dtype == torch.int64
        assert len(src_block_ids_h2d) == len(dst_block_ids_h2d)
        if src_block_ids_disk2h is not None:
            assert src_block_ids_disk2h.dtype == torch.int64
            assert dst_block_ids_disk2h.dtype == torch.int64
            assert len(src_block_ids_disk2h) == len(dst_block_ids_disk2h)

        # Use unified layerwise transfer C++ interface
        ssd_block_ids = src_block_ids_disk2h if src_block_ids_disk2h is not None else torch.empty(0, dtype=torch.int64)
        cpu_block_ids_d2h = dst_block_ids_disk2h if dst_block_ids_disk2h is not None \
            else torch.empty(0, dtype=torch.int64)

        self.layerwise_transfer_group.layerwise_transfer(
            ssd_block_ids,
            cpu_block_ids_d2h,
            self.ssd_layer_stride_in_bytes,
            self.ssd_kv_stride_in_bytes,
            self.num_blocks_per_file,
            self.round_robin,
            32,  # num_threads_per_device
            dst_block_ids_h2d,
            src_block_ids_h2d,
            self.cpu_kv_stride_in_bytes,
            self.cpu_layer_stride_in_bytes,
            self.cpu_block_stride_in_bytes,
            self.cpu_chunk_size_in_bytes,
            self.h2d_cta_num,
            self.use_ce_transfer_h2d,
            self.num_layers,
            layer_granularity,
            self.is_mla,
            counter_id,
        )

    def launch_transfer(self, transfer_op: WorkerLayerwiseTransferOp) -> None:
        layer_granularity = transfer_op.layer_granularity
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        src_block_ids_h2d = torch.from_numpy(transfer_op.src_block_ids_h2d).to(dtype=torch.int64).pin_memory()
        dst_block_ids_h2d = torch.from_numpy(transfer_op.dst_block_ids_h2d).to(dtype=torch.int64).pin_memory()

        if transfer_op.src_block_ids_disk2h.size > 0:
            src_block_ids_disk2h = torch.from_numpy(transfer_op.src_block_ids_disk2h).to(dtype=torch.int64)
            dst_block_ids_disk2h = torch.from_numpy(transfer_op.dst_block_ids_disk2h).to(dtype=torch.int64)
        else:
            src_block_ids_disk2h = None
            dst_block_ids_disk2h = None

        self._transfer_impl(
            src_block_ids_h2d,
            dst_block_ids_h2d,
            src_block_ids_disk2h,
            dst_block_ids_disk2h,
            layer_granularity,
            transfer_op.counter_id,
        )
