from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
from queue import Queue
import threading
from flexkv.common.transfer import TransferOp, TransferType, DeviceType, TransferDescriptor
from flexkv.common.storage import AccessibleHandle, AccessHandleType
from flexkv.c_ext import transfer_kv_layers, transfer_kv_blocks_ssd
import time
from threading import Thread
import queue
from flexkv.common.debug import debuginfo
import torch.cuda.nvtx as nvtx
class TransferWorker(ABC):
    def __init__(self, worker_id: int, finished_ops_queue: Queue):
        self.worker_id = worker_id
        self.transfer_queue = Queue()
        self.finished_ops_queue = finished_ops_queue
        self.transfer_thread = threading.Thread(target=self._transfer_worker)
        self.transfer_thread.start()

    def submit_transfer(self, transfer: TransferOp):
        self.transfer_queue.put(transfer)
    """
    def _get_layer_ptrs(self, layer_blocks: List[torch.Tensor]) -> torch.Tensor:
        layer_ptrs = torch.zeros(
            self.num_layers,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        for lay_id in range(self.num_layers):
            layer_ptrs[lay_id] = layer_blocks[lay_id][0].data_ptr()
        return layer_ptrs
    """

    @abstractmethod
    def _transfer_impl(self, transfer: TransferOp)->None:
        pass

    @abstractmethod
    def launch_transfer(self, transfers: List[TransferOp])->None:
        pass

    def _transfer_worker(self):
        while True:
            batch_op = None
            try:
                op = self.transfer_queue.get(timeout=0.001)
                if op is None:
                    break
                batch_op = op
            except queue.Empty:
                continue

            # TODO: support batch transfer
            if batch_op:
                try:
                    self.launch_transfer(batch_op)
                except Exception as e:
                    debuginfo.error(f"Error launching transfer: {e}\n"
                                    f"Failed transfer op: {batch_op}")
                    # self.transfer_queue.put(batch_op)
                self.finished_ops_queue.put(batch_op)

    def shutdown(self):
        self.transfer_queue.put(None)
        self.transfer_thread.join()

class GPUCPUTransferWorker(TransferWorker):
    def __init__(
        self,
        worker_id: int,
        gpu_blocks_ptrs: torch.Tensor,
        cpu_blocks_ptrs: torch.Tensor,
        finished_ops_queue: Queue,
        gpu_kv_shape: Tuple[int, ...],
        cpu_kv_shape: Tuple[int, ...],
        dtype: torch.dtype,
        gpu_device_id: int = -1,
    ):
        assert len(gpu_blocks_ptrs) == len(cpu_blocks_ptrs)
        self.finished_ops_queue = finished_ops_queue
        #NOTE: this may be different for different frameworks or kvcache layout
        self.num_layers = len(gpu_blocks_ptrs)
        self.num_gpu_blocks = gpu_kv_shape[2]
        self.num_cpu_blocks = cpu_kv_shape[2]
        self.block_size = cpu_kv_shape[3]
        self.dtype = dtype

        self.gpu_layer_ptrs = gpu_blocks_ptrs
        self.cpu_layer_ptrs = cpu_blocks_ptrs
        self.gpu_kv_stride_in_bytes = (
            self.num_gpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.cpu_kv_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.gpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize

        self.transfer_queue: Queue = Queue()

        self.transfer_stream = torch.cuda.Stream()
        self.transfer_sms = 4

        self.gpu_device_id = gpu_device_id
        if gpu_device_id != -1:
            torch.cuda.set_device(gpu_device_id) #TODO is this enough?
            self.transfer_stream = torch.cuda.Stream(device=f"cuda:{gpu_device_id}")

        self.transfer_thread = Thread(target=self._transfer_worker, daemon=True)
        self.transfer_thread.start()

    def _transfer_impl(
        self,
        gpu_descriptor: TransferDescriptor,
        cpu_descriptor: TransferDescriptor,
        transfer_type: TransferType,
        layer_id: int = -1,
        layer_granularity: int = -1,
        non_blocking: bool = False,
    ) -> None:
        assert gpu_descriptor.device_type == DeviceType.GPU
        assert cpu_descriptor.device_type == DeviceType.CPU

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        gpu_block_id_list = gpu_descriptor.physical_block_ids.pin_memory().to(dtype=torch.int64)
        cpu_block_id_list = cpu_descriptor.physical_block_ids.pin_memory().to(dtype=torch.int64)

        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)

        chunk_size_in_bytes = self.block_size * self.dtype.itemsize
        nvtx.range_push("Transfer KV Layers")
        if transfer_type == TransferType.H2D:
            transfer_kv_layers(
                gpu_block_id_list,
                self.gpu_layer_ptrs[layer_id_list].pin_memory(),
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                cpu_block_id_list,
                self.cpu_layer_ptrs[layer_id_list].pin_memory(),
                self.cpu_kv_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                chunk_size_in_bytes,
                self.transfer_sms,
                True,
                False,
            )
        elif transfer_type == TransferType.D2H:
            transfer_kv_layers(
                cpu_block_id_list,
                self.cpu_layer_ptrs[layer_id_list].pin_memory(),
                self.cpu_kv_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                gpu_block_id_list,
                self.gpu_layer_ptrs[layer_id_list].pin_memory(),
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                chunk_size_in_bytes,
                self.transfer_sms,
                False,
                False,
            )
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type}")
        nvtx.range_pop()
        # if non_blocking:
        #torch.cuda.synchronize()  # TODO: remove this ?

    def launch_transfer(self, transfer_op: TransferOp):
        with torch.cuda.stream(self.transfer_stream):
            if transfer_op.transfer_type == TransferType.H2D:
                cpu_descriptor = transfer_op.src_descriptor
                gpu_descriptor = transfer_op.dst_descriptor
            elif transfer_op.transfer_type == TransferType.D2H:
                cpu_descriptor = transfer_op.dst_descriptor
                gpu_descriptor = transfer_op.src_descriptor
            else:
                raise ValueError(f"Invalid transfer type: {transfer_op.transfer_type}")
            start_time = time.time()
            nvtx.range_push("GPU CPUTransfer")
            self._transfer_impl(
                gpu_descriptor,
                cpu_descriptor,
                transfer_op.transfer_type,
                layer_id=transfer_op.layer_id,
                layer_granularity=transfer_op.layer_granularity,
                non_blocking=True,
            )
            nvtx.range_pop()
            self.transfer_stream.synchronize()
            # debuginfo.info("TRANSFER stream synchronized")
            end_time = time.time()
            transfer_size = (
                len(gpu_descriptor.physical_block_ids)
                * self.block_size
                * self.dtype.itemsize
                * 2
                * transfer_op.layer_granularity
            )
            debuginfo.info(
                f"gpu cpu tranfer request: {transfer_op.transfer_op_id} finished "
                f"request type is {transfer_op.transfer_type} "
                f"transfer data size is {transfer_size} bytes "
                f"transfer time is {end_time - start_time:.4f} s "
                f"transfer bandwidth is "
                f"{transfer_size / (end_time - start_time) / 1e9:.2f} GB/s"
            )

class CPUSSDDiskTransferWorker(TransferWorker):
    def __init__(
        self,
        worker_id: int,
        cpu_block_ptrs: torch.Tensor,
        ssd_file: str,
        cpu_kv_shape: Tuple[int, ...],
        ssd_kv_shape: Tuple[int, ...],
        dtype: torch.dtype,
        finished_ops_queue: Queue = None,
    ):

        #NOTE: this may be different for different frameworks or kvcache layout
        self.num_layers = cpu_kv_shape[0]
        self.num_cpu_blocks = cpu_kv_shape[2]
        self.num_ssd_blocks = ssd_kv_shape[2]

        self.block_size = cpu_kv_shape[3]
        self.dtype = dtype

        self.cpu_layer_ptrs = cpu_block_ptrs
        self.ssd_file = ssd_file

        self.cpu_layer_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize * 2
        )
        self.ssd_layer_stride_in_bytes = (
            self.num_ssd_blocks * self.block_size * self.dtype.itemsize * 2
        )
        self.cpu_kv_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.ssd_kv_stride_in_bytes = (
            self.num_ssd_blocks * self.block_size * self.dtype.itemsize
        )
        self.ssd_block_stride_in_bytes = self.block_size * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize

        self.chunk_size_in_bytes = self.block_size * self.dtype.itemsize

        self.transfer_queue: Queue = Queue()
        self.finished_ops_queue = finished_ops_queue

        self.transfer_thread = Thread(target=self._transfer_worker, daemon=True)
        self.transfer_thread.start()

    def _transfer_impl(
        self,
        cpu_descriptor: TransferDescriptor,
        ssd_descriptor: TransferDescriptor,
        transfer_type: TransferType,
        layer_id: int = -1,
        layer_granularity: int = -1,
        non_blocking: bool = False,
    ) -> None:
        debuginfo.info(f"ssd transfer {transfer_type} happens")
        assert ssd_descriptor.device_type == DeviceType.SSD
        assert cpu_descriptor.device_type == DeviceType.CPU

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        # this means partial read hit cpu and other hit ssd
        # or partial write hit ssd and none hit cpu
        debuginfo.info(f"ssd transfer {transfer_type} happens")
        ssd_block_id_list = ssd_descriptor.physical_block_ids.pin_memory().to(dtype=torch.int64)
        cpu_block_id_list = cpu_descriptor.physical_block_ids.pin_memory().to(dtype=torch.int64)
        if len(ssd_block_id_list) != len(cpu_block_id_list):
            assert len(ssd_block_id_list) < len(cpu_block_id_list)
            cpu_block_id_list = cpu_descriptor.physical_block_ids[
                -len(ssd_block_id_list) :
            ]
            debuginfo.debug(
                f"cpu block id num {len(cpu_block_id_list)} "
                f"ssd block id num {len(ssd_block_id_list)}"
            )

        if len(ssd_block_id_list) == 0:
            return

        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)

        transfer_kv_blocks_ssd(
            filename=self.ssd_file,
            cpu_layer_id_list=layer_id_list,
            cpu_layer_ptrs_tensor=self.cpu_layer_ptrs,
            ssd_block_ids=ssd_block_id_list,
            cpu_block_ids=cpu_block_id_list,
            cpu_kv_stride_in_bytes=self.cpu_kv_stride_in_bytes,
            ssd_layer_stride_in_bytes=self.ssd_layer_stride_in_bytes,
            ssd_block_stride_in_bytes=self.ssd_block_stride_in_bytes,
            ssd_kv_stride_in_bytes=self.ssd_kv_stride_in_bytes,
            block_size_in_bytes=self.chunk_size_in_bytes,
            total_layers=self.num_layers,
            is_read=(transfer_type == TransferType.DISK2H),
            verbose=False
        )

    def launch_transfer(self, transfer_op: TransferOp):
        if transfer_op.transfer_type == TransferType.DISK2H:
            cpu_descriptor = transfer_op.dst_descriptor
            ssd_descriptor = transfer_op.src_descriptor
        else:
            cpu_descriptor = transfer_op.src_descriptor
            ssd_descriptor = transfer_op.dst_descriptor
        start_time = time.time()
        nvtx.range_push("CPU SSD Transfer")
        self._transfer_impl(
            cpu_descriptor,
            ssd_descriptor,
            transfer_op.transfer_type,
            layer_id=transfer_op.layer_id,
            layer_granularity=transfer_op.layer_granularity,
            non_blocking=True,
        )
        nvtx.range_pop()
        end_time = time.time()
        transfer_size = (
            len(ssd_descriptor.physical_block_ids)
            * self.block_size
            * self.dtype.itemsize
            * 2
            * self.num_layers
        )
        debuginfo.info(
            f"ssd tranfer request: {transfer_op.transfer_op_id} finished "
            f"request type is {transfer_op.transfer_type} "
            f"transfer data size is {transfer_size} bytes "
            f"transfer time is {end_time - start_time:.4f} s "
            f"transfer bandwidth is "
            f"{transfer_size / (end_time - start_time) / 1e9:.2f} GB/s"
        )
