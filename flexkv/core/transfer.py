from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Deque, Optional
from queue import Queue, Empty
from threading import Thread
from collections import deque
from flexkv.core.debug_utils import debuginfo, debug_timing
from flexkv.core.mempool import file_storage
import time

import torch

from flexkv.core.block import BlockStatus, BlockMeta
from flexkv.core.utils import DeviceType
from flexkv.c_ext import transfer_kv_layers, transfer_kv_blocks_ssd


@dataclass
class TransferDescriptor:
    device: DeviceType = DeviceType.CPU
    physical_block_ids: torch.Tensor = torch.tensor([], dtype=torch.int64)
    blockmeta_list: Optional[List[BlockMeta]] = None

    def __add__(self, other: "TransferDescriptor") -> "TransferDescriptor":
        if self.device != other.device:
            raise ValueError(
                f"Cannot merge TransferDescriptors with different "
                f"devices: {self.device} vs {other.device}"
            )

        merged_block_ids = torch.cat(
            [self.physical_block_ids, other.physical_block_ids]
        )

        merged_meta_list = None
        if self.blockmeta_list is not None and other.blockmeta_list is not None:
            merged_meta_list = self.blockmeta_list + other.blockmeta_list
        elif self.blockmeta_list is not None:
            merged_meta_list = self.blockmeta_list
        elif other.blockmeta_list is not None:
            merged_meta_list = other.blockmeta_list

        return TransferDescriptor(
            device=self.device,
            physical_block_ids=merged_block_ids,
            blockmeta_list=merged_meta_list,
        )


class TransferType(Enum):
    H2D = auto()  # Host to Device transfer
    D2H = auto()  # Device to Host transfer
    H2DISK = auto()  # Host to Disk transfer
    DISK2H = auto()  # Disk to Host transfer


@dataclass
class TransferRequest:
    transfer_id: int
    # for get requests:
    # DISK2H and H2D share the same request_id
    # for put requests:
    # H2DISK and D2H use different request_ids
    request_id: int
    type: TransferType
    src_descriptor: TransferDescriptor
    dst_descriptor: TransferDescriptor
    # used for make h2d transfer after disk2h tranfer
    return_mask: Optional[torch.Tensor]
    additional_descriptor: Optional[TransferDescriptor] = None


class Transfer(ABC):
    def __init__(self):
        self.next_transfer_id: int = 0
        self.transfer_queue: Queue = Queue()
        self.transfer_thread = Thread(target=self._transfer_worker, daemon=True)
        self.transfer_thread.start()

    def _get_next_transfer_id(self) -> int:
        transfer_id = self.next_transfer_id
        self.next_transfer_id += 1
        return transfer_id

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

    @abstractmethod
    def submit_transfer(self, request: TransferRequest) -> None:
        """
        submit a transfer request to the transfer queue
        """
        pass

    @abstractmethod
    def transfer(self, request: TransferRequest) -> None:
        """
        transfer data from source to destination
        """
        pass

    def _transfer_worker(self):
        while True:
            transfers = []
            write_transfer = None

            request = self.transfer_queue.get()
            if request is None:
                break
            # TODO we may need better scheduling of transfers here
            if request.type == TransferType.D2H or (
                request.type == TransferType.H2DISK
            ):
                write_transfer = request
            else:
                transfers.append(request)

            if not write_transfer:
                while len(transfers) < self.max_batch_size:
                    try:
                        request = self.transfer_queue.get_nowait()
                        if request is None:
                            break
                        if request.type == TransferType.H2D or (
                            request.type == TransferType.DISK2H
                        ):
                            transfers.append(request)
                        else:
                            self.transfer_queue.put(request)
                            break
                    except Empty:
                        break

            batch = [write_transfer] if write_transfer else transfers
            debuginfo.info(f"TRANSFER batch {batch[0].type} type - collected")
            self._process_transfers(batch)
            # time.sleep(0.0001)

    @abstractmethod
    def _process_transfers(self, transfers: List[TransferRequest]) -> None:
        """
        process a batch of transfers
        """
        pass

    @abstractmethod
    def pop_completed_transfers(self) -> List[TransferRequest]:
        """
        pop completed transfers from the transfer queue
        """
        pass

    def shutdown(self):
        self.transfer_queue.put(None)
        self.transfer_thread.join()


class GPUTransfer(Transfer):
    def __init__(
        self,
        gpu_blocks: List[torch.Tensor],
        cpu_blocks: List[torch.Tensor],
        max_finished: int = 1000,
        max_batch_size: int = 32,
        transfer_mode: str = "ce",
    ):
        assert len(gpu_blocks) > 0
        assert len(gpu_blocks) == len(cpu_blocks)
        assert gpu_blocks[0].is_cuda
        assert cpu_blocks[0].is_pinned
        assert gpu_blocks[0].dtype == cpu_blocks[0].dtype

        gpu_block_shape = gpu_blocks[0].shape
        cpu_block_shape = cpu_blocks[0].shape

        assert gpu_block_shape[0] == cpu_block_shape[0] == 2
        # note that vllm blocks have 5 dimentions
        # like [2, blocks, tokens_per_block, head_num, head_size]
        assert gpu_block_shape[2] == cpu_block_shape[2] or cpu_block_shape[
            2
        ] == (gpu_block_shape[2] * gpu_block_shape[3] * gpu_block_shape[4])

        self.num_layers = len(gpu_blocks)
        self.num_gpu_blocks = gpu_block_shape[1]
        self.num_cpu_blocks = cpu_block_shape[1]
        self.block_size = cpu_block_shape[2]
        self.dtype = gpu_blocks[0].dtype

        self.gpu_blocks: List[torch.Tensor] = gpu_blocks
        self.cpu_blocks: List[torch.Tensor] = cpu_blocks

        self.gpu_layer_ptrs = self._get_layer_ptrs(gpu_blocks)
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        self.gpu_kv_stride_in_bytes = (
            self.num_gpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.cpu_kv_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.gpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize

        self.transfer_queue: Queue = Queue()
        self.finished_queue: Deque = deque(maxlen=max_finished)
        self.next_transfer_id: int = 0
        self.max_batch_size = max_batch_size

        self.ssd_transfer: Optional[SSDTransfer] = None

        self.transfer_stream = torch.cuda.Stream()
        self.transfer_sms = 8
        self.transfer_mode = transfer_mode

        self.transfer_thread = Thread(target=self._transfer_worker, daemon=True)
        self.transfer_thread.start()

    def set_ssd_transfer(self, ssd_transfer: Transfer):
        self.ssd_transfer = ssd_transfer

    def transfer(
        self,
        gpu_descriptor: TransferDescriptor,
        cpu_descriptor: TransferDescriptor,
        transfer_type: TransferType,
        layer_id_list: Optional[List[int]] = None,
        non_blocking: bool = False,
    ) -> None:
        # debuginfo.info(f"running transfer gpu_descriptor: {gpu_descriptor}, "
        #      f"cpu_descriptor: {cpu_descriptor}, "
        #       f"transfer_type: {transfer_type}, "
        #      f"layer_id_list: {layer_id_list}, "
        #      f"non_blocking: {non_blocking}")
        assert gpu_descriptor.device == DeviceType.GPU
        assert cpu_descriptor.device == DeviceType.CPU

        gpu_block_id_list = gpu_descriptor.physical_block_ids.pin_memory()
        cpu_block_id_list = cpu_descriptor.physical_block_ids.pin_memory()
        block_metas_to_transfer = cpu_descriptor.blockmeta_list

        assert block_metas_to_transfer is not None
        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        if layer_id_list is None:
            layer_id_list = list(range(self.num_layers))

        chunk_size_in_bytes = self.block_size * self.dtype.itemsize

        # print(f"gpu_block_id_list: {gpu_block_id_list}")
        # print(f"cpu_block_id_list: {cpu_block_id_list}")
        if transfer_type == TransferType.H2D:
            transfer_kv_layers(
                gpu_block_id_list,
                self.gpu_layer_ptrs,
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                cpu_block_id_list,
                self.cpu_layer_ptrs,
                self.cpu_kv_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                chunk_size_in_bytes,
                self.transfer_sms,
                True, # is_host_to_device
                self.transfer_mode == "ce", # use_ce_transfer
            )
        elif transfer_type == TransferType.D2H:
            transfer_kv_layers(
                cpu_block_id_list,
                self.cpu_layer_ptrs,
                self.cpu_kv_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                gpu_block_id_list,
                self.gpu_layer_ptrs,
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                chunk_size_in_bytes,
                self.transfer_sms,
                False, # is_host_to_device
                self.transfer_mode == "ce", # use_ce_transfer
            )
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type}")

        # if non_blocking:
        torch.cuda.synchronize()  # TODO: remove this ?

    # submit a transfer request to the transfer queue
    # this is a non-blocking call
    def submit_transfer(
        self,
        request_id: int,
        type: TransferType,
        src_descriptor: TransferDescriptor,
        dst_descriptor: TransferDescriptor,
        additional_descriptor: Optional[TransferDescriptor] = None,
        return_mask: Optional[torch.Tensor] = None,
        transfer_id: Optional[int] = None,
    ) -> int:
        if transfer_id is None:
            transfer_id = self._get_next_transfer_id()
        transfer_request = TransferRequest(
            transfer_id=transfer_id,
            request_id=request_id,
            type=type,
            src_descriptor=src_descriptor,
            dst_descriptor=dst_descriptor,
            additional_descriptor=additional_descriptor,
            return_mask=return_mask,
        )
        self.transfer_queue.put(transfer_request)
        debuginfo.info(f"TRANSFER {transfer_id} - submitted")
        return transfer_id

    def _process_transfers(self, transfers):
        if not isinstance(transfers, list):
            transfers = [transfers]

        # try:
        with torch.cuda.stream(self.transfer_stream):
            # we can implement a tranfer kernel here
            # to process multiple transfers in parallel
            """
            read_gpu_descriptor = TransferDescriptor(
                device=DeviceType.GPU,
            )
            read_cpu_descriptor = TransferDescriptor(
                device=DeviceType.CPU,
            )
            write_gpu_descriptor = TransferDescriptor(
                device=DeviceType.GPU,
            )
            write_cpu_descriptor = TransferDescriptor(
                device=DeviceType.CPU,
            )
            for transfer in transfers:
                if transfer.type == TransferType.H2D:
                    read_gpu_descriptor = (
                        read_gpu_descriptor + transfer.dst_descriptor
                    )
                    read_cpu_descriptor = (
                        read_cpu_descriptor + transfer.src_descriptor
                    )
                else:
                    write_cpu_descriptor = (
                        write_cpu_descriptor + transfer.dst_descriptor
                    )
                    write_gpu_descriptor = (
                        write_gpu_descriptor + transfer.src_descriptor
                    )
            if read_gpu_descriptor.physical_block_ids.numel() > 0:
                self.transfer(
                    read_gpu_descriptor,
                    read_cpu_descriptor,
                    TransferType.H2D,
                    layer_id_list=None,
                    non_blocking=True,
                )
            if write_gpu_descriptor.physical_block_ids.numel() > 0:
                self.transfer(
                    write_gpu_descriptor,
                    write_cpu_descriptor,
                    TransferType.D2H,
                    layer_id_list=None,
                    non_blocking=True,
                )
            """
            for request in transfers:
                if request.type == TransferType.H2D:
                    cpu_descriptor = request.src_descriptor
                    gpu_descriptor = request.dst_descriptor
                elif request.type == TransferType.D2H:
                    cpu_descriptor = request.dst_descriptor
                    gpu_descriptor = request.src_descriptor
                else:
                    raise ValueError(f"Invalid transfer type: {request.type}")
                self.transfer(
                    gpu_descriptor,
                    cpu_descriptor,
                    request.type,
                    layer_id_list=None,
                    non_blocking=True,
                )
                self.transfer_stream.synchronize()
                debuginfo.info("TRANSFER stream synchronized")
                # TODO double check this logic
                self.finished_queue.append(request)
                if request.type == TransferType.D2H:
                    add_desc = request.additional_descriptor
                    if add_desc is not None and (
                        len(add_desc.physical_block_ids) > 0
                    ):
                        self.ssd_transfer.submit_transfer(
                            request.request_id,
                            TransferType.H2DISK,
                            request.dst_descriptor,
                            request.additional_descriptor,
                        )
                    else:
                        for bm in request.dst_descriptor.blockmeta_list:
                            bm.status = BlockStatus.AVAILABLE
                else:
                    # note: only when the transfer is the end of this
                    # request, we can release the blocks
                    # and block_metas always in cpu descriptor
                    for bm in request.src_descriptor.blockmeta_list:
                        bm.status = BlockStatus.AVAILABLE

        # except Exception as e:
        #    print(f"Error processing transfers: {e}")

    def pop_completed_transfers(self) -> List[TransferRequest]:
        completed = []
        while self.finished_queue:
            completed.append(self.finished_queue.popleft())
        return completed


class SSDTransfer(Transfer):
    def __init__(
        self,
        cpu_blocks: List[torch.Tensor],
        ssd_file: file_storage,
        max_finished: int = 1000,
        max_batch_size: int = 32,
        gpu_transfer: GPUTransfer = None,
        finished_queue: Deque = None,
    ):
        assert len(cpu_blocks) > 0
        assert len(cpu_blocks) == ssd_file.num_layers
        assert ssd_file.block_size == cpu_blocks[0].shape[2]
        assert ssd_file.dtype == cpu_blocks[0].dtype
        assert cpu_blocks[0].is_pinned

        cpu_block_shape = cpu_blocks[0].shape

        self.num_layers = len(cpu_blocks)
        self.num_cpu_blocks = cpu_block_shape[1]
        self.num_ssd_blocks = ssd_file.num_blocks

        self.block_size = cpu_block_shape[2]
        self.dtype = cpu_blocks[0].dtype

        self.cpu_blocks: List[torch.Tensor] = cpu_blocks
        self.ssd_file = ssd_file

        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
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
        assert gpu_transfer is not None
        if finished_queue is not None:
            self.finished_queue = finished_queue
        else:
            self.finished_queue = deque(maxlen=max_finished)
        self.gpu_transfer = gpu_transfer

        self.next_transfer_id: int = 100000
        self.max_batch_size = max_batch_size

        self.transfer_thread = Thread(target=self._transfer_worker, daemon=True)
        self.transfer_thread.start()

    # submit a transfer request to the transfer queue
    # this is a non-blocking call
    # TODO: before submitting a transfer, we need to check if the concerned
    # blocks are available in the memory, i.e., not locked by other transfers
    def submit_transfer(
        self,
        request_id: int,
        type: TransferType,
        src_descriptor: TransferDescriptor,
        dst_descriptor: TransferDescriptor,
        additional_descriptor: Optional[TransferDescriptor] = None,
        return_mask: Optional[torch.Tensor] = None,
        transfer_id: Optional[int] = None,
    ) -> int:
        if transfer_id is None:
            transfer_id = self._get_next_transfer_id()
        transfer_request = TransferRequest(
            transfer_id=transfer_id,
            request_id=request_id,
            type=type,
            src_descriptor=src_descriptor,
            dst_descriptor=dst_descriptor,
            additional_descriptor=additional_descriptor,
            return_mask=return_mask,
        )
        self.transfer_queue.put(transfer_request)
        debuginfo.info(f"TRANSFER {transfer_id} - submitted")
        return transfer_id

    def transfer(
        self,
        cpu_descriptor: TransferDescriptor,
        ssd_descriptor: TransferDescriptor,
        transfer_type: TransferType,
        layer_id_list: Optional[List[int]] = None,
        non_blocking: bool = False,
    ) -> None:
        debuginfo.info(f"ssd transfer {transfer_type} happens")
        assert ssd_descriptor.device == DeviceType.SSD
        assert cpu_descriptor.device == DeviceType.CPU
        # this means partial read hit cpu and other hit ssd
        # or partial write hit ssd and none hit cpu
        debuginfo.info(f"ssd transfer {transfer_type} happens")
        ssd_block_id_list = ssd_descriptor.physical_block_ids
        cpu_block_id_list = cpu_descriptor.physical_block_ids
        if len(ssd_block_id_list) != len(cpu_block_id_list):
            assert len(ssd_block_id_list) < len(cpu_block_id_list)
            cpu_block_id_list = cpu_descriptor.physical_block_ids[
                -len(ssd_block_id_list) :
            ]
            debuginfo.debug(
                f"cpu block id num {len(cpu_block_id_list)} "
                f"ssd block id num {len(ssd_block_id_list)}"
            )

        block_metas_to_transfer = cpu_descriptor.blockmeta_list

        assert block_metas_to_transfer is not None

        if len(ssd_block_id_list) == 0:
            return

        #if layer_id_list is None:
        #    layer_id_list = list(range(self.num_layers))

        transfer_kv_blocks_ssd(
            filename=self.ssd_file.file_path,
            cpu_layer_ptrs_tensor=self.cpu_layer_ptrs,
            ssd_block_ids=ssd_block_id_list,
            cpu_block_ids=cpu_block_id_list,
            cpu_kv_stride_in_bytes=self.cpu_kv_stride_in_bytes,
            ssd_layer_stride_in_bytes=self.ssd_layer_stride_in_bytes,
            ssd_block_stride_in_bytes=self.ssd_block_stride_in_bytes,
            ssd_kv_stride_in_bytes=self.ssd_kv_stride_in_bytes,
            block_size_in_bytes=self.chunk_size_in_bytes,
            is_read=(transfer_type == TransferType.DISK2H),
            verbose=False
        )



    def _process_transfers(self, transfers):
        if not isinstance(transfers, list):
            transfers = [transfers]
        # try:
        # since we need descriptor slicing,
        # cannot merge multiple transfers here
        for request in transfers:
            if request.type == TransferType.DISK2H:
                cpu_descriptor = request.dst_descriptor
                ssd_descriptor = request.src_descriptor
            else:
                cpu_descriptor = request.src_descriptor
                ssd_descriptor = request.dst_descriptor
            start_time = time.time()
            self.transfer(
                cpu_descriptor,
                ssd_descriptor,
                request.type,
                layer_id_list=None,
                non_blocking=True,
            )
            end_time = time.time()
            ssd_transfer_size = (
                len(ssd_descriptor.physical_block_ids)
                * self.block_size
                * self.dtype.itemsize
                * 2
                * self.num_layers
            )
            debuginfo.info(
                f"ssd tranfer request: {request.request_id} finished "
                f"request type is {request.type} "
                f"transfer data size is {ssd_transfer_size} bytes "
                f"transfer time is {end_time - start_time:.4f} s "
                f"transfer bandwidth is "
                f"{ssd_transfer_size / (end_time - start_time) / 1e9:.2f} GB/s"
            )
            # print(f"processing transfer now, request: {request}")
            if request.type == TransferType.DISK2H and (
                request.additional_descriptor is not None
            ):
                # print(f"SUBMITTING H2D transfer for request "
                #      f"src_descriptor: {request.dst_descriptor}, "
                #      f"dst_descriptor: {request.additional_descriptor}, "
                #      f"return_mask: {request.return_mask}")
                self.gpu_transfer.submit_transfer(
                    request.request_id,
                    TransferType.H2D,
                    src_descriptor=request.dst_descriptor,
                    dst_descriptor=request.additional_descriptor,
                    return_mask=request.return_mask,
                    transfer_id=request.transfer_id,
                )
            else:
                # note: only when the transfer is the end of this
                # request, we can release the blocks
                # and block_metas always in cpu descriptor
                for bm in request.src_descriptor.blockmeta_list:
                    bm.status = BlockStatus.AVAILABLE
                # self.finished_queue.append(request)

        # except Exception as e:
        #     print(f"Error processing transfers: {e}")

    def pop_completed_transfers(self) -> List[TransferRequest]:
        completed = []
        while self.finished_queue:
            completed.append(self.finished_queue.popleft())
        return completed


if __name__ == "__main__":
    debuginfo.set_level("DEBUG")
    num_layers = 10
    num_gpu_blocks = 48
    num_cpu_blocks = 48
    block_size = 16 * 768 * 128  # tokens_per_block * num_kv_heads * head_dim
    dtype = torch.float16
    gpu_blocks = [
        torch.ones(2, num_gpu_blocks, block_size, dtype=dtype).cuda()
        for _ in range(num_layers)
    ]
    cpu_blocks = [
        torch.zeros(2, num_cpu_blocks, block_size, dtype=dtype).pin_memory()
        for _ in range(num_layers)
    ]
    transfer = GPUTransfer(
        gpu_blocks=gpu_blocks,
        cpu_blocks=cpu_blocks,
        transfer_mode="sm",
    )

    num_blocks_to_transfer = 20
    gpu_physical_block_ids = torch.randperm(
        num_gpu_blocks,
        device=torch.device("cpu"),
        pin_memory=True,
    )[:num_blocks_to_transfer]
    cpu_physical_block_ids = torch.randperm(
        num_cpu_blocks,
        device=torch.device("cpu"),
        pin_memory=True,
    )[:num_blocks_to_transfer]
    block_metas = [
        BlockMeta(
            cpu_block_id=cpu_physical_block_ids[i],
            status=BlockStatus.LOCKED,
        )
        for i in range(num_blocks_to_transfer)
    ]
    cpu_descriptor = TransferDescriptor(
        device=DeviceType.CPU,
        physical_block_ids=cpu_physical_block_ids,
        blockmeta_list=block_metas,
    )
    gpu_descriptor = TransferDescriptor(
        device=DeviceType.GPU,
        physical_block_ids=gpu_physical_block_ids,
        blockmeta_list=[],
    )
    start_time = time.time()
    tid = transfer.submit_transfer(
        0, TransferType.D2H, gpu_descriptor, cpu_descriptor
    )
    print(f"Transfer ID: {tid}")
    while True:
        completed_transfers = transfer.pop_completed_transfers()
        if len(completed_transfers) > 0:
            break
        time.sleep(0.0001)

    end_time = time.time()
    elapsed_time = end_time - start_time
    data_size_in_GB = (
        num_blocks_to_transfer
        * block_size
        * num_layers
        * dtype.itemsize
        * 2
        / 1e9
    )
    bandwidth = data_size_in_GB / elapsed_time
    print(
        f"Transfer time: {elapsed_time:.6f} s, "
        f"data size: {data_size_in_GB:.2f} GB, "
        f"bandwidth: {bandwidth:.2f} GB/s"
    )

    for i in range(num_layers):
        for j in range(num_blocks_to_transfer):
            cpu_tensor_k = cpu_blocks[i][0, cpu_physical_block_ids[j], :].cuda()
            cpu_tensor_v = cpu_blocks[i][1, cpu_physical_block_ids[j], :].cuda()
            gpu_tensor_k = gpu_blocks[i][0, gpu_physical_block_ids[j], :]
            gpu_tensor_v = gpu_blocks[i][1, gpu_physical_block_ids[j], :]
            # print(cpu_tensor_k, gpu_tensor_k)
            assert torch.allclose(cpu_tensor_k, gpu_tensor_k)
            assert torch.allclose(cpu_tensor_v, gpu_tensor_v)
