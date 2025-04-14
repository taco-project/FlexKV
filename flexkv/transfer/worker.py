from abc import ABC, abstractmethod
from typing import List
import torch
import queue
import threading
from flexkv.common.transfer import TransferOp, TransferType, DeviceType
from flexkv.common.storage import AccessibleHandle, AccessHandleType

class TransferWorker(ABC):
    def __init__(self, worker_id: int, finished_queue: deque):
        self.worker_id = worker_id
        self.transfer_queue = queue.Queue()
        self.max_batch_size = 32
        self.finished_queue = finished_queue
        self.transfer_thread = threading.Thread(target=self._transfer_worker)
        self.transfer_thread.start()
    
    def submit_transfer(self, transfer: TransferOp):
        self.transfer_queue.put(transfer)

    @abstractmethod
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
    def transfer(self, transfer: TransferOp)->None:
        pass
    
    @abstractmethod
    def _process_transfers(self, transfers: List[TransferOp])->None:
        pass

    def _transfer_worker(self):
        while True:
            transfers = []
            write_transfer = None

            request = self.transfer_queue.get()
            if request is None:
                break
            if request.type % 2 == 1: #write request
                write_transfer = request
            else: #read request
                transfers.append(request)

            if not write_transfer:
                while len(transfers) < self.max_batch_size:
                    try:
                        request = self.transfer_queue.get_nowait()
                        if request is None:
                            break
                        if request.type % 2 == 0:
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

    def shutdown(self):
        self.transfer_queue.put(None)
        self.transfer_thread.join()

class GPUCPUTransferWorker(TransferWorker):
    def __init__(
        self,
        worker_id: int,
        gpu_handle: AccessibleHandle,
        cpu_handle: AccessibleHandle,
        finished_queue: Deque,
        max_batch_size: int = 32,
        gpu_device_id: int = -1,
    ):
        assert gpu_handle.handle_type == AccessHandleType.TENSOR_LIST
        assert cpu_handle.handle_type == AccessHandleType.TENSOR_LIST
        
        assert gpu_handle.dtype == cpu_handle.dtype

        gpu_block_shape = gpu_handle.kv_shape
        cpu_block_shape = cpu_handle.kv_shape

        print("gpu_block_shape", gpu_block_shape)
        print("cpu_block_shape", cpu_block_shape)
        
        #NOTE: this may be different for different frameworks or kvcache layout
        self.num_layers = gpu_handle.kv_shape[0]
        self.num_gpu_blocks = gpu_handle.kv_shape[2]
        self.num_cpu_blocks = cpu_handle.kv_shape[2]
        self.block_size = cpu_handle.kv_shape[3]
        self.dtype = gpu_handle.dtype

        self.gpu_layer_ptrs = gpu_handle.data
        self.cpu_layer_ptrs = cpu_handle.data
        self.gpu_kv_stride_in_bytes = (
            self.num_gpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.cpu_kv_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.gpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize

        self.transfer_queue: Queue = Queue()
        self.max_batch_size = max_batch_size

        self.transfer_stream = torch.cuda.Stream()
        self.transfer_sms = 4

        self.gpu_device_id = gpu_device_id
        if gpu_device_id != -1:
            torch.cuda.set_device(gpu_device_id) #TODO is this enough?
            self.transfer_stream = torch.cuda.Stream(device=f"cuda:{gpu_device_id}")

        self.transfer_thread = Thread(target=self._transfer_worker, daemon=True)
        self.transfer_thread.start()

    def transfer(
        self,
        gpu_descriptor: TransferDescriptor,
        cpu_descriptor: TransferDescriptor,
        transfer_type: TransferType,
        layer_id_list: Optional[List[int]] = None,
        non_blocking: bool = False,
    ) -> None:
        assert gpu_descriptor.device_type == DeviceType.GPU
        assert cpu_descriptor.device_type == DeviceType.CPU

        gpu_block_id_list = gpu_descriptor.physical_block_ids.pin_memory()
        cpu_block_id_list = cpu_descriptor.physical_block_ids.pin_memory()

        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        if layer_id_list is None:
            layer_id_list = list(range(self.num_layers))

        chunk_size_in_bytes = self.block_size * self.dtype.itemsize

        if transfer_type == TransferType.H2D:
            transfer_kv_layers(
                gpu_block_id_list,
                self.gpu_layer_ptrs[layer_id_list],
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                cpu_block_id_list,
                self.cpu_layer_ptrs[layer_id_list],
                self.cpu_kv_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                chunk_size_in_bytes,
                self.transfer_sms,
            )
        elif transfer_type == TransferType.D2H:
            transfer_kv_layers(
                cpu_block_id_list,
                self.cpu_layer_ptrs[layer_id_list],
                self.cpu_kv_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                gpu_block_id_list,
                self.gpu_layer_ptrs[layer_id_list],
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                chunk_size_in_bytes,
                self.transfer_sms,
            )
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type}")

        # if non_blocking:
        torch.cuda.synchronize()  # TODO: remove this ?

    def _process_transfers(self, transfers):
        if not isinstance(transfers, list):
            transfers = [transfers]

        with torch.cuda.stream(self.transfer_stream):
            # we can implement a tranfer kernel here
            # to process multiple transfers in parallel
            for op in transfers:
                if op.transfer_type == TransferType.H2D:
                    cpu_descriptor = op.src_descriptor
                    gpu_descriptor = op.dst_descriptor
                elif op.transfer_type == TransferType.D2H:
                    cpu_descriptor = op.dst_descriptor
                    gpu_descriptor = op.src_descriptor
                else:
                    raise ValueError(f"Invalid transfer type: {op.transfer_type}")
                self.transfer(
                    gpu_descriptor,
                    cpu_descriptor,
                    op.transfer_type,
                    layer_id_list=gpu_descriptor.layers,
                    non_blocking=True,
                )
                #self.transfer_stream.synchronize()
                debuginfo.info("TRANSFER stream synchronized")
                self.finished_queue.append(op)
                '''
                if op.transfer_type == TransferType.D2H:
                    add_desc = op.additional_descriptor
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
                '''

class CPUSSDDiskTransferWorker(TransferWorker):
    def __init__(
        self,
        worker_id: int,
        cpu_handle: AccessibleHandle,
        ssd_handle: AccessibleHandle,
        max_batch_size: int = 32,
        finished_queue: Deque = None,
    ):
        cpu_block_shape = cpu_handle.kv_shape
        ssd_block_shape = ssd_handle.kv_shape

        assert ssd_handle.dtype == cpu_handle.dtype

        #NOTE: this may be different for different frameworks or kvcache layout
        self.num_layers = cpu_handle.kv_shape[0]
        self.num_cpu_blocks = cpu_handle.kv_shape[2]
        self.num_ssd_blocks = ssd_handle.kv_shape[2]

        self.block_size = cpu_handle.kv_shape[3]
        self.dtype = cpu_handle.dtype

        self.cpu_layer_ptrs = cpu_handle.data
        self.ssd_file = ssd_handle.data

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
        self.finished_queue = finished_queue

        self.max_batch_size = max_batch_size

        self.transfer_thread = Thread(target=self._transfer_worker, daemon=True)
        self.transfer_thread.start()

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

        if len(ssd_block_id_list) == 0:
            return

        if layer_id_list is None:
            layer_id_list = list(range(self.num_layers))
        #TODO add layer_id_list support

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
            is_read=(transfer_type == TransferType.DISK2H),
            verbose=False
        )


    def _process_transfers(self, transfers):
        if not isinstance(transfers, list):
            transfers = [transfers]
        # try:
        # since we need descriptor slicing,
        # cannot merge multiple transfers here
        for op in transfers:
            if op.transfer_type == TransferType.DISK2H:
                cpu_descriptor = op.dst_descriptor
                ssd_descriptor = op.src_descriptor
            else:
                cpu_descriptor = op.src_descriptor
                ssd_descriptor = op.dst_descriptor
            start_time = time.time()
            self.transfer(
                cpu_descriptor,
                ssd_descriptor,
                op.transfer_type,
                layer_id_list=cpu_descriptor.layers,
                non_blocking=True,
            )
            self.finished_queue.append(op)
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
            '''
            if op.transfer_type == TransferType.DISK2H and (
                op.additional_descriptor is not None
            ):
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
            '''