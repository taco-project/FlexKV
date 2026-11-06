"""Peer-to-peer CPU <-> CPU / peer-SSD -> CPU transfers.

Moved here whole. Unlike the other edges this worker owns a control plane of its
own (ZMQ server/client, Redis metadata, a mooncake RDMA engine), which is why it
is the one module that pulls those dependencies in.
"""

import json
import os
import threading
import time
from multiprocessing.connection import Connection
from typing import Dict, List, Optional, Tuple, Union

import nvtx
import torch
import zmq
from torch.multiprocessing import Queue as MPQueue

from flexkv import c_ext
from flexkv.c_ext import transfer_kv_blocks_ssd
from flexkv.cache.redis_meta import RedisMeta
from flexkv.common.config import (
    CacheConfig,
    GLOBAL_CONFIG_FROM_ENV,
    MooncakeTransferEngineConfig,
)
from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferType
from flexkv.mooncakeEngineWrapper import MoonCakeTransferEngineWrapper
from flexkv.storage.allocator import HugePageTensorHandle, materialize_worker_tensor
from flexkv.transfer.host_buffer import allocate_host_buffer
from flexkv.transfer.utils import (
    NodeMetaInfo,
    RDMATaskInfo,
    RemoteSSD2HMetaInfo,
    group_blocks_by_node,
    group_blocks_by_node_and_segment,
    split_contiguous_blocks,
)
from flexkv.transfer.worker_op import WorkerTransferOp
from flexkv.transfer.zmqHelper import (
    NotifyMsg,
    NotifyStatus,
    SSDZMQClient,
    SSDZMQServer,
)
from flexkv.transfer.workers.runtime import TransferWorkerBase


class PEER2CPUTransferWorker(TransferWorkerBase):
    def __init__(self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
        cpu_kv_layout: KVCacheLayout,
        remote_kv_layout: KVCacheLayout,
        dtype: torch.dtype,
        cache_config: CacheConfig,
        ssd_kv_layout: KVCacheLayout = None,
        ssd_files: Dict[int, List[str]] = None,  # ssd_device_id -> file_paths
        num_blocks_per_file: int = 0,
        mooncake_config_path: str = None,
    ):
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        self._pin_op_buffer()
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        # For multi-group layouts, get_chunk_size() is invalid;
        # use get_block_stride() which works for both single and multi-group BLOCKFIRST.
        if getattr(cpu_kv_layout, "layer_groups", None) is not None:
            self.block_size = cpu_kv_layout.get_block_stride()
        else:
            self.block_size = cpu_kv_layout.get_chunk_size()
        self.dtype = dtype
        self.cpu_kv_layout = cpu_kv_layout
        self.remote_kv_layout = remote_kv_layout

        self.kv_dim = cpu_kv_layout.kv_dim
        self.num_kv_heads = cpu_kv_layout.num_kv_heads
        # Bytes per KV block (all layers); used by transfer tracing for bw.
        self._bytes_per_block = self.block_size * self.dtype.itemsize * self.num_layers * self.kv_dim

        self.cpu_blocks = cpu_blocks  ## shared memory
        self.cache_config = cache_config
        self.dst_buffer_ptr = self.cpu_blocks.data_ptr()

        self.mooncake_transfer_engine = None
        # self.zmq_listen_addr = ""

        self.zmq_listen_addr = (
            f"tcp://{cache_config.local_zmq_ip}:{cache_config.local_zmq_port}"
        )

        ## initialize distributed environment
        if self.cache_config.enable_kv_sharing:
            # step1: initialize the redis meta client for node info
            self.redis_meta_client = RedisMeta(
                self.cache_config.redis_host,
                self.cache_config.redis_port,
                self.cache_config.redis_password,
                self.cache_config.local_ip,
                node_ttl_seconds=getattr(self.cache_config, 'node_ttl_seconds', 0),
            )
            self.redis_meta_client.set_node_id(self.cache_config.distributed_node_id)

            # Connect nodeinfo so the listener/heartbeat threads start and
            # current_node_id_set is populated — required for is_node_active()
            # checks during P2P transfers.
            if not self.redis_meta_client.nodeinfo.connect():
                flexkv_logger.warning(
                    "PEER2CPUTransferWorker: failed to connect RedisNodeInfo listener"
                )
            else:
                self.redis_meta_client.nodeinfo.scan_active_nodes()

            # Persistent NodeMetaInfo Pool for skip redis operation when getting
            # NodeMetaInfo according to node_id
            # assuming that every flexkv progress has unique node id
            self.node_metas: Dict[int, NodeMetaInfo] = {}
            assert self.redis_meta_client is not None


            # step2: initialize mooncake transfer engine for the whole flexkv
            # NOTE: prefer explicit parameter > cache_config > env variable
            # (spawn subprocesses may lose env vars, but cache_config is pickle-serialized)
            if mooncake_config_path is None:
                mooncake_config_path = getattr(self.cache_config, 'mooncake_config_path', None)
            if mooncake_config_path is None:
                mooncake_config_path = os.environ.get("MOONCAKE_CONFIG_PATH")
            if mooncake_config_path is None:
                raise RuntimeError(
                    "MOONCAKE_CONFIG_PATH is not set. Please either pass mooncake_config_path "
                    "parameter, set cache_config.mooncake_config_path, or set the "
                    "MOONCAKE_CONFIG_PATH environment variable."
                )
            self.mooncake_config = MooncakeTransferEngineConfig.from_file(
                mooncake_config_path
            )
            self.mooncake_transfer_engine = MoonCakeTransferEngineWrapper(
                self.mooncake_config
            )
            assert (
                self.mooncake_transfer_engine is not None
            ), "PEER2CPUTransferWorker: initilaize mooncake transfer engine failed"

            # step3: register local cpu buffer to mooncake transfer engine
            total_cpu_blocks_size = (
                self.cpu_blocks.numel() * self.cpu_blocks.element_size()
            )
            regist_buffer_status = self.mooncake_transfer_engine.regist_buffer(
                self.cpu_blocks.data_ptr(), total_cpu_blocks_size
            )
            assert (
                regist_buffer_status == 0
            ), "PEER2CPUTransferWorker: regist cpu buffer to mooncake transfer engine"

        ## when enable p2p ssd, we need start a zmq server to recive the meta info from remote node,
        # and allocate a cpu buffer for ssd to cpu copy
        if self.cache_config.enable_p2p_ssd:
            assert ssd_kv_layout is not None, "Invalid ssd kv layout!"
            ## init the cpu buffer for ssd to cpu copy
            # NOTE: now we allocate 500 blocks for test
            self.tmp_cpu_buffer_layout = KVCacheLayout(
                type=self.cpu_kv_layout.type,
                num_layer=self.cpu_kv_layout.num_layer,
                num_block=self.cache_config.num_tmp_cpu_blocks,
                tokens_per_block=self.cpu_kv_layout.tokens_per_block,
                num_head=self.cpu_kv_layout.num_head,
                head_size=self.cpu_kv_layout.head_size,
                kv_dim=self.cpu_kv_layout.kv_dim,
                num_kv_heads=self.cpu_kv_layout.num_kv_heads,
                _kv_shape=self.cpu_kv_layout.kv_shape,
            )
            # Allocate the temporary SSD->CPU staging buffer.
            #
            # Two backends are supported:
            #  (a) HugePage-backed mmap (when ``cache_config.use_hugepage_tmp_buffer``
            #      is True and the kernel has huge pages reserved). We still need
            #      to pin it for CUDA via ``cudaHostRegister`` because the region
            #      is not allocated through PyTorch's pinned-memory allocator.
            #  (b) Pinned ``torch.empty`` (the original behavior, default).
            tmp_num_elements = self.tmp_cpu_buffer_layout.get_total_elements()
            self._tmp_cpu_buffer_handle = allocate_host_buffer(
                num_elements=tmp_num_elements,
                dtype=self.dtype,
                use_hugepage=self.cache_config.use_hugepage_tmp_buffer,
                hugepage_size_bytes=self.cache_config.hugepage_size_bytes,
            )
            self.tmp_cpu_buffer = self._tmp_cpu_buffer_handle.tensor

            self.mooncake_transfer_engine.regist_buffer(
                self.tmp_cpu_buffer.data_ptr(),
                self.tmp_cpu_buffer.numel() * self.tmp_cpu_buffer.element_size(),
            )

            ## start the zmq server and client
            self.zmq_server = SSDZMQServer(cache_config.local_zmq_ip, cache_config.local_zmq_port, self.ssd_handle_loop)
            self.zmq_client = SSDZMQClient(cache_config.local_zmq_ip, cache_config.local_zmq_port+1)

            ## ssd copy to temp cpu buffer related
            self.ssd_files = ssd_files
            self.num_blocks_per_file = num_blocks_per_file
            self.num_files = sum(len(file_list) for file_list in ssd_files.values())

            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

            self.chunk_size_in_bytes = (
                self.tmp_cpu_buffer_layout.get_chunk_size() * self.dtype.itemsize
            )
            self.block_stride_in_bytes = (
                self.tmp_cpu_buffer_layout.get_block_stride() * self.dtype.itemsize
            )
            self.cpu_kv_stride_in_bytes = (
                self.tmp_cpu_buffer_layout.get_kv_stride() * self.dtype.itemsize
            )
            self.cpu_layer_stride_in_bytes = (
                self.tmp_cpu_buffer_layout.get_layer_stride() * self.dtype.itemsize
            )
            self.ssd_kv_stride_in_bytes = (
                ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            )
            self.ssd_layer_stride_in_bytes = (
                ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            )


            self.round_robin = 1
            # initialize ssd ioctx
            try:
                self.ioctx = c_ext.SSDIOCTX(
                    ssd_files,
                    len(ssd_files),
                    GLOBAL_CONFIG_FROM_ENV.iouring_entries,
                    GLOBAL_CONFIG_FROM_ENV.iouring_flags,
                )
            except Exception as e:
                flexkv_logger.error(f"Error setting ssd ioctx: {e}\n")
                raise RuntimeError("SSD Worker init failed") from e

        ## step4: regist node info into redis server
        ## Must be done after P2P SSD init so we can register the correct
        ## ssd_buffer_base_ptr (tmp_cpu_buffer) when P2P SSD is enabled.
        if self.cache_config.enable_kv_sharing:
            ssd_buffer_ptr = (
                self.tmp_cpu_buffer.data_ptr()
                if self.cache_config.enable_p2p_ssd
                else 0
            )
            self.regist_node_meta(
                self.cpu_blocks.data_ptr(),
                ssd_buffer_ptr,
                self.zmq_listen_addr,
            )

        ## unique task id counter for remote ssd to cpu transfer task
        self.remote_ssd_task_id_counter = 0
        self.task_id_lock = threading.Lock()

    #============================ common part ========================
    def gen_task_id(self) -> int:
        """
        generate a unique task id for remote ssd to cpu transfer task
        Returns:
            int: task id
        """
        with self.task_id_lock:
            old_value = self.remote_ssd_task_id_counter
            self.remote_ssd_task_id_counter += 1
            return old_value

    def shutdown(self):
        """Best-effort cleanup; tolerant of partially-failed ``__init__``."""
        try:
            zmq_server = getattr(self, "zmq_server", None)
            if zmq_server is not None:
                zmq_server.shutdown()
            zmq_client = getattr(self, "zmq_client", None)
            if zmq_client is not None:
                zmq_client.shutdown()
            engine = getattr(self, "mooncake_transfer_engine", None)
            cpu_blocks = getattr(self, "cpu_blocks", None)
            if engine is not None and cpu_blocks is not None:
                try:
                    engine.unregist_buffer(cpu_blocks.data_ptr())
                except Exception as e:
                    flexkv_logger.warning(
                        f"PEER2CPUTransferWorker unregist cpu buffer failed: {e}"
                    )
                cache_config = getattr(self, "cache_config", None)
                if cache_config is not None and getattr(cache_config, "enable_p2p_ssd", False):
                    tmp_buf = getattr(self, "tmp_cpu_buffer", None)
                    if tmp_buf is not None:
                        try:
                            engine.unregist_buffer(tmp_buf.data_ptr())
                        except Exception as e:
                            flexkv_logger.warning(
                                f"PEER2CPUTransferWorker unregist tmp buffer failed: {e}"
                            )
                    tmp_handle = getattr(self, "_tmp_cpu_buffer_handle", None)
                    if tmp_handle is not None:
                        try:
                            tmp_handle.release()
                        except Exception as e:
                            flexkv_logger.warning(
                                f"PEER2CPUTransferWorker release tmp handle failed: {e}"
                            )
            if getattr(self, "redis_meta_client", None) is not None:
                try:
                    self.unregist_node_meta()
                except Exception as e:
                    flexkv_logger.warning(
                        f"PEER2CPUTransferWorker unregist_node_meta failed: {e}"
                    )
        except Exception as e:
            flexkv_logger.error(f"PEER2CPUTransferWorker shutdown error: {e}")
        finally:
            super().shutdown()

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        task_info_list = self.op_parser(transfer_op)

        start_time = time.time()
        transfered_size = 0
        transfer_finished = True

        for task_info in task_info_list:
            # NOTE: here one task_info represent data transfer from one node
            ret = self._batch_transfer_impl(
                task_info,
                transfer_op.transfer_type,
            )
            if not ret:
                transfer_finished = False
                break
            transfered_size += task_info.data_size

        end_time = time.time()

        self._log_transfer_performance(
            transfer_op,
            transfered_size,
            start_time,
            end_time,
        )
        return transfer_finished

    # Timeout for a single RDMA batch transfer (seconds).
    # Prevents indefinite blocking when a remote node becomes unreachable
    # but its node:<id> TTL hasn't expired yet.
    RDMA_TRANSFER_TIMEOUT_SECONDS = 30

    def _batch_transfer_impl(self,
        task_info: RDMATaskInfo,
        transfer_type: TransferType,
        **kwargs,):
        if transfer_type == TransferType.PEERH2H:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.mooncake_transfer_engine.batch_transfer_sync_read,
                    task_info.peer_engine_addr, task_info.src_ptrs, task_info.dst_ptrs, task_info.data_lens
                )
                try:
                    ret = future.result(timeout=self.RDMA_TRANSFER_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    flexkv_logger.error(
                        f"RDMA batch transfer to {task_info.peer_engine_addr} timed out "
                        f"after {self.RDMA_TRANSFER_TIMEOUT_SECONDS}s"
                    )
                    return False
            if ret != 0:
                flexkv_logger.error(f"RDMA transfer failed with error code: {ret}")
                return False
        elif transfer_type == TransferType.PEERSSD2H:
          # remote ssd to local cpu transfer by two side zmq and one side rdma write
            # step1: construct the meta info
            remote_ssd_to_cpu_meta = RemoteSSD2HMetaInfo(
                task_id=task_info.task_id,
                cpu_block_ids=task_info.dst_block_ids,
                ssd_block_ids=task_info.src_block_ids,
                peer_engine_addr=task_info.local_engine_addr,
                peer_cpu_base_ptr=self.dst_buffer_ptr,
                peer_zmq_status_addr=self.zmq_client.get_addr(),
                data_size=task_info.data_size,
            )
            #flexkv_logger.info(
            #    f"[PEERSSD2H] Sending meta: task_id={task_info.task_id}, "
            #    f"ssd_block_ids={task_info.src_block_ids}, cpu_block_ids={task_info.dst_block_ids}, "
            #    f"peer_engine_addr={task_info.local_engine_addr}, peer_zmq_addr={task_info.peer_zmq_addr}"
            #)
            ## step2: send the meta info to remote node
            if not self.zmq_client.send_meta_info(remote_ssd_to_cpu_meta, task_info.peer_zmq_addr):
                flexkv_logger.error(
                    f"Send remote ssd to cpu meta info to {task_info.peer_zmq_addr} failed"
                )
                return False

            ## step3: wait for remote node to send data transfer complete notify
            ret = self.zmq_client.wait_transfer_notify(
                task_info.peer_engine_addr, task_info.task_id
            )
            if not ret:
                flexkv_logger.error(
                    f"Wait remote ssd to cpu transfer task {task_info.task_id} "
                    f"notify from {task_info.peer_engine_addr} failed with error code: {ret}"
                )
                return False
        else:
            raise ValueError(
                f"Invalid transfer type: {transfer_type} for PEER2CPUTransferWorker"
            )
        return True

    def _transfer_impl(
        self,
        task_info: RDMATaskInfo,
        transfer_type: TransferType,
        **kwargs,
    ):
        if transfer_type == TransferType.PEERH2H:
            # remote cpu to local cpu transfer by one-side rdma read
            for i in range(len(task_info.src_ptrs)):
                ret = self.mooncake_transfer_engine.transfer_sync_read(
                    task_info.peer_engine_addr,
                    task_info.src_ptrs[i],
                    task_info.dst_ptrs[i],
                    task_info.data_lens[i],
                )
                if ret != 0:
                    flexkv_logger.error(f"transfer_sync_write failed with error code: {ret}")
                    return False
        elif transfer_type == TransferType.PEERSSD2H:
            # remote ssd to local cpu transfer by two side zmq and one side rdma write
            # step1: construct the meta info
            remote_ssd_to_cpu_meta = RemoteSSD2HMetaInfo(
                task_id=task_info.task_id,
                cpu_block_ids=task_info.dst_block_ids,
                ssd_block_ids=task_info.src_block_ids,
                peer_engine_addr=task_info.local_engine_addr,
                peer_cpu_base_ptr=self.dst_buffer_ptr,
                peer_zmq_status_addr=self.zmq_client.get_addr(),
                data_size=task_info.data_size,
            )
            flexkv_logger.info(
                f"[_transfer_impl] Sending task_id={task_info.task_id}, "
                f"ssd_block_ids={task_info.src_block_ids}, "
                f"cpu_block_ids={task_info.dst_block_ids} to {task_info.peer_zmq_addr}"
            )
            ## step2: send the meta info to remote node
            if not self.zmq_client.send_meta_info(remote_ssd_to_cpu_meta, task_info.peer_zmq_addr):
                flexkv_logger.error(
                    f"Send remote ssd to cpu meta info to {task_info.peer_zmq_addr} failed"
                )
                return False

            ## step3: wait for remote node to send data transfer complete notify
            ret = self.zmq_client.wait_transfer_notify(
                task_info.peer_engine_addr, task_info.task_id
            )
            if not ret:
                flexkv_logger.error(
                    f"Wait remote ssd to cpu transfer task {task_info.task_id} "
                    f"notify from {task_info.peer_engine_addr} failed with error code: {ret}"
                )
                return False

        else:
            raise ValueError(
                f"Invalid transfer type: {transfer_type} for PEER2CPUTransferWorker"
            )

        return True

    def op_parser(
        self, transfer_op: WorkerTransferOp
    ) -> List[RDMATaskInfo]:
        """
        parse the transfer op to a list of RDMATaskInfo
        1. group the blocks by remote node id, each segment is a list of
           continuous blocks (segment is the smallest transmission unit)
        2. using corresponding distributed op parser to parse the op and create RDMATaskInfo for each segment
        5. return the list of RDMATaskInfo
        Parameters:
            transfer_op (WorkerTransferOp): the transfer op to be parsed
        Returns:
            List[RDMATaskInfo]: the list of RDMATaskInfo
        """
        assert (
            transfer_op.transfer_type == TransferType.PEERH2H
            or transfer_op.transfer_type == TransferType.PEERSSD2H
        ), f"PEER2CPUTransferWorker only support PEERH2H or PEERSSD2H, but get {transfer_op.transfer_type}"

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op, False)

        assert len(src_block_ids) == len(dst_block_ids)

        src_block_node_ids = transfer_op.src_block_node_ids
        # Convert to plain list — the compiled utils.so (pybind11) requires list,
        # not numpy.ndarray.
        if hasattr(src_block_node_ids, 'tolist'):
            src_block_node_ids = src_block_node_ids.tolist()

        # step1: group the blocks by remote node id and remote block source type,
        # each segment is a list of continuous blocks
        #flexkv_logger.info(
        #    f"[PEER2CPUTransferWorker] src_block_ids: {src_block_ids} \n \
        #                        dst_block_ids: {dst_block_ids} \n \
        #                        src_block_node_ids: {src_block_node_ids} \n"
        #)
        task_info_list = []

        if transfer_op.transfer_type == TransferType.PEERH2H:
            groups = group_blocks_by_node_and_segment(
                src_block_ids, dst_block_ids, src_block_node_ids
            )
            task_info_list = self._dist_cpu_op_parser(groups)
        elif transfer_op.transfer_type == TransferType.PEERSSD2H:
            groups = group_blocks_by_node(
                src_block_ids, dst_block_ids, src_block_node_ids
            )
            task_info_list = self._dist_ssd_op_parser(groups)
        else:
            raise RuntimeError(
                f"Unsurpported transfer_type {transfer_op.transfer_type} in PEER2CPUTransferWorker"
            )

        return task_info_list

    #========================== distrbuted ssd related ==========================
    #========================== local behaviors
    def _dist_ssd_op_parser(self, groups: Dict[int, Dict[str, List[int]]]):
        """
        Distributed ssd op parser
        1. for each segment, get the remote ssd blocks and local cpu blocks
        2. create RDMATaskInfo for each segment
        Args:
            groups (Dict[int, Dict[str, List[int]]]): the grouped blocks

        Returns:
            task_info_list: the list of RDMATaskInfo, each task refers to one data transfer operation
        """
        ## parse ssd
        # TODO: now we only support blockwise layout, need support layerwise layout

        task_info_list = []

        for node_id, segment in groups.items():
            ##NOTE: for ssd scenario, each node will only have one set of src and dst block ids
            peer_node_info = self.get_node_meta(node_id)
            if peer_node_info is None:
                return []
            peer_zmq_addr = peer_node_info.zmq_addr
            peer_engine_addr = peer_node_info.engine_addr
            assert (
                peer_zmq_addr != ""
            ), f"Node {node_id} zmq addr not found in redis server"

            src_blocks = segment["src"]
            dst_blocks = segment["dst"]
            assert len(src_blocks) == len(dst_blocks)

            data_size = self.cpu_kv_layout.get_block_stride() * self.dtype.itemsize * len(src_blocks)
            ssd_task_id = self.gen_task_id()
            task_info_list.append(
                RDMATaskInfo(
                    ssd_task_id,
                    self.mooncake_transfer_engine.get_engine_addr(),
                    # for ssd transfer, peer engine addr refers to local mooncake engine
                    peer_engine_addr,
                    peer_zmq_addr,
                    None,
                    None,
                    src_blocks,
                    dst_blocks,
                    [], # not used in ssd transfer
                    data_size=data_size
                )
            )
        return task_info_list


    #=============================remote behaviors

    def meta_info_parser(self, recv_msg: str):
        recv_dict = json.loads(recv_msg)
        return RemoteSSD2HMetaInfo.from_dict(recv_dict)

    def ssd_handle_loop(self):
        flexkv_logger.info(
            f"Node {self.cache_config.distributed_node_id} Listening on {self.zmq_listen_addr}"
        )
        while not self.zmq_server.shutdown_event.is_set():
            recv_meta = None
            failure_msg = None
            try:
                ## step1: recv and parse the message into meta info
                try:
                    message = self.zmq_server.listen_socket.recv().decode("utf-8")
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                if not message:
                    self.zmq_server.listen_socket.send(b"ERROR")
                    continue

                recv_meta = self.meta_info_parser(message)
                if not recv_meta:
                    self.zmq_server.listen_socket.send(b"ERROR")
                    flexkv_logger.warning("Can not parse RemoteSSD2HMetaInfo using recieved message")
                    continue

                flexkv_logger.info(
                    f"[ssd_handle_loop] Received task_id={recv_meta.task_id}, "
                    f"ssd_block_ids={recv_meta.ssd_block_ids}, "
                    f"cpu_block_ids={recv_meta.cpu_block_ids}"
                )

                self.zmq_server.listen_socket.send(b"OK")

                failure_msg = NotifyMsg(
                    mooncake_engine_addr=self.mooncake_transfer_engine.get_engine_addr(),
                    task_id=recv_meta.task_id,
                    status=NotifyStatus.FAIL,
                )
                success_msg = NotifyMsg(
                    mooncake_engine_addr=self.mooncake_transfer_engine.get_engine_addr(),
                    task_id=recv_meta.task_id,
                    status=NotifyStatus.SUCCESS,
                )

                # step2: ckeck the recieved info, early return if check error
                nvtx_range = nvtx.start_range(message="ssd_handle_loop. check and load_data", color="orange")
                if len(recv_meta.ssd_block_ids) == 0 or len(recv_meta.cpu_block_ids) == 0 \
                    or len(recv_meta.cpu_block_ids)!=len(recv_meta.ssd_block_ids):
                        flexkv_logger.warning(
                            "Invalid cpu_block_ids or ssd_block_ids, skipping this transfer..."
                        )
                        self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                        continue

                # TODO: we need to support dynamic temp buffer or split the ssd
                # transfer request if number of ssd blocks is larger than
                # num_tmp_cpu_blocks. Now we just refuse this transfer by
                # returning a failure status.
                if len(recv_meta.ssd_block_ids)>self.cache_config.num_tmp_cpu_blocks:
                    flexkv_logger.warning(
                            f"The number of ssd_block_ids is larger than "
                            f"{self.cache_config.num_tmp_cpu_blocks}, can not do transfer now"
                        )
                    self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                    continue

                ## step3: do copy data from ssd to cpu
                # NOTE: this block ids is a corresponding relationship with
                # self.tmp_cpu_buffer, for every transfer req we reuse the local cpu buffer
                local_cpu_buffer_block_ids = torch.arange(0, len(recv_meta.ssd_block_ids), dtype = torch.int64)
                local_cpu_start_idx = 0

                # seperate the blocks to get the longest continuous blocks
                groups = split_contiguous_blocks(recv_meta.ssd_block_ids, recv_meta.cpu_block_ids)

                all_copy_complete = True
                src_ptr_list = []
                dst_ptr_list = []
                data_size_list = []

                for item in groups:
                    # in this loop we do two things:
                    # 1. copy ssd data to cpu for each segment
                    # 2. calculate the start ptr of local cpu blocks and dst cpu blocks for each segment and record them
                    ssd_block_ids_per_seg = torch.tensor(item["src"], dtype=torch.int64)
                    dst_cpu_block_ids_per_seg = torch.tensor(item["dst"], dtype=torch.int64)

                    if len(ssd_block_ids_per_seg) == 0:
                        all_copy_complete = False
                        break
                    # get corresponding temp cpu block ids
                    local_cpu_buffer_block_ids_per_seg = local_cpu_buffer_block_ids[
                        local_cpu_start_idx: local_cpu_start_idx + len(ssd_block_ids_per_seg)
                    ]
                    local_cpu_start_idx += len(ssd_block_ids_per_seg)

                    layer_id_list = torch.arange(
                        0, self.num_layers, dtype=torch.int32
                    )
                    if not self.copy_ssd_data_to_dram(
                        layer_id_list, ssd_block_ids_per_seg, local_cpu_buffer_block_ids_per_seg
                    ):
                        flexkv_logger.error("Copy ssd data to dram failed!")
                        all_copy_complete = False
                        break

                    src_ptrs, src_block_size = self.get_cpu_buffer_block_start_ptr(
                        local_cpu_buffer_block_ids_per_seg,
                        self.tmp_cpu_buffer.data_ptr(),
                    )

                    dst_ptrs, dst_block_size = self.get_cpu_buffer_block_start_ptr(
                        dst_cpu_block_ids_per_seg,
                        recv_meta.peer_cpu_base_ptr,
                    )
                    assert src_block_size == dst_block_size, "Block size mismatch between src and dst"

                    for _ in range(len(src_ptrs)):
                        data_size_list.append(src_block_size * len(local_cpu_buffer_block_ids_per_seg))
                    src_ptr_list.extend(src_ptrs)
                    dst_ptr_list.extend(dst_ptrs)
                    assert len(src_ptr_list) == len(data_size_list) and len(dst_ptr_list) == len(data_size_list)

                nvtx.end_range(nvtx_range)
                nvtx_range = nvtx.start_range(message="ssd_handle_loop. write_data_back_to_peer", color="orange")
                ## step4: do rdma transfer and send notify
                if not all_copy_complete:
                    self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                    continue

                if not self.write_data_back_to_peer(
                    recv_meta.peer_engine_addr, src_ptr_list, dst_ptr_list, data_size_list
                ):
                    self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                    flexkv_logger.error("Failed to write data back to peer")
                    continue

                self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, success_msg)
                nvtx.end_range(nvtx_range)
            except Exception as e:
                flexkv_logger.error(f"Unexpected error in ssd_handle_loop: {e}")
                # Send failure notify so the peer doesn't block waiting forever
                try:
                    if recv_meta is not None:
                        self.zmq_server.send_transfer_status(
                            recv_meta.peer_zmq_status_addr, failure_msg
                        )
                except Exception:
                    pass
                time.sleep(0.001)

    def copy_ssd_data_to_dram(
        self, layer_id_list: torch.Tensor, ssd_block_id_list: torch.Tensor, cpu_block_id_list: torch.Tensor
    ):
        assert len(ssd_block_id_list) == len(cpu_block_id_list)
        flexkv_logger.info(f"copy ssd blocks:{ssd_block_id_list} to cpu blocks: {cpu_block_id_list}" )
        try:
            transfer_kv_blocks_ssd(
                self.ioctx,
                layer_id_list,
                self.tmp_cpu_buffer.data_ptr(),  ## copy ssd data to tmp cpu buffer
                ssd_block_id_list,
                cpu_block_id_list,
                self.cpu_layer_stride_in_bytes,
                self.cpu_kv_stride_in_bytes,
                self.ssd_layer_stride_in_bytes,
                self.ssd_kv_stride_in_bytes,
                self.chunk_size_in_bytes,
                self.block_stride_in_bytes,
                True,
                self.num_blocks_per_file,
                self.round_robin,
                32,
                self.kv_dim,
                ssd_io_opt=GLOBAL_CONFIG_FROM_ENV.ssd_io_opt,
            )
        except Exception as e:
            flexkv_logger.error(f"Copy data from ssd to cpu failed: {e}")
            return False
        return True

    def write_data_back_to_peer(
        self,
        peer_address: str,
        src_ptr_list: List[int],
        dst_ptr_list: List[int],
        data_size_list: List[int]
    ):
        flexkv_logger.info(
            f"Write data back to peer from src: {src_ptr_list} to {dst_ptr_list}"
        )
        ret = self.mooncake_transfer_engine.batch_transfer_sync_write(
            peer_address, src_ptr_list, dst_ptr_list, data_size_list
        )
        return ret == 0


    #============================== distrbuted cpu related ==========================

    def _dist_cpu_op_parser(
        self,
        groups: Dict[int, List[Dict[str, List[int]]]],
    ):
        """
        Distributed cpu op parser
        1. for each segment, get the remote cpu ptrs and local cpu ptrs
        2. create RDMATaskInfo for each segment

        Inputs:
            groups (Dict[int, List[Dict[str, List[int]]]]): the grouped blocks

        Returns:
            task_info_list: the list of RDMATaskInfo, each task refers to the data transfer of one node
        """

        task_info_list = []

        for node_id, segments in groups.items():
            # step1: get the remote meta info
            src_meta = self.get_node_meta(node_id)
            if src_meta is None:
                # Skip this node's blocks instead of aborting all nodes.
                # In multi-node P2P, one dead node should not prevent fetching
                # blocks from other healthy nodes.
                flexkv_logger.warning(
                    f"[PEER2CPUTransferWorker] Skipping node {node_id}: "
                    f"meta unavailable, will skip {len(segments)} segment(s)"
                )
                continue
            peer_engine_addr = src_meta.engine_addr
            src_ptr_list = []
            dst_ptr_list = []
            data_size_list = []
            for seg in segments:
                src_blocks = seg["src"]
                dst_blocks = seg["dst"]

                # step2: calculate the src and dst block start ptrs
                src_block_start_ptrs, src_data_size_per_block = (
                    self.get_cpu_buffer_block_start_ptr(
                        src_blocks,
                        src_meta.cpu_bufer_base_ptr,  # the cpu buffer ptr on remote machine
                    )
                )


                dst_block_start_ptrs, dst_data_size_per_block = (
                    self.get_cpu_buffer_block_start_ptr(
                        dst_blocks,
                        self.dst_buffer_ptr,  # the cpu buffer ptr on local machine
                    )
                )

                assert (
                    src_data_size_per_block == dst_data_size_per_block
                ), "src and dst blocks have different layout"


                for _ in range(len(src_block_start_ptrs)):
                    data_size = src_data_size_per_block * len(src_blocks)
                    data_size_list.append(data_size)
                src_ptr_list.extend(src_block_start_ptrs)
                dst_ptr_list.extend(dst_block_start_ptrs)
                assert len(data_size_list) == len(src_ptr_list) and len(data_size_list) == len(dst_ptr_list)

            flexkv_logger.info(
                f"[PEER2CPUTransferWorker]: remote cpu op parser "
                f"src_ptr_list: {src_ptr_list}, dst_ptr_list: {dst_ptr_list} "
            )
            # step3: create RDMATaskInfo for each segment
            # NOTE: block wise layout: only one start ptr for each segment
            #       layer wise layout: multiple start ptrs for each segment,
            #       the number of start ptrs equals num_layers * kv_dim
            task_info_list.append(
                  RDMATaskInfo(
                    0,
                    "",
                    peer_engine_addr,
                    "",
                    src_ptr_list,
                    dst_ptr_list,
                    [],    # src_block_ids unused for PEERH2H (uses ptrs)
                    [],    # dst_block_ids unused for PEERH2H (uses ptrs)
                    data_size_list,
                    data_size = sum(data_size_list)
                )
            )

        return task_info_list

    #================================== utils =================================
    def get_cpu_buffer_block_start_ptr(
        self,
        cpu_blocks: List[int],
        cpu_base_ptr: int,
    ) -> Tuple[List[int], int]:
        """
        Get the cpu buffer block start ptrs for the given cpu blocks.
        We have two layout types in flexkv, layerwise and blockwise.
        1) For layerwise layout, although the cpu blocks are continuous, we need to
        calculate the start ptrs for each layer and each kv dim. So
        the number of start ptrs equals self.num_layers * kv_dim.
        2) For blockwise layout, the cpu blocks are continuous, so we only need to
        calculate the start ptr for the first block. So
        the number of start ptrs is 1.
        3) For other layout types, raise error.

        Parameters:
            cpu_blocks (List[int]): the list of cpu block ids, continuous
            cpu_base_ptr (int): the base ptr of the cpu buffer
        Returns:
            Tuple(List[int], int): the list of cpu buffer block start ptrs and
                data size per block (used for calculate total data size)
        """

        # assuming that remote cpu buffer layout is the same as local cpu buffer layout
        assert self.cpu_kv_layout.type == self.remote_kv_layout.type
        src_block_ptrs = []

        # Get the first block ID and handle different input types
        if isinstance(cpu_blocks, torch.Tensor):
            block_id_int = int(cpu_blocks[0].item())
        elif isinstance(cpu_blocks, list):
            first_elem = cpu_blocks[0]
            if isinstance(first_elem, torch.Tensor):
                block_id_int = int(first_elem.item())
            else:
                block_id_int = int(first_elem)
        else:
            raise ValueError(f"Invalid cpu_blocks type: {type(cpu_blocks)}")

        if self.cpu_kv_layout.type == KVCacheLayoutType.LAYERFIRST:
            for layer_id in range(0, self.num_layers):
                for kv_id in range(self.kv_dim):
                    element_offset = (
                        (
                            ((layer_id * self.kv_dim) + kv_id)
                            * self.cpu_kv_layout.num_block
                            + block_id_int
                        )
                        * self.cpu_kv_layout.get_block_stride()
                        * self.dtype.itemsize
                    )
                    src_block_ptrs.append(cpu_base_ptr + element_offset)

        elif self.cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST:
            block_volume = self.cpu_kv_layout.get_block_stride()
            element_offset = block_id_int * block_volume * self.dtype.itemsize
            src_block_ptrs.append(cpu_base_ptr + element_offset)
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.cpu_kv_layout.type}")
        data_size_per_block = self.cpu_kv_layout.get_block_stride() * self.dtype.itemsize

        return  src_block_ptrs, data_size_per_block

    ### redis client helper functions
    def regist_node_meta(
        self, cpu_buffer_base_ptr: int, ssd_buffer_base_ptr: int, zmq_addr: str
    ):
        self.redis_meta_client.regist_node_meta(
            self.redis_meta_client.get_node_id(),
            self.mooncake_transfer_engine.get_engine_addr(),
                                                zmq_addr, cpu_buffer_base_ptr, ssd_buffer_base_ptr)
        #NOTE: maybe useless
        node_meta_info = NodeMetaInfo(
            self.redis_meta_client.get_node_id(),
            self.mooncake_transfer_engine.get_engine_addr(),
            zmq_addr,
            cpu_buffer_base_ptr,
            ssd_buffer_base_ptr
        )
        self.node_metas[self.redis_meta_client.get_node_id()] = node_meta_info
        flexkv_logger.info(f"Registered node {self.redis_meta_client.get_node_id()} to Redis.")

    def unregist_node_meta(self, node_id: int = None) -> None:
        self.redis_meta_client.unregist_node_meta(self.redis_meta_client.get_node_id())
        flexkv_logger.info(f"Unregistered node {self.redis_meta_client.get_node_id()} from Redis.")

    def get_node_meta(self, node_id: int) -> Optional[NodeMetaInfo]:
        """Get the node meta info by node id.

        Before returning cached or freshly-fetched meta, we verify that the
        node is still active (its node:<id> key exists in Redis and has not
        expired).  This prevents RDMA transfers to stale addresses after a
        remote node has crashed.
        """
        # ===== Active-node validation (Scheme 4) =====
        if not self.redis_meta_client.is_node_active(node_id):
            # Node is no longer active – purge cached meta if any
            if node_id in self.node_metas:
                del self.node_metas[node_id]
                flexkv_logger.warning(
                    f"Node {node_id} is no longer active, removed cached meta."
                )
            else:
                flexkv_logger.warning(
                    f"Node {node_id} is not active, skipping meta fetch."
                )
            return None

        if node_id not in self.node_metas:
            ## fetch from redis
            node_redis_data = self.redis_meta_client.get_node_meta(node_id)
            if not node_redis_data:
                flexkv_logger.error(f"Node {node_id} meta not found in Redis.")
                return None

            node_meta = NodeMetaInfo.from_dict(node_redis_data)

            self.node_metas[node_id] = node_meta
            flexkv_logger.info(f"Fetched node {node_id} meta from Redis.")

        return self.node_metas[node_id]
