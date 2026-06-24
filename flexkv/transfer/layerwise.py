import time
import os
import socket
import struct
from torch.multiprocessing import Queue as MPQueue
from multiprocessing.connection import Connection
from typing import List, Any, Dict, Union, Optional, Tuple

import torch

from flexkv.c_ext import LayerwiseTransferGroup
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.config import ModelConfig, GLOBAL_CONFIG_FROM_ENV

from flexkv.transfer.worker_op import WorkerLayerwiseTransferOp
from flexkv.transfer.worker import TransferWorkerBase, cudaHostRegister


def build_layerwise_eventfd_socket_path(model_config: ModelConfig) -> str:
    """Construct the LayerwiseWorker's UDS socket path.

    Disambiguated by ``(pp_rank, dp_rank)`` so multiple PP stages and DP
    replicas on the same host each get their own endpoint.

    We deliberately do NOT embed ``node_rank`` in the path: Unix domain
    sockets are kernel-local, so two FlexKV instances on different
    physical hosts cannot collide even when ``/tmp`` happens to be on a
    shared filesystem (NFS and friends propagate the inode, not the
    socket endpoint).  Deployments that stack multiple containers on one
    host with a shared ``/tmp`` should disambiguate via the
    ``FLEXKV_LAYERWISE_EVENTFD_SOCKET`` env var (e.g. embed ``$HOSTNAME``
    or the container id in the base path).

    Must stay in sync with the sglang-side consumer at
    ``sglang.srt.mem_cache.storage.flexkv.flexkv_connector``, which
    imports this helper directly so the two ends cannot drift.  Both
    sides derive the path from the same ``ModelConfig`` fields, so no
    env-var plumbing between processes is required.
    """
    base = os.environ.get(
        'FLEXKV_LAYERWISE_EVENTFD_SOCKET',
        '/tmp/flexkv_layerwise_eventfd.sock',
    )
    suffix = ""
    if model_config.pp_size > 1:
        suffix += f"_pp{model_config.pp_rank}"
    if model_config.dp_size > 1:
        suffix += f"_dp{model_config.dp_rank}"
    if not suffix:
        return base
    root, ext = os.path.splitext(base)
    return f"{root}{suffix}{ext}"


def _recv_fds(sock: socket.socket, num_fds: int) -> Tuple[List[int], bytes]:
    """Receive multiple fds + extra_data via Unix domain socket (SCM_RIGHTS)."""
    data_buf = bytearray(256)
    anc_buf_size = socket.CMSG_SPACE(num_fds * struct.calcsize("i"))

    nbytes, ancdata, flags, addr = sock.recvmsg_into([data_buf], anc_buf_size, 0)
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
                 pp_rank: int,
                 pp_size: int,
                 dp_size: int,
                 dp_rank: int,
                 layerwise_eventfd_socket: str,
                 num_blocks_per_file: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 h2d_cta_num: int = 4,
                 d2h_cta_num: int = 4,
                 enable_eventfd: bool = True,
                 is_nsa_cp: bool = False,
                 indexer_gpu_blocks: Optional[List[List[TensorSharedHandle]]] = None,
                 indexer_cpu_blocks: Optional[torch.Tensor] = None,
                 indexer_gpu_kv_layouts: Optional[List[KVCacheLayout]] = None,
                 indexer_cpu_kv_layout: Optional[KVCacheLayout] = None,
                 indexer_dtype: Optional[torch.dtype] = None,
                 indexer_ssd_files: Optional[Dict[int, List[str]]] = None,
                 indexer_ssd_kv_layout: Optional[KVCacheLayout] = None,
                 indexer_num_blocks_per_file: int = 0) -> None:
        flexkv_logger.debug(
            f"[LayerwiseWorker] __init__ started: worker_id={worker_id}, "
            f"tp_group_size={tp_group_size}, dp_group_id={dp_group_id}, "
            f"pp_rank={pp_rank}, pp_size={pp_size}, "
            f"enable_eventfd={enable_eventfd}, "
            f"num_gpu_blocks={[len(b) for b in gpu_blocks]}")
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
        self.pp_rank = pp_rank
        self.pp_size = pp_size if pp_size > 0 else 1
        self.dp_group_id = dp_group_id
        self.dp_size = dp_size if dp_size > 0 else 1
        self.dp_rank = dp_rank
        # Pre-computed UDS socket path.  Both ends (this worker and the
        # sglang connector) derive the path from the same ModelConfig
        # fields (pp_rank / dp_rank / node_rank / is_multinode_tp), so no
        # env-var plumbing between processes is required.
        self.layerwise_eventfd_socket = layerwise_eventfd_socket
        self.is_nsa_cp = is_nsa_cp

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

        flexkv_logger.debug(f"[LayerwiseWorker] About to receive eventfds, enable_eventfd={enable_eventfd}")
        if enable_eventfd:
            layer_eventfds_tensor = self._receive_eventfds_from_sglang(tp_group_size)
        else:
            layer_eventfds_tensor = torch.empty(0, dtype=torch.int32)
        flexkv_logger.debug(f"[LayerwiseWorker] Eventfds received, tensor shape={layer_eventfds_tensor.shape}")

        # initialize CPU storage
        flexkv_logger.info(f"[LayerwiseWorker] Pinning CPU Memory: "
                           f"{cpu_blocks.numel() * cpu_blocks.element_size() / (1024 ** 3):.2f} GB")
        cudaHostRegister(cpu_blocks)
        flexkv_logger.debug("[LayerwiseWorker] CPU memory pinned successfully")
        self.cpu_blocks = cpu_blocks

        self.cpu_chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize
        # Full CPU strides (for SSD->CPU, which transfers all TP ranks' data)
        self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        # TP-divided CPU strides (for CPU->GPU, each rank reads its own portion)
        if self.is_nsa_cp:
            # CP: no head partitioning, every rank gets the full KV cache
            cpu_kv_layout_tp = cpu_kv_layout
            self.cpu_tp_stride_in_bytes = 0
        else:
            # TP: partition by heads, each rank reads a different head slice
            if cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST and not self.is_mla:
                cpu_kv_layout_tp = cpu_kv_layout.div_head(self.tp_group_size)
            else:
                cpu_kv_layout_tp = cpu_kv_layout
            self.cpu_tp_stride_in_bytes = self.cpu_block_stride_in_bytes // self.tp_group_size
        self.h2d_cpu_kv_stride_in_bytes = cpu_kv_layout_tp.get_kv_stride() * self.dtype.itemsize
        self.h2d_cpu_layer_stride_in_bytes = cpu_kv_layout_tp.get_layer_stride() * self.dtype.itemsize

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

        # Create LayerwiseTransferGroup which handles both SSD->CPU and CPU->GPU transfers
        flexkv_logger.debug("[LayerwiseWorker] Creating LayerwiseTransferGroup...")

        # Initialize indexer fuse support
        self.enable_indexer = (indexer_gpu_blocks is not None and indexer_cpu_blocks is not None)
        indexer_constructor_kwargs = {}
        if self.enable_indexer:
            assert indexer_gpu_kv_layouts is not None
            assert indexer_cpu_kv_layout is not None
            assert indexer_dtype is not None

            # Import indexer GPU tensor handles
            imported_indexer_gpu_blocks = []
            for handles_in_one_gpu in indexer_gpu_blocks:
                blocks_in_one_gpu = []
                for handle in handles_in_one_gpu:
                    blocks_in_one_gpu.append(handle.get_tensor())
                imported_indexer_gpu_blocks.append(blocks_in_one_gpu)

            # Pin indexer CPU memory
            flexkv_logger.info(
                f"[LayerwiseWorker] Pinning indexer CPU Memory: "
                f"{indexer_cpu_blocks.numel() * indexer_cpu_blocks.element_size() / (1024 ** 3):.4f} GB")
            cudaHostRegister(indexer_cpu_blocks)

            # Compute indexer GPU stride tensors
            indexer_gpu_kv_strides = [layout.get_kv_stride() * indexer_dtype.itemsize
                                     for layout in indexer_gpu_kv_layouts]
            indexer_gpu_block_strides = [layout.get_block_stride() * indexer_dtype.itemsize
                                        for layout in indexer_gpu_kv_layouts]
            indexer_gpu_layer_strides = [layout.get_layer_stride() * indexer_dtype.itemsize
                                        for layout in indexer_gpu_kv_layouts]
            indexer_gpu_chunk_sizes = [layout.get_chunk_size() * indexer_dtype.itemsize
                                      for layout in indexer_gpu_kv_layouts]

            # Compute indexer CPU strides.
            # Indexer is always is_mla=True (1 head, ReplicatedLinear weights),
            # so all TP ranks hold identical data and no head-partitioning is needed.
            # Therefore indexer has no tp_stride — cpu_startoff is always 0.
            self.indexer_cpu_block_stride_in_bytes = indexer_cpu_kv_layout.get_block_stride() * indexer_dtype.itemsize
            self.indexer_cpu_layer_stride_in_bytes = indexer_cpu_kv_layout.get_layer_stride() * indexer_dtype.itemsize
            self.indexer_h2d_cpu_kv_stride_in_bytes = indexer_cpu_kv_layout.get_kv_stride() * indexer_dtype.itemsize
            self.indexer_h2d_cpu_layer_stride_in_bytes = indexer_cpu_kv_layout.get_layer_stride() * indexer_dtype.itemsize

            self.indexer_gpu_blocks = imported_indexer_gpu_blocks
            self.indexer_cpu_blocks = indexer_cpu_blocks
            self.indexer_gpu_kv_strides_tensor = torch.tensor(indexer_gpu_kv_strides, dtype=torch.int64)
            self.indexer_gpu_block_strides_tensor = torch.tensor(indexer_gpu_block_strides, dtype=torch.int64)
            self.indexer_gpu_layer_strides_tensor = torch.tensor(indexer_gpu_layer_strides, dtype=torch.int64)
            self.indexer_gpu_chunk_sizes_tensor = torch.tensor(indexer_gpu_chunk_sizes, dtype=torch.int64)

            flexkv_logger.info(
                f"[LayerwiseWorker] Indexer fuse enabled: "
                f"gpu_blocks={len(imported_indexer_gpu_blocks)}, "
                f"cpu_size={indexer_cpu_blocks.numel() * indexer_cpu_blocks.element_size() / (1024 ** 2):.2f} MB, "
                f"chunk_size={indexer_gpu_chunk_sizes[0]} bytes, "
                f"cpu_block_stride={self.indexer_cpu_block_stride_in_bytes} bytes, "
                f"cpu_layer_stride={self.indexer_cpu_layer_stride_in_bytes} bytes")
        else:
            self.indexer_cpu_block_stride_in_bytes = 0
            self.indexer_cpu_layer_stride_in_bytes = 0
            self.indexer_h2d_cpu_kv_stride_in_bytes = 0
            self.indexer_h2d_cpu_layer_stride_in_bytes = 0
            self.indexer_gpu_blocks = []
            self.indexer_cpu_blocks = torch.Tensor()
            self.indexer_gpu_kv_strides_tensor = torch.empty(0, dtype=torch.int64)
            self.indexer_gpu_block_strides_tensor = torch.empty(0, dtype=torch.int64)
            self.indexer_gpu_layer_strides_tensor = torch.empty(0, dtype=torch.int64)
            self.indexer_gpu_chunk_sizes_tensor = torch.empty(0, dtype=torch.int64)

        # Initialize indexer SSD support
        self.enable_indexer_ssd = (
            self.enable_indexer and
            indexer_ssd_files is not None and len(indexer_ssd_files) > 0 and
            indexer_ssd_kv_layout is not None
        )
        if self.enable_indexer_ssd:
            assert indexer_dtype is not None
            self.indexer_ssd_files = indexer_ssd_files
            self.indexer_num_blocks_per_file = indexer_num_blocks_per_file

            indexer_ssd_kv_layout_per_file = indexer_ssd_kv_layout.div_block(
                sum(len(fl) for fl in indexer_ssd_files.values()), padding=True)
            self.indexer_ssd_kv_stride_in_bytes = indexer_ssd_kv_layout_per_file.get_kv_stride() * indexer_dtype.itemsize
            self.indexer_ssd_layer_stride_in_bytes = indexer_ssd_kv_layout_per_file.get_layer_stride() * indexer_dtype.itemsize
            self.indexer_cpu_chunk_size_in_bytes = indexer_cpu_kv_layout.get_chunk_size() * indexer_dtype.itemsize

            flexkv_logger.info(
                f"[LayerwiseWorker] Indexer SSD fuse enabled: "
                f"num_files={sum(len(fl) for fl in indexer_ssd_files.values())}, "
                f"num_blocks_per_file={indexer_num_blocks_per_file}, "
                f"ssd_kv_stride={self.indexer_ssd_kv_stride_in_bytes}, "
                f"ssd_layer_stride={self.indexer_ssd_layer_stride_in_bytes}, "
                f"cpu_chunk_size={self.indexer_cpu_chunk_size_in_bytes}")
        else:
            self.indexer_ssd_files = {}
            self.indexer_num_blocks_per_file = 0
            self.indexer_ssd_kv_stride_in_bytes = 0
            self.indexer_ssd_layer_stride_in_bytes = 0
            self.indexer_cpu_chunk_size_in_bytes = 0

        self.layerwise_transfer_group = LayerwiseTransferGroup(
            self.num_gpus, self.gpu_blocks, cpu_blocks, ssd_files,
            dp_group_id, self.num_layers,
            gpu_kv_strides_tensor, gpu_block_strides_tensor,
            gpu_layer_strides_tensor, gpu_chunk_sizes_tensor,
            GLOBAL_CONFIG_FROM_ENV.iouring_entries,
            GLOBAL_CONFIG_FROM_ENV.iouring_flags,
            layer_eventfds_tensor, tp_group_size,
            self.indexer_gpu_blocks, self.indexer_cpu_blocks,
            self.indexer_gpu_kv_strides_tensor, self.indexer_gpu_block_strides_tensor,
            self.indexer_gpu_layer_strides_tensor, self.indexer_gpu_chunk_sizes_tensor,
            self.indexer_ssd_files)
        flexkv_logger.info(f"[LayerwiseWorker] __init__ completed successfully, worker_id={worker_id}")

    def _receive_eventfds_from_sglang(self, tp_group_size: int,
                                       max_retries: int = 180,
                                       retry_interval: float = 1.0) -> torch.Tensor:
        """Receive eventfds from SGLang via Unix socket (FlexKV as server)."""
        socket_path = self.layerwise_eventfd_socket

        rank_parts = []
        if int(self.tp_group_size) > 1:
            rank_parts.append("tp_rank=0")
        if int(self.pp_size) > 1:
            rank_parts.append(f"pp_rank={int(self.pp_rank)}")
        if int(self.dp_size) > 1:
            rank_parts.append(f"dp_rank={int(self.dp_rank)}")
        rank_label = f" [{', '.join(rank_parts)}]" if rank_parts else ""

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
            # Use a larger backlog to accommodate client retries on failed connections
            server_sock.listen(tp_group_size * 3)
            os.chmod(socket_path, 0o777)
            flexkv_logger.info(
                f"[LayerwiseWorker] Eventfd server created{rank_label}: "
                f"socket={socket_path}, waiting for {tp_group_size} connection(s)")
        except Exception as e:
            flexkv_logger.error(
                f"[LayerwiseWorker] Failed to bind/listen on {socket_path}{rank_label}: {e}")
            server_sock.close()
            return torch.empty(0, dtype=torch.int32)

        # Use a per-connection timeout instead of a global one so that
        # failed connections can be retried by the client without the server
        # giving up too early.  The total deadline is still bounded.
        per_conn_timeout = 30  # seconds per accept() call
        total_deadline = time.time() + max_retries * retry_interval
        server_sock.settimeout(per_conn_timeout)
        all_rank_eventfds: Dict[int, Dict[int, List[int]]] = {}
        num_layers, num_counters = self.num_layers, 3
        conn_idx = 0

        try:
            # Keep accepting until we have eventfds from all ranks or deadline.
            while len(all_rank_eventfds) < tp_group_size:
                if time.time() > total_deadline:
                    flexkv_logger.error(
                        f"[LayerwiseWorker] Deadline exceeded on {socket_path}{rank_label}, "
                        f"received {len(all_rank_eventfds)}/{tp_group_size} ranks")
                    break

                remaining = total_deadline - time.time()
                server_sock.settimeout(min(per_conn_timeout, max(remaining, 1)))

                try:
                    conn, _ = server_sock.accept()
                    conn_idx += 1
                    flexkv_logger.info(
                        f"[LayerwiseWorker] Accepted connection "
                        f"{conn_idx} (registered {len(all_rank_eventfds)}/{tp_group_size}) "
                        f"on {socket_path}{rank_label}")
                except socket.timeout:
                    flexkv_logger.warning(
                        f"[LayerwiseWorker] Timeout waiting for connection on {socket_path}{rank_label}, "
                        f"registered {len(all_rank_eventfds)}/{tp_group_size}, retrying...")
                    continue

                try:
                    with conn:
                        # Accept both 16-byte (legacy: tp_rank, tp_size, num_layers, num_counters)
                        # and 24-byte (new: tp_rank, tp_size, cp_rank, cp_size, num_layers, num_counters)
                        metadata = conn.recv(24)
                        if len(metadata) < 16:
                            flexkv_logger.error(
                                f"[LayerwiseWorker] Incomplete metadata on {socket_path}{rank_label}: "
                                f"{len(metadata)} bytes")
                            continue

                        if len(metadata) >= 24:
                            tp_rank, _, cp_rank, cp_size, recv_num_layers, recv_num_counters = \
                                struct.unpack("iiiiii", metadata[:24])
                        else:
                            tp_rank, _, recv_num_layers, recv_num_counters = \
                                struct.unpack("iiii", metadata[:16])
                            cp_rank, cp_size = 0, 1

                        # Use cp_rank as the connection key when CP is active,
                        # otherwise use tp_rank
                        rank_key = cp_rank if cp_size > 1 else tp_rank
                        if not all_rank_eventfds:
                            num_layers, num_counters = recv_num_layers, recv_num_counters

                        flexkv_logger.debug(
                            f"[LayerwiseWorker] Connection {conn_idx}: "
                            f"tp_rank={tp_rank}, cp_rank={cp_rank}, cp_size={cp_size}, "
                            f"num_layers={recv_num_layers}, "
                            f"num_counters={recv_num_counters}")

                        rank_eventfds = {}
                        for _ in range(recv_num_counters):
                            fds, extra_data = _recv_fds(conn, recv_num_layers)
                            counter_id = struct.unpack("i", extra_data[:4])[0]
                            rank_eventfds[counter_id] = fds
                            flexkv_logger.debug(
                                f"[LayerwiseWorker] Received counter_id={counter_id}, "
                                f"num_fds={len(fds)} from rank_key={rank_key}")

                        all_rank_eventfds[rank_key] = rank_eventfds
                        # Send ACK to client so it knows the fds were received
                        try:
                            conn.sendall(b"\x01")
                        except Exception:
                            pass
                        flexkv_logger.info(
                            f"[LayerwiseWorker] Received all eventfds from rank_key={rank_key} "
                            f"(tp_rank={tp_rank}, cp_rank={cp_rank}) on {socket_path}")
                except Exception as e:
                    # Send NACK so client knows to retry
                    try:
                        conn.sendall(b"\x00")
                    except Exception:
                        pass
                    flexkv_logger.warning(
                        f"[LayerwiseWorker] Failed to receive eventfds from connection {conn_idx} "
                        f"on {socket_path}{rank_label}: {e}. "
                        f"Client will retry, continuing accept loop...")
                    continue
        except Exception as e:
            flexkv_logger.error(
                f"[LayerwiseWorker] Fatal error in accept loop on {socket_path}{rank_label}: {e}")
        finally:
            server_sock.close()
            cleanup_socket()

        if not all_rank_eventfds:
            flexkv_logger.warning(
                f"[LayerwiseWorker] No connections received on {socket_path}{rank_label}")
            return torch.empty(0, dtype=torch.int32)

        # Build tensor: [num_counters, tp_size, num_layers]
        eventfds_list = []
        for counter_id in range(num_counters):
            for tp_rank in range(tp_group_size):
                fds = all_rank_eventfds.get(tp_rank, {}).get(counter_id, [-1] * num_layers)
                eventfds_list.extend(fds)

        tensor = torch.tensor(eventfds_list, dtype=torch.int32)
        flexkv_logger.info(
            f"[LayerwiseWorker] Eventfd setup complete{rank_label}: "
            f"socket={socket_path}, tensor_shape={tensor.shape}, "
            f"counters={num_counters}, tp_size={tp_group_size}, layers={num_layers}"
        )
        return tensor

    def _transfer_impl(self,
                      src_block_ids_h2d: torch.Tensor,
                      dst_block_ids_h2d: torch.Tensor,
                      src_block_ids_disk2h: Optional[torch.Tensor],
                      dst_block_ids_disk2h: Optional[torch.Tensor],
                      layer_granularity: int,
                      counter_id: int = 0,
                      indexer_src_block_ids: Optional[torch.Tensor] = None,
                      indexer_dst_block_ids: Optional[torch.Tensor] = None,
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

        # Prepare indexer block_ids for fused transfer
        indexer_gpu_block_id_tensor = torch.Tensor()
        indexer_cpu_block_id_tensor = torch.Tensor()
        if self.enable_indexer and indexer_dst_block_ids is not None and len(indexer_dst_block_ids) > 0:
            indexer_gpu_block_id_tensor = indexer_dst_block_ids
            indexer_cpu_block_id_tensor = indexer_src_block_ids

        # Prepare indexer SSD block_ids for fused DISK2H transfer
        indexer_ssd_block_ids_tensor = torch.Tensor()
        indexer_cpu_block_ids_d2h_tensor = torch.Tensor()
        if self.enable_indexer_ssd and src_block_ids_disk2h is not None:
            # Indexer SSD block_ids mirror main KV's DISK2H block_ids (1:1 mapping)
            indexer_ssd_block_ids_tensor = ssd_block_ids
            indexer_cpu_block_ids_d2h_tensor = cpu_block_ids_d2h

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
            self.h2d_cpu_kv_stride_in_bytes,
            self.h2d_cpu_layer_stride_in_bytes,
            self.cpu_tp_stride_in_bytes,
            self.h2d_cta_num,
            self.use_ce_transfer_h2d,
            self.num_layers,
            layer_granularity,
            self.is_mla,
            counter_id,
            indexer_gpu_block_id_tensor,
            indexer_cpu_block_id_tensor,
            self.indexer_cpu_block_stride_in_bytes,
            self.indexer_cpu_layer_stride_in_bytes,
            self.indexer_h2d_cpu_kv_stride_in_bytes,
            self.indexer_h2d_cpu_layer_stride_in_bytes,
            indexer_ssd_block_ids_tensor,
            indexer_cpu_block_ids_d2h_tensor,
            self.indexer_ssd_layer_stride_in_bytes,
            self.indexer_ssd_kv_stride_in_bytes,
            self.indexer_cpu_chunk_size_in_bytes,
            self.indexer_num_blocks_per_file,
        )

    def launch_transfer(self, transfer_op: WorkerLayerwiseTransferOp) -> bool:
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

        # Extract indexer block_ids if available
        indexer_src_block_ids = None
        indexer_dst_block_ids = None
        if self.enable_indexer and transfer_op.indexer_src_block_ids.size > 0:
            indexer_src_block_ids = torch.from_numpy(
                transfer_op.indexer_src_block_ids).to(dtype=torch.int64).pin_memory()
            indexer_dst_block_ids = torch.from_numpy(
                transfer_op.indexer_dst_block_ids).to(dtype=torch.int64).pin_memory()

        num_h2d_blocks = len(src_block_ids_h2d)

        start_time = time.time()
        self._transfer_impl(
            src_block_ids_h2d,
            dst_block_ids_h2d,
            src_block_ids_disk2h,
            dst_block_ids_disk2h,
            layer_granularity,
            transfer_op.counter_id,
            indexer_src_block_ids=indexer_src_block_ids,
            indexer_dst_block_ids=indexer_dst_block_ids,
        )
        end_time = time.time()

        kv_dim = 2 if not self.is_mla else 1
        transfer_size = self.cpu_chunk_size_in_bytes * self.num_layers * num_h2d_blocks * kv_dim

        if self.is_nsa_cp or self.is_mla:
            transfer_size *= self.tp_group_size

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )

        return True
