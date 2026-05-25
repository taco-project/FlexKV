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
from flexkv.common.config import (
    ModelConfig, GLOBAL_CONFIG_FROM_ENV,
    LayerGroupSpec, build_layer_member_map,
)
from flexkv.common.transfer import WorkerKey
from flexkv.storage.allocator import HugePageTensorHandle, materialize_worker_tensor

from flexkv.transfer.worker_op import WorkerLayerwiseTransferOp
from flexkv.transfer.worker import TransferWorkerBase, cudaHostRegister


def build_layerwise_eventfd_socket_path(
    dp_client_id: int,
    pp_rank: int,
    model_config: ModelConfig,
) -> str:
    """Construct the LayerwiseWorker's UDS socket path."""
    base = os.environ.get(
        'FLEXKV_LAYERWISE_EVENTFD_SOCKET',
        '/tmp/flexkv_layerwise_eventfd.sock',
    )
    suffix = ""
    if model_config.pp_size > 1:
        suffix += f"_pp{pp_rank}"
    if model_config.instance_num > 1 or model_config.dp_size > 1:
        suffix += f"_dp{dp_client_id}"
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
                 cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
                 ssd_files: Dict[int, List[str]],
                 gpu_kv_layouts: List[KVCacheLayout],
                 cpu_kv_layout: KVCacheLayout,
                 ssd_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 tp_group_size: int,
                 layerwise_eventfd_socket: str,
                 num_blocks_per_file: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 h2d_cta_num: int = 4,
                 d2h_cta_num: int = 4,
                 enable_eventfd: bool = True,
                 layer_groups: Optional[List[LayerGroupSpec]] = None,
                 gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]] = None,
                 gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]] = None) -> None:
        flexkv_logger.debug(
            f"[LayerwiseWorker] __init__ started: worker_id={worker_id}, "
            f"tp_group_size={tp_group_size}, "
            f"enable_eventfd={enable_eventfd}, "
            f"num_gpu_blocks={[len(b) for b in gpu_blocks]}, "
            f"multi_group={'yes' if layer_groups is not None else 'no'}")
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        assert len(gpu_blocks) == tp_group_size, f"len(gpu_blocks) = {len(gpu_blocks)}, tp_group_size = {tp_group_size}"
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            blocks_in_one_gpu = []
            for handle in handles_in_one_gpu:
                blocks_in_one_gpu.append(handle.get_tensor())
            imported_gpu_blocks.append(blocks_in_one_gpu)
        self.gpu_blocks = imported_gpu_blocks
        self.dtype = dtype # note this should be quantized data type (uint8 in multi-group)
        self.is_mla = gpu_kv_layouts[0].is_mla
        self.kv_dim = 1 if self.is_mla else 2

        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        # Pre-computed UDS socket path.  Both ends (this worker and the
        # sglang connector) derive the path from the same ModelConfig
        # fields (pp_rank / dp_rank / node_rank / is_multinode_tp), so no
        # env-var plumbing between processes is required.
        self.layerwise_eventfd_socket = layerwise_eventfd_socket

        # ``num_layers`` is the *original* layer count for this PP stage —
        # the framework's per-layer eventfd index space. Same field is used
        # in both single-group and multi-group modes; in multi-group it
        # equals ``cpu_kv_layout.num_layer`` (== num_layers_per_pp_stage).
        self.num_layers = gpu_kv_layouts[0].num_layer

        self.has_multi_group = (
            layer_groups is not None
            and gpu_blocks_per_group is not None
            and gpu_layouts_per_group is not None
        )

        num_blocks_first_gpu = len(imported_gpu_blocks[0]) if imported_gpu_blocks else 0
        if num_blocks_first_gpu == 1:
            self.gpu_block_type_ = 1  # TRTLLM
        elif num_blocks_first_gpu == self.num_layers:
            self.gpu_block_type_ = 0  # VLLM
        elif num_blocks_first_gpu == self.num_layers * 2:
            self.gpu_block_type_ = 2  # SGLANG
        elif self.has_multi_group:
            # In multi-group mode the legacy ``gpu_blocks`` list is the
            # aggregated per-rank view; per-group tensors live in
            # ``gpu_blocks_per_group``. Skip the legacy-shape check.
            self.gpu_block_type_ = -1
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

        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h
        self.h2d_cta_num = h2d_cta_num
        self.d2h_cta_num = d2h_cta_num

        # initialize SSD storage (file count etc. — strides handled per-group below)
        self.enable_ssd = len(ssd_files) > 0
        self.ssd_files = ssd_files
        if self.enable_ssd:
            self.num_blocks_per_file = num_blocks_per_file
            self.num_files = sum(len(file_list) for file_list in ssd_files.values())
            self.round_robin = 1
        else:
            self.num_blocks_per_file = 0
            self.round_robin = 1

        if self.has_multi_group:
            self._init_multi_group(
                cpu_kv_layout=cpu_kv_layout,
                ssd_kv_layout=ssd_kv_layout,
                ssd_files=ssd_files,
                layer_groups=layer_groups,
                gpu_blocks_per_group=gpu_blocks_per_group,
                gpu_layouts_per_group=gpu_layouts_per_group,
                cpu_blocks=cpu_blocks,
                layer_eventfds_tensor=layer_eventfds_tensor,
                tp_group_size=tp_group_size,
            )
        else:
            self._init_single_group(
                gpu_kv_layouts=gpu_kv_layouts,
                cpu_kv_layout=cpu_kv_layout,
                ssd_kv_layout=ssd_kv_layout,
                ssd_files=ssd_files,
                cpu_blocks=cpu_blocks,
                layer_eventfds_tensor=layer_eventfds_tensor,
                tp_group_size=tp_group_size,
            )

        flexkv_logger.info(f"[LayerwiseWorker] __init__ completed successfully, worker_id={worker_id}")

    # ------------------------------------------------------------------
    # Single-group init (legacy)
    # ------------------------------------------------------------------
    def _init_single_group(
        self,
        gpu_kv_layouts: List[KVCacheLayout],
        cpu_kv_layout: KVCacheLayout,
        ssd_kv_layout: KVCacheLayout,
        ssd_files: Dict[int, List[str]],
        cpu_blocks: torch.Tensor,
        layer_eventfds_tensor: torch.Tensor,
        tp_group_size: int,
    ) -> None:
        # here the chunk size doesn't include the layer info
        self.gpu_chunk_sizes_in_bytes = [gpu_kv_layout.get_chunk_size() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_kv_strides_in_bytes = [gpu_kv_layout.get_kv_stride() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_block_strides_in_bytes = [gpu_kv_layout.get_block_stride() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_layer_strides_in_bytes = [gpu_kv_layout.get_layer_stride() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]

        self.cpu_chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize
        # Full CPU strides (for SSD->CPU, which transfers all TP ranks' data)
        self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        # TP-divided CPU strides (for CPU->GPU, each rank reads its own portion)
        if cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST and not self.is_mla:
            cpu_kv_layout_tp = cpu_kv_layout.div_head(self.tp_group_size)
        else:
            cpu_kv_layout_tp = cpu_kv_layout
        self.cpu_tp_stride_in_bytes = self.cpu_block_stride_in_bytes // self.tp_group_size
        self.h2d_cpu_kv_stride_in_bytes = cpu_kv_layout_tp.get_kv_stride() * self.dtype.itemsize
        self.h2d_cpu_layer_stride_in_bytes = cpu_kv_layout_tp.get_layer_stride() * self.dtype.itemsize

        if self.enable_ssd:
            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)
            self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            self.ssd_block_stride_in_bytes = ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize
        else:
            self.ssd_kv_stride_in_bytes = 0
            self.ssd_layer_stride_in_bytes = 0
            self.ssd_block_stride_in_bytes = 0

        gpu_kv_strides_tensor = torch.tensor(self.gpu_kv_strides_in_bytes, dtype=torch.int64)
        gpu_block_strides_tensor = torch.tensor(self.gpu_block_strides_in_bytes, dtype=torch.int64)
        gpu_chunk_sizes_tensor = torch.tensor(self.gpu_chunk_sizes_in_bytes, dtype=torch.int64)
        gpu_layer_strides_tensor = torch.tensor(self.gpu_layer_strides_in_bytes, dtype=torch.int64)

        flexkv_logger.debug("[LayerwiseWorker] Creating LayerwiseTransferGroup (single-group)...")

        self.layerwise_transfer_group = LayerwiseTransferGroup(
            self.num_gpus, self.gpu_blocks, cpu_blocks, ssd_files,
            self.num_layers,
            gpu_kv_strides_tensor, gpu_block_strides_tensor,
            gpu_layer_strides_tensor, gpu_chunk_sizes_tensor,
            GLOBAL_CONFIG_FROM_ENV.iouring_entries,
            GLOBAL_CONFIG_FROM_ENV.iouring_flags,
            layer_eventfds_tensor, tp_group_size)

    # ------------------------------------------------------------------
    # Multi-group init
    # ------------------------------------------------------------------
    def _init_multi_group(
        self,
        cpu_kv_layout: KVCacheLayout,
        ssd_kv_layout: KVCacheLayout,
        ssd_files: Dict[int, List[str]],
        layer_groups: List[LayerGroupSpec],
        gpu_blocks_per_group: List[List[List[TensorSharedHandle]]],
        gpu_layouts_per_group: List[List[KVCacheLayout]],
        cpu_blocks: torch.Tensor,
        layer_eventfds_tensor: torch.Tensor,
        tp_group_size: int,
    ) -> None:
        """Initialize per-group params for the multi-group LayerwiseTransferGroup.

        CPU/SSD buffers are byte-flat (uint8) in multi-group mode. Each group's
        CPU region starts at ``cpu_offset_bytes`` inside a block and is sized
        per ``g.num_kv_heads * g.head_size * dtype_size_g`` (per-token). The
        SSD layout mirrors the CPU layout.
        """
        if cpu_kv_layout.type != KVCacheLayoutType.BLOCKFIRST:
            raise ValueError(
                "[LayerwiseWorker multi-group] only BLOCKFIRST CPU layout is "
                f"supported, got {cpu_kv_layout.type}"
            )
        if self.enable_ssd and ssd_kv_layout.type != KVCacheLayoutType.BLOCKFIRST:
            raise ValueError(
                "[LayerwiseWorker multi-group] only BLOCKFIRST SSD layout is "
                f"supported, got {ssd_kv_layout.type}"
            )

        kv_dim = self.kv_dim
        tpb = cpu_kv_layout.tokens_per_block
        num_original_layers = self.num_layers

        # CSR mapping
        layer_member_map = build_layer_member_map(layer_groups, num_original_layers)
        csr_offsets = list(layer_member_map.offsets)
        csr_group_idx = list(layer_member_map.group_idx)
        csr_local_id = list(layer_member_map.local_id)

        # Block-stride is identical for CPU and SSD in multi-group BLOCKFIRST
        # (kv_shape[1] = bytes_per_block, already accounts for tp_size and per-group dtypes).
        cpu_block_stride = cpu_kv_layout.get_block_stride()
        cpu_tp_stride = cpu_block_stride // tp_group_size

        group_num_layers: List[int] = []
        group_cpu_offset_bytes: List[int] = []
        group_ssd_offset_bytes: List[int] = []
        group_cpu_layer_strides: List[int] = []
        group_cpu_kv_strides: List[int] = []
        group_ssd_layer_strides: List[int] = []
        group_ssd_kv_strides: List[int] = []
        group_chunk_sizes: List[int] = []
        group_h2d_cpu_kv_strides: List[int] = []
        group_h2d_cpu_layer_strides: List[int] = []
        group_cpu_block_strides: List[int] = []
        group_cpu_tp_strides: List[int] = []
        group_gpu_kv_strides: List[int] = []
        group_gpu_block_strides: List[int] = []
        group_gpu_layer_strides: List[int] = []
        group_gpu_chunk_sizes: List[int] = []

        gpu_blocks_per_group_tensors: List[List[List[torch.Tensor]]] = []

        offset_bytes = 0
        for gi, g in enumerate(layer_groups):
            dtype_size_g = g.dtype.itemsize
            chunk_elements = tpb * g.num_kv_heads * g.head_size
            chunk_size_bytes = chunk_elements * dtype_size_g
            layer_stride_bytes = kv_dim * chunk_size_bytes
            kv_stride_bytes = chunk_size_bytes

            group_num_layers.append(g.num_layers)
            group_cpu_offset_bytes.append(offset_bytes)
            group_ssd_offset_bytes.append(offset_bytes)  # CPU and SSD share layout
            group_cpu_layer_strides.append(layer_stride_bytes)
            group_cpu_kv_strides.append(kv_stride_bytes)
            group_ssd_layer_strides.append(layer_stride_bytes)
            group_ssd_kv_strides.append(kv_stride_bytes)
            group_chunk_sizes.append(chunk_size_bytes)
            # In BLOCKFIRST + TP-divided layout, each rank reads only its own
            # head shard. The kv/layer strides for H2D are the per-group, per-token
            # widths — they're identical to the SSD->CPU strides because the
            # per-rank chunk is laid out the same way internally.
            group_h2d_cpu_kv_strides.append(kv_stride_bytes)
            group_h2d_cpu_layer_strides.append(layer_stride_bytes)
            group_cpu_block_strides.append(cpu_block_stride)
            group_cpu_tp_strides.append(cpu_tp_stride)

            # Materialize GPU tensors for this group, per device
            group_blocks_per_device = gpu_blocks_per_group[gi]
            group_layouts_per_device = gpu_layouts_per_group[gi]
            if len(group_blocks_per_device) != self.num_gpus:
                raise ValueError(
                    f"[LayerwiseWorker multi-group] gpu_blocks_per_group[{gi}] "
                    f"has {len(group_blocks_per_device)} devices, expected "
                    f"{self.num_gpus}"
                )

            imported_group_blocks: List[List[torch.Tensor]] = []
            for d, handles_for_device in enumerate(group_blocks_per_device):
                tensors_for_device = [h.get_tensor() for h in handles_for_device]
                imported_group_blocks.append(tensors_for_device)
            gpu_blocks_per_group_tensors.append(imported_group_blocks)

            for d, layout in enumerate(group_layouts_per_device):
                # Per-GPU GPU-side strides, in bytes (per-group dtype).
                group_gpu_kv_strides.append(layout.get_kv_stride() * dtype_size_g)
                group_gpu_block_strides.append(layout.get_block_stride() * dtype_size_g)
                group_gpu_layer_strides.append(layout.get_layer_stride() * dtype_size_g)
                group_gpu_chunk_sizes.append(layout.get_chunk_size() * dtype_size_g)

            offset_bytes += g.num_layers * layer_stride_bytes

        # Store for transfer_impl / logging
        self.layer_groups = layer_groups
        self.group_cpu_block_stride = cpu_block_stride
        self.csr_offsets = csr_offsets
        self.csr_group_idx = csr_group_idx
        self.csr_local_id = csr_local_id

        flexkv_logger.info(
            f"[LayerwiseWorker multi-group] {len(layer_groups)} groups, "
            f"num_original_layers={num_original_layers}, "
            f"block_stride={cpu_block_stride}, tp_stride={cpu_tp_stride}"
        )

        flexkv_logger.debug("[LayerwiseWorker] Creating LayerwiseTransferGroup (multi-group)...")
        self.layerwise_transfer_group = LayerwiseTransferGroup(
            num_gpus=self.num_gpus,
            gpu_blocks_per_group=gpu_blocks_per_group_tensors,
            cpu_blocks=cpu_blocks,
            ssd_files=ssd_files,
            num_original_layers=num_original_layers,
            csr_offsets=csr_offsets,
            csr_group_idx=csr_group_idx,
            csr_local_id=csr_local_id,
            group_num_layers=group_num_layers,
            group_cpu_offset_bytes=group_cpu_offset_bytes,
            group_ssd_offset_bytes=group_ssd_offset_bytes,
            group_cpu_layer_strides=group_cpu_layer_strides,
            group_cpu_kv_strides=group_cpu_kv_strides,
            group_ssd_layer_strides=group_ssd_layer_strides,
            group_ssd_kv_strides=group_ssd_kv_strides,
            group_chunk_sizes=group_chunk_sizes,
            group_h2d_cpu_kv_strides=group_h2d_cpu_kv_strides,
            group_h2d_cpu_layer_strides=group_h2d_cpu_layer_strides,
            group_cpu_block_strides=group_cpu_block_strides,
            group_cpu_tp_strides=group_cpu_tp_strides,
            group_gpu_kv_strides=group_gpu_kv_strides,
            group_gpu_block_strides=group_gpu_block_strides,
            group_gpu_layer_strides=group_gpu_layer_strides,
            group_gpu_chunk_sizes=group_gpu_chunk_sizes,
            iouring_entries=GLOBAL_CONFIG_FROM_ENV.iouring_entries,
            iouring_flags=GLOBAL_CONFIG_FROM_ENV.iouring_flags,
            layer_eventfds_tensor=layer_eventfds_tensor,
            tp_size=tp_group_size,
        )

    def _receive_eventfds_from_sglang(self, tp_group_size: int,
                                       max_retries: int = 180,
                                       retry_interval: float = 1.0) -> torch.Tensor:
        """Receive eventfds from SGLang via Unix socket (FlexKV as server)."""
        socket_path = self.layerwise_eventfd_socket

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
                f"[LayerwiseWorker] Eventfd server created: "
                f"socket={socket_path}, waiting for {tp_group_size} connection(s)")
        except Exception as e:
            flexkv_logger.error(
                f"[LayerwiseWorker] Failed to bind/listen on {socket_path}: {e}")
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
                        f"[LayerwiseWorker] Deadline exceeded on {socket_path}, "
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
                        f"on {socket_path}")
                except socket.timeout:
                    flexkv_logger.warning(
                        f"[LayerwiseWorker] Timeout waiting for connection on {socket_path}, "
                        f"registered {len(all_rank_eventfds)}/{tp_group_size}, retrying...")
                    continue

                try:
                    with conn:
                        # Receive 16-byte metadata: tp_rank_per_node, tp_size_per_node,
                        # num_layers, num_counters
                        metadata = conn.recv(16)
                        if len(metadata) < 16:
                            flexkv_logger.error(
                                f"[LayerwiseWorker] Incomplete metadata on {socket_path}: "
                                f"expected 16 bytes, got {len(metadata)}")
                            continue

                        rank_key, tp_size_per_node_recv, recv_num_layers, recv_num_counters = \
                            struct.unpack("iiii", metadata[:16])

                        if not all_rank_eventfds:
                            num_layers, num_counters = recv_num_layers, recv_num_counters

                        flexkv_logger.debug(
                            f"[LayerwiseWorker] Connection {conn_idx}: "
                            f"tp_rank_per_node={rank_key}, "
                            f"tp_size_per_node={tp_size_per_node_recv}, "
                            f"num_layers={recv_num_layers}, "
                            f"num_counters={recv_num_counters}")

                        rank_eventfds = {}
                        for _ in range(recv_num_counters):
                            fds, extra_data = _recv_fds(conn, recv_num_layers)
                            counter_id = struct.unpack("i", extra_data[:4])[0]
                            rank_eventfds[counter_id] = fds
                            flexkv_logger.debug(
                                f"[LayerwiseWorker] Received counter_id={counter_id}, "
                                f"num_fds={len(fds)} from tp_rank_per_node={rank_key}")

                        all_rank_eventfds[rank_key] = rank_eventfds
                        # Send ACK to client so it knows the fds were received
                        try:
                            conn.sendall(b"\x01")
                        except Exception:
                            pass
                        flexkv_logger.info(
                            f"[LayerwiseWorker] Received all eventfds from tp_rank_per_node={rank_key} "
                            f"on {socket_path}")
                except Exception as e:
                    # Send NACK so client knows to retry
                    try:
                        conn.sendall(b"\x00")
                    except Exception:
                        pass
                    flexkv_logger.warning(
                        f"[LayerwiseWorker] Failed to receive eventfds from connection {conn_idx} "
                        f"on {socket_path}: {e}. "
                        f"Client will retry, continuing accept loop...")
                    continue
        except Exception as e:
            flexkv_logger.error(
                f"[LayerwiseWorker] Fatal error in accept loop on {socket_path}: {e}")
        finally:
            server_sock.close()
            cleanup_socket()

        if not all_rank_eventfds:
            flexkv_logger.warning(
                f"[LayerwiseWorker] No connections received on {socket_path}")
            return torch.empty(0, dtype=torch.int32)

        # Build tensor: [num_counters, tp_size, num_layers]
        eventfds_list = []
        for counter_id in range(num_counters):
            for tp_rank in range(tp_group_size):
                fds = all_rank_eventfds.get(tp_rank, {}).get(counter_id, [-1] * num_layers)
                eventfds_list.extend(fds)

        tensor = torch.tensor(eventfds_list, dtype=torch.int32)
        flexkv_logger.info(
            f"[LayerwiseWorker] Eventfd setup complete: "
            f"socket={socket_path}, tensor_shape={tensor.shape}, "
            f"counters={num_counters}, tp_size_per_rank={tp_group_size}, layers={num_layers}"
        )
        return tensor

    def _transfer_impl(self,
                      src_block_ids_h2d: torch.Tensor,
                      dst_block_ids_h2d: torch.Tensor,
                      src_block_ids_disk2h: Optional[torch.Tensor],
                      dst_block_ids_disk2h: Optional[torch.Tensor],
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

        if self.has_multi_group:
            self.layerwise_transfer_group.layerwise_transfer_multi_group(
                ssd_block_ids=ssd_block_ids,
                cpu_block_ids_d2h=cpu_block_ids_d2h,
                num_blocks_per_file=self.num_blocks_per_file,
                round_robin=self.round_robin,
                num_threads_per_device=32,
                gpu_block_id_tensor=dst_block_ids_h2d,
                cpu_block_id_tensor=src_block_ids_h2d,
                transfer_cta_num=self.h2d_cta_num,
                use_ce_transfer=self.use_ce_transfer_h2d,
                is_mla=self.is_mla,
                counter_id=counter_id,
            )
            return

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
            1,  # layer_granularity: LAYERWISE protocol fires one eventfd per layer
            self.is_mla,
            counter_id,
        )

    def launch_transfer(self, transfer_op: WorkerLayerwiseTransferOp) -> bool:
        src_block_ids_h2d = torch.from_numpy(transfer_op.src_block_ids_h2d).to(dtype=torch.int64).pin_memory()
        dst_block_ids_h2d = torch.from_numpy(transfer_op.dst_block_ids_h2d).to(dtype=torch.int64).pin_memory()

        if transfer_op.src_block_ids_disk2h.size > 0:
            src_block_ids_disk2h = torch.from_numpy(transfer_op.src_block_ids_disk2h).to(dtype=torch.int64)
            dst_block_ids_disk2h = torch.from_numpy(transfer_op.dst_block_ids_disk2h).to(dtype=torch.int64)
        else:
            src_block_ids_disk2h = None
            dst_block_ids_disk2h = None

        num_h2d_blocks = len(src_block_ids_h2d)

        start_time = time.time()
        self._transfer_impl(
            src_block_ids_h2d,
            dst_block_ids_h2d,
            src_block_ids_disk2h,
            dst_block_ids_disk2h,
            transfer_op.counter_id,
        )
        end_time = time.time()

        if self.has_multi_group:
            # Multi-group: full block byte size already accounts for tp_size and all groups
            transfer_size = self.group_cpu_block_stride * num_h2d_blocks
        else:
            transfer_size = self.cpu_chunk_size_in_bytes * self.num_layers * num_h2d_blocks * self.kv_dim
            if self.is_mla:
                transfer_size *= self.tp_group_size

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )

        return True
