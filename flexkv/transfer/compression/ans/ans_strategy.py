from __future__ import annotations

import time
from typing import Any, Dict, Optional

import nvtx
import torch

from flexkv.common.storage import KVCacheLayoutType
from flexkv.common.transfer import TransferType
from flexkv.c_ext import transfer_kv_blocks_ssd_packed
from flexkv.transfer.compression.ans import ans_utils
from flexkv.transfer.compression.common.size_table import bind_size_table
from flexkv.transfer.compression.common.strategy import CompressionStrategy


class NvcompGpuCpuStrategy(CompressionStrategy):
    """ANS-compressed CPU<->GPU transfer for a TP group of any size.

    There used to be a second strategy (``NvcompGpuCpuTpStrategy``) because
    there used to be a second worker.  The two bound to different attribute
    shapes -- scalar ``gpu_kv_stride_in_bytes`` and module-level
    ``transfer_kv_blocks_ans_{comp,decomp}`` here, list-shaped
    ``gpu_chunk_sizes_in_bytes`` and ``tp_group_transfer_ans`` there -- so
    every field added to one had to be mirrored into the other.  Now that
    ``GPUCPUTransferWorker`` covers tp==1 as ``num_gpus == 1``, one strategy
    on ``tp_group_transfer_ans`` covers both: the cpp side already dispatches
    per rank, and with a single rank the ``num_kv_heads > 1`` branch reduces
    to exactly the old non-TP call (offset 0, table slice 0).

    The size table follows the same rule: canonical 3-D ``[blocks, layers,
    kv]`` when tp==1 or KV is replicated across ranks, per-rank 4-D otherwise
    (see ``ans_utils.size_table_shape``).  Only the 4-D form has a rank
    stride, and only then does the kernel need one.
    """

    def __init__(self, cpu_size_table: torch.Tensor):
        self._cpu_size_table = cpu_size_table
        self._table_ptr = 0
        self._table_rank_stride = 0
        self._table_block_stride = 0
        self._table_layer_stride = 0

    def attach(self, worker) -> None:
        if getattr(worker, "tp_group_transfer_groups", None) is not None:
            raise RuntimeError(
                "nvcomp is not supported for multi-group (heterogeneous KV) "
                "layouts: the per-group transfers bypass the compressor.")
        batch_size, data_type = ans_utils.tp_worker_config(
            gpu_chunk_sizes_in_bytes=worker.gpu_chunk_sizes_in_bytes,
            dtype=worker.dtype,
            tp_size=worker.num_gpus,
        )
        # A 4-D per-rank table is required exactly when the ranks hold
        # different bytes: head-sharded KV spread over more than one GPU.
        needs_rank_stride = worker.num_kv_heads > 1 and worker.num_gpus > 1
        (self._table_ptr, self._table_rank_stride,
         self._table_block_stride,
         self._table_layer_stride) = bind_size_table(
            self._cpu_size_table,
            "[GPUCPUTransferWorker] cpu_size_table",
            register=True,
            expected_dims=(4,) if needs_rank_stride else (3,),
        )
        if self._table_ptr == 0:
            raise RuntimeError(
                "GPUCPUTransferWorker: nvcomp is enabled but "
                "cpu_size_table was not supplied.")
        worker.tp_transfer_thread_group.init_nvcomp(batch_size, data_type)

    def run(self, worker, op, src_block_ids, dst_block_ids) -> None:
        start_time = time.time()
        compressed_bytes = self._dispatch(
            worker, src_block_ids, dst_block_ids, op.transfer_type)
        end_time = time.time()
        uncomp_size = (
            worker.chunk_size_in_bytes
            * worker.num_layers
            * op.valid_block_num
            * worker.kv_dim
        )
        worker._log_transfer_performance(
            op, int(compressed_bytes), start_time, end_time,
            uncompressed_size=uncomp_size)

    def _dispatch(self, worker, src_block_ids, dst_block_ids, transfer_type) -> int:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2D:
            gpu_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
            use_ce_transfer = worker.use_ce_transfer_h2d
            transfer_num_cta = worker.transfer_num_cta_h2d
        elif transfer_type == TransferType.D2H:
            gpu_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
            use_ce_transfer = worker.use_ce_transfer_d2h
            transfer_num_cta = worker.transfer_num_cta_d2h
        else:
            raise ValueError(
                f"Invalid transfer type: {transfer_type} for GPUCPUTransferWorker")

        if len(gpu_block_id_list) == 0:
            return 0

        nvtx_range = nvtx.start_range(
            message=f"NvcompGpuCpuStrategy.run[{transfer_type.name}]",
            color="purple")
        try:
            # Keyword args, not positional: the cpp signature takes
            # num_kv_heads between kv_dim and the size-table pointer, and the
            # old positional call omitted it -- the four table values shifted
            # up by one and the call was one argument short of the binding.
            # It never fired because nvcomp+TP had no test that reached here.
            return int(worker.tp_transfer_thread_group.tp_group_transfer_ans(
                gpu_block_id_tensor=gpu_block_id_list,
                cpu_block_id_tensor=cpu_block_id_list,
                cpu_kv_stride_in_bytes=worker.cpu_kv_stride_in_bytes,
                cpu_layer_stride_in_bytes=worker.cpu_layer_stride_in_bytes,
                cpu_block_stride_in_bytes=worker.cpu_block_stride_in_bytes,
                cpu_tp_stride_in_bytes=worker.cpu_tp_stride_in_bytes,
                transfer_num_cta=transfer_num_cta,
                is_host_to_device=(transfer_type == TransferType.H2D),
                use_ce_transfer=use_ce_transfer,
                layer_id=0,
                layer_granularity=worker.num_layers,
                kv_dim=worker.kv_dim,
                num_kv_heads=worker.num_kv_heads,
                cpu_size_table_tp_ptr=self._table_ptr,
                cpu_size_table_tp_rank_stride=self._table_rank_stride,
                cpu_size_table_block_stride=self._table_block_stride,
                cpu_size_table_layer_stride=self._table_layer_stride,
            ))
        finally:
            nvtx.end_range(nvtx_range)


class NvcompCpuSsdStrategy(CompressionStrategy):
    def __init__(
        self,
        *,
        cpu_size_table,
        ssd_size_table,
        cpu_size_table_tp,
        ssd_size_table_tp,
        tp_size: int,
    ):
        self._cpu_size_table = cpu_size_table
        self._ssd_size_table = ssd_size_table
        self._cpu_size_table_tp = cpu_size_table_tp
        self._ssd_size_table_tp = ssd_size_table_tp
        self._tp_size = tp_size

    def attach(self, worker) -> None:
        (self._write_threads,
         self._read_threads) = ans_utils.ssd_packed_threads(worker.num_kv_heads == 1)

        selected_cpu = (self._cpu_size_table_tp
                        if self._cpu_size_table_tp is not None
                        else self._cpu_size_table)
        selected_ssd = (self._ssd_size_table_tp
                        if self._ssd_size_table_tp is not None
                        else self._ssd_size_table)
        ans_utils.check_worker_nvcomp_enable(
            True,
            path="cpu_ssd",
            cpu_size_table=selected_cpu,
            ssd_size_table=selected_ssd,
            tp_size=self._tp_size if self._cpu_size_table_tp is not None else 1,
        )

        (self._cpu_table_ptr, _,
         self._cpu_table_block_stride,
         self._cpu_table_layer_stride) = bind_size_table(
            self._cpu_size_table,
            "[CPUSSDDiskTransferWorker] cpu_size_table",
            expected_dims=(3,),
        )
        (self._ssd_table_ptr, _,
         self._ssd_table_block_stride,
         self._ssd_table_layer_stride) = bind_size_table(
            self._ssd_size_table,
            "[CPUSSDDiskTransferWorker] ssd_size_table",
            expected_dims=(3,),
        )

        self._per_rank_chunk_in_bytes = 0
        self._cpu_table_rank_stride_tp = 0
        self._ssd_table_rank_stride_tp = 0
        if self._cpu_size_table_tp is not None:
            assert self._ssd_size_table_tp is not None, \
                "cpu_size_table_tp and ssd_size_table_tp must be supplied together"
            self._per_rank_chunk_in_bytes = (
                worker.chunk_size_in_bytes // self._tp_size)
            (_, self._cpu_table_rank_stride_tp,
             self._cpu_table_block_stride_tp,
             self._cpu_table_layer_stride_tp) = bind_size_table(
                self._cpu_size_table_tp,
                "[CPUSSDDiskTransferWorker] cpu_size_table_tp",
                expected_dims=(4,),
            )
            (_, self._ssd_table_rank_stride_tp,
             self._ssd_table_block_stride_tp,
             self._ssd_table_layer_stride_tp) = bind_size_table(
                self._ssd_size_table_tp,
                "[CPUSSDDiskTransferWorker] ssd_size_table_tp",
                expected_dims=(4,),
            )

    def run(self, worker, op, src_block_ids, dst_block_ids) -> None:
        start_time = time.time()
        transfer_size = self._dispatch(
            worker, src_block_ids, dst_block_ids, op.transfer_type)
        end_time = time.time()
        uncomp_size = (
            worker.chunk_size_in_bytes
            * worker.num_layers
            * op.valid_block_num
            * worker.kv_dim
        )
        worker._log_transfer_performance(
            op, int(transfer_size), start_time, end_time,
            uncompressed_size=uncomp_size)

    def _build_ssd_packed_kwargs(
        self,
        worker,
        transfer_type: TransferType,
    ) -> Dict[str, Any]:
        num_threads = (
            self._read_threads
            if transfer_type == TransferType.DISK2H
            else self._write_threads
        )
        use_ranked_table = self._cpu_size_table_tp is not None and worker.num_kv_heads > 1
        use_canonical_rank0_table = self._cpu_size_table_tp is not None and worker.num_kv_heads == 1

        if use_ranked_table:
            if worker.cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
                cpu_layer_stride = worker.cpu_layer_stride_in_bytes // self._tp_size
                cpu_kv_stride = worker.cpu_kv_stride_in_bytes // self._tp_size
                cpu_tp_rank_stride = worker.block_stride_in_bytes // self._tp_size
            elif worker.cpu_layout_type == KVCacheLayoutType.LAYERFIRST:
                cpu_layer_stride = worker.cpu_layer_stride_in_bytes
                cpu_kv_stride = worker.cpu_kv_stride_in_bytes
                cpu_tp_rank_stride = self._per_rank_chunk_in_bytes
            else:
                raise RuntimeError(
                    "CPUSSDDiskTransferWorker: nvcomp+SSD only supports "
                    "BLOCKFIRST/LAYERFIRST layouts; requested "
                    f"layout={worker.cpu_layout_type.name}.")

            return {
                "cpu_layer_stride_in_bytes": cpu_layer_stride,
                "cpu_kv_stride_in_bytes": cpu_kv_stride,
                "chunk_size_in_bytes": self._per_rank_chunk_in_bytes,
                "num_threads_per_device": num_threads,
                "cpu_size_table_ptr": self._cpu_size_table_tp.data_ptr(),
                "cpu_size_table_rank_stride": self._cpu_table_rank_stride_tp,
                "cpu_size_table_block_stride": self._cpu_table_block_stride_tp,
                "cpu_size_table_layer_stride": self._cpu_table_layer_stride_tp,
                "ssd_size_table_ptr": self._ssd_size_table_tp.data_ptr(),
                "ssd_size_table_rank_stride": self._ssd_table_rank_stride_tp,
                "ssd_size_table_block_stride": self._ssd_table_block_stride_tp,
                "ssd_size_table_layer_stride": self._ssd_table_layer_stride_tp,
                "kv_dim": worker.kv_dim,
                "tp_size": self._tp_size,
                "cpu_tp_rank_stride_in_bytes": cpu_tp_rank_stride,
            }

        if use_canonical_rank0_table:
            cpu_table_ptr = self._cpu_size_table_tp.data_ptr()
            ssd_table_ptr = self._ssd_size_table_tp.data_ptr()
            cpu_block_stride = self._cpu_table_block_stride_tp
            cpu_layer_stride_tbl = self._cpu_table_layer_stride_tp
            ssd_block_stride = self._ssd_table_block_stride_tp
            ssd_layer_stride_tbl = self._ssd_table_layer_stride_tp
        else:
            cpu_table_ptr = self._cpu_table_ptr
            ssd_table_ptr = self._ssd_table_ptr
            cpu_block_stride = self._cpu_table_block_stride
            cpu_layer_stride_tbl = self._cpu_table_layer_stride
            ssd_block_stride = self._ssd_table_block_stride
            ssd_layer_stride_tbl = self._ssd_table_layer_stride

        return {
            "cpu_layer_stride_in_bytes": worker.cpu_layer_stride_in_bytes,
            "cpu_kv_stride_in_bytes": worker.cpu_kv_stride_in_bytes,
            "chunk_size_in_bytes": worker.chunk_size_in_bytes,
            "num_threads_per_device": num_threads,
            "cpu_size_table_ptr": cpu_table_ptr,
            "cpu_size_table_rank_stride": 0,
            "cpu_size_table_block_stride": cpu_block_stride,
            "cpu_size_table_layer_stride": cpu_layer_stride_tbl,
            "ssd_size_table_ptr": ssd_table_ptr,
            "ssd_size_table_rank_stride": 0,
            "ssd_size_table_block_stride": ssd_block_stride,
            "ssd_size_table_layer_stride": ssd_layer_stride_tbl,
            "kv_dim": worker.kv_dim,
            "tp_size": 1,
            "cpu_tp_rank_stride_in_bytes": 0,
        }

    def _dispatch(self, worker, src_block_ids, dst_block_ids, transfer_type) -> int:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2DISK:
            ssd_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
        elif transfer_type == TransferType.DISK2H:
            ssd_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
        else:
            raise ValueError(
                f"Invalid transfer type: {transfer_type} for CPUSSDDiskTransferWorker")

        layer_id_list = torch.arange(worker.num_layers, dtype=torch.int32)
        packed_kwargs = self._build_ssd_packed_kwargs(worker, transfer_type)
        compressed_bytes = transfer_kv_blocks_ssd_packed(
            ioctx=worker.ioctx,
            cpu_layer_id_list=layer_id_list,
            cpu_tensor_ptr=worker.cpu_layer_ptrs[0].item(),
            ssd_block_ids=ssd_block_id_list,
            cpu_block_ids=cpu_block_id_list,
            block_stride_in_bytes=worker.block_stride_in_bytes,
            is_read=(transfer_type == TransferType.DISK2H),
            num_blocks_per_file=worker.num_blocks_per_file,
            layout_type=worker.cpu_layout_type.value,
            total_layers=worker.num_layers,
            round_robin=worker.round_robin,
            **packed_kwargs,
        )
        worker._last_nvcomp_ssd_path = (
            "packed_blockfirst"
            if worker.cpu_layout_type == KVCacheLayoutType.BLOCKFIRST
            else "packed_layerfirst")
        return int(compressed_bytes)
