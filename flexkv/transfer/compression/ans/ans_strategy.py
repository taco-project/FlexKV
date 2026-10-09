from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

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

    # ANS goes through ``tp_group_transfer_ans`` on the thread group; the
    # region batch has no compressed entry point. So this strategy is the one
    # thing that keeps the thread group on the live path for a uniform pool.
    needs_gpu_cpu_thread_group = True

    def __init__(self, cpu_size_table: torch.Tensor,
                 group_plans: Optional[Sequence[ans_utils.GroupPlan]] = None):
        self._cpu_size_table = cpu_size_table
        # Resolved in the engine process and carried here, rather than re-read
        # from the env inside each worker: one list, so the engine's accept
        # decision and the worker's binding cannot disagree. Empty/None on a
        # uniform-KV model, which has no groups to select among.
        self._group_plans: List[ans_utils.GroupPlan] = list(group_plans or ())
        self._table_ptr = 0
        self._table_rank_stride = 0
        self._table_block_stride = 0
        self._table_layer_stride = 0
        # Set on a multi-group worker: one geometry dict per compressed group,
        # in group order. Empty on a uniform worker, where those numbers live
        # on the worker itself.
        self._groups: List[Dict[str, Any]] = []

    def compressed_group_indices(self) -> List[int]:
        """The groups ``_attach_multi_group`` bound to; empty if uniform.

        Read off what this instance actually holds strides for, rather than
        recomputed, so the worker splits its transfer on exactly the groups
        that were bound.
        """
        return [g['group_index'] for g in self._groups]

    def _uniform_geometry(self, worker) -> Dict[str, Any]:
        """The worker's own scalars, published by
        ``_init_uniform(expose_on_self=True)``. Only valid when the worker has
        no layer groups -- a multi-group worker never sets them, because each
        group carries its own base offset and stride set."""
        return {
            'thread_group': worker.tp_transfer_thread_group,
            'cpu_kv_stride': worker.cpu_kv_stride_in_bytes,
            'cpu_layer_stride': worker.cpu_layer_stride_in_bytes,
            'cpu_block_stride': worker.cpu_block_stride_in_bytes,
            'cpu_tp_stride': worker.cpu_tp_stride_in_bytes,
            'num_layers': worker.num_layers,
            'chunk_size': worker.chunk_size_in_bytes,
            # Uniform KV is the whole table; no group offset.
            'table_ptr': self._table_ptr,
        }

    def attach(self, worker) -> None:
        groups = getattr(worker, "tp_group_transfer_groups", None)
        if groups is not None:
            self._attach_multi_group(worker, groups)
            return
        self._attach_uniform(worker)

    def _attach_multi_group(self, worker, groups) -> None:
        """Bind to each compressed group of a heterogeneous-KV worker.

        The engine already decided which groups are compressible -- see
        ``ans_utils.select_nvcomp_groups``, which runs before any worker is
        spawned, and whose result was handed to this instance's constructor.
        Re-checking the *shape* here is still worth it: this process is the
        first place a group's realized geometry (after TP sharding and the IPC
        import) is visible, and a mismatch would otherwise surface as a
        wrong-length decompress rather than an error.
        """
        if not self._group_plans:
            raise RuntimeError(
                "nvcomp: attached to a multi-group worker with no compressed "
                "group plan; the engine should have declined nvcomp instead.")
        self._bind_table(worker)

        # Table rows are handed out by prefix sum over *every* group's layer
        # count, so two compressed groups can never write each other's lengths.
        # The engine computed those offsets from the declared specs; this
        # re-derives them from the realized per-group layer counts and requires
        # the two to agree, which is what makes the kernel's base pointer
        # arithmetic safe.
        realized_offsets = []
        running = 0
        for g in groups:
            realized_offsets.append(running)
            running += g['num_layers']
        if running > self._table_layer_rows:
            raise RuntimeError(
                f"nvcomp: the worker's layer groups span {running} table rows "
                f"but the CPU size table has only {self._table_layer_rows}; "
                "the table was sized for a different group layout.")

        for plan in self._group_plans:
            if plan.group_index >= len(groups):
                raise RuntimeError(
                    f"nvcomp: compressed group index {plan.group_index} is out "
                    f"of range for a worker with {len(groups)} layer groups.")
            group = dict(groups[plan.group_index])
            if group['num_layers'] != plan.num_layers:
                raise RuntimeError(
                    f"nvcomp: compressed layer group {plan.group_index} covers "
                    f"{group['num_layers']} layers in this worker but the "
                    f"engine sized its size-table rows for {plan.num_layers}.")
            if realized_offsets[plan.group_index] != plan.table_layer_offset:
                raise RuntimeError(
                    f"nvcomp: compressed layer group {plan.group_index} starts "
                    f"at table row {realized_offsets[plan.group_index]} in this "
                    f"worker but the engine assigned it row "
                    f"{plan.table_layer_offset}; compressing it would index the "
                    "table out of step with the other groups.")
            group['group_index'] = plan.group_index
            # One key name for the dispatcher, whichever shape it came from.
            group['thread_group'] = group['tp_thread_group']
            # Fold the group's row offset into the base pointer. The kernel
            # indexes ``base + block*block_stride + (0 + layer)*layer_stride``
            # with local layer ids starting at 0 for every group, so a shifted
            # base is exactly the per-group window -- and needs no change to
            # any C++ signature, since the base is already a per-call argument.
            group['table_ptr'] = (
                self._table_ptr
                + plan.table_layer_offset * self._table_layer_stride
                * self._cpu_size_table.element_size())
            self._groups.append(group)
            # The group's own dtype and per-device chunk, not the worker's. A
            # DSA model pairs a bf16 latent group with an fp8 indexer, so the
            # ANS symbol type has to come from the group being compressed;
            # ``worker.dtype`` is the registration-wide dtype and happens to
            # match group 0 today, which is exactly the kind of coincidence
            # that breaks silently once more than one group is compressed. The
            # chunk must be per-device: ``chunk_size`` is the whole-group chunk
            # the host strides over, tp_size times what one rank holds.
            group['tp_thread_group'].init_nvcomp(
                *ans_utils.tp_worker_config(
                    gpu_chunk_sizes_in_bytes=group['gpu_chunk_sizes'],
                    dtype=group['dtype'],
                    tp_size=worker.num_gpus,
                ))

    def _attach_uniform(self, worker) -> None:
        batch_size, data_type = ans_utils.tp_worker_config(
            gpu_chunk_sizes_in_bytes=worker.gpu_chunk_sizes_in_bytes,
            dtype=worker.dtype,
            tp_size=worker.num_gpus,
        )
        self._bind_table(worker)
        worker.tp_transfer_thread_group.init_nvcomp(batch_size, data_type)

    def _bind_table(self, worker) -> None:
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
        # How many layer rows the table actually has, so a per-group window can
        # be checked against it rather than trusted. Layer is the last-but-one
        # dim in both the canonical 3-D and the per-rank 4-D shape.
        self._table_layer_rows = self._cpu_size_table.shape[-2]

    def run(self, worker, op, src_block_ids, dst_block_ids) -> None:
        geoms = self._groups or [self._uniform_geometry(worker)]
        start_time = time.time()
        compressed_bytes = 0
        uncomp_size = 0
        for geom in geoms:
            compressed_bytes += self._dispatch(
                worker, src_block_ids, dst_block_ids, op.transfer_type, geom)
            # Only the compressed groups' bytes. On a multi-group worker any
            # remaining groups moved separately, on the uncompressed path, and
            # reported their own sizes -- folding them in here would understate
            # the ratio for the groups that were actually compressed.
            uncomp_size += (
                geom['chunk_size']
                * geom['num_layers']
                * op.valid_block_num
                * worker.kv_dim
            )
        end_time = time.time()
        worker._log_transfer_performance(
            op, int(compressed_bytes), start_time, end_time,
            uncompressed_size=uncomp_size)

    def _dispatch(self, worker, src_block_ids, dst_block_ids, transfer_type,
                  geom) -> int:
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
            message=(f"NvcompGpuCpuStrategy.run[{transfer_type.name}]"
                     f"[group{geom.get('group_index', 0)}]"),
            color="purple")
        try:
            # Keyword args past the block-id tensors.  #257 inserted
            # ``num_kv_heads`` between ``kv_dim`` and the size-table group, so
            # a positional call silently shifts the four table arguments by one
            # and comes up one short -- a TypeError before any byte moves.  The
            # unit tests are the only other caller and they pass it; this is
            # the live TP+nvcomp path.
            return int(geom['thread_group'].tp_group_transfer_ans(
                gpu_block_id_list, cpu_block_id_list,
                cpu_kv_stride_in_bytes=geom['cpu_kv_stride'],
                cpu_layer_stride_in_bytes=geom['cpu_layer_stride'],
                cpu_block_stride_in_bytes=geom['cpu_block_stride'],
                cpu_tp_stride_in_bytes=geom['cpu_tp_stride'],
                transfer_num_cta=transfer_num_cta,
                is_host_to_device=transfer_type == TransferType.H2D,
                use_ce_transfer=use_ce_transfer,
                layer_id=0,
                layer_granularity=geom['num_layers'],
                kv_dim=worker.kv_dim,
                num_kv_heads=worker.num_kv_heads,
                # This group's window into the table, not the table base: the
                # offset is folded into the pointer in ``_attach_multi_group``
                # so the kernel's local-layer-id arithmetic lands on the rows
                # that belong to this group.
                cpu_size_table_tp_ptr=geom['table_ptr'],
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
