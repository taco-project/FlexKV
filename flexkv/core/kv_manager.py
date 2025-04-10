from typing import Any, List, Optional, Tuple

import torch

from flexkv.core.utils import DeviceType
from flexkv.core.mempool import CPUMemPool, SSDMemPool
from flexkv.core.index import TokenToBlockIndex
from flexkv.core.evictor import create_evictor, EvictionPolicy
from flexkv.core.transfer import (
    GPUTransfer,
    TransferDescriptor,
    SSDTransfer,
)
from flexkv.core.block import BlockStatus, BlockMeta, BlockLocation
from flexkv.core.debug_utils import debug_timing, debuginfo
from flexkv.core.async_handler import (
    AsyncRequestHandler,
    RequestType,
    Response,
)


class KVManager:
    def __init__(
        self,
        num_cpu_blocks: int,
        num_ssd_blocks: int,
        tokens_per_block: int,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        gpu_physical_blocks: List[torch.Tensor],
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ):
        assert num_cpu_blocks > 0
        assert num_ssd_blocks > 0
        assert tokens_per_block > 0
        assert num_layers > 0
        assert num_kv_heads > 0
        assert head_size > 0

        self.num_cpu_blocks = num_cpu_blocks
        self.num_ssd_blocks = num_ssd_blocks
        self.tokens_per_block = tokens_per_block
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype

        self.block_size = (
            self.tokens_per_block * self.num_kv_heads * self.head_size
        )
        self.cpu_mem_pool = CPUMemPool(
            self.num_layers,
            self.num_cpu_blocks,
            self.block_size,
            self.dtype,
            pin_memory=True,
        )
        self.ssd_mem_pool = SSDMemPool(
            self.num_layers,
            self.num_ssd_blocks,
            self.block_size,
            self.dtype,
        )
        self.local_index = TokenToBlockIndex(self.tokens_per_block)
        self.evictor = create_evictor(eviction_policy)
        self.gpu_transfer = None

        if len(gpu_physical_blocks) > 0:
            self.add_gpu_blocks(gpu_physical_blocks)

    def reset(self) -> None:
        self.local_index.reset()
        self.cpu_mem_pool.reset()
        self.ssd_mem_pool.reset()

    @property
    def is_ready(self) -> bool:
        return self.gpu_transfer is not None

    def add_gpu_blocks(self, gpu_physical_blocks: List[torch.Tensor]):
        if not self.is_ready():
            self.gpu_transfer = GPUTransfer(
                gpu_blocks=gpu_physical_blocks,
                cpu_blocks=self.cpu_mem_pool.get_physical_blocks(),
            )
            # TODO: modified the finished_queue
            self.ssd_transfer = SSDTransfer(
                cpu_blocks=self.cpu_mem_pool.get_physical_blocks(),
                ssd_file=self.ssd_mem_pool.get_physical_blocks(),
                gpu_transfer=self.gpu_transfer,
                finished_queue=None,
            )
            self.gpu_transfer.set_ssd_transfer(self.ssd_transfer)
            self.request_handler = AsyncRequestHandler(
                get_func=self._get_impl,
                put_func=self._put_impl,
                gpu_transfer=self.gpu_transfer,
                ssd_transfer=self.ssd_transfer,
            )
        else:
            raise RuntimeError("KVManager is already ready")

    def async_get(
        self,
        token_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        gpu_physical_block_ids: torch.Tensor,
    ) -> int:
        return self.request_handler.submit_request(
            RequestType.GET, token_ids, token_mask, gpu_physical_block_ids
        )

    def async_put(
        self,
        token_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        gpu_physical_block_ids: torch.Tensor,
    ) -> int:
        return self.request_handler.submit_request(
            RequestType.PUT, token_ids, token_mask, gpu_physical_block_ids
        )

    @debug_timing("sync get")
    def get(
        self,
        token_ids: List[int],
        token_mask: Optional[torch.Tensor],
        dst_descriptor: Any,
    ) -> List[BlockMeta]:
        block_metas, _ = self._get_impl(token_ids, token_mask, dst_descriptor)
        # TODO: should we use check_and_set automic operation here?
        self.gpu_transfer.to_gpu(block_metas, dst_descriptor, True)
        return block_metas

    @debug_timing("sync put")
    def put(
        self,
        token_ids: List[int],
        token_mask: Optional[torch.Tensor],
        src_descriptor: Any,
    ) -> List[BlockMeta]:
        _, block_metas = self._put_impl(token_ids, token_mask, src_descriptor)
        self.gpu_transfer.from_gpu(src_descriptor, block_metas, True)
        return block_metas

    def get_results(self, request_ids: List[int]) -> List[Response]:
        return self.request_handler.get_results(request_ids)

    def wait_until_finished(self, request_ids: List[int]):
        return self.request_handler.wait_until_finished(request_ids)

    def _evict_cpu_blocks(self, num_required_blocks: int) -> int:
        if num_required_blocks > self.cpu_mem_pool.num_free_blocks:
            while self.cpu_mem_pool.num_free_blocks < num_required_blocks:
                # TODO if we can maintain a list of leave block metas
                evictable_blocks = (
                    self.local_index.get_cpu_leaf_blocks_to_offload()
                )
                if len(evictable_blocks) == 0:
                    break
                evicted_block_metas = self.evictor.evict(
                    evictable_blocks,
                    num_required_blocks - self.cpu_mem_pool.num_free_blocks,
                )
                self.cpu_mem_pool.free_blocks(
                    [
                        block_meta.cpu_block_id
                        for block_meta in evicted_block_metas
                    ]
                )
                self.local_index.offload_to_ssd(evicted_block_metas)

    def _evict_blocks(self, num_required_blocks: int) -> int:
        if num_required_blocks > self.ssd_mem_pool.num_free_blocks:
            while self.ssd_mem_pool.num_free_blocks < num_required_blocks:
                evictable_blocks = self.local_index.get_evictable_leaf_blocks()
                if len(evictable_blocks) == 0:
                    break
                evicted_block_metas = self.evictor.evict(
                    evictable_blocks,
                    num_required_blocks - self.ssd_mem_pool.num_free_blocks,
                )
                self.local_index.remove(evicted_block_metas)

                self.cpu_mem_pool.free_blocks(
                    [
                        block_meta.cpu_block_id
                        for block_meta in evicted_block_metas
                        if block_meta.location == BlockLocation.CPU
                    ]
                )

                self.ssd_mem_pool.free_blocks(
                    [
                        block_meta.ssd_block_id
                        for block_meta in evicted_block_metas
                    ]
                )

    def _mempool_internal_get(
        self, block_metas_to_get: List[BlockMeta]
    ) -> TransferDescriptor:
        block_metas_to_get_s2h = []
        last_cpu_block_meta = None
        for block_meta in block_metas_to_get:
            if block_meta.location == BlockLocation.SSD:
                block_metas_to_get_s2h.append(block_meta)
                block_meta.status = BlockStatus.LOCKED  # transfer to cpu

            if block_meta.location == BlockLocation.CPU:
                last_cpu_block_meta = block_meta

        num_required_blocks = len(block_metas_to_get_s2h)

        if last_cpu_block_meta:
            last_cpu_block_meta.status = BlockStatus.LOCKED
        self._evict_cpu_blocks(num_required_blocks)
        if last_cpu_block_meta:
            last_cpu_block_meta.status = BlockStatus.AVAILABLE

        assert num_required_blocks <= self.cpu_mem_pool.num_free_blocks

        cpu_block_ids = self.cpu_mem_pool.allocate_blocks(num_required_blocks)
        self.local_index.upload_to_cpu(block_metas_to_get_s2h, cpu_block_ids)

        ssd_descriptor = TransferDescriptor(
            blockmeta_list=block_metas_to_get_s2h,
            physical_block_ids=torch.tensor(
                [
                    block_meta.ssd_block_id
                    for block_meta in block_metas_to_get_s2h
                ],
                dtype=torch.int64,
                pin_memory=True,
            ),
            device=DeviceType.SSD,
        )

        return ssd_descriptor

    def _get_impl(
        self,
        token_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        gpu_physical_block_ids: torch.Tensor,
    ) -> Tuple[
        TransferDescriptor, TransferDescriptor, TransferDescriptor, torch.Tensor
    ]:
        if token_mask is None:
            token_mask = torch.ones(len(token_ids), dtype=torch.bool)
        if not len(token_mask) == len(token_ids):
            raise ValueError(
                "token_mask should have the same length as token_ids"
            )

        aligned_token_mask = self._check_and_align_token_mask(
            token_mask, prefix_mask_check=True
        )
        block_mask_range = self._get_block_range(aligned_token_mask)

        cached_block_metas, _ = self.local_index.match(token_ids)
        actual_transfer_block_start = max(0, block_mask_range[0])
        actual_transfer_block_end = min(
            len(cached_block_metas), block_mask_range[1]
        )
        if actual_transfer_block_start >= actual_transfer_block_end:
            return_mask = torch.zeros_like(token_mask, dtype=torch.bool)
            return (
                TransferDescriptor(),
                TransferDescriptor(),
                TransferDescriptor(),
                return_mask,
            )

        block_metas_to_get_s2h = []
        last_cpu_block_meta = None
        for block_meta in cached_block_metas:
            if block_meta.location == BlockLocation.SSD:
                block_metas_to_get_s2h.append(block_meta)
                block_meta.status = BlockStatus.LOCKED  # transfer to cpu

            if block_meta.location == BlockLocation.CPU:
                last_cpu_block_meta = block_meta

        num_cpu_required_blocks = len(block_metas_to_get_s2h)

        if last_cpu_block_meta:
            last_cpu_block_meta.status = BlockStatus.LOCKED
        self._evict_cpu_blocks(num_cpu_required_blocks)
        if last_cpu_block_meta:
            last_cpu_block_meta.status = BlockStatus.AVAILABLE

        num_cpu_actual_blocks = min(
            num_cpu_required_blocks, self.cpu_mem_pool.num_free_blocks
        )

        cpu_block_ids = self.cpu_mem_pool.allocate_blocks(num_cpu_actual_blocks)
        block_metas_to_get_s2h = block_metas_to_get_s2h[:num_cpu_actual_blocks]
        self.local_index.upload_to_cpu(block_metas_to_get_s2h, cpu_block_ids)

        ssd_descriptor = TransferDescriptor(
            blockmeta_list=block_metas_to_get_s2h,
            physical_block_ids=torch.tensor(
                [
                    block_meta.ssd_block_id
                    for block_meta in block_metas_to_get_s2h
                ],
                dtype=torch.int64,
                pin_memory=True,
            ),
            device=DeviceType.SSD,
        )

        actual_transfer_block_end -= (
            num_cpu_required_blocks - num_cpu_actual_blocks
        )

        # we assume that the prefixes are always cached
        block_metas_to_transfer = cached_block_metas[
            actual_transfer_block_start:actual_transfer_block_end
        ]
        actual_get_mask = torch.zeros_like(token_mask)
        actual_get_mask[
            actual_transfer_block_start
            * self.tokens_per_block : actual_transfer_block_end
            * self.tokens_per_block
        ] = True

        return_mask = token_mask & actual_get_mask

        gpu_physical_blocks = gpu_physical_block_ids[
            actual_transfer_block_start:actual_transfer_block_end
        ]

        assert len(gpu_physical_blocks) == len(block_metas_to_transfer)

        for block_meta in block_metas_to_transfer:
            block_meta.status = BlockStatus.LOCKED

        # ssd_descriptor = self._mempool_internal_get(block_metas_to_transfer)

        gpu_descriptor = TransferDescriptor(
            blockmeta_list=[],
            physical_block_ids=(
                gpu_physical_blocks.pin_memory().to(torch.int64)
            ),
            device=DeviceType.GPU,
        )
        cpu_descriptor = TransferDescriptor(
            blockmeta_list=block_metas_to_transfer,
            physical_block_ids=torch.tensor(
                [
                    block_meta.cpu_block_id
                    for block_meta in block_metas_to_transfer
                    # if block_meta.location == BlockLocation.CPU
                ],
                pin_memory=True,
                dtype=torch.int64,
            ),
            device=DeviceType.CPU,
        )
        return gpu_descriptor, cpu_descriptor, ssd_descriptor, return_mask

    def _put_impl(
        self,
        token_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        gpu_physical_block_ids: torch.Tensor,
    ) -> Tuple[TransferDescriptor, TransferDescriptor, torch.Tensor]:
        if token_mask is None:
            token_mask = torch.ones(len(token_ids), dtype=torch.bool)
        if not len(token_mask) == len(token_ids):
            raise ValueError(
                "token_mask should have the same length as token_ids"
            )

        if not token_mask[0]:
            raise ValueError("prefix should be put into kvcache")

        aligned_token_mask = self._check_and_align_token_mask(
            token_mask, prefix_mask_check=True
        )

        masked_token_ids = token_ids[: aligned_token_mask.sum()]

        cached_block_metas, num_required_blocks = self.local_index.match(
            masked_token_ids
        )

        cpu_cached_block_metas = []
        ssd_cached_block_metas = []
        for block_meta in cached_block_metas:
            if block_meta.location == BlockLocation.CPU:
                cpu_cached_block_metas.append(block_meta)
            elif block_meta.location == BlockLocation.SSD:
                ssd_cached_block_metas.append(block_meta)

        prev_block_meta = None
        if cached_block_metas:
            prev_block_meta = cached_block_metas[-1]
            prev_block_meta.status = BlockStatus.LOCKED  # prevent eviction
        if cpu_cached_block_metas:
            cpu_cached_block_metas[-1].status = BlockStatus.LOCKED

        num_cpu_required_blocks = num_required_blocks + len(
            ssd_cached_block_metas
        )
        self._evict_cpu_blocks(num_cpu_required_blocks)
        num_cpu_inserted_blocks = min(
            num_cpu_required_blocks, self.cpu_mem_pool.num_free_blocks
        )
        # cpu blocks num after insertion is still less than ssd block num?
        num_ssd_inserted_blocks = max(
            0, num_cpu_inserted_blocks - len(ssd_cached_block_metas)
        )
        self._evict_blocks(num_ssd_inserted_blocks)
        assert self.ssd_mem_pool.num_free_blocks >= num_ssd_inserted_blocks

        if prev_block_meta:
            prev_block_meta.status = BlockStatus.AVAILABLE
        if cpu_cached_block_metas:
            cpu_cached_block_metas[-1].status = BlockStatus.AVAILABLE

        cpu_block_ids = self.cpu_mem_pool.allocate_blocks(
            num_cpu_inserted_blocks
        )
        ssd_block_ids = self.ssd_mem_pool.allocate_blocks(
            num_ssd_inserted_blocks
        )

        num_inserted_blocks = num_ssd_inserted_blocks

        num_uploaded_blocks = min(
            num_cpu_inserted_blocks, len(ssd_cached_block_metas)
        )
        self.local_index.upload_to_cpu(
            ssd_cached_block_metas[:num_uploaded_blocks],
            cpu_block_ids[:num_uploaded_blocks],
        )

        block_metas_to_write_ssd = self.local_index.insert(
            masked_token_ids[
                len(cached_block_metas) * self.tokens_per_block : (
                    len(cached_block_metas) + num_inserted_blocks
                )
                * self.tokens_per_block
            ],
            cpu_block_ids[-num_inserted_blocks:],
            ssd_block_ids,
            prev_block_meta,
        )

        block_metas_to_transfer = (
            ssd_cached_block_metas + block_metas_to_write_ssd
        )

        for block in block_metas_to_transfer:
            block.status = BlockStatus.LOCKED

        gpu_physical_blocks = gpu_physical_block_ids[
            len(cpu_cached_block_metas) : len(cpu_cached_block_metas)
            + len(block_metas_to_transfer)
        ]

        gpu_descriptor = TransferDescriptor(
            blockmeta_list=[],
            physical_block_ids=(
                gpu_physical_blocks.pin_memory().to(torch.int64)
            ),
            device=DeviceType.GPU,
        )
        cpu_descriptor = TransferDescriptor(
            blockmeta_list=block_metas_to_transfer,
            physical_block_ids=torch.tensor(
                [
                    block_meta.cpu_block_id
                    for block_meta in block_metas_to_transfer
                ],
                pin_memory=True,
                dtype=torch.int64,
            ),
            device=DeviceType.CPU,
        )
        ssd_descriptor = TransferDescriptor(
            blockmeta_list=block_metas_to_write_ssd,
            physical_block_ids=torch.tensor(
                [
                    block_meta.ssd_block_id
                    for block_meta in block_metas_to_write_ssd
                ],
                dtype=torch.int64,
                pin_memory=True,
            ),
            device=DeviceType.SSD,
        )
        return_mask = torch.zeros_like(token_mask, dtype=torch.bool)
        put_block_num = len(cached_block_metas) + num_inserted_blocks
        return_mask[: (put_block_num) * self.tokens_per_block] = True
        return gpu_descriptor, cpu_descriptor, ssd_descriptor, return_mask

    def _check_and_align_token_mask(
        self, token_mask: torch.Tensor, prefix_mask_check: bool = False
    ) -> torch.Tensor:
        """Check and align the token mask.

        This method is used to check if the token mask contains consecutive True
        values and align it to the tokens_per_block.

        For example, if the token mask is FFFF-FFFT-TTTT-TF (token_per_block=4),
        it will be aligned to FFFF-TTTT-TTTT-TT.

        Args:
            token_mask: A boolean tensor of shape (num_tokens,).
            prefix_mask_check: Whether to check the mask has prefix True values.
        Returns:
            aligned token mask.
        """

        if token_mask.ndim != 1:
            raise ValueError("token_mask should be a 1D tensor")
        if token_mask.dtype != torch.bool:
            raise ValueError("token_mask should be a boolean tensor")

        mask_diff = torch.diff(token_mask)
        if mask_diff.sum() > 1 or (  # T..T or T..TF..F or F..FT..T
            mask_diff.sum() == 2
            and not token_mask[0]
            and not token_mask[-1]  # or F..FT..TF..F
        ):
            raise ValueError(
                "token_mask should contain consecutive True values"
            )

        if prefix_mask_check and (
            mask_diff.sum() > 1
            or (  # T..TF..F
                mask_diff.sum() == 2
                and not token_mask[0]
                and not token_mask[-1]  # or F..FT..TF..F
            )
        ):
            raise ValueError(
                "token_mask should have prefix True values such as 'TTTTFFFF'"
            )

        aligned_token_mask = torch.zeros_like(token_mask)

        num_chunks = (len(token_mask) - 1) // self.tokens_per_block + 1
        for i in range(num_chunks):
            aligned_token_mask[
                i * self.tokens_per_block : (i + 1) * self.tokens_per_block
            ] = token_mask[
                i * self.tokens_per_block : (i + 1) * self.tokens_per_block
            ].any()

        return aligned_token_mask

    def _get_block_range(
        self, aligned_token_mask: torch.Tensor
    ) -> Tuple[int, int]:
        block_ids = torch.nonzero(
            aligned_token_mask[:: self.tokens_per_block]
        ).flatten()
        if len(block_ids) == 0:
            return 0, 0
        return block_ids[0], block_ids[-1] + 1

    # just for test
    def get_physical_blocks(self) -> List[torch.Tensor]:
        return self.cpu_mem_pool.get_physical_blocks()

    def shutdown(self):
        self.gpu_transfer.shutdown()
        self.request_handler.shutdown()
