import hashlib
import time
import torch
from typing import Dict, List, Optional, Tuple

from flexkv.core.block import BlockMeta, BlockStatus, BlockLocation


class TokenToBlockIndex:
    def __init__(self, tokens_per_block: int, invalid_watermark: int = 100):
        self.token2block: Dict[str, BlockMeta] = {}
        self.tokens_per_block = tokens_per_block
        self.num_invalid = 0
        self.invalid_watermark = invalid_watermark
        self.quick_prune : bool = True
        self.quick_prune_threshold : int = tokens_per_block * 20
        self.quick_prune_block_num : int = 2

    def reset(self)->None:
        self.token2block.clear()
        self.num_invalid = 0

    def match(self, token_ids: torch.Tensor) -> Tuple[List[BlockMeta], int]:
        if self.quick_prune and len(token_ids) > self.quick_prune_threshold:
            keys = self._tokens_to_block_hashes(
                token_ids[: self.tokens_per_block * self.quick_prune_block_num]
            )
            for key in keys:
                if key not in self.token2block or (
                    self.token2block[key].status == BlockStatus.INVALID
                ):
                    return [], len(token_ids) // self.tokens_per_block
        block_metas = []
        last_cached_block_index, last_cached_block = (
            self._get_last_cached_block(token_ids)
        )
        if last_cached_block:
            block_metas.append(last_cached_block)
            while last_cached_block.prev_block:
                assert (
                    last_cached_block.prev_block.status != BlockStatus.INVALID
                ), "Invalid non-leaf block"
                block_metas.append(last_cached_block.prev_block)
                last_cached_block = last_cached_block.prev_block
        assert len(block_metas) == last_cached_block_index + 1
        num_cached_blocks = last_cached_block_index + 1
        num_uncached_blocks = (
            len(token_ids) // self.tokens_per_block - num_cached_blocks
        )

        current_time = time.time()
        for block_meta in block_metas:
            block_meta.last_access_time = current_time
        return block_metas[::-1], num_uncached_blocks

    def insert(
        self,
        inserted_token_ids: torch.Tensor,
        cpu_block_ids: torch.Tensor,
        ssd_block_ids: torch.Tensor,
        prev_block_meta: Optional[BlockMeta] = None,
    ) -> List[BlockMeta]:
        keys = self._tokens_to_block_hashes(
            inserted_token_ids,
            prev_block_meta.block_hash if prev_block_meta else "",
        )
        current_time = time.time()
        assert len(cpu_block_ids) == len(keys)
        assert len(ssd_block_ids) == len(keys)

        inserted_block_metas = []
        for key, cpu_block_id, ssd_block_id in zip(
            keys, cpu_block_ids, ssd_block_ids
        ):
            assert (
                key not in self.token2block
                or self.token2block[key].status == BlockStatus.INVALID
            ), f"Key {key} already exists in the index"
            new_block_meta = BlockMeta(
                block_hash=key,
                last_access_time=current_time,
                cpu_block_id=cpu_block_id,
                ssd_block_id=ssd_block_id,
                prev_block=prev_block_meta,
                reference_count=0,
                status=BlockStatus.AVAILABLE,
                location=BlockLocation.CPU,  # inserted blocks are always on CPU
            )
            self.token2block[key] = new_block_meta
            inserted_block_metas.append(new_block_meta)
            if prev_block_meta:
                prev_block_meta.reference_count += 1
            prev_block_meta = new_block_meta
        return inserted_block_metas

    def remove(self, block_metas: List[BlockMeta]):
        # remove blocks from the index, which means that
        # the blocks are not on the CPU or SSD
        for block_meta in block_metas:
            assert block_meta.reference_count == 0, (
                "Only leaf blocks can be removed"
            )
            assert block_meta.status == BlockStatus.AVAILABLE, (
                "Only IN_USE blocks can be removed"
            )
            block_meta.status = BlockStatus.INVALID
            self.num_invalid += 1
            if block_meta.prev_block:
                block_meta.prev_block.reference_count -= 1
        if self.num_invalid >= self.invalid_watermark:
            self._delete_invalid_blocks()

    def offload_to_ssd(self, block_metas: List[BlockMeta]):
        for block_meta in block_metas:
            assert block_meta.location == BlockLocation.CPU, (
                "Only CPU blocks can be offloaded to SSD"
            )
            assert block_meta.status != BlockStatus.INVALID, "Invalid block"
            block_meta.location = BlockLocation.SSD
            block_meta.cpu_block_id = -1

    def upload_to_cpu(
        self, block_metas: List[BlockMeta], cpu_block_ids: List[int]
    ):
        for block_meta, cpu_block_id in zip(block_metas, cpu_block_ids):
            assert block_meta.location == BlockLocation.SSD, (
                "Only SSD blocks can be uploaded to CPU"
            )
            assert block_meta.status != BlockStatus.INVALID, "Invalid block"
            block_meta.location = BlockLocation.CPU
            block_meta.cpu_block_id = cpu_block_id

    def _delete_invalid_blocks(self):
        for key in list(self.token2block.keys()):
            if self.token2block[key].status == BlockStatus.INVALID:
                del self.token2block[key]
                self.num_invalid -= 1
        assert self.num_invalid == 0

    def _tokens_to_block_hashes(
        self, token_ids: torch.Tensor, prefix_hash: str = ""
    ) -> List[str]:
        block_hashes = []
        # we give up the last block if it is not full
        for i in range(len(token_ids) // self.tokens_per_block):
            block_hash = hashlib.md5(
                (
                    str(prefix_hash)
                    + str(
                        token_ids[
                            i * self.tokens_per_block : (i + 1)
                            * self.tokens_per_block
                        ]
                    )
                ).encode()
            ).hexdigest()
            block_hashes.append(str(block_hash))
            prefix_hash = block_hash
        return block_hashes

    def _get_last_cached_block(
        self, token_ids: torch.Tensor
    ) -> Tuple[int, Optional[BlockMeta]]:
        keys = self._tokens_to_block_hashes(token_ids)
        left, right = 0, len(keys) - 1
        last_cached_block_index = -1
        while left <= right:
            mid = (left + right) // 2
            if (
                keys[mid] in self.token2block
                and self.token2block[keys[mid]].status != BlockStatus.INVALID
            ):
                last_cached_block_index = mid
                left = mid + 1
            else:
                right = mid - 1
        return last_cached_block_index, self.token2block[
            keys[last_cached_block_index]
        ] if last_cached_block_index >= 0 else None

    def get_evictable_leaf_blocks(self):
        return [
            block
            for block in self.token2block.values()
            if block.reference_count == 0
            and block.status == BlockStatus.AVAILABLE
        ]

    def get_cpu_leaf_blocks_to_offload(self):
        cpu_leaf_blocks = []
        for block in self.token2block.values():
            if (
                block.status != BlockStatus.INVALID
                and block.location == BlockLocation.SSD
                and block.prev_block is not None
                and block.prev_block.status == BlockStatus.AVAILABLE
                and block.prev_block.location == BlockLocation.CPU
            ):
                cpu_leaf_blocks.append(block.prev_block)

            if (
                block.status == BlockStatus.AVAILABLE
                and block.location == BlockLocation.CPU
                and block.reference_count == 0
            ):
                cpu_leaf_blocks.append(block)

        return cpu_leaf_blocks
