from typing import List, Tuple, Optional, Dict
from queue import Queue
import torch
import time
import threading

from flexkv.common.transfer import DeviceType, TransferOpGraph
from flexkv.cache.index import TokenToBlockIndex
from flexkv.cache.mempool import Mempool
from flexkv.common.block import BlockMeta, SequenceMeta, BlockStatus
from flexkv.common.config import CacheConfig
from flexkv.cache.transfer_pattern import create_read_transfer_graph, create_write_transfer_graph
from flexkv.common.request import cacheEngineRequestType, cacheEngineRequest

class CacheEngine:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int):
        self.device_type = device_type

        self.index = TokenToBlockIndex(tokens_per_block=tokens_per_block)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

    def reset(self):
        self.index.reset()
        self.mempool.reset()

    def match(self, sequence_meta: SequenceMeta, maximum_status: int = BlockStatus.LOCKED.value) -> torch.Tensor:
        physical_block_ids = self.index.match_prefix(sequence_meta,
                                              update_cache_info=True,
                                              maximum_status=maximum_status)
        return physical_block_ids

    def match_length(self, sequence_meta: SequenceMeta, maximum_status: int = BlockStatus.LOCKED.value) -> int:
        return self.index.match_length(sequence_meta, maximum_status)

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: torch.Tensor,
               match_length: int = -1) -> None:
        if match_length == -1:
            # in insert, we can use the default maximum_status for match
            match_length = self.index.match_length(sequence_meta=sequence_meta)
        self.index.insert(sequence_meta, match_length, physical_block_ids)

    def take(self, num_required_blocks: int, locked_blocks: torch.Tensor = None, as_buffer: bool = False) -> List[int]:
        if num_required_blocks > self.mempool.num_free_blocks:
            self.mempool.recycle_blocks(
                self.index.evict(
                    num_evicted=num_required_blocks - self.mempool.num_free_blocks,
                    locked_blocks=locked_blocks
                )
            )
        if num_required_blocks > self.mempool.num_free_blocks:
            raise ValueError("Not enough free blocks to take")
        blocks_ids = self.mempool.allocate_blocks(num_required_blocks)
        if not as_buffer:
            self.index.set_blocks_as_in_put(blocks_ids)
        return blocks_ids

    def cleanup(self, block_ids: torch.Tensor) -> None:
        # TODO: for now we ignore the buffer blocks because we don't have them now
        # actually we should return those blocks to the mempool
        """
        selected_status = self.index._status[block_ids]
        buffer_positions = torch.nonzero(selected_status == BlockStatus.UNREGISTERED.value).squeeze(-1)
        block_id_positions = torch.nonzero(selected_status > BlockStatus.UNREGISTERED.value).squeeze(-1)
        buffer_block_ids = torch.index_select(block_ids, 0, buffer_positions)
        self.mempool.recycle_blocks(buffer_block_ids)
        block_ids = torch.index_select(block_ids, 0, block_id_positions)
        """
        self.index._lock_cnt[block_ids] -= 1
        selected_lock_cnt = self.index._lock_cnt[block_ids]
        to_be_freed_blocks = torch.nonzero(selected_lock_cnt == 0).squeeze(-1)
        to_be_freed_blocks = torch.index_select(block_ids, 0, to_be_freed_blocks)
        self.index._status[to_be_freed_blocks] = BlockStatus.AVAILABLE.value

class GlobalCacheEngine:
    def __init__(self, cache_config: CacheConfig):
        self.cache_config = cache_config
        self.tokens_per_block = cache_config.tokens_per_block

        self.cpu_cache_engine = None
        self.ssd_cache_engine = None
        self.remote_cache_engine = None

        # self.task_queue = Queue()
        # self.finished_queue = Queue()
        # self.running = True

        if not cache_config.enable_cpu and not cache_config.enable_ssd:
            raise ValueError("Either enable_cpu or enable_ssd must be True")

        if not cache_config.enable_cpu and not cache_config.use_gds:
            raise ValueError("use_gds must be True if enable_cpu is False")

        if cache_config.enable_cpu:
            self.cpu_cache_engine = CacheEngine(DeviceType.CPU,
                                                cache_config.num_cpu_blocks,
                                                cache_config.tokens_per_block)
        if cache_config.enable_ssd:
            self.ssd_cache_engine = CacheEngine(DeviceType.SSD,
                                                cache_config.num_ssd_blocks,
                                                cache_config.tokens_per_block)
        if cache_config.enable_remote:
            raise NotImplementedError("Remote cache is not implemented")

        # self._worker_thread = threading.Thread(target=self._worker_loop)
        # self._worker_thread.start()
    def cleanup_engines(self, block_ids_to_unlock: Dict[DeviceType, torch.Tensor]):
        if DeviceType.CPU in block_ids_to_unlock:
            self.cpu_cache_engine.cleanup(block_ids_to_unlock[DeviceType.CPU])
        if DeviceType.SSD in block_ids_to_unlock:
            self.ssd_cache_engine.cleanup(block_ids_to_unlock[DeviceType.SSD])

    """
    def _worker_loop(self):
        while self.running:
            request = self.task_queue.get()
            if request is None:
                break
            elif request.request_type == cacheEngineRequestType.GET:
                graph, return_mask = self.get(request.token_ids, request.token_mask, request.slot_mapping)
                self.finished_queue.put((request.request_id, graph, return_mask))
            elif request.request_type == cacheEngineRequestType.PUT:
                graph, return_mask = self.put(request.token_ids, request.token_mask, request.slot_mapping)
                self.finished_queue.put((request.request_id, graph, return_mask))
            elif request.request_type == cacheEngineRequestType.CLEANUP:
                if DeviceType.CPU in request.block_ids_to_unlock:
                    self.cpu_cache_engine.cleanup(request.block_ids_to_unlock[DeviceType.CPU])
                if DeviceType.SSD in request.block_ids_to_unlock:
                    self.ssd_cache_engine.cleanup(request.block_ids_to_unlock[DeviceType.SSD])
            time.sleep(0.001)

    def submit_request(self, request: cacheEngineRequest):
        self.task_queue.put(request)

    def shutdown(self):
        self.running = False
        self.task_queue.put(None)
        self._worker_thread.join()

    def get_finished_requests(self, timeout: Optional[float] = None) -> List[Tuple[int, TransferOpGraph, torch.Tensor]]:
        finished_requests = []

        if self.finished_queue.empty():
            return finished_requests

        try:
            request_id, graph, mask = self.finished_queue.get(timeout=timeout)
            finished_requests.append((request_id, graph, mask))

            while not self.finished_queue.empty():
                request_id, graph, mask = self.finished_queue.get_nowait()
                finished_requests.append((request_id, graph, mask))
        except Queue.Empty:
            pass

        return finished_requests
    """

    def reset(self):
        if self.cpu_cache_engine:
            self.cpu_cache_engine.reset()
        if self.ssd_cache_engine:
            self.ssd_cache_engine.reset()
        if self.remote_cache_engine:
            self.remote_cache_engine.reset()

    def get(self,
            token_ids: torch.Tensor,
            token_mask: torch.Tensor,
            slot_mapping: torch.Tensor) -> Tuple[TransferOpGraph, torch.Tensor]:
        self._check_input(token_ids, token_mask, slot_mapping)

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        start_idx, end_idx = self._get_block_range(token_mask)
        #NOTE: the slot_mapping starts from token that need to be fetched
        gpu_block_mapping = self._slot_to_block_mapping(slot_mapping)[:end_idx-start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_matched_blocks, ssd_matched_blocks, _ = self.match(sequence_meta,
                                                               start_idx,
                                                               end_idx,
                                                               maximum_status=BlockStatus.IN_GET.value)

        num_transfer_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
        assert num_transfer_blocks <= len(gpu_block_mapping)

        gpu_blocks_to_transfer = gpu_block_mapping[:num_transfer_blocks]
        cpu_blocks_to_transfer = cpu_matched_blocks
        # WARNING: here we lock all the concerned blocks in the global cache engine
        # this may be not a good idea, beacause we should perhaps do this inside the index
        # during the match process to make this more modular and decoupled
        self.cpu_cache_engine.index.set_blocks_as_in_get(cpu_blocks_to_transfer)
        ssd_blocks_to_transfer = ssd_matched_blocks[len(cpu_matched_blocks):]
        self.ssd_cache_engine.index.set_blocks_as_in_get(ssd_blocks_to_transfer)
        # TODO: lock the status of the taken blocks
        if len(cpu_matched_blocks) < len(ssd_matched_blocks):
            extra_cpu_blocks = self.cpu_cache_engine.take(
                num_required_blocks=len(ssd_matched_blocks) - len(cpu_matched_blocks),
                locked_blocks=cpu_matched_blocks[-1:] if len(cpu_matched_blocks) > 0 else None,
                as_buffer=False
            ) # here we still put the buffer in the cpu index, so no buffer is needed
            cpu_blocks_to_transfer = torch.cat([cpu_blocks_to_transfer, extra_cpu_blocks])
            # here we still put the temporal buffer in the cpu cache engine for later access
            assert sequence_meta.num_blocks == len(cpu_matched_blocks) + start_idx + len(extra_cpu_blocks)
            self.cpu_cache_engine.insert(sequence_meta, extra_cpu_blocks, len(cpu_matched_blocks) + start_idx)
        # NOTE: for now in build transfer graph, we assume that cpu works as a cache for ssd
        transfer_graph = create_read_transfer_graph(ssd_blocks_to_transfer,
                                                    cpu_blocks_to_transfer,
                                                    gpu_blocks_to_transfer)

        return_mask = torch.zeros_like(token_mask)
        return_mask[start_idx* self.tokens_per_block:
                    (start_idx + len(gpu_blocks_to_transfer)) * self.tokens_per_block] = True

        return transfer_graph, return_mask

    def put(self,
            token_ids: torch.Tensor,
            token_mask: torch.Tensor,
            slot_mapping: torch.Tensor) -> Tuple[TransferOpGraph, torch.Tensor]:
        self._check_input(token_ids, token_mask, slot_mapping)

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        # WARNING the start_idx and end_idx is token level, not block level
        start_idx, end_idx = self._get_block_range(token_mask)

        # the mask should has a prefix of True
        assert start_idx == 0

        gpu_block_mapping = self._slot_to_block_mapping(slot_mapping)[:end_idx-start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_matched_blocks, ssd_matched_blocks, _ = self.match(sequence_meta,
                                                               start_idx,
                                                               end_idx,
                                                               maximum_status=BlockStatus.IN_PUT.value)

        # cpu works as a cache for ssd
        assert len(cpu_matched_blocks) <= len(ssd_matched_blocks)

        gpu_blocks_to_transfer = gpu_block_mapping[len(cpu_matched_blocks):]
        # lock the last block of cpu and ssd to avoid eviction in this turn
        cpu_blocks_to_transfer = self.cpu_cache_engine.take(
            num_required_blocks=len(gpu_block_mapping) - len(cpu_matched_blocks),
            locked_blocks=cpu_matched_blocks[-1:] if len(cpu_matched_blocks) > 0 else None
        )
        ssd_blocks_to_transfer = self.ssd_cache_engine.take(
            num_required_blocks=len(gpu_block_mapping) - len(ssd_matched_blocks),
            locked_blocks=ssd_matched_blocks[-1:] if len(ssd_matched_blocks) > 0 else None
        )
        assert len(cpu_blocks_to_transfer) + start_idx + len(cpu_matched_blocks) == sequence_meta.num_blocks
        assert len(ssd_blocks_to_transfer) + start_idx + len(ssd_matched_blocks) == sequence_meta.num_blocks

        self.cpu_cache_engine.insert(sequence_meta, cpu_blocks_to_transfer, len(cpu_matched_blocks) + start_idx)
        self.ssd_cache_engine.insert(sequence_meta, ssd_blocks_to_transfer, len(ssd_matched_blocks) + start_idx)
        # lock the blocks to avoid eviction during the transfer
        # already done in take() function
        # self.cpu_cache_engine.index.set_blocks_as_in_put(cpu_blocks_to_transfer)
        # self.ssd_cache_engine.index.set_blocks_as_in_put(ssd_blocks_to_transfer)

        transfer_graph = create_write_transfer_graph(ssd_blocks_to_transfer,
                                                    cpu_blocks_to_transfer,
                                                    gpu_blocks_to_transfer)

        return_mask = torch.zeros_like(token_mask)
        return_mask[start_idx* self.tokens_per_block:
                    (start_idx + len(gpu_block_mapping)) * self.tokens_per_block] = True

        return transfer_graph, return_mask

    def match(self, sequence_meta: SequenceMeta,
              start_idx: int,
              end_idx: int,
              maximum_status: int = BlockStatus.LOCKED.value) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cpu_match_results = []
        ssd_match_results = []
        remote_match_results = []
        # TODO: avoid redundant match?
        if self.cpu_cache_engine:
            cpu_match_results = self.cpu_cache_engine.match(sequence_meta, maximum_status)[start_idx:end_idx]
        if self.ssd_cache_engine:
            ssd_match_results = self.ssd_cache_engine.match(sequence_meta, maximum_status)[start_idx:end_idx]
        if self.remote_cache_engine:
            remote_match_results = self.remote_cache_engine.match(sequence_meta, maximum_status)[start_idx:end_idx]

        return cpu_match_results, ssd_match_results, remote_match_results

    def _check_input(self,
                      token_ids: torch.Tensor,
                      token_mask: torch.Tensor,
                      slot_mapping: torch.Tensor) -> None:
        assert token_ids.ndim == 1
        assert token_mask.ndim == 1
        assert slot_mapping.ndim == 1
        assert len(token_ids) == len(token_mask)
        assert len(slot_mapping) == token_mask.sum().item()

    def _slot_to_block_mapping(self,
                              slot_mapping: torch.Tensor) -> torch.Tensor:
        block_mapping = slot_mapping // self.tokens_per_block
        block_mapping = torch.unique(block_mapping)
        return block_mapping

    def _get_block_range(self,
                         token_mask: torch.Tensor) -> Tuple[int, int]:
        mask_idx = torch.where(token_mask)[0]
        start_idx = mask_idx[0].item() // self.tokens_per_block
        end_idx = mask_idx[-1].item() // self.tokens_per_block
        return start_idx, end_idx + 1
