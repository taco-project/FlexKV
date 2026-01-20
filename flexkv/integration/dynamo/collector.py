from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple
from queue import Queue
import threading
import numpy as np

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent

class KVEventCollector:
    def __init__(self) -> None:
        self.events: List['KVCacheEvent'] = []
        self._event_lock = threading.Lock()
        self._work_queue: Queue[Optional[Tuple[
            str,
            np.ndarray,
            int,
            Optional[int],
            Optional[str]
        ]]] = Queue()
        self._worker = threading.Thread(target=self._run_worker,
                                        name='kv-event-collector',
                                        daemon=True)
        self._worker.start()

    def _run_worker(self) -> None:
        while True:
            item = self._work_queue.get()
            if item is None: # Deliberately enqueued to signal shutdown
                self._work_queue.task_done()
                break
            event_type, block_hashes, block_size, lora_id, medium = item
            event: 'KVCacheEvent' = self._create_event(
                event_type=event_type,
                block_hashes=block_hashes,
                block_size=block_size,
                lora_id=lora_id,
                medium=medium,
            )
            with self._event_lock:
                self.events.append(event)
            self._work_queue.task_done()

    def close(self) -> None:
        self._work_queue.put(None)
        self._worker.join()

    def _create_event(
        self,
        event_type: str,
        block_hashes: np.ndarray,
        block_size: int,
        lora_id: Optional[int] = None,
        medium: Optional[str] = None,
    ) -> 'KVCacheEvent':
        from vllm.distributed.kv_events import BlockRemoved, BlockStored

        if event_type == 'BlockStored':
            return BlockStored(
                block_hashes=block_hashes.tolist(),
                parent_block_hash=None,
                token_ids=[],
                block_size=block_size,
                lora_id=lora_id,
                medium=medium,
            )
        elif event_type == 'BlockRemoved':
            return BlockRemoved(
                block_hashes=block_hashes.tolist(),
                medium=medium,
            )

        raise ValueError(f'Unknown event type: {event_type}')

    def publish_stored(
        self,
        block_hashes: np.ndarray,
        block_size: int = 16,
        lora_id: Optional[int] = None,
        medium: Optional[str] = 'CPU',
    ) -> None:
        self._work_queue.put((
            'BlockStored',
            block_hashes,
            block_size,
            lora_id,
            medium,
        ))

    def publish_removed(
        self,
        block_hashes: np.ndarray,
        medium: Optional[str] = 'CPU',
    ) -> None:
        self._work_queue.put((
            'BlockRemoved',
            block_hashes,
            0,    # Unused
            None, # Unused
            medium,
        ))

    def take_events(self) -> Iterable['KVCacheEvent']:
        with self._event_lock:
            events = self.events
            self.events = []
        return events
