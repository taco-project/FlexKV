from queue import Queue, Empty
from threading import Thread, Condition
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Callable, Generic, TypeVar, Deque, Dict
from collections import deque
from flexkv.core.debug_utils import debuginfo
from flexkv.core.transfer import GPUTransfer, TransferType, SSDTransfer

# from flexkv.core.block import BlockStatus
import time
import torch

T = TypeVar("T")


class RequestType(Enum):
    GET = auto()
    PUT = auto()


@dataclass
class Request:
    request_id: int
    type: RequestType
    token_ids: torch.Tensor
    token_mask: Optional[torch.Tensor]
    gpu_physical_block_ids: torch.Tensor


@dataclass
class Response:
    request_id: int
    type: RequestType
    return_mask: Optional[torch.Tensor]


class AsyncRequestHandler(Generic[T]):
    def __init__(
        self,
        get_func: Callable[[torch.Tensor, Any], T],
        put_func: Callable[[torch.Tensor, Any], T],
        gpu_transfer: "GPUTransfer",
        ssd_transfer: "SSDTransfer",
        # maximum number of finished requests
        max_finished: int = 1000,
        # maximum number of requests to process in one batch
        max_batch_size: int = 10,
    ):
        self.running = True
        self.get_func = get_func
        self.put_func = put_func
        self.gpu_transfer = gpu_transfer  # ok here?
        self.ssd_transfer = ssd_transfer
        self.request_queue: Queue = Queue()
        self.finished_queue: Deque = deque(maxlen=max_finished)
        self.next_request_id: int = 0
        self.max_batch_size = max_batch_size
        # we assume that one request only has one transfer request
        self.pending_requests: Dict[int, int] = {}  # request_id -> transfer_ids
        self.transfer_to_request: Dict[
            int, int
        ] = {}  # transfer_id -> request_id

        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.check_transfer_completion_thread = Thread(
            target=self._check_completions, daemon=True
        )
        self.check_transfer_completion_thread.start()
        self.cond = Condition()

    def _get_next_request_id(self) -> int:
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id

    def _get_current_request_id(self) -> int:
        return self.next_request_id

    def submit_request(
        self,
        type: RequestType,
        token_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        gpu_physical_block_ids: torch.Tensor,
    ) -> int:
        request_id = self._get_next_request_id()
        request = Request(
            request_id=request_id,
            type=type,
            token_ids=token_ids,
            token_mask=token_mask,
            gpu_physical_block_ids=gpu_physical_block_ids,
        )
        self.request_queue.put(request)
        debuginfo.info(f"REQUEST {request_id} - submitted")
        return request_id

    def get_results(self, request_ids: List[int]) -> List[Response]:
        completed_responses = []
        request_id_set = set(request_ids)

        for response in self.finished_queue:
            if response.request_id in request_id_set:
                completed_responses.append(response)
                request_id_set.remove(response.request_id)

                if not request_id_set:
                    break

        return completed_responses

    def _worker(self):
        while True:
            requests = []
            write_request = None

            request = self.request_queue.get()
            if request is None:
                break

            if request.type == RequestType.PUT:
                write_request = request
            else:
                requests.append(request)

            if not write_request:
                while len(requests) < self.max_batch_size:
                    try:
                        request = self.request_queue.get_nowait()
                        if request is None:
                            break
                        if request.type == RequestType.GET:
                            requests.append(request)
                        else:
                            self.request_queue.put(request)
                            break
                    except Empty:
                        break

            batch = write_request if write_request else requests
            if write_request:
                debuginfo.info(f"WRITE batch {batch.request_id} - collected")
            else:
                request_ids = ", ".join(
                    [str(request.request_id) for request in batch]
                )
                debuginfo.info(f"READ batch {request_ids} - collected")
            self._process_batch(batch)
            time.sleep(0.0001)

    def _process_batch(self, batch):
        if not isinstance(batch, list):
            batch = [batch]

        # TODO we need a parallel version later
        # to process multiple requests at the same time
        for request in batch:
            # get op
            if request.type == RequestType.GET:
                # all concerned status should already be locked
                (gpu_desc, cpu_desc, ssd_desc, return_mask) = self.get_func(
                    request.token_ids,
                    request.token_mask,
                    request.gpu_physical_block_ids,
                )
                # empty result
                if (
                    len(cpu_desc.physical_block_ids) == 0
                    and len(ssd_desc.physical_block_ids) == 0
                ):
                    # or (
                    #     # note blockmetas should always on cpu descriptor
                    #     cpu_desc.blockmeta_list[0].status == \
                    #                                   BlockStatus.LOCKED
                    # ):
                    response = Response(
                        request_id=request.request_id,
                        type=request.type,
                        return_mask=torch.zeros_like(
                            request.token_ids, dtype=torch.bool
                        ),
                    )
                    debuginfo.info(
                        f"REQUEST {request.request_id} - "
                        f"get op returns empty results. finished"
                    )
                    self.finished_queue.append(response)
                    continue
                # only cpu transfer
                elif len(ssd_desc.physical_block_ids) == 0:
                    transfer_id = self.gpu_transfer.submit_transfer(
                        request.request_id,
                        TransferType.H2D,
                        cpu_desc,
                        gpu_desc,
                        additional_descriptor=None,
                        return_mask=return_mask,
                    )
                # ssd transfer and cpu transfer
                else:
                    # note:
                    # h2d transfer will be called after disk2h finishes
                    transfer_id = self.ssd_transfer.submit_transfer(
                        request.request_id,
                        TransferType.DISK2H,
                        ssd_desc,
                        cpu_desc,
                        gpu_desc,
                        return_mask,
                    )
                debuginfo.debug(
                    f"REQUEST {request.request_id} - "
                    f"block to transfer: {cpu_desc.physical_block_ids}"
                )
            else:
                (gpu_desc, cpu_desc, ssd_desc, return_mask) = self.put_func(
                    request.token_ids,
                    request.token_mask,
                    request.gpu_physical_block_ids,
                )
                # the kvcaches are already in kvpool
                if len(cpu_desc.physical_block_ids) == 0:
                    response = Response(
                        request_id=request.request_id,
                        type=request.type,
                        return_mask=torch.ones_like(
                            request.token_ids, dtype=torch.bool
                        ),
                    )
                    self.finished_queue.append(response)
                    continue
                # actual transfer is happening
                # after this d2h, h2Disk will be called
                transfer_id = self.gpu_transfer.submit_transfer(
                    request.request_id,
                    TransferType.D2H,
                    gpu_desc,
                    cpu_desc,
                    ssd_desc,
                    return_mask,
                )
                debuginfo.debug(
                    f"REQUEST {request.request_id} - "
                    f"block to transfer: {cpu_desc.physical_block_ids}"
                )
            # we maintain two mappings of [transfer_id] <-> [request_id]
            # self.transfer_to_request[transfer_id] = request.request_id
            self.pending_requests[request.request_id] = transfer_id

    def _check_completions(self):
        while self.running:
            # try:
            completed_transfers = self.gpu_transfer.pop_completed_transfers()

            for transfer in completed_transfers:
                # request_id = self.transfer_to_request[transfer.transfer_id]
                request_id = transfer.request_id
                response = Response(
                    request_id=request_id,
                    type=transfer.type,
                    return_mask=transfer.return_mask,
                )
                print(f"REQUEST {request_id} - {transfer.type} op transfer ")
                print(f"transfer id {transfer.transfer_id}")
                debuginfo.info(
                    f"REQUEST {request_id} - "
                    f"{transfer.type} op transfer completed with "
                    f"processed tokens: {transfer.return_mask.sum()}"
                )
                self.finished_queue.append(response)
                with self.cond:
                    self.cond.notify_all()

                # del self.transfer_to_request[transfer.transfer_id]
                del self.pending_requests[request_id]

            if not completed_transfers:
                time.sleep(0.0001)

            # except Exception as e:
            #     debuginfo.error(f"Error in check_completions: {e}")
            #     if not self.running:
            #         break

    def wait_until_finished(self, request_ids: List[int]):
        if not request_ids:
            return

        pending_ids = set(request_ids)
        current_request_id = self._get_current_request_id()
        assert current_request_id >= max(request_ids)

        with self.cond:
            while True:
                completed = True
                for response in self.finished_queue:
                    if response.request_id in pending_ids:
                        pending_ids.remove(response.request_id)
                if len(pending_ids) == 0:
                    break

                completed = False
                if not completed:
                    self.cond.wait(timeout=0.0001)

    def shutdown(self):
        self.running = False
        self.request_queue.put(None)
        self.worker_thread.join(timeout=5)
        self.check_transfer_completion_thread.join(timeout=5)

        if self.worker_thread.is_alive():
            debuginfo.warning("Worker thread didn't terminate properly")
        if self.check_transfer_completion_thread.is_alive():
            debuginfo.warning("Check complete thread didn't terminate properly")
