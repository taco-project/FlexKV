import os
import time
from typing import TYPE_CHECKING, Optional, Literal, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch

from flexkv.kvmanager import KVManager
from flexkv.server.client import KVTPClient
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.request import KVResponseStatus
from flexkv.common.debug import flexkv_logger
from flexkv.integration.stats import FlexKVStats
from flexkv.integration.utils import cdiv
from flexkv.integration.config import FlexKVConfig

# vllm
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata, KVConnectorRole)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
    from vllm.v1.outputs import KVConnectorOutput


logger = flexkv_logger


@dataclass
class FlexKVResponse:
    task_id: int
    task_type: Literal["get", "put"]
    request: "Request"
    success: bool


@dataclass
class FlexKVTask(ABC):
    task_id: int = 0
    request: "Request" = 0

    # slot mapping
    slot_mapping: Optional[np.ndarray] = None

    # timer
    match_start_time: float = 0
    match_end_time: float = 0
    task_launch_time: float = 0
    task_finished_time: float = 0

    @property
    def match_cost(self) -> float:
        return (self.match_end_time - self.match_start_time)

    @property
    def task_execute_cost(self) -> float:
        return (self.task_finished_time - self.task_launch_time)

    @property
    @abstractmethod
    def task_type(self) -> str:
        ...

    def __str__(self):
        return (f"FlexKVTask(task_id={self.task_id}, "
                f"request={self.request.request_id}, "
                f"match_cost {self.match_cost*1000:.2f} ms, "
                f"task execute cost {self.task_execute_cost*1000:.2f} ms)")


@dataclass(kw_only=True)
class FlexKVGetTask(FlexKVTask):
    num_computed_tokens: int
    num_new_matched_tokens: int

    @property
    def task_type(self) -> str:
        return "get"

    def __str__(self):
        return (f"FlexKVGetTask(task_id={self.task_id}, "
                f"request={self.request.request_id}, "
                f"num_computed_tokens={self.num_computed_tokens}, "
                f"num_new_matched_tokens={self.num_new_matched_tokens}, "
                f"match_cost {self.match_cost*1000:.2f} ms, "
                f"task execute cost {self.task_execute_cost*1000:.2f} ms)")


@dataclass(kw_only=True)
class FlexKVPutTask(FlexKVTask):
    num_matched_tokens: int
    num_unmatched_tokens: int

    @property
    def task_type(self) -> str:
        return "put"

    def __str__(self):
        return (f"FlexKVPutTask(task_id={self.task_id}, "
                f"request={self.request.request_id}, "
                f"num_matched_tokens={self.num_matched_tokens}, "
                f"num_unmatched_tokens={self.num_unmatched_tokens}, "
                f"match_cost {self.match_cost*1000:.2f} ms, "
                f"task execute cost {self.task_execute_cost*1000:.2f} ms)")


class FlexKVSchedulerConnector:
    def __init__(
        self,
        flexkv_config: FlexKVConfig
    ):
        logger.info(f"Start init FlexKVSchedulerConnector with {flexkv_config}")
        self.flexkv_config = flexkv_config
        self.server_recv_port = flexkv_config.server_recv_port
        self.tp_size = flexkv_config.tp_size
        self.block_size = flexkv_config.block_size
        self.model_config = ModelConfig(
            num_layers=flexkv_config.num_layers,
            num_kv_heads=flexkv_config.num_kv_heads,
            head_size=flexkv_config.head_size,
            use_mla=flexkv_config.use_mla,
            dtype=flexkv_config.dtype,
            tp_size=flexkv_config.tp_size,
        )
        if "tokens_per_block" in flexkv_config.cache_config:
            assert flexkv_config.cache_config.pop("tokens_per_block") == flexkv_config.block_size
        self.cache_config = CacheConfig(
            tokens_per_block=flexkv_config.block_size,
            **flexkv_config.cache_config,
        )
        self.flexkv_manager = KVManager(model_config=self.model_config,
                                        cache_config=self.cache_config,
                                        gpu_register_port=flexkv_config.server_recv_port)
        self.flexkv_manager.start()
        # self.dp_client = KVDPClient(self.server_recv_port, self.model_config)

        # request_id -> task_id
        self.req_id_to_task_dict: dict[str, int] = {}
        # launched but unfinished tasks
        self.get_tasks: dict[int, FlexKVGetTask] = {}
        self.put_tasks: dict[int, FlexKVPutTask] = {}
        # unlaunched tasks
        self.tasks_to_launch: dict[int, FlexKVTask] = {}
        self.tasks_to_cancel: dict[int, FlexKVTask] = {}

        self.flexkv_stats = FlexKVStats(flexkv_config.num_log_interval_requests)

        while not self.is_ready():
            logger.info("Waiting for flexkv init...")
            time.sleep(5)

        logger.info("Finish init FlexKVSchedulerConnector")

    def is_ready(
        self,
    ) -> bool:
        " Ask flexkv is ready "
        return self.flexkv_manager.is_ready()

    def shutdown(self) -> None:
        self.flexkv_manager.shutdown()

    @property
    def dp_client_id(self) -> int:
        return self.flexkv_manager.dp_client_id

    ####################
    #### Get Method ####
    ####################

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Args:
            request: Request to get.
            num_computed_tokens: Number of prefix tokens have already been computed,
                                which means not need to transfer from flexkv.

        Returns:
            tuple[int, bool]: A tuple containing two integer values representing the
                            number of new matched tokens and whether it is necessary
                            to get the new matched blocks from flexkv.
        """
        task_id, num_new_matched_tokens = self._get_match(request=request,
                                                          num_computed_tokens=num_computed_tokens)
        self.flexkv_stats.record_get(num_prompt_tokens=request.num_prompt_tokens,
                                     num_gpu_matched_tokens=num_computed_tokens,
                                     num_flexkv_matched_tokens=num_new_matched_tokens)

        if not self._need_to_get(num_prompt_tokens=request.num_prompt_tokens,
                                   num_computed_tokens=num_computed_tokens,
                                   num_new_matched_tokens=num_new_matched_tokens):
            return 0, False

        return num_new_matched_tokens, True


    def _get_match(
        self,
        request: "Request",
        num_computed_tokens: int = 0,
    ) -> tuple[int, int]:
        """
        Args:
            request: Request to get.
            num_computed_tokens: Number of prefix tokens have already been computed,
                                which means not need to transfer from flexkv.

        Returns:
            tuple[int, int]:  A tuple containing two integer values representing
                            the task_id and number of new matched tokens.
        """
        match_start_time = time.perf_counter()
        num_tokens_to_get = (cdiv(request.num_prompt_tokens+1, self.block_size)-1)*self.block_size
        token_ids = request.prompt_token_ids[:num_tokens_to_get]

        assert num_computed_tokens <= num_tokens_to_get
        assert num_computed_tokens % self.block_size == 0

        if num_tokens_to_get == num_computed_tokens:
            return -1, 0

        np_token_ids = np.array(token_ids)
        np_token_mask = np.ones_like(np_token_ids, dtype=bool)
        np_token_mask[:num_computed_tokens] = False
        task_id, matched_mask = self.flexkv_manager.get_match(token_ids=np_token_ids,
                                                         token_mask=np_token_mask)
        num_new_matched_tokens = matched_mask.sum().item()

        # Auto cancel if not call update_state_after_alloc()
        match_end_time = time.perf_counter()
        logger.debug(f"Get match cost {(match_end_time-match_start_time)*1000:.2f} ms.")
        if num_new_matched_tokens > 0:
            self.req_id_to_task_dict[request.request_id] = task_id
            self.tasks_to_cancel[task_id] = FlexKVGetTask(task_id=task_id,
                                                        request=request,
                                                        num_computed_tokens=num_computed_tokens,
                                                        num_new_matched_tokens=num_new_matched_tokens,
                                                        match_start_time=match_start_time,
                                                        match_end_time=match_end_time)

            logger.debug(f"FlexKV create get task: {self.tasks_to_cancel[task_id]}")

        return task_id, num_new_matched_tokens

    def _need_to_get(
        self,
        num_prompt_tokens: int,
        num_computed_tokens: int,
        num_new_matched_tokens: int,
    ) -> bool:
        """
        Determine whether it is necessary to get the new matched blocks from flexkv.
        """
        return num_new_matched_tokens > 0

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_new_matched_tokens: int,
    ) -> None:
        """
        Compute slot mapping and prepare to launch task.
        Only call after get_num_new_matched_tokens().

        Args:
            request: Request to get.
            blocks: All blocks of the request.
            num_new_matched_tokens: Number of new matched tokens returned by
            get_num_new_matched_tokens().

        Returns:
            None.
        """
        if num_new_matched_tokens == 0:
            return
        # prepare to launch task
        task_id = self.req_id_to_task_dict[request.request_id]
        task: FlexKVGetTask = self.tasks_to_cancel.pop(task_id)
        self.tasks_to_launch[task_id] = task

        # compute slot_mapping
        num_computed_blocks = task.num_computed_tokens // self.block_size
        num_blocks_to_get = num_new_matched_tokens // self.block_size
        all_block_ids = blocks.get_block_ids()[0]
        block_ids_to_get = all_block_ids[num_computed_blocks:num_computed_blocks+num_blocks_to_get]
        task.slot_mapping = np.array(block_ids_to_get).repeat(self.block_size)*self.block_size

    def wait_for_all_get_tasks(self) -> list[FlexKVResponse]:
        """
        Blocking wait for all get tasks.

        Returns:
            list[FlexKVResponse]: Responses of all get tasks.
        """
        return self._blocking_waiting_for_tasks(self.get_tasks)

    ####################
    #### Put Method ####
    ####################

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> bool:
        """
        Args:
            request: Request to put.
            blocks: All block_ids of the request.

        Returns:
            bool: whether thire is unfinished task for this request.
        """
        # Task not finished, can't free blocks
        if request.request_id in self.req_id_to_task_dict:
            return True

        # Abnormal finished, don't put
        if not (request.is_finished() and request.get_finished_reason() < 2):
            return False

        task_id, num_matched_tokens, num_unmatched_tokens = self._put_match(request=request)

        self.flexkv_stats.record_put(num_all_tokens=request.num_tokens,
                                     num_unmatched_tokens=num_unmatched_tokens)

        if not self._need_to_put(num_all_tokens=request.num_tokens,
                                num_matched_tokens=num_matched_tokens,
                                num_unmatched_tokens=num_unmatched_tokens):
            return False

        # prepare to launch task
        task: FlexKVPutTask = self.tasks_to_cancel.pop(task_id)
        self.tasks_to_launch[task_id] = task

        # compute slot mapping
        # num_blocks_to_put = (num_matched_tokens+num_unmatched_tokens) // self.block_size
        num_matched_blocks = num_matched_tokens // self.block_size
        num_unmatched_tokens = num_unmatched_tokens // self.block_size
        block_ids_to_put = block_ids[num_matched_blocks:num_matched_blocks+num_unmatched_tokens]
        task.slot_mapping = np.array(block_ids_to_put).repeat(self.block_size)*self.block_size

        return True

    def _put_match(
        self,
        request: "Request"
    ) -> tuple[int, int, int]:
        """
        Args:
            request: Request to put.

        Returns:
            tuple[int, int, int]:  A tuple containing three integer values representing
                            the task_id, number of matched tokens and number of unmatched tokens.
        """
        match_start_time = time.perf_counter()
        num_tokens_to_put = (cdiv(request.num_tokens+1, self.block_size)-1)*self.block_size
        token_ids = request.all_token_ids[:num_tokens_to_put]

        if num_tokens_to_put == 0:
            return -1, 0, 0

        np_token_ids = np.array(token_ids)
        task_id, unmatched_mask = self.flexkv_manager.put_match(token_ids=np_token_ids)

        num_unmatched_tokens = unmatched_mask.sum().item()
        num_matched_tokens = num_tokens_to_put - num_unmatched_tokens

        # Auto cancel if not need to put.
        match_end_time = time.perf_counter()
        logger.debug(f"Put match cost {(match_end_time-match_start_time)*1000:.2f} ms.")

        if num_unmatched_tokens > 0:
            self.req_id_to_task_dict[request.request_id] = task_id
            self.tasks_to_cancel[task_id] = FlexKVPutTask(task_id=task_id,
                                                        request=request,
                                                        num_matched_tokens=num_matched_tokens,
                                                        num_unmatched_tokens=num_unmatched_tokens,
                                                        match_start_time=match_start_time,
                                                        match_end_time=match_end_time)
            logger.debug(f"FlexKV create put task: {self.tasks_to_cancel[task_id]}")

        return task_id, num_matched_tokens, num_unmatched_tokens

    def _need_to_put(
        self,
        num_all_tokens: int,
        num_matched_tokens: int,
        num_unmatched_tokens: int,
    ) -> bool:
        """
        Determine whether it is necessary to put the unmatched blocks from flexkv.
        """
        return num_unmatched_tokens > 0

    def wait_for_all_put_tasks(self) -> list[FlexKVResponse]:
        """
        Blocking wait for all put tasks.

        Returns:
            list[FlexKVResponse]: Responses of all put tasks.
        """
        return self._blocking_waiting_for_tasks(self.put_tasks)

    #######################
    #### Common Method ####
    #######################

    def cancel_tasks(self) -> None:
        """
        Cancel tasks in self.cancel_tasks.
        Call before launch_tasks() to delete req_id in self.req_id_to_task_dict
        """
        # TODO: check if this method is inproc.
        if len(self.tasks_to_cancel) == 0:
            return
        for task in self.tasks_to_cancel.values():
            del self.req_id_to_task_dict[task.request.request_id]
            logger.info(f"FlexKV Cancel task: {task}")
        self.flexkv_manager.cancel(task_ids=list(self.tasks_to_cancel.keys()))
        self.tasks_to_cancel.clear()

    def launch_tasks(self) -> None:
        """
        Launch tasks in self.unlaunched_tasks
        """
        if len(self.tasks_to_launch) == 0:
            return
        task_launch_time = time.perf_counter()
        task_ids: list[int] = []
        slot_mappings: list[np.ndarray] = []

        for task_id, task in self.tasks_to_launch.items():
            logger.info(f"FlexKV Launch task: {task}")
            task.task_launch_time = task_launch_time
            task_ids.append(task_id)
            slot_mappings.append(task.slot_mapping)
            if isinstance(task, FlexKVGetTask):
                self.get_tasks[task_id] = task
            else:
                self.put_tasks[task_id] = task
        self.flexkv_manager.launch(task_ids=task_ids,
                                   slot_mappings=slot_mappings)
        self.tasks_to_launch.clear()

    def query_finished_task(self) -> tuple[set[str], set[str]]:
        """
        Get response of finished task.

        Returns:
            list[FlexKVResponse]: Responses of finished tasks.
        """
        if len(self.req_id_to_task_dict) == 0:
            return set(), set()
        logger.debug(f"unfinished task: {self.req_id_to_task_dict}")
        task_ids = list(self.get_tasks.keys()) + list(self.put_tasks.keys())
        responses_from_manager = self.flexkv_manager.try_wait(task_ids)
        task_finished_time = time.perf_counter()
        # responses_to_return: list[FlexKVResponse] = []
        finished_sending = set()
        finished_recving = set()
        num_failed_tasks = 0
        for task_id, response in responses_from_manager.items():
            success = (response.status == KVResponseStatus.SUCCESS)
            if task_id in self.get_tasks:
                task = self.get_tasks.pop(task_id)
                finished_recving.add(task.request.request_id)
            else:
                task = self.put_tasks.pop(task_id)
                finished_sending.add(task.request.request_id)
            del self.req_id_to_task_dict[task.request.request_id]
            task.task_finished_time = task_finished_time
            if success:
                logger.info(f"{task} finished successfully.")
            else:
                logger.error(f"{task} failed, status: {response.status}.")
                num_failed_tasks += 1
            # responses_to_return.append(FlexKVResponse(task_id=task_id, task_type=task.task_type,
            #                                             request=task.request, success=success))
        self.flexkv_stats.record_faild(num_failed_requests=num_failed_tasks)
        return finished_sending, finished_recving

    def _blocking_waiting_for_tasks(self, task_dict: dict[int, FlexKVTask]) -> list[FlexKVResponse]:
        """
        Blocking wait for tasks in task_dict.

        Returns:
            list[FlexKVResponse]: Responses of all tasks in task_dict.
        """
        if len(task_dict) == 0:
            return []

        task_ids = list(task_dict.keys())
        response_from_manager = self.flexkv_manager.wait(task_ids=task_ids)
        task_finished_time = time.perf_counter()
        responses_to_return: list[FlexKVResponse] = []
        for task_id, response in response_from_manager.items():
            success = (response.status == KVResponseStatus.SUCCESS)
            task = task_dict.pop(task_id)
            task.task_finished_time = task_finished_time
            if success:
                logger.info(f"{task} finished successfully.")
            else:
                logger.error(f"{task} failed, status: {response.status}.")
            responses_to_return.append(FlexKVResponse(task_id=task_id, task_type=task.task_type,
                                                      request=task.request, success=success))
        return responses_to_return


class FlexKVWorkerConnector:
    def __init__(
        self,
        flexkv_config: FlexKVConfig,
    ):
        current_device_id = torch.cuda.current_device()
        self.flexkv_config = flexkv_config
        logger.info(f"Start init FlexKVWorkerConnector to {flexkv_config.server_recv_port}")
        self.tp_client = KVTPClient(flexkv_config.server_recv_port, 0, current_device_id)
        logger.info("Finish init FlexKVWorkerConnector")

    def register_to_server(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Start register kv_caches")
        gpu_blocks = list(kv_caches.values())
        num_layer = len(kv_caches)
        if self.flexkv_config.use_mla:
            assert gpu_blocks[0].ndim == 3, (
                f"expect kv cached tensor has 3 dim but get shape={gpu_blocks[0].shape}.")
            num_blocks = gpu_blocks[0].shape[0]
            block_size = gpu_blocks[0].shape[1]
            num_kv_heads = 1
            head_size = gpu_blocks[0].shape[2]
        else:
            assert gpu_blocks[0].ndim == 5, (
                f"expect kv cached tensor has 5 dim but get shape={gpu_blocks[0].shape}.")
            num_blocks = gpu_blocks[0].shape[1]
            block_size = gpu_blocks[0].shape[2]
            num_kv_heads = gpu_blocks[0].shape[3]
            head_size = gpu_blocks[0].shape[4]
        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERWISE,
            num_layer=num_layer,
            num_block=num_blocks,
            tokens_per_block=block_size,
            num_head=num_kv_heads,
            head_size=head_size,
            is_mla=self.flexkv_config.use_mla,
        )
        self.tp_client.register_to_server(gpu_blocks, gpu_layout)
        logger.info("Finish register kv_caches")


class FlexKVConnectorV1Impl:
    def __init__(self, vllm_config: "VllmConfig", role: "KVConnectorRole"):
        self.role = role
        flexkv_config = FlexKVConfig.from_env()
        flexkv_config.post_init_from_vllm_config(vllm_config)

        if role == KVConnectorRole.SCHEDULER:
            self.connector = FlexKVSchedulerConnector(flexkv_config)
        elif role == KVConnectorRole.WORKER:
            self.connector = FlexKVWorkerConnector(flexkv_config)
        else:
            raise ValueError(f"Unrecognized KVConnectorRole: {role}.")

    def shutdown(self):
        if self.role == KVConnectorRole.SCHEDULER:
            self.connector.shutdown()

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        pass

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return None, None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        self.connector.register_to_server(kv_caches)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self.connector.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        self.connector.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
            self, scheduler_output: "SchedulerOutput") -> "KVConnectorMetadata":
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        self.connector.cancel_tasks()
        self.connector.launch_tasks()
        return KVConnectorMetadata()

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """

        finished_sending, finished_recving = self.connector.query_finished_task()
        connector_output.finished_sending = finished_sending
        connector_output.finished_recving = finished_recving


    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return self.connector.request_finished(request, block_ids), None
