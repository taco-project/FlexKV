import os
from typing import Any, Optional, Tuple
import time
from typing import TYPE_CHECKING, Optional, Literal, Any, List, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch
import traceback
from flexkv.kvmanager import KVManager
from flexkv.server.client import KVTPClient
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.request import KVResponseStatus
from flexkv.common.debug import flexkv_logger
from flexkv.integration.stats import FlexKVStats
from flexkv.integration.utils import cdiv
from flexkv.integration.config import FlexKVConfig
from flexkv.integration.tensorrt_llm.meta import(
    FlexKVResponse, FlexKVTask, FlexKVGetTask, FlexKVPutTask, FlexKVConnectorMetadata)
from flexkv.transfer_manager import TransferManagerOnRemote

from flexkv.integration.tensorrt_llm.utils import RequestWrapper, get_mapping_from_config
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.bindings.executor import ExecutorConfig
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
    KvCacheConnectorScheduler, KvCacheConnectorWorker,
    SchedulerOutput)

class FlexKVSchedulerConnector(KvCacheConnectorScheduler):
    def __init__(self, config: ExecutorConfig):
        # Set environment variable for FlexKV with TensorRT-LLM，this must before KVManager initialization
        os.environ['FLEXKV_WITH_TRTLLM'] = '1'
        flexkv_config = FlexKVConfig.from_env() 
        flexkv_config.post_init_from_trt_config(config)
        self.node_rank, self.tp_rank, self.dp_rank = get_rank_info_from_trt_config(config)

        flexkv_logger.info(f"Start init FlexKVSchedulerConnector with {flexkv_config}")
        self.server_recv_port = flexkv_config.server_recv_port
        self.tp_size = flexkv_config.model_config.tp_size
        self.dp_size = flexkv_config.model_config.dp_size
        self.block_size = flexkv_config.cache_config.tokens_per_block
        self.model_config = flexkv_config.model_config
        self.cache_config = flexkv_config.cache_config
        self.flexkv_manager = KVManager(model_config=self.model_config,
                                        cache_config=self.cache_config,
                                        server_recv_port=flexkv_config.server_recv_port,
                                        dp_client_id=self.dp_rank)
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

        self.flexkv_stats = FlexKVStats(os.getenv('FLEXKV_NUM_LOG_INTERVAL_REQUESTS', 200))

        flexkv_logger.info("Finish init FlexKVSchedulerConnector")

    def wait_for_initialization(self):
        """
        Wait for the FlexKV manager to be initialized.
        """
        while not self.flexkv_manager.is_ready():
            time.sleep(0.1)
        flexkv_logger.info("FlexKV manager is ready")

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
        _request: "LlmRequest",
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
        request = RequestWrapper(_request)
        task_id, num_new_matched_tokens = self._get_match(_request=_request,
                                                          num_computed_tokens=num_computed_tokens)
        
        self.flexkv_stats.record_get(num_prompt_tokens=request.num_prompt_tokens,
                                     num_gpu_matched_tokens=num_computed_tokens,
                                     num_flexkv_matched_tokens=num_new_matched_tokens)

        if not self._need_to_get(num_prompt_tokens=request.num_prompt_tokens,
                                   num_computed_tokens=num_computed_tokens,
                                   num_new_matched_tokens=num_new_matched_tokens):
            return 0, False

        return num_new_matched_tokens, True

    def _extract_namespace(self, _request: "LlmRequest") -> Optional[List[str]]:
        """
        Extract namespace information from TensorRT-LLM LlmRequest for cache isolation.
        
        This method extracts namespace components from multiple sources in priority order:
        1. lora_task_id: LoRA task identifier for multi-tenant LoRA serving
        2. cache_salt_id: Explicit cache isolation identifier
        3. namespace_info: User-defined namespace hierarchy (string with "|" separator)
        
        The namespace components are combined to form a hierarchical namespace path,
        enabling fine-grained KV cache isolation across different tenants, users, or sessions.
        
        Args:
            _request: LlmRequest object from TensorRT-LLM
            
        Returns:
            Optional[List[str]]: Ordered list of namespace components forming the hierarchy,
                                or None if no namespace information is available
                                
        Example:
            If request has lora_task_id=123, cache_salt_id="session_1",
            namespace_info="tenant_A|user_1", the result will be:
            ["123", "session_1", "tenant_A", "user_1"]
        """
        namespace_info = []

        if hasattr(_request, 'lora_task_id'):
            lora_task_id = getattr(_request, 'lora_task_id', None)
            if lora_task_id is not None:
                namespace_info.append(str(lora_task_id))

        if hasattr(_request, 'cache_salt_id') and _request.cache_salt_id is not None:
            cache_salt_id = _request.cache_salt_id
            if cache_salt_id is not None:
                namespace_info.append(str(cache_salt_id))

        if hasattr(_request, 'namespace_info') and _request.namespace_info is not None:
            user_namespace = _request.namespace_info
            # namespace_info in TensorRT-LLM is a string, potentially with "|" separator
            if isinstance(user_namespace, str):
                # Split by "|" to support hierarchical namespaces like "tenant_A|user_1"
                namespace_components = user_namespace.split('|')
                namespace_info.extend([comp.strip() for comp in namespace_components if comp.strip()])
            else:
                namespace_info.append(str(user_namespace))
        
        if len(namespace_info) == 0:
            return None
        
        return namespace_info

    def _get_match(
        self,
        _request: "LlmRequest",
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
        request = RequestWrapper(_request)
        flexkv_logger.info(f"Get match request: {request}")
        
        match_start_time = time.perf_counter()
        num_tokens_to_get = (request.num_prompt_tokens // self.block_size) * self.block_size
        if num_tokens_to_get == 0:
            return -1, 0

        assert num_computed_tokens <= num_tokens_to_get, (
            f"{num_computed_tokens=} must less equal to {num_tokens_to_get=}")
        assert num_computed_tokens % self.block_size == 0,(
            f"{num_computed_tokens=}, {self.block_size=}")

        if num_tokens_to_get == num_computed_tokens:
            return -1, 0

        # Use cached numpy token sequence to avoid repeated list->numpy conversion in hot path
        np_all_token_ids = request.np_token_ids
        if np_all_token_ids.shape[0] < num_tokens_to_get:
            raise ValueError(
                f"Request {request.req_id} tokens shorter than expected: "
                f"{np_all_token_ids.shape[0]=}, {num_tokens_to_get=}"
            )
        np_token_ids = np_all_token_ids[:num_tokens_to_get]
        np_token_mask = np.ones_like(np_token_ids, dtype=bool)
        np_token_mask[:num_computed_tokens] = False
        
        # Extract namespace info for cache isolation
        namespace = self._extract_namespace(_request)
        
        task_id, matched_mask = self.flexkv_manager.get_match(
            token_ids=np_token_ids,
            token_mask=np_token_mask,
            namespace=namespace,
        )
        num_new_matched_tokens = matched_mask.sum().item()

        # Auto cancel if not call update_state_after_alloc()
        match_end_time = time.perf_counter()
        flexkv_logger.debug(f"Get match cost {(match_end_time-match_start_time)*1000:.2f} ms.")
        if num_new_matched_tokens > 0:
            self.req_id_to_task_dict[request.req_id] = task_id
            self.tasks_to_cancel[task_id] = FlexKVGetTask(task_id=task_id,
                                                        request=request,
                                                        num_computed_tokens=num_computed_tokens,
                                                        num_new_matched_tokens=num_new_matched_tokens,
                                                        match_start_time=match_start_time,
                                                        match_end_time=match_end_time)

            flexkv_logger.debug(f"FlexKV create get task: {self.tasks_to_cancel[task_id]}")

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

    # def update_state_after_alloc(
    #     self,
    #     _request: "LlmRequest",
    #     blocks: "KVCacheBlocks",
    # ) -> None:
    def update_state_after_alloc(
        self,
        _request: "LlmRequest",
        block_ids: List[int],
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
        request = RequestWrapper(_request)
        if request.req_id not in self.req_id_to_task_dict:
            flexkv_logger.info(f"Request {request.req_id} didn't launch task, skip update state after alloc.")
            return

        # prepare to launch task
        task_id = self.req_id_to_task_dict[request.req_id]
        task: FlexKVGetTask = self.tasks_to_cancel.pop(task_id)
        self.tasks_to_launch[task_id] = task

        # compute slot_mapping
        num_computed_blocks = task.num_computed_tokens // self.block_size
        num_blocks_to_get = task.num_new_matched_tokens // self.block_size
        block_ids_to_get = block_ids[num_computed_blocks:num_computed_blocks+num_blocks_to_get]
        task.slot_mapping = np.array(block_ids_to_get).repeat(self.block_size)*self.block_size

    def wait_for_all_get_tasks(self) -> list[FlexKVResponse]:
        return self._blocking_waiting_for_tasks(self.get_tasks)

    ####################
    #### Put Method ####
    ####################

    def request_finished(
        self,
        _request: "LlmRequest",
        block_ids: list[int],
    ) -> bool:
        """
        Args:
            request: Request to put.
            blocks: All block_ids of the request.

        Returns:
            bool: whether thire is unfinished task for this request.
        """
        request = RequestWrapper(_request)
        
        # Task not finished, can't free blocks
        if request.req_id in self.req_id_to_task_dict:
            return True

        # Abnormal finished, don't put
        if not (request.is_finished() and request.is_finished_normal()):
            return False

        task_id, num_matched_tokens, num_unmatched_tokens = self._put_match(_request=_request)

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
        num_matched_blocks = num_matched_tokens // self.block_size
        num_unmatched_blocks = num_unmatched_tokens // self.block_size
        block_ids_to_put = block_ids[num_matched_blocks:num_matched_blocks+num_unmatched_blocks]
        flexkv_logger.info(f"{num_matched_blocks=}, {num_matched_blocks+num_unmatched_blocks=}, {len(block_ids)=}")
        task.slot_mapping = np.array(block_ids_to_put).repeat(self.block_size)*self.block_size
        flexkv_logger.info(f"{task_id=}, {num_matched_tokens=}, {num_unmatched_tokens=}, {len(block_ids_to_put)=}, {self.block_size=}, {task.slot_mapping.shape=}")
        return True

    def _put_match(
        self,
        _request: "LlmRequest"
    ) -> tuple[int, int, int]:
        """
        Args:
            request: Request to put.

        Returns:
            tuple[int, int, int]:  A tuple containing three integer values representing
                            the task_id, number of matched tokens and number of unmatched tokens.
        """
        request = RequestWrapper(_request)
        flexkv_logger.info(f"Put match request: {request}")
        match_start_time = time.perf_counter()
        num_tokens_to_put = (cdiv(request.num_tokens, self.block_size) - 1) * self.block_size

        if num_tokens_to_put == 0:
            return -1, 0, 0

        # Use cached numpy token sequence
        np_all_token_ids = request.np_token_ids
        if np_all_token_ids.shape[0] < num_tokens_to_put:
            raise ValueError(
                f"Request {request.req_id} tokens shorter than expected: "
                f"{np_all_token_ids.shape[0]=}, {num_tokens_to_put=}"
            )
        np_token_ids = np_all_token_ids[:num_tokens_to_put]
        
        # Extract namespace info for cache isolation
        namespace = self._extract_namespace(_request)
        
        task_id, unmatched_mask = self.flexkv_manager.put_match(
            token_ids=np_token_ids,
            namespace=namespace,
        )

        num_unmatched_tokens = unmatched_mask.sum().item()
        num_matched_tokens = num_tokens_to_put - num_unmatched_tokens

        # Auto cancel if not need to put.
        match_end_time = time.perf_counter()
        flexkv_logger.debug(f"Put match cost {(match_end_time-match_start_time)*1000:.2f} ms. {num_unmatched_tokens=}")
        
        if num_unmatched_tokens > 0:
            self.req_id_to_task_dict[request.req_id] = task_id
            self.tasks_to_cancel[task_id] = FlexKVPutTask(task_id=task_id,
                                                        request=request,
                                                        num_matched_tokens=num_matched_tokens,
                                                        num_unmatched_tokens=num_unmatched_tokens,
                                                        match_start_time=match_start_time,
                                                        match_end_time=match_end_time)
            flexkv_logger.debug(f"FlexKV create put task: {self.tasks_to_cancel[task_id]}")

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
        if len(self.tasks_to_cancel) == 0:
            return
        for task in self.tasks_to_cancel.values():
            del self.req_id_to_task_dict[task.request.req_id]
            flexkv_logger.info(f"FlexKV Cancel task: {task}")
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
            flexkv_logger.info(f"FlexKV Launch task: {task}")
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
        task_ids = list(self.get_tasks.keys()) + list(self.put_tasks.keys())
        responses_from_manager = self.flexkv_manager.try_wait(task_ids)
        task_finished_time = time.perf_counter()
        finished_sending = set()
        finished_recving = set()
        num_failed_tasks = 0
        for task_id, response in responses_from_manager.items():
            success = (response.status == KVResponseStatus.SUCCESS)
            if task_id in self.get_tasks:
                task = self.get_tasks.pop(task_id)
                finished_recving.add(task.request.req_id)
            else:
                task = self.put_tasks.pop(task_id)
                finished_sending.add(task.request.req_id)
            del self.req_id_to_task_dict[task.request.req_id]
            task.task_finished_time = task_finished_time
            if success:
                flexkv_logger.info(f"{task} finished successfully.")
            else:
                flexkv_logger.error(f"{task} failed, status: {response.status}.")
                num_failed_tasks += 1
        flexkv_logger.debug(f"unfinished task: {self.req_id_to_task_dict}")
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
                flexkv_logger.info(f"{task} finished successfully.")
            else:
                flexkv_logger.error(f"{task} failed, status: {response.status}.")
            responses_to_return.append(FlexKVResponse(task_id=task_id, task_type=task.task_type,
                                                      request=task.request, success=success))
        return responses_to_return

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        self.cancel_tasks()
        self.launch_tasks()
        finished_sending, finished_recving = self.query_finished_task()
        metadata = FlexKVConnectorMetadata(
            finished_sending=list(finished_sending),
            finished_recving=list(finished_recving))
        return metadata
    
    @property
    def dp_client_id(self) -> int:
        return self.flexkv_manager.dp_client_id

class FlexKVWorkerConnector(KvCacheConnectorWorker):
    def __init__(self, config: ExecutorConfig):
        flexkv_config = FlexKVConfig.from_env()
        self.node_rank, self.tp_rank, self.dp_rank = get_rank_info_from_trt_config(config)
        flexkv_config.post_init_from_trt_config(config)
        dp_client_id = self.dp_rank
        
        current_device_id = torch.cuda.current_device() + dp_client_id * flexkv_config.model_config.tp_size
        self.flexkv_config = flexkv_config
        
        # For multi-node TP on remote node (node B), worker0 needs to create TransferManagerOnRemote process
        self.remote_process = None
        if self._need_to_create_remote_process():
            flexkv_logger.info("Multi-node TP detected on remote node, worker0 creating TransferManagerOnRemote process")
            self.remote_process = TransferManagerOnRemote.create_process()
            flexkv_logger.info(f"TransferManagerOnRemote process created, PID: {self.remote_process.pid}")
        
        flexkv_logger.info(f"Start init FlexKVWorkerConnector to {flexkv_config.gpu_register_port}, dp_client_id: {dp_client_id}")
        self.tp_client = KVTPClient(flexkv_config.gpu_register_port, dp_client_id, current_device_id)
        flexkv_logger.info("Finish init FlexKVWorkerConnector")
    
    def _need_to_create_remote_process(self) -> bool:
        """Check if need to create TransferManagerOnRemote process.
        
        Returns True when all of the following conditions are met:
        - Multi-node TP is detected (tp_size > gpus_per_node)
        - Current node is not master node (node_rank > 0)
        - Current worker is worker0 in TP group (tp_rank == 0)
        
        Returns:
            bool: True if need to create TransferManagerOnRemote process, False otherwise.
        """
        try:
            is_master_node = self.node_rank == 0
            is_first_worker = self.tp_rank % 8 == 0
            is_multinode_tp = self.flexkv_config.model_config.tp_size > torch.cuda.device_count()
            flexkv_logger.info(f"{is_master_node=}, {is_first_worker=}, {is_multinode_tp=}")
            
            return is_multinode_tp and not is_master_node and is_first_worker
        except Exception as e:
            flexkv_logger.error(f"Failed to get node info from flexkv_config: {e}")
            flexkv_logger.error(traceback.format_exc())
            raise e

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        # vllm kv_caches: dict[str, torch.Tensor]
        # trt kv_caches: torch.Tensor
        
        # shepe = ITensor::makeShape({mNumPrimaryBlocks, pool.numLayers, mKVFactor, blockSize});
        # 1. mNumPrimaryBlocks{blocksInPrimaryPool}            blocksInPrimaryPool = tc::ceilDiv(maxTokens, tokensPerBlock);
        # 2. layer_num
        # 3. mKVFactor{mCacheType == CacheType::kSELFKONLY ? 1 : 2}
        # 4. blockSize((numKvHeads * sizePerHead * tokensPerBlock) / quantSize)
    
        flexkv_logger.info(f"Start register kv_caches, shape: {kv_cache_tensor.shape}")
        
        # Get actual device from tensor (more reliable in MPI environment)
        logical_device_id = kv_cache_tensor.device.index
        flexkv_logger.debug(f"Tensor is on device: {kv_cache_tensor.device}, logical device.index={logical_device_id}")
        flexkv_logger.debug(f"self.tp_client.device_id (from init): {self.tp_client.device_id}")
        
        # Get physical GPU ID (in case CUDA_VISIBLE_DEVICES is set)
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible_devices:
            # Map logical ID to physical ID
            visible_gpus = [int(x) for x in cuda_visible_devices.split(',')]
            physical_device_id = visible_gpus[logical_device_id] if logical_device_id < len(visible_gpus) else logical_device_id
            flexkv_logger.debug(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}, mapping logical {logical_device_id} -> physical {physical_device_id}")
        else:
            physical_device_id = logical_device_id
            flexkv_logger.debug(f"No CUDA_VISIBLE_DEVICES set, using logical device ID {logical_device_id}")
        
        # Use physical device ID for registration
        correct_device_id = physical_device_id
        
        if self.flexkv_config.model_config.use_mla:
            assert kv_cache_tensor.ndim == 4, (f"expect kv cached tensor has 4 dim but get shape={kv_cache_tensor.shape}")

        num_blocks = kv_cache_tensor.shape[0]
        num_layers = kv_cache_tensor.shape[1]
        kv_dim = kv_cache_tensor.shape[2]
        block_size = self.flexkv_config.cache_config.tokens_per_block
        num_kv_heads = 1 if self.flexkv_config.model_config.use_mla else self.flexkv_config.model_config.num_kv_heads
        head_size = self.flexkv_config.model_config.head_size
        if self.flexkv_config.model_config.use_mla:
            assert kv_dim == 1, (f"expect kv_dim eqals to 1 when using MLA but get kv_dim={kv_dim}")
        
        gpu_blocks = [kv_cache_tensor] # convert to list for flexkv register 
 
        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.BLOCKFIRST,
            num_layer=num_layers,
            num_block=num_blocks,
            tokens_per_block=block_size,
            num_head=num_kv_heads,
            head_size=head_size,
            is_mla=self.flexkv_config.model_config.use_mla,
        )
        flexkv_logger.info(f"gpu_layout: {gpu_layout}")
        # Use correct device_id from tensor's actual device
        self.tp_client.register_to_server(gpu_blocks, gpu_layout, override_device_id=correct_device_id)
        flexkv_logger.info(f"Finish register kv_caches on device {correct_device_id}")

    def start_load_kv(self, stream: torch.cuda.Stream):
        return
        
    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        return
    
    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        return

    def wait_for_save(self, stream: torch.cuda.Stream):
        return
    
    def get_finished(
            self, finished_gen_req_ids: List[int],
            started_loading_req_ids: List[int]) -> Tuple[List[int], List[int]]:
        
        finished_sending = self._metadata.finished_sending
        finished_recving = self._metadata.finished_recving
        return finished_sending, finished_recving
    
    def shutdown(self) -> None:
        """Shutdown the worker connector and cleanup resources.
        
        For node B worker0, this will shutdown the TransferManagerOnRemote process.
        """
        if hasattr(self, 'remote_process') and self.remote_process is not None:
            flexkv_logger.info("Shutting down TransferManagerOnRemote process on node B worker0")
            try:
                # Terminate the process
                if hasattr(self.remote_process, '_popen'):
                    self.remote_process._popen.terminate()
                    self.remote_process._popen.wait(timeout=5.0)
                    if self.remote_process._popen.poll() is None:
                        flexkv_logger.warning("TransferManagerOnRemote process did not terminate, killing it")
                        self.remote_process._popen.kill()
                        self.remote_process._popen.wait()
                else:
                    # Fallback for subprocess.Popen
                    self.remote_process.terminate()
                    self.remote_process.wait(timeout=5.0)
                    if self.remote_process.poll() is None:
                        flexkv_logger.warning("TransferManagerOnRemote process did not terminate, killing it")
                        self.remote_process.kill()
                        self.remote_process.wait()
                flexkv_logger.info("TransferManagerOnRemote process shutdown complete")
            except Exception as e:
                flexkv_logger.error(f"Error shutting down TransferManagerOnRemote process: {e}")
            finally:
                self.remote_process = None
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'remote_process') and self.remote_process is not None:
            self.shutdown()

def get_rank_info_from_trt_config(config: Any) -> Tuple[int, int, int]:
    mapping = get_mapping_from_config(config)

    enable_attention_dp = bool(getattr(mapping, "enable_attention_dp", False))
    node_rank = int(getattr(mapping, "node_rank", 0))
    tp_rank = int(getattr(mapping, "tp_rank", 0)) if not enable_attention_dp else 0
    dp_rank = int(getattr(mapping, "rank", 0)) if enable_attention_dp else 0
    return node_rank, tp_rank, dp_rank
