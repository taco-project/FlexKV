from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import resource
import sys
from typing import Any

import numpy as np
import torch

from .gpu_worker import _fill_blocks, _verify_blocks, _zero_blocks


@dataclass
class _CacheNode:
    children: dict[tuple[int, ...], "_CacheNode"] = field(default_factory=dict)
    references: int = 0
    # effective_rank -> group -> layer tensor for one logical KV block
    data: dict[int, list[list[torch.Tensor]]] = field(default_factory=dict)
    # effective_rank -> layer tensor for the sequence's SWA slot
    swa_data: dict[int, list[torch.Tensor]] = field(default_factory=dict)


@dataclass
class _Task:
    operation: str
    tokens: tuple[int, ...]
    nodes: list[_CacheNode]
    missing_start: int = 0


class _SuccessStatus:
    name = "SUCCESS"


class _SuccessResponse:
    status = _SuccessStatus()


class CpuTorchWorker:
    """In-process stand-in for one inference-engine GPU KV pool."""

    def __init__(self, config, dp_rank: int, effective_rank: int, device_id: int):
        from .device import torch_dtype

        self.config = config
        self.dp_rank = dp_rank
        self.effective_rank = effective_rank
        self.device_id = device_id
        self.tensors_per_group: list[list[torch.Tensor]] = []
        for group in config.preset.groups:
            group_tpb = config.tokens_per_block // group.compress_ratio
            shape = (
                config.model.gpu_blocks_per_rank,
                group_tpb,
                group.num_kv_heads,
                group.head_size,
            )
            self.tensors_per_group.append([
                torch.zeros(shape, dtype=torch_dtype(group.dtype))
                for _ in range(group.num_layers)
            ])
        self.swa_tensors: list[torch.Tensor] | None = None
        if config.features.swa:
            shape = (
                config.cache.swa_num_slots,
                config.tokens_per_block,
                1,
                config.preset.swa_head_size,
            )
            self.swa_tensors = [
                torch.zeros(shape, dtype=torch.uint8)
                for _ in range(config.preset.swa_num_layers)
            ]

    def request(self, command: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
        del timeout
        operation = command["operation"]
        block_ids = [int(value) for value in command.get("block_ids", [])]
        patterns = [int(value) for value in command.get("patterns", [])]
        if operation == "seed":
            for group_id, tensors in enumerate(self.tensors_per_group):
                _fill_blocks(tensors, block_ids, patterns, self.effective_rank, group_id)
            if self.swa_tensors is not None and command.get("swa_block_ids"):
                _fill_blocks(
                    self.swa_tensors,
                    [int(value) for value in command["swa_block_ids"]],
                    [patterns[-1]],
                    self.effective_rank,
                    len(self.tensors_per_group),
                )
            return {"ok": True}
        if operation == "zero":
            max_layers = command.get("max_layers")
            for tensors in self.tensors_per_group:
                _zero_blocks(tensors, block_ids, max_layers)
            if self.swa_tensors is not None and command.get("swa_block_ids"):
                _zero_blocks(
                    self.swa_tensors,
                    [int(value) for value in command["swa_block_ids"]],
                    max_layers,
                )
            return {"ok": True}
        if operation == "verify":
            errors = []
            max_layers = int(command.get("max_layers", 2))
            max_bytes = int(command.get("max_bytes", 256))
            for group_id, tensors in enumerate(self.tensors_per_group):
                errors.extend(_verify_blocks(
                    tensors,
                    block_ids,
                    patterns,
                    self.effective_rank,
                    group_id,
                    max_layers,
                    max_bytes,
                ))
            if self.swa_tensors is not None and command.get("swa_block_ids"):
                errors.extend(_verify_blocks(
                    self.swa_tensors,
                    [int(value) for value in command["swa_block_ids"]],
                    [patterns[-1]],
                    self.effective_rank,
                    len(self.tensors_per_group),
                    max_layers,
                    max_bytes,
                ))
            return {"ok": not errors, "errors": errors[:20]}
        if operation == "memory":
            tensors = [tensor for group in self.tensors_per_group for tensor in group]
            tensors.extend(self.swa_tensors or [])
            allocated = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
            rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return {
                "ok": True,
                "allocated_bytes": allocated,
                "reserved_bytes": allocated,
                "max_allocated_bytes": allocated,
                # macOS reports bytes, Linux reports KiB.
                "rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            }
        if operation == "shutdown":
            return {"ok": True}
        raise ValueError(f"Unknown CPU stub worker operation {operation!r}")

    def shutdown(self) -> None:
        pass


class CpuStubManager:
    """CPU implementation of the KVManager API used by the stress driver.

    The manager maintains the same block-prefix/refcount model as PrefixOracle
    and copies real PyTorch tensor blocks between worker pools and an in-memory
    cache. It intentionally does not exercise FlexKV IPC, native transfer, SSD,
    or eventfd implementations.
    """

    def __init__(self, config, dp_rank: int, workers: list[CpuTorchWorker], capacity_blocks: int):
        self.config = config
        self.dp_rank = dp_rank
        self.workers = [worker for worker in workers if worker.dp_rank == dp_rank]
        self.capacity_blocks = capacity_blocks
        self.root = _CacheNode()
        self.sequences: deque[tuple[tuple[int, ...], ...]] = deque()
        self.logical_blocks = 0
        self.tasks: dict[int, _Task] = {}
        self.next_task_id = 1

    def start(self) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def _keys(self, tokens) -> tuple[tuple[int, ...], ...]:
        values = tuple(int(value) for value in tokens)
        tpb = self.config.tokens_per_block
        aligned = len(values) // tpb * tpb
        return tuple(tuple(values[start:start + tpb]) for start in range(0, aligned, tpb))

    def _new_task(self, task: _Task, *, retain: bool = True) -> int:
        task_id = self.next_task_id
        self.next_task_id += 1
        if retain:
            self.tasks[task_id] = task
        return task_id

    def get_match(self, tokens, swa_aware: bool = False):
        del swa_aware
        keys = self._keys(tokens)
        node = self.root
        nodes = []
        for key in keys:
            child = node.children.get(key)
            if child is None:
                break
            nodes.append(child)
            node = child
        hit_tokens = len(nodes) * self.config.tokens_per_block
        mask = np.zeros(len(tokens), dtype=np.int64)
        mask[:hit_tokens] = 1
        task_id = self._new_task(
            _Task("get", tuple(int(v) for v in tokens), nodes),
            retain=bool(nodes),
        )
        return task_id, mask

    def put_match(self, tokens):
        keys = self._keys(tokens)
        node = self.root
        nodes = []
        for key in keys:
            child = node.children.get(key)
            if child is None:
                break
            nodes.append(child)
            node = child
        matched_blocks = len(nodes)
        mask = np.zeros(len(tokens), dtype=bool)
        mask[matched_blocks * self.config.tokens_per_block:len(keys) * self.config.tokens_per_block] = True
        task_id = self._new_task(
            _Task("put", tuple(int(v) for v in tokens), nodes, missing_start=matched_blocks),
            retain=matched_blocks < len(keys),
        )
        return task_id, mask

    def _mapping_blocks(self, slot_mapping, expected_blocks: int) -> list[int]:
        slots = np.asarray(slot_mapping, dtype=np.int64).reshape(-1)
        tpb = self.config.tokens_per_block
        if len(slots) != expected_blocks * tpb:
            raise ValueError(
                f"CPU stub expected {expected_blocks * tpb} slots, got {len(slots)}"
            )
        return [int(slots[index * tpb] // tpb) for index in range(expected_blocks)]

    def _copy_worker_block_to_node(self, worker: CpuTorchWorker, block_id: int,
                                   node: _CacheNode) -> None:
        node.data[worker.effective_rank] = [
            [tensor[block_id].clone() for tensor in group]
            for group in worker.tensors_per_group
        ]

    def _copy_node_to_worker_block(self, node: _CacheNode, worker: CpuTorchWorker,
                                   block_id: int) -> None:
        cached_groups = node.data.get(worker.effective_rank)
        if cached_groups is None:
            raise RuntimeError(
                f"CPU stub cache is missing rank {worker.effective_rank} data"
            )
        for tensors, cached_layers in zip(worker.tensors_per_group, cached_groups):
            for tensor, cached in zip(tensors, cached_layers):
                tensor[block_id].copy_(cached)

    def _commit_put(self, task: _Task, slot_mapping, swa_slot_mapping=None) -> None:
        keys = self._keys(task.tokens)
        missing = len(keys) - task.missing_start
        source_blocks = self._mapping_blocks(slot_mapping, missing)
        node = self.root
        node.references += 1
        for index, key in enumerate(keys):
            child = node.children.get(key)
            if child is None:
                child = _CacheNode()
                node.children[key] = child
            child.references += 1
            if index >= task.missing_start:
                block_id = source_blocks[index - task.missing_start]
                for worker in self.workers:
                    self._copy_worker_block_to_node(worker, block_id, child)
            node = child
        if swa_slot_mapping is not None and len(np.asarray(swa_slot_mapping).reshape(-1)):
            swa_block = int(np.asarray(swa_slot_mapping).reshape(-1)[0])
            for worker in self.workers:
                if worker.swa_tensors is not None:
                    node.swa_data[worker.effective_rank] = [
                        tensor[swa_block].clone() for tensor in worker.swa_tensors
                    ]
        self.sequences.append(keys)
        self.logical_blocks += len(keys)
        while self.capacity_blocks and self.logical_blocks > self.capacity_blocks and self.sequences:
            self._remove(self.sequences.popleft())

    def _remove(self, keys: tuple[tuple[int, ...], ...]) -> None:
        node = self.root
        path = []
        node.references -= 1
        for key in keys:
            child = node.children.get(key)
            if child is None:
                return
            path.append((node, key, child))
            child.references -= 1
            node = child
        for parent, key, child in reversed(path):
            if child.references == 0:
                del parent.children[key]
            else:
                break
        self.logical_blocks -= len(keys)

    def _load_get(self, task: _Task, slot_mapping, swa_slot_mapping=None) -> None:
        target_blocks = self._mapping_blocks(slot_mapping, len(task.nodes))
        for node, block_id in zip(task.nodes, target_blocks):
            for worker in self.workers:
                self._copy_node_to_worker_block(node, worker, block_id)
        if task.nodes and swa_slot_mapping is not None and len(np.asarray(swa_slot_mapping).reshape(-1)):
            terminal = task.nodes[-1]
            swa_block = int(np.asarray(swa_slot_mapping).reshape(-1)[0])
            for worker in self.workers:
                cached = terminal.swa_data.get(worker.effective_rank)
                if cached is None or worker.swa_tensors is None:
                    continue
                for tensor, source in zip(worker.swa_tensors, cached):
                    tensor[swa_block].copy_(source)

    def launch(self, task_ids, slot_mappings, swa_slot_mappings=None, **kwargs):
        del kwargs
        swa_mappings = swa_slot_mappings or [None] * len(task_ids)
        for task_id, slot_mapping, swa_mapping in zip(task_ids, slot_mappings, swa_mappings):
            task = self.tasks.pop(int(task_id))
            if task.operation == "put":
                self._commit_put(task, slot_mapping, swa_mapping)
            elif task.operation == "get":
                self._load_get(task, slot_mapping, swa_mapping)
            else:
                raise ValueError(f"Unknown CPU stub task operation {task.operation!r}")
        return list(task_ids)

    def wait(self, task_ids, **kwargs):
        del kwargs
        return {int(task_id): _SuccessResponse() for task_id in task_ids}

    def put_async(self, tokens, slot_mapping):
        task_id, mask = self.put_match(tokens)
        if mask.any():
            self._commit_put(self.tasks.pop(task_id), np.asarray(slot_mapping)[mask])
        return [task_id]

    def get_async(self, tokens, slot_mapping):
        task_id, mask = self.get_match(tokens)
        if int(mask.sum()) != len(tokens):
            raise KeyError(f"CPU stub async GET matched {int(mask.sum())}/{len(tokens)} tokens")
        self._load_get(self.tasks.pop(task_id), slot_mapping)
        return [task_id]


def start_cpu_workers(config) -> list[CpuTorchWorker]:
    workers = []
    effective_size = config.model.tp_size * config.model.cp_size
    for dp_rank in range(config.model.dp_size):
        for effective_rank in range(effective_size):
            device_id = dp_rank * effective_size + effective_rank
            workers.append(CpuTorchWorker(config, dp_rank, effective_rank, device_id))
    return workers
