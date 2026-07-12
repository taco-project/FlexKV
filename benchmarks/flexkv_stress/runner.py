from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import random
import time
import zlib

from .eventfd import EventfdGroup
from .gpu_worker import GpuWorker, command_dp, start_gpu_workers
from .metrics import OperationResult, TurnResult
from .workload import PrefixOracle, SlotAllocator, Turn, align_down


def _block_patterns(tokens, tokens_per_block: int) -> list[int]:
    import numpy as np
    values = np.asarray(tokens, dtype=np.int64)
    patterns = []
    for start in range(0, len(values), tokens_per_block):
        patterns.append(zlib.crc32(values[start:start + tokens_per_block].tobytes()) & 0x7FFFFFFF)
    return patterns


def build_flexkv_configs(config):
    import torch
    from flexkv.common.config import CacheConfig, LayerGroupSpec, ModelConfig, SWAPoolConfig

    from .device import torch_dtype

    groups = [
        LayerGroupSpec(
            num_layers=group.num_layers,
            num_kv_heads=group.num_kv_heads,
            head_size=group.head_size,
            layer_indices=list(group.layer_indices),
            sliding_window=group.sliding_window,
            dtype=torch_dtype(group.dtype),
            compress_ratio=group.compress_ratio,
        )
        for group in config.preset.groups
    ]
    model = ModelConfig(
        num_layers=config.preset.num_layers,
        num_kv_heads=config.preset.num_kv_heads,
        head_size=config.preset.head_size,
        use_mla=config.preset.use_mla,
        dtype=torch_dtype(config.model.dtype or config.preset.dtype),
        tp_size=config.kv_tp_size,
        dp_size=config.model.dp_size,
        cp_size=config.model.cp_size,
        layer_groups=groups,
    )
    cpu_blocks = config.cache.num_cpu_blocks or max(
        1, int(config.cache.cpu_cache_gb * 1024**3 / config.bytes_per_block)
    )
    ssd_blocks = config.cache.num_ssd_blocks
    if not ssd_blocks and config.cache.ssd_cache_gb > 0:
        ssd_blocks = max(1, int(config.cache.ssd_cache_gb * 1024**3 / config.bytes_per_block))
    if ssd_blocks and config.cache.ssd_cache_dir:
        devices = len(config.cache.ssd_cache_dir)
        ssd_blocks = ((ssd_blocks + devices - 1) // devices) * devices
    for directory in config.cache.ssd_cache_dir:
        Path(directory).mkdir(parents=True, exist_ok=True)
    swa = None
    if config.features.swa:
        swa = SWAPoolConfig(
            enabled=True,
            num_slots=config.cache.swa_num_slots,
            num_ssd_slots=config.cache.swa_ssd_slots if config.features.ssd else 0,
            num_swa_layers=config.preset.swa_num_layers,
            bytes_per_token_per_layer=config.preset.swa_head_size,
        )
    cache = CacheConfig(
        tokens_per_block=config.tokens_per_block,
        eviction_policy=config.cache.eviction_policy,
        enable_cpu=True,
        enable_ssd=config.features.ssd,
        enable_gds=config.cache.enable_gds,
        num_cpu_blocks=cpu_blocks,
        num_ssd_blocks=ssd_blocks,
        ssd_cache_dir=config.cache.ssd_cache_dir,
        swa=swa,
        enable_swa_transfer=config.features.swa,
    )
    return model, cache


def _socket_path(base: str, dp_rank: int, dp_size: int) -> str:
    if dp_size == 1:
        return base
    root, extension = os.path.splitext(base)
    return f"{root}_dp{dp_rank}{extension}"


class FlexKVRuntime:
    def __init__(self, config):
        self.config = config
        self.managers = []
        self.workers: list[GpuWorker] = []
        self.eventfds: dict[int, EventfdGroup] = {}
        self.model_config = None
        self.cache_config = None

    def start(self) -> None:
        import torch
        if self.config.features.cpu_stub:
            from types import SimpleNamespace

            from .cpu_stub import CpuStubManager, start_cpu_workers

            cpu_blocks = self.config.cache.num_cpu_blocks or max(
                1,
                int(self.config.cache.cpu_cache_gb * 1024**3 / self.config.bytes_per_block),
            )
            self.model_config = None
            self.cache_config = SimpleNamespace(
                num_cpu_blocks=cpu_blocks,
                num_ssd_blocks=0,
            )
            self.workers = start_cpu_workers(self.config)
            capacity = max(
                1,
                int(
                    (self.cache_config.num_cpu_blocks + self.cache_config.num_ssd_blocks)
                    / self.config.model.dp_size
                ),
            )
            self.managers = [
                CpuStubManager(self.config, dp_rank, self.workers, capacity)
                for dp_rank in range(self.config.model.dp_size)
            ]
            for manager in self.managers:
                manager.start()
            logging.getLogger("flexkv_stress").warning(
                "CPU stub mode: exercising stress control flow and PyTorch byte copies; "
                "CUDA/ROCm IPC, native transfers, SSD, and eventfd signaling are not under test"
            )
            return
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA/ROCm device is available; use --dry-run in this environment")
        if torch.cuda.device_count() < self.config.required_gpus:
            raise RuntimeError(
                f"Need {self.config.required_gpus} GPUs for composite-TP×DP "
                f"(kv_tp={self.config.kv_tp_size}, cp={self.config.model.cp_size}), "
                f"found {torch.cuda.device_count()}"
            )
        base_socket = os.environ.setdefault(
            "FLEXKV_LAYERWISE_EVENTFD_SOCKET",
            f"/tmp/flexkv_stress_eventfd_{os.getpid()}.sock",
        )
        os.environ["FLEXKV_ENABLE_LAYERWISE_TRANSFER"] = "1" if self.config.features.layerwise else "0"
        os.environ.setdefault("FLEXKV_SERVER_RECV_PORT", f"ipc:///tmp/flexkv_stress_server_{os.getpid()}")

        from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
        from flexkv.kvmanager import KVManager
        GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = self.config.features.layerwise
        GLOBAL_CONFIG_FROM_ENV.server_recv_port = os.environ["FLEXKV_SERVER_RECV_PORT"]
        self.model_config, self.cache_config = build_flexkv_configs(self.config)

        if self.config.features.layerwise:
            effective_size = self.config.workers_per_dp
            for dp_rank in range(self.config.model.dp_size):
                group = EventfdGroup(
                    _socket_path(base_socket, dp_rank, self.config.model.dp_size),
                    effective_size,
                    self.config.preset.num_layers,
                )
                group.start()
                self.eventfds[dp_rank] = group

        for dp_rank in range(self.config.model.dp_size):
            manager = KVManager(self.model_config, self.cache_config, dp_client_id=dp_rank)
            manager.start()
            self.managers.append(manager)
        self.workers = start_gpu_workers(self.config, self.managers[0].gpu_register_port)

        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if all(manager.is_ready() for manager in self.managers):
                break
            time.sleep(0.2)
        else:
            raise TimeoutError("FlexKV managers did not become ready in 180 seconds")
        for group in self.eventfds.values():
            group.wait_ready()

    def close(self) -> None:
        for manager in reversed(self.managers):
            try:
                manager.shutdown()
            except Exception:
                pass
        for worker in self.workers:
            worker.shutdown()
        for group in self.eventfds.values():
            group.close()


@dataclass
class _Launch:
    task_id: int
    slots: object
    swa_slots: object | None = None


class ManagerDriver:
    def __init__(self, config, dp_rank: int, manager, workers: list[GpuWorker], eventfds=None,
                 oracle_capacity_blocks: int = 0):
        self.config = config
        self.dp_rank = dp_rank
        self.manager = manager
        self.workers = workers
        self.eventfds = eventfds
        self.oracle = PrefixOracle(config.tokens_per_block, oracle_capacity_blocks)
        self.slots = SlotAllocator(config.model.gpu_blocks_per_rank, config.tokens_per_block)
        self.swa_slots = SlotAllocator(config.cache.swa_num_slots, config.tokens_per_block)
        self.counter_id = 0
        self.last_stored: tuple[object, list[int]] | None = None

    def _operation(self, operations: list[OperationResult], name: str, started: float,
                   success: bool, tokens: int = 0, hit_tokens: int = 0,
                   transfer_bytes: int = 0, error: str = "") -> None:
        operations.append(OperationResult(
            name, success, (time.perf_counter() - started) * 1000,
            tokens=tokens, hit_tokens=hit_tokens,
            transfer_bytes=transfer_bytes, error=error,
        ))

    def _swa_mapping(self):
        blocks, slots = self.swa_slots.allocate(self.config.tokens_per_block)
        return blocks.tolist(), slots

    def _launch_bytes(self, pending: list[_Launch]) -> int:
        import numpy as np

        tpb = self.config.tokens_per_block
        main_blocks = sum(
            len(np.unique(np.asarray(item.slots, dtype=np.int64).reshape(-1) // tpb))
            for item in pending
        )
        swa_blocks = sum(
            len(np.unique(np.asarray(item.swa_slots, dtype=np.int64).reshape(-1) // tpb))
            for item in pending if item.swa_slots is not None
        )
        return (
            main_blocks * self.config.bytes_per_block
            + swa_blocks * self.config.swa_bytes_per_block
        )

    def _launch(self, pending: list[_Launch], operation: str,
                operations: list[OperationResult], layerwise: bool = False) -> bool:
        if not pending:
            return True
        started = time.perf_counter()
        transfer_bytes = self._launch_bytes(pending)
        try:
            as_batch = len(pending) > 1 or layerwise
            returned = self.manager.launch(
                task_ids=[item.task_id for item in pending],
                slot_mappings=[item.slots for item in pending],
                swa_slot_mappings=[item.swa_slots for item in pending] if self.config.features.swa else None,
                as_batch=as_batch,
                layerwise_transfer=layerwise,
                counter_id=self.counter_id,
            )
            responses = self.manager.wait(
                returned,
                timeout=self.config.concurrency.request_timeout_seconds,
                completely=True,
            )
            success = bool(responses) and all(
                getattr(response.status, "name", str(response.status)) == "SUCCESS"
                for response in responses.values()
            )
            if layerwise:
                values = self.eventfds.read_counter(self.counter_id) if self.eventfds else []
                success = success and (not values or all(sum(rank) > 0 for rank in values))
                self.counter_id = (self.counter_id + 1) % 3
            self._operation(
                operations, operation, started, success,
                transfer_bytes=transfer_bytes,
            )
            return success
        except Exception as exc:
            self._operation(
                operations, operation, started, False,
                transfer_bytes=transfer_bytes, error=str(exc),
            )
            return False

    def _match_get(self, tokens, operations: list[OperationResult]):
        import numpy as np
        started = time.perf_counter()
        try:
            result = self.manager.get_match(
                np.asarray(tokens, dtype=np.int64),
                swa_aware=self.config.features.swa,
            )
            if result is None:
                raise RuntimeError("get_match returned None")
            task_id, mask = result
            hit_tokens = int(mask.sum())
            self._operation(operations, "get_match", started, True, len(tokens), hit_tokens)
            return task_id, mask, hit_tokens
        except Exception as exc:
            self._operation(operations, "get_match", started, False, len(tokens), error=str(exc))
            return -1, None, 0

    def execute_batch(self, turns: list[Turn], round_id: int,
                      sample_keys: set[tuple[int, int]]) -> tuple[list[TurnResult], list[OperationResult]]:
        import numpy as np
        operations: list[OperationResult] = []
        states = []

        # GET the prefix available before this turn, matching SGLang's load phase.
        get_pending = []
        for turn in turns:
            expected = self.oracle.match(turn.get_tokens)
            task_id, mask, actual = self._match_get(turn.get_tokens, operations)
            state = {
                "turn": turn, "expected": expected, "actual": actual,
                "match_ok": (
                    task_id >= 0
                    and abs(actual - expected)
                    <= self.config.validation.hit_tolerance_blocks * self.config.tokens_per_block
                ),
                "get_ok": True, "put_ok": True, "put_unmatched": 0,
                "match_ms": operations[-1].latency_ms, "get_ms": 0.0, "put_ms": 0.0,
                "sample": (turn.conversation_id, turn.turn_id) in sample_keys,
                "validation_ok": True,
            }
            if task_id >= 0 and actual:
                blocks, slots = self.slots.allocate(actual)
                swa_mapping = None
                if self.config.features.swa:
                    _swa_blocks, swa_mapping = self._swa_mapping()
                get_pending.append(_Launch(task_id, slots, swa_mapping))
            states.append(state)
        get_started = time.perf_counter()
        get_ok = self._launch(
            get_pending, "launch_get", operations,
            layerwise=self.config.features.layerwise,
        )
        get_ms = (time.perf_counter() - get_started) * 1000
        for state in states:
            state["get_ok"] = get_ok
            state["get_ms"] = get_ms

        # Simulate inference writes, then PUT only the unmatched slots.
        put_pending = []
        put_metadata = []
        for state in states:
            turn = state["turn"]
            aligned = align_down(len(turn.put_tokens), self.config.tokens_per_block)
            tokens = turn.put_tokens[:aligned]
            blocks, slots = self.slots.allocate(aligned)
            patterns = _block_patterns(tokens, self.config.tokens_per_block)
            swa_mapping = None
            swa_blocks = []
            if self.config.features.swa:
                swa_blocks, swa_mapping = self._swa_mapping()
            command_dp(self.workers, self.dp_rank, {
                "operation": "seed", "block_ids": blocks.tolist(), "patterns": patterns,
                "swa_block_ids": swa_blocks, "swa_patterns": patterns[-1:],
            })
            started = time.perf_counter()
            state["save_started"] = started
            try:
                result = self.manager.put_match(np.asarray(tokens, dtype=np.int64))
                if result is None:
                    raise RuntimeError("put_match returned None")
                task_id, unmatched = result
                unmatched = np.asarray(unmatched, dtype=bool)
                state["put_unmatched"] = int(unmatched.sum())
                self._operation(operations, "put_match", started, True, aligned, 0)
                if unmatched.any():
                    put_pending.append(_Launch(task_id, slots[unmatched], swa_mapping))
                put_metadata.append((state, tokens, patterns))
            except Exception as exc:
                state["put_ok"] = False
                self._operation(operations, "put_match", started, False, aligned, error=str(exc))
        put_ok = self._launch(put_pending, "launch_put", operations)
        for state, tokens, _patterns in put_metadata:
            state["put_ok"] = state["put_ok"] and put_ok
            # Service save latency starts at put_match and ends after launch_put/wait.
            state["put_ms"] = (time.perf_counter() - state["save_started"]) * 1000
            if state["put_ok"]:
                self.oracle.put(tokens)
                self.last_stored = (tokens, _patterns)

        # Immediate read-back catches transfer corruption; sampling controls byte verification.
        if self.config.conversation.read_after_put:
            read_pending = []
            read_metadata = []
            for state, tokens, patterns in put_metadata:
                task_id, _mask, actual = self._match_get(tokens, operations)
                if task_id < 0 or actual == 0:
                    state["get_ok"] = False
                    continue
                blocks, slots = self.slots.allocate(actual)
                actual_patterns = patterns[:len(blocks)]
                swa_mapping = None
                swa_blocks = []
                if self.config.features.swa:
                    swa_blocks, swa_mapping = self._swa_mapping()
                if state["sample"]:
                    command_dp(self.workers, self.dp_rank, {
                        "operation": "zero", "block_ids": blocks.tolist(),
                        "swa_block_ids": swa_blocks,
                        "max_layers": self.config.validation.sampled_layers_per_group,
                    })
                read_pending.append(_Launch(task_id, slots, swa_mapping))
                read_metadata.append((state, blocks.tolist(), actual_patterns, swa_blocks))
            read_ok = self._launch(
                read_pending, "launch_readback", operations,
                layerwise=self.config.features.layerwise,
            )
            for state, blocks, patterns, swa_blocks in read_metadata:
                state["get_ok"] = state["get_ok"] and read_ok
                if state["sample"] and read_ok:
                    verify_started = time.perf_counter()
                    responses = command_dp(self.workers, self.dp_rank, {
                        "operation": "verify",
                        "block_ids": blocks[:self.config.validation.sampled_blocks_per_request],
                        "patterns": patterns[:self.config.validation.sampled_blocks_per_request],
                        "swa_block_ids": swa_blocks, "swa_patterns": patterns[-1:],
                        "max_layers": self.config.validation.sampled_layers_per_group,
                        "max_bytes": self.config.validation.bytes_per_sample,
                    })
                    state["validation_ok"] = all(response.get("ok", False) for response in responses)
                    errors = [error for response in responses for error in response.get("errors", [])]
                    self._operation(
                        operations,
                        "validate_readback",
                        verify_started,
                        state["validation_ok"],
                        error="; ".join(errors[:5]),
                    )

        results = []
        for state in states:
            turn = state["turn"]
            success = state["match_ok"] and state["get_ok"] and state["put_ok"] and state["validation_ok"]
            results.append(TurnResult(
                round_id=round_id,
                conversation_id=turn.conversation_id,
                turn_id=turn.turn_id,
                added_input_tokens=turn.added_input_tokens,
                output_tokens=turn.output_tokens,
                total_tokens=len(turn.put_tokens),
                expected_hit_tokens=state["expected"],
                actual_hit_tokens=state["actual"],
                hit_delta_tokens=state["actual"] - state["expected"],
                put_unmatched_tokens=state["put_unmatched"],
                match_ms=state["match_ms"], get_ms=state["get_ms"], put_ms=state["put_ms"],
                success=success,
                validation_sampled=state["sample"] and self.config.conversation.read_after_put,
                validation_ok=state["validation_ok"],
                query_tokens=len(turn.get_tokens),
            ))
        return results, operations

    def async_probe(self, round_id: int) -> list[OperationResult]:
        """Exercise KVManager's direct put/get APIs with a small unique sequence."""
        import numpy as np
        operations: list[OperationResult] = []
        length = self.config.tokens_per_block * 2
        tokens = np.arange(length, dtype=np.int64) + (round_id + 1) * 10_000_000 + self.dp_rank * 100_000
        patterns = _block_patterns(tokens, self.config.tokens_per_block)
        source_blocks, source_slots = self.slots.allocate(length)
        command_dp(self.workers, self.dp_rank, {
            "operation": "seed", "block_ids": source_blocks.tolist(), "patterns": patterns,
        })
        started = time.perf_counter()
        try:
            put_task = self.manager.put_async(tokens, source_slots)
            put_response = self.manager.wait(
                put_task, timeout=self.config.concurrency.request_timeout_seconds, completely=True
            )
            put_ok = bool(put_response) and all(
                getattr(value.status, "name", str(value.status)) == "SUCCESS"
                for value in put_response.values()
            )
            self._operation(
                operations, "put_async", started, put_ok, tokens=length,
                transfer_bytes=(length // self.config.tokens_per_block) * self.config.bytes_per_block,
            )
        except Exception as exc:
            self._operation(operations, "put_async", started, False, tokens=length, error=str(exc))
            return operations

        target_blocks, target_slots = self.slots.allocate(length)
        command_dp(self.workers, self.dp_rank, {
            "operation": "zero", "block_ids": target_blocks.tolist(),
        })
        started = time.perf_counter()
        try:
            get_task = self.manager.get_async(tokens, target_slots)
            get_response = self.manager.wait(
                get_task, timeout=self.config.concurrency.request_timeout_seconds, completely=True
            )
            get_ok = bool(get_response) and all(
                getattr(value.status, "name", str(value.status)) == "SUCCESS"
                for value in get_response.values()
            )
            self._operation(
                operations, "get_async", started, get_ok, tokens=length,
                hit_tokens=length,
                transfer_bytes=(length // self.config.tokens_per_block) * self.config.bytes_per_block,
            )
            if get_ok:
                verified = command_dp(self.workers, self.dp_rank, {
                    "operation": "verify", "block_ids": target_blocks.tolist(), "patterns": patterns,
                    "max_layers": self.config.validation.sampled_layers_per_group,
                    "max_bytes": self.config.validation.bytes_per_sample,
                })
                if not all(response.get("ok", False) for response in verified):
                    operations[-1].success = False
                    errors = [error for response in verified for error in response.get("errors", [])]
                    operations[-1].error = "async GET byte verification failed: " + "; ".join(errors[:5])
        except Exception as exc:
            self._operation(operations, "get_async", started, False, tokens=length, error=str(exc))
        return operations

    def ssd_probe(self) -> list[OperationResult]:
        """Force a direct-mode CPU miss and verify SSD -> CPU -> GPU reload."""
        import numpy as np
        operations: list[OperationResult] = []
        if self.last_stored is None:
            return operations
        tokens, patterns = self.last_stored
        started = time.perf_counter()
        try:
            self.manager._clear_cpu_cache()
            self._operation(operations, "clear_cpu_for_ssd_probe", started, True)
            task_id, _mask, actual = self._match_get(tokens, operations)
            if task_id < 0 or actual != len(tokens):
                raise RuntimeError(f"SSD probe expected {len(tokens)} hit tokens, got {actual}")
            blocks, slots = self.slots.allocate(actual)
            swa_mapping = None
            swa_blocks = []
            if self.config.features.swa:
                swa_blocks, swa_mapping = self._swa_mapping()
            command_dp(self.workers, self.dp_rank, {
                "operation": "zero", "block_ids": blocks.tolist(),
                "swa_block_ids": swa_blocks,
                "max_layers": self.config.validation.sampled_layers_per_group,
            })
            ok = self._launch(
                [_Launch(task_id, slots, swa_mapping)],
                "ssd_reload_get",
                operations,
                layerwise=self.config.features.layerwise,
            )
            if ok:
                responses = command_dp(self.workers, self.dp_rank, {
                    "operation": "verify",
                    "block_ids": blocks[:self.config.validation.sampled_blocks_per_request].tolist(),
                    "patterns": patterns[:self.config.validation.sampled_blocks_per_request],
                    "swa_block_ids": swa_blocks, "swa_patterns": patterns[-1:],
                    "max_layers": self.config.validation.sampled_layers_per_group,
                    "max_bytes": self.config.validation.bytes_per_sample,
                })
                if not all(response.get("ok", False) for response in responses):
                    operations[-1].success = False
                    errors = [error for response in responses for error in response.get("errors", [])]
                    operations[-1].error = "SSD reload byte verification failed: " + "; ".join(errors[:5])
        except Exception as exc:
            self._operation(operations, "ssd_probe", started, False, error=str(exc))
        return operations


class StressRunner:
    def __init__(self, config, runtime: FlexKVRuntime):
        self.config = config
        self.runtime = runtime
        self.drivers = [
            ManagerDriver(
                config, dp_rank, runtime.managers[dp_rank], runtime.workers,
                runtime.eventfds.get(dp_rank),
                max(
                    1,
                    int(
                        (runtime.cache_config.num_cpu_blocks + runtime.cache_config.num_ssd_blocks)
                        / config.model.dp_size
                    ),
                ),
            )
            for dp_rank in range(config.model.dp_size)
        ]
        self.rng = random.Random(config.run.seed ^ 0xF1E5)

    def _sample_keys(self, turns: list[Turn]) -> set[tuple[int, int]]:
        sampled = {
            (turn.conversation_id, turn.turn_id)
            for turn in turns if self.rng.random() < self.config.validation.sample_rate
        }
        needed = min(self.config.validation.min_samples_per_round, len(turns)) - len(sampled)
        if needed > 0:
            remaining = [turn for turn in turns if (turn.conversation_id, turn.turn_id) not in sampled]
            for turn in self.rng.sample(remaining, needed):
                sampled.add((turn.conversation_id, turn.turn_id))
        return sampled

    def run_round(self, conversations, round_id: int) -> tuple[list[TurnResult], list[OperationResult]]:
        all_turns = [turn for conversation in conversations for turn in conversation.turns]
        sample_keys = self._sample_keys(all_turns)
        results: list[TurnResult] = []
        operations: list[OperationResult] = []
        max_turns = max((len(conversation.turns) for conversation in conversations), default=0)
        for turn_id in range(max_turns):
            by_dp = [[] for _ in self.drivers]
            for conversation in conversations:
                if turn_id < len(conversation.turns):
                    dp_rank = conversation.conversation_id % len(self.drivers)
                    by_dp[dp_rank].append(conversation.turns[turn_id])

            def run_dp(dp_rank: int):
                dp_results, dp_operations = [], []
                batch_size = (
                    min(self.config.concurrency.batch_size, self.config.concurrency.max_inflight_per_dp)
                    if self.config.features.concurrency else 1
                )
                for start in range(0, len(by_dp[dp_rank]), batch_size):
                    batch = by_dp[dp_rank][start:start + batch_size]
                    batch_started = time.perf_counter()
                    batch_results, batch_operations = self.drivers[dp_rank].execute_batch(
                        batch, round_id, sample_keys
                    )
                    dp_results.extend(batch_results)
                    dp_operations.extend(batch_operations)
                    if self.config.concurrency.target_qps > 0:
                        target_seconds = len(batch) / self.config.concurrency.target_qps
                        time.sleep(max(0, target_seconds - (time.perf_counter() - batch_started)))
                return dp_results, dp_operations

            active = [rank for rank, turns in enumerate(by_dp) if turns]
            if self.config.features.concurrency and len(active) > 1:
                with ThreadPoolExecutor(max_workers=len(active)) as executor:
                    futures = [executor.submit(run_dp, rank) for rank in active]
                    for future in futures:
                        batch_results, batch_operations = future.result()
                        results.extend(batch_results)
                        operations.extend(batch_operations)
            else:
                for rank in active:
                    batch_results, batch_operations = run_dp(rank)
                    results.extend(batch_results)
                    operations.extend(batch_operations)
        interval = self.config.features.async_api_probe_interval_rounds
        if (
            interval > 0
            and round_id % interval == 0
            and not self.config.features.swa
            and not self.config.features.layerwise
        ):
            for driver in self.drivers:
                operations.extend(driver.async_probe(round_id))
        ssd_interval = self.config.cache.force_ssd_reload_interval_rounds
        if (
            self.config.features.ssd
            and self.config.model.dp_size == 1
            and ssd_interval > 0
            and round_id % ssd_interval == 0
        ):
            operations.extend(self.drivers[0].ssd_probe())
        return results, operations
