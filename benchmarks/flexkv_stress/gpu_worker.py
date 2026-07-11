from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import traceback
from typing import Any


def _value(base: int, rank: int, group_id: int, layer_id: int) -> int:
    return 1 + (base + rank * 17 + group_id * 29 + layer_id * 7) % 127


def _fill_blocks(tensors, block_ids: list[int], patterns: list[int], rank: int, group_id: int) -> None:
    for layer_id, tensor in enumerate(tensors):
        for block_id, pattern in zip(block_ids, patterns):
            tensor[block_id].fill_(_value(pattern, rank, group_id, layer_id))


def _zero_blocks(tensors, block_ids: list[int], max_layers: int | None = None) -> None:
    selected = tensors if max_layers is None else tensors[:max_layers]
    for tensor in selected:
        for block_id in block_ids:
            tensor[block_id].zero_()


def _verify_blocks(tensors, block_ids: list[int], patterns: list[int], rank: int,
                   group_id: int, max_layers: int, max_bytes: int) -> list[str]:
    import torch
    errors = []
    for layer_id, tensor in enumerate(tensors[:max_layers]):
        for block_id, pattern in zip(block_ids, patterns):
            expected = _value(pattern, rank, group_id, layer_id)
            actual = tensor[block_id].reshape(-1)
            itemsize = max(1, actual.element_size())
            actual = actual[:max(1, max_bytes // itemsize)]
            if not bool(torch.all(actual == expected)):
                errors.append(
                    f"rank={rank} group={group_id} layer={layer_id} block={block_id} expected={expected}"
                )
    return errors


def _worker_main(connection, config, dp_rank: int, effective_rank: int, device_id: int,
                 gpu_register_port: str) -> None:
    try:
        import torch
        from flexkv.common.config import LayerGroupSpec
        from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
        from flexkv.server.client import KVTPClient

        from .device import DeviceBackend, torch_dtype

        backend = DeviceBackend.detect(torch)
        backend.set_device(device_id)
        groups = config.preset.groups
        num_blocks = config.model.gpu_blocks_per_rank
        layer_specs = []
        layouts = []
        tensors_per_group = []
        for group in groups:
            group_tpb = config.tokens_per_block // group.compress_ratio
            spec = LayerGroupSpec(
                num_layers=group.num_layers,
                num_kv_heads=group.num_kv_heads,
                head_size=group.head_size,
                layer_indices=list(group.layer_indices),
                sliding_window=group.sliding_window,
                dtype=torch_dtype(group.dtype),
                compress_ratio=group.compress_ratio,
            )
            layout = KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=group.num_layers,
                num_block=num_blocks,
                tokens_per_block=group_tpb,
                num_head=group.num_kv_heads,
                head_size=group.head_size,
                is_mla=True,
            )
            tensors = [
                backend.empty(
                    (num_blocks, group_tpb, group.num_kv_heads, group.head_size),
                    torch_dtype(group.dtype),
                    device_id,
                    fill_zero=True,
                )
                for _ in range(group.num_layers)
            ]
            layer_specs.append(spec)
            layouts.append(layout)
            tensors_per_group.append(tensors)

        swa_tensors = None
        swa_layout = None
        if config.features.swa:
            swa_slots = config.cache.swa_num_slots
            swa_layout = KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=config.preset.swa_num_layers,
                num_block=swa_slots,
                tokens_per_block=config.tokens_per_block,
                num_head=1,
                head_size=config.preset.swa_head_size,
                is_mla=True,
            )
            swa_tensors = [
                backend.empty(
                    (swa_slots, config.tokens_per_block, 1, config.preset.swa_head_size),
                    torch.uint8,
                    device_id,
                    fill_zero=True,
                )
                for _ in range(config.preset.swa_num_layers)
            ]

        tp_client = KVTPClient(gpu_register_port, dp_rank, device_id, pp_rank=0)
        flat_tensors = [tensor for group in tensors_per_group for tensor in group]
        tp_client.register_to_server(
            kv_caches=flat_tensors,
            kv_layout=layouts[0],
            layer_groups=layer_specs,
            gpu_layouts=layouts,
            handles_per_group=tensors_per_group,
            swa_caches=swa_tensors,
            swa_layout=swa_layout,
        )
        backend.synchronize(device_id)
        connection.send({
            "ok": True,
            "backend": backend.name,
            "device_id": device_id,
            "groups": [tuple(tensor.shape) for tensor in (group[0] for group in tensors_per_group)],
        })

        while True:
            command = connection.recv()
            operation = command["operation"]
            if operation == "shutdown":
                connection.send({"ok": True})
                break
            block_ids = [int(v) for v in command.get("block_ids", [])]
            patterns = [int(v) for v in command.get("patterns", [])]
            if operation == "seed":
                for group_id, tensors in enumerate(tensors_per_group):
                    _fill_blocks(tensors, block_ids, patterns, effective_rank, group_id)
                if swa_tensors is not None and command.get("swa_block_ids"):
                    _fill_blocks(
                        swa_tensors,
                        [int(v) for v in command["swa_block_ids"]],
                        [patterns[-1]],
                        effective_rank,
                        len(tensors_per_group),
                    )
                backend.synchronize(device_id)
                connection.send({"ok": True})
            elif operation == "zero":
                max_layers = command.get("max_layers")
                for tensors in tensors_per_group:
                    _zero_blocks(tensors, block_ids, max_layers)
                if swa_tensors is not None and command.get("swa_block_ids"):
                    _zero_blocks(
                        swa_tensors, [int(v) for v in command["swa_block_ids"]], max_layers
                    )
                backend.synchronize(device_id)
                connection.send({"ok": True})
            elif operation == "verify":
                errors = []
                max_layers = int(command.get("max_layers", 2))
                max_bytes = int(command.get("max_bytes", 256))
                for group_id, tensors in enumerate(tensors_per_group):
                    errors.extend(_verify_blocks(
                        tensors, block_ids, patterns, effective_rank, group_id, max_layers, max_bytes
                    ))
                if swa_tensors is not None and command.get("swa_block_ids"):
                    errors.extend(_verify_blocks(
                        swa_tensors,
                        [int(v) for v in command["swa_block_ids"]],
                        [patterns[-1]],
                        effective_rank,
                        len(tensors_per_group),
                        max_layers,
                        max_bytes,
                    ))
                connection.send({"ok": not errors, "errors": errors[:20]})
            elif operation == "memory":
                import resource
                rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                # Linux reports KiB; GPU execution environments for FlexKV are Linux.
                connection.send({"ok": True, **backend.memory(device_id), "rss_bytes": rss * 1024})
            else:
                raise ValueError(f"Unknown GPU worker operation {operation!r}")
    except BaseException as exc:
        try:
            connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        except Exception:
            pass
        raise


@dataclass
class GpuWorker:
    dp_rank: int
    effective_rank: int
    device_id: int
    process: mp.Process
    connection: Any

    def request(self, command: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
        if not self.process.is_alive():
            raise RuntimeError(f"GPU worker {self.device_id} exited with code {self.process.exitcode}")
        self.connection.send(command)
        if not self.connection.poll(timeout):
            raise TimeoutError(f"GPU worker {self.device_id} timed out on {command['operation']}")
        response = self.connection.recv()
        if not response.get("ok", False) and command["operation"] != "verify":
            raise RuntimeError(response.get("error", str(response)))
        return response

    def shutdown(self) -> None:
        if self.process.is_alive():
            try:
                self.request({"operation": "shutdown"}, timeout=10)
            except Exception:
                pass
            self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)


def start_gpu_workers(config, gpu_register_port: str) -> list[GpuWorker]:
    context = mp.get_context("spawn")
    workers = []
    effective_size = config.model.tp_size * config.model.cp_size
    for dp_rank in range(config.model.dp_size):
        for effective_rank in range(effective_size):
            device_id = dp_rank * effective_size + effective_rank
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(child, config, dp_rank, effective_rank, device_id, gpu_register_port),
                daemon=False,
            )
            process.start()
            worker = GpuWorker(dp_rank, effective_rank, device_id, process, parent)
            if not parent.poll(180):
                worker.shutdown()
                raise TimeoutError(f"GPU worker {device_id} did not register in 180 seconds")
            response = parent.recv()
            if not response.get("ok"):
                worker.shutdown()
                raise RuntimeError(response.get("traceback", response.get("error", str(response))))
            workers.append(worker)
    return workers


def command_dp(workers: list[GpuWorker], dp_rank: int, command: dict[str, Any],
               timeout: float = 60) -> list[dict[str, Any]]:
    return [worker.request(command, timeout) for worker in workers if worker.dp_rank == dp_rank]
