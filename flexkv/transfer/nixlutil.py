"""NIXL helpers for FlexKV (FILE plugins: GDS_MT, POSIX, 3FS).

See https://github.com/ai-dynamo/nixl
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from flexkv.common.debug import flexkv_logger

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as e:
    nixl_agent = None  # type: ignore[misc, assignment]
    nixl_agent_config = None  # type: ignore[misc, assignment]
    _NIXL_IMPORT_ERROR = e
else:
    _NIXL_IMPORT_ERROR = None

NIXL_GPU_FILE_BACKENDS = frozenset({"GDS_MT"})
NIXL_CPU_FILE_BACKENDS = frozenset({"POSIX", "3FS"})

# AUTO: try in this order
_FILE_PLUGIN_ORDER = ("3FS", "POSIX", "GDS_MT")


def require_nixl() -> None:
    if nixl_agent is None or nixl_agent_config is None:
        raise ImportError(
            "NIXL is required for NixlTransferWorker. Install per "
            "https://github.com/ai-dynamo/nixl/blob/main/README.md"
        ) from _NIXL_IMPORT_ERROR


class NixlOpts:
    """Extra dict: flat keys, or nested ``plugin: { name: { ... } }``."""

    def __init__(self, raw: Optional[Dict[str, Any]] = None) -> None:
        self.raw: Dict[str, Any] = raw or {}

    def resolve_plugin(self) -> str:
        pl = self.raw.get("plugin")
        if isinstance(pl, dict):
            for name, body in pl.items():
                if isinstance(body, dict) and body.get("active") in (
                    True,
                    "true",
                    "True",
                ):
                    return str(name).upper()
        return os.environ.get("FLEXKV_NIXL_BACKEND_PLUGIN", "auto").upper()

    def init_params(self, plugin_name: str) -> Dict[str, str]:
        pl = self.raw.get("plugin")
        if isinstance(pl, dict):
            sec = pl.get(plugin_name.lower())
            sec = sec if isinstance(sec, dict) else {}
        else:
            sec = {k: v for k, v in self.raw.items() if k != "plugin"}
        return {k: str(v) for k, v in sec.items() if k != "active"}


def _pick_plugin(requested: str, available: List[str]) -> Optional[str]:
    u = requested.upper()
    if u == "AUTO":
        for p in _FILE_PLUGIN_ORDER:
            if p in available:
                return p
        return None
    return u if u in available else None


def _nixl_register(
    agent: Any,
    buffers: Union[List[Tuple[Any, ...]], torch.Tensor, List[torch.Tensor]],
    mem_type: Optional[str] = None,
) -> bool:
    if isinstance(buffers, list) and not buffers:
        return False
    descs = agent.get_reg_descs(buffers, mem_type)
    if descs is None:
        flexkv_logger.error("NIXL: get_reg_descs failed")
        return False
    try:
        agent.register_memory(descs)
        return True
    except Exception as e:
        flexkv_logger.error(f"NIXL register_memory failed: {e}")
        return False


def _open_xfer_files(
    file_paths: List[str],
    region_offsets: List[int],
    region_lens: List[int],
) -> Tuple[Optional[Dict[str, int]], List[Tuple[int, int, int]]]:
    path_to_fd: Dict[str, int] = {}
    xfer: List[Tuple[int, int, int]] = []
    for path, off, ln in zip(file_paths, region_offsets, region_lens):
        if path not in path_to_fd:
            try:
                path_to_fd[path] = os.open(path, os.O_RDWR)
            except OSError as e:
                flexkv_logger.error(f"NIXL: open {path} failed: {e}")
                for fd in path_to_fd.values():
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                return None, []
        xfer.append((off, ln, path_to_fd[path]))
    return path_to_fd, xfer


def _close_xfer_files(path_to_fd: Dict[str, int]) -> None:
    for fd in path_to_fd.values():
        try:
            os.close(fd)
        except OSError:
            pass


def _nixl_run_xfer(
    agent: Any,
    agent_name: str,
    direction: str,
    mem_side_descs: Any,
    file_xfer_tuples: List[Tuple[int, int, int]],
) -> bool:
    storage_descs = agent.get_xfer_descs(file_xfer_tuples, "FILE")
    if storage_descs is None or mem_side_descs is None:
        flexkv_logger.error("NIXL: get_xfer_descs returned None")
        return False
    try:
        req = agent.initialize_xfer(
            direction, mem_side_descs, storage_descs, agent_name
        )
    except Exception as e:
        flexkv_logger.error(f"NIXL initialize_xfer failed: {e}")
        return False
    try:
        state = agent.transfer(req)
        while state != "DONE":
            state = agent.check_xfer_state(req)
            if state == "ERR":
                agent.release_xfer_handle(req)
                flexkv_logger.error("NIXL transfer ERR")
                return False
            time.sleep(0.0001)
        agent.release_xfer_handle(req)
        return True
    except Exception as e:
        flexkv_logger.error(f"NIXL transfer failed: {e}")
        try:
            agent.release_xfer_handle(req)
        except Exception:
            pass
        return False


class NixlAgentSession:
    """nixl_agent + one FILE backend + xfer helpers."""

    def __init__(self, backend_plugin: str, extra_config: Optional[Dict[str, Any]]) -> None:
        require_nixl()
        self.agent_name = f"flexkv_nixl_{uuid.uuid4()}"
        opts = NixlOpts(extra_config)
        req = (
            backend_plugin
            if str(backend_plugin).upper() != "AUTO"
            else opts.resolve_plugin()
        )
        self.agent = nixl_agent(self.agent_name, nixl_agent_config(backends=[]))
        avail = list(self.agent.get_plugin_list())
        name = _pick_plugin(str(req).upper(), avail)
        if name is None:
            raise RuntimeError(
                f"NIXL: no plugin (requested={req}, available={avail})"
            )
        params = opts.init_params(name)
        try:
            self.agent.create_backend(name, params)
        except Exception as e:
            raise RuntimeError(
                f"NIXL: create_backend({name!r}) failed: {e}"
            ) from e
        self.backend_name = name
        flexkv_logger.info(f"NIXL backend {name} initparams={params}")

    def xfer_dram_file(
        self,
        direction: str,
        dram_ptr_len: List[Tuple[int, int]],
        file_paths: List[str],
        region_lens: List[int],
        region_offsets: List[int],
    ) -> bool:
        if not dram_ptr_len or not file_paths:
            return True
        path_to_fd, xfer_tuples = _open_xfer_files(
            file_paths, region_offsets, region_lens
        )
        if path_to_fd is None:
            return False
        try:
            reg_file = [(0, 0, path_to_fd[p], p) for p in path_to_fd]
            if not _nixl_register(self.agent, reg_file, "FILE"):
                return False
            dram_reg = [(p, ln, 0, "") for p, ln in dram_ptr_len]
            if not _nixl_register(self.agent, dram_reg, "DRAM"):
                return False
            dram_descs = self.agent.get_xfer_descs(
                [(p, ln, 0) for p, ln in dram_ptr_len], "DRAM"
            )
            return _nixl_run_xfer(
                self.agent,
                self.agent_name,
                direction,
                dram_descs,
                xfer_tuples,
            )
        finally:
            _close_xfer_files(path_to_fd)

    def xfer_vram_file(
        self,
        direction: str,
        gpu_tensors: List[torch.Tensor],
        file_paths: List[str],
        region_lens: List[int],
        region_offsets: List[int],
    ) -> bool:
        if not gpu_tensors or not file_paths:
            return True
        path_to_fd, xfer_tuples = _open_xfer_files(
            file_paths, region_offsets, region_lens
        )
        if path_to_fd is None:
            return False
        try:
            reg_file = [(0, 0, path_to_fd[p], p) for p in path_to_fd]
            if not _nixl_register(self.agent, reg_file, "FILE"):
                return False
            if not _nixl_register(self.agent, gpu_tensors):
                return False
            vram_descs = self.agent.get_xfer_descs(gpu_tensors)
            if vram_descs is None:
                flexkv_logger.error("NIXL: get_xfer_descs(VRAM) failed")
                return False
            return _nixl_run_xfer(
                self.agent,
                self.agent_name,
                direction,
                vram_descs,
                xfer_tuples,
            )
        finally:
            _close_xfer_files(path_to_fd)


def remap_ssd_block_id(
    ssd_block_id: int, num_devices: int, round_robin: int
) -> Tuple[int, int]:
    d = (ssd_block_id // round_robin) % num_devices
    block_in_dev = (
        (ssd_block_id // round_robin) // num_devices
    ) * round_robin + (ssd_block_id % round_robin)
    return d, block_in_dev


def file_path_for_ssd_block(
    ssd_files: Dict[int, List[str]],
    ssd_block_id: int,
    num_devices: int,
    num_files_per_device: int,
    round_robin: int,
) -> Tuple[str, int]:
    _, block_id_in_device = remap_ssd_block_id(
        ssd_block_id, num_devices, round_robin
    )
    file_idx = block_id_in_device % num_files_per_device
    block_in_file = block_id_in_device // num_files_per_device
    device_id = (ssd_block_id // round_robin) % num_devices
    path = ssd_files[device_id][file_idx]
    return path, block_in_file


def kv_chunk_byte_offset_in_block(
    layer_id: int,
    kv_idx: int,
    mem_block_id: int,
    layer_stride_b: int,
    kv_stride_b: int,
    block_stride_b: int,
    is_mla: bool,
) -> int:
    if is_mla:
        return mem_block_id * block_stride_b + layer_id * layer_stride_b
    return (
        mem_block_id * block_stride_b
        + layer_id * layer_stride_b
        + kv_idx * kv_stride_b
    )


def ssd_chunk_byte_offset_in_file(
    layer_id: int,
    kv_idx: int,
    block_in_file: int,
    ssd_layer_stride_b: int,
    ssd_kv_stride_b: int,
    block_stride_b: int,
    is_mla: bool,
) -> int:
    if is_mla:
        return block_in_file * block_stride_b + layer_id * ssd_layer_stride_b
    return (
        block_in_file * block_stride_b
        + layer_id * ssd_layer_stride_b
        + kv_idx * ssd_kv_stride_b
    )


def gpu_chunk_u8_view(
    gpu_blocks: List[torch.Tensor],
    gpu_block_type: int,
    num_layers: int,
    gpu_block_id: int,
    layer_id: int,
    kv_idx: int,
    gpu_kv_stride_b: int,
    gpu_block_stride_b: int,
    gpu_layer_stride_b: int,
    chunk_size_b: int,
    is_mla: bool,
) -> torch.Tensor:
    if is_mla:
        kv_idx = 0
    if gpu_block_type == 0:
        t = gpu_blocks[layer_id]
        off = kv_idx * gpu_kv_stride_b + gpu_block_id * gpu_block_stride_b
    elif gpu_block_type == 1:
        t = gpu_blocks[0]
        off = (
            gpu_block_id * gpu_block_stride_b
            + layer_id * gpu_layer_stride_b
            + kv_idx * gpu_kv_stride_b
        )
    elif gpu_block_type == 2:
        t = gpu_blocks[kv_idx * num_layers + layer_id]
        off = gpu_block_id * gpu_block_stride_b
    else:
        raise ValueError(f"Invalid gpu_block_type {gpu_block_type}")

    if off + chunk_size_b > t.numel() * t.element_size():
        raise ValueError("GPU chunk slice out of bounds for NIXL transfer")

    flat = t.view(torch.uint8).reshape(-1)
    return flat[off : off + chunk_size_b]
