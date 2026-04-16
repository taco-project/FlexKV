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
# Upstream NIXL registers 3FS as plugin "HF3FS"; "3FS" is accepted as an alias.
NIXL_CPU_FILE_BACKENDS = frozenset({"POSIX", "HF3FS", "3FS"})

# AUTO: try in this order (HF3FS is the real plugin name in nixl.git)
_FILE_PLUGIN_ORDER = ("HF3FS", "POSIX", "GDS_MT")


def normalize_nixl_file_plugin_name(name: str) -> str:
    """Map user-facing names to NIXL create_backend plugin names."""
    u = str(name).upper()
    if u == "3FS":
        return "HF3FS"
    return u


def build_nixl_extra_config(cache_config: Any) -> Dict[str, Any]:
    """Merge ``CacheConfig.nixl_*`` HF3FS fields into ``nixl_extra_config`` for ``NixlAgentSession``."""
    out: Dict[str, Any] = dict(getattr(cache_config, "nixl_extra_config", None) or {})
    mp = getattr(cache_config, "nixl_hf3fs_mount_point", None)
    mc = getattr(cache_config, "nixl_hf3fs_mem_config", None)
    ios = getattr(cache_config, "nixl_hf3fs_iopool_size", None)
    if mp is None and mc is None and ios is None:
        return out
    pl = out.get("plugin")
    if not isinstance(pl, dict):
        pl = {}
        out["plugin"] = pl
    hf = pl.get("hf3fs")
    if not isinstance(hf, dict):
        hf = {}
        pl["hf3fs"] = hf
    if mp is not None and "mount_point" not in hf:
        hf["mount_point"] = str(mp)
    if mc is not None and "mem_config" not in hf:
        hf["mem_config"] = str(mc)
    if ios is not None and "iopool_size" not in hf:
        hf["iopool_size"] = str(int(ios))
    # Flat keys for NixlOpts.init_params HF3FS fallback
    if mp is not None and "mount_point" not in out:
        out["mount_point"] = str(mp)
    if mc is not None and "mem_config" not in out:
        out["mem_config"] = str(mc)
    if ios is not None and "iopool_size" not in out:
        out["iopool_size"] = str(int(ios))
    return out


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
                    return normalize_nixl_file_plugin_name(str(name).upper())
        return normalize_nixl_file_plugin_name(
            os.environ.get("FLEXKV_NIXL_BACKEND_PLUGIN", "auto").upper()
        )

    def init_params(self, plugin_name: str) -> Dict[str, str]:
        pl = self.raw.get("plugin")
        if isinstance(pl, dict):
            sec = pl.get(plugin_name.lower())
            sec = sec if isinstance(sec, dict) else {}
        else:
            sec = {k: v for k, v in self.raw.items() if k != "plugin"}
        base = {k: str(v) for k, v in sec.items() if k != "active"}
        # HF3FS: allow flat mount_point / mem_config / iopool_size on the merged config dict.
        if plugin_name.upper() == "HF3FS":
            for k in ("mount_point", "mem_config", "iopool_size"):
                if k not in base and k in self.raw:
                    base[k] = str(self.raw[k])
        return base


def _pick_plugin(requested: str, available: List[str]) -> Optional[str]:
    avail = set(available)
    u = normalize_nixl_file_plugin_name(requested)
    if u == "AUTO":
        for p in _FILE_PLUGIN_ORDER:
            if p in avail:
                return p
        return None
    return u if u in avail else None


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
    flexkv_logger.info(
        f"NIXL: _nixl_run_xfer (direction={direction}, agent={agent_name}, "
        f"file_regions={len(file_xfer_tuples)})"
    )
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
        raw_req = str(backend_plugin).upper()
        if raw_req != "AUTO":
            req = normalize_nixl_file_plugin_name(backend_plugin)
        else:
            req = opts.resolve_plugin()
            if str(req).upper() == "3FS":
                req = "HF3FS"
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
        self._nixl_extra_config: Dict[str, Any] = dict(extra_config or {})
        flexkv_logger.info(f"NIXL backend {name} initparams={params}")
        # Opened at prepare_all_ssd_files; DRAM/VRAM registered at prepare_dram_cpu /
        # prepare_vram_gpu. xfer_* only runs initialize_xfer + transfer.
        self._path_to_fd: Optional[Dict[str, int]] = None

    def prepare_all_ssd_files(self, ssd_files: Dict[int, List[str]]) -> bool:
        """Open every unique backing file and register FILE memory once (worker init)."""
        paths: List[str] = []
        for fl in ssd_files.values():
            paths.extend(fl)
        unique = sorted(set(paths))
        if not unique:
            self._path_to_fd = {}
            return True
        path_to_fd: Dict[str, int] = {}
        for path in unique:
            try:
                path_to_fd[path] = os.open(path, os.O_RDWR)
            except OSError as e:
                flexkv_logger.error(f"NIXL: open {path} failed: {e}")
                _close_xfer_files(path_to_fd)
                return False
        reg_file = [(0, 0, path_to_fd[p], p) for p in path_to_fd]
        if not _nixl_register(self.agent, reg_file, "FILE"):
            _close_xfer_files(path_to_fd)
            return False
        self._path_to_fd = path_to_fd
        return True

    def prepare_dram_cpu(self, cpu_tensor: torch.Tensor) -> bool:
        """Register the full pinned CPU pool as DRAM once (after cudaHostRegister)."""
        sz = int(cpu_tensor.numel() * cpu_tensor.element_size())
        ptr = int(cpu_tensor.data_ptr())
        if self.backend_name == "HF3FS":
            try:
                page = int(os.sysconf("SC_PAGESIZE"))
            except (AttributeError, OSError, ValueError):
                page = 4096
            if page > 0 and (ptr % page != 0 or sz % page != 0):
                flexkv_logger.warning(
                    "NIXL HF3FS: CPU pool not page-aligned (ptr %% page=%s, size %% page=%s); "
                    "DRAM_ZC fast path may fall back to bounce buffers. "
                    "Consider page-aligned allocation for the CPU KV pool.",
                    ptr % page,
                    sz % page,
                )
        dram_reg = [(ptr, sz, 0, "")]
        ok = _nixl_register(self.agent, dram_reg, "DRAM")
        if ok and self.backend_name == "HF3FS":
            ex = self._nixl_extra_config
            plg = ex.get("plugin", {})
            hf = plg.get("hf3fs", {}) if isinstance(plg, dict) else {}
            flat_mp = ex.get("mount_point")
            mount = hf.get("mount_point") if isinstance(hf, dict) else None
            mount = mount or flat_mp
            if not mount:
                flexkv_logger.warning(
                    "NIXL HF3FS: no mount_point in nixl_extra_config / "
                    "CacheConfig.nixl_hf3fs_mount_point; backend default applies "
                    "(see NIXL hf3fs_backend)."
                )
        return ok

    def prepare_vram_gpu(self, gpu_tensors: List[torch.Tensor]) -> bool:
        """Register GPU KV tensors once (POSIX path does not use this)."""
        if not gpu_tensors:
            return False
        return _nixl_register(self.agent, gpu_tensors)

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
        if self._path_to_fd is None:
            flexkv_logger.error("NIXL: xfer_dram_file called without prepare_all_ssd_files")
            return False
        xfer_tuples: List[Tuple[int, int, int]] = []
        for path, off, ln in zip(file_paths, region_offsets, region_lens):
            fd = self._path_to_fd.get(path)
            if fd is None:
                flexkv_logger.error(
                    f"NIXL: file {path!r} not in session (not opened at init)"
                )
                return False
            xfer_tuples.append((off, ln, fd))
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
        if self._path_to_fd is None:
            flexkv_logger.error("NIXL: xfer_vram_file called without prepare_all_ssd_files")
            return False
        xfer_tuples: List[Tuple[int, int, int]] = []
        for path, off, ln in zip(file_paths, region_offsets, region_lens):
            fd = self._path_to_fd.get(path)
            if fd is None:
                flexkv_logger.error(
                    f"NIXL: file {path!r} not in session (not opened at init)"
                )
                return False
            xfer_tuples.append((off, ln, fd))
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
