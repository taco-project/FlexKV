# GPU Backend Integration Guide

This document is for developers who want to add support for a new GPU/NPU vendor in FlexKV.

---

## Overview

FlexKV uses a layered GPU abstraction architecture that isolates all vendor-specific code inside individual backend modules, keeping the core logic hardware-agnostic. Three backends are built-in:

| Backend | Hardware | Status |
|---------|----------|--------|
| `nvidia` | NVIDIA A/H/B series (CUDA) | Production-ready (default) |
| `musa` | MTT S series (MUSA) | Experimental |
| `generic` | Any platform (pure PyTorch CPU fallback) | Available |

To add a new vendor (Ascend, Cambricon, other CUDA-compatible vendors, etc.), follow the steps in this document.

---

## Five Core Design Principles

Before you start, please understand the five core principles that govern all backend implementations:

**Principle 1 — Vendor code is self-contained**

All implementation files for a vendor (C++ kernels, Python adapters, CMakeLists) must live under `gpu_backend/<vendor>/`. Do not scatter vendor-specific code across shared paths.

**Principle 2 — All access goes through the GpuBackend interface**

Application code (`worker.py`, `transfer_engine.py`, etc.) must only call `GpuBackend` abstract methods — no `cuda*`, `musa*`, `acl*`, or any other vendor API should appear in shared code.

**Principle 3 — Independent wheel packages, installed on demand**

A build targeting a specific vendor (`FLEXKV_GPU_BACKEND=<vendor>`) should produce a standalone wheel (e.g., `flexkv-nvidia-*.whl`). Users install only what they need; no other vendor's SDK is required at build time.

**Principle 4 — Prefer CUDA-compatible vendors to reuse NvidiaBackend**

> **Read this before deciding whether to create a new backend.**

Many third-party vendors (e.g., Baidu Kunlun) expose a CUDA-compatible interface via an adaptation layer, where `torch.cuda.is_available()` and `import flexkv.c_ext` both work normally. **For these vendors, reuse `NvidiaBackend` directly — no new code is needed.** Simply add an alias in `_ENV_MAP`:

```python
# flexkv/gpu_backend/__init__.py
_ENV_MAP = {
    ...
    "kunlun": "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",  # CUDA-compatible
}
```

To check compatibility, run these two commands on the target hardware. If both succeed, the vendor is CUDA-compatible and does not need a new backend:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"  # True
python3 -c "import flexkv.c_ext; print('ok')"                # ok
```

Only vendors with a fully independent runtime (MUSA, AscendCL, etc.) need a new backend.

**Principle 5 — Performance first; the abstraction layer must not add overhead**

All `GpuBackend` methods are thin wrappers. No extra Python-level dispatch, locking, or memory copies are allowed on hot paths. Critical paths (`transfer_kv_blocks`, `layout_transform`, `host_register`) call the underlying C extension or runtime library directly. The overhead introduced by the abstraction layer must be within single-digit microseconds per call.

---

## Architecture

```
FlexKV Application Layer (Python)
        │
        ▼
flexkv/gpu_backend/          ← Python abstraction layer
  interface.py               ← GpuBackend ABC (15 abstract methods)
  __init__.py                ← Auto-detection + current_backend singleton
  nvidia/backend.py          ← NvidiaBackend
  musa/backend.py            ← MusaBackend
  generic/backend.py         ← GenericBackend
        │
        ▼
csrc/gpu_compat.h            ← C++ compile-time macro switching
csrc/gpu_backend/
  nvidia/                    ← CUDA kernels + cuFile GDS
  musa/                      ← MUSA kernels + muFile GDS
  generic/                   ← CPU fallback stub
        │
        ▼
build_backends/              ← Build system (one Builder per vendor)
  base.py                    ← GPUBuilder ABC
  cuda_builder.py
  musa_builder.py
  generic_builder.py
```

---

## Backend Auto-Detection

At runtime, FlexKV selects a backend using the following priority order:

1. **Environment variable** `FLEXKV_GPU_BACKEND` (nvidia / musa / generic / ...)
2. **entry_points plugins** (`flexkv.gpu_backends` group, registered by third-party packages)
3. **Built-in probe** (tries Nvidia → MUSA in order)
4. **Default**: `NvidiaBackend`

The recommended approach for new vendors is the **entry_points plugin mechanism** — it requires no changes to the FlexKV main repository.

---

## Adding a New Backend: Step-by-Step

> Before starting, verify that your vendor actually needs a new backend (see Principle 4).

### Step 1 — Python Layer (`flexkv/gpu_backend/<vendor>/`)

Create a new directory and implement the `GpuBackend` ABC defined in `flexkv/gpu_backend/interface.py`.

**NvidiaBackend as a reference example:**

```python
# flexkv/gpu_backend/nvidia/backend.py

import ctypes
import torch
from flexkv.gpu_backend.interface import GpuBackend

class NvidiaBackend(GpuBackend):
    """NVIDIA CUDA backend — production default"""

    try:
        import flexkv.c_ext as _ext  # type: ignore
        _HAS_CUDA_EXT = True
    except ImportError:
        _ext = None
        _HAS_CUDA_EXT = False

    # ── Device queries ────────────────────────────────────
    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()

    @classmethod
    def device_name(cls) -> str:
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NVIDIA (unavailable)"

    def set_device(self, device_id: int) -> None:
        torch.cuda.set_device(device_id)

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    # ── Stream ────────────────────────────────────────────
    def create_stream(self):
        return torch.cuda.Stream()

    def destroy_stream(self, stream) -> None:
        pass

    def stream_context(self, stream):
        return torch.cuda.stream(stream)

    def get_current_stream(self):
        return torch.cuda.current_stream()

    def synchronize(self, stream=None) -> None:
        torch.cuda.synchronize()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    # ── Pinned memory ─────────────────────────────────────
    def register_host_tensor(self, tensor: torch.Tensor) -> None:
        _lib = ctypes.CDLL("libcudart.so")
        ret = _lib.cudaHostRegister(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_size_t(tensor.numel() * tensor.element_size()), 1)
        if ret != 0:
            raise RuntimeError(f"cudaHostRegister failed with error code {ret}")

    def unregister_host_tensor(self, tensor: torch.Tensor) -> None:
        ctypes.CDLL("libcudart.so").cudaHostUnregister(ctypes.c_void_p(tensor.data_ptr()))

    def alloc_pinned(self, size: int):
        ptr = ctypes.create_string_buffer(size)
        ctypes.CDLL("libcudart.so").cudaHostRegister(
            ctypes.c_void_p(ctypes.addressof(ptr)), ctypes.c_size_t(size), 1)
        return ptr

    def free_pinned(self, ptr) -> None:
        ctypes.CDLL("libcudart.so").cudaHostUnregister(ctypes.c_void_p(ctypes.addressof(ptr)))

    # ── IPC (cross-process GPU memory sharing) ────────────
    def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
        # Call libcudart cudaIpcGetMemHandle, returns a 64-byte handle
        handle = (ctypes.c_byte * 64)()
        ret = ctypes.CDLL("libcudart.so").cudaIpcGetMemHandle(
            handle, ctypes.c_void_p(tensor.data_ptr()))
        if ret != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed with error code {ret}")
        return bytes(handle)

    def open_ipc_handle(self, handle, shape, dtype, device_id):
        # Call libcudart cudaIpcOpenMemHandle, reconstruct as Tensor
        ...

    def close_ipc_handle(self, ptr: int) -> None:
        ctypes.CDLL("libcudart.so").cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))

    # ── KV transfer (calls C extension) ───────────────────
    def transfer_kv_blocks(self, src, dst, block_ids, stream=None) -> None:
        if self._HAS_CUDA_EXT:
            self._ext.transfer_kv_blocks(src, dst, block_ids,
                                         stream.cuda_stream if stream else 0)
        else:
            for s, d in zip(src, dst):
                d.copy_(s, non_blocking=True)

    def layout_transform(self, src, dst, stream=None) -> None:
        if self._HAS_CUDA_EXT:
            self._ext.layout_transform(src, dst, stream.cuda_stream if stream else 0)
        else:
            dst.copy_(src)

    # ── GDS (Direct Storage) ──────────────────────────────
    def supports_direct_storage(self) -> bool:
        return self._HAS_CUDA_EXT and hasattr(self._ext, 'cufile_read')

    def read_direct(self, fd: int, tensor, offset: int) -> int:
        return self._ext.cufile_read(fd, tensor, offset)

    def write_direct(self, fd: int, tensor, offset: int) -> int:
        return self._ext.cufile_write(fd, tensor, offset)

    def is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.is_cuda
```

**Key points:**
- Implement **all** abstract methods from `interface.py` — failing to do so raises `TypeError` at instantiation.
- For capabilities not supported by your hardware (e.g., GDS), return `False` or raise `NotImplementedError`. Do not silently ignore errors.
- If IPC is unavailable, use the SHM fallback (see `generic/backend.py`).

### Step 2 — Register the Backend

**Option A: Built-in (modify the main repository)**

Edit `flexkv/gpu_backend/__init__.py`:

```python
_ENV_MAP = {
    "nvidia": "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",
    "cuda":   "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",
    "musa":   "flexkv.gpu_backend.musa.backend.MusaBackend",
    "generic":"flexkv.gpu_backend.generic.backend.GenericBackend",
    # CUDA-compatible vendor — reuse NvidiaBackend directly:
    "kunlun": "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",
    # Vendor with independent runtime — needs its own backend:
    "vendor_x": "flexkv.gpu_backend.vendor_x.backend.VendorXBackend",
}
```

**Option B: entry_points plugin (recommended — zero changes to FlexKV)**

Declare the entry point in your own package's `pyproject.toml`:

```toml
[project.entry-points."flexkv.gpu_backends"]
vendor_x = "flexkv_vendor_x.backend:VendorXBackend"
```

FlexKV discovers and loads it automatically at startup.

### Step 3 — C++ Layer (`csrc/gpu_backend/<vendor>/`)

If your vendor has GPU-specific kernels, follow the layout of `csrc/gpu_backend/nvidia/`:

```
csrc/gpu_backend/<vendor>/
  CMakeLists.txt
  transfer.<ext>            # GPU kernel (.cu / .mu / .bang / ...)
  layout_transform.<ext>    # GDS layout kernel (omit if no GDS support)
  gds_manager.<ext>         # Direct Storage manager (optional)
  tp_transfer_thread_group.cpp
```

`csrc/gpu_compat.h` provides compile-time macro switching:

```cpp
// csrc/gpu_compat.h (excerpt)
#if defined(FLEXKV_BACKEND_MUSA)
  #include <musa_runtime.h>
  using gpu_stream_t = musaStream_t;
  #define gpu_current_stream()   flexkv::musa_current_stream()
  #define gpu_get_last_error()   musaGetLastError()
  #define GPU_SUCCESS            musaSuccess
#else  // default: CUDA
  #include <cuda_runtime.h>
  #include <ATen/cuda/CUDAContext.h>
  using gpu_stream_t = cudaStream_t;
  #define gpu_current_stream()   at::cuda::getCurrentCUDAStream()
  #define gpu_get_last_error()   cudaGetLastError()
  #define GPU_SUCCESS            cudaSuccess
#endif
```

Add an `#elif defined(FLEXKV_BACKEND_XXX)` branch for your vendor.

### Step 4 — Build System (`build_backends/<vendor>_builder.py`)

Subclass `GPUBuilder` from `build_backends/base.py`:

```python
# build_backends/base.py (interface)
class GPUBuilder(ABC):
    @abstractmethod
    def get_extension_class(self): ...   # Return Extension class
    @abstractmethod
    def get_sources(self, **kw) -> list: ...   # Source file list
    @abstractmethod
    def get_compile_args(self, **kw) -> dict: ...
    @abstractmethod
    def get_link_args(self, **kw) -> list: ...
    @abstractmethod
    def get_build_ext_class(self): ...
    def configure_env(self): ...         # Optional: pre-build env setup
```

Use `build_backends/cuda_builder.py` as a reference. Register your builder in `setup.py`:

```python
_BUILDER_MAP = {
    "cuda":    "build_backends.cuda_builder.CUDABuilder",
    "musa":    "build_backends.musa_builder.MUSABuilder",
    "generic": "build_backends.generic_builder.GenericBuilder",
    # New vendor:
    "vendor_x": "build_backends.vendor_x_builder.VendorXBuilder",
}
```

---

## GDS (GPU Direct Storage) Support

GDS allows file data to be read/written directly to GPU memory without CPU involvement, significantly reducing IO latency.

| Platform | Solution | Notes |
|----------|----------|-------|
| NVIDIA | cuFile GDS | Production-ready; requires `FLEXKV_STORAGE_BACKEND=cufile` |
| All others | POSIX fallback (pread/pwrite + H2D memcpy) | Auto-degraded; functionally equivalent, lower performance |

**Fallback behavior**: For vendors without GDS support, simply return `False` from `gds_available()`. FlexKV automatically uses the POSIX IO path — no additional handling required, and functionality is unaffected.

**Recommended**: If your platform has a Direct Storage capability (GPU DMA directly to/from persistent storage), it is strongly recommended to implement the `gds_read` / `gds_write` interface. This can significantly improve KV cache swap throughput. See `csrc/gpu_backend/nvidia/gds_manager.cpp` as a reference.

---

## IPC (Cross-Process GPU Memory Sharing) Support

IPC allows multiple processes to share the same GPU memory region without copying, which is critical for FlexKV's distributed KV cache.

| Platform | API | Support Level |
|----------|-----|---------------|
| NVIDIA | `cudaIpcGetMemHandle` | Full |
| AMD ROCm | `hipIpcGetMemHandle` | Mostly compatible |
| Others | No native equivalent | Degrades to SHM fallback |

**Fallback behavior**: Platforms without native IPC automatically fall back to a `multiprocessing.shared_memory`-based SHM implementation. Data is routed through CPU (GPU→CPU→GPU), which preserves correctness but introduces an extra copy. See `SHMFallbackMixin` in `generic/backend.py`.

**Recommended**: If your platform supports any form of cross-process GPU memory sharing (even with a different API name), it is strongly recommended to implement native IPC. This eliminates the SHM fallback copy overhead, with especially significant performance gains in high-concurrency scenarios.

---

## Building and Testing

```bash
# NVIDIA CUDA with GDS (recommended default)
FLEXKV_GPU_BACKEND=cuda FLEXKV_STORAGE_BACKEND=cufile pip install -e .

# NVIDIA CUDA with POSIX IO (no cuFile/GDS required)
FLEXKV_GPU_BACKEND=cuda FLEXKV_STORAGE_BACKEND=posix pip install -e .

# New vendor example (AMD ROCm)
FLEXKV_GPU_BACKEND=rocm pip install -e .

# Run backend unit tests
cd FlexKV
PYTHONPATH=. python3 -m pytest tests/test_gpu_backend.py --noconftest -v
```

Test coverage:
- `TestGpuBackendInterface` — ABC completeness + cannot instantiate directly
- `TestNvidiaBackend` — is_available, device_name, stream, IPC, transfer (requires CUDA GPU; skipped otherwise)
- `TestGenericBackend` — 11 cases, all passing without GPU hardware
- `TestAutoDetection` — env-var override (generic / cuda / rocm / invalid values)

---

## FAQ

**Q: My vendor's hardware is CUDA-compatible. Do I need to create a new backend?**

No. As long as `torch.cuda.is_available()` and `import flexkv.c_ext` both work normally, simply map your vendor name to `NvidiaBackend` in `_ENV_MAP` (see Principle 4).

**Q: I just want to support a new card but have no custom kernels. Can I still use FlexKV?**

Yes. Subclass `GenericBackend` and override only the device query methods. `transfer_kv_blocks` automatically falls back to `torch`-native `copy_`, and GDS / IPC will also use their respective fallback paths.

**Q: What is the difference between an entry_points plugin and a built-in backend?**

They are functionally identical. entry_points plugins are maintained in the vendor's own package, independent of FlexKV release cycles. This is the recommended approach for external vendors.

**Q: Can `FLEXKV_GPU_BACKEND` and `FLEXKV_STORAGE_BACKEND` be set independently?**

Yes. They are orthogonal: `FLEXKV_GPU_BACKEND` controls compute and memory; `FLEXKV_STORAGE_BACKEND` controls the IO path.
