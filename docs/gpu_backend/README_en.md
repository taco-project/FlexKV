# FlexKV GPU Backend Onboarding Guide

> Audience: engineers porting FlexKV to a new GPU vendor.
>
> This document only describes the **current state** of the abstraction;
> it does not cover historical changes or migration steps.
>
> Chinese version: [README_zh.md](./README_zh.md).

---

## 1. Five Core Principles

| # | Principle | Meaning |
| - | --------- | ------- |
| 1 | **Group code by vendor** | All GPU vendor-specific code lives under `csrc/gpu_backend/<vendor>/` and `flexkv/gpu_backend/<vendor>/`. The two trees are independent. FlexKV upper layers (`worker.py / transfer_engine.py / cache_engine.py / common/memory_handle.py / kvtask.py / storage/allocator.py / integration/* / kvmanager.py`) contain **no** literal `cuda*` / `hip*` / `musa*` calls. |
| 2 | **One unified abstraction** | Python: the `GpuBackend` ABC. C++: `csrc/gpu_backend/backend_bindings.h` exposes a per-vendor pybind registration function. Upper layers only call `current_backend.xxx()` and only import vendor-agnostic symbols from `flexkv.c_ext`. |
| 3 | **Per-vendor wheels, opt-in install** | Each build compiles exactly one vendor's C extension, selected by `FLEXKV_GPU_BACKEND`. Wheels can be shipped independently as `flexkv-nvidia` / `flexkv-rocm` / `flexkv-musa`; building ROCm needs no CUDA SDK and vice-versa. |
| 4 | **Prefer CUDA-ecosystem reuse** | Hardware that already speaks the CUDA API (e.g. Baidu Kunlunxin) can reuse `NvidiaBackend / NvidiaBuilder` directly, no new backend needed. Use aliases like `FLEXKV_GPU_BACKEND=kunlun`. |
| 5 | **Performance first, zero overhead** | The abstraction layer is a thin wrapper. Hot paths (`transfer_kv_blocks` / `layout_transform` / `register_host_tensor`) are single-line `return self._ext.xxx(...)` forwards — no Python-level dict lookups, locks, or copies. On the C++ side, vendor selection happens at compile time via `FLEXKV_BACKEND_<VENDOR>` macros, so there are no runtime branches. |

> **Naming note**: `csrc/gpu_backend/nvidia/gtensor_handler.cuh` defines `enum class BackendType { VLLM, TRTLLM, SGLANG }`, which describes the **LLM framework backend** (vLLM / TRT-LLM / SGLang) and is unrelated to GPU vendors. Throughout this document, the GPU hardware vendor is captured by `GpuVendor`.

---

## 2. Directory Layout

```
FlexKV/
├── flexkv/
│   └── gpu_backend/                       ← Python abstraction
│       ├── __init__.py                    ← auto-detect + current_backend singleton
│       ├── interface.py                   ← GpuBackend ABC + GpuVendor enum
│       ├── nvidia/backend.py              ← NvidiaBackend
│       ├── rocm/backend.py                ← RocmBackend
│       ├── musa/backend.py                ← MusaBackend
│       └── generic/backend.py             ← GenericBackend (PyTorch fallback)
│
├── csrc/
│   ├── bindings.cpp                       ← GPU-vendor agnostic. Registers
│   │                                        only the cross-vendor CPU
│   │                                        bindings (SSD / Hasher /
│   │                                        RadixTree / Pcfs / Redis / ...)
│   │                                        and ends with one call to
│   │                                        register_active_backend_bindings(m)
│   ├── hash.cpp / radix_tree.cpp /
│   ├── transfer_ssd.cpp /
│   ├── eviction_strategy.cpp /
│   ├── pcfs/* /
│   ├── monitoring/* /
│   ├── dist/*                             ← GPU-agnostic, shared by all vendors
│   │
│   └── gpu_backend/                       ← C++ abstraction
│       ├── backend_bindings.h             ← register_<vendor>_bindings(m) interface
│       ├── gpu_types.h                    ← cross-vendor macros / typedefs:
│       │                                    gpu_stream_t / gpuMallocHost /
│       │                                    gpuStreamCreate / gpuSetDevice /
│       │                                    gpuGetLastError / gpuSuccess / ...
│       ├── nvidia/
│       │   ├── nvidia_bindings.cpp        ← register_nvidia_bindings(m)
│       │   ├── transfer.cu / transfer.cuh
│       │   ├── gtensor_handler.cuh
│       │   ├── tp_transfer_thread_group.cpp/.h  ← uses gpu_* macros
│       │   └── gds/                       ← cuFile / GDS
│       │       ├── gds_manager.cpp/.h
│       │       ├── tp_gds_transfer_thread_group.cpp/.h
│       │       └── layout_transform.cu/.cuh
│       ├── rocm/
│       │   ├── rocm_bindings.cpp          ← register_rocm_bindings(m)
│       │   └── .hipified/                 ← generated at build time by hipify-perl
│       ├── musa/
│       │   └── transfer.mu
│       └── generic/                       ← reserved
│
├── build_backends/                        ← compile-time vendor dispatch
│   ├── __init__.py                        ← load_builder(name) entry point
│   ├── base.py                            ← GPUBuilder ABC
│   ├── cuda_builder.py                    ← NvidiaBuilder
│   ├── rocm_builder.py                    ← RocmBuilder
│   ├── musa_builder.py                    ← MusaBuilder (skeleton)
│   └── generic_builder.py                 ← GenericBuilder (no C extension)
│
└── setup.py                               ← reads FLEXKV_GPU_BACKEND,
                                             load_builder(name) → builds one vendor
```

**Invariants**

- `bindings.cpp` does **not** `#include` any vendor header and does **not** call any `cuda*` / `hip*` API. NVIDIA-/ROCm-specific bindings live in their respective `<vendor>_bindings.cpp`.
- `flexkv.c_ext` is the unified Python extension module name (used by both the NVIDIA and ROCm wheels). MUSA uses a separate name `flexkv.c_ext_musa` because of toolchain differences.
- Upper-layer code **never** imports `flexkv.c_ext` directly; it always goes through `current_backend`.
- `csrc/transfer_ssd.cpp` and friends (Pcfs / Redis / RadixTree / Hasher) are GPU-agnostic, registered by `bindings.cpp`, and shared by every wheel.

---

## 3. Abstractions and What They Do

The abstraction exists at three layers simultaneously: the **Python runtime** (`GpuBackend`), the **C++ pybind layer** (`backend_bindings.h`), and the **build system** (`GPUBuilder`).

### 3.1 `flexkv.gpu_backend.GpuBackend` (Python runtime abstraction)

ABC defined in `flexkv/gpu_backend/interface.py`. It is the **single contract** between the upper layers and the vendor implementations. Each vendor provides `class <Vendor>Backend(GpuBackend)` in `flexkv/gpu_backend/<vendor>/backend.py`.

Methods are organized into 7 groups:

| # | Group | Key methods | Purpose |
| - | ----- | ----------- | ------- |
| 1 | **Meta** | `is_available()` / `device_name()` / `torch_device_type()` / `vendor` | Auto-detection, `torch.device(...)` construction, and tensor membership checks (NVIDIA & ROCm both report `"cuda"`; MUSA reports `"musa"`). |
| 2 | **Devices** | `set_device / current_device / device_count / synchronize / is_initialized / init_runtime / empty_cache / is_gpu_tensor / get_device_capability / make_device / detect_arch_list` | Replaces every `torch.cuda.*` call upstream; `detect_arch_list` is consumed by the builder to populate the arch list at compile time. |
| 3 | **Streams** | `create_stream / destroy_stream / get_current_stream / stream_handle / stream_context` | `stream_handle` exposes the underlying uintptr_t for the C extension; `stream_context` is a `with`-style context. |
| 4 | **Pinned host memory** | `register_host_tensor / unregister_host_tensor / alloc_pinned / free_pinned` | Replaces `cudaHostRegister / cudaHostUnregister`; implementations use ctypes to dlopen the runtime (`libcudart.so` / `libamdhip64.so` / `libmusa.so`). |
| 5 | **Hot path (KV transfer)** | `transfer_kv_blocks / transfer_kv_blocks_ssd / layout_transform` | Single-line forwards into `flexkv.c_ext.*`. The default `transfer_kv_blocks_ssd` lives in the ABC and is shared by every vendor. |
| 6 | **Cross-process IPC** | `supports_ipc / get_ipc_handle / open_ipc_handle / close_ipc_handle` | `cudaIpc* / hipIpc* / musaIpc*` handles are not binary-compatible across vendors, so each vendor implements them independently. Backends without IPC return `False`. See §3.6. |
| 7 | **Direct storage** | `supports_direct_storage / gds_create_manager` | NVIDIA uses cuFile / GDS. Other vendors return `False`; callers fall back to the POSIX/io_uring path (`transfer_kv_blocks_ssd`, GPU-agnostic). |

`GpuVendor` enum: `NVIDIA / ROCM / MUSA / GENERIC`, used purely as metadata.

> Beyond the 7 groups above, `GpuBackend` exposes a small **device-visibility
> env-var** abstraction in group 2
> (`visible_devices_env_vars / get_visible_device_map / strip_visible_devices`)
> that replaces every hardcoded `CUDA_VISIBLE_DEVICES` literal in the upper
> layers. See §3.7.

### 3.2 `csrc/gpu_backend/backend_bindings.h` (C++ pybind registration)

```cpp
#if defined(FLEXKV_BACKEND_NVIDIA)
void register_nvidia_bindings(pybind11::module_& m);
#elif defined(FLEXKV_BACKEND_ROCM)
void register_rocm_bindings (pybind11::module_& m);
#elif defined(FLEXKV_BACKEND_MUSA)
void register_musa_bindings (pybind11::module_& m);
#endif

inline void register_active_backend_bindings(pybind11::module_& m);
```

Effects:

- `bindings.cpp` only calls `register_active_backend_bindings(m)` and is **completely vendor-agnostic**.
- Each vendor implements `register_<vendor>_bindings` inside `csrc/gpu_backend/<vendor>/<vendor>_bindings.cpp`, registering at minimum `transfer_kv_blocks` and `TPTransferThreadGroup`. NVIDIA additionally registers `GDSManager / TPGDSTransferThreadGroup / transfer_kv_blocks_gds`.
- The builder injects `-DFLEXKV_BACKEND_<VENDOR>` on the compile command line; selection happens at compile time, with zero runtime overhead.

### 3.3 `csrc/gpu_backend/gpu_types.h` (C++ cross-vendor types/macros)

Unifies vendor runtime APIs under a `gpu_*` family:

```cpp
gpu_stream_t / gpuError_t / gpuSuccess
gpuMallocHost   gpuFreeHost
gpuSetDevice    gpuStreamCreate
gpuGetLastError gpuGetErrorString
```

This lets cross-vendor source files such as `tp_transfer_thread_group.cpp` be written **once**; the vendor-specific runtime call (cuda*/hip*/musa*) is selected by `-DFLEXKV_BACKEND_<VENDOR>`.

### 3.4 `build_backends.GPUBuilder` (build-time abstraction)

ABC in `build_backends/base.py`. `setup.py` calls `load_builder(name)` to obtain an instance and queries it for:

| Method | Purpose |
| ------ | ------- |
| `is_available()` | Whether the current environment supports this builder (e.g. `torch.version.hip is not None`). |
| `get_extension_class()` | Returns `CUDAExtension` / `MUSAExtension` / plain `Extension`. |
| `get_extension_name()` | Returns `"flexkv.c_ext"` or `"flexkv.c_ext_musa"`, i.e. the produced `.so` name. |
| `get_sources(**opts)` | Vendor-specific source list (`.cpp/.cu/.mu/.hipified/...`). |
| `get_compile_args(**opts)` | Returns `{"cxx": [...], "nvcc": [...]}`; auto-injects `vendor_macro()` (i.e. `-DFLEXKV_BACKEND_<VENDOR>`). |
| `get_link_args(**opts)` | Link libraries (`-lxxhash` / `-lcuda` / `-lcufile` / `-lhiredis` / `-lhifs_client_sdk` / ...). |
| `get_include_dirs(**opts)` | Include paths; must include at least `csrc/` and `csrc/gpu_backend/`. |
| `get_build_ext_class()` | Returns the `BuildExtension` subclass. |
| `configure_env()` | Pre-build environment side effects, e.g. probing `TORCH_CUDA_ARCH_LIST` / `PYTORCH_ROCM_ARCH`, or running hipify. |
| `vendor_macro()` | Returns `-DFLEXKV_BACKEND_<NAME>`, kept in sync with `gpu_types.h` / `backend_bindings.h`. |

### 3.5 GDS and GDR — how the two "GPU-direct" paths are abstracted

FlexKV has **two independent "GPU-direct"-style fast paths**. They look similar
on paper (both bypass the CPU bounce buffer) but live at different layers and
are abstracted differently. Knowing where each one fits prevents new vendors
from re-implementing the wrong piece.

| Path  | Counterparts on the wire     | Vendor coupling                                  | Where it is implemented in FlexKV                          |
| ----- | ---------------------------- | ------------------------------------------------ | ---------------------------------------------------------- |
| GDS   | GPU memory ⇄ local SSD       | **Tightly coupled** to the GPU vendor (cuFile)   | `csrc/gpu_backend/<vendor>/gds/` + `<vendor>_bindings.cpp` |
| GDR   | GPU/CPU memory ⇄ remote NIC  | **Loosely coupled** — RDMA-side is vendor-neutral | `flexkv/mooncakeEngineWrapper.py` + Mooncake transfer engine |

#### 3.5.1 GDS (GPU Direct Storage) — vendor-private, opt-in at compile time

GDS is the cuFile path: NVIDIA RDMAs SSD blocks straight into device memory.
There is **no portable counterpart** today (ROCm has no cuFile, MUSA has no
direct-storage SDK, Generic has no GPU at all), so GDS must stay
vendor-private.

The abstraction has three layers, all *capability-gated* — never assumed:

1. **Python contract** — `GpuBackend.supports_direct_storage()` and
   `GpuBackend.gds_create_manager(...)` (group 7 in §3.1). Default
   implementation returns `False` / raises `NotImplementedError`. Only
   `NvidiaBackend` overrides them, and only when `flexkv.c_ext` was built
   with `FLEXKV_ENABLE_GDS=1`. Upper-layer code **must** branch on
   `current_backend.supports_direct_storage()` before touching
   `gds_create_manager()`.
2. **C++ binding** — the GDS classes (`GDSManager`,
   `TPGDSTransferThreadGroup`, `transfer_kv_blocks_gds`) are registered
   inside `register_nvidia_bindings(m)` under `#ifdef FLEXKV_ENABLE_GDS`.
   `csrc/bindings.cpp` and `register_<other-vendor>_bindings` know nothing
   about cuFile. ROCm / MUSA / Generic wheels simply don't expose those
   symbols.
3. **Source layout** — every GDS source file lives under
   `csrc/gpu_backend/nvidia/gds/` (`gds_manager.{h,cpp}`,
   `tp_gds_transfer_thread_group.{h,cpp}`, `layout_transform.{cuh,cu}`).
   Other vendors **must not** create a sibling directory unless their
   hardware actually has a cuFile-equivalent SDK; otherwise the no-op
   fallback below is used.

**Fallback for non-GDS vendors**: `transfer_kv_blocks_ssd` (registered by
`csrc/bindings.cpp`, GPU-agnostic, POSIX / io_uring backed). Upper layers
call it via `current_backend.transfer_kv_blocks_ssd(...)`, so there is no
runtime branch in the hot path — the worker simply picks the GDS or SSD
worker once at construction time based on `supports_direct_storage()`.

#### 3.5.2 GDR (GPU Direct RDMA) — vendor-neutral, runtime-detected

The cross-node KV-cache reuse path uses
[Mooncake transfer engine](https://github.com/kvcache-ai/Mooncake), which
is a thin wrapper over libibverbs. The RDMA NIC DMAs *directly* into the
buffer that Mooncake's `register_memory(ptr, size)` was given — whether
that buffer sits in CPU pinned memory or in GPU device memory is
**transparent to FlexKV's abstraction**. The buffer's pointer and size are
the only inputs; how it was allocated is the vendor backend's business.

Because of that, GDR does **not** need a per-vendor C++ binding or a new
abstract method on `GpuBackend`. The seam is much smaller:

1. **Buffer producers go through `GpuBackend`.** Whenever upper layers need
   a buffer that will later be exposed to Mooncake, they obtain it from the
   abstraction:
   - CPU pinned bounce buffer → `GpuBackend.alloc_pinned` /
     `register_host_tensor` (group 4).
   - GPU device buffer (for true GPU-direct RDMA) → ordinary
     `torch.empty(..., device=current_backend.make_device(idx))` plus
     `current_backend.stream_handle(...)` for sync.
   - Cross-process GPU buffer → `get_ipc_handle / open_ipc_handle`
     (group 6, only on backends where `supports_ipc()` is True).
2. **`MoonCakeTransferEngineWrapper.regist_buffer(ptr, size)` is vendor-agnostic.**
   It just calls `engine.register_memory(ptr, size)`. The wrapper is GPU-vendor
   neutral by construction; nothing under `flexkv/gpu_backend/<vendor>/`
   imports it.
3. **The decision "GDR or staged copy" lives in the worker, not in the abstraction.**
   `flexkv/transfer/worker.py` selects between `PEER2CPUTransferWorker`
   (Mooncake) and the fallback (CPU staging + standard SSD/CPU paths) once
   at construction time, driven by config (`enable_p2p_cpu / enable_p2p_ssd`)
   and `MOONCAKE_CONFIG_PATH`. There is **no** vendor-specific GDR API in
   the hot path.

Practical consequence for backend authors: **you do not need to add a GDR
hook**. As long as your `GpuBackend` correctly implements groups 1–4 (so
that pinned host memory and device pointers are obtained safely) and group
6 (if you want cross-process GPU RDMA), Mooncake will work out of the box.
NIC discovery / GID / PD creation are entirely Mooncake's responsibility.

#### 3.5.3 Quick decision matrix

| Scenario                                       | Path used at runtime                     | What the new vendor must implement                                |
| ---------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------ |
| Local SSD, vendor has cuFile                   | GDS (`gds_create_manager`)               | `csrc/gpu_backend/<vendor>/gds/*` + extend the binding under `#ifdef` |
| Local SSD, vendor has no cuFile equivalent     | POSIX/io_uring (`transfer_kv_blocks_ssd`) | Nothing — inherit the default                                      |
| Remote node, NIC supports GPU Direct RDMA      | Mooncake GDR (`register_memory(gpu_ptr)`) | Nothing on the GpuBackend; ensure `supports_ipc` if cross-process  |
| Remote node, NIC only supports CPU pinned RDMA | Mooncake on pinned bounce buffer         | Make sure `register_host_tensor / alloc_pinned` are correct        |

### 3.6 Cross-process device-memory sharing (IPC) — plumbing KV-cache from the trainer process into the FlexKV server

In FlexKV's multi-instance / server-client mode the **LLM inference framework's
worker process** (vLLM / TRT-LLM / SGLang) owns the actual KV-cache device
memory, while the **FlexKV server / TransferManager runs in a separate
process**. To let the server read/write the worker's KV-cache directly, the GPU
pointer must travel **across processes** — that is exactly what the IPC
abstraction solves. Every related call funnels through `TensorSharedHandle` in
[`flexkv/common/memory_handle.py`](../../flexkv/common/memory_handle.py),
which is the **only Python file in FlexKV that touches vendor IPC primitives**.

#### 3.6.1 Two paths

`TensorSharedHandle.__init__(force_direct_ipc: bool)` chooses one of the two
paths at construction time:

| Path | Producer | Consumer | When to use |
| ---- | -------- | -------- | ----------- |
| **PyTorch reductions (default fast path)** | `torch.multiprocessing.reductions.reduce_tensor(tensor)` | `rebuild_func(*rebuild_args)` | `force_direct_ipc=False`. No vendor-private IPC API needed; works across vendors. |
| **Vendor IPC handle (direct path)** | `_gpu_backend.get_ipc_handle(tensor)` | `_gpu_backend.open_ipc_handle(handle, dtype, shape, strides, storage_offset, device_id)` | `force_direct_ipc=True`. Bypasses PyTorch reductions' refcounting/cleanup hooks; lifetime is managed by the caller (used for long-lived registrations such as the transfer engine). |

Both paths are **transparent to upper layers**: callers only see the
`torch.Tensor` produced by `TensorSharedHandle.create_tensor()` and don't care
which path was taken.

#### 3.6.2 The five APIs exposed by the abstraction (`GpuBackend` group 6)

```python
def supports_ipc(self) -> bool:
    """Whether the hardware/runtime supports cross-process device-memory sharing."""

def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
    """Serialize the GPU tensor into an opaque byte string (CUDA 64B / HIP 64B /
    MUSA ?B). Can be shipped to a peer process via ZeroMQ / pipe / shared file."""

def open_ipc_handle(
    self,
    handle: bytes,
    dtype: torch.dtype,
    shape: Tuple[int, ...],
    strides: Tuple[int, ...],
    storage_offset: int,
    device_id: int,
) -> torch.Tensor:
    """Reconstruct the tensor in the peer process. `device_id` MUST be the
    **physical** GPU id — see §3.7."""

def close_ipc_handle(self, ptr: int) -> None:
    """Release the mapping created by open_ipc_handle. Called on
    TensorSharedHandle teardown / explicit close."""
```

Together with `is_gpu_tensor` (group 2), these five methods are everything the
IPC path needs — **`memory_handle.py` contains zero `cuda*` / `hip*` / `musa*`
literals**.

#### 3.6.3 Cross-vendor caveats and constraints

1. **Binary handles are not vendor-portable.** `cudaIpcMemHandle_t` (64 B),
   `hipIpcMemHandle_t` (64 B) and the MUSA equivalent may share the *size* but
   field layout is **not** guaranteed compatible, and opening one with the
   other vendor's runtime is undefined. Producer and consumer must use the
   **same** backend; FlexKV's `current_backend` is a process-local singleton,
   which satisfies this naturally.
2. **Always pass the physical device id.** The `device_id=N` argument of
   `open_ipc_handle` is the **physical** GPU index. When the worker process
   was started with `CUDA_VISIBLE_DEVICES` set but the server process can see
   all GPUs, the visibility map from §3.7 must be applied first to translate
   logical → physical, otherwise the mapping ends up on the wrong card.
   `server.py` / `transfer_manager.py` already strip the masks via
   `_gpu_backend.strip_visible_devices(env)` so the children see all GPUs —
   that is a precondition.
3. **Backends without IPC.** If `supports_ipc()` returns `False`
   (e.g. `GenericBackend`, or a single-card-only device), the three handle
   methods fall back to the abstract base implementation that **raises
   `RuntimeError("... does not support IPC")`**. The upper-layer
   server-client mode is gated upstream by `is_gpu_tensor()`, so the IPC
   methods are simply never called — you can safely leave them unimplemented.
4. **Why two paths?** The reductions path also calls vendor IPC under the
   hood, but adds PyTorch's own refcounting (`torch.cuda.Event` /
   `cudaIpcOpenEventHandle` / …). The direct path skips that machinery,
   which is **more reliable for long-lived KV-cache registrations** because
   PyTorch's cleanup hook can release the mapping prematurely under certain
   `multiprocessing` topologies.

#### 3.6.4 Minimal effort for a new vendor

```python
class FooBackend(GpuBackend):
    def supports_ipc(self) -> bool:
        return True   # or False, depending on the hardware

    def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
        # call fooIpcGetMemHandle(...), return the bytes
        ...

    def open_ipc_handle(self, handle, dtype, shape, strides,
                        storage_offset, device_id):
        # call fooIpcOpenMemHandle(...), wrap into a torch.Tensor
        ...

    def close_ipc_handle(self, ptr: int) -> None:
        # call fooIpcCloseMemHandle(ptr)
        ...
```

Once these four methods are correct, FlexKV's server-client / multi-instance
mode just works — **no change is required in `memory_handle.py` or any of the
upper-layer call sites**.

### 3.7 GPU device-visibility env vars (`CUDA_VISIBLE_DEVICES`-style)

Every GPU vendor ships its own visibility-mask environment variable to limit
which GPUs a process sees on a multi-GPU host:

| Vendor | Env var |
| ------ | ------- |
| NVIDIA | `CUDA_VISIBLE_DEVICES` |
| AMD ROCm/HIP | `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, plus `CUDA_VISIBLE_DEVICES` for backwards compatibility |
| Moore Threads MUSA | `MUSA_VISIBLE_DEVICES` |
| Future vendor X | `X_VISIBLE_DEVICES` |

FlexKV upper layers need this mask in two flavors:

1. **Read & parse** — in server-client mode, translate the worker's *logical*
   device id to a *physical* device id
   ([`vllm_v1_adapter.py`](../../flexkv/integration/vllm/vllm_v1_adapter.py)
   `FlexKVWorkerConnector` and
   [`trtllm_adapter.py`](../../flexkv/integration/tensorrt_llm/trtllm_adapter.py)
   KV-cache registration).
2. **Strip** — the FlexKV server / cross-process TransferManager must see
   **all** physical GPUs (they perform cross-device IPC), so the mask is
   removed from the env before spawning those subprocesses.

If the upper layers naively poked at `os.environ['CUDA_VISIBLE_DEVICES']`,
ROCm users would get burned: their actual mask might be in
`HIP_VISIBLE_DEVICES`, completely invisible to FlexKV. To fix this once and
for all, FlexKV pushes the responsibility down into `GpuBackend` and the
**business code carries zero vendor literals**.

#### 3.7.1 The three APIs (extension of `GpuBackend` group 2)

```python
@classmethod
def visible_devices_env_vars(cls) -> Tuple[str, ...]:
    """Vendor-specific env-var names that mask GPU visibility.
       The first entry is the canonical/preferred one (used when *writing*
       a mask); all entries participate in *reading* and *stripping*.
       Default returns (), which makes the two helpers below behave as
       no-ops."""

def get_visible_device_map(self) -> Optional[List[int]]:
    """Read the mask -> return [phys_id, ...]; returns None if no mask is set.
       Falls back to None when the mask uses UUIDs (e.g. 'GPU-xxxx') so
       the caller can use the default branch."""

def strip_visible_devices(self, env: Dict[str, str]) -> Dict[str, str]:
    """Remove every visibility mask from `env` in place and return `env`.
       Used by parents that need to spawn children visible to all GPUs."""
```

`get_visible_device_map / strip_visible_devices` already have **complete
default implementations** in the abstract base class — they only depend on
`visible_devices_env_vars()`, so a **new vendor only has to override that one
classmethod**.

#### 3.7.2 What each vendor declares

```python
# NVIDIA
@classmethod
def visible_devices_env_vars(cls) -> Tuple[str, ...]:
    return ("CUDA_VISIBLE_DEVICES",)

# ROCm — HIP is canonical; ROCR is the lower-level ROCr runtime mask;
# the CUDA name is honored too for backwards compatibility with old scripts.
@classmethod
def visible_devices_env_vars(cls) -> Tuple[str, ...]:
    return ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

# MUSA
@classmethod
def visible_devices_env_vars(cls) -> Tuple[str, ...]:
    return ("MUSA_VISIBLE_DEVICES",)
```

#### 3.7.3 Upper-layer call sites (zero literals in business code)

| Call site | Purpose | API |
| --------- | ------- | --- |
| `flexkv/integration/vllm/vllm_v1_adapter.py` | When a worker registers its KV-cache with the server, map `local_device` to physical id | `_gpu_backend.get_visible_device_map()` |
| `flexkv/integration/tensorrt_llm/trtllm_adapter.py` | Same, on the TRT-LLM path | same |
| `flexkv/server/server.py` | Strip the mask before spawning the server child so it sees all GPUs | `_gpu_backend.strip_visible_devices(env)` |
| `flexkv/transfer_manager.py` | Strip the mask before spawning the TransferManager child | same |

> **Invariant**: FlexKV business code (adapter / server / transfer_manager /
> memory_handle) contains **no occurrence** of `CUDA_VISIBLE_DEVICES`,
> `HIP_VISIBLE_DEVICES`, or `MUSA_VISIBLE_DEVICES` as string literals. Those
> strings are only allowed inside `flexkv/gpu_backend/<vendor>/backend.py`.

---

## 4. Building & Publishing per Vendor

### 4.1 Default (NVIDIA CUDA)

> ⚠️ **Important**: `pip install -e .` and `pip install flexkv` **build and install the NVIDIA CUDA wheel by default**.
>
> When `FLEXKV_GPU_BACKEND` is unset, `setup.py` defaults to `"nvidia"`, picks `NvidiaBuilder`, compiles `csrc/gpu_backend/nvidia/` with `nvcc`, and produces `flexkv.c_ext` (NVIDIA build).
>
> No additional configuration is required — behavior matches the pre-refactor experience.

```bash
# Default NVIDIA CUDA wheel
pip install -e .

# Equivalent to
FLEXKV_GPU_BACKEND=nvidia pip install -e .
```

Optional compile switches (all off by default):

| Env var | Effect |
| ------- | ------ |
| `FLEXKV_ENABLE_GDS=1`     | Compile GPU Direct Storage / cuFile (NVIDIA only) |
| `FLEXKV_ENABLE_P2P=1`     | Compile distributed RadixTree + Redis metadata channel |
| `FLEXKV_ENABLE_METRICS=1` | Compile prometheus-cpp monitoring |
| `FLEXKV_ENABLE_CFS=1`     | Compile PCFS (Baidu file system) remote path |
| `FLEXKV_ENABLE_CPUTEST=1` | Skip `-lcuda` for CI containers without a CUDA driver |
| `TORCH_CUDA_ARCH_LIST`    | Explicit arch list (e.g. `sm_80;sm_90`); auto-detected via `NvidiaBackend.detect_arch_list()` if unset |

### 4.2 AMD ROCm / HIP

`csrc/gpu_backend/rocm/.hipified/` is generated at build time by `RocmBuilder.configure_env()` running `hipify-perl` on the NVIDIA sources (`transfer.cu / transfer.cuh / gtensor_handler.cuh`). It is gitignored.

```bash
# Prerequisites: ROCm PyTorch (torch.version.hip is not None) + ROCm SDK + hipify-perl
FLEXKV_GPU_BACKEND=rocm pip install -e .

# Optional: pin architecture (defaults to gfx90a;gfx942)
PYTORCH_ROCM_ARCH=gfx90a;gfx942 \
FLEXKV_GPU_BACKEND=rocm pip install -e .
```

Output: `flexkv.c_ext` (ROCm build via hipcc). No GDS.

### 4.3 Moore Threads MUSA

```bash
# Prerequisites: torch_musa + MUSA SDK
MUSA_HOME=/usr/local/musa \
FLEXKV_GPU_BACKEND=musa pip install -e .
```

Output: `flexkv.c_ext_musa` (mcc build; coexists with NVIDIA wheel).

### 4.4 Generic (no C extension, CI / CPU-only)

```bash
FLEXKV_GPU_BACKEND=generic pip install -e .
```

No C extension; hot paths fall back to pure PyTorch. Useful for CI, doc builds, or GPU-less containers.

### 4.5 Third-party / CUDA-compatible hardware

If your hardware is **already CUDA-compatible** (e.g. Baidu Kunlunxin via a `libcudart.so` shim), reuse the NVIDIA path:

```bash
FLEXKV_GPU_BACKEND=kunlun pip install -e .   # alias → NvidiaBuilder + NvidiaBackend
```

If the hardware is not CUDA-compatible, add a new vendor following §5. You may also ship the backend class as an **out-of-tree** wheel and register it via the Python entry-point group `flexkv.gpu_backends`; `current_backend` will pick it up during auto-detection without modifying the FlexKV main repo.

### 4.6 Release matrix

| Wheel | Install | C extension | Compiler |
| ----- | ------- | ----------- | -------- |
| `flexkv` / `flexkv-nvidia` | `pip install -e .` *(default)* | `flexkv.c_ext` | nvcc |
| `flexkv-rocm` | `FLEXKV_GPU_BACKEND=rocm pip install -e .` | `flexkv.c_ext` | hipcc |
| `flexkv-musa` | `FLEXKV_GPU_BACKEND=musa MUSA_HOME=… pip install -e .` | `flexkv.c_ext_musa` | mcc |
| `flexkv-cpu`  | `FLEXKV_GPU_BACKEND=generic pip install -e .` | (none) | — |

**No cross-pollution**: building `flexkv-rocm` does not require a CUDA SDK; building `flexkv-nvidia` does not require a ROCm SDK. Vendor source trees under `csrc/gpu_backend/<vendor>/` are mutually independent.

---

## 5. Checklist for Onboarding a New Vendor

Complete the following four steps and your new vendor will work like NVIDIA / ROCm:

1. **C++ implementation** — under `csrc/gpu_backend/<vendor>/`:
   - GPU kernels and host runtime calls (`transfer.cu/.cuh/.h`, etc.);
   - `<vendor>_bindings.cpp` implementing `register_<vendor>_bindings(pybind11::module_&)`, registering at least `transfer_kv_blocks` and `TPTransferThreadGroup`;
   - Add a `#if defined(FLEXKV_BACKEND_<VENDOR>)` branch in `csrc/gpu_backend/backend_bindings.h`;
   - To reuse `tp_transfer_thread_group.cpp`, ensure `gpu_types.h` provides the `gpu_*` macros and `gpu_stream_t / gpuError_t / gpuSuccess` aliases for your vendor.
2. **Python implementation** — `flexkv/gpu_backend/<vendor>/backend.py` containing `class <Vendor>Backend(GpuBackend)`. Implement at least every `@abstractmethod` in groups 1–5 of §3.1. **Don't forget to override `visible_devices_env_vars()`** (a one-line classmethod returning the vendor's visibility-mask names — see §3.7.2); without it, server-client device-id mapping and child-process mask stripping silently degrade to no-ops.
3. **Builder** — `build_backends/<vendor>_builder.py` containing `class <Vendor>Builder(GPUBuilder)` with sources / compile_args / link_args / include_dirs / configure_env. Register the alias in both `build_backends/__init__.py:load_builder` and `flexkv/gpu_backend/__init__.py:_FORCE_MAP / _BUILTIN_BACKENDS`.
4. **Optional IPC / GDS** — if the hardware supports them, override `supports_ipc / get_ipc_handle / open_ipc_handle / close_ipc_handle` (see §3.6) or `supports_direct_storage / gds_create_manager`, and expose the corresponding C++ classes from your vendor binding.
5. **GDR (cross-node RDMA) is automatic** — you do **not** need to add a GDR hook. As long as steps 1–4 are correct, `MoonCakeTransferEngineWrapper` (GPU-vendor neutral) will register the pinned / device buffer through libibverbs without any vendor-specific code. See §3.5 for the rationale.

Throughout the process, only `csrc/bindings.cpp`, `flexkv/gpu_backend/interface.py`, and `flexkv/gpu_backend/__init__.py` ever need to be touched in the FlexKV upper layers — **no other upper-layer code requires changes**.
