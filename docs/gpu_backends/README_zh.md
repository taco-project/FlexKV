# GPU Backend 接入指南

本文档面向希望为 FlexKV 新增 GPU/NPU 厂商支持的开发者。

---

## 一、背景

FlexKV 采用分层 GPU 抽象架构，将所有厂商相关代码隔离在各自的 backend 模块中，核心逻辑与硬件无关。目前内置三个 backend：

| Backend | 硬件 | 状态 |
|---------|------|------|
| `nvidia` | NVIDIA A/H/B 系列（CUDA） | 生产就绪（默认） |
| `musa` | 摩尔线程 MTT S 系列（MUSA） | 实验性 |
| `generic` | 任意平台（纯 PyTorch CPU fallback） | 可用 |

如需接入新厂商（昇腾、寒武纪、其他 CUDA 兼容厂商等），按本文档操作即可。

---

## 二、五大设计原则

在开始接入之前，请先理解 FlexKV GPU Backend 的核心设计原则，所有 backend 实现必须遵守：

**原则一：代码按厂商集中**

每个厂商的所有实现（C++ kernel、Python 适配、CMakeLists）集中到自己的 `gpu_backend/<vendor>/` 目录，互不干扰。不要把厂商特有代码散落在公共路径中。

**原则二：GpuBackend 基类统一对外抽象**

上层代码（`worker.py`、`transfer_engine.py` 等）只调用 `GpuBackend` 抽象接口，不出现任何 `cuda*`、`musa*`、`acl*` 等厂商专有 API。

**原则三：独立 wheel 包按需安装**

通过 `FLEXKV_GPU_BACKEND=<vendor>` 编译得到的产物应可以打包为独立的 wheel（如 `flexkv-nvidia-*.whl`），用户只安装自己需要的版本，编译时不依赖其他厂商的 SDK。

**原则四：优先兼容 CUDA 生态，沿用 nvidia backend**

> **这是最重要的一条原则，请先阅读本条再决定是否需要新建 backend。**

很多第三方厂商（如百度昆仑芯等）已通过适配层兼容了 CUDA 生态，`torch.cuda.is_available()` 和 `import flexkv.c_ext` 在这些平台上均能正常工作。**对于此类厂商，直接沿用 `NvidiaBackend` 即可，无需新增任何代码**，只需在 `_ENV_MAP` 中添加别名：

```python
# flexkv/gpu_backend/__init__.py
_ENV_MAP = {
    ...
    "kunlun": "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",  # 兼容 CUDA 生态
}
```

判断标准：在目标硬件上执行以下两条命令，若均输出正常，则视为 CUDA 生态兼容：

```bash
python3 -c "import torch; print(torch.cuda.is_available())"  # True
python3 -c "import flexkv.c_ext; print('ok')"                # ok
```

只有 CUDA 生态覆盖不到的厂商（使用完全独立 runtime，如 MUSA、AscendCL 等）才需要新建 backend。

**原则五：性能优先，抽象层不引入额外开销**

所有 `GpuBackend` 方法均为 thin wrapper，不在热路径上做额外的 Python 层调度、加锁或内存拷贝。关键路径（`transfer_kv_blocks`、`layout_transform`、`host_register`）直接调用底层 C extension 或 runtime 库。抽象层引入的额外调用开销应在 μs 量级以内。

---

## 三、整体架构

```
FlexKV 应用层 (Python)
        │
        ▼
flexkv/gpu_backend/          ← Python 抽象层
  interface.py               ← GpuBackend ABC（15 个抽象方法）
  __init__.py                ← 自动探测 + current_backend 单例
  nvidia/backend.py          ← NvidiaBackend
  musa/backend.py            ← MusaBackend
  generic/backend.py         ← GenericBackend
        │
        ▼
csrc/gpu_compat.h            ← C++ 编译时宏切换
csrc/gpu_backend/
  nvidia/                    ← CUDA kernels + cuFile GDS
  musa/                      ← MUSA kernels + muFile GDS
  generic/                   ← CPU fallback stub
        │
        ▼
build_backends/              ← 编译体系（每厂商一个 Builder）
  base.py                    ← GPUBuilder ABC
  cuda_builder.py
  musa_builder.py
  generic_builder.py
```

---

## 四、Backend 自动探测逻辑

FlexKV 在运行时按以下优先级选择 backend：

1. **环境变量** `FLEXKV_GPU_BACKEND`（nvidia / musa / generic / ...）
2. **entry_points 插件**（`flexkv.gpu_backends` group，第三方包注册）
3. **内置探测**（依次尝试 Nvidia → MUSA）
4. **默认**：`NvidiaBackend`

接入新厂商时，推荐使用 **entry_points 插件方式**，无需修改 FlexKV 主仓库代码。

---

## 五、新增 Backend：分步说明

> 在开始之前，请确认你的厂商确实需要新建 backend（参见原则四）。

### 5.1 Python 层（`flexkv/gpu_backend/<vendor>/`）

新建目录并实现 `GpuBackend` ABC（`flexkv/gpu_backend/interface.py`）。

以 **NvidiaBackend** 为例：

```python
# flexkv/gpu_backend/nvidia/backend.py

import ctypes
import torch
from flexkv.gpu_backend.interface import GpuBackend

class NvidiaBackend(GpuBackend):
    """NVIDIA CUDA backend — production default implementation"""

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
        # Calls libcudart cudaIpcGetMemHandle, returns 64-byte handle
        handle = (ctypes.c_byte * 64)()
        ret = ctypes.CDLL("libcudart.so").cudaIpcGetMemHandle(
            handle, ctypes.c_void_p(tensor.data_ptr()))
        if ret != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed with error code {ret}")
        return bytes(handle)

    def open_ipc_handle(self, handle, shape, dtype, device_id):
        # Calls libcudart cudaIpcOpenMemHandle, reconstructs as Tensor
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

**关键点**：
- 实现 `interface.py` 中全部抽象方法（否则实例化时抛 `TypeError`）
- 不支持的能力（如 GDS）返回 `False` / 抛 `NotImplementedError`，不要静默忽略
- IPC 不可用时，使用 `SHM fallback`（见 `generic/backend.py`）

### 5.2 注册 Backend

**方式 A：内置（修改主仓库）**

编辑 `flexkv/gpu_backend/__init__.py`：

```python
_ENV_MAP = {
    "nvidia": "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",
    "cuda":   "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",
    "musa":   "flexkv.gpu_backend.musa.backend.MusaBackend",
    "generic":"flexkv.gpu_backend.generic.backend.GenericBackend",
    # 新增厂商（CUDA 生态兼容，直接指向 NvidiaBackend）：
    "kunlun": "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",
    # 新增厂商（独立 runtime，需要自建 backend）：
    "vendor_x": "flexkv.gpu_backend.vendor_x.backend.VendorXBackend",
}
```

**方式 B：entry_points 插件（推荐，零侵入）**

在厂商自己的 Python 包 `pyproject.toml` 中：

```toml
[project.entry-points."flexkv.gpu_backends"]
vendor_x = "flexkv_vendor_x.backend:VendorXBackend"
```

FlexKV 启动时会自动发现并加载，无需改动主仓库。

### 5.3 C++ 层（`csrc/gpu_backend/<vendor>/`）

如果新厂商有自定义 GPU kernel，参照 `csrc/gpu_backend/nvidia/` 的目录结构：

```
csrc/gpu_backend/<vendor>/
  CMakeLists.txt
  transfer.<ext>          # GPU kernel（.cu/.mu/.bang/...）
  layout_transform.<ext>  # GDS layout kernel（无 GDS 支持则省略）
  gds_manager.<ext>       # Direct Storage 管理（可选）
  tp_transfer_thread_group.cpp
```

`csrc/gpu_compat.h` 提供编译时宏切换：

```cpp
// csrc/gpu_compat.h（节选）
#if defined(FLEXKV_BACKEND_MUSA)
  #include <musa_runtime.h>
  using gpu_stream_t = musaStream_t;
  #define gpu_current_stream()   flexkv::musa_current_stream()
  #define gpu_get_last_error()   musaGetLastError()
  #define GPU_SUCCESS            musaSuccess
#else  // 默认 CUDA
  #include <cuda_runtime.h>
  #include <ATen/cuda/CUDAContext.h>
  using gpu_stream_t = cudaStream_t;
  #define gpu_current_stream()   at::cuda::getCurrentCUDAStream()
  #define gpu_get_last_error()   cudaGetLastError()
  #define GPU_SUCCESS            cudaSuccess
#endif
```

新厂商在此文件中添加对应的 `#elif defined(FLEXKV_BACKEND_XXX)` 分支即可。

### 5.4 编译 Builder（`build_backends/<vendor>_builder.py`）

继承 `build_backends/base.py` 的 `GPUBuilder` ABC：

```python
# build_backends/base.py（接口）
class GPUBuilder(ABC):
    @abstractmethod
    def get_extension_class(self): ...   # 返回 Extension 类
    @abstractmethod
    def get_sources(self, **kw) -> list: ...   # 源文件列表
    @abstractmethod
    def get_compile_args(self, **kw) -> dict: ...
    @abstractmethod
    def get_link_args(self, **kw) -> list: ...
    @abstractmethod
    def get_build_ext_class(self): ...
    def configure_env(self): ...         # 可选：编译前环境配置
```

参照 `build_backends/cuda_builder.py` 实现对应厂商的 Builder，然后在 `setup.py` 的 `_BUILDER_MAP` 中注册：

```python
_BUILDER_MAP = {
    "cuda":    "build_backends.cuda_builder.CUDABuilder",
    "musa":    "build_backends.musa_builder.MUSABuilder",
    "generic": "build_backends.generic_builder.GenericBuilder",
    # 新增：
    "vendor_x": "build_backends.vendor_x_builder.VendorXBuilder",
}
```

---

## 六、GDS（Direct Storage）支持说明

GDS（GPU Direct Storage）允许文件数据绕过 CPU 直接读写 GPU 显存，可显著降低 IO 延迟。

| 平台 | 方案 | 备注 |
|------|------|------|
| NVIDIA | cuFile GDS | 生产级，需 `FLEXKV_STORAGE_BACKEND=cufile` |
| 其他平台 | POSIX fallback（pread/pwrite + H2D memcpy） | 自动降级，功能等价，性能较低 |

**降级说明**：不支持 GDS 的厂商只需在 `gds_available()` 中返回 `False`，FlexKV 会自动走 POSIX IO 路径，无需任何额外处理，功能不受影响。

**建议实现**：如果你的平台有类似 GDS 的 Direct Storage 能力（GPU 直接 DMA 读写持久化存储），强烈建议实现对应的 `gds_read` / `gds_write` 接口，可以大幅提升 KV Cache 换入换出的吞吐量。参考 `csrc/gpu_backend/nvidia/gds_manager.cpp` 的实现方式。

---

## 七、IPC（跨进程显存共享）支持说明

IPC（Inter-Process Communication）允许多个进程直接共享同一块 GPU 显存，无需拷贝，是 FlexKV 分布式 KV Cache 的关键能力。

| 平台 | API | 支持程度 |
|------|-----|---------|
| NVIDIA | `cudaIpcGetMemHandle` | 完整 |
| AMD ROCm | `hipIpcGetMemHandle` | 基本兼容 |
| 其他 | 无原生等价 | 降级为 SHM fallback |

**降级说明**：不支持原生 IPC 的平台会自动降级为基于 `multiprocessing.shared_memory` 的 SHM fallback，数据通过 CPU 中转（GPU→CPU→GPU），功能保持完整，但会引入一次额外的内存拷贝开销。参考 `generic/backend.py` 中的 `SHMFallbackMixin` 实现。

**建议实现**：如果你的平台支持跨进程显存共享（即便 API 名称与 CUDA 不同），强烈建议实现原生 IPC 接口，可以消除 SHM fallback 的额外拷贝开销，尤其在高并发场景下性能差异显著。

---

## 八、编译与测试

```bash
# NVIDIA CUDA + GDS（默认推荐）
FLEXKV_GPU_BACKEND=cuda FLEXKV_STORAGE_BACKEND=cufile pip install -e .

# NVIDIA CUDA + POSIX IO（不需要 cuFile/GDS）
FLEXKV_GPU_BACKEND=cuda FLEXKV_STORAGE_BACKEND=posix pip install -e .

# 新厂商（以 AMD ROCm 为例）
FLEXKV_GPU_BACKEND=rocm pip install -e .

# 运行 backend 单元测试
cd FlexKV
PYTHONPATH=. python3 -m pytest tests/test_gpu_backend.py --noconftest -v
```

测试覆盖：
- `TestGpuBackendInterface`：ABC 完整性 + 不可直接实例化
- `TestNvidiaBackend`：is_available、device_name、stream、IPC、transfer（需 CUDA GPU，否则 skip）
- `TestGenericBackend`：11 个用例，无 GPU 环境下全部通过
- `TestAutoDetection`：环境变量覆盖（generic / cuda / rocm / 非法值）

---

## 九、常见问题

**Q：我的厂商硬件兼容 CUDA，需要新建 backend 吗？**

不需要。只要 `torch.cuda.is_available()` 和 `import flexkv.c_ext` 均正常，直接在 `_ENV_MAP` 中将你的厂商名称映射到 `NvidiaBackend` 即可（参见原则四）。

**Q：我只想适配一块新卡，但没有自定义 kernel，能不能直接用？**

可以。继承 `GenericBackend` 并覆盖设备查询类方法即可，`transfer_kv_blocks` 自动降级为 `torch` 原生 `copy_`，GDS 和 IPC 也会走各自的 fallback 路径。

**Q：entry_points 插件和内置 backend 有什么区别？**

功能完全等价。entry_points 插件由厂商自己的包维护，不依赖 FlexKV 主仓库发版，推荐第三方厂商使用此方式。

**Q：`FLEXKV_GPU_BACKEND` 和 `FLEXKV_STORAGE_BACKEND` 可以独立设置吗？**

可以。两者正交：`FLEXKV_GPU_BACKEND` 控制计算和内存，`FLEXKV_STORAGE_BACKEND` 控制 IO 路径。
