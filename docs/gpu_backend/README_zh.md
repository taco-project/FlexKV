# FlexKV GPU Backend 接入指南

> 本文档面向**适配新 GPU 后端**的工程同学，给出 FlexKV 的当前抽象边界与接入流程。
>
> 内容只描述**当前代码的最终形态**，不包含历史变更、不包含迁移过程。
>
> 英文版见 [README_en.md](./README_en.md)。

---

## 1. 五大原则

| # | 原则 | 含义 |
| - | ---- | ---- |
| 1 | **代码按厂商集中** | 所有 GPU 厂商相关实现都放在 `csrc/gpu_backend/<vendor>/` 与 `flexkv/gpu_backend/<vendor>/` 两棵子树里，互不依赖。FlexKV 上层（`worker.py / transfer_engine.py / cache_engine.py / common/memory_handle.py / kvtask.py / storage/allocator.py / integration/* / kvmanager.py`）不出现任何 `cuda*` / `hip*` / `musa*` 字面调用。 |
| 2 | **统一抽象接口** | Python 侧通过 `GpuBackend` ABC 暴露能力，C++ 侧通过 `csrc/gpu_backend/backend_bindings.h` 暴露 pybind 注册函数。上层只调用 `current_backend.xxx()`、只 import `flexkv.c_ext` 暴露的 vendor-agnostic 符号。 |
| 3 | **每个厂商独立 wheel，按需安装** | 每次构建只编译一个厂商的 C 扩展，由环境变量 `FLEXKV_GPU_BACKEND` 决定。可分别发布 `flexkv-nvidia` / `flexkv-rocm` / `flexkv-musa` 等 wheel；编译 ROCm 时不需要 CUDA SDK，反之亦然。 |
| 4 | **优先兼容 CUDA 生态** | 已经兼容 CUDA 接口的第三方厂商（如百度昆仑芯）直接复用 `NvidiaBackend / NvidiaBuilder`，无需新建 backend。配置 `FLEXKV_GPU_BACKEND=kunlun` 等 alias 即可。 |
| 5 | **性能优先，零额外开销** | 抽象层是 thin wrapper：hot path（`transfer_kv_blocks` / `layout_transform` / `register_host_tensor`）必须是单层 `return self._ext.xxx(...)` 转发，不在 Python 层做 dict 查表/锁/拷贝；C++ 侧编译期通过 `FLEXKV_BACKEND_<VENDOR>` 宏切换 vendor 实现，无运行时分支。 |

> **命名说明**：`csrc/gpu_backend/nvidia/gtensor_handler.cuh` 中的 `enum class BackendType { VLLM, TRTLLM, SGLANG }` 指的是**LLM 框架后端**（vLLM / TRT-LLM / SGLang），与 GPU 厂商无关。本文档统一用 `GpuVendor` 表示 GPU 硬件厂商。

---

## 2. 目录结构

```
FlexKV/
├── flexkv/
│   └── gpu_backend/                       ← Python 抽象层
│       ├── __init__.py                    ← 自动探测 + current_backend 单例
│       ├── interface.py                   ← GpuBackend ABC + GpuVendor enum
│       ├── nvidia/backend.py              ← NvidiaBackend
│       ├── rocm/backend.py                ← RocmBackend
│       ├── musa/backend.py                ← MusaBackend
│       └── generic/backend.py             ← GenericBackend（PyTorch fallback）
│
├── csrc/
│   ├── bindings.cpp                       ← GPU-vendor agnostic：仅注册
│   │                                        CPU 跨厂商绑定（SSD/Hasher/
│   │                                        RadixTree/Pcfs/Redis/...），
│   │                                        最后调一行
│   │                                        register_active_backend_bindings(m)
│   ├── hash.cpp / radix_tree.cpp /
│   ├── transfer_ssd.cpp /
│   ├── eviction_strategy.cpp /
│   ├── pcfs/* /
│   ├── monitoring/* /
│   ├── dist/*                             ← 与 GPU 无关，跨厂商共用
│   │
│   └── gpu_backend/                       ← C++ 抽象层
│       ├── backend_bindings.h             ← register_<vendor>_bindings(m) 接口
│       ├── gpu_types.h                    ← gpu_stream_t / gpuMallocHost /
│       │                                    gpuStreamCreate / gpuSetDevice /
│       │                                    gpuGetLastError / gpuSuccess ...
│       │                                    一组跨厂商宏与类型别名
│       ├── nvidia/
│       │   ├── nvidia_bindings.cpp        ← register_nvidia_bindings(m)
│       │   ├── transfer.cu / transfer.cuh
│       │   ├── gtensor_handler.cuh
│       │   ├── tp_transfer_thread_group.cpp/.h  ← 用 gpu_* 宏，跨厂商共用源
│       │   └── gds/                       ← cuFile / GDS 实现
│       │       ├── gds_manager.cpp/.h
│       │       ├── tp_gds_transfer_thread_group.cpp/.h
│       │       └── layout_transform.cu/.cuh
│       ├── rocm/
│       │   ├── rocm_bindings.cpp          ← register_rocm_bindings(m)
│       │   └── .hipified/                 ← 构建时由 hipify-perl 自动生成
│       ├── musa/
│       │   └── transfer.mu
│       └── generic/                       ← 预留
│
├── build_backends/                        ← 编译期 vendor 分发
│   ├── __init__.py                        ← load_builder(name) 入口
│   ├── base.py                            ← GPUBuilder ABC
│   ├── cuda_builder.py                    ← NvidiaBuilder
│   ├── rocm_builder.py                    ← RocmBuilder
│   ├── musa_builder.py                    ← MusaBuilder（骨架）
│   └── generic_builder.py                 ← GenericBuilder（无 C 扩展）
│
└── setup.py                               ← 读取 FLEXKV_GPU_BACKEND，
                                             load_builder(name) → 编译一个 vendor
```

**关键不变量**

- `bindings.cpp` 不 `#include` 任何 vendor 头文件、不调用任何 `cuda*` / `hip*` API；NVIDIA/ROCm 专属绑定全部在各自 `<vendor>_bindings.cpp` 中。
- `flexkv.c_ext` 是统一的 Python 扩展模块名（NVIDIA wheel 与 ROCm wheel 都暴露这个名字）；MUSA 由于工具链不同，使用独立模块名 `flexkv.c_ext_musa`。
- 上层代码**永远**不直接 `import flexkv.c_ext`，统一通过 `current_backend` 间接调用。
- `csrc/transfer_ssd.cpp` 等 SSD/Pcfs/Redis/RadixTree 全部是 GPU-agnostic，由 `bindings.cpp` 统一注册，所有厂商 wheel 共用。

---

## 3. 抽象类与其作用

抽象在三个层面同时存在：**Python 运行时**（`GpuBackend`）、**C++ pybind 绑定**（`backend_bindings.h`）、**编译期**（`GPUBuilder`）。

### 3.1 `flexkv.gpu_backend.GpuBackend`（Python 运行时抽象）

`flexkv/gpu_backend/interface.py` 定义的 ABC，是上层与厂商实现之间的**唯一接口**。每个厂商在 `flexkv/gpu_backend/<vendor>/backend.py` 中提供一个 `class <Vendor>Backend(GpuBackend)`。

按职责分 7 组：

| # | 组别 | 主要方法 | 作用 |
| - | ---- | -------- | ---- |
| 1 | **元信息** | `is_available()` / `device_name()` / `torch_device_type()` / `vendor` | 自动探测、`torch.device(...)` 构造与 tensor 归属判断（NVIDIA/ROCm 都是 `"cuda"`，MUSA 是 `"musa"`）。 |
| 2 | **设备管理** | `set_device / current_device / device_count / synchronize / is_initialized / init_runtime / empty_cache / is_gpu_tensor / get_device_capability / make_device / detect_arch_list` | 替换上层所有 `torch.cuda.*` 调用；`detect_arch_list` 供 builder 在编译期填 arch 列表。 |
| 3 | **流管理** | `create_stream / destroy_stream / get_current_stream / stream_handle / stream_context` | `stream_handle` 把 stream 转成 uintptr_t 供 C 扩展使用；`stream_context` 是 `with` 上下文。 |
| 4 | **锁页主机内存** | `register_host_tensor / unregister_host_tensor / alloc_pinned / free_pinned` | 替换 `cudaHostRegister / cudaHostUnregister`；用 ctypes 直接调底层运行时（`libcudart.so` / `libamdhip64.so` / `libmusa.so`）。 |
| 5 | **Hot path（KV 传输）** | `transfer_kv_blocks / transfer_kv_blocks_ssd / layout_transform` | 单层转发到 `flexkv.c_ext.*`；`transfer_kv_blocks_ssd` 在 ABC 中已给出默认实现，所有厂商共用。 |
| 6 | **跨进程显存共享 (IPC)** | `supports_ipc / get_ipc_handle / open_ipc_handle / close_ipc_handle` | `cudaIpc* / hipIpc* / musaIpc*` 二进制 handle 互不兼容，每个厂商独立实现；不支持 IPC 的 backend 返回 `False`。详见 §3.6。 |
| 7 | **Direct Storage** | `supports_direct_storage / gds_create_manager` | NVIDIA 走 cuFile / GDS；其他厂商目前均无对等能力，返回 `False`，调用方退化到 POSIX/io_uring 路径（`transfer_kv_blocks_ssd`，与 GPU 无关）。 |

`GpuVendor` 枚举：`NVIDIA / ROCM / MUSA / GENERIC`，仅作元数据用。

> 除上述 7 组外，`GpuBackend` 还在第 2 组中暴露了一组**可见性环境变量**抽象
> （`visible_devices_env_vars / get_visible_device_map / strip_visible_devices`），
> 用来彻底取代上层硬编码的 `CUDA_VISIBLE_DEVICES` 字面量。详见 §3.7。

### 3.2 `csrc/gpu_backend/backend_bindings.h`（C++ pybind 注册抽象）

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

作用：

- `bindings.cpp` 只调用 `register_active_backend_bindings(m)`，**完全不感知**当前是哪个 vendor。
- 每个 vendor 在 `csrc/gpu_backend/<vendor>/<vendor>_bindings.cpp` 中实现自己的 `register_<vendor>_bindings`，注册 `transfer_kv_blocks` / `TPTransferThreadGroup` / 以及（NVIDIA 限定）`GDSManager / TPGDSTransferThreadGroup / transfer_kv_blocks_gds` 等。
- 选择哪一个由 builder 在命令行注入的 `-DFLEXKV_BACKEND_<VENDOR>` 宏决定，编译期分发，零运行时开销。

### 3.3 `csrc/gpu_backend/gpu_types.h`（C++ 跨厂商类型/宏）

把厂商运行时 API 统一抽象到一组 `gpu_*` 名字：

```cpp
gpu_stream_t / gpuError_t / gpuSuccess
gpuMallocHost  gpuFreeHost
gpuSetDevice   gpuStreamCreate
gpuGetLastError gpuGetErrorString
```

这样像 `tp_transfer_thread_group.cpp` 这样**逻辑跨厂商但要调运行时的源文件**只需写一份，由 `-DFLEXKV_BACKEND_<VENDOR>` 决定底层映射到 cuda*/hip*/musa*。

### 3.4 `build_backends.GPUBuilder`（编译期抽象）

`build_backends/base.py` 定义的 ABC。`setup.py` 调用 `load_builder(name)` 拿到一个 builder 实例，然后向它询问：

| 方法 | 作用 |
| ---- | ---- |
| `is_available()` | 当前环境是否能用此 builder（如 `torch.version.hip is not None`）。 |
| `get_extension_class()` | 返回 `CUDAExtension` / `MUSAExtension` / `Extension`。 |
| `get_extension_name()` | 返回 `"flexkv.c_ext"` 或 `"flexkv.c_ext_musa"`，决定生成的 `.so` 名。 |
| `get_sources(**opts)` | 返回当前 vendor 的所有源文件清单（`.cpp/.cu/.mu/.hipified/...`）。 |
| `get_compile_args(**opts)` | 返回 `{"cxx": [...], "nvcc": [...]}`，自动注入 `vendor_macro()`（即 `-DFLEXKV_BACKEND_<VENDOR>`）。 |
| `get_link_args(**opts)` | 链接库列表（`-lxxhash` / `-lcuda` / `-lcufile` / `-lhiredis` / `-lhifs_client_sdk` / ...）。 |
| `get_include_dirs(**opts)` | include 路径，至少包含 `csrc/` 与 `csrc/gpu_backend/`。 |
| `get_build_ext_class()` | 返回 `BuildExtension` 子类。 |
| `configure_env()` | 编译前的环境变量副作用，例如 `TORCH_CUDA_ARCH_LIST` / `PYTORCH_ROCM_ARCH` 探测，或触发 hipify。 |
| `vendor_macro()` | 返回 `-DFLEXKV_BACKEND_<NAME>`，与 `gpu_types.h` / `backend_bindings.h` 对齐。 |

### 3.5 GDS 与 GDR ——两条「GPU-direct」路径如何抽象

FlexKV 有**两条独立的「GPU-direct」快路径**，名字相近但层次完全不同，抽象方式
也不一样。先搞清楚它们各自落在哪一层，能避免新厂商在错误的位置实现错误的东
西。

| 路径  | 直连对象               | 与 GPU 厂商的耦合              | 在 FlexKV 中的实现位置                                       |
| ----- | ---------------------- | ------------------------------ | ------------------------------------------------------------ |
| GDS   | GPU 显存 ⇄ 本地 SSD    | **强耦合**（cuFile）          | `csrc/gpu_backend/<vendor>/gds/` + `<vendor>_bindings.cpp`   |
| GDR   | GPU/CPU 内存 ⇄ 远端 NIC | **弱耦合**——RDMA 侧厂商无关 | `flexkv/mooncakeEngineWrapper.py` + Mooncake transfer engine |

#### 3.5.1 GDS（GPU Direct Storage）——厂商私有，编译期 opt-in

GDS 即 cuFile：NVIDIA 把 SSD 数据 RDMA 直推到设备显存里。其他厂商目前**没有
对等能力**（ROCm 没有 cuFile，MUSA 没有 direct-storage SDK，Generic 没有
GPU），因此 GDS 必须保持厂商私有。

抽象分三层，全部走**能力探测**，绝不假定存在：

1. **Python 接口**：`GpuBackend.supports_direct_storage()` 与
   `GpuBackend.gds_create_manager(...)`（§3.1 第 7 组）。默认实现返回
   `False` / 抛 `NotImplementedError`；只有 `NvidiaBackend` 在 `flexkv.c_ext`
   带 `FLEXKV_ENABLE_GDS=1` 编译时才覆写它们。上层代码**必须**先判断
   `current_backend.supports_direct_storage()`，再决定是否调用
   `gds_create_manager()`。
2. **C++ 绑定**：`GDSManager / TPGDSTransferThreadGroup /
   transfer_kv_blocks_gds` 等符号都注册在 `register_nvidia_bindings(m)` 内的
   `#ifdef FLEXKV_ENABLE_GDS` 分支里。`csrc/bindings.cpp` 与其他厂商的
   `register_<vendor>_bindings` 完全不感知 cuFile。ROCm / MUSA / Generic
   wheel 不暴露这些符号。
3. **源码布局**：所有 GDS 源文件都放在 `csrc/gpu_backend/nvidia/gds/` 下
   （`gds_manager.{h,cpp}`、`tp_gds_transfer_thread_group.{h,cpp}`、
   `layout_transform.{cuh,cu}`）。其他厂商**不要**在自己目录下复制一份
   同名子目录，除非硬件确实有 cuFile-equivalent SDK；否则走下方 fallback。

**非 GDS 厂商的回退路径**：`transfer_kv_blocks_ssd`（由 `csrc/bindings.cpp`
注册，与 GPU 无关，POSIX / io_uring 实现）。上层通过
`current_backend.transfer_kv_blocks_ssd(...)` 调用，hot path 内**没有运行时
分支**——worker 在构造期就根据 `supports_direct_storage()` 选好走 GDS
worker 还是走 SSD worker。

#### 3.5.2 GDR（GPU Direct RDMA）——厂商无关，运行期探测

跨节点 KV-cache 复用走的是
[Mooncake transfer engine](https://github.com/kvcache-ai/Mooncake)，本质是
libibverbs 的薄封装。RDMA NIC 直接 DMA 进 Mooncake `register_memory(ptr,
size)` 注册的 buffer——这块 buffer 究竟落在 CPU pinned 内存还是 GPU 显存里，
**对 FlexKV 抽象层是透明的**。Mooncake 只关心「指针 + 长度」，buffer 怎么来
的是 vendor backend 的事。

因此 GDR **不需要**新增一个厂商 C++ 绑定，也**不需要**在 `GpuBackend` 上加新
抽象方法。接缝小到这种程度：

1. **Buffer 的产生统一走 `GpuBackend`。** 上层凡是要把某块 buffer 后续交给
   Mooncake 的，buffer 必须从抽象层拿：
   - CPU pinned 中转 buffer → `GpuBackend.alloc_pinned` /
     `register_host_tensor`（第 4 组）。
   - GPU 设备 buffer（真正走 GPU-direct RDMA）→ 普通的
     `torch.empty(..., device=current_backend.make_device(idx))` 加上
     `current_backend.stream_handle(...)` 做同步。
   - 跨进程 GPU buffer → `get_ipc_handle / open_ipc_handle`（第 6 组，仅在
     `supports_ipc()` 为 True 的 backend 上）。
2. **`MoonCakeTransferEngineWrapper.regist_buffer(ptr, size)` 与厂商无关**：
   只是转发到 `engine.register_memory(ptr, size)`。这个 wrapper 天然是
   GPU-vendor 中立的；`flexkv/gpu_backend/<vendor>/` 下没有任何文件 import
   它。
3. **「走 GDR 还是走中转拷贝」的决策落在 worker，不落在抽象层**：
   `flexkv/transfer/worker.py` 在构造期根据 config（`enable_p2p_cpu /
   enable_p2p_ssd`）和 `MOONCAKE_CONFIG_PATH` 选择启用
   `PEER2CPUTransferWorker`（Mooncake）还是回退到 CPU 中转 + 标准
   SSD/CPU 路径。hot path 中**没有**任何 vendor-specific 的 GDR 调用。

新厂商接入时的实际意义：**你不需要新增 GDR 钩子**。只要 `GpuBackend` 把
1–4 组（pinned 主机内存与设备指针）实现正确，必要时实现 6 组（IPC，用于跨
进程 GPU RDMA），Mooncake 就能直接工作。NIC 发现 / GID / PD 创建完全是
Mooncake 自己的事。

#### 3.5.3 决策矩阵

| 场景                                          | 运行时走的路径                              | 新厂商要实现什么                                                  |
| --------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------- |
| 本地 SSD，硬件有 cuFile                       | GDS（`gds_create_manager`）                 | `csrc/gpu_backend/<vendor>/gds/*` + 在 binding 里 `#ifdef` 暴露   |
| 本地 SSD，硬件无 cuFile 等价物                | POSIX/io_uring（`transfer_kv_blocks_ssd`）  | 什么都不做——继承默认实现                                         |
| 跨节点，NIC 支持 GPU Direct RDMA              | Mooncake GDR（`register_memory(gpu_ptr)`）  | `GpuBackend` 无需修改；如需跨进程则确认 `supports_ipc` 正确       |
| 跨节点，NIC 只支持 CPU pinned RDMA            | Mooncake on pinned 中转 buffer              | 确认 `register_host_tensor / alloc_pinned` 实现正确               |

### 3.6 跨进程显存共享（IPC）—— 把 KV-cache 从训练进程注入 FlexKV server

FlexKV 的多实例 / server-client 模式里，**LLM 推理框架的 worker 进程**（vLLM / TRT-LLM / SGLang）持有
真正的 KV-cache 显存，而 **FlexKV server / TransferManager 是另一个独立进程**。要让 server 直接读写
worker 已分配好的 KV-cache，必须把 GPU 指针**跨进程**传递过去，这就是 IPC 抽象要解决的问题。所有相关逻
辑都汇聚在 [`flexkv/common/memory_handle.py`](../../flexkv/common/memory_handle.py) 里的 `TensorSharedHandle`，**它是 FlexKV 中唯一接触 vendor IPC 原语的 Python 文件**。

#### 3.6.1 两条路径

`TensorSharedHandle.__init__(force_direct_ipc: bool)` 在构造期决定走哪条路径：

| 路径 | 构造方式 | 还原方式 | 何时使用 |
| ---- | -------- | -------- | -------- |
| **PyTorch reductions（默认快路径）** | `torch.multiprocessing.reductions.reduce_tensor(tensor)` | `rebuild_func(*rebuild_args)` | `force_direct_ipc=False`，无需 backend 私有 IPC API，跨厂商通用 |
| **Vendor IPC handle（直路径）** | `_gpu_backend.get_ipc_handle(tensor)` | `_gpu_backend.open_ipc_handle(handle, dtype, shape, strides, storage_offset, device_id)` | `force_direct_ipc=True`，绕过 PyTorch reductions 的引用计数与 cleanup hook，由调用方自管生命周期（用于 transfer_engine 这种长生命周期注册） |

两条路径**对上层完全透明**：调用方只看到 `TensorSharedHandle.create_tensor()` 还原出的 `torch.Tensor`，不
关心底层是哪一条。

#### 3.6.2 抽象类暴露的 5 个 API（`GpuBackend` 第 6 组）

```python
def supports_ipc(self) -> bool:
    """硬件 / 运行时是否支持跨进程显存共享。"""

def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
    """把当前进程持有的 GPU tensor 序列化成不透明字节串（CUDA 64B / HIP 64B / MUSA ?B），
    可通过 ZeroMQ / pipe / 共享文件传给目标进程。"""

def open_ipc_handle(
    self,
    handle: bytes,
    dtype: torch.dtype,
    shape: Tuple[int, ...],
    strides: Tuple[int, ...],
    storage_offset: int,
    device_id: int,
) -> torch.Tensor:
    """在目标进程内还原 tensor。注意 device_id 必须是**物理 GPU id**（参见 §3.7）。"""

def close_ipc_handle(self, ptr: int) -> None:
    """显式释放 open_ipc_handle 占用的 mapping；由 TensorSharedHandle 析构 / 显式 close 调用。"""
```

加上 `is_gpu_tensor`（第 2 组），构成 IPC 路径所需的全部抽象——**`memory_handle.py` 不出现任何 `cuda*` /
`hip*` / `musa*` 字面量**。

#### 3.6.3 关键约束与跨厂商注意事项

1. **二进制 handle 不可跨厂商**：CUDA 的 `cudaIpcMemHandle_t`（64B）、HIP 的 `hipIpcMemHandle_t`（64B）、
   MUSA 的同名结构虽然大小可能相同，但**不保证字段布局兼容**，更不保证能跨厂商打开。生产/还原必须由
   **同一个 backend** 完成；FlexKV 的 `current_backend` 是 process-local 单例，天然满足这一约束。
2. **必须使用物理 device id**：`open_ipc_handle(..., device_id=N)` 的 `N` 是**物理 GPU 编号**。当 worker 进
   程被 `CUDA_VISIBLE_DEVICES` 屏蔽过、而 server 进程能看到所有 GPU 时，必须先经过 §3.7 的可见性映射把
   逻辑 id 翻译成物理 id，否则会打开到错误的卡上。`server.py` / `transfer_manager.py` 已经通过
   `_gpu_backend.strip_visible_devices(env)` 让子进程看到全部 GPU，这是前置条件。
3. **不支持 IPC 的 backend 怎么办**：`supports_ipc()` 返回 `False` 的 backend（如 `GenericBackend` 或某些
   只有单卡的硬件），在 `get_ipc_handle / open_ipc_handle / close_ipc_handle` 上 fallback 到抽象基类的默
   认实现——**直接抛 `RuntimeError("... does not support IPC")`**。上层 server-client 模式在
   `is_gpu_tensor()` 这一关就会被拒，永远走不到 IPC 调用，因此可以安全地不实现这三个方法。
4. **快路径与直路径的等价性**：reductions 路径底层同样调用 vendor IPC，但走 PyTorch 自己的进程间共享内
   存（`torch.cuda.Event` / `cudaIpcOpenEventHandle` 等）做引用计数；直路径绕过这个机制，**对长生命周期
   的 KV-cache 注册更稳定**（PyTorch 的 cleanup hook 在某些 multiprocessing 拓扑下会过早释放 mapping）。

#### 3.6.4 新厂商的最小实现成本

```python
class FooBackend(GpuBackend):
    def supports_ipc(self) -> bool:
        return True   # 或 False，按硬件能力

    def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
        # 调 fooIpcGetMemHandle(...)，返回 bytes
        ...

    def open_ipc_handle(self, handle, dtype, shape, strides, storage_offset, device_id):
        # 调 fooIpcOpenMemHandle(...)，封装回 torch.Tensor
        ...

    def close_ipc_handle(self, ptr: int) -> None:
        # 调 fooIpcCloseMemHandle(ptr)
        ...
```

只要这四个方法正确，FlexKV 的 server-client / 多实例模式就自动可用，**`memory_handle.py` 与所有上层调用
点都不需要任何修改**。

### 3.7 GPU 可见性环境变量（`CUDA_VISIBLE_DEVICES` 类）抽象

每家 GPU 厂商都有自己的"可见性掩码"环境变量，用来在多卡机器上限制进程可见的 GPU：

| 厂商 | 环境变量 |
| ---- | -------- |
| NVIDIA | `CUDA_VISIBLE_DEVICES` |
| AMD ROCm/HIP | `HIP_VISIBLE_DEVICES`、`ROCR_VISIBLE_DEVICES`，并出于历史兼容也读 `CUDA_VISIBLE_DEVICES` |
| 摩尔线程 MUSA | `MUSA_VISIBLE_DEVICES` |
| 未来厂商 X | `X_VISIBLE_DEVICES` |

FlexKV 上层在两类场景中需要用到这种掩码：

1. **读取并解析**：在 server-client 模式下，把 worker 进程的"逻辑 device id"翻译成"物理 device id"
   （[`vllm_v1_adapter.py`](../../flexkv/integration/vllm/vllm_v1_adapter.py) 的 `FlexKVWorkerConnector` 与
   [`trtllm_adapter.py`](../../flexkv/integration/tensorrt_llm/trtllm_adapter.py) 的 KV-cache 注册流程）。
2. **擦除**：FlexKV server / 跨进程 TransferManager 必须看到**所有**物理 GPU（要做跨卡 IPC），所以在
   spawn 子进程前要把掩码从 env 里抹掉。

如果上层直接写 `os.environ['CUDA_VISIBLE_DEVICES']`，ROCm 用户就会踩坑：他们设的可能是
`HIP_VISIBLE_DEVICES`，被 FlexKV 完全无视。因此 FlexKV 把这件事下沉到 `GpuBackend`，**业务代码里 0
vendor 字面量**。

#### 3.7.1 抽象类暴露的 3 个 API（`GpuBackend` 第 2 组的扩展）

```python
@classmethod
def visible_devices_env_vars(cls) -> Tuple[str, ...]:
    """该厂商的可见性掩码环境变量名清单。
       第一个是首选名（写入时使用），所有名字都参与读取/擦除。
       默认实现返回 ()，让下面两个 helper 退化为 no-op。"""

def get_visible_device_map(self) -> Optional[List[int]]:
    """读取掩码 -> 解析为 [phys_id, ...]；未设置返回 None。
       自动忽略 UUID 形式的 mask（如 'GPU-xxxx'），返回 None 让调用方走默认分支。"""

def strip_visible_devices(self, env: Dict[str, str]) -> Dict[str, str]:
    """把 env 中所有可见性掩码就地删除并返回 env，用于 spawn 全局可见的子进程。"""
```

`get_visible_device_map / strip_visible_devices` 在抽象基类已经给出**完整的默认实现**——它们只依赖
`visible_devices_env_vars()`，所以**新厂商只需要重写这一个 classmethod**。

#### 3.7.2 各厂商重写示例

```python
# NVIDIA
@classmethod
def visible_devices_env_vars(cls) -> Tuple[str, ...]:
    return ("CUDA_VISIBLE_DEVICES",)

# ROCm —— HIP 是首选；ROCR 是 ROCr 运行时层；CUDA 名字也读，用于兼容老脚本
@classmethod
def visible_devices_env_vars(cls) -> Tuple[str, ...]:
    return ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

# MUSA
@classmethod
def visible_devices_env_vars(cls) -> Tuple[str, ...]:
    return ("MUSA_VISIBLE_DEVICES",)
```

#### 3.7.3 上层调用点（业务代码已 0 字面量）

| 调用点 | 用途 | API |
| ------ | ---- | --- |
| `flexkv/integration/vllm/vllm_v1_adapter.py` | worker 把 KV-cache 注册给 server 时，把 `local_device` 映射成物理 id | `_gpu_backend.get_visible_device_map()` |
| `flexkv/integration/tensorrt_llm/trtllm_adapter.py` | 同上，TRT-LLM 路径 | 同上 |
| `flexkv/server/server.py` | spawn server 子进程前擦除掩码，让 server 看到所有 GPU | `_gpu_backend.strip_visible_devices(env)` |
| `flexkv/transfer_manager.py` | spawn TransferManager 子进程前擦除掩码 | 同上 |

> **关键不变量**：FlexKV 业务代码（adapter / server / transfer_manager / memory_handle）里**不出现任何**
> `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` / `MUSA_VISIBLE_DEVICES` 字面量。这些字符串只允许出现在
> `flexkv/gpu_backend/<vendor>/backend.py` 一处。

---

## 4. 各厂商如何编译与发布

### 4.1 默认行为（NVIDIA CUDA）

> ⚠️ **重要**：`pip install -e .` 与 `pip install flexkv` **默认编译并安装 NVIDIA CUDA 版本**。
>
> 当 `FLEXKV_GPU_BACKEND` 环境变量未设置时，`setup.py` 取默认值 `"nvidia"`，调用 `NvidiaBuilder`，使用 `nvcc` 编译 `csrc/gpu_backend/nvidia/`，产物是 `flexkv.c_ext`（NVIDIA 版本）。
>
> 不需要任何额外配置，与重构前的行为完全一致。

```bash
# 默认 NVIDIA CUDA wheel
pip install -e .

# 等价于
FLEXKV_GPU_BACKEND=nvidia pip install -e .
```

可选编译开关（默认全部关闭）：

| 环境变量 | 作用 |
| -------- | ---- |
| `FLEXKV_ENABLE_GDS=1`     | 编译 GPU Direct Storage / cuFile（仅 NVIDIA） |
| `FLEXKV_ENABLE_P2P=1`     | 编译分布式 RadixTree + Redis 元数据通道 |
| `FLEXKV_ENABLE_METRICS=1` | 编译 prometheus-cpp 监控埋点 |
| `FLEXKV_ENABLE_CFS=1`     | 编译 PCFS（百度文件系统）远端路径 |
| `FLEXKV_ENABLE_CPUTEST=1` | 跳过 `-lcuda`，用于无 CUDA 驱动的 CI 容器 |
| `TORCH_CUDA_ARCH_LIST`    | 显式指定 `sm_80;sm_90` 等；不设则由 `NvidiaBackend.detect_arch_list()` 自动探测 |

### 4.2 AMD ROCm / HIP

`csrc/gpu_backend/rocm/.hipified/` 在编译时由 `RocmBuilder.configure_env()` 自动调用 `hipify-perl` 从 NVIDIA 源（`transfer.cu / transfer.cuh / gtensor_handler.cuh`）生成，已加入 `.gitignore`。

```bash
# 前置：已安装 ROCm PyTorch（torch.version.hip is not None）+ ROCm SDK + hipify-perl
FLEXKV_GPU_BACKEND=rocm pip install -e .

# 可选：指定架构（不设则用默认 gfx90a;gfx942）
PYTORCH_ROCM_ARCH=gfx90a;gfx942 \
FLEXKV_GPU_BACKEND=rocm pip install -e .
```

产物：`flexkv.c_ext`（ROCm 版本，由 hipcc 编译）。无 GDS。

### 4.3 摩尔线程 MUSA

```bash
# 前置：已安装 torch_musa + MUSA SDK
MUSA_HOME=/usr/local/musa \
FLEXKV_GPU_BACKEND=musa pip install -e .
```

产物：`flexkv.c_ext_musa`（由 mcc 编译；与 NVIDIA wheel 共存不冲突）。

### 4.4 Generic（无 C 扩展，CI / CPU-only）

```bash
FLEXKV_GPU_BACKEND=generic pip install -e .
```

无 C 扩展，所有 hot path 退化到纯 PyTorch 实现。用于 CI、文档构建、无 GPU 容器。

### 4.5 第三方 / CUDA 兼容硬件

如果你的硬件**已经兼容 CUDA 接口**（例如百度昆仑芯走 `libcudart.so` 的兼容层），优先复用 NVIDIA 路径：

```bash
FLEXKV_GPU_BACKEND=kunlun pip install -e .   # alias → NvidiaBuilder + NvidiaBackend
```

如果硬件不兼容 CUDA，需要**新增一个 vendor**，按 §5 接入流程操作。也可以通过 Python entry-point `flexkv.gpu_backends` 把 backend 类挂到 FlexKV 主仓库之外的独立 wheel 中（`current_backend` 自动探测时会优先查 entry-point），这样能完全不改主仓库代码。

### 4.6 发布矩阵

| Wheel | 安装命令 | C 扩展 | 编译器 |
| ---- | -------- | ------ | ------ |
| `flexkv` / `flexkv-nvidia` | `pip install -e .` *(默认)* | `flexkv.c_ext` | nvcc |
| `flexkv-rocm` | `FLEXKV_GPU_BACKEND=rocm pip install -e .` | `flexkv.c_ext` | hipcc |
| `flexkv-musa` | `FLEXKV_GPU_BACKEND=musa MUSA_HOME=… pip install -e .` | `flexkv.c_ext_musa` | mcc |
| `flexkv-cpu`  | `FLEXKV_GPU_BACKEND=generic pip install -e .` | （无） | — |

**零污染保证**：编译 `flexkv-rocm` 时不需要 CUDA SDK；编译 `flexkv-nvidia` 时不需要 ROCm SDK；不同 vendor 的源文件在 `csrc/gpu_backend/<vendor>/` 下互相隔离。

---

## 5. 接入新 GPU 厂商的工作清单

按顺序完成下述四步，新厂商即可像 NVIDIA / ROCm 一样正常工作：

1. **C++ 实现**：在 `csrc/gpu_backend/<vendor>/` 下提供
   - GPU kernel 与 host runtime 调用（`transfer.cu/.cuh/.h` 等）；
   - `<vendor>_bindings.cpp`，实现 `register_<vendor>_bindings(pybind11::module_&)`，注册至少 `transfer_kv_blocks` 与 `TPTransferThreadGroup`；
   - 在 `csrc/gpu_backend/backend_bindings.h` 中加 `#if defined(FLEXKV_BACKEND_<VENDOR>)` 分支；
   - 如需共用 `tp_transfer_thread_group.cpp`，确认 `gpu_types.h` 已为本 vendor 提供 `gpu_*` 宏与 `gpu_stream_t / gpuError_t / gpuSuccess` 别名。
2. **Python 实现**：在 `flexkv/gpu_backend/<vendor>/backend.py` 中实现 `class <Vendor>Backend(GpuBackend)`，至少覆盖 §3.1 中第 1–5 组的所有 `@abstractmethod`。**别忘了重写 `visible_devices_env_vars()`**（仅一行 classmethod，返回该厂商的可见性掩码名清单，详见 §3.7.2），否则 server-client 模式下的 device-id 映射与子进程 mask 擦除都会退化为 no-op。
3. **Builder 实现**：在 `build_backends/<vendor>_builder.py` 中实现 `class <Vendor>Builder(GPUBuilder)`，提供 sources / compile_args / link_args / include_dirs / configure_env；并在 `build_backends/__init__.py:load_builder` 与 `flexkv/gpu_backend/__init__.py:_FORCE_MAP / _BUILTIN_BACKENDS` 注册 alias。
4. **可选 IPC / GDS**：若硬件支持，覆写 `supports_ipc / get_ipc_handle / open_ipc_handle / close_ipc_handle`（详见 §3.6）或 `supports_direct_storage / gds_create_manager` 等，并在 vendor binding 中暴露相应类。
5. **GDR（跨节点 RDMA）自动可用**：**不需要**新增 GDR 钩子。只要 1–4 步实现正确，`MoonCakeTransferEngineWrapper`（与 GPU 厂商无关）就会通过 libibverbs 注册 pinned / 设备 buffer，不需要任何 vendor-specific 代码。详见 §3.5。

整个过程中：`csrc/bindings.cpp`、`flexkv/gpu_backend/interface.py`、`flexkv/gpu_backend/__init__.py` 之外的 FlexKV 上层代码**不需要任何改动**。
