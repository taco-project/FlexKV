# FlexKV HugePage 使用指南

## 一、功能概述

FlexKV 当前支持两类 HugePage 配置项：

- `use_hugepage_cpu_buffer`
  控制通用 CPU KV Cache 是否优先使用 HugePage 分配。
- `use_hugepage_tmp_buffer`
  控制 `enable_p2p_ssd=true` 场景下临时 CPU staging buffer 是否优先使用 HugePage 分配。
- `hugepage_size_bytes`
  控制上述两类内存申请时使用的 HugePage 大小。

两类开关可以独立启用，也可以同时启用。

当前实现上的共享方式：

- `use_hugepage_cpu_buffer` 的主 CPU KV cache 由 TransferManager 用匿名 hugetlb memfd（`memfd_create(MFD_HUGETLB)`）分配，不依赖 hugetlbfs 挂载点，也不会在任何目录下创建文件。`spawn` 出来的 worker 通过 `/proc/<TransferManager pid>/fd/<fd>` 重新打开同一个 memfd 并 mmap，因此仍是同一块 HugePage-backed 映射。
- 进程无论如何退出（包括 SIGKILL），内核都会关闭 fd、解除映射，最后一个引用消失后大页自动归还池中，不会残留文件或占用。
- 纯匿名 `MAP_HUGETLB` 只用于单进程或不可共享场景；一旦主 CPU cache 被传给 `TransferEngine` 的子进程，PyTorch 会把普通 CPU tensor 序列化成新的 shared-memory storage，因此主 CPU cache 不走这条路径。
- `use_hugepage_tmp_buffer` 优先走匿名 `MAP_HUGETLB`，失败时回退到 hugetlb memfd，不涉及跨进程共享。

---

## 二、适用场景

建议在以下场景启用 HugePage：

- CPU KV Cache 容量较大，希望降低页表和 TLB 开销。
- 已启用 `enable_p2p_ssd=true`，并希望优化 `SSD -> CPU -> GPU` 数据路径中的临时 staging buffer。
- 已完成宿主机 HugePage 预留，具备稳定的系统运行条件。

如果机器没有预留 HugePage，或者当前并不依赖 CPU KV Cache / p2p SSD 路径上的性能收益，不建议启用。

---

## 三、前置条件

### 3.1 宿主机已预留 HugePage

先检查系统状态：

```bash
grep -E 'HugePages_|Hugepagesize' /proc/meminfo
```

以 2 MiB HugePage 为例，预留 4096 个页，即约 8 GiB：

```bash
sudo sysctl -w vm.nr_hugepages=4096
```

如果使用 1 GiB HugePage，通常需要在内核启动参数中预留，例如：

```text
default_hugepagesz=1G hugepagesz=1G hugepages=N
```

多 NUMA 机器上开启 `FLEXKV_ENABLE_CPU_CACHE_NUMA_SPREAD=1`（默认关闭）后，会把 CPU KV 池按节点均分，并在启动时逐节点预分配大页。因此每个节点都要预留足够的大页，可通过 `/sys/devices/system/node/node*/hugepages/hugepages-*/nr_hugepages` 按节点查看或设置。某节点不足时会打印告警，超出部分溢出到其他节点。详见 [配置指南 · CPU KV 池 NUMA 分布](../flexkv_config_reference/README_zh.md)。

### 3.2 无需挂载 hugetlbfs

大页通过 `memfd_create(MFD_HUGETLB)` 直接从内核大页池申请，页大小由 `hugepage_size_bytes` 显式指定，不需要 hugetlbfs 挂载点。旧版本使用的 `FLEXKV_HUGETLBFS_DIR` 已不再生效。

要求：

- Linux >= 4.14（支持 `MFD_HUGETLB` 及按页大小选择），Python >= 3.8（`os.memfd_create`）。
- TransferManager 与其 transfer worker 处于同一 PID namespace、同一 uid（父子进程天然满足），worker 需能打开 `/proc/<pid>/fd/<fd>`。

升级前的旧版本若被强杀，可能在 `/dev/hugepages` 或 `/mnt/hugepages` 下残留 `flexkv_hugepage_*` 文件，确认对应进程已退出后可手动删除。

### 3.3 tmp buffer 场景需要 CUDA 运行时

`use_hugepage_tmp_buffer` 对应的 staging buffer 在 HugePage 分配成功后还会执行 `cudaHostRegister`。因此这一路径要求：

- CUDA runtime 可用
- `libcudart.so` 可正常加载
- 容器或宿主机的 `memlock` 限制不要过小

可先做基础检查：

```bash
python3 - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

主 CPU HugePage 池的分配本身使用 hugetlb memfd，`spawn` worker 通过 `/proc/<pid>/fd/<fd>` 重新打开它。GPU/CPU 传输 worker 随后会对 CPU 池调用 `cudaHostRegister`，因此将该池用于 GPU 传输仍需要可用的 CUDA runtime 和合适的注册限制。

---

## 四、配置方式

FlexKV 已将 HugePage 作为正式用户配置项，支持配置文件和环境变量两种方式。

### 4.1 配置文件

YAML 示例：

```yaml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data/flexkv_ssd/
enable_p2p_ssd: true
use_hugepage_cpu_buffer: true
use_hugepage_tmp_buffer: true
hugepage_size_bytes: 2097152
```

JSON 示例：

```json
{
  "cpu_cache_gb": 32,
  "ssd_cache_gb": 1024,
  "ssd_cache_dir": "/data/flexkv_ssd/",
  "enable_p2p_ssd": true,
  "use_hugepage_cpu_buffer": true,
  "use_hugepage_tmp_buffer": true,
  "hugepage_size_bytes": 2097152
}
```

### 4.2 环境变量

```bash
export FLEXKV_USE_HUGEPAGE_CPU_BUFFER=1
export FLEXKV_USE_HUGEPAGE_TMP_BUFFER=1
export FLEXKV_HUGEPAGE_SIZE_BYTES=2097152
```

说明：

- `FLEXKV_USE_HUGEPAGE_CPU_BUFFER=1`
  为通用 CPU KV Cache 启用 HugePage。
- `FLEXKV_USE_HUGEPAGE_TMP_BUFFER=1`
  为 p2p SSD 场景下的临时 CPU staging buffer 启用 HugePage。
- `FLEXKV_HUGEPAGE_SIZE_BYTES=2097152`
  表示使用 2 MiB HugePage。

如果宿主机准备的是 1 GiB HugePage，可设置为：

```bash
export FLEXKV_HUGEPAGE_SIZE_BYTES=1073741824
```

### 4.3 两个开关的选择原则

- 只需要优化通用 CPU KV Cache：开启 `use_hugepage_cpu_buffer`。
- 只需要优化 p2p SSD 的临时 staging buffer：开启 `use_hugepage_tmp_buffer`，同时确保 `enable_p2p_ssd=true`。
- 两条路径都需要：两个开关同时开启。

---

## 五、推荐启用顺序

首次接入建议按以下顺序进行：

1. 先在宿主机准备 2 MiB HugePage。
2. 先只启用 `use_hugepage_cpu_buffer=true`，验证通用 CPU KV Cache 可正常工作。
3. 如果还需要验证 p2p SSD 路径，再启用 `use_hugepage_tmp_buffer=true`。
4. 在确认功能稳定后，再根据机器环境评估是否切换到 1 GiB HugePage。

推荐第一轮验证优先使用 2 MiB HugePage。它的系统准备成本更低，排障也更直接。

---

## 六、如何确认已经生效

### 6.1 检查日志

如果 tmp staging buffer 成功使用 HugePage，日志会出现类似信息：

```text
[PEER2CPUTransferWorker] tmp_cpu_buffer allocated on HugePages: 2.000 GB
```

如果主 CPU KV cache 成功使用 HugePage，通常会先看到类似日志：

```text
HugePage allocate total_size: ... GB (page_size=2MiB)
```

如果 tmp staging buffer 的 HugePage 路径失败并回退，日志会出现类似信息：

```text
[PEER2CPUTransferWorker] HugePage allocation for tmp_cpu_buffer failed (...); falling back to torch.empty(pin_memory=True).
```

如果 `use_hugepage_cpu_buffer=true` 但所需页大小的大页池不可用（未配置该页大小，或空闲大页不足），日志会出现类似信息：

```text
HugePage allocation failed (HugePage: memfd_create(MFD_HUGETLB, page_size=...) failed: ...); falling back to regular CPU memory.
```

运行期间可通过 `ls -l /proc/<TransferManager pid>/fd | grep memfd:flexkv_hugepage` 查看大页池对应的 memfd。

### 6.2 检查 HugePage 计数

在服务启动前后分别执行：

```bash
grep -E 'HugePages_Total|HugePages_Free|Hugepagesize' /proc/meminfo
```

如果 HugePage 分配生效，通常可以观察到：

- `HugePages_Total` 不变
- `HugePages_Free` 下降

服务退出并释放资源后，`HugePages_Free` 应恢复到接近启动前的水平。

### 6.3 运行测试

如果机器已具备 HugePage 和 CUDA 条件，可执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/hugepage -rs
```

该测试集可用于验证：

- HugePage 分配与释放
- CPU KV Cache 的 HugePage 配置路径
- tmp staging buffer 的 HugePage 配置路径
- HugePage 失败后的回退行为

---

## 七、常见配置错误

### 7.1 开启了 `use_hugepage_tmp_buffer`，但实际上没有生效

请依次检查：

- 是否同时设置了 `enable_p2p_ssd=true`
- 宿主机是否预留了足够的 HugePage
- CUDA runtime 是否可用
- `memlock` 限制是否过小

### 7.2 开启了 HugePage，但服务没有报错也没有性能收益

这通常意味着 HugePage 路径已经回退到普通内存分配。

FlexKV 对 HugePage 采用的是“失败自动回退”策略，因此不能仅以服务是否启动成功来判断功能是否生效，必须结合日志和 `/proc/meminfo` 一起确认。

另外需要区分两类原因：

- `use_hugepage_cpu_buffer` 场景下，`hugepage_size_bytes` 对应页大小的大页池必须已配置且空闲页充足，否则 memfd 分配或 mmap 失败并回退。
- `use_hugepage_tmp_buffer` 场景下，同样依赖对应页大小的大页池。

### 7.3 使用 1 GiB HugePage 后启动失败或无法生效

最常见原因是宿主机并未真正准备 1 GiB HugePage 池。请确认：

- 内核启动参数已正确设置
- 宿主机已实际预留 1 GiB HugePage
- `hugepage_size_bytes` 与系统中实际可用的 HugePage 类型一致

如果是首次接入，建议先回到 2 MiB HugePage 完成功能验证，再切换到 1 GiB。

---

## 八、最小可用配置示例

如果你的目标是同时验证通用 CPU KV Cache 和 p2p SSD tmp buffer 的 HugePage 功能，可以使用以下最小配置：

```yaml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data/flexkv_ssd/
enable_p2p_ssd: true
use_hugepage_cpu_buffer: true
use_hugepage_tmp_buffer: true
hugepage_size_bytes: 2097152
```

如果你当前只想验证通用 CPU KV Cache，则可以只保留：

```yaml
cpu_cache_gb: 32
use_hugepage_cpu_buffer: true
hugepage_size_bytes: 2097152
```
