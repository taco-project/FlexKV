# FlexKV HugePage User Guide

## 1. Overview

FlexKV currently exposes three HugePage-related configuration fields:

- `use_hugepage_cpu_buffer`
  Controls whether the main CPU KV cache should be allocated from HugePages.
- `use_hugepage_tmp_buffer`
  Controls whether the temporary CPU staging buffer in the `enable_p2p_ssd=true` path should be allocated from HugePages.
- `hugepage_size_bytes`
  Controls the HugePage size used by both allocation paths.

The two HugePage switches are independent. They can be enabled separately or together.

There is one important implementation constraint today:

- `use_hugepage_cpu_buffer` for the main CPU KV cache must use a `hugetlbfs`-backed file so the same HugePage mapping can be reopened inside `spawn` workers.
- Anonymous `MAP_HUGETLB` is not sufficient for the main CPU cache path because once that tensor is sent into `TransferEngine` workers, PyTorch serializes ordinary CPU tensors into new shared-memory storage, which breaks the original HugePage backing.
- `use_hugepage_tmp_buffer` does not have that cross-process sharing requirement, so it may still succeed through either anonymous HugePages or `hugetlbfs`.

---

## 2. Recommended Use Cases

HugePage is recommended in the following situations:

- The CPU KV cache is large and CPU-side page table or TLB overhead is non-trivial.
- `enable_p2p_ssd=true` is enabled and you want to optimize the temporary staging buffer in the `SSD -> CPU -> GPU` path.
- The host has already been prepared with reserved HugePages and a working hugetlbfs mount.

HugePage is not recommended when the host has no HugePage reservation or when the target workload does not materially benefit from CPU cache or p2p SSD path optimization.

---

## 3. Prerequisites

### 3.1 HugePages Must Be Reserved on the Host

Check the current HugePage status:

```bash
grep -E 'HugePages_|Hugepagesize' /proc/meminfo
```

For 2 MiB HugePages, the following command reserves 4096 pages, which is about 8 GiB:

```bash
sudo sysctl -w vm.nr_hugepages=4096
```

If you plan to use 1 GiB HugePages, they usually need to be reserved through kernel boot parameters, for example:

```text
default_hugepagesz=1G hugepagesz=1G hugepages=N
```

### 3.2 hugetlbfs Must Be Mounted

FlexKV uses `/mnt/hugepages` as the default hugetlbfs mount point.

For `use_hugepage_cpu_buffer`, this is a hard requirement rather than a recommendation. If `FLEXKV_HUGETLBFS_DIR` points to a normal filesystem, FlexKV now rejects it and falls back instead of silently treating regular 4 KiB pages as HugePages.

Check the mount status:

```bash
mount | grep hugetlbfs
ls -ld /mnt/hugepages
```

If hugetlbfs is not mounted yet:

```bash
sudo mkdir -p /mnt/hugepages
sudo mount -t hugetlbfs none /mnt/hugepages
```

If your actual hugetlbfs mount point is different, set it explicitly:

```bash
export FLEXKV_HUGETLBFS_DIR=/path/to/hugetlbfs
```

### 3.3 CUDA Runtime Is Required for the tmp Buffer Path

The `use_hugepage_tmp_buffer` path performs `cudaHostRegister` after HugePage allocation succeeds. This path therefore requires:

- A working CUDA runtime
- `libcudart.so` to be discoverable
- A sufficiently large `memlock` limit on the host or in the container

Basic check:

```bash
python3 - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

Note: `use_hugepage_cpu_buffer` does not depend on `cudaHostRegister`.

Additional note: although `use_hugepage_cpu_buffer` does not require CUDA runtime, it does require a writable `hugetlbfs` mount because the main CPU KV cache must be reopened from the same HugePage-backed file inside spawned workers.

---

## 4. Configuration

HugePage is now a formal user-facing configuration surface in FlexKV. It can be configured through either configuration files or environment variables.

### 4.1 Configuration File

YAML example:

```yaml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data/flexkv_ssd/
enable_p2p_ssd: true
use_hugepage_cpu_buffer: true
use_hugepage_tmp_buffer: true
hugepage_size_bytes: 2097152
```

JSON example:

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

### 4.2 Environment Variables

```bash
export FLEXKV_USE_HUGEPAGE_CPU_BUFFER=1
export FLEXKV_USE_HUGEPAGE_TMP_BUFFER=1
export FLEXKV_HUGEPAGE_SIZE_BYTES=2097152
```

Meaning:

- `FLEXKV_USE_HUGEPAGE_CPU_BUFFER=1`
  Enables HugePage allocation for the main CPU KV cache.
- `FLEXKV_USE_HUGEPAGE_TMP_BUFFER=1`
  Enables HugePage allocation for the temporary CPU staging buffer in the p2p SSD path.
- `FLEXKV_HUGEPAGE_SIZE_BYTES=2097152`
  Uses 2 MiB HugePages.

If the host is configured for 1 GiB HugePages:

```bash
export FLEXKV_HUGEPAGE_SIZE_BYTES=1073741824
```

### 4.3 How to Choose Between the Two Switches

- If you only want to optimize the main CPU KV cache, enable `use_hugepage_cpu_buffer`.
- If you only want to optimize the temporary staging buffer in the p2p SSD path, enable `use_hugepage_tmp_buffer` and make sure `enable_p2p_ssd=true` is set.
- If both paths matter, enable both switches.

---

## 5. Recommended Enablement Order

For a first rollout, the recommended order is:

1. Prepare 2 MiB HugePages on the host and confirm that hugetlbfs is mounted correctly.
2. Enable `use_hugepage_cpu_buffer=true` first and verify that the main CPU KV cache works correctly.
3. If you also need to validate the p2p SSD path, then enable `use_hugepage_tmp_buffer=true`.
4. Only after the feature is stable should you evaluate switching to 1 GiB HugePages.

For initial validation, 2 MiB HugePages are recommended because host setup is simpler and troubleshooting is more straightforward.

---

## 6. How to Verify It Is Working

### 6.1 Check the Logs

If the tmp staging buffer successfully uses HugePages, logs will contain a message similar to:

```text
[PEER2CPUTransferWorker] tmp_cpu_buffer allocated on HugePages: 2.000 GB
```

If the main CPU KV cache successfully uses HugePages, you will typically also see a log similar to:

```text
HugePage allocate total_size: ... GB (page_size=2MiB)
```

If the HugePage path for the tmp staging buffer fails and falls back, logs will contain a message similar to:

```text
[PEER2CPUTransferWorker] HugePage allocation for tmp_cpu_buffer failed (...); falling back to torch.empty(pin_memory=True).
```

If `use_hugepage_cpu_buffer=true` but the hugetlbfs mount is invalid, logs will contain a message similar to:

```text
HugePage allocation failed (HugePage: /path is not a hugetlbfs mount ...); falling back to regular CPU memory.
```

### 6.2 Check HugePage Counters

Before and after the service starts, run:

```bash
grep -E 'HugePages_Total|HugePages_Free|Hugepagesize' /proc/meminfo
```

If HugePage allocation is active, you will typically observe:

- `HugePages_Total` unchanged
- `HugePages_Free` decreased

After the service exits and releases resources, `HugePages_Free` should return close to its original value.

### 6.3 Run the Test Suite

If the machine already satisfies the HugePage and CUDA requirements, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/hugepage -rs
```

This test suite validates:

- HugePage allocation and release
- HugePage configuration flow for the CPU KV cache
- HugePage configuration flow for the tmp staging buffer
- Fallback behavior when HugePage allocation cannot be used

---

## 7. Common Configuration Errors

### 7.1 `use_hugepage_tmp_buffer` Is Enabled but Does Not Take Effect

Check the following items in order:

- `enable_p2p_ssd=true` is actually enabled
- The host has enough HugePages reserved
- hugetlbfs is mounted
- `FLEXKV_HUGETLBFS_DIR` points to the correct mount
- CUDA runtime is available
- `memlock` is not too small

### 7.2 HugePage Is Enabled but There Is No Error and No Performance Gain

This usually means the HugePage path has already fallen back to regular memory.

FlexKV treats HugePage as a best-effort optimization with automatic fallback. Service startup success alone is not sufficient evidence that HugePage is active. You must confirm through logs and `/proc/meminfo`.

Also distinguish the two common cases:

- For `use_hugepage_cpu_buffer`, a writable `hugetlbfs` mount is mandatory even if the host has reserved HugePages.
- For `use_hugepage_tmp_buffer`, anonymous HugePages may still work even without `hugetlbfs`.

### 7.3 1 GiB HugePages Do Not Work After Configuration

The most common reason is that the host does not actually have a 1 GiB HugePage pool available. Confirm the following:

- Kernel boot parameters are set correctly
- The host has a real 1 GiB HugePage reservation
- `hugepage_size_bytes` matches the HugePage type actually available on the machine

For initial rollout, it is better to validate functionality with 2 MiB HugePages first and move to 1 GiB only afterward.

---

## 8. Minimal Working Examples

If you want to validate both the main CPU KV cache and the p2p SSD tmp buffer HugePage paths, the following is a minimal example:

```yaml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data/flexkv_ssd/
enable_p2p_ssd: true
use_hugepage_cpu_buffer: true
use_hugepage_tmp_buffer: true
hugepage_size_bytes: 2097152
```

If you only want to validate the main CPU KV cache path:

```yaml
cpu_cache_gb: 32
use_hugepage_cpu_buffer: true
hugepage_size_bytes: 2097152
```
