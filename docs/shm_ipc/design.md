# FlexKV Shared Memory IPC Channel

## 背景与动机

FlexKV 在 multi-DP 场景下使用 server-client 模式：多个 DP client 通过 IPC 与一个 KV Server 进程通信。当前实现基于 ZMQ (IPC socket) + pickle 序列化，round-trip 延迟约 100-200us，主要开销：

| 环节 | 耗时 |
|------|------|
| pickle 序列化 (client 侧) | ~22us |
| ZMQ IPC 传输 | ~30-50us |
| pickle 反序列化 (server 侧) | ~22us |
| server 处理 | <1us (get_match 等轻量操作) |
| 响应返回 (同上) | ~50us |

对于 `get_match`/`put_match` 这类同步操作，推理引擎会阻塞等待结果，IPC 延迟直接加到 prefill 关键路径上。目标是将 round-trip 延迟降到 ~10-50us 量级。

## 方案设计

### 核心思路

用 `/dev/shm/` 命名共享内存文件 + Linux futex 替代 ZMQ+pickle：
- 共享内存：零拷贝，client 写入的数据 server 直接可见，无需内核态数据拷贝
- 二进制序列化：`struct.pack_into` + `np.ndarray.tobytes()` 替代 pickle，省去对象创建开销
- futex：轻量跨进程通知，无 busy wait 时接近零 CPU 开销

### 与 ZMQ 并存

SHM IPC 作为可选传输层，通过环境变量 `FLEXKV_SHM_IPC=1` 启用。ZMQ 路径完全不变。TP client 的 GPU tensor handle 注册仍走 ZMQ（一次性操作，不敏感）。

### 架构图

```
  DP Client 0              DP Client 1              DP Client N
      |                        |                        |
      v                        v                        v
  ShmChannel_0            ShmChannel_1            ShmChannel_N
  /dev/shm/flexkv_shm_    /dev/shm/flexkv_shm_    /dev/shm/flexkv_shm_
  ch_{id}_0               ch_{id}_1               ch_{id}_N
      |                        |                        |
      +----------+-------------+------------------------+
                 |
                 v
          ShmControlBlock            <-- wake counter (futex)
          /dev/shm/flexkv_shm_ctrl_{id}
                 |
                 v
            KV Server
         (poll loop: check flags -> process -> futex sleep when idle)
```

## 共享内存布局

### ShmControlBlock（全局，4KB）

| 偏移 | 大小 | 字段 | 说明 |
|------|------|------|------|
| 0 | 4B | `wake_counter` | 任意 client atomic 自增 + futex_wake；server futex_wait |
| 64 | 4B | `server_ready` | server 初始化完成后置 1，client 启动时 poll 等待 |

### ShmChannel（每 client 一个，~1GB）

```
[0..64)          async_write_pos   (uint64, client 写)    ─┐ 各占独立 cache line
[64..128)        async_read_pos    (uint64, server 写)     │ 避免 false sharing
[128..192)       sync_req_flag     (int32, client 置 1)    │
[192..256)       sync_resp_flag    (int32, server 置 1)   ─┘
[256..256+1GB)   async ring buffer (4096 slots x 256KB)    # fire-and-forget 消息
[+1GB..+256KB)   sync request slot (256KB)                 # 同步请求
[..+256KB)       sync response slot (256KB)                # 同步响应
```

**注意**: 当前 async ring buffer 占用 1GB（4096 slots x 256KB），多 client 时内存开销较大。这是后续需要优化的重点（见"已知问题"部分）。

### 消息二进制格式

Request header（38 bytes）:
```c
struct {
    uint8_t  msg_type;          // ShmMsgType enum
    int32_t  dp_client_id;
    int64_t  task_id;
    int32_t  n_tokens;
    uint8_t  flags;             // bit0: has_slot_mapping, bit1: has_token_mask
                                // bit2: completely, bit3: as_batch
    int32_t  layer_granularity;
    int32_t  n_task_ids;
    int64_t  batch_id;
    double   wait_timeout;
    int32_t  n_namespace;
};
// 紧跟 variable-length payload:
//   token_ids:    n_tokens * 8 bytes (int64)
//   slot_mapping: n_tokens * 8 bytes (int64, if has_slot_mapping)
//   token_mask:   n_tokens * 1 byte  (bool, if has_token_mask)
//   task_ids:     n_task_ids * 8 bytes (int64)
//   namespace:    [len(4B) + utf8_bytes] * n_namespace
```

Response header（23 bytes）:
```c
struct {
    int32_t  status_code;
    int64_t  task_id;
    uint8_t  is_ready;
    uint8_t  has_mask;
    int32_t  mask_len;
    int32_t  n_kv_responses;
    int32_t  error_msg_len;
};
```

## 两种通信模式

### 1. Async (fire-and-forget)

用于 `put_async`, `get_async`, `prefetch_async`, `launch_tasks`, `cancel_tasks`。

```
Client                          Server
  |                                |
  |-- write msg to ring[wp] ----> |
  |-- advance write_pos --------> |
  |-- notify_server() ----------> |  (increment wake_counter + futex_wake)
  |   (return immediately)        |
  |                                |-- poll: read_pos != write_pos
  |                                |-- read msg from ring[rp]
  |                                |-- advance read_pos
```

SPSC lock-free ring buffer，power-of-2 slot count，`write_pos` 和 `read_pos` 在不同 cache line 上避免 false sharing。ring 满时 client 侧 spin 等待（backpressure）。

### 2. Sync (request-response)

用于 `get_match`, `put_match`, `wait`, `try_wait`, `is_ready`, `shutdown`。

```
Client                          Server
  |                                |
  |-- clear resp_flag (=0) -----> |
  |-- write request to sync slot  |
  |-- set req_flag (=1) --------> |
  |-- notify_server() ----------> |
  |                                |-- poll: req_flag == 1
  |                                |-- read request, clear req_flag (=0)
  |                                |-- process request
  |                                |-- write response to sync slot
  |                                |-- set resp_flag (=1)
  |                                |-- futex_wake(resp_flag)
  |-- spin-wait resp_flag == 1    |
  |   (10000 spins, then futex)   |
  |-- read response               |
  |-- clear resp_flag (=0)        |
```

关键设计点：
- **Flag 所有权**: `req_flag` 由 client 置 1、server 清 0；`resp_flag` 由 server 置 1、client 清 0。避免竞态。
- **Spin-wait + adaptive futex**: client 先 spin 10000 次（~3-5us），超时再 futex_wait。实际场景下 server 响应通常在 spin 阶段内返回。
- **Server 端无 syscall 轮询**: server 检查 `req_flag` 和 ring buffer 都是纯内存读，只在连续空闲 2000 轮后才 futex_wait。

### Server 主循环（关键的 futex 顺序）

```python
while running:
    wake_snapshot = ctrl.get_wake_counter()  # 1. 先读 counter

    for ch in channels:                      # 2. 再检查工作
        req = ch.check_sync_request()
        ...
    for ch in channels:
        msg = ch.async_recv()
        ...

    if no_work:
        idle_spins += 1
        if idle_spins >= 2000:
            ctrl.futex_wait_on_wake(wake_snapshot)  # 3. 用之前的 snapshot
```

**必须先读 wake_counter，再检查工作**。如果反过来，notify 可能在 "无工作" 和 `futex_wait` 之间到达，导致 server 永久阻塞（已踩过的坑）。

## Benchmark 结果

环境：Ubuntu 22.04, Python 3.10, CPU: Intel 系列（单机，无 GPU 参与）

### get_match round-trip 延迟（SHM IPC）

| token 数量 | mean | p50 | p99 |
|-----------|------|-----|-----|
| 512 | 15.2us | 15.7us | 22.1us |
| 1,024 | 19.0us | 17.1us | 28.0us |
| 2,048 | 19.7us | 19.5us | 25.1us |
| 4,096 | 23.7us | 23.5us | 28.8us |
| 8,192 | 31.2us | 31.0us | 36.3us |
| 16,384 | 45.1us | 45.2us | 50.3us |

对比 ZMQ+pickle 的 ~100-200us，SHM IPC 在 512 token 下有约 **10x** 的延迟改善。延迟随 token 数线性增长，主要是 `tobytes()`/`frombuffer` 的 memcpy 开销（16K tokens = 128KB payload）。

### 延迟分解估算（512 tokens）

| 环节 | 耗时 |
|------|------|
| pack_request (client, struct.pack_into + tobytes) | ~3us |
| notify_server (atomic inc + futex_wake) | ~1us |
| server 唤醒 + 读 flag | ~2us |
| unpack_request (server, struct.unpack + frombuffer) | ~3us |
| server 处理 (构造 mask) | <1us |
| pack_response + set resp_flag + futex_wake | ~2us |
| client spin 检测 + unpack_response | ~3us |
| **总计** | **~15us** |

## 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `flexkv/server/shm_channel.py` | 核心 SHM IPC 模块：ShmControlBlock, ShmChannel, 消息序列化/反序列化, futex 封装 |
| `tests/test_shm_channel.py` | 17 个测试：序列化 round-trip, ring buffer, sync request/response, 多进程, 延迟 benchmark |

### 修改文件

| 文件 | 改动 |
|------|------|
| `flexkv/common/config.py` | GLOBAL_CONFIG_FROM_ENV 增加 `shm_ipc` 字段 |
| `flexkv/server/client.py` | 新增 `ShmKVDPClient` 类（与 `KVDPClient` 同接口） |
| `flexkv/server/server.py` | 新增 `_run_shm()`, `_handle_shm_sync_request()`, `_handle_shm_async_request()`；`run()` 根据 `shm_mode` 分发；`create_server()` 传递 `shm_mode` |
| `flexkv/kvmanager.py` | `__init__` 根据 `FLEXKV_SHM_IPC=1` 选择 `ShmKVDPClient` vs `KVDPClient` |

## 如何启用

```bash
export FLEXKV_SHM_IPC=1
# 其余配置不变，正常启动 FlexKV
```

## 如何运行测试

```bash
cd FlexKV
# 安装依赖（如果还没有）
pip install numpy pytest torch --index-url https://download.pytorch.org/whl/cpu
pip install pyzmq

# 运行 SHM channel 单元测试（不需要 GPU）
python3 -m pytest tests/test_shm_channel.py -v --noconftest
```

## 已知问题与后续优化方向

### 1. 内存占用过大（高优先级）

当前每个 ShmChannel 占 ~1GB（4096 ring slots x 256KB/slot）。8 个 DP client 就是 8GB 纯 SHM。

优化方向：
- 减小 `ASYNC_SLOT_SIZE`：大多数消息远小于 256KB，可以按实际最大 token 数动态计算，或用 16KB-64KB
- 减少 `ASYNC_RING_SLOTS`：4096 对于 backpressure 场景过多，256-512 通常够用
- 可变长 slot：header 记录实际消息长度，slot 大小按最大可能的消息计算

### 2. Python 序列化仍是瓶颈

`struct.pack_into`/`unpack_from` + `tobytes()` 在 16K tokens 时占 ~30us。用 C++/pybind11 重写序列化可以进一步压缩到 ~3-5us：
- 直接 memcpy numpy buffer 到 SHM（避免 Python 层 tobytes 创建临时对象）
- SIMD 优化 mask 拷贝

### 3. 尚未进行端到端集成验证

当前只跑了 SHM channel 层面的单元测试和 benchmark。需要在真实 FlexKV 集成环境（多 GPU, 真实推理负载）下验证：
- `FLEXKV_SHM_IPC=1` 完整 e2e 流程
- 与 ZMQ 模式的性能对比
- 多实例 (`FLEXKV_INSTANCE_NUM > 1`) 场景

### 4. futex syscall number 硬编码

`_SYS_FUTEX = 202` 仅适用于 x86_64。如果需要支持 aarch64 需要改为 98。可以通过 `platform.machine()` 动态选择。

### 5. 错误处理

当前 SHM channel 层面的错误处理较简单（主要是 assert 和 exception）。生产使用需要考虑：
- SHM 文件残留清理（进程异常退出时）
- client 连接超时 / server 崩溃检测
- ring buffer 满时的策略（当前是 spin-wait，可能需要超时机制）
