# NSA Indexer 去重与启动校验

SGLang 0.5.10 的配套接入使用 `FLEXKV_DEDUP_INDEXER_GROUP` 控制 Indexer 去重。
在一些特殊机型和场景下分支保留原默认行为：**默认开启**。上游同名开关默认关闭，部署时应显式设置。

```bash
export FLEXKV_DEDUP_INDEXER_GROUP=1
```

设为 `0` 保留原完整 Indexer 注册布局。两侧均接受去除首尾空白后的
`1/true/yes/on` 为开启，其余值视为关闭。旧字段 `FLEXKV_PRUNE_UNUSED_INDEXER`
已移除，不再读取；原部署若用旧字段关闭优化，必须改为新字段的 `0`。
SGLang 与 FlexKV 必须使用相匹配的接入版本。

去重根据缓存池的 skip-topk 元数据与原始层号进行，不按模型路径或模型名称匹配。
不修改模型的 top-k 频率、KV 内容、量化方式或数值计算。

## 校验边界

优化开启时，在 manager 启动和 GPU buffer 注册前检查：

- 原始 Indexer 列表、层数、skip 表与 active 层号的一致性；缺失 active buffer 不能被 alias 归一化掩盖。
- active buffer 为二维、shape/dtype/device 一致、连续且满足 8 字节行/指针对齐；每个 Indexer page 对应一个主 KV block。
- 本 rank 持有的 active buffer 地址区间不得重叠；同一分配中相邻、不重叠的连续视图可以使用。
- 物理 CP 非 owner 的 active buffer 可以为空；shared 层可以为 None 或 alias。接收 scratch 可被所有非 owner 层共同引用，但不得与本地持有的 active buffer 重叠。
- rank-local GPU device ordinal 不参与跨 rank 比较；参与去重的 cache ranks 必须具有相同的 Indexer 几何与 dtype。

本地失败通过原有 eligibility MIN vote 通知其他 cache ranks，一致拒绝启动；
已有的最终 membership vote 同时核对几何指纹。合法 NSA 启动不增加 collective，
serving 请求、H2D/D2H 与逐层消费不增加校验或同步。关闭优化时保留原未裁剪行为。

这不扩大物理 CP、PP、SWA 或显式 layer groups 的支持范围，也不补齐 DSpark 草稿
Host/L3 缓存等其他独立功能。

## CPU 验证

```bash
python3 -B tests/test_indexer_preflight_standalone.py -v
python3 -B tests/test_sglang_constructor_standalone.py -v
python3 -B tests/test_multinode_cp_cache_standalone.py -v
```

测试执行实际 selector/constructor 和 CPU collective harness，注入指针重叠、
缺失 buffer、几何不一致及单 rank 故障；合法 CP scratch 复用不会被误判。
Tensor/IPC/设备端点使用替身，不等同于真实一些特殊机型和场景 或模型精度、性能验收。
