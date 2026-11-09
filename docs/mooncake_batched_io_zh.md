# Mooncake I/O 批量大小限制

[English](mooncake_batched_io.md)

Mooncake Store 的 JSON 配置支持可选的 `max_get_batch_bytes` 和
`max_put_batch_bytes`，单位是每次 SDK 调用的原始数据字节数。两者默认均为 `0`
（不限制），保持原有批处理行为。例如：

```json
{
  "max_get_batch_bytes": 67108864,
  "max_put_batch_bytes": 67108864
}
```

将这些字段添加到已有的 Store 配置中。正数上限必须能够容纳最大的单个 KV/SWA
block；超大 block 会在提交任何数据批次之前被拒绝。一个 key 不会被拆分，失败的
key 不会自动重试。上限计算请求数据字节之和，不包括分配器对齐、provider 元数据或
RDMA 注册开销，应根据实际 provider 选择。

只有 SDK 为每个 key 返回的字节数与请求值完全一致，Get 才算成功。短读、零字节、
超大响应和负返回码均表示该 key 失败。Put 按 key 记录成功（返回 `0`），并跳过
已有 key。返回结果数量不匹配时会抛错，而不是静默截断；传输后端将该操作标记为
失败，但之前完成的 Put 批次不会回滚。
