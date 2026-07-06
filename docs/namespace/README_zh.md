# FlexKV Namespace 隔离使用指南

本文档介绍如何使用 FlexKV 的 namespace 隔离功能，实现跨不同租户、用户、会话或 LoRA 适配器的细粒度 KV 缓存隔离。

## 版本要求

### vLLM

- 对于 vLLM，请使用 `examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector-namespace.patch` 补丁以支持 namespace 功能
### TensorRT-LLM

**重要提示**：Namespace 隔离功能仅在 **TensorRT-LLM v1.1.0rc5** 中支持。

- 如果您使用的是 TensorRT-LLM v1.1.0rc2，请使用 `examples/trtllm_adaption/trtllm_v1.1.0rc2.patch` 补丁，但该版本不支持 namespace 功能
- 如果您需要使用 namespace 功能，请使用 TensorRT-LLM v1.1.0rc5，并应用 `examples/trtllm_adaption/trtllm_v1.1.0rc5-namespace.patch` 补丁
## 概述

FlexKV namespace 隔离允许您将 KV 缓存划分为独立的命名空间，确保不同上下文的缓存条目互不干扰。此功能适用于：

- **多租户服务**：隔离不同租户或组织之间的 KV 缓存
- **LoRA 适配器隔离**：在多 LoRA 服务中为不同适配器分离缓存
- **基于会话的隔离**：为不同用户会话维护独立缓存

## 工作原理

FlexKV 在块哈希级别实现 namespace 隔离。在计算每个 KV 缓存块的哈希时：

1. **无 namespace**：`hash = H(token_ids)`
2. **有 namespace**：`hash = H(namespace_id || token_ids)`

这确保了：
- 相同 token 但不同 namespace 产生不同的哈希
- 相同 token 和相同 namespace 产生相同的哈希
- 前缀匹配在相同 namespace 内正确工作

## 参数说明

FlexKV 的 namespace 功能可以包含以下信息：

### `lora_request.lora_name`
- **类型**：`string`
- **说明**：使用 LoRA 适配器时，LoRA 名称会自动包含在 namespace 中
- **示例**：`"adapter_A"`

### `cache_salt`
- **类型**：`string`
- **说明**：会话级缓存隔离标识符
- **示例**：`"session_xyz"`

### `namespace_info`
- **类型**：`string` 或 `List[string]`
- **说明**：用户自定义的 namespace 标识符，用于缓存隔离
- **示例**：`"tenant_A"` 或 `["tenant_A", "user_123"]`

## 使用示例

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "你好！"}],
    "namespace_info": ["tenant_A", "user_123"]
  }'
```
