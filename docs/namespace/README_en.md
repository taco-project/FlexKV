# FlexKV Namespace Isolation Guide

This document describes how to use FlexKV's namespace isolation feature to enable fine-grained KV cache isolation across different tenants, users, sessions, or LoRA adapters.

## Version Requirements

### vLLM

- For vLLM, please use the `examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector-namespace.patch` patch to support namespace functionality
### TensorRT-LLM

**Important**: Namespace isolation feature is only supported in **TensorRT-LLM v1.1.0rc5**.

- If you are using TensorRT-LLM v1.1.0rc2, please use the `examples/trtllm_adaption/trtllm_v1.1.0rc2.patch` patch, but this version does not support namespace feature
- If you need namespace functionality, please use TensorRT-LLM v1.1.0rc5 and apply the `examples/trtllm_adaption/trtllm_v1.1.0rc5-namespace.patch` patch
## Overview

FlexKV namespace isolation allows you to partition the KV cache into isolated namespaces, ensuring that cache entries from different contexts don't interfere with each other. This feature is essential for:

- **Multi-tenant serving**: Isolate KV cache between different tenants or organizations
- **LoRA adapter isolation**: Separate cache for different LoRA adapters in multi-LoRA serving
- **Session-based isolation**: Maintain separate cache for different user sessions

## How It Works

FlexKV implements namespace isolation at the block hash level. When computing the hash for each KV cache block:

1. **Without namespace**: `hash = H(token_ids)`
2. **With namespace**: `hash = H(namespace_id || token_ids)`

This ensures that:
- Same tokens with different namespaces produce different hashes
- Same tokens with same namespace produce identical hashes
- Prefix matching works correctly within the same namespace

## Parameters

FlexKV's namespace feature can include the following information:

### `lora_request.lora_name`
- **Type**: `string`
- **Description**: When using LoRA adapters, the LoRA name is automatically included in the namespace
- **Example**: `"adapter_A"`

### `cache_salt`
- **Type**: `string`
- **Description**: Session-based cache isolation identifier
- **Example**: `"session_xyz"`

### `namespace_info`
- **Type**: `string` or `List[string]`
- **Description**: User-defined namespace identifier for cache isolation
- **Example**: `"tenant_A"` or `["tenant_A", "user_123"]`

## Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "namespace_info": ["tenant_A", "user_123"]
  }'
```
