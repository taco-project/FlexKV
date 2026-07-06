# FlexKV 基准测试脚本

此脚本（[./run_benchmark.sh](./run_benchmark.sh)）用于运行启用了 FlexKV 的 vLLM 服务器并执行多轮对话基准测试。

## 安装vLLM+FlexKV

- 参见 [README_zh.md](../README_zh.md)

## 使用方法

```bash
./run_benchmark.sh --vllm-path <path> --model-path <path> [options]
```

### 必需参数

| 参数 | 说明 |
|----------|-------------|
| `--vllm-path <path>` | vLLM 安装路径 |
| `--model-path <path>` | 模型目录路径 |

### 可选参数

#### vLLM 配置

| 参数 | 说明 | 默认值 |
|----------|-------------|---------|
| `--vllm-port <port>` | vLLM 服务器端口 | 30001 |
| `--tensor-parallel-size, --tp-size <size>` | 张量并行大小 | 1 |
| `--max-num-seqs <num>` | 最大序列数量 | 128 |
| `--max-model-len <len>` | 最大模型长度 | 4096 |
| `--gpu-memory-util <ratio>` | GPU 内存利用率 | 0.8 |

#### FlexKV 配置

| 参数 | 说明 | 默认值 |
|----------|-------------|---------|
| `--enable-flexkv <0\|1>` | 启用 FlexKV | 1 |
| `--flexkv-cpu-cache-gb <size>` | FlexKV CPU 缓存大小（GB） | 64 |
| `--flexkv-ssd-cache-gb <size>` | FlexKV SSD 缓存大小（GB） | 1024 |
| `--flexkv-ssd-cache-dir <path>` | FlexKV SSD 缓存目录 | `$HOME/.cache/flexkv_ssd_cache/` |
| `--flexkv-enable-gds <0\|1>` | 启用 FlexKV GDS（GPUDirect Storage） | 0 |

#### 数据集配置

| 参数 | 说明 | 默认值 |
|----------|-------------|---------|
| `--dataset-name <name>` | 数据集名称 | sharegpt（仅支持 sharegpt 数据集） |
| `--dataset-path <path>` | 数据集路径 | `$HOME/.cache/sharegpt/` |

#### 基准测试配置

| 参数 | 说明 | 默认值 |
|----------|-------------|---------|
| `--request-rate <rate>` | 请求速率（每秒请求数） | 32 |
| `--workers <num>` | Worker 数量 | 32 |
| `--max-turns <num>` | 最大轮次数 | 5 |
| `--concurrency <num>` | 并发数 | 1 |
| `--profile` | 启用性能分析 | - |
| `--profile-duration <seconds>` | 性能分析持续时间 | 600 |
| `--profile-delay <seconds>` | 性能分析延迟 | 60 |

#### 日志配置

| 参数 | 说明 | 默认值 |
|----------|-------------|---------|
| `--log-dir <path>` | 日志文件目录 | `../logs` |

#### 帮助

| 参数 | 说明 |
|----------|-------------|
| `-h, --help` | 显示帮助信息 |

## 脚本功能

1. **数据集准备**：如果尚未可用，下载并转换 ShareGPT 数据集
2. **FlexKV 配置**：设置 FlexKV 环境变量
3. **vLLM 服务器启动**：使用指定配置启动 vLLM 服务器，并等待服务器就绪（最多 5 分钟）
4. **基准测试执行**：运行多轮对话基准测试
5. **清理**：停止 vLLM 服务器并显示结果

## 输出文件

脚本会在日志目录中生成以下日志文件：

- `vllm_server_YYYYMMDD_HHMMSS.log` - vLLM 服务器日志
- `benchmark_YYYYMMDD_HHMMSS.log` - 基准测试结果和摘要

如果启用性能分析，还会生成以下文件：
- `vllm_profile_YYYYMMDD_HHMMSS.nsys-rep` - 性能分析报告
