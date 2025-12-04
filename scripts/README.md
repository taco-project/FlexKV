# FlexKV Benchmark Script

This script ([./run_benchmark.sh](./run_benchmark.sh)) is used to run FlexKV-enabled vLLM server with multi-turn benchmarks.

## Install vLLM+FlexKV

- See [README.md](../README.md)

## Usage

```bash
./run_benchmark.sh --vllm-path <path> --model-path <path> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--vllm-path <path>` | Path to vLLM installation |
| `--model-path <path>` | Path to the model directory |

### Optional Arguments

#### vLLM Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--vllm-port <port>` | vLLM server port | 30001 |
| `--tensor-parallel-size, --tp-size <size>` | Tensor parallel size | 1 |
| `--max-num-seqs <num>` | Maximum number of sequences | 128 |
| `--max-model-len <len>` | Maximum model length | 4096 |
| `--gpu-memory-util <ratio>` | GPU memory utilization ratio | 0.8 |

#### FlexKV Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--enable-flexkv <0\|1>` | Enable FlexKV | 1 |
| `--flexkv-cpu-cache-gb <size>` | FlexKV CPU cache size in GB | 64 |
| `--flexkv-ssd-cache-gb <size>` | FlexKV SSD cache size in GB | 1024 |
| `--flexkv-ssd-cache-dir <path>` | FlexKV SSD cache directory | `$HOME/.cache/flexkv_ssd_cache/` |
| `--flexkv-enable-gds <0\|1>` | Enable FlexKV GDS (GPUDirect Storage) | 0 |

#### Dataset Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-name <name>` | Dataset name | sharegpt (only support sharegpt dataset) |
| `--dataset-path <path>` | Dataset path | `$HOME/.cache/sharegpt/` |

#### Benchmark Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--request-rate <rate>` | Request rate (requests per second) | 32 |
| `--workers <num>` | Number of workers | 32 |
| `--max-turns <num>` | Maximum number of turns | 5 |
| `--concurrency <num>` | Concurrency | 1 |
| `--profile` | Enable profiling | - |
| `--profile-duration <seconds>` | Profiling duration | 600 |
| `--profile-delay <seconds>` | Profiling delay | 60 |

#### Logging Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--log-dir <path>` | Directory for log files | `../logs` |

#### Help

| Argument | Description |
|----------|-------------|
| `-h, --help` | Display help information |

## What the Script Does

1. **Dataset Preparation**: Downloads and converts the ShareGPT dataset if not already available
2. **FlexKV Configuration**: Sets up FlexKV environment variables
3. **vLLM Server Launch**: Starts the vLLM server with specified configuration and Waits for the server to be ready (up to 5 minutes)
4. **Benchmark Execution**: Runs multi-turn conversation benchmark
5. **Cleanup**: Stops the vLLM server and displays results

## Output Files

The script generates the following log files in the log directory:

- `vllm_server_YYYYMMDD_HHMMSS.log` - vLLM server logs
- `benchmark_YYYYMMDD_HHMMSS.log` - Benchmark results and summary

If profiling is enabled, the following file will also be generated:
- `vllm_profile_YYYYMMDD_HHMMSS.nsys-rep` - Profiling report
