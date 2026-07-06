# Using FlexKV in TensorRT-LLM
## 1. Environment Setup

### 1.1 Install TensorRT-LLM (Tag v1.1.0rc2)
We are currently working with the community to merge TensorRT-LLM adaptation code. Before it is merged into the main branch, there are two methods:
#### 1.1.1 Method 1
You can use the patch we provide and recompile:
```bash
cd TensorRT-LLM
git apply FLEXKV_DIR/examples/trtllm_adaption/trtllm_v1.1.0rc2.patch
```
Note: For TensorRT-LLM compilation instructions, please refer to [here](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#build-from-source-linux)

#### 1.1.2 Method 2
You can also install our pre-compiled package:
```bash
pip install https://flexkv-1252113659.cos.ap-shanghai.myqcloud.com/TensorRT-LLM/tensorrt_llm-1.1.0rc2-cp312-cp312-linux_x86_64.whl
```

## 2. Running

### 2.1 Configure FlexKV

First, set the environment variable `TENSORRT_LLM_USE_FLEXKV` to enable FlexKV:
```bash
export TENSORRT_LLM_USE_FLEXKV=1
```

FlexKV can be configured through environment variables and configuration files. For details, please refer to [`docs/flexkv_config_reference/README_en.md`](../../docs/flexkv_config_reference/README_en.md). Below are two simple configuration examples.
##### Example 1: Enable CPU Offloading Only
Use 32GB of CPU memory as secondary cache.
```bash
unset FLEXKV_CONFIG_PATH
export FLEXKV_CPU_CACHE_GB=32
```
##### Example 2: Enable SSD Offloading
Use 32GB of CPU memory and 1TB of SSD storage as secondary and tertiary caches respectively. (Assuming the machine has two SSDs mounted at /data0 and /data1.)
```bash
# generate config
cat <<EOF > ./flexkv_config.yml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data0/flexkv_ssd/;/data1/flexkv_ssd/
enable_gds: false
EOF
export FLEXKV_CONFIG_PATH="./flexkv_config.yml"
```

### 2.2 Launch TensorRT-LLM
#### 2.2.1. Method 1: Using Our Provided Example Script
```bash
cd FLEXKV_DIR/examples/trtllm_adaption
bash launch.sh YOUR_MODEL_PATH
```
Note: The `launch.sh` script will launch both TensorRT-LLM and FlexKV, and configure FlexKV through `flexkv_config.json` in the same directory.
#### 2.2.2. Method 2: Custom Launch
After configuring FlexKV according to the instructions in section [2.1](#21-configure-flexkv), add the following content to your `extra-llm-api-config.yml`:
```txt
kv_cache_config:
  enable_partial_reuse: false
kv_connector_config:
  connector_module: "flexkv.integration.tensorrt_llm.trtllm_adapter"
  connector_scheduler_class: "FlexKVSchedulerConnector"
  connector_worker_class: "FlexKVWorkerConnector"
```

### 2.3 Potential TensorRT-LLM Issues
If you send a request to TensorRT-LLM that exceeds the `max_seq_len` length, you may encounter an error similar to the following:
```
[W] `default_max_tokens` (-40205) should be greater than 0, `default_max_tokens` (-40205) = max_seq_len (40961) - `splited_prompt_len` (81166) - `query_token_len` (0)
[W] User-specified `max_tokens` (16384) is greater than deduced `default_max_tokens` (-40205), using default_max_tokens instead.
[E] submit request failed: [TensorRT-LLM][ERROR] Assertion failed: mMaxNewTokens > 0
```
This is caused by the TensorRT-LLM framework itself not filtering requests that exceed the `max_seq_len` length, and is not related to FlexKV. We are currently working with the community to fix this issue.
