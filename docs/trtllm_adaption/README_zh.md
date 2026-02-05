# 在 TensorRT-LLM 中使用 FlexKV
## 1. 环境准备

### 1.1 安装 TensorRT-LLM（Tag 为 v1.1.0rc2）
目前我们正在推动社区合入 TensorRT-LLM 侧的适配代码，在合入主分支之前，有如下两种方法：
#### 1.1.1 方法一
您可以使用我们提供的 patch，然后重新编译：
```bash
cd TensorRT-LLM
git apply FLEXKV_DIR/examples/trtllm_adaption/trtllm_v1.1.0rc2.patch
```
注：TensorRT-LLM 的编译方式可以参考[这里](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#build-from-source-linux)

#### 1.1.2 方法二
您也可以安装我们预先编译好的包：
```bash
pip install https://flexkv-1252113659.cos.ap-shanghai.myqcloud.com/TensorRT-LLM/tensorrt_llm-1.1.0rc2-cp312-cp312-linux_x86_64.whl
```

## 2. 运行

### 2.1 配置FlexKV

首先设置环境变量`TENSORRT_LLM_USE_FLEXKV`以启用FlexKV
```bash
export TENSORRT_LLM_USE_FLEXKV=1
```

可以通过环境变量和配置文件两种方式配置FlexKV，具体请参考[`docs/flexkv_config_reference/README_zh.md`](../../docs/flexkv_config_reference/README_zh.md)，下面提供了两个简单的配置示例。
##### 示例一：仅启用CPU卸载
使用32GB的CPU内存作为二级缓存。
```bash
unset FLEXKV_CONFIG_PATH
export FLEXKV_CPU_CACHE_GB=32
```
##### 示例二：启用SSD卸载
使用32GB的CPU内存和1T的SSD存储分别作为二级和三级缓存。（假设机器有两个SSD，并分别挂载在/data0和/data1两个路径上。）
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

### 2.2 启动 TensorRT-LLM
#### 2.2.1. 方式一：使用我们提供的示例脚本
```bash
cd FLEXKV_DIR/examples/trtllm_adaption
bash launch.sh YOUR_MODEL_PATH
```
注：`launch.sh` 脚本会同时启动 TensorRT-LLM 和 FlexKV，并通过同路径下的`flexkv_config.json`进行FlexKV的配置
#### 2.2.2. 方式二：自定义启动
按照 [2.1](#21-配置flexkv) 节的指示配置好FlexKV，接着在您的 `extra-llm-api-config.yml`加入下面的内容：
```txt
kv_cache_config:
  enable_partial_reuse: false
kv_connector_config:
  connector_module: "flexkv.integration.tensorrt_llm.trtllm_adapter"
  connector_scheduler_class: "FlexKVSchedulerConnector"
  connector_worker_class: "FlexKVWorkerConnector"
```

### 2.3 TensorRT-LLM 潜在的问题
如果您向 TensorRT-LLM 发送了超过 `max_seq_len` 长度的请求，会出现类似下面的报错：
```
[W] `default_max_tokens` (-40205) should be greater than 0, `default_max_tokens` (-40205) = max_seq_len (40961) - `splited_prompt_len` (81166) - `query_token_len` (0)
[W] User-specified `max_tokens` (16384) is greater than deduced `default_max_tokens` (-40205), using default_max_tokens instead.
[E] submit request failed: [TensorRT-LLM][ERROR] Assertion failed: mMaxNewTokens > 0
```
这是 TensorRT-LLM 框架本身没有过滤超过 `max_seq_len` 长度的请求导致的，和 FlexKV 本身无关，目前我们正在推动社区修复这个问题。
