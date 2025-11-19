# 1. 环境准备
## 1.1. 编译和安装 FlexKV
从源码编译和安装：
```bash
cd FLEXKV_DIR
bash build.sh
```
## 1.2. 安装 TensorRT-LLM（Tag 为 v1.1.0.rc2）
目前我们正在推动社区合入 TensorRT-LLM 侧的适配代码，在合入主分支之前，有如下两种方法：
### 1.2.1 方法一
您可以使用我们提供的 patch，然后重新编译：
```bash
cd TENSORRT_LLM_DIR
git apply examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch
```
### 1.2. 方法二
您也可以安装我们预先编译好的包：
```bash
pip install https://flexkv-1252113659.cos.ap-shanghai.myqcloud.com/TensorRT-LLM/tensorrt_llm-1.1.0rc2-cp312-cp312-linux_x86_64.whl
```
注：TensorRT-LLM 的编译方式可以参考[这里](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#build-from-source-linux)

# 2. 运行
## 2.1 相关文件介绍
`examples/trtllm_adaption` 目录中有我们提供的启动脚本的示例，其中的文件包括：
- `vllm_0_10_1_1-flexkv-connector.patch`：TensorRT-LLM 适配 FLexKV 的 patch

- `flexkv_config.json`：FLexKV 相关的配置
- `extra-llm-api-config.yml`：TensorRT-LLM 额外的一些启动配置
- `launch.sh`：示例启动脚本
## 2.2 启动方式
### 2.2.1. 方式一：使用我们提供的脚本
启动方式为：
```bash
cd FLEXKV_DIR/examples/trtllm_adaption
bash launch.sh YOUR_MODEL_PATH
```
注：`launch.sh` 脚本会同时启动 TensorRT-LLM 和 FlexKV
### 2.2.2. 方式二：改造您的脚本
首先，在您的 TensorRT-LLM 启动脚本中额外引入如下的两个环境变量：
```bash
export FLEXKV_CONFIG_PATH="./flexkv_config.json"
export TENSORRT_LLM_USE_FLEXKV=1
```
然后，在您的 `extra-llm-api-config.yml` 中添加如下内容：
```txt
kv_cache_config:
  enable_partial_reuse: false
kv_connector_config:
  connector_module: "flexkv.integration.tensorrt_llm.trtllm_adapter"
  connector_scheduler_class: "FlexKVSchedulerConnector"
  connector_worker_class: "FlexKVWorkerConnector"
```

注：如果您想了解 flexkv 详细的配置，可以参考 `docs/flexkv_config_reference/README_zh.md`


## 2.3 TensorRT-LLM 潜在的问题
如果您向 TensorRT-LLM 发送了超过 `max_seq_len` 长度的请求，会出现类似下面的报错：
```
[W] `default_max_tokens` (-40205) should be greater than 0, `default_max_tokens` (-40205) = max_seq_len (40961) - `splited_prompt_len` (81166) - `query_token_len` (0)
[W] User-specified `max_tokens` (16384) is greater than deduced `default_max_tokens` (-40205), using default_max_tokens instead.
[E] submit request failed: [TensorRT-LLM][ERROR] Assertion failed: mMaxNewTokens > 0
```
这是 TensorRT-LLM 框架本身没有过滤超过 `max_seq_len` 长度的请求导致的，和 FlexKV 本身无关，目前我们正在推动社区修复这个问题。
