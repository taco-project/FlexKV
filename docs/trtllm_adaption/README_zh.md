# 1. 环境准备
## 1.1. 编译和安装 FlexKV
从源码编译和安装：
```bash
cd FLEXKV_DIR
bash build.sh
```
## 1.2. 安装 TensorRT-LLM（Tag 为 v1.1.0.rc2）
目前我们正在推动社区合入 TensorRT-LLM 侧的适配代码，在合入主分支之前，有如下两种方法：

1. 您可以使用我们提供的 patch，然后重新编译：
```bash
cd FLEXKV_DIR
git apply examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch
```
2. 也可以安装我们预先编译好的包：
```bash
TODO
```

# 2. 运行
## 2.1 相关文件介绍
`examples/trtllm_adaption` 目录中有我们提供的启动脚本的示例，其中的文件包括：
- `vllm_0_10_1_1-flexkv-connector.patch`：TensorRT-LLM 适配 FLexKV 的 patch

- `flexkv_config.json`：FLexKV 相关的配置
- `extra-llm-api-config-cg.yml`：TensorRT-LLM 额外的一些启动配置
- `launch.sh`：示例启动脚本
## 2.2 启动方式
启动方式为：
```bash
cd examples/trtllm_adaption
bash launch.sh
```

总结来说，您只需要在原有的 TensorRT-LLM 启动脚本中额外引入如下的两个环境变量即可：
```bash
export FLEXKV_CONFIG_PATH="./flexkv_config.json"
export TENSORRT_LLM_USE_FLEXKV=1
```

如果您想了解 flexkv 详细的配置，可以参考 `docs/flexkv_config_reference/README_zh.md`


## 2.3 TensorRT-LLM 潜在的问题
如果您向 TensorRT-LLM 发送了超过 `max_seq_len` 长度的请求，会遇到如下问题：
```
TODO
```
这是 TensorRT-LLM 框架本身没有过滤超过 `max_seq_len` 长度的请求导致的，和 FlexKV 本身无关，目前我们正在推动社区修复这个问题。
