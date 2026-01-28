# FlexKV 分布式 KVCache 重用功能

FlexKV 当前支持分布式 KVCache 的重用功能，在一个多节点 serving 的环境中，只需开启一些选项就能够实现多个节点之间的 KVCache 共享。

FlexKV 实现了 KVCache index 管理的功能，通过在本地构建一个全局索引的快照，能够快速进行全局 KVCache 的查询和复用。这里使用了 Mooncake 来进行实际的数据传输，使用了 Redis 进行元数据管理。

这里以在一个节点上启动两个 serving 的实例作为例子，介绍有分布式支持的 FlexKV 的运行方法。

## 环境搭建

### 1. 创建 Docker 容器

下载 vLLM 的镜像并创建一个 container：

```bash
docker run -it --name flexkv_dist_env \
    -v /home/FlexKV:/workspace \
    --gpus all \
    --ipc=host \
    --net=host \
    --device=/dev/infiniband/uverbs0 \
    --device=/dev/infiniband/uverbs1 \
    --device=/dev/infiniband/uverbs2 \
    --device=/dev/infiniband/uverbs3 \
    --device=/dev/infiniband/uverbs4 \
    --device=/dev/infiniband/uverbs5 \
    --device=/dev/infiniband/uverbs6 \
    --device=/dev/infiniband/uverbs7 \
    --device=/dev/infiniband/rdma_cm \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --privileged \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.10.1.1
```

### 2. 安装必要的依赖包

在容器内安装必要的包（用于分布式通信和 Redis 支持等）：

```bash
# RDMA 和基础库
apt update && apt install -y libibverbs-dev ibverbs-utils rdma-core

# io_uring 和 xxhash
apt install -y liburing-dev libxxhash0 libxxhash-dev

# Redis 客户端库
apt install -y libhiredis-dev

# JSON 和其他依赖
apt install -y libjsoncpp-dev libgflags-dev libgflags2.2

# Redis 工具和 Python 包
apt install -y redis-tools
pip install redis pandas datasets
```

### 3. 安装 Mooncake

推荐从源码安装，以便启用 Redis 支持：

```bash
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
bash dependencies.sh

mkdir build && cd build
cmake .. -DUSE_TENT=ON -DUSE_REDIS=ON
make -j
sudo make install
```

### 4. 安装 vLLM 和 FlexKV

下载 vLLM，切换到 v0.10.1.1 版本，应用 FlexKV 的 patch，然后从源码安装 vLLM 和 FlexKV。

详细步骤请参考：[FlexKV vLLM 适配文档](https://github.com/taco-project/FlexKV/blob/main/docs/vllm_adapter/README_zh.md#%E8%BF%90%E8%A1%8C)

## 运行

### 1. 启动 Redis 服务

在一个节点启动 Redis 服务。需要启动两个实例：一个给 Mooncake Engine 使用，一个给 FlexKV 使用。

```bash
# FlexKV 使用的 Redis (端口 6379)
redis-server --port 6379 --bind 10.6.131.12 --requirepass redis-serving-passwd

# Mooncake 使用的 Redis (端口 6380)
redis-server --port 6380 --bind 10.6.131.12 --requirepass redis-serving-passwd
```

### 2. 启动多个 vLLM 实例

通过脚本启动多个 vLLM instance。

> **注意**：需要根据实际情况配置 IP、端口、节点数目、FlexKV 容量配置，以及准备 FlexKV 和 Mooncake 的配置文件、vLLM 的 serving 配置等。这些都在 `start_multi_node_serving.sh` 脚本中，可以参考并修改。

```bash
cd FlexKV/scripts/multi-nodes

# 启动 2 个 instance，服务端口号从 30001 开始
bash start_multi_node_serving.sh 2 30001
```

### 3. 开始 Serving 服务

向 30001 开始的连续端口的 vLLM instance 发送 benchmark 请求即可。

## 最佳配置

> TODO: 待补充
