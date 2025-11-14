# FlexKV Distributed KVCache Reuse

FlexKV supports distributed KVCache reuse functionality. In a multi-node serving environment, you can enable KVCache sharing across multiple nodes by simply configuring a few options.

FlexKV implements KVCache index management by building a local snapshot of the global index, enabling fast global KVCache lookup and reuse. It uses Mooncake for actual data transfer and Redis for metadata management.

This guide uses an example of starting two serving instances on a single node to demonstrate how to run FlexKV with distributed support.

## Environment Setup

### 1. Create Docker Container

Pull the vLLM image and create a container:

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

### 2. Install Required Dependencies

Install the necessary packages inside the container (for distributed communication and Redis support):

```bash
# RDMA and base libraries
apt update && apt install -y libibverbs-dev ibverbs-utils rdma-core

# io_uring and xxhash
apt install -y liburing-dev libxxhash0 libxxhash-dev

# Redis client library
apt install -y libhiredis-dev

# JSON and other dependencies
apt install -y libjsoncpp-dev libgflags-dev libgflags2.2

# Redis tools and Python packages
apt install -y redis-tools
pip install redis pandas datasets
```

### 3. Install Mooncake

We recommend building from source to enable Redis support:

```bash
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
bash dependencies.sh

mkdir build && cd build
cmake .. -DUSE_TENT=ON -DUSE_REDIS=ON
make -j
sudo make install
```

### 4. Install vLLM and FlexKV

Clone vLLM, checkout version v0.10.1.1, apply the FlexKV patch, then build vLLM and FlexKV from source.

For detailed instructions, please refer to: [FlexKV vLLM Adapter Documentation](https://github.com/taco-project/FlexKV/blob/main/docs/vllm_adapter/README_en.md)

## Running

### 1. Start Redis Services

Start Redis services on one node. You need to start two instances: one for Mooncake Engine and one for FlexKV.

```bash
# Redis for FlexKV (port 6379)
redis-server --port 6379 --bind 10.6.131.12 --requirepass redis-serving-passwd

# Redis for Mooncake (port 6380)
redis-server --port 6380 --bind 10.6.131.12 --requirepass redis-serving-passwd
```

### 2. Start Multiple vLLM Instances

Use the script to start multiple vLLM instances.

> **Note**: You need to configure the IP, ports, number of nodes, FlexKV capacity settings, and prepare FlexKV and Mooncake configuration files as well as vLLM serving configurations according to your actual environment. All these settings are in the `start_multi_node_serving.sh` script, which you can refer to and modify.

```bash
cd FlexKV/scripts/multi-nodes

# Start 2 instances, with service ports starting from 30001
bash start_multi_node_serving.sh 2 30001
```

### 3. Start Serving

Send benchmark requests to the vLLM instances on consecutive ports starting from 30001.

## Best Practices

> TODO: To be added

