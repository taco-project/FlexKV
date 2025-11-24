#!/bin/bash
redis-cli -h 10.6.131.12 -p 6379 -a redis-serving-passwd FLUSHALL
redis-cli -h 10.6.131.12 -p 6380 -a redis-serving-passwd FLUSHALL
export FLEXKV_CONFIG_PATH="/workspace/FlexKV/node1.yml"
export FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_1"
export LD_LIBRARY_PATH=/workspace/sdk/:$LD_LIBRARY_PATH
export PYTHONPATH=/workspace/sdk/:$PYTHONPATH
export MOONCAKE_CONFIG_PATH="/workspace/FlexKV/tests/test_dist_e2e/mooncake_config_r.json"
export MC_REDIS_PASSWORD="redis-serving-passwd"
export MC_LEGACY_RPC_PORT_BINDING=12841
#export FLEXKV_IORING_ENTRIES=0
#VLLM_USE_V1=1 python3 -m vllm.entrypoints.cli.main serve /workspace/Qwen3-8B \
VLLM_USE_V1=1 python3 -m vllm.entrypoints.openai.api_server \
     --model /workspace/Qwen3-8B \
     --tensor-parallel-size 2 \
     --trust-remote-code \
     --port 30001 \
     --max-num-seqs 128 \
     --max-num-batched-tokens 8192 \
     --max_model_len 8192 \
     --max-seq-len-to-capture 8192 \
     --gpu-memory-utilization 0.4 \
     --enable-chunked-prefill \
     --enable-prefix-caching \
     --kv-transfer-config \
        '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
