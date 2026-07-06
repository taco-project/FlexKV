mkdir -p logs


# FlexKV config             TODO 未测试
# export FLEXKV_CPU_CACHE_GB=32
# export FLEXKV_SSD_CACHE_GB=1024
# export FLEXKV_SSD_CACHE_DIR=/data/flexkv_ssd/
# export FLEXKV_ENABLE_GDS=false
export FLEXKV_CONFIG_PATH=./flexkv_config.json

MODEL_PATH=Qwen3/Qwen3-32B

VLLM_USE_V1=1 python -m vllm.entrypoints.cli.main serve $MODEL_PATH \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --port 30001 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --max_model_len 8192 \
    --max-seq-len-to-capture 8192 \
    --gpu-memory-utilization 0.8 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --kv-transfer-config \
    '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}' | tee logs/vllm.log
