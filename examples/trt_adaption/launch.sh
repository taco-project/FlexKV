# export PYTHONPATH=/path/to/your/connector/module:$PYTHONPATH
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d%H%M%S)
MODEL_PATH=/cfs_zhongwei/models/deepseek-ai/DeepSeek-V3.1-W4AFP8-MTP

cat <<EOF > ./flexkv_config.json
{
    "server_recv_port": "ipc:///tmp/flexkv_test",
    "cache_config": {
          "enable_cpu": true,
          "num_cpu_blocks": 10240
    },
    "num_log_interval_requests": 200
}
EOF

export FLEXKV_CONFIG_PATH="./flexkv_config.json"
export MODEL_PATH=$MODEL_PATH
# export PMIX_MCA_gds=hash
export CUDA_LAUNCH_BLOCKING=1

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# export MPI4PY_RC_INITIALIZE=false

trtllm-serve serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 6000 \
    --backend pytorch \
    --tp_size 8 \
    --ep_size 8 \
    --max_seq_len 49152 \
    --max_num_tokens 24576 \
    --max_batch_size 16 \
    --kv_cache_free_gpu_memory_fraction 0.75 \
    --extra_llm_api_options extra-llm-api-config.yml 2>&1 | tee logs/$TIMESTAMP.log 
