mkdir -p logs
TIMESTAMP=$(date +%Y.%m.%d-%H:%M:%S)
MODEL_PATH=/cfs_zhongwei/models/deepseek-ai/DeepSeek-V3.1-W4AFP8-MTP
MAX_SEQ_LEN=49152
MAX_NUM_TOKENS=24576
TP_SIZE=8
EP_SIZE=$TP_SIZE

# MODEL_PATH=/cfs_zhongwei/models/Qwen3-0.6B
# MAX_SEQ_LEN=4096
# MAX_NUM_TOKENS=4096




export FLEXKV_CONFIG_PATH="./flexkv_config.json"
export MODEL_PATH=$MODEL_PATH
# export CUDA_LAUNCH_BLOCKING=1
export TENSORRT_LLM_USE_FLEXKV=1

trtllm-serve serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 6000 \
    --backend pytorch \
    --tp_size $TP_SIZE \
    --ep_size $EP_SIZE \
    --max_seq_len $MAX_SEQ_LEN \
    --max_num_tokens $MAX_NUM_TOKENS \
    --max_batch_size 16 \
    --kv_cache_free_gpu_memory_fraction 0.75 \
    --extra_llm_api_options extra-llm-api-config-cg.yml 2>&1 | tee logs/$TIMESTAMP.log 
