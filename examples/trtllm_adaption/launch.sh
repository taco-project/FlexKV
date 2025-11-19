mkdir -p logs
TIMESTAMP=$(date +%Y.%m.%d-%H:%M:%S)
MODEL_PATH=/cfs_zhongwei/models/deepseek-ai/DeepSeek-V3.1-W4AFP8-MTP-TRTLLM
MAX_SEQ_LEN=155648
MAX_NUM_TOKENS=16384


# MODEL_PATH=/cfs_zhongwei/models/Qwen3-0.6B
# MAX_SEQ_LEN=8192
# MAX_NUM_TOKENS=8192

BATCH_SIZE=4
TP_SIZE=8
EP_SIZE=$TP_SIZE

export FLEXKV_CONFIG_PATH="./flexkv_config.json"
export MODEL_PATH=$MODEL_PATH
# export CUDA_LAUNCH_BLOCKING=1
export TENSORRT_LLM_USE_FLEXKV=1
export TLLM_LOG_FIRST_RANK_ONLY=0

trtllm-serve serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 6000 \
    --backend pytorch \
    --tp_size $TP_SIZE \
    --ep_size $EP_SIZE \
    --max_seq_len $MAX_SEQ_LEN \
    --max_num_tokens $MAX_NUM_TOKENS \
    --max_batch_size $BATCH_SIZE \
    --extra_llm_api_options extra-llm-api-config-cg.yml 2>&1 | tee logs/$TIMESTAMP.log 
