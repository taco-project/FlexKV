MODEL_PATH=YOUR_MODEL_PATH
DATASET_PATH=YOUR_DATASET_PATH

for workers in 32 64 128 256; do
    concurrency_multiplier=8
    if [ $workers -gt 128 ]; then
        concurrency_multiplier=4
    fi
    vllm bench serve \
        --backend vllm \
        --model $MODEL_PATH \
        --dataset-name random \
        --dataset-path $DATASET_PATH \
        --random-input-len 1005 \
        --random-output-len 635 \
        --num-prompts $((workers*concurrency_multiplier)) \
        --max-concurrency $workers \
        --host 0.0.0.0 \
        --port 30001 \
        --ignore-eos
done