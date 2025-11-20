mkdir -p logs
TIMESTAMP=$(date +%Y.%m.%d-%H:%M:%S)
MODEL_PATH=${1:-YOUR_MODEL_PATH}

BATCH_SIZE=4
TP_SIZE=16
EP_SIZE=$TP_SIZE
MAX_SEQ_LEN=8192
MAX_NUM_TOKENS=8192
HOSTFILE=/cfs_zhongwei/rongwei/scripts/flexkv/trtllm_hostfile

export FLEXKV_CONFIG_PATH="./flexkv_config.json"
export TENSORRT_LLM_USE_FLEXKV=1
export FLEXKV_MASTER_HOST="172.16.0.30"
export FLEXKV_MASTER_PORTS="5556,5557,5558"


mpirun -np 16 \
--hostfile $HOSTFILE \
-mca plm_rsh_args "-p 3441" \       # TODO：1. 配置 ssh 免密；2.要改成 ssh 免密的端口号
-mca btl tcp,self \
-mca btl_tcp_if_include eth0 \
-x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
-x GLOO_SOCKET_IFNAME=eth0 \
-x NCCL_DEBUG=INFO \
-x NCCL_IBEXT_DISABLE=0 \
-x NCCL_IB_GID_INDEX=3 \
-x NCCL_IB_DISABLE=0 \
-x NCCL_NET_GDR_LEVEL=2 \
-x NCCL_IB_QPS_PER_CONNECTION=4 \
-x NCCL_IB_TC=160 \
-x NCCL_IB_TIMEOUT=22 \
-x NCCL_SOCKET_IFNAME=eth0 \
-x OMPI_MCA_btl=tcp,self \
-x OMPI_MCA_btl_tcp_if_include=eth0 \
--allow-run-as-root \
trtllm-llmapi-launch trtllm-serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 6000 \
    --backend pytorch \
    --tp_size $TP_SIZE \
    --ep_size $EP_SIZE \
    --max_seq_len $MAX_SEQ_LEN \
    --max_num_tokens $MAX_NUM_TOKENS \
    --max_batch_size $BATCH_SIZE \
    --extra_llm_api_options extra-llm-api-config.yml \
