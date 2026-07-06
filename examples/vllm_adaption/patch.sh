VLLM_DIR=YOUR_VLLM_DIR
FLEXKV_DIR=YOUR_FLEXKV_DIR

cd $VLLM_DIR && git apply --reject $FLEXKV_DIR/examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch