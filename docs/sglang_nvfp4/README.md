# FlexKV + SGLang + GLM-5.2 + NVFP4 KV Cache

## SGLang Version

Use the patched SGLang branch with DSA FP4 support:

```bash
git clone -b glm5.2-nvfp4 https://github.com/linhu-nv/sglang.git
cd sglang && pip install -e "python[all]"
```

## Build FlexKV

```bash
cd FlexKV
pip install -e .
```

## Configure FlexKV

```bash
cat <<EOF > flexkv_config.yml
cpu_cache_gb: 32
EOF
```

## Launch

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/GLM-5.2-FP8 \
  --tp 8 \
  --trust-remote-code \
  --mem-fraction-static 0.7 \
  --context-length 4096 \
  --enable-flexkv \
  --flexkv-config-file ./flexkv_config.yml
```

With FP4 KV cache:

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/GLM-5.2-FP8 \
  --tp 8 \
  --trust-remote-code \
  --kv-cache-dtype fp4_e2m1 \
  --mem-fraction-static 0.7 \
  --context-length 4096 \
  --enable-flexkv \
  --flexkv-config-file ./flexkv_config.yml
```

## Verify

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/GLM-5.2-FP8",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

## Notes

- `--tp 8` required: FP8 weights ~89.5 GB/GPU, TP<8 will OOM on H20
- `--mem-fraction-static` must be > 0.648
- First launch takes 10–20 min for DeepGEMM JIT and CUDA graph capture
- `nvidia/GLM-5.2-NVFP4` has NVFP4-quantized *weights*, requires Blackwell (SM120+); use `GLM-5.2-FP8` on H20
