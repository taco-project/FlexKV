# SGLang speculative draft replay benchmark

`../bench_sglang_draft_replay.py` compares cold computation with device, host,
and remote cache replay. It requires the SGLang draft-aware FlexKV adapter and
the matching FlexKV connector. The fixture in this directory is for an
exclusively owned benchmark server: it drains transfers, resets cache metadata
between independent experiments, and clears **only L2** when establishing an
L3 replay. Keep its HTTP listener on loopback.

## Freeze inputs

```bash
python scripts/bench_sglang_draft_replay.py prepare \
  --dataset /datasets/sharegpt.json --tokenizer /models/DeepSeek-V4.1-Flash \
  --samples 200 --min-tokens 512 --max-tokens 4096 \
  --output /results/dsv41-inputs.jsonl
```

The default input format is the first user message encoded as raw completion
tokens. Use `--chat-template` only when the tokenizer provides a suitable chat
template. Reuse the resulting JSONL unchanged for both arms. Its companion
manifest records the dataset hash, tokenizer, selection seed, and token lengths.
GLM and DeepSeek need their own tokenizer-specific frozen input files.

For a focused reproduction, add `--replay-page-size 256 --replay-suffix-tokens 16`
for V4.1, or page size `64` for GLM. This truncates real prompts to a whole-page
prefix plus 16 tokens. With decoder bounded replay disabled, the short remaining
prefill cannot rebuild an entire missing draft window. Treat this deliberately
constructed workload separately from natural-length prompts; retain both input
manifests and never compare different frozen input files.

## Start matching server arms

Add this fixture to `PYTHONPATH`, before the source checkouts, and enable it only
for the benchmark processes:

```bash
export PYTHONPATH="$PWD/scripts/sglang_draft_replay:$SGLANG_CHECKOUT/python:$PWD"
export FLEXKV_BENCHMARK_CONTROL_PORT=31933
export FLEXKV_BENCHMARK_STATE_DIR=/results/draft-replay-control
export FLEXKV_ENABLE_LAYERWISE_TRANSFER=1
export FLEXKV_ENABLE_MPS=0
export FLEXKV_CONFIG_PATH=/results/flexkv.json
```

The `sitecustomize.py` fixture installs itself only when the control port is
nonzero. Its opt-in `FLEXKV_BENCHMARK_DRAFT_CACHE=0` switch removes the draft pool
argument at the connector boundary, reproducing target-only backup while
retaining the corrected bigram keys and replay boundaries in both arms. Set it
to `1` for the fixed arm. This switch belongs to the fixture, not the production
connector. Restart the model between arms; retain the same model, launch flags,
transfer mode, input file, and sampling settings.

For the full H20 experiments, the relevant model flags are:

| Model | Required speculative/cache flags |
| --- | --- |
| GLM-5.2-FP8 | `--tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` |
| DeepSeek-V4.1-Flash | `--tp-size 8 --ep-size 8 --kv-cache-dtype fp8_e4m3 --page-size 256 --attention-backend dsv4 --dsv4-prefill-backend flashmla_sparse_q8 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6` |

Add `--enable-flexkv --enable-cache-report --enable-metrics`, and choose context,
token capacity, and memory settings that fit the complete checkpoint. On H20,
set `SGLANG_DSV4_KV_LAYOUT=v4`. The V4.1 intermediate expert dimension in TP8
alone does not meet the SM90 MXFP4 kernel alignment; EP8 retains whole experts.

Example FlexKV configuration, with an independently deployed Mooncake Store:

```json
{
  "cpu_cache_gb": 4,
  "ssd_cache_gb": 0,
  "enable_gds": false,
  "swa": {"num_cpu_slots": 64, "num_remote_slots": 128},
  "use_mooncake_store_backend": true,
  "mooncake_store_config_path": "/results/mooncake-client.json",
  "enable_chunked_prefetch": true,
  "prefetch_options": {"policy": "wait_complete"}
}
```

The connector resolves `wait_complete` to its whole-task prefetch path. This
benchmark does not establish chunked prefetch overlap performance. Both arms
must use the same policy. Check that Python imports the modified `connector.py`;
a release `connector.so` in the same directory can shadow source changes.

## Measure and compare

For GLM, use `--page-size 64 --key-mode bigram --reprefill-tail 0`.
For V4.1 with decoder bounded replay enabled and a 128-token window, use
`--page-size 256 --key-mode token --reprefill-tail 128`. To reproduce the full
prefill/SWA snapshot path, launch **both** arms with
`--no-enable-decoder-swa-bounded-replay` and use `--reprefill-tail 0` instead.
Read the resolved server configuration: bounded replay can be enabled by
default and is not numerically equivalent to full prefill.

```bash
python scripts/bench_sglang_draft_replay.py measure \
  --url http://127.0.0.1:31931 --control-url http://127.0.0.1:31933 \
  --inputs /results/dsv41-inputs.jsonl --output /results/before \
  --page-size 256 --key-mode token --reprefill-tail 128 \
  --output-tokens 128 --repeats 3
# Restart with FLEXKV_BENCHMARK_DRAFT_CACHE=1, then run with --output /results/after.
python scripts/bench_sglang_draft_replay.py compare \
  --before /results/before --after /results/after \
  --output /results/comparison.json
```

Each prompt follows cold -> boundary seed -> L1 -> L2 -> L3. The seed ends at
the exact reusable boundary so that the hybrid SWA snapshot belongs to that
prefix. A different real prompt overwrites reusable device slots before L2/L3,
preventing stale draft bytes at lucky allocator addresses from hiding missing
backup. Seed and churn requests are excluded from the acceptance statistics.
Use at least two distinct inputs. `--no-churn` is an explicitly weaker control.

An L3 sample requires a complete remote metadata prefix before L2 clearing,
empty L2 allocation afterward, reported storage-hit tokens on the request,
and a positive remote main-KV payload publication counter. SWA payloads are
additional and appear in native transfer logs; this counter is not NIC wire
traffic or total main+SWA bytes. A miss, short output,
missing speculative counters, or missing output IDs invalidates the run. The
script saves the failed sample and refuses to compare incomplete runs.

For long runs on a small exclusively owned Store, `--store-control-url` can
point to a separately supplied `/clear` endpoint returning `{"valid": true}`.
It is called only after all-tier metadata reset between independent experiments,
never between a seed and its L3 replay. This avoids accumulating unreachable
objects from unique salts. Do not direct this option at a shared Store. The
ordinary L3 transition uses the fixture's L2-only reset and preserves Store data.

Reports include verify-weighted acceptance length, strict draft acceptance
rate, per-request counters, cache tiers, remote bytes, elapsed client time, and
paired output IDs. Comparisons resample prompt IDs, retaining repeats together,
to obtain a paired bootstrap interval. They also report the replay-to-cold gap
for each arm. Complete output equality is separate from acceptance; neither a
cache hit nor an acceptance gain establishes model quality or production
correctness. Single-host TCP L3 results do not validate multi-host RDMA.
