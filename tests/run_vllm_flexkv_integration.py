"""Run a real vLLM -> FlexKV -> vLLM cache round trip.

The first request computes and offloads its KV cache.  Local vLLM prefix
caching is disabled, so the identical second request must load its prefix from
FlexKV.  The script requires a non-zero external cache hit and byte-correct KV
is validated indirectly by requiring deterministic generated token IDs to be
identical before and after the round trip.
"""

import argparse
import json
import time

from vllm import LLM, SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--attention-backend")
    parser.add_argument("--max-model-len", type=int, default=512)
    args = parser.parse_args()

    llm_kwargs = {}
    if args.attention_backend:
        llm_kwargs["attention_config"] = {"backend": args.attention_backend}

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=False,
        tensor_parallel_size=1,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=0.2,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        disable_custom_all_reduce=True,
        kv_transfer_config={
            "kv_connector": "FlexKVConnectorV1",
            "kv_role": "kv_both",
        },
        **llm_kwargs,
    )
    prompt = (
        "FlexKV integration correctness test. "
        "The same long prefix is intentionally repeated so that it spans many "
        "KV cache blocks and exercises an external cache load. "
    ) * 16
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        seed=1234,
    )

    first = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
    time.sleep(1.0)
    second = llm.generate([prompt], sampling_params, use_tqdm=False)[0]

    first_token_ids = list(first.outputs[0].token_ids)
    second_token_ids = list(second.outputs[0].token_ids)
    result = {
        "kv_cache_dtype": args.kv_cache_dtype,
        "attention_backend": args.attention_backend,
        "prompt_tokens": len(first.prompt_token_ids),
        "first_cached_tokens": first.num_cached_tokens,
        "second_cached_tokens": second.num_cached_tokens,
        "first_output_token_ids": first_token_ids,
        "second_output_token_ids": second_token_ids,
    }
    print("FLEXKV_VLLM_RESULT=" + json.dumps(result, sort_keys=True))

    if not second.num_cached_tokens:
        raise AssertionError("second vLLM request did not report a FlexKV cache hit")
    if first_token_ids != second_token_ids:
        raise AssertionError(
            "generated token IDs changed after the FlexKV KV-cache round trip"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
