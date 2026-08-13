#!/usr/bin/env python3
"""Benchmark FlexKV prefix cache (CPU restore) across serving frameworks.

Sends the same prompt three times and checks:
  1. Cold start  — pure compute, no cache anywhere.
  2. GPU hit     — framework native prefix cache (VRAM).
  3. FlexKV hit  — GPU cache flushed, restore from FlexKV CPU pool.

The first --max-tokens output tokens from each request are compared; a
mismatch across the three requests is a hard failure.

Usage:
  python benchmarks/benchmark_prefix_cache.py \
      --model /data/models/qwen/Qwen3-8B \
      --port 8000

  # Force a framework (skip auto-detect)
  python benchmarks/benchmark_prefix_cache.py --model ... --framework sglang
"""

import argparse
import json
import random
import string
import sys
import time
from enum import Enum

import requests


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------

class Framework(str, Enum):
    SGLANG = "sglang"
    VLLM = "vllm"
    TRTLLM = "trtllm"
    UNKNOWN = "unknown"


def detect_framework(base_url: str, timeout: float = 5.0) -> tuple:
    """Probe well-known endpoints to identify the serving framework and model.

    Returns (Framework, model_name_or_None, max_model_len_or_None).
    """
    model_name = None
    max_model_len = None

    # All three frameworks expose /v1/models (OpenAI-compatible).
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                model_name = models[0].get("id")
                # vLLM sometimes includes max_model_len in model metadata.
                max_model_len = models[0].get("max_model_len")
    except Exception:
        pass

    # SGLang exposes /get_server_info with context_len.
    try:
        resp = requests.get(f"{base_url}/get_server_info", timeout=timeout)
        if resp.status_code == 200 and "sglang" in resp.text.lower():
            info = resp.json()
            if max_model_len is None:
                max_model_len = info.get("context_len")
            return Framework.SGLANG, model_name, max_model_len
    except Exception:
        pass

    # vLLM has /health; TRT-LLM typically does not.
    try:
        health = requests.get(f"{base_url}/health", timeout=timeout)
        if health.status_code == 200:
            return Framework.VLLM, model_name, max_model_len
    except Exception:
        pass

    # /v1/models worked but neither sglang nor vllm signature found → assume TRT-LLM.
    if model_name is not None:
        return Framework.TRTLLM, model_name, max_model_len

    return Framework.UNKNOWN, None, None


# ---------------------------------------------------------------------------
# Framework-specific cache flush
# ---------------------------------------------------------------------------

def flush_cache_sglang(base_url: str) -> None:
    """SGLang provides a direct /flush_cache endpoint."""
    print(">>> [sglang] Flushing GPU prefix cache via /flush_cache ...")
    try:
        resp = requests.post(f"{base_url}/flush_cache", timeout=30)
        if resp.status_code == 200:
            print("    flush OK")
        else:
            print(f"    flush returned HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as e:
        print(f"    flush error: {e}")
    time.sleep(1)


def flush_cache_vllm(base_url: str, model: str, api_url: str) -> None:
    """vLLM has no standard flush endpoint.

    Try /reset_prefix_cache first (some vLLM forks expose it); fall back to
    evicting the GPU cache by sending many distinct large prompts.
    """
    # Attempt direct flush (vLLM fork / newer versions).
    try:
        resp = requests.post(f"{base_url}/reset_prefix_cache", timeout=10)
        if resp.status_code == 200:
            print(">>> [vllm] Flushed GPU prefix cache via /reset_prefix_cache")
            time.sleep(1)
            return
    except Exception:
        pass

    # Eviction fallback.
    print(">>> [vllm] Evicting GPU prefix cache with random prompts ...")
    eviction_count = 20
    prompt_len = 8000
    for i in range(eviction_count):
        random_text = "".join(
            random.choices(string.ascii_letters + string.digits + " ", k=prompt_len)
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"evict-{i}"},
                {"role": "user", "content": random_text[:4000]},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        try:
            requests.post(api_url, json=payload, timeout=60)
        except Exception:
            pass
        if (i + 1) % 5 == 0:
            print(f"    evicted {i + 1}/{eviction_count}")
    print("    eviction done")
    time.sleep(1)


def flush_cache_trtllm(base_url: str) -> None:
    """TRT-LLM does not expose a standard cache-flush endpoint.

    Try /flush_cache (some builds); otherwise warn and skip.
    """
    try:
        resp = requests.post(f"{base_url}/flush_cache", timeout=10)
        if resp.status_code == 200:
            print(">>> [trtllm] Flushed cache via /flush_cache")
            time.sleep(1)
            return
    except Exception:
        pass
    print(">>> [trtllm] No flush endpoint available; relying on eviction by cold prompt divergence")


def flush_cache(framework: Framework, base_url: str, model: str, api_url: str) -> None:
    if framework == Framework.SGLANG:
        flush_cache_sglang(base_url)
    elif framework == Framework.VLLM:
        flush_cache_vllm(base_url, model, api_url)
    elif framework == Framework.TRTLLM:
        flush_cache_trtllm(base_url)
    else:
        print(">>> [unknown] Attempting /flush_cache ...")
        try:
            requests.post(f"{base_url}/flush_cache", timeout=10)
        except Exception:
            pass
    print()


# ---------------------------------------------------------------------------
# Request helper
# ---------------------------------------------------------------------------

def measure_step(
    step_name: str,
    api_url: str,
    model: str,
    messages: list,
    max_tokens: int,
) -> tuple:
    """Measure TTFT and capture output for one step.

    Returns (ttft, output_text, total_time).
    """
    print(f">>> {step_name}")
    t0 = time.time()

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    try:
        response = requests.post(api_url, json=payload, stream=True, timeout=300)
    except requests.exceptions.RequestException as e:
        print(f"    [ERROR] request failed: {e}")
        return float("inf"), "", float("inf")

    if response.status_code != 200:
        print(f"    [ERROR] HTTP {response.status_code}: {response.text[:200]}")
        return float("inf"), "", float("inf")

    ttft = None
    output_parts = []
    chunk_count = 0

    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: ") or decoded == "data: [DONE]":
            continue
        try:
            data = json.loads(decoded[6:])
        except json.JSONDecodeError:
            continue

        if "error" in data:
            print(f"    [ERROR] server: {data['error']}")
            return float("inf"), "", float("inf")

        if "choices" not in data or len(data["choices"]) == 0:
            continue

        delta = data["choices"][0].get("delta", {})
        if chunk_count < 3:
            print(f"    [Debug Chunk {chunk_count}] delta: {delta}")
        chunk_count += 1

        content = delta.get("content", "")
        if content and ttft is None:
            ttft = time.time() - t0

        if content:
            output_parts.append(content)

    output_text = "".join(output_parts)

    if ttft is None:
        print(f"    [WARN] got {chunk_count} chunks but no content")
        return float("inf"), "", float("inf")

    total = time.time() - t0
    print(f"    output: {repr(output_text[:80])}{'...' if len(output_text) > 80 else ''}")
    print(f"    TTFT: {ttft:.4f}s | total: {total:.4f}s\n")
    return ttft, output_text, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_prompt(target_tokens: int = 4096) -> list:
    """Build a deterministic prompt of approximately *target_tokens* tokens.

    Uses a repeating paragraph (~120 tokens per repetition at ~4 chars/token).
    """
    para = (
        "Quantum mechanics is a fundamental theory in physics that provides "
        "a description of the physical properties of nature at the scale of "
        "atoms and subatomic particles.\n"
        "It is the foundation of all quantum physics including quantum chemistry, "
        "quantum field theory, quantum technology, and quantum information science.\n"
        "Classical physics, the collection of theories that existed before the "
        "advent of quantum mechanics, describes many aspects of nature at an "
        "ordinary (macroscopic) scale, but is not sufficient for describing them "
        "at small (atomic and subatomic) scales.\n"
    )
    # ~480 chars/paragraph ≈ 120 tokens. Repeat to reach target.
    chars_per_token = 4
    repeats = max(1, target_tokens * chars_per_token // len(para))
    base_text = para * repeats

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Please summarize the following text in one sentence:\n{base_text}"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark FlexKV prefix cache (CPU restore) across frameworks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=None, help="Model name/path (auto-detected from /v1/models if omitted)")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument(
        "--framework",
        choices=["auto", "sglang", "vllm", "trtllm"],
        default="auto",
        help="Serving framework (default: auto-detect)",
    )
    parser.add_argument("--max-tokens", type=int, default=10, help="Max tokens to generate per request (default: 10)")
    parser.add_argument("--prompt-tokens", type=int, default=None, help="Approximate prompt length in tokens (auto-sized to model max if omitted)")
    parser.add_argument("--flexkv-wait", type=float, default=3.0, help="Seconds to wait for FlexKV async D2H (default: 3)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"

    # Detect framework and model.
    if args.framework == "auto":
        print("Auto-detecting framework and model ...")
        framework, detected_model, max_model_len = detect_framework(base_url)
        if framework == Framework.UNKNOWN:
            print("[ERROR] Could not auto-detect framework. Use --framework to specify.")
            return 1
        print(f"Detected: framework={framework.value}, model={detected_model}, max_model_len={max_model_len}")
    else:
        framework = Framework(args.framework)
        # Still try to auto-detect model name and max len.
        _, detected_model, max_model_len = detect_framework(base_url)
        print(f"Framework: {framework.value}, model={detected_model}, max_model_len={max_model_len}")

    # Resolve model name: CLI arg takes priority, then auto-detected.
    model = args.model or detected_model
    if not model:
        print("[ERROR] Could not determine model name. Use --model to specify.")
        return 1
    print(f"Using model: {model}\n")

    # Resolve prompt token count.
    if args.prompt_tokens is not None:
        prompt_tokens = args.prompt_tokens
    else:
        # Auto-size: use half of max_model_len (or 4096 fallback), capped at 32768.
        if max_model_len and max_model_len > 0:
            prompt_tokens = min(max_model_len // 2, 32768)
        else:
            prompt_tokens = 4096
        # Leave room for max_tokens + safety margin.
        if max_model_len and prompt_tokens + args.max_tokens + 256 > max_model_len:
            prompt_tokens = max(256, max_model_len - args.max_tokens - 256)
    print(f"Prompt tokens: ~{prompt_tokens}\n")

    messages = build_prompt(prompt_tokens)

    print(f"=== FlexKV prefix cache benchmark: {model} ({framework.value}) ===\n")

    # Clean start.
    flush_cache(framework, base_url, model, api_url)

    # Step 1: Cold start.
    ttft_cold, out_cold, _ = measure_step(
        "Step 1: Cold start (pure compute)", api_url, model, messages, args.max_tokens
    )

    # Wait for FlexKV async D2H.
    print(f">>> Waiting {args.flexkv_wait}s for FlexKV async write to CPU ...\n")
    time.sleep(args.flexkv_wait)

    # Step 2: GPU prefix cache hit.
    ttft_gpu, out_gpu, _ = measure_step(
        "Step 2: GPU prefix cache hit", api_url, model, messages, args.max_tokens
    )

    # Step 3: Flush GPU cache, keep FlexKV CPU cache.
    flush_cache(framework, base_url, model, api_url)

    # Step 4: FlexKV cache hit (GPU miss, restore from CPU).
    ttft_flexkv, out_flexkv, _ = measure_step(
        "Step 3: FlexKV cache hit (restore from CPU)", api_url, model, messages, args.max_tokens
    )

    # ---- Report ----
    print("=" * 60)
    print("Results")
    print("=" * 60)

    if float("inf") in (ttft_cold, ttft_gpu, ttft_flexkv):
        print("FAIL: one or more steps failed (see errors above).")
        return 1

    print(f"1. Cold start TTFT      : {ttft_cold:.4f}s")
    print(f"2. GPU cache hit TTFT   : {ttft_gpu:.4f}s")
    print(f"3. FlexKV restore TTFT  : {ttft_flexkv:.4f}s")
    print("-" * 60)

    # Output consistency check.
    outputs = [out_cold, out_gpu, out_flexkv]
    all_match = all(o == outputs[0] for o in outputs[1:])
    print(f"Output tokens match across 3 requests: {'YES' if all_match else 'NO'}")
    if not all_match:
        print("  [WARN] Outputs differ — temperature=0 should be deterministic.")
        for i, o in enumerate(outputs):
            print(f"  Request {i+1}: {repr(o[:100])}")
        print()

    # TTFT comparison.
    if ttft_cold > ttft_flexkv > ttft_gpu:
        print("PASS: cold > FlexKV restore > GPU hit (as expected)")
        savings = (ttft_cold - ttft_flexkv) / ttft_cold * 100
        print(f"FlexKV reduced TTFT by {savings:.1f}% vs cold start")
        return 0 if all_match else 1
    elif ttft_cold > ttft_flexkv:
        savings = (ttft_cold - ttft_flexkv) / ttft_cold * 100
        print(f"FlexKV reduced TTFT by {savings:.1f}% vs cold start")
        print("(GPU hit was not faster than FlexKV restore, possibly due to eviction)")
        return 0 if all_match else 1
    else:
        print("FAIL: FlexKV restore was slower than cold start.")
        print("Possible causes: FlexKV async write not finished, or CPU restore too slow.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
