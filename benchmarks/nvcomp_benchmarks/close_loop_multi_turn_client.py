#!/usr/bin/env python3
"""
Async multi-turn closed-loop OpenAI client for sglang server. Designed for
SWE-bench Oracle prefix-cache stress tests.

For Poisson open-loop arrival use open_loop-multi_turn_client.py instead.

Closed-loop behavior:
  - `N = num_clients` async workers each own a conversation and serve its
    turns *sequentially* — each turn must wait for the previous turn's
    response to finish before sending the next user message. So real
    in-flight ≈ min(num_clients, num_active_conversations).
  - Each turn streams `/v1/chat/completions` with stream_options.include_usage,
    so we get per-token timing + accurate prompt/completion token counts.

Input: prep JSON (list of {id, messages: [...alternating user/assistant...]}).
Output: bench.json (list of per-(conv, turn) records).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import aiohttp


@dataclass
class TurnRecord:
    conv_id: str
    turn_idx: int
    success: bool
    error: str | None = None
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    latency_ms: float = 0.0
    prompt_num_tokens: int = 0
    output_num_tokens: int = 0
    output_num_chunks: int = 0
    output_text: str = ""           # short, just for sanity-check sampling
    # Absolute wall-clock timestamps (unix seconds, float, ms-precision) so
    # bench.json records can be aligned 1:1 with server.log events without
    # the ±0.5s slop that comes from server.log's second-resolution HTTP
    # access lines. worker_id = closed-loop worker that owned this turn.
    worker_id: int = -1
    t_send_unix: float = 0.0        # right before session.post()
    t_first_token_unix: float = 0.0 # when first visible delta arrives
    t_end_unix: float = 0.0         # after streaming finishes / errors out
    # Sleep actually sampled by the worker before this turn's send.
    # wait_sleep_s:     uniform[0, wait_time_per_conv], only on turn_idx=0
    # prev_turn_gap_s:  exp(mean=think_time_per_turn), only on turn_idx>0
    wait_sleep_s: float = 0.0
    prev_turn_gap_s: float = 0.0


async def _stream_turn(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    timeout_s: float,
) -> TurnRecord:
    """Stream one turn. Returns a fully populated TurnRecord."""
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.0,        # deterministic-ish, removes sampling noise
    }
    rec = TurnRecord(conv_id="", turn_idx=-1, success=False)
    chunks_text: list[str] = []
    first_token_time: float | None = None
    last_chunk_time: float | None = None
    n_token_chunks = 0
    usage = None

    t_start = time.perf_counter()
    rec.t_send_unix = time.time()
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                rec.error = f"http {resp.status}: {body[:200]}"
                return rec
            async for raw in resp.content:
                # SSE: lines like "data: {...}\n\n"
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # Usage in final chunk
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    # GLM-5.1 reasoning models emit visible tokens via
                    # `reasoning_content` (chain-of-thought) BEFORE any
                    # `content` arrives. For TTFT/TPOT purposes, the first
                    # *visible* token is the first non-empty delta of either
                    # field — anything else under-reports TTFT badly.
                    piece = delta.get("content") or delta.get("reasoning_content")
                    if piece:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                            rec.t_first_token_unix = time.time()
                        last_chunk_time = now
                        n_token_chunks += 1
                        chunks_text.append(piece)
    except asyncio.TimeoutError:
        rec.t_end_unix = time.time()
        rec.error = f"timeout after {timeout_s}s"
        return rec
    except Exception as e:
        rec.t_end_unix = time.time()
        rec.error = f"{type(e).__name__}: {e}"
        return rec
    t_end = time.perf_counter()
    rec.t_end_unix = time.time()

    rec.success = True
    rec.latency_ms = (t_end - t_start) * 1e3
    if first_token_time is not None:
        rec.ttft_ms = (first_token_time - t_start) * 1e3
    rec.output_text = "".join(chunks_text)[:120]
    if usage:
        rec.prompt_num_tokens = int(usage.get("prompt_tokens", 0))
        rec.output_num_tokens = int(usage.get("completion_tokens", 0))
    else:
        rec.output_num_tokens = n_token_chunks  # fallback
    rec.output_num_chunks = n_token_chunks
    # Industry TPOT convention: average time per generated token after the
    # first token. OpenAI-compatible streaming deltas often map 1:1 to tokens
    # in practice, but the protocol only guarantees text deltas, so use the
    # usage token count when available and keep chunk count only as diagnostics.
    if (
        first_token_time is not None
        and last_chunk_time is not None
        and rec.output_num_tokens > 1
    ):
        rec.tpot_ms = (last_chunk_time - first_token_time) * 1e3 / (rec.output_num_tokens - 1)
    return rec


async def _worker(
    worker_id: int,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    max_tokens: int,
    timeout_s: float,
    results: list[TurnRecord],
    stop_after_reqs: int | None,
    counter: dict[str, int],
    max_turns_per_conv: int | None = None,
    think_time_per_turn: float = 0.0,
    wait_time_per_conv: float = 0.0,
    rng: random.Random | None = None,
):
    while True:
        try:
            conv = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        # Random sleep before EVERY new conv (including the first). This
        # decorrelates worker timing in two ways: (1) initial burst at T=0
        # is broken because each worker sleeps a different random amount
        # before its first conv; (2) ongoing conv-switch events stay
        # decorrelated since each worker independently stagger after every
        # conv completion.
        wait_sleep_sampled = 0.0
        if wait_time_per_conv > 0 and rng is not None:
            wait_sleep_sampled = rng.uniform(0, wait_time_per_conv)
            await asyncio.sleep(wait_sleep_sampled)
        try:
            cid: str = conv["id"]
            messages: list[dict] = list(conv["messages"])  # copy; we'll edit
            # The prep JSON alternates user/assistant placeholders.
            # We replay turn by turn: send messages[: 2k+1], read response,
            # overwrite assistant placeholder at 2k+1, repeat.
            num_turns = sum(1 for m in messages if m.get("role") == "user")
            if max_turns_per_conv is not None:
                num_turns = min(num_turns, max_turns_per_conv)
            sent_idx = 0          # number of user msgs already sent
            think_sleep_sampled = 0.0  # gap sampled after turn k, applied to turn k+1
            for turn in range(num_turns):
                # Take history up through this user turn (inclusive).
                next_user_pos = None
                count = 0
                for i, m in enumerate(messages):
                    if m.get("role") == "user":
                        count += 1
                        if count == turn + 1:
                            next_user_pos = i; break
                assert next_user_pos is not None
                chat = messages[: next_user_pos + 1]
                # Don't include the placeholder assistant turns past this point.
                rec = await _stream_turn(
                    session, base_url, model_name, chat, max_tokens, timeout_s
                )
                rec.conv_id = cid
                rec.turn_idx = turn
                rec.worker_id = worker_id
                rec.wait_sleep_s = wait_sleep_sampled if turn == 0 else 0.0
                rec.prev_turn_gap_s = think_sleep_sampled if turn > 0 else 0.0
                results.append(rec)
                counter["done"] += 1
                if rec.success and turn + 1 < num_turns:
                    # Overwrite the next assistant placeholder so subsequent
                    # turns carry the model's real response in their history.
                    asst_pos = next_user_pos + 1
                    if asst_pos < len(messages) and messages[asst_pos]["role"] == "assistant":
                        messages[asst_pos] = {
                            "role": "assistant",
                            "content": rec.output_text or " ",
                        }
                    # Simulate user "think time" between turns. Exponential
                    # so most turns are quick but occasional pauses are long
                    # — matches measured human chat distribution.
                    if think_time_per_turn > 0 and rng is not None:
                        think_sleep_sampled = rng.expovariate(1.0 / think_time_per_turn)
                        await asyncio.sleep(think_sleep_sampled)
                    else:
                        think_sleep_sampled = 0.0
                if stop_after_reqs is not None and counter["done"] >= stop_after_reqs:
                    return
            counter["conv_done"] += 1
            print(f"[bench] conv {counter['conv_done']}/{counter['total_convs']} done "
                  f"(worker={worker_id}, id={cid})", flush=True)
        finally:
            queue.task_done()


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _print_round_summary(tag: str, results: list[TurnRecord], wall: float) -> None:
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    ttfts = [r.ttft_ms for r in ok]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms > 0]
    e2es = [r.latency_ms for r in ok]
    out_tok = sum(r.output_num_tokens for r in ok)
    in_tok = sum(r.prompt_num_tokens for r in ok)
    total_tok = in_tok + out_tok
    print(f"[{tag}] requests done={len(results)} ok={len(ok)} fail={len(fail)}  wall={wall:.1f}s")
    print(f"[{tag}] tokens prompt={in_tok} completion={out_tok}")
    if wall > 0:
        print(f"[{tag}] throughput effective prompt={in_tok / wall:.1f} "
              f"output={out_tok / wall:.1f} total={total_tok / wall:.1f} tok/s")
    if ttfts:
        print(f"[{tag}] ttft  mean={statistics.mean(ttfts):.0f}ms  p50={_pct(ttfts,0.5):.0f}  "
              f"p95={_pct(ttfts,0.95):.0f}  p99={_pct(ttfts,0.99):.0f}")
    if tpots:
        print(f"[{tag}] tpot  mean={statistics.mean(tpots):.1f}ms  p95={_pct(tpots,0.95):.1f}")
    if e2es:
        print(f"[{tag}] e2e   mean={statistics.mean(e2es):.0f}ms  p95={_pct(e2es,0.95):.0f}")
    if fail:
        sample_errs: dict[str, int] = {}
        for r in fail:
            sample_errs.setdefault(r.error or "unknown", 0)
            sample_errs[r.error or "unknown"] += 1
        for e, n in sorted(sample_errs.items(), key=lambda x: -x[1])[:5]:
            print(f"[{tag}] fail x{n}: {e[:120]}")


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "p50": _pct(xs, 0.5),
        "p90": _pct(xs, 0.9),
        "p99": _pct(xs, 0.99),
    }


def _build_output_doc(args, results: list[TurnRecord], wall: float) -> dict:
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    ttfts = [r.ttft_ms for r in ok]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms > 0]
    e2es = [r.latency_ms for r in ok]
    out_tok = sum(r.output_num_tokens for r in ok)
    in_tok = sum(r.prompt_num_tokens for r in ok)
    total_tok = in_tok + out_tok
    turn_indices = sorted({r.turn_idx for r in results})
    turns_per_conv = (max(turn_indices) + 1) if turn_indices else 0
    return {
        "metadata": {
            "num_conversations": args.num_conversations or 0,
            "num_clients": args.num_clients,
            "think_time_per_turn": getattr(args, "think_time_per_turn", 0.0),
            "wait_time_per_conv": getattr(args, "wait_time_per_conv", 0.0),
            "seed": getattr(args, "seed", 42),
            "rounds": args.rounds,
            "max_tokens": args.max_tokens,
            "turns_per_conv": turns_per_conv,
            "total_turns": len(results),
            "wall_seconds": wall,
        },
        "summary": {
            "ok": len(ok),
            "fail": len(fail),
            "ttft_ms": _stats(ttfts),
            "tpot_ms": _stats(tpots),
            "latency_ms": _stats(e2es),
            "prompt_tokens_total": in_tok,
            "completion_tokens_total": out_tok,
            "total_tokens_total": total_tok,
            "throughput": {
                "round_wall_seconds": wall,
                "effective_prompt_tok_per_s": in_tok / wall if wall > 0 else 0.0,
                "effective_output_tok_per_s": out_tok / wall if wall > 0 else 0.0,
                "effective_total_tok_per_s": total_tok / wall if wall > 0 else 0.0,
            },
        },
        "details": [asdict(r) for r in results],
    }


async def _run_one_round(
    args,
    session: aiohttp.ClientSession,
    convs: list[dict],
    active: int,
) -> tuple[list[TurnRecord], float]:
    """Replay the same conv set once via closed-loop workers. N async
    workers pull convs from a shared queue, each worker serially runs one
    conv's turns. Returns (results, wall_seconds).

    Deep-copies each conv so per-turn assistant-placeholder overwrites
    don't leak into the next round's history.
    """
    results: list[TurnRecord] = []
    counter = {"done": 0, "conv_done": 0, "total_convs": len(convs)}
    convs_copy = [{"id": c["id"], "messages": [dict(m) for m in c["messages"]]}
                  for c in convs]
    max_turns_per_conv = getattr(args, "max_turns_per_conv", None)

    rng = random.Random(args.seed)
    think_time_per_turn = getattr(args, "think_time_per_turn", 0.0)
    wait_time_per_conv = getattr(args, "wait_time_per_conv", 0.0)

    queue: asyncio.Queue = asyncio.Queue()
    for c in convs_copy:
        queue.put_nowait(c)

    t0 = time.perf_counter()
    workers = [
        asyncio.create_task(
            _worker(
                i, queue, session, args.base_url, args.model_name,
                args.max_tokens, args.request_timeout_sec, results,
                args.max_num_requests, counter, max_turns_per_conv,
                think_time_per_turn, wait_time_per_conv, rng,
            )
        )
        for i in range(active)
    ]
    await asyncio.gather(*workers)
    wall = time.perf_counter() - t0
    return results, wall


async def run_bench(args) -> int:
    with open(args.input_file) as f:
        all_convs: list[dict] = json.load(f)

    # Shuffle the input pool deterministically before cycling. Without
    # this, conv arrival order is fixed by JSON file order, so any prefix
    # bias in the dataset shows up as systematic burst pattern. Same seed
    # → reproducible across baseline/nvcomp runs (so the SAME conv hits
    # the server at the SAME relative time in both variants).
    random.Random(args.seed).shuffle(all_convs)

    if args.max_active_conversations is None:
        active = args.num_clients
    else:
        active = min(args.max_active_conversations, args.num_clients)

    # Pick N convs; cycle through input if N > len(all_convs). Recycled
    # instances get an "#k" id suffix so detail records stay unique while
    # the underlying prompt prefix (and thus FlexKV cache hit) stays the
    # same. Useful when sweep arrivals exceed the input-file conv count.
    N = args.num_conversations or len(all_convs)
    convs = []
    for i in range(N):
        src = all_convs[i % len(all_convs)]
        if i >= len(all_convs):
            convs.append({"id": f"{src['id']}#{i // len(all_convs)}",
                          "messages": src["messages"]})
        else:
            convs.append(src)

    rounds = max(1, args.rounds)
    conn = aiohttp.TCPConnector(limit=max(active * 4, 32))
    timeout = aiohttp.ClientTimeout(total=None)
    final_results: list[TurnRecord] = []
    final_wall = 0.0
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        for ri in range(1, rounds + 1):
            tag = f"round {ri}/{rounds}" + (" — warmup" if ri < rounds else " — measure")
            print(f"\n[bench] === {tag} ===")
            results, wall = await _run_one_round(args, session, convs, active)
            _print_round_summary(f"r{ri}", results, wall)
            if ri == rounds:
                final_results = results
                final_wall = wall
            if ri < rounds and args.inter_round_sleep_sec > 0:
                print(f"[bench] sleeping {args.inter_round_sleep_sec}s before next round...")
                await asyncio.sleep(args.inter_round_sleep_sec)

    # Only the LAST round's per-turn records are persisted as bench.json.
    # Earlier rounds are warmup — their stats live in stdout only. This
    # matches online_benchmark_client.py's `warmup = round 1, benchmark =
    # last round` convention.
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    output = _build_output_doc(args, final_results, final_wall)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[bench] === final (round {rounds}) — written to {args.output_file} ===")
    _print_round_summary("final", final_results, final_wall)

    fail = [r for r in final_results if not r.success]
    return 0 if not fail else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", required=True,
                    help="prep JSON: list of {id, messages}")
    ap.add_argument("--output-file", required=True,
                    help="bench.json output")
    ap.add_argument("--base-url", default="http://127.0.0.1:30001",
                    help="sglang server base URL")
    ap.add_argument("--model-name", default="glm-5.1-fp8",
                    help="served-model-name as configured in sglang server")
    ap.add_argument("--num-clients", type=int, default=8,
                    help="async workers (= true in-flight conv concurrency)")
    ap.add_argument("--max-active-conversations", type=int, default=None,
                    help="cap (defaults to num_clients)")
    ap.add_argument("--num-conversations", type=int, default=None,
                    help="cap on convs taken from input (default all)")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="output max_tokens per turn (long-input/short-output)")
    ap.add_argument("--max-num-requests", type=int, default=None,
                    help="early-stop after this many turn requests completed")
    ap.add_argument("--max-turns-per-conv", type=int, default=None,
                    help="cap turns sent per conv (default no cap, replay all "
                         "turns from the input JSON). Set to 1 for warmup runs "
                         "that only need to populate the cold 128k prefix into "
                         "the FlexKV CPU pool.")
    ap.add_argument("--request-timeout-sec", type=float, default=600.0)
    ap.add_argument("--rounds", type=int, default=1,
                    help="Number of times to replay the same conv set in one "
                         "process. Only the LAST round's per-turn records go to "
                         "the output bench.json (earlier rounds are warmup). "
                         "Modeled after online_benchmark_client.py's --rounds.")
    ap.add_argument("--inter-round-sleep-sec", type=float, default=0.0,
                    help="Seconds to sleep between rounds (default 0). For "
                         "this design we don't insert /flush_cache between "
                         "rounds — the workload naturally re-routes through "
                         "FlexKV CPU pool by round 2 if working set > GPU "
                         "radix capacity.")
    ap.add_argument("--think-time-per-turn", type=float, default=0.0,
                    help="Mean inter-turn delay in seconds (exponential dist). "
                         "Simulates user reading/thinking between turns. "
                         "0 (default)=back-to-back torture test; 15s is a "
                         "reasonable SWE-bench-agent value; 30-60s matches "
                         "real chat workloads. Decorrelates worker arrivals "
                         "so scheduler doesn't see synchronized ready bursts.")
    ap.add_argument("--wait-time-per-conv", type=float, default=0.0,
                    help="Each worker sleeps uniform [0, N] sec before EVERY "
                         "new conv (including the first). Decorrelates "
                         "worker timing: (1) initial T=0 burst is broken "
                         "since each worker waits a different random "
                         "amount; (2) conv-switch events stay decorrelated "
                         "thereafter. Set to ~20s (a typical per-req cycle "
                         "for ISL=128k) for cleanest de-sync.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for conv shuffling + jitter RNG. Same "
                         "seed → reproducible across baseline/nvcomp runs.")
    args = ap.parse_args()

    rc = asyncio.run(run_bench(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
