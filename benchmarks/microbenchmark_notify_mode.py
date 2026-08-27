"""Benchmark: layerwise notification modes (hostfunc vs polling).

Measures end-to-end H2D transfer time under:
  - notify_mode: hostfunc (default) vs polling
  - engine:      CUDA kernel vs CE (cudaMemcpyAsync)
  - data scale:  small / medium / large

Run:
    python benchmarks/bench_notify_mode.py
    python benchmarks/bench_notify_mode.py --scales small --notify hostfunc polling
    python benchmarks/bench_notify_mode.py --json results.json
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.transfer.region_batch import (
    RegionSpec,
    build_region_batch,
    make_requests,
)

# The consumer's end of the eventfd protocol lives with the tests; a benchmark
# of the notify path has to be that consumer, or ``LayerNotifier`` no-ops and
# the numbers below measure the transfer with notification switched off.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from eventfd_probe import Fds  # noqa: E402

DTYPE = torch.float16
ES = DTYPE.itemsize

SCALES = {
    # (num_layers, num_blocks, tpb, head_dim)
    "small":  (8,  16, 16, 512),
    "medium": (32, 64, 16, 512),
    "large":  (80, 256, 16, 512),
}

NOTIFY_MODES = ["hostfunc", "polling"]
ENGINES = {"cuda": False, "ce": True}
WARMUP_ITERS = 3
BENCH_ITERS = 20


# ---------- data prep ----------

def make_tensors(num_layers, num_blocks, tpb, head_dim, num_gpus):
    heads_per_rank = 1
    kv_dim = 1  # MLA: K/V combined
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=heads_per_rank,
        head_size=head_dim, kv_dim=kv_dim)
    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=1,
        head_size=head_dim, kv_dim=kv_dim)

    all_gpu = []
    for g in range(num_gpus):
        full = torch.zeros(
            (num_layers, kv_dim, num_blocks, tpb, heads_per_rank, head_dim),
            dtype=DTYPE, device=f"cuda:{g}")
        all_gpu.append([full[i] for i in range(num_layers)])

    cpu_kv = torch.zeros(tuple(cpu_layout.kv_shape), dtype=DTYPE, pin_memory=True)
    return gpu_layout, cpu_layout, all_gpu, cpu_kv


def make_group(cpu_kv, all_gpu, num_gpus, gpu_layout, cpu_layout, num_layers):
    """One region over the whole KV pool -- the shape a single-group model has."""
    spec = RegionSpec(
        name="kv",
        cpu_ptr=cpu_kv.data_ptr(),
        cpu_kv_stride=cpu_layout.get_kv_stride() * ES,
        cpu_layer_stride=cpu_layout.get_layer_stride() * ES,
        cpu_block_stride=cpu_layout.get_block_stride() * ES,
        cpu_tp_stride=(cpu_layout.get_block_stride() * ES) // num_gpus,
        gpu_block_ptrs_flat=[all_gpu[g][l].data_ptr()
                             for g in range(num_gpus)
                             for l in range(num_layers)],
        num_tensors_per_gpu=num_layers,
        gpu_kv_strides=[gpu_layout.get_kv_stride() * ES] * num_gpus,
        gpu_block_strides=[gpu_layout.get_block_stride() * ES] * num_gpus,
        gpu_layer_strides=[gpu_layout.get_layer_stride() * ES] * num_gpus,
        gpu_chunk_sizes=[gpu_layout.get_chunk_size() * ES] * num_gpus,
        num_layers=num_layers,
        kv_dim=1,
        num_kv_heads=1,
    )
    return build_region_batch(
        [spec], list(range(num_gpus)), is_blockfirst=True, num_kv_heads=1)


def bench_one(num_layers, num_blocks, tpb, head_dim, num_gpus, use_ce,
              notify_mode, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    gpu_layout, cpu_layout, all_gpu, cpu_kv = make_tensors(
        num_layers, num_blocks, tpb, head_dim, num_gpus)

    block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()

    group = make_group(
        cpu_kv, all_gpu, num_gpus, gpu_layout, cpu_layout, num_layers)

    # One request per layer, each tagged with the layer it closes: that tag is
    # what makes this a per-layer transfer rather than a whole-block one.
    requests = []
    for layer in range(num_layers):
        req = make_requests(
            1, block_ids, block_ids, True,
            transfer_num_cta=4, use_ce_transfer=use_ce,
            layer_id=layer, layer_granularity=1)[0]
        req.milestone_layer = layer
        requests.append(req)

    fds = Fds(num_counters=1, tp_size=num_gpus, num_layers=num_layers)
    group.set_layer_eventfds(
        fds.tensor(), num_gpus, num_layers, notify_mode)

    def run_once():
        group.submit_layerwise(requests, [], 0)
        ok, err = group.wait_layer_completion(120.0)
        if not ok:
            raise RuntimeError(f"layerwise transfer did not complete: {err}")
        # Drain what this iteration posted, or the fds accumulate units across
        # iterations and a real consumer would be reading stale readiness.
        for layer in range(num_layers):
            fds.units(layer)

    for _ in range(warmup):
        run_once()
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    del group
    fds.close()
    times_sorted = sorted(times)
    n = len(times_sorted)

    def _pctile(p):
        if n == 0:
            return float("nan")
        k = (n - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, n - 1)
        return times_sorted[f] + (times_sorted[c] - times_sorted[f]) * (k - f)

    return {
        "min":  times_sorted[0],
        "p50":  _pctile(50),
        "p90":  _pctile(90),
        "max":  times_sorted[-1],
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if n > 1 else 0.0,
        "n":    n,
    }


# ---------- output ----------

def _print_results_table(all_results, scales_order, engines_order, notifies_order):
    """Print one clean table per scale, then a summary table."""
    for scale_name in scales_order:
        cfg = SCALES[scale_name]
        num_layers, num_blocks, tpb, head_dim = cfg
        data_mb = (num_layers * num_blocks * tpb * head_dim * ES) / (1024**2)
        print(f"\n[{scale_name}] layers={num_layers} blocks={num_blocks} "
              f"tpb={tpb} hd={head_dim}  data≈{data_mb:.1f}MB/gpu")

        # header
        print(f"  {'engine':<6} {'notify':<9} {'p50_ms':>7} {'min_ms':>7} "
              f"{'p90_ms':>7} {'max_ms':>7} {'stdev':>6}")
        print(f"  {'-'*6:<6} {'-'*9:<9} {'-'*7:>7} {'-'*7:>7} "
              f"{'-'*7:>7} {'-'*7:>7} {'-'*6:>6}")

        for engine in engines_order:
            for notify in notifies_order:
                stats = all_results.get((scale_name, engine), {}).get(notify)
                if stats:
                    print(f"  {engine:<6} {notify:<9} "
                          f"{stats['p50']:7.2f} {stats['min']:7.2f} "
                          f"{stats['p90']:7.2f} {stats['max']:7.2f} "
                          f"{stats['stdev']:6.2f}")

        # per-engine verdict if both modes present
        if "hostfunc" in notifies_order and "polling" in notifies_order:
            for engine in engines_order:
                h = all_results.get((scale_name, engine), {}).get("hostfunc")
                p = all_results.get((scale_name, engine), {}).get("polling")
                if h and p:
                    sp = h["p50"] / p["p50"]
                    print(f"  {engine:<6} -> polling {sp:.2f}x "
                          f"({'faster' if sp > 1 else 'slower'})")

    # summary
    if "hostfunc" in notifies_order and "polling" in notifies_order:
        print(f"\n{'SUMMARY':<8} {'engine':<6} {'hostfunc':>9} {'polling':>9} "
              f"{'speedup':>8}")
        print(f"{'-'*8:<8} {'-'*6:<6} {'-'*9:>9} {'-'*9:>9} {'-'*8:>8}")
        for scale_name in scales_order:
            for engine in engines_order:
                h = all_results.get((scale_name, engine), {}).get("hostfunc")
                p = all_results.get((scale_name, engine), {}).get("polling")
                if h and p:
                    sp = h["p50"] / p["p50"]
                    print(f"{scale_name:<8} {engine:<6} "
                          f"{h['p50']:8.2f}ms {p['p50']:8.2f}ms {sp:7.2f}x")
        print()


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark layerwise notify modes (hostfunc vs polling).")
    parser.add_argument("--scales", nargs="+", default=list(SCALES),
                        choices=list(SCALES))
    parser.add_argument("--notify", nargs="+", default=NOTIFY_MODES,
                        choices=NOTIFY_MODES)
    parser.add_argument("--engines", nargs="+", default=list(ENGINES),
                        choices=list(ENGINES))
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="also write machine-readable results to PATH")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print("No GPU available")
        sys.exit(1)

    print("FlexKV Layerwise Notify-Mode Benchmark")
    print(f"  GPUs={num_gpus}  warmup={WARMUP_ITERS}  iters={BENCH_ITERS}  "
          f"dtype={str(DTYPE)}  (cuda=kernel / ce=cudaMemcpyAsync)")

    all_results = {}

    for scale_name in args.scales:
        cfg = SCALES[scale_name]
        num_layers, num_blocks, tpb, head_dim = cfg

        for engine_name in args.engines:
            use_ce = ENGINES[engine_name]
            for notify_mode in args.notify:
                try:
                    stats = bench_one(num_layers, num_blocks, tpb, head_dim,
                                       num_gpus, use_ce, notify_mode)
                    all_results.setdefault((scale_name, engine_name), {})[notify_mode] = stats
                except Exception as e:
                    print(f"[{scale_name}/{engine_name}/{notify_mode}] ERROR: {e}")

    _print_results_table(all_results, args.scales, args.engines, args.notify)

    if args.json:
        out = {
            "config": {
                "num_gpus": num_gpus,
                "warmup": WARMUP_ITERS,
                "iters": BENCH_ITERS,
                "dtype": str(DTYPE),
            },
            "results": [
                {"scale": s, "engine": e, "notify": n, **stats}
                for (s, e), modes in sorted(all_results.items())
                for n, stats in modes.items()
            ],
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"-> results written to {args.json}")


if __name__ == "__main__":
    main()
