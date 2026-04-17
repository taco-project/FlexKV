#!/usr/bin/env python3
"""
FlexKV IPC Round-Trip Benchmark (DP-concurrent)

Measures get_match batch latency with dp_size concurrent DP clients:
  - direct : dp_size independent KVManagers (no shared server, no IPC)
  - zmq    : dp_size KVDPClients   -> 1 shared KVServer (ZMQ IPC)
  - shm    : dp_size ShmKVDPClients -> 1 shared KVServer (SHM IPC)

Usage:
    python roundtrip_bench.py --mode direct --dp-size 4
    python roundtrip_bench.py --mode zmq    --dp-size 4 --tp-size 1
    python roundtrip_bench.py --mode shm    --dp-size 4 --tp-size 1
"""

import argparse
import csv
import os
import sys
import time
import subprocess
import pickle
import textwrap
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

# ── Port helpers ───────────────────────────────────────────────

IPC_PREFIX = "ipc:///tmp/flexkv_rt_bench"


def make_ports(tag, index=0):
    """Generate a unique (server_recv_port, gpu_register_port) pair."""
    base = f"{IPC_PREFIX}_{tag}_{index}"
    return base, base + "_gpu"


# ── Configs ────────────────────────────────────────────────────


def make_configs(tp_size=1, dp_size=1):
    from flexkv.common.config import ModelConfig, CacheConfig

    mc = ModelConfig(
        num_layers=4,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
        use_mla=False,
        tp_size=tp_size,
        dp_size=dp_size,
    )
    cc = CacheConfig(
        tokens_per_block=16,
        enable_cpu=True,
        enable_ssd=False,
        num_cpu_blocks=256,
    )
    return mc, cc


# ── Server subprocess (zmq / shm) ─────────────────────────────


def start_server(model_config, cache_config, gpu_register_port,
                 server_recv_port, total_clients, shm_mode, log_path):
    """Start a KVServer subprocess with explicit ports and client count."""
    args_blob = pickle.dumps(
        (model_config, cache_config, gpu_register_port,
         server_recv_port, total_clients, shm_mode)
    )

    import flexkv
    flexkv_root = str(Path(flexkv.__file__).parent.parent)

    script = textwrap.dedent(f"""\
        import sys, pickle
        sys.path.insert(0, {flexkv_root!r})
        from flexkv.server.server import KVServer
        mc, cc, grp, srp, tc, sm = pickle.loads({args_blob!r})
        server = KVServer(mc, cc, grp, srp, tc, sm)
        server.run()
    """)

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_fh = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=log_fh, stderr=subprocess.STDOUT, env=env,
        )
    except Exception:
        log_fh.close()
        raise
    return proc, log_fh


def stop_server(server, log_fh):
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    log_fh.close()


# ── Wait helpers ──────────────────────────────────────────────


def wait_for_zmq_socket(server_recv_port, server_proc=None, timeout=30):
    sock_path = server_recv_port.replace("ipc://", "")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(sock_path):
            return
        if server_proc and server_proc.poll() is not None:
            raise RuntimeError(
                f"Server exited early (rc={server_proc.returncode}), check log")
        time.sleep(0.1)
    raise TimeoutError(f"ZMQ socket not ready: {sock_path}")


def wait_for_shm_files(server_recv_port, dp_size=1, server_proc=None, timeout=60):
    from flexkv.server.shm_channel import _sanitize_server_id

    safe = _sanitize_server_id(server_recv_port)
    needed = [f"/dev/shm/flexkv_shm_ctrl_{safe}"]
    for i in range(dp_size):
        needed.append(f"/dev/shm/flexkv_shm_ch_{safe}_{i}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(os.path.exists(p) for p in needed):
            return
        if server_proc and server_proc.poll() is not None:
            raise RuntimeError(
                f"Server exited early (rc={server_proc.returncode}), check log")
        time.sleep(0.1)
    raise TimeoutError(f"SHM files not created within {timeout}s: {needed}")


def wait_ready(check_fn, label, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if check_fn():
                print(f"  [{label}] ready")
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"[{label}] FlexKV not ready within {timeout}s")


# ── TP client (GPU block registration) ─────────────────────────


def _tp_worker(model_config, cache_config, num_gpu_blocks, tp_rank,
               ready_event, gpu_register_port, device_id, dp_client_id=0):
    from flexkv.server.client import KVTPClient
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    tp_client = KVTPClient(gpu_register_port, dp_client_id=dp_client_id, device_id=device_id)

    layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )
    if not model_config.use_mla and model_config.tp_size > 1:
        layout = layout.div_head(model_config.tp_size)

    blocks = [
        torch.empty(tuple(layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
        for _ in range(model_config.num_layers)
    ]
    tp_client.register_to_server(blocks, layout)
    ready_event.set()
    while True:
        time.sleep(3600)


def start_tp_clients(model_config, cache_config, num_gpu_blocks,
                     gpu_register_port, base_device=0, dp_client_id=0):
    tp_size = model_config.tp_size
    num_gpus = torch.cuda.device_count()
    ctx = mp.get_context("spawn")
    procs = []
    for tp_rank in range(tp_size):
        device_id = (base_device + tp_rank) % num_gpus
        ready = ctx.Event()
        p = ctx.Process(
            target=_tp_worker,
            args=(model_config, cache_config, num_gpu_blocks, tp_rank,
                  ready, gpu_register_port, device_id, dp_client_id),
            daemon=True,
        )
        p.start()
        procs.append(p)
        if not ready.wait(timeout=120):
            raise TimeoutError(
                f"TP rank={tp_rank} dp={dp_client_id} timed out (port={gpu_register_port})")
    return procs


def terminate_procs(procs):
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()


# ── Global cleanup ────────────────────────────────────────────


def cleanup_all():
    import signal
    import glob as _g

    my_pid = os.getpid()
    for pat in ("flexkv", "multiprocessing.spawn", "multiprocessing.resource_tracker"):
        try:
            result = subprocess.run(
                ["pgrep", "-f", pat], capture_output=True, text=True)
            for pid_str in result.stdout.strip().split("\n"):
                if pid_str and int(pid_str) != my_pid:
                    try:
                        os.kill(int(pid_str), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except FileNotFoundError:
            pass

    for f in _g.glob("/tmp/flexkv_rt_bench*"):
        try:
            os.unlink(f)
        except OSError:
            pass
    for f in _g.glob("/dev/shm/flexkv_shm_*"):
        try:
            os.unlink(f)
        except OSError:
            pass
    time.sleep(0.5)


# ── Benchmark core ─────────────────────────────────────────────


def compute_stats(latencies_ns):
    us = np.array(latencies_ns, dtype=np.float64) / 1000.0
    return dict(
        mean=np.mean(us),
        p50=np.percentile(us, 50),
        p90=np.percentile(us, 90),
        p99=np.percentile(us, 99),
        min=np.min(us),
        max=np.max(us),
    )


def _bench_loop(get_match_fn, token_sizes, warmup, reqs_per_client, num_rounds,
                warmup_barrier, start_barrier, end_barrier):
    """Worker-side: warmup followed by barrier-synced measurement rounds."""
    for n in token_sizes:
        token_ids = np.random.randint(0, 32000, size=n, dtype=np.int64)
        for _ in range(warmup):
            get_match_fn(token_ids, None, -1)
        warmup_barrier.wait()
        for _ in range(num_rounds):
            start_barrier.wait()
            for _ in range(reqs_per_client):
                get_match_fn(token_ids, None, -1)
            end_barrier.wait()


def _collect_timing(dp_size, batch_size, token_sizes, num_rounds,
                    warmup_barrier, start_barrier, end_barrier):
    """Main-process: synchronise with workers via barriers, collect wall-clock timing."""
    reqs_per_client = batch_size // dp_size
    actual_total = reqs_per_client * dp_size
    print(f"  [bench] dp_size={dp_size}  reqs_per_client={reqs_per_client}  "
          f"total_reqs={actual_total}")

    results = {}
    for n in token_sizes:
        warmup_barrier.wait()
        batch_wall_ns = []
        for _ in range(num_rounds):
            start_barrier.wait()
            t0 = time.perf_counter_ns()
            end_barrier.wait()
            t1 = time.perf_counter_ns()
            batch_wall_ns.append(t1 - t0)

        results[n] = dict(
            batch_stats=compute_stats(batch_wall_ns),
            per_req_stats=compute_stats(
                (np.array(batch_wall_ns, dtype=np.float64)
                 / actual_total).astype(np.int64).tolist()
            ),
            raw_batch_ns=batch_wall_ns,
            batch_size=batch_size,
            dp_size=dp_size,
        )
        bs = results[n]["batch_stats"]
        ps = results[n]["per_req_stats"]
        print(
            f"  tokens={n:>5}  "
            f"batch: mean={bs['mean']:.0f}us  p50={bs['p50']:.0f}us  "
            f"p99={bs['p99']:.0f}us  |  "
            f"per_req: mean={ps['mean']:.1f}us"
        )
    return results


def print_table(mode, results):
    dp_size = next(iter(results.values()))["dp_size"]
    batch_size = next(iter(results.values()))["batch_size"]
    hdr = (
        f"{'Tokens':>8}  {'Batch Mean':>12}  {'Batch P50':>12}  "
        f"{'Batch P90':>12}  {'Batch P99':>12}  "
        f"{'Per-Req Mean':>14}  {'Per-Req P50':>14}  {'Per-Req P99':>14}"
    )
    print(f"\n{'=' * len(hdr)}")
    print(f"  Mode: {mode} | dp_size: {dp_size} | batch_size: {batch_size}")
    print(f"{'=' * len(hdr)}")
    print("  (all values in microseconds)")
    print(hdr)
    print("-" * len(hdr))
    for n in sorted(results):
        bs = results[n]["batch_stats"]
        ps = results[n]["per_req_stats"]
        print(
            f"{n:>8}  {bs['mean']:>12.0f}  {bs['p50']:>12.0f}  "
            f"{bs['p90']:>12.0f}  {bs['p99']:>12.0f}  "
            f"{ps['mean']:>14.1f}  {ps['p50']:>14.1f}  {ps['p99']:>14.1f}"
        )


def save_csv(mode, dp_size, results, log_dir):
    batch_size = next(iter(results.values()))["batch_size"]
    path = os.path.join(log_dir, f"results_{mode}_dp{dp_size}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "mode", "dp_size", "tokens", "batch_size",
            "batch_mean_us", "batch_p50_us", "batch_p99_us",
            "per_req_mean_us", "per_req_p50_us", "per_req_p99_us",
        ])
        for n in sorted(results):
            bs = results[n]["batch_stats"]
            ps = results[n]["per_req_stats"]
            w.writerow([
                mode, dp_size, n, batch_size,
                f"{bs['mean']:.1f}", f"{bs['p50']:.1f}", f"{bs['p99']:.1f}",
                f"{ps['mean']:.1f}", f"{ps['p50']:.1f}", f"{ps['p99']:.1f}",
            ])
    print(f"CSV saved to {path}")
    return path


# ── Mode: direct (multi-process, no GIL contention) ─────────


def _direct_bench_worker(rank, tp_size, num_gpu_blocks, recv_port, gpu_port,
                         token_sizes, warmup, reqs_per_client, num_rounds,
                         ready_event, warmup_barrier,
                         start_barrier, end_barrier):
    """Worker process: own KVManager + own GIL, runs get_match calls."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    from flexkv.kvmanager import KVManager
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
    GLOBAL_CONFIG_FROM_ENV.server_client_mode = False

    mc, cc = make_configs(tp_size=tp_size)
    km = KVManager(mc, cc,
                   server_recv_port=recv_port,
                   gpu_register_port=gpu_port)
    km.start()
    tp_procs = start_tp_clients(mc, cc, num_gpu_blocks, gpu_port,
                                base_device=rank * tp_size)
    while not km.is_ready():
        time.sleep(0.1)
    ready_event.set()

    _bench_loop(km.get_match, token_sizes, warmup, reqs_per_client, num_rounds,
                warmup_barrier, start_barrier, end_barrier)

    km.shutdown()
    terminate_procs(tp_procs)


def run_direct(args):
    """Direct mode baseline: dp_size independent KVManagers in separate processes.

    Each process has its own Python GIL, matching a real dp=N deployment.
    Barriers synchronize round start/end for wall-clock timing.
    """
    cleanup_all()
    dp_size = args.dp_size
    reqs_per_client = args.batch_size // dp_size

    ctx = mp.get_context("spawn")
    warmup_barrier = ctx.Barrier(dp_size + 1)
    start_barrier = ctx.Barrier(dp_size + 1)
    end_barrier = ctx.Barrier(dp_size + 1)

    workers = []
    ready_events = []
    try:
        for i in range(dp_size):
            recv_port, gpu_port = make_ports("direct", i)
            ready = ctx.Event()
            p = ctx.Process(
                target=_direct_bench_worker,
                args=(i, args.tp_size, args.num_gpu_blocks,
                      recv_port, gpu_port,
                      args.token_sizes, args.warmup, reqs_per_client,
                      args.num_rounds, ready,
                      warmup_barrier, start_barrier, end_barrier),
            )
            p.start()
            workers.append(p)
            ready_events.append(ready)

        for i, ready in enumerate(ready_events):
            if not ready.wait(timeout=120):
                raise TimeoutError(f"direct-{i} not ready within 120s")
            print(f"  [direct-{i}] ready")

        results = _collect_timing(
            dp_size, args.batch_size, args.token_sizes, args.num_rounds,
            warmup_barrier, start_barrier, end_barrier)

        for p in workers:
            p.join(timeout=10)
        return results
    finally:
        terminate_procs(workers)


# ── Worker for IPC modes (zmq / shm) ──────────────────────────


def _ipc_bench_worker(rank, mode, tp_size, dp_size, num_gpu_blocks,
                      server_recv_port, gpu_register_port,
                      token_sizes, warmup, reqs_per_client, num_rounds,
                      ready_event, warmup_barrier,
                      start_barrier, end_barrier):
    """Worker process for zmq/shm: connect to shared KVServer as DP client.

    Each worker runs in a separate process — matching real vLLM deployment
    where every DP rank is an independent EngineCoreProc with its own GIL.
    """
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    mc, cc = make_configs(tp_size=tp_size, dp_size=dp_size)

    if mode == "zmq":
        wait_for_zmq_socket(server_recv_port)
        from flexkv.server.client import KVDPClient
        client = KVDPClient(server_recv_port, mc, dp_client_id=rank)
    else:
        wait_for_shm_files(server_recv_port, dp_size=dp_size)
        from flexkv.server.client import ShmKVDPClient
        client = ShmKVDPClient(server_recv_port, mc, dp_client_id=rank)

    client.start_server_and_register()
    print(f"  [{mode}-dp{rank}] registered")

    tp_procs = start_tp_clients(mc, cc, num_gpu_blocks, gpu_register_port,
                                base_device=rank * tp_size, dp_client_id=rank)
    print(f"  [{mode}-dp{rank}] {tp_size} TP client(s) registered")

    wait_ready(client.is_ready, f"{mode}-dp{rank}")
    ready_event.set()

    _bench_loop(client.get_match, token_sizes, warmup, reqs_per_client,
                num_rounds, warmup_barrier, start_barrier, end_barrier)

    if mode == "zmq":
        client.shutdown()
    terminate_procs(tp_procs)


# ── Mode: zmq ─────────────────────────────────────────────────


def run_zmq(args):
    """ZMQ mode: dp_size KVDPClients in separate processes → 1 shared KVServer."""
    cleanup_all()
    recv_port, gpu_port = make_ports("zmq")
    mc, cc = make_configs(tp_size=args.tp_size, dp_size=args.dp_size)
    log_path = os.path.join(args.log_dir, "server_zmq.log")
    dp_size = args.dp_size
    reqs_per_client = args.batch_size // dp_size

    ctx = mp.get_context("spawn")
    warmup_barrier = ctx.Barrier(dp_size + 1)
    start_barrier = ctx.Barrier(dp_size + 1)
    end_barrier = ctx.Barrier(dp_size + 1)

    server_proc = log_fh = None
    workers = []
    ready_events = []
    try:
        server_proc, log_fh = start_server(
            mc, cc, gpu_port, recv_port, dp_size,
            shm_mode=False, log_path=log_path)
        print(f"[zmq] Server pid={server_proc.pid}, log={log_path}")

        for i in range(dp_size):
            ready = ctx.Event()
            p = ctx.Process(
                target=_ipc_bench_worker,
                args=(i, "zmq", args.tp_size, dp_size, args.num_gpu_blocks,
                      recv_port, gpu_port,
                      args.token_sizes, args.warmup, reqs_per_client,
                      args.num_rounds, ready,
                      warmup_barrier, start_barrier, end_barrier),
            )
            p.start()
            workers.append(p)
            ready_events.append(ready)

        for i, ready in enumerate(ready_events):
            if not ready.wait(timeout=120):
                raise TimeoutError(f"zmq-{i} not ready within 120s")

        results = _collect_timing(
            dp_size, args.batch_size, args.token_sizes, args.num_rounds,
            warmup_barrier, start_barrier, end_barrier)

        for p in workers:
            p.join(timeout=10)
        return results
    finally:
        terminate_procs(workers)
        time.sleep(0.5)
        if server_proc is not None:
            stop_server(server_proc, log_fh)


# ── Mode: shm ─────────────────────────────────────────────────


def run_shm(args):
    """SHM mode: dp_size ShmKVDPClients in separate processes → 1 shared KVServer."""
    cleanup_all()
    recv_port, gpu_port = make_ports("shm")
    mc, cc = make_configs(tp_size=args.tp_size, dp_size=args.dp_size)
    log_path = os.path.join(args.log_dir, "server_shm.log")
    dp_size = args.dp_size
    reqs_per_client = args.batch_size // dp_size

    ctx = mp.get_context("spawn")
    warmup_barrier = ctx.Barrier(dp_size + 1)
    start_barrier = ctx.Barrier(dp_size + 1)
    end_barrier = ctx.Barrier(dp_size + 1)

    server_proc = log_fh = None
    workers = []
    ready_events = []
    try:
        server_proc, log_fh = start_server(
            mc, cc, gpu_port, recv_port, dp_size,
            shm_mode=True, log_path=log_path)
        print(f"[shm] Server pid={server_proc.pid}, log={log_path}")

        for i in range(dp_size):
            ready = ctx.Event()
            p = ctx.Process(
                target=_ipc_bench_worker,
                args=(i, "shm", args.tp_size, dp_size, args.num_gpu_blocks,
                      recv_port, gpu_port,
                      args.token_sizes, args.warmup, reqs_per_client,
                      args.num_rounds, ready,
                      warmup_barrier, start_barrier, end_barrier),
            )
            p.start()
            workers.append(p)
            ready_events.append(ready)

        for i, ready in enumerate(ready_events):
            if not ready.wait(timeout=120):
                raise TimeoutError(f"shm-{i} not ready within 120s")

        results = _collect_timing(
            dp_size, args.batch_size, args.token_sizes, args.num_rounds,
            warmup_barrier, start_barrier, end_barrier)

        for p in workers:
            p.join(timeout=10)
        return results
    finally:
        terminate_procs(workers)
        if server_proc is not None:
            stop_server(server_proc, log_fh)


# ── Main ───────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="FlexKV IPC Round-Trip Benchmark (DP-concurrent)")
    p.add_argument("--mode", choices=["direct", "zmq", "shm"], required=True)
    p.add_argument("--dp-size", type=int, default=4,
                   help="Data parallel size (number of concurrent DP clients)")
    p.add_argument("--tp-size", type=int, default=1,
                   help="Tensor parallel size (GPUs per DP rank)")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Total get_match calls per batch "
                        "(distributed across dp_size clients)")
    p.add_argument("--num-rounds", type=int, default=10,
                   help="Rounds of batch measurement for statistics")
    p.add_argument("--warmup", type=int, default=50,
                   help="Warmup iterations per DP client")
    p.add_argument("--token-sizes", type=int, nargs="+",
                   default=[128, 256, 512, 1024, 2048, 4096, 8192],
                   help="Token counts to benchmark")
    p.add_argument("--num-gpu-blocks", type=int, default=64)
    p.add_argument("--log-dir", type=str, default="./logs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ["FLEXKV_ENABLE_MPS"] = "0"

    if args.batch_size % args.dp_size != 0:
        args.batch_size = (args.batch_size // args.dp_size) * args.dp_size
        print(f"NOTE: batch_size adjusted to {args.batch_size} "
              f"(divisible by dp_size={args.dp_size})")

    print("FlexKV Round-Trip Benchmark (DP-concurrent)")
    print(f"  mode={args.mode}  dp_size={args.dp_size}  tp_size={args.tp_size}")
    print(f"  batch_size={args.batch_size}  num_rounds={args.num_rounds}  "
          f"warmup={args.warmup}")
    print(f"  token_sizes={args.token_sizes}")
    print(f"  log_dir={args.log_dir}")
    print()

    runner = {"direct": run_direct, "zmq": run_zmq, "shm": run_shm}[args.mode]
    try:
        results = runner(args)
    finally:
        cleanup_all()

    print_table(args.mode, results)

    raw = {str(n): np.array(results[n]["raw_batch_ns"]) for n in results}
    out = os.path.join(args.log_dir,
                       f"latencies_{args.mode}_dp{args.dp_size}.npz")
    np.savez(out, **raw)
    print(f"\nRaw batch latencies saved to {out}")

    save_csv(args.mode, args.dp_size, results, args.log_dir)


if __name__ == "__main__":
    main()
