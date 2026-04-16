#!/usr/bin/env python3
"""
FlexKV IPC Round-Trip Benchmark
Measures get_match round-trip latency across three IPC modes:
  - direct : KVManager -> KVTaskEngine  (in-process, no IPC)
  - zmq    : KVDPClient -> ZMQ pickle   -> KVServer -> KVTaskEngine
  - shm    : ShmKVDPClient -> SHM binary -> KVServer -> KVTaskEngine

Usage:
    python roundtrip_bench.py --mode direct
    python roundtrip_bench.py --mode zmq  --tp-size 2
    python roundtrip_bench.py --mode shm  --tp-size 4
"""

import argparse
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

# ── IPC endpoints ──────────────────────────────────────────────

IPC_PREFIX = "ipc:///tmp/flexkv_rt_bench"
SERVER_RECV_PORT = IPC_PREFIX
GPU_REGISTER_PORT = IPC_PREFIX + "_gpu"

# ── Configs ────────────────────────────────────────────────────


def make_configs(tp_size=1):
    from flexkv.common.config import ModelConfig, CacheConfig

    mc = ModelConfig(
        num_layers=4,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
        use_mla=False,
        tp_size=tp_size,
        dp_size=1,
    )
    cc = CacheConfig(
        tokens_per_block=16,
        enable_cpu=True,
        enable_ssd=False,
        num_cpu_blocks=256,
    )
    return mc, cc


# ── Server subprocess (zmq / shm) ─────────────────────────────


def start_server(model_config, cache_config, shm_mode, log_path):
    """Start a real FlexKV server subprocess; capture stdout+stderr to *log_path*."""
    args_blob = pickle.dumps(
        (model_config, cache_config, GPU_REGISTER_PORT, SERVER_RECV_PORT, 1, shm_mode)
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
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except Exception:
        log_fh.close()
        raise
    return proc, log_fh


def stop_server(server, log_fh):
    """Terminate server subprocess gracefully, escalate to SIGKILL if needed."""
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    log_fh.close()


def wait_for_zmq_socket(timeout=30):
    """Poll for ZMQ IPC socket file instead of blind sleep."""
    sock_path = SERVER_RECV_PORT.replace("ipc://", "")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(sock_path):
            return
        time.sleep(0.1)
    raise TimeoutError(f"ZMQ socket not ready: {sock_path}")


# ── TP client (GPU block registration) ─────────────────────────


def _tp_worker(model_config, cache_config, num_gpu_blocks, tp_rank, ready_event):
    from flexkv.server.client import KVTPClient
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

    # Use physical GPU index matching the server (which also pops CUDA_VISIBLE_DEVICES)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    device_id = tp_rank
    tp_client = KVTPClient(GPU_REGISTER_PORT, dp_client_id=0, device_id=device_id)

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


def start_tp_clients(model_config, cache_config, num_gpu_blocks):
    """Spawn one TP client process per tp_rank and wait for all to register."""
    tp_size = model_config.tp_size
    avail = torch.cuda.device_count()
    if avail < tp_size:
        raise RuntimeError(f"Need {tp_size} GPUs, only {avail} available")

    ctx = mp.get_context("spawn")
    procs = []
    for tp_rank in range(tp_size):
        ready = ctx.Event()
        p = ctx.Process(
            target=_tp_worker,
            args=(model_config, cache_config, num_gpu_blocks, tp_rank, ready),
            daemon=True,
        )
        p.start()
        procs.append(p)
        if not ready.wait(timeout=120):
            raise TimeoutError(f"TP client rank={tp_rank} registration timed out")
    print(f"  {tp_size} TP client(s) registered")
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
    """Kill residual flexkv processes and clean all IPC artifacts.
    Must run before each mode and after all tests complete."""
    import signal, glob as _g

    # Kill stale flexkv server/worker processes (not ourselves).
    # mp.spawn workers show as "multiprocessing.spawn" in cmdline,
    # not "flexkv", so we must match both patterns.
    my_pid = os.getpid()
    for proc_name in ("flexkv", "multiprocessing.spawn", "multiprocessing.resource_tracker"):
        try:
            result = subprocess.run(
                ["pgrep", "-f", proc_name], capture_output=True, text=True
            )
            for pid_str in result.stdout.strip().split("\n"):
                if pid_str and int(pid_str) != my_pid:
                    try:
                        os.kill(int(pid_str), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except FileNotFoundError:
            pass  # pgrep not available

    # Clean ZMQ IPC sockets
    for suffix in ("", "_gpu"):
        sock_path = IPC_PREFIX.replace("ipc://", "") + suffix
        try:
            os.unlink(sock_path)
        except OSError:
            pass

    # Clean SHM files
    for f in _g.glob("/dev/shm/flexkv_shm_*"):
        try:
            os.unlink(f)
        except OSError:
            pass

    time.sleep(0.5)  # let resources settle


# ── ZMQ helpers ────────────────────────────────────────────────


def cleanup_zmq_ipc():
    for suffix in ("", "_gpu"):
        sock_path = IPC_PREFIX.replace("ipc://", "") + suffix
        try:
            os.unlink(sock_path)
        except OSError:
            pass


# ── SHM helpers ────────────────────────────────────────────────


def cleanup_shm():
    import glob as _g

    for f in _g.glob("/dev/shm/flexkv_shm_*"):
        try:
            os.unlink(f)
        except OSError:
            pass


def wait_for_shm_files(timeout=60):
    from flexkv.server.shm_channel import _sanitize_server_id

    safe = _sanitize_server_id(SERVER_RECV_PORT)
    needed = [
        f"/dev/shm/flexkv_shm_ctrl_{safe}",
        f"/dev/shm/flexkv_shm_ch_{safe}_0",
    ]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(os.path.exists(p) for p in needed):
            return
        time.sleep(0.1)
    raise TimeoutError(f"SHM files not created: {needed}")


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
        req_s=1e6 / np.mean(us),
    )


def bench_get_match(get_match_fn, token_sizes, batch_size, warmup, num_rounds):
    results = {}
    for n in token_sizes:
        token_ids = np.random.randint(0, 32000, size=n, dtype=np.int64)

        for _ in range(warmup):
            get_match_fn(token_ids, None, -1)

        batch_total_ns = []
        for _ in range(num_rounds):
            t0 = time.perf_counter_ns()
            for _ in range(batch_size):
                get_match_fn(token_ids, None, -1)
            t1 = time.perf_counter_ns()
            batch_total_ns.append(t1 - t0)

        results[n] = dict(
            batch_stats=compute_stats(batch_total_ns),
            per_req_stats=compute_stats(
                (np.array(batch_total_ns, dtype=np.float64) / batch_size).astype(np.int64).tolist()
            ),
            raw_batch_ns=batch_total_ns,
            batch_size=batch_size,
        )
        bs = results[n]["batch_stats"]
        ps = results[n]["per_req_stats"]
        print(
            f"  tokens={n:>5}  batch({batch_size})="
            f"mean {bs['mean']:.0f}us  p50 {bs['p50']:.0f}us  p99 {bs['p99']:.0f}us  |  "
            f"per_req= mean {ps['mean']:.1f}us  p50 {ps['p50']:.1f}us"
        )
    return results


def print_table(mode, results):
    batch_size = next(iter(results.values()))["batch_size"]
    hdr = (
        f"{'Tokens':>8}  {'Batch Mean':>12}  {'Batch P50':>12}  {'Batch P90':>12}  "
        f"{'Batch P99':>12}  {'Per-Req Mean':>14}  {'Per-Req P50':>14}  {'Per-Req P99':>14}"
    )
    print(f"\n{'=' * len(hdr)}")
    print(f"  Mode: {mode} | Operation: get_match | Batch size: {batch_size}")
    print(f"{'=' * len(hdr)}")
    print(f"  (all values in microseconds)")
    print(hdr)
    print("-" * len(hdr))
    for n in sorted(results):
        bs = results[n]["batch_stats"]
        ps = results[n]["per_req_stats"]
        print(
            f"{n:>8}  {bs['mean']:>12.0f}  {bs['p50']:>12.0f}  {bs['p90']:>12.0f}  "
            f"{bs['p99']:>12.0f}  {ps['mean']:>14.1f}  {ps['p50']:>14.1f}  {ps['p99']:>14.1f}"
        )


# ── Wait-for-ready helper ─────────────────────────────────────


def wait_ready(check_fn, label, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if check_fn():
                print(f"[{label}] FlexKV ready")
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"[{label}] FlexKV not ready within {timeout}s")


# ── Mode: direct ───────────────────────────────────────────────


def run_direct(args):
    cleanup_all()
    mc, cc = make_configs(tp_size=args.tp_size)
    from flexkv.kvmanager import KVManager
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

    saved_scm = GLOBAL_CONFIG_FROM_ENV.server_client_mode
    GLOBAL_CONFIG_FROM_ENV.server_client_mode = False
    km = None
    tp_procs = []
    try:
        km = KVManager(mc, cc, server_recv_port=SERVER_RECV_PORT, gpu_register_port=GPU_REGISTER_PORT)
        km.start()
        tp_procs = start_tp_clients(mc, cc, args.num_gpu_blocks)
        wait_ready(km.is_ready, "direct")

        return bench_get_match(
            lambda tid, tm, lg: km.get_match(tid, tm, lg),
            args.token_sizes,
            args.batch_size,
            args.warmup,
            args.num_rounds,
        )
    finally:
        if km is not None:
            km.shutdown()
        terminate_procs(tp_procs)
        GLOBAL_CONFIG_FROM_ENV.server_client_mode = saved_scm


# ── Mode: zmq ─────────────────────────────────────────────────


def run_zmq(args):
    cleanup_all()
    mc, cc = make_configs(tp_size=args.tp_size)
    log_path = os.path.join(args.log_dir, "server_zmq.log")
    server = log_fh = client = None
    tp_procs = []
    try:
        server, log_fh = start_server(mc, cc, shm_mode=False, log_path=log_path)
        print(f"[zmq] Server pid={server.pid}, log={log_path}")
        wait_for_zmq_socket()

        from flexkv.server.client import KVDPClient

        client = KVDPClient(SERVER_RECV_PORT, mc, dp_client_id=0)
        client.start_server_and_register()
        tp_procs = start_tp_clients(mc, cc, args.num_gpu_blocks)
        wait_ready(client.is_ready, "zmq")

        return bench_get_match(
            lambda tid, tm, lg: client.get_match(tid, tm, lg),
            args.token_sizes,
            args.batch_size,
            args.warmup,
            args.num_rounds,
        )
    finally:
        if client is not None:
            client.shutdown()
            time.sleep(0.5)
        if server is not None:
            stop_server(server, log_fh)
        terminate_procs(tp_procs)
        cleanup_zmq_ipc()


# ── Mode: shm ─────────────────────────────────────────────────


def run_shm(args):
    cleanup_all()
    mc, cc = make_configs(tp_size=args.tp_size)
    log_path = os.path.join(args.log_dir, "server_shm.log")
    server = log_fh = client = None
    tp_procs = []
    try:
        server, log_fh = start_server(mc, cc, shm_mode=True, log_path=log_path)
        print(f"[shm] Server pid={server.pid}, log={log_path}")
        wait_for_shm_files()

        from flexkv.server.client import ShmKVDPClient

        client = ShmKVDPClient(SERVER_RECV_PORT, mc, dp_client_id=0)
        client.start_server_and_register()
        tp_procs = start_tp_clients(mc, cc, args.num_gpu_blocks)
        wait_ready(client.is_ready, "shm")

        return bench_get_match(
            lambda tid, tm, lg: client.get_match(tid, tm, lg),
            args.token_sizes,
            args.batch_size,
            args.warmup,
            args.num_rounds,
        )
    finally:
        if client is not None:
            client.shutdown()
            time.sleep(0.5)
        if server is not None:
            stop_server(server, log_fh)
        terminate_procs(tp_procs)
        cleanup_shm()


# ── Main ───────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="FlexKV IPC Round-Trip Benchmark")
    p.add_argument("--mode", choices=["direct", "zmq", "shm"], required=True)
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (GPUs per dp rank)")
    p.add_argument("--batch-size", type=int, default=128, help="Number of get_match calls per batch")
    p.add_argument("--num-rounds", type=int, default=10, help="Number of batch rounds for statistics")
    p.add_argument("--warmup", type=int, default=100, help="Warmup iterations per token size")
    p.add_argument(
        "--token-sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Token counts to benchmark",
    )
    p.add_argument("--num-gpu-blocks", type=int, default=64)
    p.add_argument("--log-dir", type=str, default="./logs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # MPS daemon interferes with spawned TP worker processes (CUDA device assertion).
    # Force-disable it for the benchmark.
    os.environ["FLEXKV_ENABLE_MPS"] = "0"

    print(f"FlexKV Round-Trip Benchmark")
    print(f"  mode={args.mode}  tp_size={args.tp_size}")
    print(f"  batch_size={args.batch_size}  num_rounds={args.num_rounds}  warmup={args.warmup}")
    print(f"  token_sizes={args.token_sizes}")
    print(f"  log_dir={args.log_dir}")
    print()

    runner = {"direct": run_direct, "zmq": run_zmq, "shm": run_shm}[args.mode]
    try:
        results = runner(args)
    finally:
        cleanup_all()

    print_table(args.mode, results)

    # Save raw batch timing data
    raw = {str(n): np.array(results[n]["raw_batch_ns"]) for n in results}
    out = os.path.join(args.log_dir, f"latencies_{args.mode}.npz")
    np.savez(out, **raw)
    print(f"\nRaw batch latencies saved to {out}")


if __name__ == "__main__":
    main()
