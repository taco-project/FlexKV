#!/usr/bin/env python3
"""SSD transfer microbenchmark: baseline vs opt (real C++ call).

Benchmarks real SSD I/O via ``flexkv.c_ext.transfer_kv_blocks_ssd`` for two
strategies:

  - baseline : ``ssd_io_opt=False``  -> unbatched per-(block,layer) fragmented I/O
  - opt      : ``ssd_io_opt=True``   -> path auto-selected from layout + contiguity

    * BLOCKFIRST              -> bf_single_io_per_block
    * LAYERFIRST + contiguous -> lf_layer_major_batch
    * LAYERFIRST + scattered  -> lf_vectored

Output is a single consolidated table comparing baseline vs opt across:
    layout(BF/LF) x block_order(contig/scattered) x model(MHA/MLA)
    x num_blocks x direction(SSD->CPU / CPU->SSD)
plus per-case I/O stats (BaseIOs / OptIOs / OptAvgKB, from /proc/self/io,
fallback engine only) to diagnose whether batching engaged, and a BF-vs-LF
summary with a layout recommendation at the end.

Usage:
  python benchmarks/microbenchmark_ssd_simulation.py
  python benchmarks/microbenchmark_ssd_simulation.py --read-threads 32 --write-threads 32
  python benchmarks/microbenchmark_ssd_simulation.py --sizes small --blocks 256 1024 --rounds 5
  python benchmarks/microbenchmark_ssd_simulation.py --cold-read --ssd-dir /path/to/real/nvme

  NOTE: --ssd-dir is now REQUIRED -- it must point to a real disk (your NVMe
  cache_dir), not tmpfs /tmp. There is no default.
"""
import argparse
import ctypes
import mmap
import os
import sys
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

TOKENS_PER_BLOCK = 16


@dataclass
class ModelPreset:
    name: str
    num_layers: int
    num_kv_heads: int
    head_size: int
    kv_dim: int  # 1=K/V combined (MLA), 2=K/V separate (MHA)
    dtype: torch.dtype = torch.bfloat16

    @property
    def elem_size(self) -> int:
        return self.dtype.itemsize

    @property
    def chunk_size(self) -> int:
        return TOKENS_PER_BLOCK * self.num_kv_heads * self.head_size

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_size * self.elem_size


MODEL_SIZES: Dict[str, Dict[str, int]] = {
    "small":  {"num_layers": 32, "num_kv_heads": 8,  "head_size": 128},
    "medium": {"num_layers": 80, "num_kv_heads": 8,  "head_size": 128},
    "large":  {"num_layers": 80, "num_kv_heads": 16, "head_size": 256},
}


def build_presets(sizes: List[str], mla_modes: List[str]) -> Dict[str, ModelPreset]:
    presets = {}
    for size in sizes:
        cfg = MODEL_SIZES[size]
        for mode in mla_modes:
            if mode == "MHA":
                name = f"{size}-MHA"
                presets[name] = ModelPreset(name, cfg["num_layers"], cfg["num_kv_heads"],
                                            cfg["head_size"], kv_dim=2)
            elif mode == "MLA":
                name = f"{size}-MLA"
                presets[name] = ModelPreset(name, cfg["num_layers"], 1,
                                            cfg["head_size"], kv_dim=1)
    return presets


def make_layout(layout_type: KVCacheLayoutType, preset: ModelPreset, num_blocks: int) -> KVCacheLayout:
    return KVCacheLayout(
        type=layout_type,
        num_layer=preset.num_layers,
        num_block=num_blocks,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_head=preset.num_kv_heads,
        head_size=preset.head_size,
        kv_dim=preset.kv_dim,
    )


def compute_strides(layout: KVCacheLayout, preset: ModelPreset) -> dict:
    return {
        "chunk_bytes": preset.chunk_bytes,
        "block_stride_bytes": layout.get_block_stride() * preset.elem_size,
        "layer_stride_bytes": layout.get_layer_stride() * preset.elem_size,
        "kv_stride_bytes": layout.get_kv_stride() * preset.elem_size,
        "total_bytes": layout.get_total_elements() * preset.elem_size,
    }


# SSD transfer path identifiers used by predict_opt_path().
PATH_NAMES: List[str] = [
    "bf_single_io_per_block",   # one I/O per block
    "lf_layer_major_batch",     # one I/O per (layer, kv)
    "lf_vectored",              # preadv/pwritev coalesced
    "fragmented_per_chunk",     # baseline per-(block,layer,kv)
]


def predict_opt_path(strides: dict, ssd_block_ids: np.ndarray,
                     cpu_block_ids: np.ndarray, num_threads: int,
                     num_files_per_device: int = 1) -> str:
    """Faithful mirror of the C++ path-selection tree (ssd_io_opt=True).

    Reproduces every step that decides the path, *including* the
    ``sort by (fd, in-file id)`` that ``transfer_kv_blocks_ssd`` applies to the
    (cpu, ssd) id pairs before dispatching them to the worker threads. That
    sort is exactly what a naive predictor misses: when the caller hands the
    same id array to both sides, sorting by SSD id makes *both* sides come out
    perfectly ascending, so a "scattered" workload silently collapses back to
    the contiguous layer-major path.
    """
    block = strides["block_stride_bytes"]
    layer = strides["layer_stride_bytes"]
    chunk = strides["chunk_bytes"]

    # C++: enable_block_first_transfer =
    #        ssd_io_opt && block > cpu_layer_stride && block > ssd_layer_stride
    # (this benchmark uses identical CPU and SSD layouts, so one compare)
    if block > layer:
        return PATH_NAMES[0]

    n = len(ssd_block_ids)
    order = sorted(
        range(n),
        key=lambda i: (int(ssd_block_ids[i]) % num_files_per_device,
                       int(ssd_block_ids[i]) // num_files_per_device),
    )
    ssd_sorted = [int(ssd_block_ids[i]) for i in order]
    cpu_sorted = [int(cpu_block_ids[i]) for i in order]

    per_thread = (n + num_threads - 1) // num_threads
    seen: List[str] = []
    for t in range(num_threads):
        start_block = t * per_thread
        end_block = min(start_block + per_thread, n)
        if start_block >= end_block:
            continue
        contiguous = all(
            cpu_sorted[b + 1] == cpu_sorted[b] + 1
            and ssd_sorted[b + 1] == ssd_sorted[b] + 1
            for b in range(start_block, end_block - 1)
        )
        batchable = (contiguous and num_files_per_device == 1
                     and block == chunk)
        path = PATH_NAMES[1] if batchable else PATH_NAMES[2]
        if path not in seen:
            seen.append(path)
    seen.sort(key=PATH_NAMES.index)
    return "+".join(seen) if seen else "-"


def make_block_ids(num_blocks: int,
                   block_order: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(ssd_block_ids, cpu_block_ids)`` for the requested order.

    ``contiguous`` -> identical ascending ids on both sides.
    ``scattered``  -> two *independent* permutations.

    The independence matters. ``transfer_kv_blocks_ssd`` sorts the (cpu, ssd)
    pairs by SSD id before dispatch, so a single shared shuffled array -- which
    is what this benchmark used to pass for both sides -- is fully un-shuffled
    by that sort and the vectored path is never reached. Only a CPU-side
    permutation that is independent of the SSD ordering survives the sort and
    yields genuinely fragmented CPU buffers, which is also what a real
    workload looks like (CPU and SSD blocks are allocated independently).
    """
    if block_order == "contiguous":
        ids = np.arange(num_blocks, dtype=np.int64)
        return ids, ids.copy()
    ssd_ids = np.arange(num_blocks, dtype=np.int64)
    cpu_ids = np.arange(num_blocks, dtype=np.int64)
    np.random.default_rng(42).shuffle(ssd_ids)
    np.random.default_rng(1337).shuffle(cpu_ids)
    return ssd_ids, cpu_ids


def _proc_io() -> dict:
    """Snapshot /proc/self/io counters (Linux only). Returns {} if unavailable.

    For the fallback engine (sync pread/pwrite/readv/writev) this reflects real
    I/O: syscr/syscw = read/write syscalls, rchar/wchar = bytes transferred.
    With the io_uring engine, rchar/wchar still track data volume but syscw/syscr
    count io_uring_enter calls instead of individual reads/writes, so the I/O
    *count* column is only meaningful under the fallback engine.
    """
    try:
        d = {}
        with open("/proc/self/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                d[k.strip()] = int(v.strip())
        return d
    except OSError:
        return {}


_tmpfs_warned = False


def drop_caches(ssd_path) -> None:
    # Evict cached pages via posix_fadvise(DONTNEED). No root needed; walks
    # ssd_path so it hits the file C++ reads (page cache is per-inode).
    for root, _dirs, files in os.walk(str(ssd_path)):
        for name in files:
            fp = os.path.join(root, name)
            try:
                fd = os.open(fp, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            finally:
                os.close(fd)


def _is_tmpfs(path) -> bool:
    """Return True if `path` lives on a tmpfs mount (no real device)."""
    try:
        path = os.path.abspath(str(path))
        with open("/proc/mounts", "r") as f:
            mounts = [line.split()[1] for line in f]
        # longest prefix match = the mount holding `path`
        best = None
        for m in mounts:
            if path == m or path.startswith(m + "/"):
                if best is None or len(m) > len(best):
                    best = m
        if best:
            with open("/proc/mounts", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == best:
                        return parts[2] == "tmpfs"
    except OSError:
        pass
    return False


def run_single_transfer(ioctx, transfer_fn, cpu_tensor, preset, strides,
                        ssd_block_ids, cpu_block_ids, is_read, num_threads,
                        ssd_io_opt: bool, ssd_path, cold_read: bool = False):
    """Run one transfer and return ``(ms, ios, iobytes)``."""
    layer_ids = torch.arange(0, preset.num_layers, dtype=torch.int32)
    ssd_ids_t = torch.from_numpy(ssd_block_ids.astype(np.int64))
    cpu_ids_t = torch.from_numpy(cpu_block_ids.astype(np.int64))
    if cold_read and is_read:
        drop_caches(ssd_path)
    before = _proc_io()
    start = time.perf_counter()
    transfer_fn(
        ioctx=ioctx,
        cpu_layer_id_list=layer_ids,
        cpu_tensor_ptr=cpu_tensor.data_ptr(),
        ssd_block_ids=ssd_ids_t,
        cpu_block_ids=cpu_ids_t,
        cpu_layer_stride_in_bytes=strides["layer_stride_bytes"],
        cpu_kv_stride_in_bytes=strides["kv_stride_bytes"],
        ssd_layer_stride_in_bytes=strides["layer_stride_bytes"],
        ssd_kv_stride_in_bytes=strides["kv_stride_bytes"],
        chunk_size_in_bytes=strides["chunk_bytes"],
        block_stride_in_bytes=strides["block_stride_bytes"],
        is_read=is_read,
        num_blocks_per_file=len(ssd_block_ids),
        round_robin=1,
        num_threads_per_device=num_threads,
        kv_dim=preset.kv_dim,
        ssd_io_opt=ssd_io_opt,
    )
    try:
        ioctx.get_iouring().wait_completion()
    except Exception:
        pass
    ms = (time.perf_counter() - start) * 1000
    after = _proc_io()
    if before and after:
        if is_read:
            ios = after.get("syscr", 0) - before.get("syscr", 0)
            iobytes = after.get("rchar", 0) - before.get("rchar", 0)
        else:
            ios = after.get("syscw", 0) - before.get("syscw", 0)
            iobytes = after.get("wchar", 0) - before.get("wchar", 0)
    else:
        ios, iobytes = -1, -1
    return ms, ios, iobytes


@dataclass
class CaseResult:
    size: str
    layout: str
    block_order: str
    model: str
    num_blocks: int
    direction: str
    baseline_ms: float
    opt_ms: float
    opt_path: str
    data_gb: float
    base_ios: int = -1
    opt_ios: int = -1
    opt_avg_kb: float = -1.0


def benchmark_case(preset, size_label, layout_type, num_blocks, block_order,
                   rounds, warmup, read_threads, write_threads, engine, ssd_path, SSDIOCTX, transfer_fn,
                   cold_read: bool = False) -> List[CaseResult]:
    layout = make_layout(layout_type, preset, num_blocks)
    strides = compute_strides(layout, preset)
    total_bytes = strides["total_bytes"]

    # Anonymous mmap; no fill (content irrelevant to I/O, avoids 2x temp copy).
    with mmap.mmap(-1, total_bytes, prot=mmap.PROT_READ | mmap.PROT_WRITE) as mm:
        cpu_tensor = torch.frombuffer(mm, dtype=preset.dtype)

        with open(ssd_path, "wb") as f:
            f.truncate(total_bytes)
            os.fsync(f.fileno())
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        with open(ssd_path, "r+b") as f:
            fd = f.fileno()
            libc.fallocate(fd, 0, 0, total_bytes)

        iouring_entries = 512 if engine == "iouring" else 0
        ioctx = SSDIOCTX({0: [str(ssd_path)]}, 1, iouring_entries, 0)

        ssd_block_ids, cpu_block_ids = make_block_ids(num_blocks, block_order)
        data_gb = total_bytes / (1024 ** 3)
        results = []

        for direction, is_read in [("SSD→CPU", True), ("CPU→SSD", False)]:
            n_threads = read_threads if is_read else write_threads
            # Predicted per-direction: the thread count changes the slicing and
            # therefore can change which path each slice takes.
            path = predict_opt_path(strides, ssd_block_ids, cpu_block_ids,
                                    n_threads)
            base_stats, opt_stats = [], []  # each: (ms, ios, iobytes)
            for r in range(warmup + rounds):
                s_base = run_single_transfer(ioctx, transfer_fn, cpu_tensor, preset,
                                             strides, ssd_block_ids, cpu_block_ids,
                                             is_read, n_threads, False,
                                             ssd_path, cold_read=cold_read)
                s_opt = run_single_transfer(ioctx, transfer_fn, cpu_tensor, preset,
                                            strides, ssd_block_ids, cpu_block_ids,
                                            is_read, n_threads, True,
                                            ssd_path, cold_read=cold_read)
                if r >= warmup:
                    base_stats.append(s_base)
                    opt_stats.append(s_opt)
            # Drop None before stats.
            base_stats = [s for s in base_stats if s is not None]
            opt_stats = [s for s in opt_stats if s is not None]
            if not base_stats or not opt_stats:
                # Record invalid result instead of median(None) crash.
                results.append(CaseResult(
                    size=size_label,
                    layout=layout_type.value, block_order=block_order,
                    model=("MLA" if preset.kv_dim == 1 else "MHA"), num_blocks=num_blocks,
                    direction=direction,
                    baseline_ms=-1.0, opt_ms=-1.0,
                    opt_path=path, data_gb=data_gb,
                    base_ios=-1, opt_ios=-1, opt_avg_kb=-1.0,
                ))
                continue
            base_ms = statistics.median(s[0] for s in base_stats)
            opt_ms = statistics.median(s[0] for s in opt_stats)
            has_io = base_stats[0][1] >= 0
            base_ios = int(statistics.median(s[1] for s in base_stats)) if has_io else -1
            opt_ios = int(statistics.median(s[1] for s in opt_stats)) if has_io else -1
            opt_bytes = statistics.median(s[2] for s in opt_stats)
            opt_avg_kb = (opt_bytes / opt_ios / 1024.0) if opt_ios > 0 else -1.0
            results.append(CaseResult(
                size=size_label,
                layout=layout_type.value, block_order=block_order,
                model=("MLA" if preset.kv_dim == 1 else "MHA"), num_blocks=num_blocks,
                direction=direction,
                baseline_ms=base_ms, opt_ms=opt_ms,
                opt_path=path, data_gb=data_gb,
                base_ios=base_ios, opt_ios=opt_ios, opt_avg_kb=opt_avg_kb,
            ))

    return results


def print_table(rows: List[CaseResult], read_threads: int, write_threads: int):
    print()
    print("=" * 170)
    print("  SSD Transfer: baseline (ssd_io_opt=False) vs opt (ssd_io_opt=True)   [read_threads={}, write_threads={}]".format(read_threads, write_threads))
    print("=" * 170)
    hdr = ("{:<10s} {:<7s} {:<10s} {:<6s} {:<7s} {:<9s} {:>13s} {:>10s} {:>9s} {:>9s} "
           "{:>8s} {:>8s} {:>9s}  {:<24s}").format(
        "Layout", "Size", "Order", "Model", "Blocks", "Dir", "Baseline(ms)", "Opt(ms)",
        "Speedup", "Data(GB)", "BaseIOs", "OptIOs", "OptAvgKB", "OptPath(pred)")
    print(hdr)
    print("  " + "-" * 164)
    for r in rows:
        spd = "{:.2f}x".format(r.baseline_ms / r.opt_ms) if r.opt_ms > 0 else "-"
        bios = str(r.base_ios) if r.base_ios >= 0 else "-"
        oios = str(r.opt_ios) if r.opt_ios >= 0 else "-"
        oavg = "{:.1f}".format(r.opt_avg_kb) if r.opt_avg_kb >= 0 else "-"
        print(("  {:<10s} {:<7s} {:<10s} {:<6s} {:<7d} {:<9s} {:>13.2f} {:>10.2f} "
               "{:>9s} {:>9.2f} {:>8s} {:>8s} {:>9s}  {:<24s}").format(
            r.layout, r.size, r.block_order, r.model, r.num_blocks, r.direction,
            r.baseline_ms, r.opt_ms, spd, r.data_gb, bios, oios, oavg, r.opt_path))
    print("=" * 170)


def print_summary(rows: List[CaseResult]):
    """Aggregate BF vs LF median OPT latency (ms) and recommend the lowest absolute.

    Ranking by speedup is misleading: a high speedup only means the baseline was
    bad, not that the opt path is actually faster in absolute terms. We rank by the
    median absolute OPT latency instead.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if r.opt_ms > 0:
            key = (r.layout, r.block_order, r.model, r.direction)
            groups[key].append(r.opt_ms)

    def med(xs):
        return statistics.median(xs) if xs else 0.0

    print()
    print("=" * 100)
    print("  BF vs LF -- median OPT latency (ms) across all sizes / blocks  [lower is better]")
    print("=" * 100)
    for model in ["MHA", "MLA"]:
        for direction in ["SSD→CPU", "CPU→SSD"]:
            bf = med(groups.get(("BLOCKFIRST", "contiguous", model, direction), []))
            lfc = med(groups.get(("LAYERFIRST", "contiguous", model, direction), []))
            lfs = med(groups.get(("LAYERFIRST", "scattered", model, direction), []))
            print("  {:<4s} {:<9s} : BF(contig) {:.2f}ms | LF(contig) {:.2f}ms | LF(scattered) {:.2f}ms".format(
                model, direction, bf, lfc, lfs))
    def collect(layout, order):
        vals = []
        for k, v in groups.items():
            if k[0] == layout and k[1] == order:
                vals.extend(v)
        return med(vals)

    bf_all = collect("BLOCKFIRST", "contiguous")
    lfc_all = collect("LAYERFIRST", "contiguous")
    lfs_all = collect("LAYERFIRST", "scattered")
    print("  " + "-" * 100)
    print("  OVERALL median OPT latency (ms): BF(contig) {:.2f} | LF(contig) {:.2f} | LF(scattered) {:.2f}".format(
        bf_all, lfc_all, lfs_all))

    # Pick the layout with the lowest median OPT latency (absolute, not speedup).
    candidates = [
        ("BLOCKFIRST(contiguous)", bf_all),
        ("LAYERFIRST(contiguous)", lfc_all),
        ("LAYERFIRST(scattered)", lfs_all),
    ]
    best_label, best_ms = min(candidates, key=lambda x: x[1])
    print("  RECOMMEND (by absolute OPT latency): {}: lowest median OPT latency ({:.2f}ms).".format(
        best_label, best_ms))
    print("=" * 100)


def main(args):
    try:
        from flexkv.c_ext import SSDIOCTX, transfer_kv_blocks_ssd
    except ImportError as e:
        print("ERROR: c_ext not built: {}".format(e))
        return

    if not os.path.isdir(args.ssd_dir):
        print("ERROR: --ssd-dir must be an existing directory: {}".format(args.ssd_dir))
        return
    presets = build_presets(args.sizes, args.models)
    layouts = [KVCacheLayoutType(l.upper()) for l in args.layouts]
    block_orders = args.block_orders
    engines = args.engines.split(",")
    read_threads = args.read_threads
    write_threads = args.write_threads

    tmpdir = Path(tempfile.mkdtemp(prefix="flexkv_ssd_bench_",
                                   dir=args.ssd_dir))
    print(">> Temp dir : {}".format(tmpdir))
    if args.cold_read:
        print(">> Cold-read: evicting page cache (posix_fadvise DONTNEED) before "
              "EVERY SSD->CPU read to expose true cold-device cost.")
        global _tmpfs_warned
        if not _tmpfs_warned and _is_tmpfs(tmpdir):
            _tmpfs_warned = True
            print(">> WARN: bench dir is on tmpfs -- reads are served from RAM, not a "
                  "real device. Use --ssd-dir on a real NVMe mount for representative "
                  "cold-read latency.", file=sys.stderr)
    print(">> Sizes    : {} | Models: {} | read_threads={} write_threads={}".format(
        args.sizes, args.models, read_threads, write_threads))
    print(">> Layouts  : {} | Orders: {} | Blocks: {} | Engines: {} | Rounds: {} (warmup {})".format(
        [l.value for l in layouts], block_orders, args.blocks, engines, args.rounds, args.warmup))

    all_rows: List[CaseResult] = []
    try:
        for name, preset in presets.items():
            for num_blocks in args.blocks:
                ssd_path = tmpdir / "ssd_{}_{}.bin".format(name, num_blocks)
                for engine in engines:
                    for layout_type in layouts:
                        for block_order in block_orders:
                            # BF always uses single I/O per block; skip scattered.
                            if layout_type.value == "BLOCKFIRST" and block_order == "scattered":
                                continue
                            print(">> {} blocks={} layout={} order={} engine={} ...".format(
                                name, num_blocks, layout_type.value, block_order, engine))
                            rows = benchmark_case(
                                preset, name.rsplit("-", 1)[0], layout_type, num_blocks,
                                block_order, args.rounds, args.warmup, read_threads,
                                write_threads, engine, ssd_path, SSDIOCTX, transfer_kv_blocks_ssd,
                                cold_read=args.cold_read)
                            all_rows.extend(rows)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print_table(all_rows, read_threads, write_threads)
    print_summary(all_rows)


def parse_args():
    p = argparse.ArgumentParser(description="SSD transfer microbenchmark: baseline vs opt (real C++ call)")
    p.add_argument("--sizes", nargs="+", default=["small", "medium"], choices=list(MODEL_SIZES.keys()))
    p.add_argument("--models", nargs="+", default=["MHA", "MLA"], choices=["MHA", "MLA"])
    p.add_argument("--layouts", nargs="+", default=["BLOCKFIRST", "LAYERFIRST"],
                   choices=["blockfirst", "layerfirst", "BLOCKFIRST", "LAYERFIRST"])
    p.add_argument("--block-orders", nargs="+", default=["contiguous", "scattered"],
                   choices=["contiguous", "scattered"])
    p.add_argument("--blocks", nargs="+", type=int, default=[256, 1024])
    p.add_argument("--read-threads", type=int, default=32, help="SSD read threads (default 32, matches production)")
    p.add_argument("--write-threads", type=int, default=32, help="SSD write threads (default 32, matches production)")
    p.add_argument("--engines", type=str, default="fallback", help="Comma-separated: fallback,iouring")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--cold-read", dest="cold_read", action="store_true",
                   help="Evict page cache (posix_fadvise DONTNEED, no root needed) "
                        "before every SSD->CPU read to expose true cold-device read "
                        "cost. Still requires --ssd-dir on a real disk (tmpfs has no device). "
                        "ON by default; use --no-cold-read to disable.")
    p.add_argument("--no-cold-read", dest="cold_read", action="store_false",
                   help="Disable cold-read mode (default is cold-read ON).")
    p.set_defaults(cold_read=True)
    p.add_argument("--ssd-dir", type=str, required=True,
                   help="REQUIRED: directory for the bench file. Must be an existing "
                        "directory pointing at a REAL NVMe mount (your cache_dir). "
                        "Avoid tmpfs /tmp -- tmpfs has no real device and cold-read "
                        "latency will be meaningless.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
