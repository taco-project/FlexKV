#!/usr/bin/env python3
"""Offline replay for FlexKV D2H block-id dumps (op*_D2H_g*_pid*.npz).

Compare CE memcpy vs custom kernel by running the same transfer twice with
different use_ce_transfer flags.  Intended for DSv4 op38 hang post-mortem.

Example (on h20_2, stop sglang first to free 8 GPUs):

  export PYTHONPATH=/cfs_zhongwei/leolingli/dsv4/python:/cfs_zhongwei/leolingli/dsv4/FlexKV
  /usr/bin/python3.12 FlexKV/tools/replay_d2h_dump.py \\
      --npz /cfs_zhongwei/leolingli/dsv4/flexkv_dumps/op38_D2H_g0_pid301389.npz \\
      --mode tp8 --timeout 120

  # CE off (custom kernel):
  /usr/bin/python3.12 FlexKV/tools/replay_d2h_dump.py ... --no-ce --mode tp8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from flexkv.c_ext import TPTransferThreadGroup, transfer_kv_blocks


@dataclass(frozen=True)
class GroupLayout:
    """DSv4 c4 group-0 layout from server_20260608_074415.log."""

    name: str = "c4_g0"
    num_layers: int = 21
    num_gpus: int = 8
    # GPU page geometry
    sub_page_size: int = 64
    bytes_per_page_padded: int = 37440
    effective_head_size: int = 585  # bytes_per_page_padded // sub_page_size
  # CPU BLOCKFIRST multi-group
    total_block_bytes: int = 15971328
    # Transfer knobs (match production server.sh)
    transfer_num_cta: int = 4
    is_mla: bool = True
    gpu_block_type: int = 0  # VLLM ptr layout: one tensor per layer

    @property
    def chunk_size_bytes(self) -> int:
        return self.sub_page_size * self.effective_head_size

    @property
    def cpu_kv_stride(self) -> int:
        return self.chunk_size_bytes

    @property
    def cpu_layer_stride(self) -> int:
        kv_dim = 1 if self.is_mla else 2
        return kv_dim * self.chunk_size_bytes

    @property
    def cpu_tp_stride(self) -> int:
        return self.total_block_bytes // self.num_gpus

    @property
    def gpu_kv_stride(self) -> int:
        return self.chunk_size_bytes

    @property
    def gpu_block_stride(self) -> int:
        return self.bytes_per_page_padded

    @property
    def gpu_layer_stride(self) -> int:
        return 0


def _load_block_ids(npz_path: str, remap: bool) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    data = np.load(npz_path)
    gpu = np.asarray(data["gpu_block_ids"], dtype=np.int64)
    cpu = np.asarray(data["cpu_block_ids"], dtype=np.int64)
    meta = {
        "orig_gpu_min": int(gpu.min()),
        "orig_gpu_max": int(gpu.max()),
        "orig_cpu_min": int(cpu.min()),
        "orig_cpu_max": int(cpu.max()),
        "count": int(gpu.size),
    }
    if remap:
        n = gpu.size
        gpu = np.arange(n, dtype=np.int64)
        cpu = np.arange(n, dtype=np.int64)
        meta["remapped"] = True
    else:
        meta["remapped"] = False
    return (
        torch.from_numpy(gpu),
        torch.from_numpy(cpu),
        meta,
    )


def _alloc_buffers(
    layout: GroupLayout,
    gpu_block_ids: torch.Tensor,
    cpu_block_ids: torch.Tensor,
    num_gpus: int,
) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
    max_gpu = int(gpu_block_ids.max().item()) + 1
    max_cpu = int(cpu_block_ids.max().item()) + 1

    gpu_layers: List[List[torch.Tensor]] = []
    for gi in range(num_gpus):
        dev = torch.device(f"cuda:{gi}")
        layers = [
            torch.zeros(
                (max_gpu, layout.bytes_per_page_padded),
                dtype=torch.uint8,
                device=dev,
            )
            for _ in range(layout.num_layers)
        ]
        gpu_layers.append(layers)

    cpu_bytes = max_cpu * layout.total_block_bytes + layout.num_layers * layout.cpu_layer_stride
    cpu_tensor = torch.zeros(cpu_bytes, dtype=torch.uint8, pin_memory=True)
    return gpu_layers, cpu_tensor


def _run_single_gpu(
    layout: GroupLayout,
    gpu_ids: torch.Tensor,
    cpu_ids: torch.Tensor,
    use_ce: bool,
) -> None:
    gpu_layers, cpu_tensor = _alloc_buffers(layout, gpu_ids, cpu_ids, num_gpus=1)
    gpu_ptrs = torch.tensor(
        [t.data_ptr() for t in gpu_layers[0]],
        dtype=torch.int64,
    ).pin_memory()

    torch.cuda.set_device(0)
    transfer_kv_blocks(
        gpu_ids,
        gpu_ptrs,
        layout.gpu_kv_stride,
        layout.gpu_block_stride,
        layout.gpu_layer_stride,
        cpu_ids,
        cpu_tensor,
        layout.cpu_kv_stride,
        layout.cpu_layer_stride,
        layout.total_block_bytes,
        layout.chunk_size_bytes,
        0,
        layout.num_layers,
        layout.transfer_num_cta,
        False,  # D2H
        use_ce,
        layout.is_mla,
        layout.gpu_block_type,
        True,
    )
    torch.cuda.synchronize()


def _run_tp8(
    layout: GroupLayout,
    gpu_ids: torch.Tensor,
    cpu_ids: torch.Tensor,
    use_ce: bool,
) -> None:
    gpu_layers, cpu_tensor = _alloc_buffers(
        layout, gpu_ids, cpu_ids, num_gpus=layout.num_gpus
    )
    ptrs_flat = [
        layer.data_ptr()
        for gpu in gpu_layers
        for layer in gpu
    ]
    gpu_device_ids = list(range(layout.num_gpus))
    strides = [layout.gpu_kv_stride] * layout.num_gpus
    blk_strides = [layout.gpu_block_stride] * layout.num_gpus
    layer_strides = [layout.gpu_layer_stride] * layout.num_gpus
    chunk_sizes = [layout.chunk_size_bytes] * layout.num_gpus

    group = TPTransferThreadGroup(
        layout.num_gpus,
        ptrs_flat,
        layout.num_layers,
        cpu_tensor.data_ptr(),
        layout.num_layers,
        strides,
        blk_strides,
        layer_strides,
        chunk_sizes,
        gpu_device_ids,
    )
    group.tp_group_transfer(
        gpu_ids,
        cpu_ids,
        layout.cpu_kv_stride,
        layout.cpu_layer_stride,
        layout.total_block_bytes,
        layout.cpu_tp_stride,
        layout.transfer_num_cta,
        False,  # D2H
        use_ce,
        0,
        layout.num_layers,
        layout.is_mla,
    )


def _worker_main(
    mode: str,
    npz_path: str,
    remap: bool,
    use_ce: bool,
    result_queue: mp.Queue,
) -> None:
    try:
        layout = GroupLayout()
        gpu_ids, cpu_ids, meta = _load_block_ids(npz_path, remap=remap)
        t0 = time.time()
        if mode == "single":
            _run_single_gpu(layout, gpu_ids, cpu_ids, use_ce=use_ce)
        elif mode == "tp8":
            _run_tp8(layout, gpu_ids, cpu_ids, use_ce=use_ce)
        else:
            raise ValueError(f"unknown mode: {mode}")
        elapsed = time.time() - t0
        result_queue.put(
            {
                "ok": True,
                "elapsed": elapsed,
                "meta": meta,
                "use_ce": use_ce,
                "mode": mode,
            }
        )
    except Exception as e:
        result_queue.put({"ok": False, "error": repr(e), "use_ce": use_ce, "mode": mode})


def run_with_timeout(
    mode: str,
    npz_path: str,
    remap: bool,
    use_ce: bool,
    timeout_sec: float,
) -> dict:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(
        target=_worker_main,
        args=(mode, npz_path, remap, use_ce, q),
    )
    p.start()
    p.join(timeout=timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(5)
        return {
            "ok": False,
            "hang": True,
            "use_ce": use_ce,
            "mode": mode,
            "timeout_sec": timeout_sec,
        }
    if q.empty():
        return {"ok": False, "error": "no result (process died)", "use_ce": use_ce}
    return q.get()


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay FlexKV D2H npz dump")
    parser.add_argument("--npz", required=True, help="Path to op*_D2H_g*_pid*.npz")
    parser.add_argument(
        "--mode",
        choices=("single", "tp8"),
        default="tp8",
        help="single=1-GPU transfer_kv_blocks; tp8=TPTransferThreadGroup (production path)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Seconds before hang")
    parser.add_argument(
        "--remap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remap block ids to 0..N-1 (default, saves ~100GB CPU RAM)",
    )
    parser.add_argument("--ce", dest="use_ce", action="store_true", help="CE memcpy path")
    parser.add_argument("--no-ce", dest="use_ce", action="store_false", help="Custom kernel path")
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run CE on then CE off and print comparison",
    )
    parser.set_defaults(use_ce=True)
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        print(f"ERROR: npz not found: {args.npz}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    n_gpu = torch.cuda.device_count()
    if args.mode == "tp8" and n_gpu < 8:
        print(
            f"WARNING: mode=tp8 needs 8 GPUs, found {n_gpu}. "
            "Stop sglang or use --mode single.",
            file=sys.stderr,
        )

    cases = [True, False] if args.both else [args.use_ce]
    exit_code = 0
    for use_ce in cases:
        label = "CE memcpy" if use_ce else "custom kernel"
        print(f"\n=== Replay: mode={args.mode}, path={label}, remap={args.remap} ===")
        result = run_with_timeout(
            args.mode, args.npz, args.remap, use_ce, args.timeout
        )
        if result.get("hang"):
            print(f"HANG suspected (>{args.timeout}s): {label}")
            exit_code = 2
        elif result.get("ok"):
            meta = result.get("meta", {})
            print(
                f"OK in {result['elapsed']:.3f}s, blocks={meta.get('count')}, "
                f"orig_gpu=[{meta.get('orig_gpu_min')},{meta.get('orig_gpu_max')}]"
            )
        else:
            print(f"FAILED: {result.get('error')}")
            exit_code = 1

    if args.both:
        print(
            "\nInterpretation:\n"
            "  CE hangs + kernel OK  -> bug in CE cudaMemcpyAsync loop / sync\n"
            "  Both hang             -> likely custom kernel or shared driver issue\n"
            "  CE OK in isolation    -> hang may need full server concurrency (H2D overlap)"
        )
    return exit_code


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
