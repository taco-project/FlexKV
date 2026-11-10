#!/usr/bin/env python3
"""some special machine types and scenarios XDR via torch-allocated XPU memory.

Torch (XMLIR) allocates the XPU tensor.  The GDR path is the same as
peermem_map.cpp: cudaPointerGetAttributes -> hostPointer, then ibverbs
RDMA WRITE/READ (libxdr_gdr.so).

  WRITE: host DRAM  --RDMA WRITE-->  torch XPU tensor
  READ : host DRAM  <--RDMA READ---  torch XPU tensor
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def make_pattern(n: int, seed: int):
    import numpy as np

    i = np.arange(n, dtype=np.uint32)
    return ((i + seed) & 0xFF).astype(np.uint8)


def parse_int_list(s: str) -> list:
    return [int(x) for x in str(s).replace(",", " ").split() if x]


def load_gdr(so_path: str) -> ctypes.CDLL:
    if not os.path.isfile(so_path):
        die(f"missing {so_path}; compile with test_xdr_torch.sh")
    lib = ctypes.CDLL(so_path)
    lib.xdr_gdr_run.restype = ctypes.c_int
    lib.xdr_gdr_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    return lib


def gdr_run(
    lib: ctypes.CDLL,
    dev_ptr: int,
    nbytes: int,
    ibdev: str,
    gid_index: int,
    nqps: int,
    tx_depth: int,
    iters: int,
    do_write: int,
    do_read: int,
) -> None:
    rc = lib.xdr_gdr_run(
        ctypes.c_void_p(dev_ptr),
        nbytes,
        ibdev.encode(),
        1,
        gid_index,
        nqps,
        tx_depth,
        iters,
        do_write,
        do_read,
        0,  # torch seeds the tensor; do not BAR-fill
    )
    if rc != 0:
        die(f"xdr_gdr_run rc={rc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--so", default=os.path.join(os.path.dirname(__file__), "libxdr_gdr.so"))
    p.add_argument("--ib-dev", default=os.environ.get("RDMA_DEV") or os.environ.get("IB_DEV") or "mlx5_bond_0")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gid-index", type=int, default=int(os.environ.get("GID_INDEX", "3")))
    p.add_argument("--size", type=int, default=int(os.environ.get("SIZE", str(1 << 20))))
    p.add_argument("--iters", type=int, default=int(os.environ.get("ITERS", "100")))
    p.add_argument(
        "--qps",
        default=os.environ.get("QPS", "1 2 4 8"),
        help="QP counts, space or comma separated, e.g. '1 2 4 8'",
    )
    p.add_argument("--tx-depth", type=int, default=int(os.environ.get("TX_DEPTH", "128")))
    p.add_argument("--opcode", choices=["write", "read", "both"], default="both")
    args = p.parse_args()
    qps_list = parse_int_list(args.qps)
    if not qps_list or min(qps_list) < 1:
        die("--qps must be positive integers, e.g. 1 2 4 8")

    os.environ.setdefault("BKCL_ENABLE_XDR", "1")
    os.environ.setdefault("CUDA_ENABLE_P2P_NO_UVA", "1")

    log("======================================================")
    log("some special machine types and scenarios XDR  (torch tensor + ibverbs GDR)")
    log("======================================================")
    log(f"so         : {args.so}")
    log(f"ibdev      : {args.ib_dev}")
    log(f"gpu / xpu  : {args.gpu} (process-local, after VISIBLE_DEVICES)")
    log(f"size       : {args.size} bytes")
    log(f"iters      : {args.iters}")
    log(f"qps        : {qps_list}")
    log(f"tx_depth   : {args.tx_depth}")
    log(f"opcode     : {args.opcode}")

    import torch

    log(f"torch      : {torch.__version__}")
    log(f"cuda.is_available = {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        die("torch.cuda is not available (XMLIR should expose XPU as cuda)")
    log(f"device_count = {torch.cuda.device_count()}")
    torch.cuda.set_device(args.gpu)

    nbytes = args.size
    t = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{args.gpu}")
    torch.cuda.synchronize()
    ptr = int(t.data_ptr())
    log(f"torch.empty uint8[{nbytes}]  data_ptr={hex(ptr)}  device={t.device}")
    if not (0x4000000000 <= ptr < 0x5800000000):
        log(f"WARNING: data_ptr not in some special machine types and scenarios HBM window 0x4000000000-0x5800000000")

    lib = load_gdr(args.so)
    do_write = args.opcode in ("write", "both")
    do_read = args.opcode in ("read", "both")

    for nqps in qps_list:
        log("")
        log("######################################################")
        log(f"QP = {nqps}")
        log("######################################################")

        if do_write:
            log("")
            log(f"########## torch WRITE  QPs={nqps}  host DRAM -> XPU tensor ##########")
            t.zero_()
            torch.cuda.synchronize()
            gdr_run(
                lib, ptr, nbytes, args.ib_dev, args.gid_index, nqps,
                args.tx_depth, args.iters, 1, 0,
            )
            torch.cuda.synchronize()
            got = t.detach().cpu().contiguous().numpy()
            expect = make_pattern(nbytes, 0xA5)
            bad = int((got != expect).sum())
            if bad:
                die(
                    f"torch.cpu() after GDR WRITE QPs={nqps} mismatch: "
                    f"{bad} / {nbytes} bytes"
                )
            log(f"torch.cpu() after GDR WRITE QPs={nqps}: payload matches")

        if do_read:
            log("")
            log(f"########## torch READ   QPs={nqps}  XPU tensor -> host DRAM ##########")
            expect = make_pattern(nbytes, 0x5A)
            t.copy_(torch.from_numpy(expect.copy()).to(t.device))
            torch.cuda.synchronize()
            gdr_run(
                lib, ptr, nbytes, args.ib_dev, args.gid_index, nqps,
                args.tx_depth, args.iters, 0, 1,
            )
            log(f"torch-seeded GDR READ QPs={nqps}: payload checked inside libxdr_gdr")

    # Keep tensor alive until GDR returned.
    _ = t
    log("")
    log("======================================================")
    log("ALL TORCH<->XPU XDR TESTS PASSED")
    log("======================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
