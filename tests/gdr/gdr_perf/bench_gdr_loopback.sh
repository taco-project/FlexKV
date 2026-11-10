#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# some special machine types and scenarios XDR WRITE sweep  (examples/bench_gdr_loopback.sh analog)
#
# NVIDIA: same-host ib_read_bw, GPU source / host dest, size×QP×outs CSV.
# some special machine types and scenarios:   same-host test_xdr WRITE sweep, size×QP×tx_depth CSV.
#
#   host DRAM  --RDMA WRITE-->  XPU HBM
#
# Usage:
#   ./bench_gdr_loopback.sh
#   SIZE_LIST="1048576 8388608" QP_LIST="1 2 4" TX_DEPTH_LIST="16 128" \
#       ./bench_gdr_loopback.sh
# ============================================================

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"

apply_special_platform_env
prefer_xmlir_cudart

GPU_ID="${GPU_ID:-0}"
RDMA_DEV="${RDMA_DEV:-$(first_rdma_dev)}"
GID_INDEX="${GID_INDEX:-3}"
ITERS="${ITERS:-1000}"
SIZE_LIST="${SIZE_LIST:-4096 8192 16384 32768 65536 131072 262144 1048576}"
QP_LIST="${QP_LIST:-1 2 4 8}"
TX_DEPTH_LIST="${TX_DEPTH_LIST:-1 2 4 8 16}"
SRC="${SCRIPT_DIR}/test_xdr.c"
BIN="${BIN:-${SCRIPT_DIR}/test_xdr}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results_gdr_loopback}"

export XPU_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

[[ -e "/sys/class/infiniband/${RDMA_DEV}" ]] || die "RDMA device not found: $RDMA_DEV"

build_test_xdr "$SRC" "$BIN"

timestamp="$(date +%Y%m%d_%H%M%S)"
CSV_FILE="${RESULT_DIR}/gdr_write_${timestamp}.csv"
mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "some special machine types and scenarios XDR RDMA WRITE loopback sweep"
echo "============================================================"
echo "binary     : $BIN"
echo "IB device  : $RDMA_DEV"
echo "GPU / XPU  : $GPU_ID"
echo "GID index  : $GID_INDEX"
echo "iters      : $ITERS"
echo "SIZE list  : $SIZE_LIST"
echo "QP list    : $QP_LIST"
echo "tx_depth   : $TX_DEPTH_LIST"
echo "CSV        : $CSV_FILE"
echo "============================================================"

run_xdr_sweep write

echo
echo "============================================================"
echo "DONE"
echo "CSV: $CSV_FILE"
echo "============================================================"
