#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# some special machine types and scenarios XDR loopback  (examples/test_gdr_loopback.sh analog)
#
# NVIDIA: CUDA ib_write_bw --use_cuda, GPU <-> GPU WRITE.
# some special machine types and scenarios:   xpu_malloc + XMLIR hostPointer + ibverbs.
#
#   WRITE: host DRAM  --RDMA WRITE-->  XPU HBM
#   READ : host DRAM  <--RDMA READ---  XPU HBM
#
# Multi-QP + bond slave counters.  No ib_write_bw, no Mooncake.
#
# Usage:
#   ./test_gdr_loopback.sh
#   ./test_gdr_loopback.sh write
#   ./test_gdr_loopback.sh read
#   SIZE=8388608 ITERS=1000 QPS="1 2 4 8" ./test_gdr_loopback.sh write
# ============================================================

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"

apply_special_platform_env
prefer_xmlir_cudart

MODE="${1:-both}"
case "$MODE" in
    both|write|read) ;;
    *)
        echo "Usage: $0 [both|write|read]"
        exit 1
        ;;
esac

GPU_ID="${GPU_ID:-0}"
RDMA_DEV="${RDMA_DEV:-$(first_rdma_dev)}"
GID_INDEX="${GID_INDEX:-3}"
SIZE="${SIZE:-8388608}"
ITERS="${ITERS:-1000}"
QPS="${QPS:-1 2 4 8}"
TX_DEPTH="${TX_DEPTH:-128}"
SRC="${SCRIPT_DIR}/test_xdr.c"
BIN="${BIN:-${SCRIPT_DIR}/test_xdr}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/gdr_loopback_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

export XPU_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ ! -e "/sys/class/infiniband/${RDMA_DEV}" ]]; then
    echo "ERROR: RDMA device not found: $RDMA_DEV"
    echo "Available:"
    list_rdma_devs || true
    exit 1
fi

SLAVES=( $(bond_slaves "$RDMA_DEV") )

echo "======================================================"
echo "some special machine types and scenarios XDR loopback  (xpu_malloc + ibverbs)"
echo "======================================================"
echo "source     : $SRC"
echo "binary     : $BIN"
echo "mode       : $MODE"
echo "GPU / XPU  : $GPU_ID"
echo "RDMA dev   : $RDMA_DEV"
echo "GID index  : $GID_INDEX"
echo "slaves     : ${SLAVES[*]}"
echo "size       : $SIZE bytes"
echo "iters      : $ITERS"
echo "QPs        : $QPS"
echo "tx_depth   : $TX_DEPTH"
echo "output     : $OUTDIR"
echo "XMLIR_CUDART = ${XMLIR_CUDART:-"(dlopen libcudart.so via LD_LIBRARY_PATH)"}"
echo
echo "---------- XPUs ----------"
list_xpus
echo
echo "---------- RDMA ----------"
list_rdma_devs || true
if command -v ibdev2netdev >/dev/null 2>&1; then
    ibdev2netdev || true
fi
echo

build_test_xdr "$SRC" "$BIN"

echo
echo "1 QP on mlx5_bond typically hashes onto one 200G slave (~200 Gbps)."
echo "Use QPS=2/4/8 to fill both bond members."
echo

for q in $QPS; do
    echo
    echo "######################################################"
    echo "QP = $q"
    echo "######################################################"

    before="${OUTDIR}/q${q}.before"
    after="${OUTDIR}/q${q}.after"
    log="${OUTDIR}/q${q}.log"

    snapshot_nics "$before" "${SLAVES[@]}"

    "$BIN" \
        -d "$RDMA_DEV" \
        -g 0 \
        -x "$GID_INDEX" \
        -s "$SIZE" \
        -n "$ITERS" \
        -q "$q" \
        -D "$TX_DEPTH" \
        "$MODE" \
        | tee "$log"

    snapshot_nics "$after" "${SLAVES[@]}"
    echo
    show_slave_deltas "$before" "$after" "${SLAVES[@]}"
done

echo
echo "======================================================"
echo "ALL DONE"
echo "Logs: $OUTDIR"
echo "======================================================"
