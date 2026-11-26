#!/usr/bin/env bash
set -euo pipefail

# Mapped peer-memory loopback: cudaMalloc + hostPointer + ibverbs.
# WRITE: host DRAM -> device memory; READ: device memory -> host DRAM.
# Configure the runtime and device visibility externally before launching.
# Usage: SIZE=8388608 ITERS=1000 QPS="1 2 4 8" ./test_gdr_loopback.sh write

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"

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
GID_INDEX="${GID_INDEX:-0}"
SIZE="${SIZE:-8388608}"
ITERS="${ITERS:-1000}"
QPS="${QPS:-1 2 4 8}"
TX_DEPTH="${TX_DEPTH:-128}"
SRC="${SCRIPT_DIR}/test_mapped_gdr.c"
BIN="${BIN:-${SCRIPT_DIR}/test_mapped_gdr}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/gdr_loopback_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

if [[ ! -e "/sys/class/infiniband/${RDMA_DEV}" ]]; then
    echo "ERROR: RDMA device not found: $RDMA_DEV"
    echo "Available:"
    list_rdma_devs || true
    exit 1
fi

SLAVES=( $(bond_slaves "$RDMA_DEV") )

echo "======================================================"
echo "Mapped peer-memory GDR loopback (cudaMalloc + ibverbs)"
echo "======================================================"
echo "source     : $SRC"
echo "binary     : $BIN"
echo "mode       : $MODE"
echo "GPU        : $GPU_ID (process-local)"
echo "RDMA dev   : $RDMA_DEV"
echo "GID index  : $GID_INDEX"
echo "slaves     : ${SLAVES[*]}"
echo "size       : $SIZE bytes"
echo "iters      : $ITERS"
echo "QPs        : $QPS"
echo "tx_depth   : $TX_DEPTH"
echo "output     : $OUTDIR"
echo "CUDART_LIB : ${CUDART_LIB:-"(dlopen libcudart.so via LD_LIBRARY_PATH)"}"
echo
echo "---------- GPUs ----------"
list_gpus
echo
echo "---------- RDMA ----------"
list_rdma_devs || true
if command -v ibdev2netdev >/dev/null 2>&1; then
    ibdev2netdev || true
fi
echo

build_test_mapped_gdr "$SRC" "$BIN"

echo
echo "Multiple QPs can distribute traffic across bonded NIC members."
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
        -g "$GPU_ID" \
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
