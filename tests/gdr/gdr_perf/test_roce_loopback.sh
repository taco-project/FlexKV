#!/usr/bin/env bash
set -euo pipefail

# Host DRAM <-> host DRAM RoCE control, independent of mapped peer memory.
# Usage: QPS="1 2 4 8" SIZE=8388608 ITERS=1000 ./test_roce_loopback.sh

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"

RDMA_DEV="${RDMA_DEV:-$(first_rdma_dev)}"
GID_INDEX="${GID_INDEX:-0}"
SIZE="${SIZE:-8388608}"
ITERS="${ITERS:-1000}"
QPS="${QPS:-1 2 4 8}"
TX_DEPTH="${TX_DEPTH:-128}"
SRC="${SCRIPT_DIR}/test_mapped_gdr.c"
BIN="${BIN:-${SCRIPT_DIR}/test_mapped_gdr}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/roce_loopback_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

if [[ ! -e "/sys/class/infiniband/${RDMA_DEV}" ]]; then
    echo "ERROR: RDMA device not found: $RDMA_DEV"
    list_rdma_devs || true
    exit 1
fi

SLAVES=( $(bond_slaves "$RDMA_DEV") )

echo "======================================================"
echo "RoCE same-host loopback (host memory)"
echo "======================================================"
echo "RDMA dev : $RDMA_DEV"
echo "GID index: $GID_INDEX"
echo "slaves   : ${SLAVES[*]}"
echo "QPs      : $QPS"
echo "size     : $SIZE bytes"
echo "iters    : $ITERS"
echo "tx_depth : $TX_DEPTH"
echo "output   : $OUTDIR"
echo

if [[ -e "/sys/class/infiniband/${RDMA_DEV}/ports/1/gids/${GID_INDEX}" ]]; then
    echo "GID:"
    cat "/sys/class/infiniband/${RDMA_DEV}/ports/1/gids/${GID_INDEX}"
    echo
fi

echo "Before test:"
for d in "${SLAVES[@]}"; do
    echo "---- $d ----"
    echo "tx_bytes=$(counter "$d" tx_bytes)"
    echo "rx_bytes=$(counter "$d" rx_bytes)"
done

build_test_mapped_gdr "$SRC" "$BIN"

for q in $QPS; do
    echo
    echo "======================================================"
    echo "QP = $q"
    echo "======================================================"

    before="$OUTDIR/q${q}.before"
    after="$OUTDIR/q${q}.after"
    log="$OUTDIR/q${q}.log"

    snapshot_nics "$before" "${SLAVES[@]}"

    "$BIN" \
        -d "$RDMA_DEV" \
        -x "$GID_INDEX" \
        -s "$SIZE" \
        -n "$ITERS" \
        -q "$q" \
        -D "$TX_DEPTH" \
        -H \
        write \
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
