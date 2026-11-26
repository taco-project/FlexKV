#!/usr/bin/env bash
set -euo pipefail

# Mapped peer-memory GDR with torch-allocated device memory.
# Configure runtime libraries and device visibility externally before launching.
# Usage: QPS="1 2 4 8" SIZE=8388608 ITERS=1000 ./test_mapped_gdr_torch.sh write

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
SIZE="${SIZE:-1048576}"
ITERS="${ITERS:-100}"
QPS="${QPS:-1 2 4 8}"
TX_DEPTH="${TX_DEPTH:-128}"
PYTHON="${PYTHON:-python3}"
SRC="${SCRIPT_DIR}/test_mapped_gdr.c"
SO="${SCRIPT_DIR}/libmapped_gdr.so"
CC="${CC:-cc}"

if [[ ! -e "/sys/class/infiniband/${RDMA_DEV}" ]]; then
    echo "ERROR: RDMA device not found: $RDMA_DEV"
    list_rdma_devs || true
    exit 1
fi

echo "======================================================"
echo "Mapped peer-memory GDR (torch tensor + ibverbs)"
echo "======================================================"
echo "python     : $PYTHON"
echo "src        : $SRC"
echo "so         : $SO"
echo "mode       : $MODE"
echo "GPU        : $GPU_ID (process-local)"
echo "RDMA dev   : $RDMA_DEV"
echo "GID index  : $GID_INDEX"
echo "size       : $SIZE bytes"
echo "iters      : $ITERS"
echo "qps        : $QPS (adjust with QPS=\"1 2 4 8\")"
echo "tx_depth   : $TX_DEPTH (outstanding WRs per QP)"
echo "CUDART_LIB : ${CUDART_LIB:-unset}"
echo
list_gpus
echo

need_rebuild=1
if [[ -f "$SO" && "$SO" -nt "$SRC" ]]; then
    need_rebuild=0
fi
if [[ "$need_rebuild" -eq 1 ]]; then
    echo "[compile] $CC -O2 -fPIC -shared -DMAPPED_GDR_AS_LIB $SRC -o $SO -libverbs -ldl"
    "$CC" -O2 -fPIC -shared -DMAPPED_GDR_AS_LIB "$SRC" -o "$SO" -libverbs -ldl
fi

echo
echo "Multiple QPs can distribute traffic across bonded NIC members."
echo

exec "$PYTHON" "${SCRIPT_DIR}/test_mapped_gdr_torch.py" \
    --so "$SO" \
    --ib-dev "$RDMA_DEV" \
    --gpu "$GPU_ID" \
    --gid-index "$GID_INDEX" \
    --size "$SIZE" \
    --iters "$ITERS" \
    --qps "$QPS" \
    --tx-depth "$TX_DEPTH" \
    --opcode "$MODE"
