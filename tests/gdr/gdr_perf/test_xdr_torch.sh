#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# some special machine types and scenarios XDR with torch-allocated XPU memory
#
#   torch.empty(..., device="cuda")     XMLIR XPU HBM
#   cudaPointerGetAttributes            XDR hostPointer (peermem_map.cpp)
#   ibv_reg_mr + RDMA WRITE/READ        libxdr_gdr.so
#
# Native xpu_malloc test remains: ./test_gdr_loopback.sh
#
# Usage:
#   ./test_xdr_torch.sh
#   ./test_xdr_torch.sh write
#   QPS="1 2 4 8" SIZE=8388608 ITERS=1000 ./test_xdr_torch.sh write
#   QPS=4 TX_DEPTH=128 ./test_xdr_torch.sh
# ============================================================

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"

apply_special_platform_env

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
SIZE="${SIZE:-1048576}"
ITERS="${ITERS:-100}"
QPS="${QPS:-1 2 4 8}"
TX_DEPTH="${TX_DEPTH:-128}"
PYTHON="${PYTHON:-python3}"
SRC="${SCRIPT_DIR}/test_xdr.c"
SO="${SCRIPT_DIR}/libxdr_gdr.so"
CC="${CC:-cc}"

# One card before import torch (XMLIR aborts if num_gpus is empty).
export XPU_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
LOCAL_GPU=0

prefer_xmlir_cudart

if [[ ! -e "/sys/class/infiniband/${RDMA_DEV}" ]]; then
    echo "ERROR: RDMA device not found: $RDMA_DEV"
    list_rdma_devs || true
    exit 1
fi

echo "======================================================"
echo "some special machine types and scenarios XDR  (torch tensor + ibverbs GDR)"
echo "======================================================"
echo "python     : $PYTHON"
echo "src        : $SRC"
echo "so         : $SO"
echo "mode       : $MODE"
echo "GPU / XPU  : $GPU_ID  (VISIBLE_DEVICES -> local 0)"
echo "RDMA dev   : $RDMA_DEV"
echo "GID index  : $GID_INDEX"
echo "size       : $SIZE bytes"
echo "iters      : $ITERS"
echo "qps        : $QPS  (adjust with QPS=\"1 2 4 8\")"
echo "tx_depth   : $TX_DEPTH  (outstanding WRs per QP)"
echo "XMLIR_CUDART = ${XMLIR_CUDART:-unset}"
echo
list_xpus
echo

need_rebuild=1
if [[ -f "$SO" && "$SO" -nt "$SRC" ]]; then
    need_rebuild=0
fi
if [[ "$need_rebuild" -eq 1 ]]; then
    echo "[compile] $CC -O2 -fPIC -shared -DXDR_AS_LIB $SRC -o $SO -libverbs -ldl"
    "$CC" -O2 -fPIC -shared -DXDR_AS_LIB "$SRC" -o "$SO" -libverbs -ldl
fi

echo
echo "1 QP on mlx5_bond typically hashes onto one 200G slave (~200 Gbps)."
echo "QPS=2/4/8 uses multiple QPs to fill both bond members."
echo

exec "$PYTHON" "${SCRIPT_DIR}/test_xdr_torch.py" \
    --so "$SO" \
    --ib-dev "$RDMA_DEV" \
    --gpu "$LOCAL_GPU" \
    --gid-index "$GID_INDEX" \
    --size "$SIZE" \
    --iters "$ITERS" \
    --qps "$QPS" \
    --tx-depth "$TX_DEPTH" \
    --opcode "$MODE"
