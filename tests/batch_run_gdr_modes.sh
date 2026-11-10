#!/usr/bin/env bash
# Thin PHASE wrapper. The 134ad747-style five-arm matrix is:
#   bash batch_run_ce_kernel_tp8.sh
# Usage: TP=8 DIRECTION=H2D BATCH_REQUESTS=8 bash tests/batch_run_gdr_modes.sh
set -euo pipefail
export BKCL_ENABLE_XDR="${BKCL_ENABLE_XDR:-1}"
export CUDA_ENABLE_P2P_NO_UVA="${CUDA_ENABLE_P2P_NO_UVA:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-OAM_ID}"
export MLX5_SCATTER_TO_CQE="${MLX5_SCATTER_TO_CQE:-0}"
export FLEXKV_USE_CE_TRANSFER_H2D="${FLEXKV_USE_CE_TRANSFER_H2D:-1}"
export FLEXKV_USE_CE_TRANSFER_D2H="${FLEXKV_USE_CE_TRANSFER_D2H:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PHASE=${PHASE:-compare}
DIAGNOSTICS=${DIAGNOSTICS:-0}
export FLEXKV_LOG_LEVEL=${FLEXKV_LOG_LEVEL:-$([[ "$DIAGNOSTICS" == 1 ]] && echo INFO || echo WARNING)}
export FLEXKV_TRANSFER_TRACE=0
export PYTHONUNBUFFERED=1
if [[ "$PHASE" == unit ]]; then
  exec "${PYTHON:-python3}" tests/gdr/test_contracts.py
fi
case "$PHASE" in
  smoke) BLOCKS=${BLOCKS:-8}; CONCURRENCY=${CONCURRENCY:-1,4}; ROUNDS=${ROUNDS:-2}; LAYERS=${LAYERS:-4} ;;
  compare) BLOCKS=${BLOCKS:-32,256}; CONCURRENCY=${CONCURRENCY:-1,8}; ROUNDS=${ROUNDS:-10}; LAYERS=${LAYERS:-8} ;;
  *) echo 'PHASE must be unit, smoke, or compare' >&2; exit 2 ;;
esac
DIRECTION=${DIRECTION:-both}
[[ "$DIRECTION" == both ]] && DIRECTION=H2D,D2H
OUTPUT=${OUTPUT:-../result_gdr_modes_$(date +%Y%m%d_%H%M%S)_${PHASE}}
# Refuse accidental overwrite; interrupted runs keep raw.jsonl and failure.json.
mkdir "$OUTPUT"
if [[ "$PHASE" == smoke && "${CHECK_IPC:-1}" == 1 ]]; then
  "${PYTHON:-python3}" tests/gdr/test_direct_ipc.py --tp "${TP:-8}" --qps "${QP_COUNT:-4}" \
    --kv-dim "${KV_DIM:-1}" --notify "${NOTIFY_MODE:-polling}" --nics "${NIC_NAMES:-}" \
    2>&1 | tee "$OUTPUT/direct_ipc.log"
fi
command -v nvidia-smi >/dev/null && nvidia-smi topo -m > "$OUTPUT/topology.txt" || true
"${PYTHON:-python3}" tests/gdr/compare_modes.py \
  --output "$OUTPUT" --arms "${ARMS:-CE,GDR}" \
  --directions "$DIRECTION" --tp "${TP:-8}" --blocks "$BLOCKS" --concurrency "$CONCURRENCY" \
  --layers "$LAYERS" --fragment "${FRAGMENT_BYTES:-8192}" --kv-dim "${KV_DIM:-1}" \
  --shared "${SHARED_KV:-1}" --layouts "${LAYOUTS:-BLOCKFIRST}" --scatter "${SCATTER:-1}" \
  --qps "${QP_COUNT:-4}" --batch "${BATCH_REQUESTS:-4}" --max-batch-blocks "${MAX_BATCH_BLOCKS:-8192}" \
  --dedicated "${USE_DEDICATED_GDR_THREAD:-1}" --selector "${SELECTOR:-avgbytes}" \
  --notify "${NOTIFY_MODE:-polling}" --nics "${NIC_NAMES:-}" \
  --rounds "$ROUNDS" --warmup "${WARMUP:-2}" --diagnostics "$DIAGNOSTICS" 2>&1 | tee "$OUTPUT/run.log"
