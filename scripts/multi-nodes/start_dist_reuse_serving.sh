#!/bin/bash
# ============================================================================
# FlexKV dist_reuse 多物理节点启动脚本（§2.5 / Phase 1-G）
#
# 场景（设计文档 §4.5.5 / §4.6.3）：
#   * Prefill 实例跨 2 台 GPU 机器（pp_size × tp_node_count ≤ 2）
#   * CP 在节点内做（CP 不进 sd_key，CP rank > 0 只做 GPU 注册）
#   * 每个 SD-Remote 物理机独立 Mooncake TransferEngine
#   * Master 单写 Redis，Remote 读 + 收协同 GET/PUT 指令
#
# 用法：
#   # Master（node_rank=0）
#   ./start_dist_reuse_serving.sh \
#       --nnodes 2 --node-rank 0 \
#       --master-ip 10.0.0.1 --dist-init-port 29500 \
#       --tp-size 8 --pp-size 1 --cp-size 4 \
#       --model /workspace/models/DeepSeek-V3 \
#       --redis-host 10.0.0.1 --redis-password 123456
#
#   # Remote（node_rank=1）— 脚本参数与 master 相同
#   ./start_dist_reuse_serving.sh \
#       --nnodes 2 --node-rank 1 \
#       --master-ip 10.0.0.1 --dist-init-port 29500 \
#       --tp-size 8 --pp-size 1 --cp-size 4 \
#       --model /workspace/models/DeepSeek-V3 \
#       --redis-host 10.0.0.1 --redis-password 123456
#
# 必填：
#   --nnodes / --node-rank / --master-ip / --tp-size / --pp-size / --cp-size /
#   --model / --redis-host
#
# 备注：
#   * 如果该 instance 只用 CP 跨节点（CP 跨机、TP/PP 不跨机），脚本会自动
#     把 node_rank>0 的机器放到 CP_PEER_REGISTRATION_ONLY 路径上（不启
#     TransferManagerOnRemote，sglang connector 侧按 multinode_policy
#     策略自己决定）。
#   * 脚本**不直接启动** sglang/vLLM 进程；它只做环境变量和配置文件生
#     成，然后 exec 用户指定的启动命令（--launcher-cmd / 默认是
#     sglang 的 router 入口）。这样脚本本身单测友好。
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------- argparse
NNODES=""
NODE_RANK=""
MASTER_IP=""
DIST_INIT_PORT=29500
TP_SIZE=""
PP_SIZE=""
CP_SIZE="1"
MODEL=""
REDIS_HOST=""
REDIS_PORT=6379
REDIS_PASSWORD=""
MOONCAKE_REDIS_PORT=6380       # separate metadata redis for mooncake
MOONCAKE_ENGINE_PORT_BASE=12345
LAUNCHER_CMD=""
RDMA_DEVICE="${RDMA_DEVICE:-mlx5_0}"
INSTANCE_ID=""
DRY_RUN="false"

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nnodes)              NNODES="$2"; shift 2 ;;
    --node-rank)           NODE_RANK="$2"; shift 2 ;;
    --master-ip)           MASTER_IP="$2"; shift 2 ;;
    --dist-init-port)      DIST_INIT_PORT="$2"; shift 2 ;;
    --tp-size)             TP_SIZE="$2"; shift 2 ;;
    --pp-size)             PP_SIZE="$2"; shift 2 ;;
    --cp-size)             CP_SIZE="$2"; shift 2 ;;
    --model)               MODEL="$2"; shift 2 ;;
    --redis-host)          REDIS_HOST="$2"; shift 2 ;;
    --redis-port)          REDIS_PORT="$2"; shift 2 ;;
    --redis-password)      REDIS_PASSWORD="$2"; shift 2 ;;
    --mooncake-redis-port) MOONCAKE_REDIS_PORT="$2"; shift 2 ;;
    --mooncake-engine-port-base) MOONCAKE_ENGINE_PORT_BASE="$2"; shift 2 ;;
    --rdma-device)         RDMA_DEVICE="$2"; shift 2 ;;
    --instance-id)         INSTANCE_ID="$2"; shift 2 ;;
    --launcher-cmd)        LAUNCHER_CMD="$2"; shift 2 ;;
    --dry-run)             DRY_RUN="true"; shift ;;
    -h|--help)             usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

# ---------------------------------------------------------------- validate
for v in NNODES NODE_RANK MASTER_IP TP_SIZE PP_SIZE CP_SIZE MODEL REDIS_HOST; do
  if [[ -z "${!v}" ]]; then
    echo "Missing required argument: --$(echo "$v" | tr 'A-Z_' 'a-z-')"
    usage
  fi
done

if ! [[ "$NNODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "nnodes must be >= 1"; exit 2
fi
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
  echo "node_rank must be in [0, nnodes)"; exit 2
fi

# Deployment constraint (design-doc §3.3): prefill crosses at most 2 nodes.
# Enforce the same bound here so mis-configured clusters fail fast.
if (( NNODES > 2 )); then
  echo "dist_reuse currently supports <= 2 physical nodes per instance"
  echo "(see docs/dist_reuse/dist_reuse_with_cp_pp_multinode_tp.md §3.3)"
  exit 2
fi

# -------------------------------------------------------- derived topology
# Basic rule: MASTER = node_rank 0.  Everything else is an off-master role
# whose concrete type is picked by the connector's multinode_policy
# (FlexKV/flexkv/integration/multinode_policy.py).  The script only needs
# to decide whether to emit a Mooncake config (full SD-Remote) or just an
# empty config (CP-peer stub); it does that by asking whether the
# instance has any TP-level cross-node spread.

# tp_node_count = TP_SIZE / gpus_per_node.  We auto-detect gpus_per_node
# via nvidia-smi (default 8 if unavailable).
if command -v nvidia-smi >/dev/null 2>&1; then
  GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
else
  GPUS_PER_NODE=8
fi
if (( GPUS_PER_NODE == 0 )); then GPUS_PER_NODE=8; fi

if (( TP_SIZE <= GPUS_PER_NODE )); then
  TP_NODE_COUNT=1
else
  if (( TP_SIZE % GPUS_PER_NODE != 0 )); then
    echo "tp_size ($TP_SIZE) must be a multiple of gpus_per_node ($GPUS_PER_NODE)"
    exit 2
  fi
  TP_NODE_COUNT=$((TP_SIZE / GPUS_PER_NODE))
fi

IS_MULTINODE_TP="false"
if (( TP_NODE_COUNT > 1 )) || (( PP_SIZE > 1 )); then
  # PP > 1 crossing nodes always needs a full SD-Remote on each PP-peer
  # node too.  We conservatively flag it as multinode_tp for the
  # mooncake-config emission branch.  The connector's policy module
  # still does the fine-grained role decision at runtime.
  IS_MULTINODE_TP="true"
fi

IS_MULTINODE_CP="false"
if (( CP_SIZE > 1 && NNODES > 1 )); then
  IS_MULTINODE_CP="true"
fi

# Default INSTANCE_ID if not provided (stable per-invocation value).
if [[ -z "$INSTANCE_ID" ]]; then
  INSTANCE_ID="flexkv-$(hostname -s)-${DIST_INIT_PORT}"
fi

# ---------------------------------------------------------------- layout
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/dist_reuse"
CFG_DIR="${SCRIPT_DIR}/gen/dist_reuse_node${NODE_RANK}"
mkdir -p "$LOG_DIR" "$CFG_DIR"

# ---------------------------------------------------------- mooncake config
# Only emit a real mooncake config when we need a full SD-Remote on this
# node.  CP-peer-only nodes don't touch mooncake.
NEED_FULL_SD_REMOTE="false"
if [[ "$NODE_RANK" == "0" ]]; then
  # Master always needs mooncake — it owns the TransferEngine in-process.
  NEED_FULL_SD_REMOTE="true"
else
  if [[ "$IS_MULTINODE_TP" == "true" ]]; then
    NEED_FULL_SD_REMOTE="true"
  fi
  # CP-only multi-node: off-master is peer-stub only, no mooncake.
fi

MOONCAKE_ENGINE_PORT=$((MOONCAKE_ENGINE_PORT_BASE + NODE_RANK))
MOONCAKE_CONFIG_FILE="${CFG_DIR}/mooncake_config.json"

if [[ "$NEED_FULL_SD_REMOTE" == "true" ]]; then
  cat > "$MOONCAKE_CONFIG_FILE" <<EOF
{
    "engine_ip": "$(hostname -i | awk '{print $1}')",
    "engine_port": ${MOONCAKE_ENGINE_PORT},
    "metadata_backend": "redis",
    "metadata_server": "redis://${REDIS_HOST}:${MOONCAKE_REDIS_PORT}",
    "metadata_server_auth": "${REDIS_PASSWORD}",
    "protocol": "rdma",
    "device_name": "${RDMA_DEVICE}"
}
EOF
else
  # Empty sentinel — the connector checks for file existence and picks
  # the CP-peer stub path when absent.
  : > "${MOONCAKE_CONFIG_FILE}"
fi

# --------------------------------------------------------- flexkv env vars
# The sglang connector reads these at import time; we export them so
# whatever --launcher-cmd the user passes inherits them.
export FLEXKV_ENABLE_SHARING_DOMAIN=1
export FLEXKV_INSTANCE_ID="${INSTANCE_ID}"
export FLEXKV_REDIS_HOST="${REDIS_HOST}"
export FLEXKV_REDIS_PORT="${REDIS_PORT}"
export FLEXKV_REDIS_PASSWORD="${REDIS_PASSWORD}"
export FLEXKV_MASTER_HOST="${MASTER_IP}"
export FLEXKV_DIST_INIT_ADDR="${MASTER_IP}:${DIST_INIT_PORT}"
export FLEXKV_NNODES="${NNODES}"
export FLEXKV_NODE_RANK="${NODE_RANK}"
export FLEXKV_TP_NODE_COUNT="${TP_NODE_COUNT}"
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_FILE}"
export MC_REDIS_PASSWORD="${REDIS_PASSWORD}"

# --------------------------------------------------------- summary + launch
echo "================================================================"
echo "FlexKV dist_reuse — node_rank=${NODE_RANK}/${NNODES}"
echo "  model              : ${MODEL}"
echo "  tp / pp / cp       : ${TP_SIZE} / ${PP_SIZE} / ${CP_SIZE}"
echo "  tp_node_count      : ${TP_NODE_COUNT}  (gpus_per_node=${GPUS_PER_NODE})"
echo "  is_multinode_tp    : ${IS_MULTINODE_TP}"
echo "  is_multinode_cp    : ${IS_MULTINODE_CP}"
echo "  need_full_sd_remote: ${NEED_FULL_SD_REMOTE}"
echo "  instance_id        : ${INSTANCE_ID}"
echo "  master_ip          : ${MASTER_IP}"
echo "  dist_init          : ${MASTER_IP}:${DIST_INIT_PORT}"
echo "  redis              : ${REDIS_HOST}:${REDIS_PORT} (flexkv) / ${MOONCAKE_REDIS_PORT} (mooncake)"
echo "  mooncake cfg       : ${MOONCAKE_CONFIG_FILE}"
echo "  launcher           : ${LAUNCHER_CMD:-<default: print env and exit>}"
echo "================================================================"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] not executing launcher."
  exit 0
fi

if [[ -z "$LAUNCHER_CMD" ]]; then
  echo "No --launcher-cmd provided; env is set, exec the launcher yourself:"
  echo ""
  echo "  # env exported:"
  env | grep -E '^(FLEXKV_|MOONCAKE_|MC_)' | sort
  echo ""
  exit 0
fi

# exec the user launcher — stdout/stderr to log.
LOG_FILE="${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
echo "Launching: ${LAUNCHER_CMD}"
echo "Log: ${LOG_FILE}"
# shellcheck disable=SC2086
exec bash -c "${LAUNCHER_CMD}" >"${LOG_FILE}" 2>&1
