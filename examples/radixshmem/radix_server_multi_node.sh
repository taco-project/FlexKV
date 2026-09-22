#!/bin/bash
# The operator's radix-server on ONE node of a cluster; run it on every node with
# the same --cluster-id / --registry. Peers dial the IP resolved from
# --rpc-interface (node identity defaults to node<ip>); the index control plane
# uses --index-dev, the KV bytes move over --transfer-dev. The first node that
# receives a geometry publishes it to etcd, the others adopt it; slot counts
# may differ per node (different budgets). FlexKV's ready_timeout_s must cover
# --bootstrap-timeout.
set -eu
radix-server --name "${NAME:-/flexkv}" \
  --data-bytes "${DATA_BYTES:-64G}" \
  ${SWA_RATIO:+--swa-ratio "$SWA_RATIO"} \
  --expected-min-nodes "${NODES:-2}" --num-rht-shards "${NODES:-2}" --rht-slots 4 \
  --registry "${REGISTRY:-etcd://10.0.0.1:2379}" --cluster-id "${CLUSTER_ID:-flexkv_prod}" \
  --rpc-interface "${RPC_INTERFACE:-bond0}" --index-dev "${INDEX_DEV:-mlx5_bond_0}" --gid-idx "${GID_IDX:-3}" \
  --transfer-dev "${TRANSFER_DEV:-mlx5_0}" \
  --bootstrap-timeout "${BOOTSTRAP_TIMEOUT:-600}"
